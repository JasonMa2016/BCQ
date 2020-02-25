import time
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

from BC import Policy
from PPO import PPO
from DDPG import DDPG

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class DRIL(object):
    def __init__(self, state_dim, action_dim, policy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = args.device
        # self.config = args

        self.state_dim = state_dim
        self.action_dim = action_dim

        # BC imitator ensemble
        self.ensemble = []

        # initialize actor
        self.policy = policy
        self.policy_minibatch_loss = nn.MSELoss() # ???

    def set_expert(self, expert_traj):
        """
        Set the expert trajectories.
        :param expert_traj:
        :param num_traj
        :return:
        """
        self.expert_traj = expert_traj
        self.expert_states = []
        self.expert_actions = []
        for i in range(len(self.expert_traj)):
            self.expert_states.append(self.expert_traj[i][0])
            self.expert_actions.append(self.expert_traj[i][2])
        self.expert_actions = torch.FloatTensor(self.expert_actions)
        self.expert_states = torch.FloatTensor(self.expert_states)

    def set_ensemble(self, model_paths):
        for model_path in model_paths:
            imitator = Policy(self.state_dim, self.action_dim)
            imitator.load_state_dict(
                torch.load(model_path))
            self.ensemble.append(imitator)

    def select_action(self, state):
        return self.policy.actor.select_action(state)

    def train_minibatch(self, batch_size=100):
        """
        Update the imitator policy from expert minibatch using BC loss
        :param minibatch_size: size of the expert minibatch
        :return:
        """
        indices = np.random.choice(len(self.expert_traj), batch_size)
        expert_minibatch = self.expert_traj[indices]

        expert_state_actions = torch.DoubleTensor(expert_minibatch)
        expert_states = expert_state_actions[:,:self.state_dim].to(self.device)
        expert_actions = expert_state_actions[:,self.state_dim:].to(self.device)

        predicted_actions = self.policy.actor.select_action(expert_states)

        self.policy.actor_optimizer.zero_grad()
        loss = self.policy_minibatch_loss(predicted_actions, expert_actions)
        loss.backward()
        self.policy.actor_optimizer.step()

        return loss.to('cpu').detach().numpy()

    def train(self, replay_buffer, agent, batch_size=100):
        """
        Train DRIL. One step of supervised loss and uncertainty cost, respectively.
        :param batch:
        :param minibatch_size:
        :return:
        """

        # Sample replay buffer / batch
        state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state_np).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state_np).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device)

        # supervised lost
        predicted_actions = self.policy.actor.select_action(state)
        self.policy.actor_optimizer.zero_grad()
        loss = self.policy_minibatch_loss(predicted_actions, action)
        loss.backward()
        self.policy.actor_optimizer.step()

        # uncertainty cost
        # pretty hacky right now...
        memory = agent.simple_collect_samples(2048)
        batch = np.array(memory.memory)
        states, next_states, actions, rewards, masks = [], [], [], [], []

        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            masks.append(batch[i][2])
            next_states.append(batch[i][3])
            rewards.append(batch[i][4])

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        masks = np.array(masks)

        new_reward = []
        for imitator in self.ensemble:
            with torch.no_grad():
                action_log_probs = imitator.get_log_prob(torch.FloatTensor(states), torch.FloatTensor(actions))
                new_reward.append(action_log_probs)
        new_reward = torch.stack(new_reward, dim=2)
        rewards = - torch.var(new_reward, dim=2)
        batch = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'masks': masks,
            'rewards': rewards
        }
        self.policy.train(batch)

        # return {"bc_loss":supervised_loss,
        #         "uncertainty_cost": uncertainty_cost.to('cpu').detach().numpy()}


class DRIL_PPO(object):
    def __init__(self, args, state_dim, action_dim, is_dict_action=False):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = args.device
        # self.config = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_dict_action = is_dict_action

        # BC imitator ensemble
        self.ensemble = []

        # initialize actor
        self.policy = PPO(args, state_dim, action_dim, is_dict_action)
        self.policy_minibatch_loss = nn.MSELoss() # ???

    def set_expert(self, expert_traj):
        """
        Set the expert trajectories.
        :param expert_traj:
        :param num_traj
        :return:
        """
        self.expert_traj = expert_traj
        self.expert_states = []
        self.expert_actions = []
        for i in range(len(self.expert_traj)):
            self.expert_states.append(self.expert_traj[i][0])
            self.expert_actions.append(self.expert_traj[i][2])
        self.expert_actions = torch.FloatTensor(self.expert_actions)
        self.expert_states = torch.FloatTensor(self.expert_states)

    def set_ensemble(self, model_paths):
        for model_path in model_paths:
            imitator = Policy(self.state_dim, self.action_dim)
            imitator.load_state_dict(
                torch.load(model_path))
            self.ensemble.append(imitator)

    def select_action(self, state):
        return self.policy.actor.select_action(state)

    def train_minibatch(self, batch_size=100):
        """
        Update the imitator policy from expert minibatch using BC loss
        :param minibatch_size: size of the expert minibatch
        :return:
        """
        indices = np.random.choice(len(self.expert_traj), batch_size)
        expert_minibatch = self.expert_traj[indices]

        expert_state_actions = torch.DoubleTensor(expert_minibatch)
        expert_states = expert_state_actions[:,:self.state_dim].to(self.device)
        expert_actions = expert_state_actions[:,self.state_dim:].to(self.device)

        predicted_actions = self.policy.actor.select_action(expert_states)

        self.policy.actor_optimizer.zero_grad()
        loss = self.policy_minibatch_loss(predicted_actions, expert_actions)
        loss.backward()
        self.policy.actor_optimizer.step()

        return loss.to('cpu').detach().numpy()

    def train(self, replay_buffer, agent, batch_size=100):
        """
        Train DRIL. One step of supervised loss and uncertainty cost, respectively.
        :param batch:
        :param minibatch_size:
        :return:
        """

        # Sample replay buffer / batch
        state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state_np).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state_np).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device)

        # supervised lost
        predicted_actions = self.policy.actor.select_action(state)
        self.policy.actor_optimizer.zero_grad()
        loss = self.policy_minibatch_loss(predicted_actions, action)
        loss.backward()
        self.policy.actor_optimizer.step()

        # uncertainty cost
        # pretty hacky right now...
        memory = agent.simple_collect_samples(2048)
        batch = np.array(memory.memory)
        states, next_states, actions, rewards, masks = [], [], [], [], []

        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            masks.append(batch[i][2])
            next_states.append(batch[i][3])
            rewards.append(batch[i][4])

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        masks = np.array(masks)

        new_reward = []
        for imitator in self.ensemble:
            with torch.no_grad():
                action_log_probs = imitator.get_log_prob(torch.FloatTensor(states), torch.FloatTensor(actions))
                new_reward.append(action_log_probs)
        new_reward = torch.stack(new_reward, dim=2)
        rewards = - torch.var(new_reward, dim=2)
        batch = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'masks': masks,
            'rewards': rewards
        }
        self.policy.train(batch)

        # return {"bc_loss":supervised_loss,
        #         "uncertainty_cost": uncertainty_cost.to('cpu').detach().numpy()}