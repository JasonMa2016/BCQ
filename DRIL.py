import time
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

from BC import Policy
from PPO import PPO


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class DRIL(object):
    def __init__(self, args, state_dim, action_dim, is_dict_action):
        self.device =  args.device
        self.config = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_dict_action = is_dict_action

        # BC imitator ensemble
        self.ensemble = []

        # initialize actor
        self.policy = PPO(args, state_dim, action_dim, is_dict_action)
        self.policy_minibatch_loss = nn.MSELoss() # ???

    def set_expert(self, expert_traj, num_trajs):
        """
        Set the expert trajectories.
        :param expert_traj:
        :return:
        """
        self.expert_traj_pool = expert_traj
        self.expert_traj = np.vstack(expert_traj[:num_trajs])

    def set_ensemble(self, model_paths):
        for model_path in model_paths:
            imitator = Policy(self.state_dim, self.action_dim)
            imitator.load_state_dict(
                torch.load(model_path))
            self.ensemble.append(imitator)

    def train_minibatch(self, minibatch_size=100):
        """
        Update the imitator policy from expert minibatch using BC loss
        :param minibatch_size: size of the expert minibatch
        :return:
        """
        indices = np.random.choice(len(self.expert_traj), minibatch_size)
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

    def train(self, batch):
        """
        Train DRIL. One step of supervised loss and uncertainty cost, respectively.
        :param batch:
        :param minibatch_size:
        :return:
        """
        # supervised_loss = self.train_minibatch()
        supervised_loss = 0
        # compute uncertainty cost
        states = np.stack(batch.state)
        actions = np.stack(batch.action)
        new_reward = []
        for imitator in self.ensemble:
            with torch.no_grad():
                action_log_probs = imitator.actor.get_log_prob(torch.DoubleTensor(states), torch.DoubleTensor(actions))
                action_probs = torch.exp(action_log_probs)
                new_reward.append(action_probs)
        new_reward = torch.stack(new_reward, dim=2)
        new_reward_var = torch.var(new_reward, dim=2)

        print(np.mean(batch.reward))
        batch = batch._replace(reward=new_reward_var)
        print(np.mean(batch.reward))
        # batch._replace(reward=-new_reward_var) # figure out if this is right
        uncertainty_cost = torch.mean(new_reward_var)
        # update using PPO
        batch = Transition(*zip(*[]))
        self.policy.train(batch)

        return {"bc_loss":supervised_loss,
                "uncertainty_cost": uncertainty_cost.to('cpu').detach().numpy()}