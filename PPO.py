import torch
import numpy as np
import math
import pickle
import time

from core.common import *
from models.mlp_policy import Policy
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_critic import Value

from models.cnn_policy import CNNPolicy
from models.cnn_discriminator import CNNDiscriminator
from models.cnn_critic import CNNCritic


class PPO(object):
    def __init__(self, args, state_dim, action_dim, is_dict_action=False, is_atari=False):

        self.device = args.device
        self.config = args
        if is_atari:
            self.actor = CNNPolicy(state_dim, action_dim).to(self.device)
            self.critic = CNNCritic(state_dim).to(self.device)
        else:
            self.actor = DiscretePolicy(state_dim, action_dim).to(self.device) if is_dict_action else \
                Policy(state_dim, action_dim, log_std=self.config.log_std).to(self.device)
            self.critic = Value(state_dim).to(self.device)

        # initialize optimizer for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

        # optimization epoch number and batch size for PPO
        self.optim_epochs = 10
        self.optim_batch_size = 64

    def train(self, batch):
        """
        Train the policy using the given batch.
        :param batch:
        :return:
        """

        states = torch.DoubleTensor(np.stack(batch.state)).to(self.device)
        actions = torch.DoubleTensor(np.stack(batch.action)).to(self.device)
        rewards = torch.DoubleTensor(np.stack(batch.reward)).to(self.device)
        masks = torch.DoubleTensor(np.stack(batch.mask)).to(self.device)

        with torch.no_grad():
            values = self.critic(states)
            fixed_log_probs = self.actor.get_log_prob(states, actions)

        # get advantage estimation from the trajectories
        advantages, returns = estimate_advantages(rewards, masks, values, self.config.gamma, self.config.tau, self.device)

        # compute minibatch size
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))

        # PPO updates
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
                self.ppo_step(states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)

    def ppo_step(self, states, actions, returns, advantages, fixed_log_probs):
        """
        A PPO policy gradient update step.
        :param states:
        :param actions:
        :param returns:
        :param advantages:
        :param fixed_log_probs:
        :return:
        """
        # update critic, for now assume one epoch
        values_pred = self.critic(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in self.critic.parameters():
            value_loss += param.pow(2).sum() * self.config.l2_reg
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update actor
        log_probs = self.actor.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        self.actor.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_optimizer.step()
