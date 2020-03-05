import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils_local
from BC import Policy
from DDPG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG_SQIL(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.supervised_loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, expert_buffer, iterations=250, batch_size=100, discount=0.99, tau=0.005, original=False):

        for it in range(iterations):
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            expert_state, expert_next_state, expert_action, expert_reward, expert_done = expert_buffer.sample(batch_size)
            expert_state = torch.FloatTensor(expert_state).to(device)
            expert_action = torch.FloatTensor(expert_action).to(device)
            expert_next_state = torch.FloatTensor(expert_next_state).to(device)
            expert_reward = torch.FloatTensor(expert_reward).to(device)
            expert_done = torch.FloatTensor(1 - expert_done).to(device)


            state = torch.cat([state, expert_state], dim=0)
            action = torch.cat([action, expert_action], dim=0)
            next_state = torch.cat([next_state, expert_next_state], dim=0)
            done = torch.cat([done, expert_done], dim=0)

            if original:
                reward = torch.FloatTensor(np.zeros(reward.size())).to(device)
                expert_reward = torch.FloatTensor(np.ones(reward.size())).to(device)
            else:
                reward = - torch.FloatTensor(np.ones(reward.size())).to(device)
                expert_reward = torch.FloatTensor(np.ones(expert_reward.size())).to(device)
            reward = torch.cat([reward, expert_reward], dim=0)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    # def train(self, replay_buffer, expert=True, iterations=500, batch_size=100, discount=0.99, tau=0.005, original=False):
    #
    #     for it in range(iterations):
    #
    #         # Each of these are batches
    #         state, next_state, action, reward, done = replay_buffer.sample(batch_size)
    #         state = torch.FloatTensor(state).to(device)
    #         action = torch.FloatTensor(action).to(device)
    #         next_state = torch.FloatTensor(next_state).to(device)
    #         reward = torch.FloatTensor(reward).to(device)
    #         done = torch.FloatTensor(1 - done).to(device)
    #
    #         if not original:
    #             reward = - torch.FloatTensor(np.ones(reward.size())).to(device)
    #             if expert:
    #                 reward = - reward
    #         else:
    #             reward = torch.FloatTensor(np.zeros(reward.size())).to(device)
    #             if expert:
    #                 reward = torch.FloatTensor(np.ones(reward.size())).to(device)
    #
    #         # Compute the target Q value
    #         target_Q = self.critic_target(next_state, self.actor_target(next_state))
    #         target_Q = reward + (done * discount * target_Q).detach()
    #
    #         # Get current Q estimate
    #         current_Q = self.critic(state, action)
    #
    #         # Compute critic loss
    #         critic_loss = F.mse_loss(current_Q, target_Q)
    #
    #         # Optimize the critic
    #         self.critic_optimizer.zero_grad()
    #         critic_loss.backward()
    #         self.critic_optimizer.step()
    #
    #         # Compute actor loss
    #         actor_loss = -self.critic(state, self.actor(state)).mean()
    #
    #         # Optimize the actor
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer.step()
    #
    #         # Update the frozen target models
    #         for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    #
    #         for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
    #             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def save_best(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor_best.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic_best.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


# class DDPG_SQIL_ORIGINAL(object):
#     def __init__(self, state_dim, action_dim, max_action):
#         self.actor = Actor(state_dim, action_dim, max_action).to(device)
#         self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
#
#         self.critic = Critic(state_dim, action_dim).to(device)
#         self.critic_target = Critic(state_dim, action_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
#
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.supervised_loss = nn.MSELoss()
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def select_action(self, state):
#         state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         return self.actor(state).cpu().data.numpy().flatten()
#
#     def train(self, replay_buffer, expert=True, iterations=500, batch_size=100, discount=0.99, tau=0.005):
#
#         for it in range(iterations):
#
#             # Each of these are batches
#             state, next_state, action, reward, done = replay_buffer.sample(batch_size)
#             state = torch.FloatTensor(state).to(device)
#             action = torch.FloatTensor(action).to(device)
#             next_state = torch.FloatTensor(next_state).to(device)
#             reward = torch.FloatTensor(reward).to(device)
#             done = torch.FloatTensor(1 - done).to(device)
#
#             reward = torch.FloatTensor(np.zeros(reward.size())).to(device)
#             if expert:
#                 reward = torch.FloatTensor(np.ones(reward.size())).to(device)
#
#             # Compute the target Q value
#             target_Q = self.critic_target(next_state, self.actor_target(next_state))
#             target_Q = reward + (done * discount * target_Q).detach()
#
#             # Get current Q estimate
#             current_Q = self.critic(state, action)
#
#             # Compute critic loss
#             critic_loss = F.mse_loss(current_Q, target_Q)
#
#             # Optimize the critic
#             self.critic_optimizer.zero_grad()
#             critic_loss.backward()
#             self.critic_optimizer.step()
#
#             # Compute actor loss
#             actor_loss = -self.critic(state, self.actor(state)).mean()
#
#             # Optimize the actor
#             self.actor_optimizer.zero_grad()
#             actor_loss.backward()
#             self.actor_optimizer.step()
#
#             # Update the frozen target models
#             for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#                 target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#
#             for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#                 target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#
#     def save(self, filename, directory):
#         torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
#         torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
#
#     def load(self, filename, directory):
#         self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
#         self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))