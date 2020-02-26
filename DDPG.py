import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils_local
from BC import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a)) 
		return a


# Returns a Q-value for given state/action pair
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		q = self.l3(q)
		return q


class DDPG(object):
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


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations=500, batch_size=100, discount=0.99, tau=0.005): 

		for it in range(iterations):

			# Each of these are batches 
			state, next_state, action, reward, done = replay_buffer.sample(batch_size)
			state 		= torch.FloatTensor(state).to(device)
			action 		= torch.FloatTensor(action).to(device)
			next_state 	= torch.FloatTensor(next_state).to(device)
			reward 		= torch.FloatTensor(reward).to(device)
			done 		= torch.FloatTensor(1 - done).to(device)

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


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class DDPG_DRIL(object):
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

	def set_ensemble(self, model_paths):
		self.ensemble = []
		for model_path in model_paths:
			imitator = Policy(self.state_dim, self.action_dim)
			imitator.load_state_dict(
				torch.load(model_path))
			self.ensemble.append(imitator)

	# def bc_train(self, replay_buffer, iterations=500, batch_size=100):
	# 	# Sample replay buffer / batch
	# 	for it in range(iterations):
	# 		state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
	# 		state = torch.FloatTensor(state_np).to(self.device)
	# 		action = torch.FloatTensor(action).to(self.device)
	# 		next_state = torch.FloatTensor(next_state_np).to(self.device)
	# 		reward = torch.FloatTensor(reward).to(self.device)
	# 		done = torch.FloatTensor(1 - done).to(self.device)
	#
	# 		# supervised lost
	# 		predicted_actions = self.actor(state)
	# 		# predicted_actions = self.select_action(state)
	# 		self.actor_optimizer.zero_grad()
	# 		loss = self.supervised_loss(predicted_actions, action)
	# 		loss.backward()
	# 		self.actor_optimizer.step()

	def train(self, replay_buffer, iterations=500, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Each of these are batches
			state, next_state, action, reward, done = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(1 - done).to(device)

			new_reward = []
			for imitator in self.ensemble:
				with torch.no_grad():
					action_probs = imitator.get_log_prob(torch.FloatTensor(state), torch.FloatTensor(action))
					new_reward.append(action_probs)
			new_reward = torch.stack(new_reward, dim=2)
			reward = - torch.var(new_reward, dim=2)
			reward = torch.FloatTensor(reward).to(device)

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

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

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

	def train(self, replay_buffer, expert=True, iterations=500, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Each of these are batches
			state, next_state, action, reward, done = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(1 - done).to(device)

			reward = - torch.FloatTensor(np.ones(reward.size())).to(device)
			if expert:
				reward = - reward

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

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class DDPG_SQIL_ORIGINAL(object):
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

	def bc_train(self, replay_buffer, iterations=500, batch_size=100):
		# Sample replay buffer / batch
		for it in range(iterations):
			state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(state_np).to(self.device)
			action = torch.FloatTensor(action).to(self.device)
			next_state = torch.FloatTensor(next_state_np).to(self.device)
			reward = torch.FloatTensor(reward).to(self.device)
			done = torch.FloatTensor(1 - done).to(self.device)

			# supervised lost
			predicted_actions = self.actor(state)
			# predicted_actions = self.select_action(state)
			self.actor_optimizer.zero_grad()
			loss = self.supervised_loss(predicted_actions, action)
			loss.backward()
			self.actor_optimizer.step()

	def train(self, replay_buffer, expert=True, iterations=500, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Each of these are batches
			state, next_state, action, reward, done = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(1 - done).to(device)

			reward = torch.FloatTensor(np.zeros(reward.size())).to(device)
			if expert:
				reward = torch.FloatTensor(np.ones(reward.size())).to(device)

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

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))