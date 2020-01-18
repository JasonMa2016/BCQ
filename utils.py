import numpy as np
import torch

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		return (np.array(state), 
			np.array(next_state), 
			np.array(action), 
			np.array(reward).reshape(-1, 1), 
			np.array(done).reshape(-1, 1))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy", allow_pickle=True)

	def set_expert(self, expert_traj):
		self.storage = expert_traj

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=50):
	rewards = []
	for _ in range(eval_episodes):
		episode_reward = 0
		obs = env.reset()
		done = False
		while not done:
			try:
				action = policy.select_action(np.array(obs))
				obs, reward, done, _ = env.step(action)
			except:
				action = policy.select_action(torch.FloatTensor(obs).unsqueeze(dim=0))
				obs, reward, done, _ = env.step(action.detach().numpy())
			episode_reward += reward
		rewards.append(episode_reward)
	rewards = np.array(rewards)
	return rewards