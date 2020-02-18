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
		# ind = np.random.choice(batch_size, batch_size, replace=False)
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

	def save_rewards(self, filename, rewards):
		np.save("./buffers/"+filename+ "_rewards" + ".npy", rewards)

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
		with torch.no_grad():
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


def evaluate_policy_with_noise(env, policy, eval_episodes=50, noise=0.3):
	rewards = []
	for _ in range(eval_episodes):
		episode_reward = 0
		obs = env.reset()
		done = False
		with torch.no_grad():
			while not done:
				try:
					action = policy.select_action(np.array(obs))
					if noise != 0:
						action = (action + np.random.normal(0, noise, size=env.action_space.shape[0])).clip(
							env.action_space.low, env.action_space.high)
					obs, reward, done, _ = env.step(action)
				except:
					action = policy.select_action(torch.FloatTensor(obs).unsqueeze(dim=0))
					obs, reward, done, _ = env.step(action.detach().numpy())
				episode_reward += reward
			rewards.append(episode_reward)
	rewards = np.array(rewards)
	return rewards


def collect_trajectories_rewards(expert_trajs, num_good_traj=5, num_bad_traj=3, good=False):

	# trajs = np.concatenate((expert_trajs[:num_good_traj], expert_trajs[-num_bad_traj:]), axis=0)
	# rewards = np.concatenate((expert_rewards[:num_bad_traj]), expert_rewards[-num_bad_traj:], axis=0)
	trajs = expert_trajs[:num_good_traj]
	flat_expert_trajs = []
	for expert_traj in trajs:
		for state_action in expert_traj:
			flat_expert_trajs.append(state_action)
	flat_trajs = {
		'mixed':list(flat_expert_trajs),
		'good':list(flat_expert_trajs)
	}

	for expert_traj in expert_trajs[-num_bad_traj:]:
		for state_action in expert_traj:
			flat_trajs['mixed'].append(state_action)
	# print(len(flat_trajs['mixed']), len(flat_trajs['good']))

	n = len(flat_trajs['mixed'])
	index = num_good_traj

	equal = False
	while not equal:
		current_traj = expert_trajs[index]
		for state_action in current_traj:
			flat_trajs['good'].append(state_action)
			if len(flat_trajs['good']) == n:
				equal = True
				break
		index += 1
	# print(len(flat_trajs['mixed']), len(flat_trajs['good']))

	if good:
		return flat_trajs['good']
	return flat_trajs['mixed']


def to_device(device, *args):
	return [x.to(device) for x in args]