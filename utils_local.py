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

    def sample_all(self):
        ind = len(self.storage)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in range(ind):
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
def evaluate_policy(env, policy, running_state,
                               eval_episodes=10, BCQ=False):
    rewards = []
    np.random.seed(2020)
    torch.manual_seed(2020)
    env.seed(2020)

    for _ in range(eval_episodes):
        episode_reward = 0
        obs = env.reset()
        obs = running_state(obs)
        done = False
        with torch.no_grad():
            while not done:
                state_var = torch.FloatTensor(obs).unsqueeze(0)
                if not BCQ:
                    action = policy(state_var)[0][0].detach().numpy()
                else:
                    action = policy.select_action(state_var)
                action = action.astype(np.float64)
                new_obs, reward, done, _ = env.step(action)
                new_obs = running_state(new_obs)
                episode_reward += reward
                obs = new_obs

            rewards.append(episode_reward)
    rewards = np.array(rewards)
    return rewards


def evaluate_policy_with_noise(env, policy, running_state,
                               eval_episodes=10,
                               noise1=0.3,
                               noise2=0.3, BCQ=False):
    rewards = []
    np.random.seed(2020)
    torch.manual_seed(2020)
    env.seed(2020)

    for _ in range(eval_episodes):
        episode_reward = 0
        obs = env.reset()
        obs = running_state(obs)
        done = False
        with torch.no_grad():
            while not done:
                state_var = torch.FloatTensor(obs).unsqueeze(0)
                # select deterministically

                if np.random.uniform(0, 1) < noise1:
                    action = env.action_space.sample()
                else:
                    if not BCQ:
                        action = policy(state_var)[0][0].detach().numpy()
                    else:
                        action = policy.select_action(state_var)
                    if noise2 != 0:
                        action = (action + np.random.normal(0, noise2, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)
                action = action.astype(np.float64)
                new_obs, reward, done, _ = env.step(action)
                new_obs = running_state(new_obs)
                episode_reward += reward

                obs = new_obs
            rewards.append(episode_reward)
    rewards = np.array(rewards)
    return rewards


def collect_trajectories_rewards(expert_trajs, num_trajs=5, type='good'):

    # trajs = np.concatenate((expert_trajs[:num_good_traj], expert_trajs[-num_bad_traj:]), axis=0)
    # rewards = np.concatenate((expert_rewards[:num_bad_traj]), expert_rewards[-num_bad_traj:], axis=0)
    trajs = expert_trajs[:num_trajs]
    flat_expert_trajs = []
    for expert_traj in trajs:
        for state_action in expert_traj:
            flat_expert_trajs.append(state_action)
    flat_trajs = {
        'imperfect': [],
        'mixed':list(flat_expert_trajs),
        'good':list(flat_expert_trajs)
    }

    for expert_traj in expert_trajs[-num_trajs:]:
        for state_action in expert_traj:
            flat_trajs['mixed'].append(state_action)
            flat_trajs['imperfect'].append(state_action)

    # print(len(flat_trajs['mixed']), len(flat_trajs['good']))

    n = len(flat_trajs['mixed'])
    index = num_trajs

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

    if type == 'good':
        return flat_trajs['good']
    elif type == 'mixed':
        return flat_trajs['mixed']
    else:
        return flat_trajs['imperfect']

def evaluate_model(env, model, running_state=None, num_trajs=50, verbose=True, render=False, floattensor=False):
    """seeding"""
    np.random.seed(2020)
    torch.manual_seed(2020)
    env.seed(2020)
    episodes_rewards = []
    episodes_timesteps = []
    for i in range(num_trajs):
        state = env.reset()
        state = running_state(state)

        episode_rewards = 0
        episode_steps = 0
        for t in range(10000):
            if floattensor:
                state_var = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_var = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                # select deterministically
                action = model(state_var)[0][0].numpy()
                # action = imitator.select_action(state_var)[0].numpy()
            action = int(action) if model.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            episode_rewards += reward
            episode_steps += 1
            if render:
                env.render()
            if done:
                break
            state = next_state

        episodes_rewards.append(episode_rewards)
        episodes_timesteps.append(episode_steps)
        # if args.verbose:
        #     print('{}\tsteps: {}\t reward: {:.2f}'.format(
        #         i, episode_steps, episode_rewards))

    episodes_rewards = np.array(episodes_rewards)
    episodes_timesteps = np.array(episodes_timesteps)
    if verbose:
        print("{} Trajectories \t reward avg: {:.2f} \t reward std: {:.2f}".format(num_trajs,
                                                                       episodes_rewards.mean(),
                                                                       episodes_rewards.std()))
    return {'rewards': episodes_rewards,
            'timesteps': episodes_timesteps}


def to_device(device, *args):
    return [x.to(device) for x in args]