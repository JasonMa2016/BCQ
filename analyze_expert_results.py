import numpy as np
import argparse
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import torch
import gym

import utils_local


def analyze_expert_trajectory(num_trajs=100, env_name='Hopper-v2',seed=0, expert_type='PPO'):
    file_name = "buffers/{}_traj{}_{}_{}".format(expert_type, num_trajs, env_name, seed)
    expert_trajs = np.load(file_name + '.npy', allow_pickle=True)
    expert_rewards = np.load(file_name + '_rewards.npy')
    expert_lengths = [len(i) for i in expert_trajs]
    fig, axs = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
    axs[0].plot([i for i in range(len(expert_rewards))], expert_rewards)
    axs[1].plot([i for i in range(len(expert_rewards))], expert_lengths)
    axs[0].set_title('{} {} Expert Trajectories Performance Distribution'.format(env_name, expert_type))
    axs[1].set_title('{} {} Expert Trajectories Duration Distribution'.format(env_name, expert_type))
    axs[0].set_ylabel('Cumulative Rewards')
    axs[1].set_ylabel('Episode Durations')

    fig.savefig('plots/{}_{}_Expert_Distribution'.format(expert_type, env_name))
    plt.close(fig)



def analyze_expert_training(model_name='DDPG', env_name='Hopper-v2', seed=0):
    file_name = 'expert_results/{}_{}_{}'.format(model_name, env_name, seed)
    reward = np.load(file_name + '_rewards.npy')
    timestep = np.load(file_name + '_timesteps.npy')
    print(reward[-10:])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestep, reward, label=env_name)
    plt.show()


if __name__ == "__main__":
    # envs = ['Hopper-v2', 'HalfCheetah-v2', 'Humanoid-v2']
    # for env in envs:
    #     analyze_expert_trajectory(env_name=env)
    analyze_expert_trajectory(env_name='Walker2d-v2')
    analyze_expert_trajectory(env_name='Humanoid-v2')
    analyze_expert_trajectory(env_name='Hopper-v2')
    # analyze_expert_training(env_name='Humanoid-v2')