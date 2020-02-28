import numpy as np
import argparse
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import torch
import gym

import utils_local

def analyze_expert_trajectory(num_trajs=100, env_name='Hopper-v2',seed=0):
    file_name = "buffers/Robust_traj{}_{}_{}".format(num_trajs, env_name, seed)
    expert_rewards = np.load(file_name + '_rewards.npy')
    print(expert_rewards)

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
    analyze_expert_training(env_name='Humanoid-v2')