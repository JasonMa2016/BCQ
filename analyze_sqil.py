import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym

from models.mlp_policy import Policy
from utils_local import *

def plot_model_reward_over_timesteps(env_name='Walker2d-v2', model_name='BC', num_trajs=5, seed=0, type='mixed'):
    file_name = 'SQIL_DDPG_NEW_{}_traj{}_seed{}_{}'.format(env_name, num_trajs, seed, type)
    data_name = 'results_sqil/' + file_name
    running_reward = np.load(data_name + '_rewards.npy')
    evaluation_reward = np.load(data_name + '_evaluation_rewards.npy')
    timestep = np.load(data_name + '_timesteps.npy')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestep, running_reward, label='running')
    ax.plot(timestep, evaluation_reward, label='evaluation')
    # file_name = 'results/SQIL_ORIGINAL_DDPG_{}_traj{}_seed{}_{}'.format(env_name, num_trajs, seed, type)
    # reward = np.load(file_name + '_rewards.npy')
    # timestep = np.load(file_name + '_timesteps.npy')
    #
    # ax.plot(timestep, reward, label='original')

    ax.set_title(file_name)
    ax.legend(loc='best')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rewards')
    plt.savefig('plots/{}.png'.format(file_name))
    plt.close()
    return


if __name__ == "__main__":
    plot_model_reward_over_timesteps(type='good')