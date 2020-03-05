import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym

from models.mlp_policy import Policy
from utils_local import *


def drbcq_performance_random(model='DRBCQ', env_name="Walker2d-v2", num_trajs=5, seed=0):
    '''
    Compare uncertainty cost vs. random reward
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''

    fig, ax = plt.subplots(figsize=(8, 4))
    types = ['DRBCQ', 'DRBCQ-mixed','random']
    for type in types:
        performance = []
        for seed in range(5):

            if type == 'random':
                file_name = "./results/" + "{}_{}_traj{}_seed{}_good_random".format(model, env_name, num_trajs, seed,
                                                                           type)
            elif type =='DRBCQ-mixed':
                file_name = "./results/" + "{}_{}_traj{}_seed{}_mixed".format(model, env_name, num_trajs, seed,
                                                                           type)
            else:
                file_name = "./results/" + "{}_{}_traj{}_seed{}_good".format(model, env_name, num_trajs, seed,
                                                                           type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            performance.append(np.mean(reward, axis=1))

        performance = np.array(performance)
        # print(performance.shape)
        perf_mu = np.mean(performance, axis=0)
        # print(perf_mu.shape)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label=type)
        ax.fill_between(timestep, perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    performance = []
    for seed in range(5):
        file_name = "./results/" + "BCQ_{}_traj{}_seed{}_good".format(env_name, num_trajs, seed,
                                                                       type)
        reward = np.load(file_name + '_rewards.npy')
        timestep = np.load(file_name + '_timesteps.npy')
        performance.append(np.mean(reward, axis=1))

    performance = np.array(performance)
    # print(performance.shape)
    perf_mu = np.mean(performance, axis=0)
    # print(perf_mu.shape)
    perf_std = np.std(performance, axis=0)
    ax.plot(timestep, perf_mu, label='BCQ')
    ax.fill_between(timestep, perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('DRBCQ {} Reward Ablation Plot'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/DRBCQ_{}_reward_ablation.png'.format(env_name))


def bc_performance(env_name='Walker2d-v2', num_trajs=5,seed=0):
    types = ['good', 'mixed']
    for type in types:
        fig, ax = plt.subplots(figsize=(8, 4))
        performance = []
        for sample in range(5):
            file_name = "./results/" + "BC_{}_traj{}_seed{}_sample{}_{}".format(env_name, num_trajs,
                                                                                seed, sample, type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            print(reward)
            performance.append(reward)

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=0)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label=type)
        ax.fill_between(timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

        ax.legend(loc='best')
        ax.set_title('BC {} Reward '.format(env_name))
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Rewards')
        # plt.show()
        plt.savefig('plots/BC_{}_reward_plot.png'.format(env_name))


if __name__ == "__main__":
    # drbcq_performance_random()
    bc_performance()