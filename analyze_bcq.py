import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym

from models.mlp_policy import Policy
from utils_local import *


def drbcq_performance_random(model='DRBCQ', env_name="Walker2d-v2", num_trajs=5):
    '''
    Compare uncertainty cost vs. random reward
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''

    fig, ax = plt.subplots(figsize=(8, 4))
    types = ['DRBCQ', 'DRBCQ-mixed','random', 'random-mixed']
    types = ['DRBCQ', 'DRBCQ-mixed']
    types = ['DRBCQ']
    for type in types:
        performance = []
        for seed in range(5):

            # if type == 'random':
            #     file_name = "./results/" + "{}_{}_traj{}_seed{}_good_random".format(model, env_name, num_trajs, seed,
            #                                                                type)
            # elif type == 'random-mixed':
            #     file_name = "./results/" + "{}_{}_traj{}_seed{}_mixed_random".format(model, env_name, num_trajs, seed,
            #                                                                type)
            if type =='DRBCQ-mixed':
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

    types = ['good', 'mixed']
    types = ['good']
    for type in types:
        performance = []
        for seed in range(5):
            file_name = "./results/" + "BCQ_{}_traj{}_seed{}_{}".format(env_name, num_trajs, seed,
                                                                           type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            performance.append(np.mean(reward, axis=1))

        performance = np.array(performance)
        # print(performance.shape)
        perf_mu = np.mean(performance, axis=0)
        # print(perf_mu.shape)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label='BCQ' + ' ' + type)
        ax.fill_between(timestep, perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    ax.legend(loc='best')
    # ax.set_title('DRBCQ {} Reward Ablation Plot'.format(env_name))
    ax.set_title('DRBCQ vs. BCQ {} Evaluation'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/DRBCQ_{}_reward_ablation.png'.format(env_name))

def drbcq_performance_ablation(env_name="Walker2d-v2", num_trajs=5, type='good'):
    '''
    Compare uncertainty cost vs. random reward
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''

    fig, ax = plt.subplots(figsize=(8, 4))

    models = ['DRBCQ', 'BCQ', 'BCQ - random']
    for model in models:
        performance = []
        for seed in range(5):
            if model == 'BCQ - random':

                file_name = "./results/" + "DRBCQ_{}_traj{}_seed{}_{}_random".format(env_name, num_trajs, seed,
                                                                           type)
            else:
                file_name = "./results/" + "{}_{}_traj{}_seed{}_{}".format(model, env_name, num_trajs, seed,
                                                                           type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            performance.append(np.mean(reward, axis=1))

        performance = np.array(performance)
        # print(performance.shape)
        perf_mu = np.mean(performance, axis=0)
        # print(perf_mu.shape)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label=model)
        ax.fill_between(timestep, perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)


    ax.legend(loc='best')
    ax.set_title('DRBCQ {} Reward Ablation Plot'.format(env_name))
    # ax.set_title('DRBCQ'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/DRBCQ_{}_random_reward_ablation_{}.png'.format(env_name, type))


def bc_performance(env_name='Walker2d-v2', num_trajs=5,seeds=5):
    types = ['good', 'mixed']
    fig, ax = plt.subplots(figsize=(8, 4))

    for type in types:
        performance = []
        for seed in range(seeds):
        # for sample in range(5):
        #     file_name = "./results/" + "BC_{}_traj{}_seed{}_sample{}_{}".format(env_name, num_trajs,
        #                                                                         seed, sample, type)

            file_name = "./results/" + "BC_{}_batch100_traj{}_seed{}_{}".format(env_name, num_trajs,
                                                                                seed, type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            timestep = timestep * 10
            performance.append(np.mean(reward, axis=1))

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=0)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label=type)
        ax.fill_between(timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('BC {} Running Performance '.format(env_name))
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Rewards')
    plt.savefig('plots/BC_{}_batch100_traj{}_reward_plot.png'.format(env_name, num_trajs))

def drbcq_performance(env_name='Walker2d-v2', num_trajs=5,seeds=5):
    types = ['good', 'mixed']
    fig, ax = plt.subplots(figsize=(8, 4))

    for type in types:
        performance = []
        for seed in range(seeds):
        # for sample in range(5):
        #     file_name = "./results/" + "BC_{}_traj{}_seed{}_sample{}_{}".format(env_name, num_trajs,
        #                                                                         seed, sample, type)

            file_name = "./results/" + "BCQ_{}_traj{}_seed{}_{}".format(env_name, num_trajs,
                                                                                seed, type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            performance.append(np.mean(reward, axis=1))

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=0)
        perf_std = np.std(performance, axis=0)
        ax.plot(timestep, perf_mu, label=type)
        ax.fill_between(timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('DRBCQ {} Reward '.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    plt.savefig('plots/BCQ_{}_traj{}_reward_plot.png'.format(env_name, num_trajs))

def compare_batch_models(models=['DRBCQ', 'BC'], env_name='Walker2d-v2', num_trajs=5):
    fig, ax = plt.subplots(figsize=(8, 4))
    types = ['good']
    for model in models:
        for type in types:
            performance = []
            if model == 'BC':
                for seed in range(3):
                    file_name = "./results/" + "BC_{}_batch100_traj{}_seed{}_{}".format(env_name, num_trajs,
                                                                               seed, type)
                    if env_name == 'Humanoid-v2':
                        file_name = "./results/" + "BC_{}_traj{}_seed0_sample{}_{}".format(env_name, num_trajs,
                                                                               seed, type)


                    reward = np.load(file_name + '_rewards.npy')
                    timestep = np.load(file_name + '_timesteps.npy')
                    performance.append(np.mean(reward, axis=1))
            else:
                for seed in range(5):
                    file_name = "./results/" + "{}_{}_traj{}_seed{}_{}".format(model, env_name, num_trajs,
                                                                               seed, type)
                    reward = np.load(file_name + '_rewards.npy')
                    timestep = np.load(file_name + '_timesteps.npy')
                    performance.append(np.mean(reward, axis=1))

            performance = np.array(performance)
            perf_mu = np.mean(performance, axis=0)
            perf_std = np.std(performance, axis=0)
            ax.plot(timestep, perf_mu, label=model)
            ax.fill_between(timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('{} Batch Imitation Learning Evaluation'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    plt.savefig('plots/batch_{}_reward_plot.png'.format(env_name))

if __name__ == "__main__":
    # drbcq_performance_random(env_name='Hopper-v2')
    # drbcq_performance_random(env_name='Humanoid-v2')
    # drbcq_performance_random(env_name='Walker2d-v2')
    # envs = ['Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
    # for env in envs:
    #     for type in ['good', 'mixed']:
    #         drbcq_performance_ablation(env_name=env, type=type)
    bc_performance(env_name='Walker2d-v2', num_trajs=5, seeds=5)
    bc_performance(env_name='Hopper-v2', num_trajs=5, seeds=5)
    bc_performance(env_name='Humanoid-v2', num_trajs=5, seeds=5)
    # drbcq_performance(env_name='Walker2d-v2')
    # drbcq_performance(env_name='Hopper-v2')
    # drbcq_performance(env_name='Humanoid-v2')


    # compare_batch_models()
    # compare_batch_models(env_name='Hopper-v2')
    # compare_batch_models(env_name='Humanoid-v2')