import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym
from utils_local import *
from BCQ import BCQ

def plot_model_best_performance_over_trajectories(expert_line=3750, env_name='Walker2d-v2', model_name='DRBCQ', buffer_type='PPO',
                                                  num_trajs=[1,3,5], seeds=[0,1,2,3,4], types=['good', 'mixed']):
    fig, ax = plt.subplots(figsize=(8, 4))

    # buffer_name = "%s_traj100_%s_0" % (buffer_type, env_name)
    #
    # expert_rewards = np.load("./buffers/" + buffer_name + "_rewards" + ".npy", allow_pickle=True)
    # expert_performance = np.mean(expert_rewards[:5])

    for i, type in enumerate(types):
        best_performance_all = []

        for traj in num_trajs:
            best_performance = []

            for seed in seeds:
                file_name = 'results/{}_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
                reward = np.load(file_name + '_rewards.npy')
                best_performance.append(max(np.mean(reward, axis=0)))
            best_performance_all.append(best_performance)
        performance = np.array(best_performance_all)

        perf_mu = np.mean(performance, axis=1)
        perf_std =np.std(performance, axis=1)

        print(perf_mu, perf_std)
        ax.plot(num_trajs, perf_mu, label=type, marker='v')
        ax.fill_between(num_trajs, perf_mu + 0.5 * perf_std, perf_mu - 0.5* perf_std, alpha=0.4)
    ax.axhline(y=expert_line, linestyle='--', label='expert')
    ax.legend(loc='best')
    ax.set_title('{} {} Best Performance '.format(model_name, env_name))
    ax.set_xlabel('Number of Expert Trajectories')
    ax.set_ylabel('Cumulative Rewards')
    plt.savefig('plots/{}_{}_best_performance.png'.format(model_name, env_name))
    plt.close()


def drbcq_performance(model_name='DRBCQ', env_name='Walker2d-v2', num_trajs=5,seeds=5):
    types = ['good', 'mixed']
    fig, ax = plt.subplots(figsize=(8, 4))

    for type in types:
        performance = []
        for seed in range(seeds):
        # for sample in range(5):
        #     file_name = "./results/" + "BC_{}_traj{}_seed{}_sample{}_{}".format(env_name, num_trajs,
        #                                                                         seed, sample, type)

            file_name = "./results/" + "{}_{}_traj{}_seed{}_{}".format(model_name, env_name, num_trajs,
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
    plt.savefig('plots/final_{}_{}_traj{}_reward_plot_final.png'.format(model_name, env_name, num_trajs))

def plot_distribution(env_name='Walker2d-v2', model_name='DRBCQ', sample_traj=50, traj=5, seed=0, type='good'):
    """
    Compare the distribution of reward/time-step of the expert and the imitator.
    :param env:
    :param imitator_model:
    :param num_trajs:
    :param seed:
    :param metric:
    :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(15,10), constrained_layout=True)
    gym_env = gym.make(env_name)

    # collect expert results
    expert_model_path = 'expert_models/{}_ppo_0.p'.format(env_name)
    expert, _, running_state, _ = pickle.load(open(expert_model_path, "rb"))
    expert_results = evaluate_model(gym_env, expert,
                                                running_state=running_state, num_trajs=sample_traj, verbose=False)

    imitator_name = '{}_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
    max_action = float(gym_env.action_space.high[0])
    imitator = BCQ(17, 6, max_action=max_action)
    imitator.load(imitator_name)

    # imitator_model_path = "imitator_models/{}_{}_traj{}_seed{}.p".format(model_name, model_name, traj, seed)
    # imitator = pickle.load(open(imitator_model_path, "rb"))[0]
    imitator_results= evaluate_model(gym_env, imitator, BCQ=True,
                                                running_state=running_state, num_trajs=sample_traj, verbose=False, floattensor=True)
    metrics = ['rewards', 'timesteps']
    for i, metric in enumerate(metrics):
        axs[i].hist(expert_results[metric], density=True, alpha=0.7, label='expert', bins=25)
        axs[i].hist(imitator_results[metric], density=True, alpha=0.5, label=model_name, bins=25)
        axs[i].set_title('{} {} Density Curve'.format(env_name, metric))
        axs[i].legend(loc='upper right')
        axs[i].set(xlabel='{}'.format(metric), ylabel='density')
    # for ax in axs.flat:
    #     ax.set(xlabel='{}'.format(metric), ylabel='density')

    fig.savefig('plots/final_{}_{}_density_comparison_plot.png'.format(model_name, env_name))
    plt.close(fig)


def analyze_bcq_with_noise(env_name='Walker2d-v2', model_name='DRBCQ', num_trajs=[1,3,5],
                                 seeds=[0,1,2,3,4], type='good', noise1=0, noise2=0.3):
    """
    Robustness test.
    :param env_name:
    :param model_name:
    :param num_trajs:
    :param seeds:
    :param type:
    :param noise1:
    :param noise2:
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    rewards = []
    rewards_noise = []
    expert_model_path = 'expert_models/{}_ppo_0.p'.format(env_name)
    expert, _, running_state, _ = pickle.load(open(expert_model_path, "rb"))

    for traj in num_trajs:
        traj_rewards = []
        traj_rewards_noise = []
        for seed in seeds:
            env = gym.make(env_name)

            # state_dim = env.observation_space.shape[0]
            # action_dim = env.action_space.shape[0]
            # max_action = float(env.action_space.high[0])

            imitator_name = '{}_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
            max_action = float(env.action_space.high[0])
            imitator = BCQ(17,6, max_action=max_action)
            imitator.load(imitator_name)

            # imitator.load_state_dict(torch.load('imitator_models/%s.p' % (imitator_name)))

            rewards_i = evaluate_policy(env, imitator, running_state, BCQ=True).mean()
            rewards_noise_i = evaluate_policy_with_noise(env, imitator, running_state, BCQ=True, noise1=noise1,
                                                         noise2=noise2).mean()

            traj_rewards.append(rewards_i)
            traj_rewards_noise.append(rewards_noise_i)

        rewards.append(traj_rewards)
        rewards_noise.append(traj_rewards_noise)

    rewards = np.array(rewards)
    rewards_noise = np.array(rewards_noise)
    lst = [rewards, rewards_noise]

    labels = ['no noise', 'noise']
    for label, array in zip(labels, lst):

        perf_mu = np.mean(array, axis=1)
        perf_std =np.std(array, axis=1)

        ax.plot(num_trajs, perf_mu, label=label, marker='v')
        ax.fill_between(num_trajs, perf_mu + 0.5 * perf_std, perf_mu - 0.5* perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('{} {} Noise Evaluation, epsilon={}'.format(env_name, model_name, noise1))
    ax.set_xlabel('Number of Expert Trajectories')
    ax.set_ylabel('Cumulative Rewards')
    plt.savefig('plots/final_{}_{}_noise1{}_noise2{}.png'.format(model_name, env_name, noise1, noise2))
    plt.close()


if __name__ == "__main__":
    envs = ['Walker2d-v2']
    for env in envs:
        # plot_model_best_performance_over_trajectories(env_name=env)
        # drbcq_performance(env_name=env)

        # plot_distribution(env_name=env)
        analyze_bcq_with_noise(env_name=env)
        analyze_bcq_with_noise(env_name=env, noise1=0.3)