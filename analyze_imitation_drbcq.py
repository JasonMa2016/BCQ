import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym

# from models.mlp_policy import Policy
from BC import Policy
from utils_local import *


def likelihood(env_name='Walker2d-v2', model_name='DRBCQ', traj=5, seed=0, in_sample=True):
    """
    Compute the likelihood of the entire trajectory
    :param env_name:
    :param model_name:
    :param traj:
    :param seed:
    :param in_sample:
    :return:
    """
    expert_path = "expert_models/{}_PPO_0.p".format(env_name)

    policy, _, running_state, expert_args = pickle.load(open(expert_path, "rb"))

    buffer_name = "PPO_traj100_%s_0" % (env_name)
    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for type in ['good', 'mixed']:
        imitator_name = '{}_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
        imitator = Policy(17,6)
        imitator.load_state_dict(torch.load('imitator_models/%s.p' % (imitator_name)))
        probs = []
        min_timestep = 10000
        for i in range(traj):
            prob = []
            if len(expert_trajs[i]) < min_timestep:
                min_timestep = len(expert_trajs[i])
            for sample in expert_trajs[i]:
                states = torch.FloatTensor(sample[0])
                actions = torch.FloatTensor(sample[2])
                # states = torch.FloatTensor(np.random.random(states.size()))
                log_prob = imitator.get_log_prob(states.unsqueeze(0),
                                                 actions.unsqueeze(0))[0][0].detach().numpy()
                prob.append(log_prob)
            probs.append(prob)

        for i in range(len(probs)):
            probs[i] = probs[i][:min_timestep]
        probs = np.array(probs)
        probs_mu = np.mean(probs, axis=0)
        probs_std = np.std(probs, axis=0)
        ax.plot([i for i in range(min_timestep)], probs_mu, label=type)
        ax.fill_between(min_timestep, probs_mu + probs_std, probs_mu - probs_std, alpha=0.4)

    ax.set_title('{} {} Imitator Log-Likelihood over Expert Trajectories'.format(env_name, model_name))
    ax.legend(loc='best')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Log Likelihood')
    plt.savefig('plots/{}_{}_likelihood.png'.format(model_name, env_name))
    plt.close()
    return


def plot_distribution(env_name='Walker2d-v2', model_name='GAIL', sample_traj=50, traj=5, seed=0, type='good'):
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
                                                running_state, num_trajs=sample_traj, verbose=False)

    imitator_name = '{}_PPO_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
    imitator = Policy(17, 6)
    imitator.load_state_dict(torch.load('imitator_models/%s_actor.pth' % (imitator_name)))

    # imitator_model_path = "imitator_models/{}_{}_traj{}_seed{}.p".format(model_name, model_name, traj, seed)
    # imitator = pickle.load(open(imitator_model_path, "rb"))[0]
    imitator_results= evaluate_model(gym_env, imitator,
                                                running_state, num_trajs=sample_traj, verbose=False, floattensor=True)
    metrics = ['rewards', 'timesteps']
    for i, metric in enumerate(metrics):
        axs[i].hist(expert_results[metric], density=True, alpha=0.7, label='expert', bins=25)
        axs[i].hist(imitator_results[metric], density=True, alpha=0.5, label=model_name, bins=25)
        axs[i].set_title('{} {} Density Curve'.format(env_name, metric))
        axs[i].legend(loc='upper right')
        axs[i].set(xlabel='{}'.format(metric), ylabel='density')
    # for ax in axs.flat:
    #     ax.set(xlabel='{}'.format(metric), ylabel='density')

    fig.savefig('plots/{}_{}_density_comparison_plot.png'.format(model_name, env_name))
    plt.close(fig)


def analyze_model_with_noise(env_name='Walker2d-v2', model_name='GAIL', num_trajs=[1,3,5],
                                 seeds=[0,1,2,3,4], type='good', noise1=0.3, noise2=0.3):
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
            imitator = Policy(17, 6)
            imitator.load_state_dict(torch.load('imitator_models/%s.p' % (imitator_name)))

            rewards_i = evaluate_policy(env, imitator, running_state).mean()
            rewards_noise_i = evaluate_policy_with_noise(env, imitator, running_state, noise1=noise1,
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
    ax.set_title('{} {} Noise Evaluation, epsilon={}, '.format(env_name, model_name))
    ax.set_xlabel('Number of Expert Trajectories')
    ax.set_ylabel('Cumulative Rewards')
    plt.savefig('plots/{}_{}_noise1{}_noise2{}.png'.format(model_name, env_name, noise1, noise2))
    plt.close()


if __name__ == "__main__":
    # likelihood('Walker2d-v2', model_name='BC')
    # # plot_distribution()
    analyze_model_with_noise(model_name='BC')