import numpy as np
import argparse
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import torch
import gym

import utils_local
from BC import BC, Policy


def plot_model_reward_over_timesteps(env_name='Hopper-v2', model_name='BC', num_trajs=5, seed=0, type='mixed'):
    file_name = 'results/{}_PPO_{}_traj{}_seed{}_{}'.format(model_name, env_name, num_trajs, seed, type)
    reward = np.load(file_name + '_rewards.npy')
    timestep = np.load(file_name + '_timesteps.npy')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestep, reward, label='new')

    # file_name = 'results/SQIL_ORIGINAL_DDPG_{}_traj{}_seed{}_{}'.format(env_name, num_trajs, seed, type)
    # reward = np.load(file_name + '_rewards.npy')
    # timestep = np.load(file_name + '_timesteps.npy')
    #
    # ax.plot(timestep, reward, label='original')

    ax.set_title(file_name)
    ax.legend(loc='best')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rewards')
    plt.savefig('plots/{}_{}_traj{}_seed{}_{}.png'.format(model_name, env_name, num_trajs, seed, type))
    plt.close()
    return

def plot_model_reward_over_timesteps_average(env_name='Hopper-v2', model_name='BC',
                                             num_trajs=5, seeds = [0,1,2,3,4], type='good'):

    # buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)
    #
    # expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)
    # buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)
    #
    # expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)
    #
    performance = []
    min_length = 10000
    min_timestep = None
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed in seeds:
        file_name = 'results/{}_PPO_{}_traj{}_seed{}_{}'.format(model_name, env_name, num_trajs, seed, type)
        reward = np.load(file_name + '_rewards.npy')
        timestep = np.load(file_name + '_timesteps.npy')
        if min_length > len(timestep):
            min_length = len(timestep)
            min_timestep = timestep
        performance.append(reward)
    for i in range(len(performance)):
        performance[i] = performance[i][:min_length]
    performance = np.array(performance)
    perf_mu = np.mean(performance, axis=0)
    perf_std = np.std(performance, axis=0)
    ax.plot(min_timestep, np.mean(performance, axis=0), label=type)
    ax.fill_between(min_timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('{}_{}_traj{}_{}_average'.format(model_name, env_name, num_trajs, seed))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/{}_{}_traj{}_{}_average.png'.format(model_name, env_name, num_trajs, type))
    plt.close()


def plot_model_reward_over_timesteps_compare_average(env_name='Hopper-v2', model_name='BC',
                                             num_trajs=5, seeds=[0, 1, 2, 3, 4]):
    # buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)
    #
    # expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)
    # buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)
    #
    # expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)
    #

    fig, ax = plt.subplots(figsize=(8, 4))
    for type in ['good', 'mixed']:
        performance = []
        min_length = 10000
        min_timestep = None

        for seed in seeds:
            file_name = 'results/{}_PPO_{}_traj{}_seed{}_{}'.format(model_name, env_name, num_trajs, seed, type)
            reward = np.load(file_name + '_rewards.npy')
            timestep = np.load(file_name + '_timesteps.npy')
            if min_length > len(timestep):
                min_length = len(timestep)
                min_timestep = timestep
            performance.append(reward)
        for i in range(len(performance)):
            performance[i] = performance[i][:min_length]
        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=0)
        perf_std = np.std(performance, axis=0)
        ax.plot(min_timestep, np.mean(performance, axis=0), label=type)
        ax.fill_between(min_timestep, perf_mu + perf_std, perf_mu - perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('{}_{}_traj{}_compare_average'.format(model_name, env_name, num_trajs))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/{}_{}_traj{}_compare_average.png'.format(model_name, env_name, num_trajs, type))
    plt.close()

def bc_evaluate_model_with_noise(env_name='Hopper-v2', model_name='BC', num_trajs=5, seed=0):
    expert_type = 'good'
    model_name = '{}_{}_traj{}_seed{}_{}'.format(model_name, env_name, num_trajs, seed, expert_type)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    imitator = Policy(state_dim, action_dim)

    imitator.load_state_dict(torch.load('./imitator_models_old/{}.p'.format(model_name)))
    rewards_noise = utils_local.evaluate_policy_with_noise(env, imitator)
    print(rewards_noise.mean(), rewards_noise.std())

    rewards= utils_local.evaluate_policy(env, imitator)
    print(rewards.mean(), rewards.std())

def bc_model_performance_plot(env_name='Hopper-v2', num_trajs=5, seed=0, samples=10):
    types = ['good', 'mixed']

    fig, ax = plt.subplots(figsize=(8, 4))
    for type in types:
        performance = []
        for sample in range(samples):
            file_name = "./results_old/" + "BC_{}_traj{}_seed{}_sample{}_{}.npy".format(env_name, num_trajs,
                                                                                   seed, sample, type)
            model_performance = np.load(file_name)
            performance.append(np.mean(model_performance, axis=1))

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=1)
        perf_std = np.std(performance, axis=1)
        ax.plot([(i+1)*1000 for i in range(10)], np.mean(performance, axis=1), label=type)
        ax.fill_between([(i+1)*1000 for i in range(10)], perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('{} Behavior Cloning Good vs. Mixed'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/BC-{}.png'.format(env_name))


def model_performance_plot(env_name='Hopper-v2', num_trajs=5, seed=0, samples=10):
    '''
    Plot model performance over seeds.
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''


def drbcq_performance_random(model='DRBCQ', env_name='Hopper-v2', num_trajs=5, seed=0):
    '''
    Compare uncertainty cost vs. random reward
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''

    fig, ax = plt.subplots(figsize=(8, 4))
    types = ['uncertainty cost', 'random']
    for type in types:
        performance = []
        for seed in range(5):

            if type == 'random':
                file_name = "./results/" + "{}_{}_traj{}_seed{}_good_random.npy".format(model, env_name, num_trajs, seed,
                                                                           type)
            else:
                file_name = "./results/" + "{}_{}_traj{}_seed{}_good.npy".format(model, env_name, num_trajs, seed,
                                                                           type)
            model_performance = np.load(file_name)
            performance.append(np.mean(model_performance, axis=1))

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=1)
        perf_std = np.std(performance, axis=1)
        ax.plot([(i+1)*1000 for i in range(len(performance))], np.mean(performance, axis=1), label=type)
        ax.fill_between([(i+1)*1000 for i in range(len(performance))], perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    performance = []
    for seed in range(5):
        file_name = "./results/" + "BCQ_{}_traj{}_seed{}_good.npy".format(env_name, num_trajs, seed,
                                                                       type)
        model_performance = np.load(file_name)
        performance.append(np.mean(model_performance, axis=1))

    performance = np.array(performance)
    perf_mu = np.mean(performance, axis=1)
    perf_std = np.std(performance, axis=1)
    ax.plot([(i+1)*1000 for i in range(len(performance))], np.mean(performance, axis=1), label='BCQ')
    ax.fill_between([(i+1)*1000 for i in range(len(performance))], perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('DRBCQ {} Reward Ablation Plot'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')
    # plt.show()
    plt.savefig('plots/DRBCQ_{}_reward_ablation.png'.format(env_name))


def drbcq_performance_bad_traj(model='DRBCQ', env_name='Hopper-v2', num_trajs=5):
    '''
    Compare uncertainty cost vs. random reward
    :param env_name:
    :param num_trajs:
    :param seed:
    :param samples:
    :return:
    '''

    fig, ax = plt.subplots(figsize=(8, 4))

    for bad_traj in [5,10,15]:
        performance = []
        for seed in range(5):
            file_name = "./results/" + "{}_{}_traj{}_badtraj{}_imit5_seed{}_mixed.npy".format(model, env_name, num_trajs, bad_traj,
                                                                                              seed)

            model_performance = np.load(file_name)
            performance.append(np.mean(model_performance, axis=1))

        performance = np.array(performance)
        perf_mu = np.mean(performance, axis=1)
        perf_std = np.std(performance, axis=1)
        ax.plot([(i+1)*1000 for i in range(len(performance))], np.mean(performance, axis=1), label=bad_traj)
        ax.fill_between([(i+1)*1000 for i in range(len(performance))], perf_mu+perf_std, perf_mu-perf_std, alpha=0.4)

    ax.legend(loc='best')
    ax.set_title('DRBCQ {} Reward Ablation Plot'.format(env_name))
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Rewards')

    # plt.show()
    plt.savefig('plots/DRBCQ_{}_bad_traj.png'.format(env_name))


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    print(cumsum)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_trajs", default=5, type=int)
    parser.add_argument("--model", default="DRBCQ")
    args = parser.parse_args()

    models = ['BCQ', 'DRBCQ', 'BC']
    types = ['mixed', 'good']
    # for seed in range(5):
    #     plot_model_reward_over_timesteps(model_name='GAIL', seed=seed)
    # # for seed in [0,1,2,5]:
    # #     plot_model_reward_over_timesteps(model_name='SQIL_DDPG', seed=seed)
    # for traj in [3]:
    #     for type in ['good', 'mixed']:
    #         plot_model_reward_over_timesteps_average(model_name='GAIL', env_name='Walker2d-v2',
    #                                                  num_trajs=traj, type=type)
    # plot_model_reward_over_timesteps_average(model_name='GAIL', type='good', num_trajs=1)
    # plot_model_reward_over_timesteps_average(model_name='GAIL', type='good', num_trajs=10)
    for traj in [1,3,5]:
        plot_model_reward_over_timesteps_compare_average(model_name='GAIL', num_trajs=traj, env_name='Walker2d-v2')
    # plot_model_reward_over_timesteps_compare_average(model_name='GAIL', num_trajs=1)
    # plot_model_reward_over_timesteps_compare_average(model_name='GAIL', num_trajs=10)

    # plot_model_reward_over_timesteps(model_name='GAIL', env_name='Walker2d-v2', seed=0, num_trajs=1)
    # plot_model_reward_over_timesteps(model_name='GAIL', env_name='Walker2d-v2', seed=0, num_trajs=3)
    # plot_model_reward_over_timesteps(model_name='GAIL', env_name='Walker2d-v2', seed=0, num_trajs=5)
    # plot_model_reward_over_timesteps(model_name='SQIL_TD3', seed=1)
    # drbcq_performance_random()
    # drbcq_performance_bad_traj()

    # fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    # # for i, seed in enumerate([1,2]):
    # for i, type in enumerate(types):
    #     for model in models:
    #         if model == 'BC':
    #             file_name = "./results/" + "{}_{}_traj{}_seed0_sample0_{}.npy".format(model, args.env_name, args.num_trajs,
    #                                                                    type)
    #             model_performance = np.load(file_name)
    #             axs[i].plot([10*i for i in range(10)], np.mean(model_performance, axis=1), label=model)
    #             axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #             axs[i].legend(loc='upper right')
    #         else:
    #             file_name = "./results/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
    #                                                                    type)
    #             model_performance = np.load(file_name)
    #             axs[i].plot([i for i in range(100)],np.mean(model_performance, axis=1), label=model)
    #             axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #             axs[i].legend(loc='upper right')

    # fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    # for i, seed in enumerate([0,1,2]):
    #     for type in ['good']:
    #         for model in models:
    #             if model == 'BC':
    #                 file_name = "./results/" + "{}_{}_traj{}_seed0_sample0_{}.npy".format(model, args.env_name, args.num_trajs,
    #                                                                        type)
    #                 model_performance = np.load(file_name)
    #                 axs[i].plot([10*i for i in range(10)], np.mean(model_performance, axis=1), label=model)
    #                 axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #                 axs[i].legend(loc='upper right')
    #             else:
    #                 file_name = "./results/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, seed,
    #                                                                        type)
    #                 model_performance = np.load(file_name)
    #                 axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=model)
    #                 axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #                 axs[i].legend(loc='upper right')
    #
    # # compare good vs mixed
    # fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    # for i, seed in enumerate([0,1,2]):
    #     for type in ['good', 'mixed']:
    #             file_name = "./results/" + "DRBCQ_{}_traj{}_seed{}_{}.npy".format(args.env_name, args.num_trajs, seed,
    #                                                                    type)
    #             model_performance = np.load(file_name)
    #             axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=type)
    #             axs[i].set_title('DRBCQ Training Curve'.format(args.env_name))
    #             axs[i].legend(loc='upper right')

    # bc_model_performance_plot()
    # bc_evaluate_model_with_noise()
    # for i, model in enumerate(models):
    #     for type in types:
    #         if model == 'BC':
    #             file_name = "./results_old/" + "{}_{}_traj{}_seed{}_sample5_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
    #                                                                    type)
    #         else:
    #             file_name = "./results_old/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
    #                                                                            type)
    #         model_performance = np.load(file_name)
    #         axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=type)
    #         axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #         axs[i].legend(loc='upper right')
    # plt.show()

    # plt.savefig('plots/{}_compare.png'.format(args.env_name))
