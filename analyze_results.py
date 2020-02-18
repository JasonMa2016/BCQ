import numpy as np
import argparse
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import torch
import gym

import utils_local
from BC import BC, Policy


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_trajs", default=5, type=int)
    parser.add_argument("--model", default="DRBCQ")
    args = parser.parse_args()

    models = ['BCQ', 'DRBCQ', 'BC']
    types = ['mixed', 'good']

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

    fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    for i, seed in enumerate([0,1,2]):
        for type in ['good']:
            for model in models:
                if model == 'BC':
                    file_name = "./results/" + "{}_{}_traj{}_seed0_sample0_{}.npy".format(model, args.env_name, args.num_trajs,
                                                                           type)
                    model_performance = np.load(file_name)
                    axs[i].plot([10*i for i in range(10)], np.mean(model_performance, axis=1), label=model)
                    axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
                    axs[i].legend(loc='upper right')
                else:
                    file_name = "./results/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, seed,
                                                                           type)
                    model_performance = np.load(file_name)
                    axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=model)
                    axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
                    axs[i].legend(loc='upper right')

    # compare good vs mixed
    fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    for i, seed in enumerate([0,1,2]):
        for type in ['good', 'mixed']:
                file_name = "./results/" + "DRBCQ_{}_traj{}_seed{}_{}.npy".format(args.env_name, args.num_trajs, seed,
                                                                       type)
                model_performance = np.load(file_name)
                axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=type)
                axs[i].set_title('DRBCQ Training Curve'.format(args.env_name))
                axs[i].legend(loc='upper right')

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

    plt.savefig('plots/{}_compare.png'.format(args.env_name))

    # print(np.std(results,axis=1))