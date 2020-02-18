import gym
import numpy as np
import torch
import argparse
import os
import time
import multiprocessing as mp

import utils_local
import DDPG
import DRIL
from BC import Policy
from core.agent import Agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--num_imitators", default=5, type=int)     # Number of BC imitators in the ensemble
    parser.add_argument("--max_timesteps", default=1e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--good", action='store_true', default=False) # Good or mixed expert trajectories

    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                        help='gae (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')

    args = parser.parse_args()

    expert_type = 'good' if args.good else 'mixed'
    file_name = "DRIL_%s_traj%s_seed%s_%s" % (args.env_name, args.num_trajs, str(args.seed), expert_type)
    # buffer_name = "%s_traj100_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)

    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)

    flat_expert_trajs = utils_local.collect_trajectories_rewards(expert_trajs, good=args.good)

    print("---------------------------------------")
    print("Settings: " + file_name)
    print("")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy and imitator ensemble
    imitator = DRIL.DRIL(args, state_dim, action_dim)
    model_paths = []
    for sample in range(args.num_imitators):
        # model_path = 'imitator_models/BC_{}_traj{}_seed{}_sample{}_{}.p'.format(args.env_name, args.num_trajs,args.seed, sample, expert_type)
        model_path = 'imitator_models/BC_{}_traj{}_seed0_sample{}_{}.p'.format(args.env_name, args.num_trajs, sample, expert_type)

        model_paths.append(model_path)
    imitator.set_ensemble(model_paths)
    imitator.set_expert(flat_expert_trajs)

    args.num_threads = mp.cpu_count() - 1
    agent = Agent(env, imitator.policy.actor, args.device, num_threads=args.num_threads)

    # Initialize batch
    replay_buffer = utils_local.ReplayBuffer()
    replay_buffer.set_expert(flat_expert_trajs)

    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < args.max_timesteps:
        t0 = time.time()
        pol_vals = imitator.train(replay_buffer, agent)
        t1 = time.time()
        rewards = utils_local.evaluate_policy(env, imitator)
        evaluations.append(rewards)
        np.save("./results/" + file_name, evaluations)

        training_iters += 1
        print("Training iterations: {}\tTraining time: {:.2f}\tReward average: {:.2f}\tReward std: {:.2f}".format(str(training_iters),
                                                                                          t1-t0,rewards.mean(),rewards.std()))
    # save the imitator
    imitator.policy.actor.to('cpu')
    torch.save(imitator.policy.actor.state_dict(), 'imitator_models/{}.p'.format(file_name))