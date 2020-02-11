import gym
import numpy as np
import torch
import argparse
import os
import time

import utils
import DDPG
import BCQ
from BC import Policy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--eval_freq", default=2e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--num_imitators", default=5, type=int)
    parser.add_argument("--max_timesteps", default=2e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--good", action='store_true', default=False)

    args = parser.parse_args()
    expert_type = 'good' if args.good else 'mixed'
    file_name = "BCQ_traj%s_%s_%s_%s" % (args.num_trajs, args.env_name, str(args.seed), expert_type)
    buffer_name = "%s_traj25_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)

    # create a flat list
    flat_expert_trajs = []
    if args.good:
        expert_trajs = expert_trajs[:args.num_trajs]
        expert_rewards = expert_rewards[:args.num_trajs]
    else:
        expert_trajs = np.concatenate((expert_trajs[:args.num_trajs],expert_trajs[-3:]), axis=0)
        expert_rewards = np.concatenate((expert_rewards[:args.num_trajs], expert_rewards[-3:]), axis=0)

    print("Expert rewards: {}".format(expert_rewards))
    print("avg: {} std: {}".format(np.mean(expert_rewards), np.std(expert_rewards)))

    for expert_traj in expert_trajs:
        for state_action in expert_traj:
            flat_expert_trajs.append(state_action)

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
    policy = BCQ.BCQ(state_dim, action_dim, max_action)

    # Initialize batch

    replay_buffer = utils.ReplayBuffer()
    replay_buffer.set_expert(flat_expert_trajs)

    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < args.max_timesteps:
        t0 = time.time()
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))
        t1 = time.time()
        rewards = utils.evaluate_policy(env, policy)
        evaluations.append(rewards)
        np.save("./results/" + file_name, evaluations)

        training_iters += args.eval_freq
        print("Training iterations: {}\tTraining time: {:.2f}\tReward average: {:.2f}\tReward std: {:.2f}".format(str(training_iters),
                                                                                          t1-t0,rewards.mean(),rewards.std()))