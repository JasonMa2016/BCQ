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
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    args = parser.parse_args()

    file_name = "DRBCQ_%s_%s" % (args.env_name, str(args.seed))
    buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
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
    policy = BCQ.DRBCQ(state_dim, action_dim, max_action)
    model_paths = []
    for sample in range(5):
        model_path = 'imitator_models/{}_sample{}_seed{}.p'.format(args.env_name, sample, args.seed)
        model_paths.append(model_path)
    policy.set_ensemble(model_paths)

    # Initialize batch
    replay_buffer = utils.ReplayBuffer()
    replay_buffer.load(buffer_name)

    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < args.max_timesteps:
        t0 = time.time()
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))
        t1 = time.time()
        avg_reward = utils.evaluate_policy(policy)
        evaluations.append(avg_reward)
        np.save("./results/" + file_name, evaluations)

        training_iters += args.eval_freq
        print("Training iterations: {}\tTraining time: {:.2f}\tAverage reward: {:.2f}".format(str(training_iters),
                                                                                              t1 - t0, avg_reward))