import gym
import numpy as np
import torch
import argparse
import os
import time

import utils_local
import DDPG
import BCQ


def evaluate_policy1(env, policy, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")              # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")              # Prepends name to filename.
    parser.add_argument("--eval_freq", default=1e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    args = parser.parse_args()

    file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))
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

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action)

    # Load buffer
    replay_buffer = utils_local.ReplayBuffer()
    replay_buffer.load(buffer_name)
    
    evaluations = []

    episode_num = 0
    done = True 

    training_iters = 0
    while training_iters < args.max_timesteps:
        t0 = time.time()
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))
        t1 = time.time()
        rewards = evaluate_policy1(env, policy)
        evaluations.append(rewards)
        np.save("./results/" + file_name, evaluations)

        training_iters += args.eval_freq
        print("Training iterations: {}\tTraining time: {:.2f}\tReward average: {:.2f}\tReward std: {:.2f}".format(str(training_iters),
                                                                                          t1-t0,rewards.mean(),rewards.std()))