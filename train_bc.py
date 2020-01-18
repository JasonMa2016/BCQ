import gym
import numpy as np
import torch
import time
import argparse
import os
from os import path
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import utils
import DDPG
import BCQ
from BC import BC


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
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

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), './imitator_models'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")              # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")              # Prepends name to filename.
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))
    buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    expert_traj = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    for sample in range(5):
        print("=======================================")
        print("BC Imitator {}".format(sample+1))
        indices = np.random.choice(len(expert_traj), len(expert_traj))
        current_expert_traj = expert_traj[indices]
        imitator = BC(args, state_dim, action_dim, max_action)
        imitator.set_expert(expert_traj)

        for i_iter in range(500):
            t0 = time.time()
            loss = imitator.train()
            t1 = time.time()
            if i_iter % 10 == 0:
                print('{}\tT_update:{:.4f}\t training loss:{:.2f}'.format(
                i_iter, t1-t0, loss))

        imitator.actor.to('cpu')
        torch.save(imitator.actor.state_dict(), 'imitator_models/{}_sample{}_seed{}.p'.format(args.env_name,
                                                                 sample,
                                                                 args.seed))
        print("=======================================")
        print("")