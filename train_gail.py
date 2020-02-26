import gym
import numpy as np
import torch
import argparse
import os
import time
import multiprocessing as mp

import utils_local
from DDPG import DDPG
from GAIL import GAIL

from core.agent import Agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--num_imitators", default=5, type=int)     # Number of BC imitators in the ensemble
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
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
    file_name = "GAIL_%s_traj%s_seed%s_%s" % (args.env_name, args.num_trajs, str(args.seed), expert_type)
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

    imitator = GAIL(args, state_dim, action_dim)
    imitator.set_expert(flat_expert_trajs)

    # args.num_threads = mp.cpu_count() - 1
    # agent = Agent(env, imitator.policy.actor, args.device, custom_reward=imitator.expert_reward,
    #               render=args.render, num_threads=args.num_threads)

    # Initialize batch
    # replay_buffer = utils_local.ReplayBuffer()
    # replay_buffer.set_expert(flat_expert_trajs)

    total_timesteps = 0
    episode_reward = 0
    episode_num = 0
    done = True

    expert_rewards = []
    expert_timesteps = []
    batch = {'states':[],
             'actions':[],
             'rewards':[],
             'masks':[]}
    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))

                if len(batch['states']) >= 2048:
                    imitator.train(batch)
                    batch = {'states': [],
                             'actions': [],
                             'rewards': [],
                             'masks': []}

            # Save policy
            if total_timesteps % 1e4 == 0:
                np.save("./results/" + file_name + '_rewards', expert_rewards)
                np.save("./results/" + file_name + '_timesteps', expert_timesteps)
                imitator.save(file_name, directory="./imitator_models")

            expert_rewards.append(episode_reward)
            expert_timesteps.append(total_timesteps)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # batch = {'states':[],
            #          'actions':[],
            #          'rewards':[],
            #          'masks':[]}
        state_var = torch.FloatTensor(obs).unsqueeze(0)

        # Perform action
        with torch.no_grad():
            action = imitator.policy.actor.select_action(state_var)
            action = action[0].numpy()
            new_obs, reward, done, _ = env.step(action)

        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        reward = imitator.expert_reward(obs, action)

        # Store data in replay buffer
        batch['states'].append(obs)
        batch['actions'].append(action)
        batch['rewards'].append(reward)
        batch['masks'].append(1-done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1

    np.save("./results/" + file_name + '_rewards', expert_rewards)
    np.save("./results/" + file_name + '_timesteps', expert_timesteps)

    # Save final policy
    imitator.save("%s" % (file_name), directory="./imitator_models")