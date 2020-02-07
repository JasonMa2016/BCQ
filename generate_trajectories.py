import gym
import numpy as np
import torch
import argparse
import os

import utils
import DDPG

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_trajs", default=25, type=int)  # Number of trajectories to be collected
    parser.add_argument("--noise1", default=0.3, type=float)  # Probability of selecting random action
    parser.add_argument("--noise2", default=0.3, type=float)  # Std of Gaussian exploration noise
    args = parser.parse_args()

    file_name = "DDPG_%s_%s" % (args.env_name, str(args.seed))
    buffer_name = "Robust_traj%s_%s_%s" % (args.num_trajs, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action)
    policy.load(file_name, "./pytorch_models")
    # print("DDPG expert performance: " + str(utils.evaluate_policy(env, policy)))

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True
    expert_trajs = []
    expert_rewards = []

    while episode_num < args.num_trajs:

        expert_traj = []
        obs = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        for t in range(10000):
            action = policy.select_action(np.array(obs))
            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward

            expert_traj.append((obs, new_obs, action, reward, done_bool))

            if done:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                break
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1


        # Store data in replay buffer
        expert_rewards.append(episode_reward)
        replay_buffer.add(expert_traj)
    print(expert_rewards)
    replay_buffer.storage = [x for _,x in sorted(zip(expert_rewards,replay_buffer.storage), reverse=True)]

    # Save final buffer
    replay_buffer.save(buffer_name)
    replay_buffer.save_rewards(buffer_name, sorted(expert_rewards, reverse=True))