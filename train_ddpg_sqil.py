import gym
import numpy as np
import torch
import argparse
import os
import time
import multiprocessing as mp

import utils_local
import SQIL


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--num_imitators", default=5, type=int)     # Number of BC imitators in the ensemble
    parser.add_argument("--max_timesteps", default=1e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--good", action='store_true', default=False) # Good or mixed expert trajectories
    parser.add_argument("--start_timesteps", default=1e3, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--new", action='store_true', default=False)


    args = parser.parse_args()

    expert_type = 'good' if args.good else 'mixed'
    model_type = 'NEW' if args.new else 'ORIGINAL'
    file_name = "SQIL_DDPG_%s_%s_traj%s_seed%s_%s" % (model_type, args.env_name, args.num_trajs, str(args.seed), expert_type)
    # buffer_name = "%s_traj100_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)

    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)
    flat_expert_trajs = utils_local.collect_trajectories_rewards(expert_trajs, num_good_traj=args.num_trajs,
                                                                 num_bad_traj=args.num_trajs, good=args.good)

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
    if args.new:
        policy = SQIL.DDPG_SQIL(state_dim, action_dim, max_action)
    else:
        policy = SQIL.DDPG_SQIL_ORIGINAL(state_dim, action_dim, ax_action)

    # Initialize buffers
    expert_buffer = utils_local.ReplayBuffer()
    expert_buffer.set_expert(flat_expert_trajs)

    replay_buffer = utils_local.ReplayBuffer()

    total_timesteps = 0
    episode_reward = 0
    episode_num = 0
    done = True

    expert_rewards = []
    expert_timesteps = []
    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(expert_buffer, expert=True)
                policy.train(replay_buffer)

            # Save policy
            if total_timesteps % 1e5 == 0:
                np.save("./results/" + file_name + '_rewards', expert_rewards)
                np.save("./results/" + file_name + '_timesteps', expert_timesteps)
                policy.save(file_name, directory="./imitator_models")

            expert_rewards.append(episode_reward)
            expert_timesteps.append(total_timesteps)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1

    np.save("./results/" + file_name + '_rewards', expert_rewards)
    np.save("./results/" + file_name + '_timesteps', expert_timesteps)

    # Save final policy
    policy.save("%s" % (file_name), directory="./imitator_models")