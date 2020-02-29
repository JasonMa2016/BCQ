import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
import utils_local

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save expert trajectory')
    parser.add_argument('--env-name', default="Walker2d-v2", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--model', default="ppo", metavar='G',
                        help='name of the expert model')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    # parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
    #                     help='maximal number of main iterations (default: 50000)')
    parser.add_argument('--num-trajs', type=int, default=100, metavar='N',
                        help='number of expert trajectories')
    parser.add_argument("--noise1", default=0.3, type=float)  # Probability of selecting random action
    parser.add_argument("--noise2", default=0.3, type=float)  # Std of Gaussian exploration noise

    args = parser.parse_args()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    state_dim = env.observation_space.shape[0]

    buffer_name = "PPO_traj%s_%s_%s" % (args.num_trajs, args.env_name, str(args.seed))

    args.model_path = "expert_models/{}_{}_{}.p".format(args.env_name, args.model, args.seed)

    policy, _, running_state, expert_args = pickle.load(open(args.model_path, "rb"))
    running_state.fix = True
    print(running_state.fix)
    expert_trajs = []
    expert_rewards = []
    print("=======================================")
    print("Settings: " + args.model_path)
    print("---------------------------------------")
    print("Generating Expert Trajectories")

    # Initialize buffer
    replay_buffer = utils_local.ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True
    expert_trajs = []
    expert_rewards = []
    noise = False

    while episode_num < args.num_trajs:
        if episode_num >= args.num_trajs / 2:
            noise = True
        expert_traj = []
        obs = env.reset()
        obs = running_state(obs)

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        for t in range(10000):
            state_var = tensor(obs).unsqueeze(0).to(dtype)
            # choose mean action, this works for continuous environment
            # discrete environment not implemented here
            if noise:
                if np.random.uniform(0,1) < args.noise1:
                    action = env.action_space.sample()
                else:
                    action = policy(state_var)[0][0].detach().numpy()
                    if args.noise2 != 0:
                        action = (action + np.random.normal(0, args.noise2, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)
            else:
                action = policy(state_var)[0][0].detach().numpy()

            action = action.astype(np.float64)
            new_obs, reward, done, _ = env.step(action)
            new_obs = running_state(new_obs)

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
    # print(expert_rewards)
    replay_buffer.storage = [x for _, x in sorted(zip(expert_rewards, replay_buffer.storage), reverse=True)]

    # Save final buffer
    replay_buffer.save(buffer_name)
    replay_buffer.save_rewards(buffer_name, sorted(expert_rewards, reverse=True))