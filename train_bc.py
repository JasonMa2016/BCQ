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


import utils_local
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
    parser.add_argument("--env_name", default="Walker2d-v2")              # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")              # Prepends name to filename.
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--eval_freq", default=1e3, type=float)         # How often (time steps) we evaluate
    # parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    parser.add_argument("--ensemble", action='store_true', default=False)
    parser.add_argument("--good", action='store_true', default=False)
    parser.add_argument("--max_iters", default=1e5, type=int)
    parser.add_argument("--batch_size", default=1e2, type=int)

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    expert_type = 'good' if args.good else 'mixed'
    file_name = "BC_%s_%s_%s" % (args.env_name, str(args.seed), expert_type)

    # buffer_name = "%s_traj100_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    # buffer_name = "%s_traj100_%s_0" % (args.buffer_type, args.env_name)
    buffer_name = "PPO_traj100_%s_0" % (args.env_name)

    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    expert_rewards = np.load("./buffers/"+buffer_name+ "_rewards" + ".npy", allow_pickle=True)

    args.model_path = "expert_models/{}_ppo_0.p".format(args.env_name)

    _, _, running_state, expert_args = pickle.load(open(args.model_path, "rb"))

    # print(expert_rewards)
    #
    # # create a flat list
    # flat_expert_trajs = []
    # if args.good:
    #     expert_trajs = expert_trajs[:args.num_trajs]
    # else:
    #     expert_trajs = np.concatenate((expert_trajs[:args.num_trajs],expert_trajs[-3:]), axis=0)
    # for expert_traj in expert_trajs:
    #     for state_action in expert_traj:
    #         flat_expert_trajs.append(state_action)
    # print(len(flat_expert_trajs))
    flat_expert_trajs = utils_local.collect_trajectories_rewards(expert_trajs, good=args.good)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if not args.ensemble:
        # single BC
        imitator = BC(args, state_dim, action_dim, max_action)
        imitator.set_expert(flat_expert_trajs)
        evaluations = []
        file_name = 'BC_{}_batch{}_traj{}_seed{}_{}'.format(args.env_name, int(args.batch_size), args.num_trajs, args.seed,
                                                             expert_type)
        print(file_name)
        expert_rewards = []
        expert_timesteps = []

        for i_iter in range(int(args.max_iters)):
            if i_iter % args.eval_freq == 0:
                imitator.actor.to('cpu')
                rewards = utils_local.evaluate_policy(env, imitator.actor, running_state)

                expert_rewards.append(rewards)
                expert_timesteps.append(i_iter)

                np.save("./results/" + file_name + '_rewards', expert_rewards)
                np.save("./results/" + file_name + '_timesteps', expert_timesteps)

                # evaluations.append(rewards)
                # np.save("./results/" + file_name, evaluations)
                if i_iter > 0:
                    print(
                        'Training iteration {}\tT_update:{:.4f}\t reward avg:{:.2f}\t reward std:{:.2f}\t training loss:{:.2f}'.format(
                            i_iter, t1 - t0, rewards.mean(), rewards.std(), loss))
                imitator.actor.to(args.device)
            t0 = time.time()
            loss = imitator.train(batch_size=int(args.batch_size))
            t1 = time.time()

        imitator.actor.to('cpu')
        torch.save(imitator.actor.state_dict(), 'imitator_models/{}.p'.format(file_name))
        print("=======================================")
        print("")
        np.save("./results/" + file_name + '_rewards', expert_rewards)
        np.save("./results/" + file_name + '_timesteps', expert_timesteps)

    else:
        # ensemble
        expert_traj = np.array(flat_expert_trajs)
        for sample in range(5):
            print("=======================================")
            print("BC Imitator {}".format(sample+1))
            indices = np.random.choice(len(expert_traj), len(expert_traj))
            # print(indices)
            # print(expert_traj)
            current_expert_traj = expert_traj[indices.astype(int)]
            imitator = BC(args, state_dim, action_dim, max_action)
            imitator.set_expert(current_expert_traj)
            evaluations = []
            file_name = 'BC_{}_traj{}_seed{}_sample{}_{}'.format(args.env_name, args.num_trajs, args.seed, sample, expert_type)
            expert_rewards = []
            expert_timesteps = []

            for i_iter in range(int(args.max_iters)):
                t0 = time.time()
                loss = imitator.train()
                t1 = time.time()
                if i_iter % args.eval_freq == 0:
                    imitator.actor.to('cpu')
                    rewards = utils_local.evaluate_policy(env, imitator.actor, running_state)

                    expert_rewards.append(rewards)
                    expert_timesteps.append(i_iter)

                    np.save("./results/" + file_name + '_rewards', expert_rewards)
                    np.save("./results/" + file_name + '_timesteps', expert_timesteps)

                    # evaluations.append(rewards)
                    # np.save("./results/" + file_name, evaluations)

                    print(
                        'Training iteration {}\tT_update:{:.4f}\t reward avg:{:.2f}\t reward std:{:.2f}\t training loss:{:.2f}'.format(
                            i_iter, t1 - t0, rewards.mean(), rewards.std(), loss))
                    imitator.actor.to(args.device)

            imitator.actor.to('cpu')
            torch.save(imitator.actor.state_dict(), 'imitator_models/{}.p'.format(file_name))
            print("=======================================")
            print("")
        np.save("./results/" + file_name + '_rewards', expert_rewards)
        np.save("./results/" + file_name + '_timesteps', expert_timesteps)
