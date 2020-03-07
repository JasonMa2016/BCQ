import numpy as np

import toy.gridworld
import toy.homework0_solution
import toy.homework1_solution


def load_env(env_name):
    if env_name == 'cliffworld':
        env = toy.homework1_solution.cliffworld()
    if env_name == 'cliffworld2':
        env = toy.homework1_solution.cliffworld2()
    if env_name == 'cliffworld3':
        env = toy.homework1_solution.cliffworld3()
    return env


if __name__ == "__main__":
    envs = ['cliffworld', 'cliffworld2', 'cliffworld3']
    for env_name in envs:
        env = load_env(env_name)

        # train expert
        res_expert = toy.homework1_solution.run_simulation(env, 'Q-learning')
        toy.homework1_solution.plot_policy(env, res_expert['Q'], env_name, 'expert')
        # collect expert trajectories
        episode_rewards, trajectories = toy.homework1_solution.evaluate_static_policy(env, res_expert['Q'])
        toy.homework1_solution.plot_trajectory(env, res_expert['Q'], trajectories, env_name, 'expert')

        # good imitator
        res_imitation = toy.homework1_solution.run_simulation_imitation(env, 'Q-learning', trajectories, min_num_episodes=200, default_reward=-10)
        episode_rewards, trajectories = toy.homework1_solution.evaluate_static_policy(env, res_imitation['Q'])
        toy.homework1_solution.plot_policy(env, res_imitation['Q'], env_name, 'good_imitator')
        toy.homework1_solution.plot_trajectory(env, res_imitation['Q'], trajectories, env_name, 'good imitator')

        # good imitator
        res_imitation = toy.homework1_solution.run_simulation_imitation(env, 'Q-learning', trajectories, default_reward=0, min_num_episodes=200)
        episode_rewards, trajectories = toy.homework1_solution.evaluate_static_policy(env, res_imitation['Q'])
        toy.homework1_solution.plot_policy(env, res_imitation['Q'], env_name, 'bad_imitator')
        toy.homework1_solution.plot_trajectory(env, res_imitation['Q'], trajectories, env_name, 'bad imitator')
