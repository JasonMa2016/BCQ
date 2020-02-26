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
    env_name = 'cliffworld'
    env = load_env(env_name)

    res_expert = toy.homework1_solution.run_simulation(env, 'Q-learning')
    toy.homework1_solution.plot_policy(env, res_expert['Q'], env_name, 'expert')
    # collect expert trajectories
    episode_rewards, trajectories = toy.homework1_solution.evaluate_static_policy(env, res_expert['Q'])
    toy.homework1_solution.plot_trajectory(env, res_expert['Q'], trajectories, env_name, 'expert')

    true_Q, true_pi = toy.homework1_solution.true_Q_function(env)
    toy.homework1_solution.plot_policy(env, true_Q, env_name=env_name, model_name='True Pi Star')