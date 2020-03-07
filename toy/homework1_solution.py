import matplotlib.pyplot as plt
import numpy as np
from toy.gridworld import GridWorld

from toy.homework0_solution import policy_iteration, Q_function, MDP

MAX_STEPS_PER_EPISODE = 100


def cliffworld():
    """Construct the "cliffworld" environment."""
    return GridWorld(
        maze=[
            '#######',
            '#.....#',
            '#.##..#',
            '#o...*#',
            '#XXXXX#',
            '#######'
        ],
        rewards={
            '*': 50,         # gain 50 points for reaching goal
            'X': -50,        # lose 50 points for falling down
            'moved': -1,     # lose 1 point for walking
            'hit-wall': -1   # lose 1 point for stumbling into a wall
        }
    )


def cliffworld2():
    """Construct the "cliffworld" environment."""
    return GridWorld(
        maze=[
            '#######',
            '#....*#',
            '#.###.#',
            '#o....#',
            '#XXXXX#',
            '#######'
        ],
        rewards={
            '*': 50,         # gain 50 points for reaching goal
            'X': -50,        # lose 50 points for falling down
            'moved': -1,     # lose 1 point for walking
            'hit-wall': -1   # lose 1 point for stumbling into a wall
        }
    )


def cliffworld3():
    """Construct the "cliffworld" environment."""
    return GridWorld(
        maze=[
            '#######',
            '#....*#',
            '#.#X#.#',
            '#o....#',
            '#XXXXX#',
            '#######'
        ],
        rewards={
            '*': 50,         # gain 50 points for reaching goal
            'X': -50,        # lose 50 points for falling down
            'moved': -1,     # lose 1 point for walking
            'hit-wall': -1   # lose 1 point for stumbling into a wall
        }
    )

def true_Q_function(env, discount=0.95):
    """Return the true value of the Q function using the actual MDP / HW0 code.

    NOTE: Please only use this for testing/comparison, not policy learning!"""
    true_T, true_R = env.as_mdp()
    true_mdp = MDP(true_T, true_R, discount)
    true_pi_star = policy_iteration(true_mdp)[-1]
    return Q_function(true_mdp, true_pi_star), true_pi_star


def true_pi_star(env, discount=0.95):
    true_T, true_R = env.as_mdp()
    true_mdp = MDP(true_T, true_R, discount)
    true_pi_star = policy_iteration(true_mdp)[-1]
    return true_pi_star


def epsilon_greedy_action(state, Q, epsilon=0.1):
    """Select a random action with probability epsilon or the action suggested
    by Q with probability 1-epsilon."""
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[1])
    else:
        return np.argmax(Q[state])


def solve_MDP(T, R, discount, iters=100):
    """Return the optimal Q function of an MDP defined by T, R, and
    discount.

    NOTE: if implemented inefficiently, this function will make Thompson
    sampling very slow. For this assignment, we recommend avoiding for loops
    over states and actions and instead performing calculations for all states
    simultaneously using np.sum and np.dot.  Although this prevents information
    flow between different states in the same outer iteration, Q should still
    converge, and the computation will be an order of magnitude faster on most
    machines."""
    Ds, Da, _ = T.shape
    Q = np.zeros((Ds, Da))
    V = np.zeros(Ds)
    expected_reward = np.sum(T * R, axis=2)
    for i in range(iters):
        expected_value_of_next_state = np.dot(T, V)
        Q = expected_reward + discount * expected_value_of_next_state
        V = np.max(Q, axis=1)
    return Q


def run_simulation(
        # Common parameters
        env,
        method,
        min_num_episodes=100,
        min_num_iters=5000,
        epsilon=0.1,
        discount=0.8,
        # SARSA/Q-learning parameters
        step_size=0.5,
        Q_initial=0.0,
        # Thompson-sampling parameters
        thompson_update_freq=50,
        thompson_dirichlet_prior=1,
        thompson_default_reward=50
    ):
    # Ensure valid parameters
    if method not in ('Thompson', 'SARSA', 'Q-learning'):
        raise ValueError("method not in {Thompson, SARSA, Q-learning}")

    # Initialize arrays for our estimate of Q and observations about T and R,
    # and our list of rewards by episode
    num_states, num_actions = env.num_states, env.num_actions
    Q = np.zeros((num_states, num_actions)) + Q_initial
    observed_T_counts = np.zeros((num_states, num_actions, num_states))
    observed_R_values = np.zeros((num_states, num_actions, num_states))
    episode_rewards = []
    num_cliff_falls = 0
    global_iter = 0

    # Loop through episodes
    while len(episode_rewards) < min_num_episodes or global_iter < min_num_iters:
        # Reset environment and episode-specific counters
        env.reset()
        episode_step = 0
        episode_reward = 0

        # Get our starting state
        s1 = env.observe()

        # Loop until the episode completes
        while not env.is_terminal(s1) and episode_step < MAX_STEPS_PER_EPISODE:
            # Take eps-best action & receive reward
            a = epsilon_greedy_action(s1, Q, epsilon)
            s2, r = env.perform_action(a)

            # Update counters
            episode_step += 1
            episode_reward += r
            observed_T_counts[s1][a][s2] += 1
            observed_R_values[s1][a][s2] = r
            num_cliff_falls += env.is_cliff(s2)

            # Use one of the RL methods to update Q
            if method == 'Q-learning':
                """Implement Q-learning update step from Section 6.4, Sutton
                and Barto (p131)"""

                # Treat the next state value as the best possible
                next_state_val = Q[s2].max()

                # Update Q
                Q[s1,a] += step_size * (r + discount * next_state_val - Q[s1,a])

            s1 = s2
            global_iter += 1

        episode_rewards.append(episode_reward)

    return { 'Q': Q,
            'num_cliff_falls': num_cliff_falls,
            'episode_rewards': np.array(episode_rewards) }


def run_simulation_imitation(
        # Common parameters
        env,
        method,
        expert,
        default_reward = -1,
        min_num_episodes=100,
        min_num_iters=5000,
        epsilon=0.1,
        discount=0.95,
        # SARSA/Q-learning parameters
        step_size=0.5,
        Q_initial=0.0,

    ):
    # Ensure valid parameters
    if method not in ('Thompson', 'SARSA', 'Q-learning'):
        raise ValueError("method not in {Thompson, SARSA, Q-learning}")

    # Initialize arrays for our estimate of Q and observations about T and R,
    # and our list of rewards by episode
    num_states, num_actions = env.num_states, env.num_actions
    Q = np.zeros((num_states, num_actions)) + Q_initial
    observed_T_counts = np.zeros((num_states, num_actions, num_states))
    observed_R_values = np.zeros((num_states, num_actions, num_states))
    episode_rewards = []
    num_cliff_falls = 0
    global_iter = 0

    # Loop through episodes
    while len(episode_rewards) < min_num_episodes or global_iter < min_num_iters:
        # Reset environment and episode-specific counters
        env.reset()
        episode_step = 0
        episode_reward = 0

        # Get our starting state
        s1 = env.observe()

        # Loop until the episode completes
        while not env.is_terminal(s1) and episode_step < MAX_STEPS_PER_EPISODE:
            # Take eps-best action & receive reward
            a = epsilon_greedy_action(s1, Q, epsilon)
            s2, r = env.perform_action(a)


            # Update counters
            episode_step += 1
            episode_reward += r
            observed_T_counts[s1][a][s2] += 1
            observed_R_values[s1][a][s2] = r
            num_cliff_falls += env.is_cliff(s2)

            r = default_reward
            if (s1, a) in expert:
                r = 1
            # Use one of the RL methods to update Q
            if method == 'Q-learning':
                """Implement Q-learning update step from Section 6.4, Sutton
                and Barto (p131)"""

                # Treat the next state value as the best possible
                next_state_val = Q[s2].max()

                # Update Q
                Q[s1,a] += step_size * (r + discount * next_state_val - Q[s1,a])

            s1 = s2
            global_iter += 1

        episode_rewards.append(episode_reward)

    return { 'Q': Q,
            'num_cliff_falls': num_cliff_falls,
            'episode_rewards': np.array(episode_rewards) }


def evaluate_static_policy(env, Q, num_episodes=100, epsilon=0):
    episode_rewards = []
    trajectories = set()
    while len(episode_rewards) < num_episodes:
        episode_reward = 0
        episode_iter = 0
        env.reset()
        trajectory = []
        s1 = env.observe()
        while not env.is_terminal(s1) and episode_iter < MAX_STEPS_PER_EPISODE:
            a = epsilon_greedy_action(s1, Q, epsilon)
            trajectories.add((s1, a))
            s2, r = env.perform_action(a)
            episode_reward += r
            episode_iter += 1
            s1 = s2
        trajectories.add((s1,2))
        # trajectories.append(trajectory)
        episode_rewards.append(episode_reward)
    return episode_rewards, trajectories


def plot_policy(env, Q, env_name, model_name):
    row_count, col_count = env.maze_dimensions
    maze_dims = (row_count, col_count)
    value_function = np.reshape(np.max(Q, 1), maze_dims)
    policy_function = np.reshape(np.argmax(Q, 1), maze_dims)
    wall_info = .5 + np.zeros(maze_dims)
    wall_mask = np.zeros(maze_dims)
    for row in range(row_count):
        for col in range(col_count):
            if env.maze.topology[row][col] == '#':
                wall_mask[row,col] = 1
    wall_info = np.ma.masked_where(wall_mask==0, wall_info)
    value_function *= (1-wall_mask)**2
    plt.imshow(value_function, interpolation='none', cmap='jet')
    plt.colorbar(label='Value Function')
    plt.imshow(wall_info, interpolation='none' , cmap='gray')
    y,x = env.maze.start_coords
    plt.text(x,y,'start', color='gray', fontsize=14, va='center', ha='center', fontweight='bold')
    y,x = env.maze.goal_coords
    plt.text(x,y,'goal', color='yellow', fontsize=14, va='center', ha='center', fontweight='bold')
    for row in range( row_count ):
        for col in range( col_count ):
            if wall_mask[row][col] == 1:
                continue
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow(col, row, dx, dy,
                shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2)
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.title('{} {} policy visualization'.format(env_name, model_name))
    plt.savefig('{}_{}.png'.format(env_name, model_name))
    plt.close()


def plot_trajectory(env, Q, trajectory, env_name, model_name):
    row_count, col_count = env.maze_dimensions
    maze_dims = (row_count, col_count)
    value_function = np.reshape(np.max(Q, 1), maze_dims)
    policy_function = np.reshape(np.argmax(Q, 1), maze_dims)
    wall_info = .5 + np.zeros(maze_dims)
    wall_mask = np.zeros(maze_dims)
    for row in range(row_count):
        for col in range(col_count):
            if env.maze.topology[row][col] == '#':
                wall_mask[row, col] = 1
    wall_info = np.ma.masked_where(wall_mask == 0, wall_info)
    value_function *= (1 - wall_mask) ** 2
    plt.imshow(value_function, interpolation='none', cmap='jet')
    plt.colorbar(label='Value Function')
    plt.imshow(wall_info, interpolation='none', cmap='gray')
    y, x = env.maze.start_coords
    plt.text(x, y, 'start', color='gray', fontsize=14, va='center', ha='center', fontweight='bold')
    y, x = env.maze.goal_coords
    plt.text(x, y, 'goal', color='yellow', fontsize=14, va='center', ha='center', fontweight='bold')
    for state,action in trajectory:
        row = int(state / col_count)
        col = state % col_count
        # print(state, action)
        # print(row,col)
        if wall_mask[row][col] == 1:
            continue
        if action == 0:
            dx = 0;
            dy = -.5
        if action == 1:
            dx = 0;
            dy = .5
        if action == 2:
            dx = .5;
            dy = 0
        if action == 3:
            dx = -.5;
            dy = 0
        plt.arrow(col, row, dx, dy,
                  shape='full', fc='w', ec='w', lw=3, length_includes_head=True, head_width=.2)
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.title('{} {} expert policy visualization'.format(env_name, model_name))
    plt.savefig('{}_{}_trajectory.png'.format(env_name, model_name))
    plt.close()
    return