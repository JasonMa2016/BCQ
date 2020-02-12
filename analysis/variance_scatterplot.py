import argparse
import torch
import numpy as np
import pandas as pd
import gym
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from BC import Policy


def sample_all(traj):
    """
    Helper function to put the trajectory into the desired format)
    :param traj:
    :return:
    """
    state, next_state, action, reward, done = [], [], [], [], []

    for i in range(len(traj)):
        s, s2, a, r, d = traj[i]
        state.append(np.array(s, copy=False))
        next_state.append(np.array(s2, copy=False))
        action.append(np.array(a, copy=False))
        reward.append(np.array(r, copy=False))
        done.append(np.array(d, copy=False))

    return (np.array(state),
            np.array(next_state),
            np.array(action),
            np.array(reward).reshape(-1, 1),
            np.array(done).reshape(-1, 1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")              # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")              # Prepends name to filename.
    parser.add_argument("--num_trajs", default=5, type=int)            # Number of expert trajectories to use
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    parser.add_argument("--ensemble", action='store_true', default=False)
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))
    buffer_name = "%s_traj25_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    expert_trajs = np.load("../buffers/"+buffer_name+".npy", allow_pickle=True)
    expert_rewards = np.load("../buffers/"+buffer_name+ "_rewards" + ".npy", allow_pickle=True)

    good_traj = expert_trajs[15]
    bad_traj = expert_trajs[-1]

    trajs = {'good': good_traj,
             'bad': bad_traj}

    trajs = {}
    for i in range(len(expert_trajs)):
        # if i < 5 or (i > 9 and i < 20):
        #     continue
        trajs[i] = expert_trajs[i]

    tips = sns.load_dataset('tips')
    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    imitator_ensemble = []
    model_paths = []

    for sample in range(10):
        model_path = '../imitator_models/BC_{}_traj{}_seed{}_sample{}_mixed.p'.format(args.env_name, args.num_trajs,
                                                                                     args.seed, sample)
        model_paths.append(model_path)

    # create BC imitator ensemble
    for model_path in model_paths:
        imitator = Policy(state_dim, action_dim)
        imitator.load_state_dict(
            torch.load(model_path))
        imitator_ensemble.append(imitator)

    # scatter plot for the good trajectory
    data = []
    for traj in trajs:
        new_reward = []
        state_np, next_state_np, action, reward, done = sample_all(trajs[traj])
        for imitator in imitator_ensemble:
            with torch.no_grad():
                action_probs = imitator.get_log_prob(torch.FloatTensor(state_np), torch.FloatTensor(action))
                new_reward.append(action_probs)
        new_reward = torch.stack(new_reward, dim=2)
        rewards = np.log(torch.var(new_reward, dim=2))
        # rewards = torch.var(new_reward, dim=2)

        for reward in rewards:
            data.append([traj, np.float(reward)])
    data = pd.DataFrame(data, columns=['type', 'reward'])

    sns.boxplot(x='type', y='reward', data=data, whis=np.inf)
    # sns.stripplot(x='type', y='reward', data=data, color=".3")

    plt.show()