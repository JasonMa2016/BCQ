import pickle
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from models.mlp_policy import Policy


def imitation_histogram():
    return


def likelihood(env_name='Walker2d-v2', model_name='GAIL', traj=5, seed=0, in_sample=True):
    expert_path = "expert_models/{}_PPO_0.p".format(env_name)

    policy, _, running_state, expert_args = pickle.load(open(expert_path, "rb"))

    buffer_name = "PPO_traj100_%s_0" % (env_name)
    expert_trajs = np.load("./buffers/"+buffer_name+".npy", allow_pickle=True)
    # expert_rewards = np.load("./buffers/"+buffer_name+"_rewards" + ".npy", allow_pickle=True)


    fig, ax = plt.subplots(figsize=(8, 4))

    for type in ['good', 'mixed']:
        imitator_name = '{}_PPO_{}_traj{}_seed{}_{}'.format(model_name, env_name, traj, seed, type)
        imitator = Policy(17,6)
        imitator.load_state_dict(torch.load('imitator_models/%s_actor.pth' % (imitator_name)))
        probs = []
        min_timestep = 10000
        for i in range(traj):
            prob = []
            if len(expert_trajs[i]) < min_timestep:
                min_timestep = len(expert_trajs[i])
            for sample in expert_trajs[i]:
                states = torch.FloatTensor(sample[0])
                actions = torch.FloatTensor(sample[2])
                # states = torch.FloatTensor(np.random.random(states.size()))
                log_prob = imitator.get_log_prob(states.unsqueeze(0),
                                                 actions.unsqueeze(0))[0][0].detach().numpy()
                prob.append(log_prob)
            probs.append(prob)

        for i in range(len(probs)):
            probs[i] = probs[i][:min_timestep]
        probs = np.array(probs)
        probs_mu = np.mean(probs, axis=0)
        probs_std = np.std(probs, axis=0)
        ax.plot([i for i in range(min_timestep)], probs_mu, label=type)
        ax.fill_between(min_timestep, probs_mu + probs_std, probs_mu - probs_std, alpha=0.4)

    ax.set_title('{} {} Imitator Log-Likelihood over Expert Trajectories'.format(env_name, model_name))
    ax.legend(loc='best')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Log Likelihood')
    plt.savefig('plots/{}_{}_likelihood.png'.format(model_name, env_name))
    plt.close()
    return

if __name__ == "__main__":
    likelihood('Walker2d-v2')