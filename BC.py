import numpy as np
import math
import torch
import torch.nn as nn


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(400, 300), activation='relu', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)

        #action = torch.normal(action_mean, action_std)
        action = torch.normal(torch.zeros_like(action_mean),
                              torch.ones_like(action_mean)) * action_std + action_mean
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


class BC(object):
    """
    A vanilla Behavior Cloning model.
    """
    def __init__(self, args, state_dim, action_dim, max_action):

        self.device = args.device
        self.config = args

        self.state_dim = state_dim
        self.actor = Policy(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=1e-3)
        self.actor_loss = nn.MSELoss()

    def set_expert(self, expert_traj):
        """
        Set the expert trajectories.
        :param expert_traj:
        :return:
        """
        self.expert_traj = expert_traj
        self.expert_states = []
        self.expert_actions = []
        for i in range(len(self.expert_traj)):
            self.expert_states.append(self.expert_traj[i][0])
            self.expert_actions.append(self.expert_traj[i][2])
        # self.expert_actions = torch.FloatTensor(self.expert_actions)
        # self.expert_states = torch.FloatTensor(self.expert_states)


    def train(self, batch_size=None):
        """
        :param num_traj:
        :return:
        """
        ind = np.random.randint(0, len(self.expert_states), size=batch_size)

        expert_states = []
        expert_actions = []

        for i in ind:
            expert_states.append(self.expert_states[i])
            expert_actions.append(self.expert_actions[i])

            # s, s2, a, r, d = self.storage[i]
            # state.append(np.array(s, copy=False))
            # next_state.append(np.array(s2, copy=False))
            # action.append(np.array(a, copy=False))
            # reward.append(np.array(r, copy=False))
            # done.append(np.array(d, copy=False))

        # return (np.array(state),
        #     np.array(next_state),
        #     np.array(action),
        #     np.array(reward).reshape(-1, 1),
        #     np.array(done).reshape(-1, 1))

        expert_states = torch.FloatTensor(self.expert_states).to(self.device)
        expert_actions = torch.FloatTensor(self.expert_actions).to(self.device)

        predicted_actions = self.actor.select_action(expert_states)
        self.actor_optimizer.zero_grad()
        loss = self.actor_loss(predicted_actions, expert_actions)
        loss.backward()
        self.actor_optimizer.step()

        return loss.to('cpu').detach().numpy()