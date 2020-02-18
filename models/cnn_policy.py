import torch.nn as nn
import torch
# from utils.math import *


class CNNPolicy(nn.Module):
    """
    Convolutional Policy from Nature DQN.
    """
    def __init__(self, state_dim, action_num):
        super().__init__()
        self.is_disc_action = True
        self.activation = torch.relu

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1)
        )

        # 3136 = 56 * 56, figure out why
        self.decoder = nn.Sequential(
            nn.Linear(3136, 512),
            nn.Linear(512, action_num)
        )

    def forward(self, x):
        if not isinstance(x.size, int):
            x = x.view(x.size(0), 4, 84, 84)
        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        action_prob = torch.softmax(x, dim=1)
        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

