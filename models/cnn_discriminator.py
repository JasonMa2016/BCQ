import numpy as np
import torch.nn as nn
import torch


class CNNDiscriminator(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
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
            nn.Linear(3137, 512),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        if not isinstance(state.size, int):
            state = state.view(state.size(0), 4, 84, 84)
        x = self.conv_layers(state)
        x = x.view(x.size(0), -1)
        # can I do this more elegantly?
        action = torch.tensor(action).unsqueeze(0)
        action = action.view(-1, 1)
        x = torch.DoubleTensor(torch.cat([x, action.double()], dim=1))
        logits = self.decoder(x)
        return logits