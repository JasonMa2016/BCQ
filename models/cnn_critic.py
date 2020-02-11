import torch.nn as nn
import torch


class CNNCritic(nn.Module):
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
            nn.Linear(3136, 512),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        if not isinstance(x.size, int):
            x = x.view(x.size(0), 4, 84, 84)
        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)
        logits = self.decoder(x)
        return logits