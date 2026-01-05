import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, embedding_dim):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, num_hiddens // 2, 4, stride=2, padding=1)

        self.down = nn.Sequential(
            nn.GroupNorm(32, num_hiddens // 2),
            nn.SiLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, 4, stride=2, padding=1),
        )

        self.res = nn.Sequential(
            ResBlock(num_hiddens),
            ResBlock(num_hiddens),
        )

        self.conv_out = nn.Conv2d(num_hiddens, embedding_dim, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.res(x)
        return self.conv_out(x)
