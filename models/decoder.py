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

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, out_channels):
        super().__init__()

        self.conv_in = nn.Conv2d(embedding_dim, num_hiddens, 3, padding=1)

        self.res = nn.Sequential(
            ResBlock(num_hiddens),
            ResBlock(num_hiddens),
            ResBlock(num_hiddens),
        )

        self.up1 = nn.Sequential(
            nn.GroupNorm(32, num_hiddens),
            nn.SiLU(),
            nn.ConvTranspose2d(
                num_hiddens, num_hiddens // 2, 4, stride=2, padding=1
            ),
        )

        self.up2 = nn.ConvTranspose2d(
            num_hiddens // 2, out_channels, 4, stride=2, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res(x)
        x = self.up1(x)
        return torch.sigmoid(self.up2(x))
