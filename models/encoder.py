import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, embedding_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)
