import torch
import torch.nn as nn


class GroupAttention(nn.Module):
    """
    Group Attention Module

    Learns attention weights for each feature map
    using global context.

    Input:
        (B, C, H, W)

    Output:
        (B, C, H, W)
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.avg_pool(x)

        weights = self.attention(weights)

        return x * weights
