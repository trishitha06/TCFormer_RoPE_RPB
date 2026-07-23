import torch
import torch.nn as nn


class DepthwiseConv(nn.Module):
    """
    Depthwise Convolution Block

    Input :
        (B, C, H, W)

    Output:
        (B, C, H, W)
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=True,
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(channels)

        self.activation = activation

        if activation:
            self.act = nn.ELU(inplace=True)

    def forward(self, x):

        x = self.depthwise(x)

        x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x
