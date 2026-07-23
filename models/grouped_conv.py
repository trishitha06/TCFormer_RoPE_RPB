import torch
import torch.nn as nn


class GroupedPointwiseConv(nn.Module):
    """
    Grouped 1x1 Convolution

    Used to reduce computation while mixing channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=4
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ELU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


#########################################################


class GroupedTemporalConv(nn.Module):
    """
    Grouped Temporal Convolution

    Applies temporal filtering independently
    on channel groups.
    """

    def __init__(
        self,
        channels,
        kernel_size=9,
        groups=4
    ):

        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=(1, kernel_size),

                padding=(0, padding),

                groups=groups,

                bias=False

            ),

            nn.BatchNorm2d(channels),

            nn.ELU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)
