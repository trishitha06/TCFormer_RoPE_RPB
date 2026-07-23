import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Residual Temporal Convolution Block
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2
    ):

        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.act1 = nn.ELU(inplace=True)

        self.drop1 = nn.Dropout(dropout)

        ##################################################

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.drop2 = nn.Dropout(dropout)

        self.act2 = nn.ELU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        out = out + identity

        out = self.act2(out)

        return out


############################################################


class TCN(nn.Module):
    """
    Residual Temporal Convolution Network

    Input:
        (B, C, T)

    Output:
        (B, C, T)
    """

    def __init__(
        self,
        channels=128,
        num_blocks=4,
        dropout=0.2
    ):

        super().__init__()

        blocks = []

        for i in range(num_blocks):

            blocks.append(

                TemporalBlock(

                    channels=channels,

                    kernel_size=3,

                    dilation=2 ** i,

                    dropout=dropout

                )

            )

        self.network = nn.Sequential(*blocks)

    def forward(self, x):

        return self.network(x)
