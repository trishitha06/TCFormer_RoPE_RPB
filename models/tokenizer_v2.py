import torch
import torch.nn as nn

from models.grouped_conv import (
    GroupedPointwiseConv,
    GroupedTemporalConv
)
from models.group_attention import GroupAttention


class MultiKernelTemporalCNN(nn.Module):
    """
    Multi-Kernel Temporal Convolution Block
    """

    def __init__(self, in_channels=1, out_channels=16):

        super().__init__()

        self.branch1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 15),
            padding=(0, 7),
            bias=False
        )

        self.branch2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 31),
            padding=(0, 15),
            bias=False
        )

        self.branch3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 63),
            padding=(0, 31),
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels * 3)

        self.act = nn.ELU(inplace=True)

    def forward(self, x):

        b1 = self.branch1(x)

        b2 = self.branch2(x)

        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim=1)

        x = self.bn(x)

        x = self.act(x)

        return x


#############################################################


class TokenizerV2(nn.Module):

    def __init__(
        self,
        num_channels=22,
        embed_dim=128
    ):

        super().__init__()

        #########################################
        # Multi Kernel CNN
        #########################################

        self.temporal = MultiKernelTemporalCNN(
            in_channels=1,
            out_channels=16
        )

        #########################################
        # Depthwise Conv
        #########################################

        self.spatial = nn.Sequential(

            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(num_channels, 1),
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ELU(inplace=True)

)

        #########################################
        # Average Pool
        #########################################

        self.pool1 = nn.AvgPool2d(
            kernel_size=(1, 4),
            stride=(1, 4)
        )

        #########################################
        # Grouped Pointwise Conv
        #########################################

        self.group_pw = GroupedPointwiseConv(
            in_channels=64,
            out_channels=64,
            groups=4
        )

        #########################################
        # Grouped Temporal Conv
        #########################################

        self.group_temp = GroupedTemporalConv(
            channels=64,
            kernel_size=9,
            groups=4
        )

        #########################################
        # Group Attention
        #########################################

        self.group_attention = GroupAttention(
            channels=64
        )

        #########################################
        # Second Average Pool
        #########################################

        self.pool2 = nn.AvgPool2d(
            kernel_size=(1, 2),
            stride=(1, 2)
        )

        #########################################
        # Pointwise Mixing
        #########################################

        self.mix = nn.Sequential(

            nn.Conv2d(
                64,
                embed_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(embed_dim),

            nn.ELU(inplace=True)

        )

    def forward(self, x):
        """
        Input:
            x : (B, 1, 22, T)

        Output:
            tokens : (B, N, embed_dim)
            cnn_features : (B, embed_dim, 1, N)
        """

        # Multi-Kernel Temporal CNN
        x = self.temporal(x)

        # Depthwise Spatial Convolution
        x = self.spatial(x)

        # Average Pool
        x = self.pool1(x)

        # Grouped Pointwise Convolution
        x = self.group_pw(x)

        # Grouped Temporal Convolution
        x = self.group_temp(x)

        # Group Attention
        x = self.group_attention(x)

        # Average Pool
        x = self.pool2(x)

        # Pointwise Feature Mixing
        cnn_features = self.mix(x)

        # Remove spatial dimension
        tokens = cnn_features.squeeze(2)

        # (B, C, T) -> (B, T, C)
        tokens = tokens.transpose(1, 2)

        return tokens, cnn_features
