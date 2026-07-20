import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    """
    Temporal convolution branch.
    """

    def __init__(self, kernel_size):

        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ELU()

        )

    def forward(self, x):

        return self.block(x)


class MultiScaleTokenizer(nn.Module):

    def __init__(

        self,

        num_channels=22,

        embed_dim=64,

        dropout=0.3

    ):

        super().__init__()

        ######################################
        # Three Temporal Scales
        ######################################

        self.branch1 = ConvBranch(32)

        self.branch2 = ConvBranch(64)

        self.branch3 = ConvBranch(128)

        ######################################
        # Spatial Filtering
        ######################################

        self.spatial = nn.Sequential(

            nn.Conv2d(

                48,

                64,

                kernel_size=(num_channels, 1),

                bias=False

            ),

            nn.BatchNorm2d(64),

            nn.ELU()

        )

        ######################################
        # Patch Embedding
        ######################################

        self.patch_embed = nn.Conv1d(

            64,

            embed_dim,

            kernel_size=1,

            bias=False

        )

        ######################################
        # Pooling
        ######################################

        self.pool = nn.AvgPool1d(

            kernel_size=8,

            stride=8

        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        x : (B,22,1000)
        """

        x = x.unsqueeze(1)

        b1 = self.branch1(x)

        b2 = self.branch2(x)

        b3 = self.branch3(x)

        x = torch.cat(

            [

                b1,

                b2,

                b3

            ],

            dim=1

        )

        x = self.spatial(x)

        x = x.squeeze(2)

        x = self.patch_embed(x)

        x = self.pool(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)

        return x