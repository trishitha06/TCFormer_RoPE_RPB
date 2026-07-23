import torch
import torch.nn as nn

from models.tcn import TCN


class ClassifierV2(nn.Module):
    """
    Classification Head

    Input:
        fused_features : (B, 2*embed_dim, N)

    Output:
        logits : (B, num_classes)
    """

    def __init__(
        self,
        embed_dim=128,
        num_classes=4,
        dropout=0.5
    ):
        super().__init__()

        ##########################################
        # Residual TCN
        ##########################################

        self.tcn = TCN(
            channels=embed_dim * 2,
            num_blocks=4,
            dropout=0.2
        )

        ##########################################
        # Global Average Pool
        ##########################################

        self.pool = nn.AdaptiveAvgPool1d(1)

        ##########################################
        # Dropout
        ##########################################

        self.dropout = nn.Dropout(dropout)

        ##########################################
        # Final Linear Layer
        ##########################################

        self.fc = nn.Linear(
            embed_dim * 2,
            num_classes
        )

    def forward(self, fused_features):

        x = self.tcn(fused_features)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        logits = self.fc(x)

        return logits
