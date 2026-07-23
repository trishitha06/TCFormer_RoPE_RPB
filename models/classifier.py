import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(
            self,
            embed_dim=64,
            num_classes=4,
            dropout=0.5
    ):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x):

        # x : (B,T,C)

        x = x.transpose(1,2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc(x)

        return x
