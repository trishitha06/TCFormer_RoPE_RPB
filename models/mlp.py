import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4,
                 dropout=0.3):

        super().__init__()

        hidden = dim * mlp_ratio

        self.net = nn.Sequential(

            nn.Linear(dim, hidden),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden, dim),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        return self.net(x)