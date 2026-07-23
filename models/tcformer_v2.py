import torch
import torch.nn as nn

from .tokenizer_v2 import TokenizerV2
from .encoder import TransformerEncoder
from .classifier_v2 import ClassifierV2


class TCFormerV2(nn.Module):

    def __init__(
        self,
        num_channels=22,
        embed_dim=128,
        depth=6,
        q_heads=8,
        kv_heads=2,
        head_dim=16,
        num_classes=4,
        dropout=0.3,
    ):

        super().__init__()

        ####################################################
        # Tokenizer
        ####################################################

        self.tokenizer = TokenizerV2(
            num_channels=num_channels,
            embed_dim=embed_dim
        )

        ####################################################
        # Positional Dropout
        ####################################################

        self.pos_dropout = nn.Dropout(dropout)

        ####################################################
        # Transformer
        ####################################################

        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        ####################################################
        # LayerNorm
        ####################################################

        self.norm = nn.LayerNorm(embed_dim)

        ####################################################
        # Pointwise Reduce
        ####################################################

        self.reduce = nn.Sequential(

            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm1d(embed_dim),

            nn.ELU(inplace=True)

        )

        ####################################################
        # Classifier
        ####################################################

        self.classifier = ClassifierV2(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    ########################################################

    def forward(self, x):

        ####################################################
        # Tokenizer
        ####################################################
        if x.dim() == 3:
            x = x.unsqueeze(1)
        tokens, cnn_features = self.tokenizer(x)

        ####################################################
        # Positional Dropout
        ####################################################

        tokens = self.pos_dropout(tokens)

        ####################################################
        # Transformer
        ####################################################

        transformer_features = self.encoder(tokens)

        ####################################################
        # Residual
        ####################################################

        transformer_features = transformer_features + tokens

        transformer_features = self.norm(
            transformer_features
        )

        ####################################################
        # (B,N,C) -> (B,C,N)
        ####################################################

        transformer_features = transformer_features.transpose(1,2)

        ####################################################
        # Reduce
        ####################################################

        transformer_features = self.reduce(
            transformer_features
        )

        ####################################################
        # CNN Branch
        ####################################################

        cnn_features = cnn_features.squeeze(2)

        ####################################################
        # Concatenate
        ####################################################

        fused = torch.cat(

            [
                transformer_features,
                cnn_features
            ],

            dim=1

        )

        ####################################################
        # Classifier
        ####################################################

        logits = self.classifier(fused)

        return logits
