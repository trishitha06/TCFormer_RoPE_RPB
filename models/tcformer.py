import torch
import torch.nn as nn

from .tokenizer import MultiScaleTokenizer
from .encoder import TransformerEncoder
from .classifier import Classifier


class TCFormer(nn.Module):

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

        ###################################################
        # Tokenizer
        ###################################################

        self.tokenizer = MultiScaleTokenizer(

            num_channels=num_channels,

            embed_dim=embed_dim,

            dropout=dropout

        )

        ###################################################
        # Positional Dropout
        ###################################################

        self.pos_dropout = nn.Dropout(dropout)

        ###################################################
        # Transformer Encoder
        ###################################################

        self.encoder = TransformerEncoder(

            depth=depth,

            embed_dim=embed_dim,

            q_heads=q_heads,

            kv_heads=kv_heads,

            head_dim=head_dim,

            dropout=dropout,

        )

        ###################################################
        # Final LayerNorm
        ###################################################

        self.norm = nn.LayerNorm(embed_dim)

        ###################################################
        # Classifier
        ###################################################

        self.classifier = Classifier(

            embed_dim=embed_dim,

            num_classes=num_classes,

            dropout=dropout

        )

    #########################################################

    def forward(self, x):

        ###########################################
        # EEG → Tokens
        ###########################################

        tokens = self.tokenizer(x)

        ###########################################

        tokens = self.pos_dropout(tokens)

        ###########################################
        # Transformer
        ###########################################

        features = self.encoder(tokens)

        ###########################################
        # Residual Token Fusion
        ###########################################

        features = features + tokens

        ###########################################

        features = self.norm(features)

        ###########################################
        # Classification
        ###########################################

        logits = self.classifier(features)

        return logits