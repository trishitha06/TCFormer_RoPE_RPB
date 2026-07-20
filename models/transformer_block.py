import torch
import torch.nn as nn

from .gqa import GroupedQueryAttention
from .mlp import MLP


##############################################################
# DropPath (Stochastic Depth)
##############################################################

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):

        super().__init__()

        self.drop_prob = drop_prob

    def forward(self, x):

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )

        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


##############################################################
# Transformer Block
##############################################################

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        q_heads,
        kv_heads,
        head_dim,
        mlp_ratio=4,
        dropout=0.3,
        attention_dropout=0.1,
        drop_path=0.1,
    ):

        super().__init__()

        ###########################################
        # Pre-Norm
        ###########################################

        self.norm1 = nn.LayerNorm(embed_dim)

        ###########################################
        # GQA + RoPE + RPB
        ###########################################

        self.attn = GroupedQueryAttention(

            embed_dim=embed_dim,

            num_query_heads=q_heads,

            num_kv_heads=kv_heads,

            head_dim=head_dim,

            dropout=attention_dropout

        )

        ###########################################

        self.drop_path1 = DropPath(drop_path)

        ###########################################

        self.norm2 = nn.LayerNorm(embed_dim)

        ###########################################

        self.mlp = MLP(

            dim=embed_dim,

            mlp_ratio=mlp_ratio,

            dropout=dropout

        )

        ###########################################

        self.drop_path2 = DropPath(drop_path)

    ###########################################################

    def forward(self, x, mask=None):

        ###########################################
        # Attention
        ###########################################

        x = x + self.drop_path1(

            self.attn(

                self.norm1(x),

                mask

            )

        )

        ###########################################
        # Feed Forward
        ###########################################

        x = x + self.drop_path2(

            self.mlp(

                self.norm2(x)

            )

        )

        return x