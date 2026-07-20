import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding
from .relative_position_bias import RelativePositionBias


class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dropout=0.1,
        use_rope=True,
        use_rpb=True,
        max_position=128,
    ):

        super().__init__()

        assert (
            num_query_heads % num_kv_heads == 0
        ), "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads

        self.use_rope = use_rope
        self.use_rpb = use_rpb

        #######################################
        # QKV Projection
        #######################################

        self.q_proj = nn.Linear(
            embed_dim,
            num_query_heads * head_dim,
            bias=True
        )

        self.k_proj = nn.Linear(
            embed_dim,
            num_kv_heads * head_dim,
            bias=True
        )

        self.v_proj = nn.Linear(
            embed_dim,
            num_kv_heads * head_dim,
            bias=True
        )

        #######################################

        self.out_proj = nn.Linear(
            num_query_heads * head_dim,
            embed_dim
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        #######################################

        if use_rope:
            self.rope = RotaryEmbedding(head_dim)

        if use_rpb:
            self.rpb = RelativePositionBias(
                num_query_heads,
                max_position
            )

        self._init_weights()

    ##########################################################

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    ##########################################################

    def forward(self, x, mask=None):

        B, N, _ = x.shape

        #######################################
        # Linear Projection
        #######################################

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        #######################################

        q = q.view(
            B,
            N,
            self.num_query_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            N,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            N,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        #######################################
        # RoPE
        #######################################

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        #######################################
        # Expand KV Heads
        #######################################

        k = k.repeat_interleave(
            self.group_size,
            dim=1
        )

        v = v.repeat_interleave(
            self.group_size,
            dim=1
        )

        #######################################
        # Attention
        #######################################

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        #######################################
        # Relative Position Bias
        #######################################

        if self.use_rpb:
            scores = scores + self.rpb(N).unsqueeze(0)

        #######################################
        # Mask (optional)
        #######################################

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        #######################################
        # Softmax
        #######################################

        attn = F.softmax(
            scores,
            dim=-1
        )

        attn = self.attn_dropout(attn)

        #######################################
        # Attention × Value
        #######################################

        out = torch.matmul(
            attn,
            v
        )

        #######################################
        # Merge Heads
        #######################################

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            N,
            self.num_query_heads * self.head_dim
        )

        #######################################
        # Output Projection
        #######################################

        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out