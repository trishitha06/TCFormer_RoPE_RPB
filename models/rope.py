import math
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, base=10000):

        super().__init__()

        self.dim = dim
        self.base = base

    def forward(self, x):

        # x : (B,H,N,D)

        B, H, N, D = x.shape

        device = x.device

        half = D // 2

        freq_seq = torch.arange(
            half,
            device=device,
            dtype=torch.float32
        )

        inv_freq = 1.0 / (
            self.base ** (freq_seq / half)
        )

        positions = torch.arange(
            N,
            device=device,
            dtype=torch.float32
        )

        sinusoid = torch.outer(
            positions,
            inv_freq
        )

        sin = sinusoid.sin()[None, None, :, :]
        cos = sinusoid.cos()[None, None, :, :]

        x1 = x[..., :half]
        x2 = x[..., half:]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)