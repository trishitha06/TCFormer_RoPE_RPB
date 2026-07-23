import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):

    def __init__(
            self,
            num_heads,
            max_position=128
    ):

        super().__init__()

        self.max_position = max_position

        self.bias_table = nn.Parameter(

            torch.zeros(
                2 * max_position - 1,
                num_heads
            )

        )

        nn.init.trunc_normal_(
            self.bias_table,
            std=0.02
        )

    def forward(self, seq_len):

        device = self.bias_table.device

        coords = torch.arange(
            seq_len,
            device=device
        )

        relative = coords[:, None] - coords[None, :]

        relative = relative.clamp(
            -self.max_position + 1,
            self.max_position - 1
        )

        relative += self.max_position - 1

        bias = self.bias_table[relative]

        bias = bias.permute(2, 0, 1)

        return bias
