import torch.nn as nn

from .transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        depth,
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

        ####################################################
        # Progressive DropPath
        ####################################################

        dpr = [

            x.item()

            for x in __import__("torch").linspace(
                0,
                drop_path,
                depth
            )

        ]

        ####################################################

        self.layers = nn.ModuleList(

            [

                TransformerBlock(

                    embed_dim=embed_dim,

                    q_heads=q_heads,

                    kv_heads=kv_heads,

                    head_dim=head_dim,

                    mlp_ratio=mlp_ratio,

                    dropout=dropout,

                    attention_dropout=attention_dropout,

                    drop_path=dpr[i]

                )

                for i in range(depth)

            ]

        )

        ####################################################

        self.norm = nn.LayerNorm(embed_dim)

    ########################################################

    def forward(self, x, mask=None):

        for layer in self.layers:

            x = layer(

                x,

                mask

            )

        x = self.norm(x)

        return x