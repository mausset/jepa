import torch
from torch import nn
from x_transformers import Decoder


class TransformerActionDecoder(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.dim = config["in_dim"]
        self.action_dim = config["action_dim"]

        self.model = Decoder(
            dim=self.dim,
            heads=int(config.get("heads", 8)),
            depth=int(config.get("depth", 4)),
            ff_glu=True,
            attn_flash=bool(config.get("attn_flash", True)),
            rotary_pos_emb=True,
        )
        self.out_projection = nn.Linear(self.dim, self.action_dim)

    def forward(self, x):
        x = self.model(x)
        return self.out_projection(x)
