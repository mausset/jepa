import torch
from torch import nn

from x_transformers import CrossAttender

# from x_transformers import CrossAttender
from einops import repeat


class AttentiveProbe(nn.Module):
    def __init__(self, dim, heads, num_classes, features):
        super().__init__()
        self.features = features

        self.encoder = CrossAttender(
            dim=dim,
            # context_dim=dim,
            heads=heads,
            depth=1,
            ff_glu=True,
            attn_flash=True,
        )
        self.ln = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.projection = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x):
        b, *_ = x.shape
        x = self.ln(x)
        cls_token = repeat(self.cls_token, "n d -> b n d", b=b)
        cls_token = self.encoder(cls_token, context=x)
        return self.projection(cls_token)


class FFNProbe(nn.Module):

    def __init__(self, dim, num_classes, **_):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes),
        )

    def forward(self, x):
        x = self.ffn(x)
        return x
