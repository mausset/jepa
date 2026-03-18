import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn

from jepa.models.modules import SwiGLUFFN


class ViTBlock(nn.Module):
    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)

        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm_q = nn.RMSNorm(self.HEAD_DIM)
        self.norm_k = nn.RMSNorm(self.HEAD_DIM)

        self.norm_attn = nn.RMSNorm(self.dim)
        self.norm_ffn = nn.RMSNorm(self.dim)

        self.ffn = SwiGLUFFN(dim, expansion=expansion)

        self.sdpa_list = [
            nn.attention.SDPBackend.FLASH_ATTENTION,
            nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        ]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, D = x.shape

        x_pre_attn = x
        x = self.norm_attn(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        with nn.attention.sdpa_kernel(self.sdpa_list):
            attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.attn_out(attn_output)

        x = x_pre_attn + attn_output
        x = x + self.ffn(self.norm_ffn(x))

        return x


class ViT(nn.Module):
    def __init__(self, encoder_args) -> None:
        super().__init__()

        self.dim = encoder_args["dim"]
        self.heads = encoder_args["heads"]
        self.depth = encoder_args["depth"]

        self.n_registers = encoder_args["n_registers"]

        self.resolution = encoder_args["resolution"]
        self.patch_size = encoder_args["patch_size"]
        self.n_patches = (self.resolution // self.patch_size) ** 2

        self.pe = nn.Parameter(torch.zeros(1, self.n_patches, self.dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

        self.registers = nn.Parameter(torch.zeros(self.n_registers, self.dim))
        nn.init.trunc_normal_(self.registers, std=0.02)

        self.patch_embed = nn.Sequential(
            Rearrange("b h w c -> b c h w"),
            nn.Conv2d(
                in_channels=3,
                out_channels=encoder_args["dim"],
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(self.dim, self.heads) for _ in range(self.depth)]
        )

    def forward_features(self, x):
        B, *_ = x.shape

        x = self.patch_embed(x)
        x = x + self.pe

        r = repeat(self.registers, "n d -> b n d", b=B)
        x, ps = pack((r, x), "b * d")

        for layer in self.blocks[:-1]:
            x = layer(x)

        r, x = unpack(x, ps, "b * d")

        return {"register": r, "feature_map": x}

    def forward(self, x):
        B, *_ = x.shape

        pe = self.pe
        x = self.patch_embed(x)
        x = x + pe

        r = repeat(self.registers, "n d -> b n d", b=B)
        x, ps = pack((r, x), "b * d")

        for layer in self.blocks:
            x = layer(x)

        r, x = unpack(x, ps, "b * d")

        return {"register": r, "feature_map": x}
