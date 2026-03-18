import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from jepa.models.modules import SwiGLUFFN
from jepa.utils.helpers import block_attention_mask


def modulate(x, scale, shift):
    return x * (1 + scale) + shift


class PredictorBlock(nn.Module):
    HEAD_DIM = 64

    def __init__(self, dim, heads, rope, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.rope = rope

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)

        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm_q = nn.RMSNorm(self.HEAD_DIM)
        self.norm_k = nn.RMSNorm(self.HEAD_DIM)

        self.norm_attn = nn.RMSNorm(self.dim)
        self.norm_ffn = nn.RMSNorm(self.dim)

        self.ffn = SwiGLUFFN(self.dim, expansion=expansion)

        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

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

        nn.init.zeros_(self.ada_ln[-1].weight)  # type: ignore
        nn.init.zeros_(self.ada_ln[-1].bias)  # type: ignore

    def forward(self, x, latent, attn_mask=None):
        B, N, D = x.shape

        attn_scale, attn_shift, attn_gate, ffn_scale, ffn_shift, ffn_gate = self.ada_ln(
            latent
        ).chunk(6, dim=-1)

        x_pre_attn = x
        x = modulate(self.norm_attn(x), attn_scale, attn_shift)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "b ... -> b 1 ...")

        with nn.attention.sdpa_kernel(self.sdpa_list):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.attn_out(attn_output)

        x = x_pre_attn + attn_gate * attn_output
        x = x + ffn_gate * self.ffn(modulate(self.norm_ffn(x), ffn_scale, ffn_shift))

        return x


class Predictor(nn.Module):
    def __init__(self, predictor_args) -> None:
        super().__init__()

        self.dim = predictor_args["dim"]
        self.heads = predictor_args["heads"]
        self.depth = predictor_args["depth"]
        self.context = predictor_args["context"]
        self.noise_dim = predictor_args["noise_dim"]

        self.rope = RotaryEmbedding(64)

        self.noise_embed = nn.Sequential(
            nn.Linear(self.noise_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    self.dim,
                    self.heads,
                    self.rope,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x):
        B, T, N, D = x.shape

        attn_mask = block_attention_mask(x)
        attn_mask = repeat(attn_mask, "m n -> b m n", b=B)

        x = rearrange(x, "b t n d ->  b (t n) d")

        latent = torch.randn(B, T * N, self.noise_dim, device=x.device)
        latent = self.noise_embed(latent)

        for block in self.blocks:
            x = block(x, latent, attn_mask=attn_mask)
        x = rearrange(x, "b (t n) d -> b t n d", n=N)

        return x
