import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

from jepa.models.modules import SwiGLUFFN
from jepa.utils.helpers import block_attention_mask


VIT_VARIANTS = {
    "vit-s": {
        "dim": 384,
        "depth": 12,
        "heads": 6,
    },
    "vit-b": {
        "dim": 768,
        "depth": 12,
        "heads": 12,
    },
    "vit-l": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
    },
}

MLP_VARIANTS = {
    "mlp": {
        "dim": 384,
        "hidden_dim": 384,
        "num_layers": 4,
    },
}


def build_predictor_config(config: dict) -> dict:
    arch = config.get("arch", "vit-s")
    if arch in VIT_VARIANTS:
        resolved = dict(VIT_VARIANTS[arch])
    elif arch in MLP_VARIANTS:
        resolved = dict(MLP_VARIANTS[arch])
    else:
        raise ValueError(f"Unknown predictor arch: {arch}")

    resolved.update(config)
    resolved["arch"] = arch
    return resolved


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
        B, T, N, D = x.shape

        attn_scale, attn_shift, attn_gate, ffn_scale, ffn_shift, ffn_gate = self.ada_ln(
            latent
        ).chunk(6, dim=-1)

        x_pre_attn = x
        x = rearrange(modulate(self.norm_attn(x), attn_scale, attn_shift), "b t n d -> b (t n) d")

        q = rearrange(self.to_q(x), "b s (h d) -> b h s d", h=self.heads)
        k = rearrange(self.to_k(x), "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(self.to_v(x), "b s (h d) -> b h s d", h=self.heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        positions = repeat(torch.arange(T, device=x.device), "t -> (t n)", n=N).float()
        freqs = self.rope.forward(positions)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "b ... -> b 1 ...")

        with nn.attention.sdpa_kernel(self.sdpa_list):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn_output = rearrange(attn_output, "b h (t n) d -> b t n (h d)", t=T, n=N)
        attn_output = self.attn_out(attn_output)

        x = x_pre_attn + attn_gate * attn_output
        x = x + ffn_gate * self.ffn(modulate(self.norm_ffn(x), ffn_scale, ffn_shift))

        return x


class TransformerPredictor(nn.Module):
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

        latent = torch.randn(B, T, N, self.noise_dim, device=x.device)
        latent = self.noise_embed(latent)

        for block in self.blocks:
            x = block(x, latent, attn_mask=attn_mask)

        return x


class MLPPredictor(nn.Module):
    def __init__(self, predictor_args) -> None:
        super().__init__()

        self.dim = predictor_args["dim"]
        self.noise_dim = predictor_args["noise_dim"]
        hidden_dim = predictor_args["hidden_dim"]
        num_layers = predictor_args["num_layers"]

        if num_layers != 4:
            raise ValueError("MLP predictor currently expects num_layers == 4.")

        self.net = nn.Sequential(
            nn.Linear(self.dim + self.noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, x):
        latent = torch.randn(*x.shape[:-1], self.noise_dim, device=x.device)
        x = torch.cat((x, latent), dim=-1)
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, predictor_args) -> None:
        super().__init__()

        predictor_args = build_predictor_config(predictor_args)
        self.arch = predictor_args["arch"]

        if self.arch == "mlp":
            self.model = MLPPredictor(predictor_args)
        else:
            self.model = TransformerPredictor(predictor_args)

    def forward(self, x):
        return self.model(x)
