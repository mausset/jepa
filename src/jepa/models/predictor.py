import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn
from vector_quantize_pytorch import FSQ

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


def build_predictor_config(config: dict) -> dict:
    arch = config.get("arch", "vit-s")
    if arch in VIT_VARIANTS:
        resolved = dict(VIT_VARIANTS[arch])
    else:
        raise ValueError(f"Unknown predictor arch: {arch}")

    resolved.update(config)
    resolved["arch"] = arch
    return resolved


def modulate(x, scale, shift):
    return x * (1 + scale) + shift


class MLPProjector(nn.Module):
    def __init__(self, dim, expansion=4, norm="bn"):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(hidden)
        elif norm == "ln":
            self.norm = nn.LayerNorm(hidden)
        elif norm in (None, "none"):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown projector norm: {norm}")
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        shape = x.shape
        x = self.fc1(x.reshape(-1, shape[-1]))
        x = self.act(self.norm(x))
        x = self.fc2(x)
        return x.reshape(shape)


class PlainBlock(nn.Module):
    """Transformer block without AdaLN conditioning, for the encoder half."""

    HEAD_DIM = 64

    def __init__(self, dim, heads, rope, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM

        self.rope = rope

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm_attn = nn.LayerNorm(self.dim)
        self.norm_ffn = nn.LayerNorm(self.dim)

        self.ffn = SwiGLUFFN(self.dim, expansion=expansion)

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

    def forward(self, x, attn_mask=None):
        B, T, N, D = x.shape

        x_flat = rearrange(self.norm_attn(x), "b t n d -> b (t n) d")

        q = rearrange(self.to_q(x_flat), "b s (h d) -> b h s d", h=self.heads)
        k = rearrange(self.to_k(x_flat), "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(self.to_v(x_flat), "b s (h d) -> b h s d", h=self.heads)

        positions = repeat(torch.arange(T, device=x.device), "t -> (t n)", n=N).float()
        freqs = self.rope.forward(positions)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "b ... -> b 1 ...")

        with nn.attention.sdpa_kernel(self.sdpa_list):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn_output = rearrange(attn_output, "b h (t n) d -> b t n (h d)", t=T, n=N)
        x = x + self.attn_out(attn_output)
        x = x + self.ffn(self.norm_ffn(x))

        return x


class PredictorBlock(nn.Module):
    """Transformer block with AdaLN conditioning, for the predictor half."""

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

        self.norm_attn = nn.LayerNorm(self.dim)
        self.norm_ffn = nn.LayerNorm(self.dim)

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
        x = rearrange(
            modulate(self.norm_attn(x), attn_scale, attn_shift), "b t n d -> b (t n) d"
        )

        q = rearrange(self.to_q(x), "b s (h d) -> b h s d", h=self.heads)
        k = rearrange(self.to_k(x), "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(self.to_v(x), "b s (h d) -> b h s d", h=self.heads)

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


class Predictor(nn.Module):
    def __init__(self, predictor_args) -> None:
        super().__init__()

        predictor_args = build_predictor_config(predictor_args)
        self.arch = predictor_args["arch"]
        self.dim = predictor_args["dim"]
        self.heads = predictor_args["heads"]
        self.depth = predictor_args["depth"]
        self.context = predictor_args["context"]
        self.use_fsq = predictor_args.get("fsq_levels") is not None
        self.fsq_levels = list(predictor_args["fsq_levels"]) if self.use_fsq else None

        enc_depth = self.depth // 2
        pred_depth = self.depth - enc_depth

        self.rope = RotaryEmbedding(64, theta=100.0)

        # --- encoder half (no AdaLN, full attention) ---
        self.enc_blocks = nn.ModuleList(
            [PlainBlock(self.dim, self.heads, self.rope) for _ in range(enc_depth)]
        )

        # --- FSQ bottleneck ---
        if self.use_fsq:
            fsq_dim = len(self.fsq_levels)
            self.fsq_mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim // 2),
                nn.GELU(),
                nn.Linear(self.dim // 2, fsq_dim),
            )
            self.fsq = FSQ(levels=self.fsq_levels)
            self.code_embed = nn.Sequential(
                nn.Linear(fsq_dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )

        # --- predictor half (AdaLN, causal block attention) ---
        self.pred_blocks = nn.ModuleList(
            [PredictorBlock(self.dim, self.heads, self.rope) for _ in range(pred_depth)]
        )
        proj_norm = predictor_args.get("projector_norm", "none")
        use_proj = predictor_args.get("projector", False)
        self.projector = (
            MLPProjector(self.dim, norm=proj_norm) if use_proj else nn.Identity()
        )

    def _enc_half(self, x):
        B = x.shape[0]
        enc_mask = block_attention_mask(x)
        enc_mask = repeat(enc_mask, "m n -> b m n", b=B)
        for block in self.enc_blocks:
            x = block(x, attn_mask=enc_mask)
        return x

    def _pred_half(self, x, latent, attn_mask):
        for block in self.pred_blocks:
            x = block(x, latent, attn_mask=attn_mask)
        x = self.projector(x)
        return x

    def _fsq_latent(self, x):
        """Compute FSQ latent from encoder output x (B, T, N, D)."""
        B, T = x.shape[:2]
        codes_raw = self.fsq_mlp(x)
        quantized_flat, _ = self.fsq(rearrange(codes_raw, "b t n d -> (b t) n d"))
        quantized = rearrange(quantized_flat, "(b t) n d -> b t n d", b=B, t=T)
        return self.code_embed(quantized[:, 1:])  # (B, T-1, N, D)

    def forward(self, x):
        B, T, N, D = x.shape
        x = self._enc_half(x)
        x_context = x[:, :-1]  # (B, T-1, N, D)

        if self.use_fsq:
            latent = self._fsq_latent(x)
        else:
            latent = torch.zeros_like(x_context)

        attn_mask = block_attention_mask(x_context)
        attn_mask = repeat(attn_mask, "m n -> b m n", b=B)
        return self._pred_half(x_context, latent, attn_mask)  # (B, T-1, N, D)

    def residual_forward(self, x):
        """Shared encoder half, then two batched predictor passes: null-latent (mean) and FSQ-latent (residual)."""
        B, T, N, D = x.shape
        x = self._enc_half(x)
        x_context = x[:, :-1]  # (B, T-1, N, D)

        null_latent = torch.zeros_like(x_context)
        fsq_latent = self._fsq_latent(x)

        attn_mask = block_attention_mask(x_context)
        attn_mask = repeat(attn_mask, "m n -> b m n", b=B)

        x_doubled = torch.cat([x_context, x_context], dim=0)
        latent_doubled = torch.cat([null_latent, fsq_latent], dim=0)
        out = self._pred_half(x_doubled, latent_doubled, attn_mask.repeat(2, 1, 1))
        pred_mean, pred_residual = out.chunk(2, dim=0)
        return pred_mean, pred_residual
