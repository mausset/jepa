import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn
from vector_quantize_pytorch import FSQ

from jepa.models.modules import SwiGLUFFN
from jepa.utils.helpers import block_attention_mask


VIT_VARIANTS = {
    "vit-t": {
        "dim": 192,
        "depth": 12,
        "heads": 3,
    },
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


BOTTLENECK_TYPES = ("none", "fsq", "vae")


def build_predictor_config(config: dict) -> dict:
    arch = config.get("arch", "vit-s")
    if arch in VIT_VARIANTS:
        resolved = dict(VIT_VARIANTS[arch])
    else:
        raise ValueError(f"Unknown predictor arch: {arch}")

    resolved.update(config)
    resolved["arch"] = arch
    return resolved


def resolve_bottleneck_type(predictor_args) -> str:
    bottleneck = predictor_args.get("bottleneck")
    if bottleneck is None:
        # Back-compat: infer from fsq_levels presence.
        return "fsq" if predictor_args.get("fsq_levels") is not None else "none"
    if bottleneck not in BOTTLENECK_TYPES:
        raise ValueError(f"Unknown bottleneck: {bottleneck!r}; choose from {BOTTLENECK_TYPES}")
    return bottleneck


class Projector(nn.Module):
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


class PredictorBlock(nn.Module):
    """Transformer block with RoPE."""

    HEAD_DIM = 64

    def __init__(self, dim, heads, rope, expansion=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM

        self.rope = rope

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.norm_attn = nn.LayerNorm(self.dim)
        self.norm_ffn = nn.LayerNorm(self.dim)

        self.ffn = SwiGLUFFN(self.dim, expansion=expansion, dropout=dropout)

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
        x = x + self.drop(self.attn_out(attn_output))
        x = x + self.ffn(self.norm_ffn(x))

        return x


class Predictor(nn.Module):
    LOGVAR_CLAMP = (-10.0, 5.0)

    def __init__(self, predictor_args) -> None:
        super().__init__()

        predictor_args = build_predictor_config(predictor_args)
        self.arch = predictor_args["arch"]
        self.dim = predictor_args["dim"]
        self.heads = predictor_args["heads"]
        self.depth = predictor_args["depth"]
        self.context = predictor_args["context"]
        self.bottleneck_type = resolve_bottleneck_type(predictor_args)

        self.rope = RotaryEmbedding(64, theta=100.0)
        self.dropout = predictor_args.get("dropout", 0.0)

        if self.bottleneck_type == "none":
            self.latent_dim = None
            self.fsq_levels = None
            self.blocks = nn.ModuleList(
                [PredictorBlock(self.dim, self.heads, self.rope, dropout=self.dropout) for _ in range(self.depth)]
            )
        else:
            enc_depth = self.depth // 2
            pred_depth = self.depth - enc_depth

            self.enc_blocks = nn.ModuleList(
                [PredictorBlock(self.dim, self.heads, self.rope, dropout=self.dropout) for _ in range(enc_depth)]
            )
            self.pred_blocks = nn.ModuleList(
                [PredictorBlock(self.dim, self.heads, self.rope, dropout=self.dropout) for _ in range(pred_depth)]
            )

            if self.bottleneck_type == "fsq":
                self.fsq_levels = list(predictor_args["fsq_levels"])
                self.latent_dim = len(self.fsq_levels)
                self.fsq = FSQ(levels=self.fsq_levels)
                head_out = self.latent_dim
            else:  # "vae"
                self.fsq_levels = None
                self.latent_dim = int(predictor_args["latent_dim"])
                head_out = 2 * self.latent_dim

            self.bottleneck_hidden = nn.Sequential(
                nn.Linear(self.dim, self.dim // 2),
                nn.GELU(),
            )
            self.bottleneck_head = nn.Linear(self.dim // 2, head_out)
            self.latent_embed = nn.Sequential(
                nn.Linear(self.latent_dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )

        proj_norm = predictor_args.get("projector_norm", "none")
        use_proj = predictor_args.get("projector", False)
        self.projector = (
            Projector(self.dim, norm=proj_norm) if use_proj else nn.Identity()
        )

    @property
    def has_bottleneck(self):
        return self.bottleneck_type != "none"

    def encoder_half(self, x):
        """Run encoder-half transformer blocks over (B, T, N, D) → (B, T, N, D)."""
        B = x.shape[0]
        mask = block_attention_mask(x)
        mask = repeat(mask, "m n -> b m n", b=B)
        for block in self.enc_blocks:
            x = block(x, attn_mask=mask)
        return x

    def predictor_half(self, x):
        """Run predictor-half transformer blocks over (B, T, N, D) → (B, T, N, D)."""
        B = x.shape[0]
        mask = block_attention_mask(x)
        mask = repeat(mask, "m n -> b m n", b=B)
        for block in self.pred_blocks:
            x = block(x, attn_mask=mask)
        return self.projector(x)

    def bottleneck(self, hidden, stochastic=True):
        """Map `hidden` (B, T, N, D) through the bottleneck.

        Returns a dict with:
            latent: (B, T, N, latent_dim) — quantized (FSQ) or reparameterized (VAE).
            kl / mu / logvar: VAE-only scalars/tensors; kl is summed over latent_dim
                and averaged over (B, T-1, N) positions.
        """
        B, T, N, D = hidden.shape
        h = self.bottleneck_hidden(hidden)
        raw = self.bottleneck_head(h)

        if self.bottleneck_type == "fsq":
            quantized_flat, _ = self.fsq(rearrange(raw, "b t n d -> (b t) n d"))
            latent = rearrange(quantized_flat, "(b t) n d -> b t n d", b=B, t=T)
            return {"latent": latent}

        mu, logvar = raw.chunk(2, dim=-1)
        logvar = logvar.clamp(*self.LOGVAR_CLAMP)
        if stochastic:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z = mu
        kl_full = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (B, T, N, latent_dim)
        kl = kl_full[:, 1:].sum(dim=-1).mean()
        return {"latent": z, "kl": kl, "mu": mu, "logvar": logvar}

    def sample_prior(self, shape, device):
        """Sample from the bottleneck's prior.

        Args:
            shape: tuple of batch dims, e.g. (B, 1, N).
        Returns:
            (*shape, latent_dim) on `device`.
        """
        if self.bottleneck_type == "fsq":
            idx = torch.randint(0, self.fsq.codebook_size, shape, device=device)
            return self.fsq.implicit_codebook.to(device)[idx]
        if self.bottleneck_type == "vae":
            return torch.randn(*shape, self.latent_dim, device=device)
        raise ValueError(f"sample_prior requires fsq or vae bottleneck, got {self.bottleneck_type!r}")

    def sample(self, x, latent=None):
        """Predict all T positions using posterior latents for frames 1..T-1 and a
        prior sample (or caller-provided `latent`) for the unknown frame T.

        Returns (B, T, N, D); caller takes [:, -1] for the frame-T prediction.
        """
        if not self.has_bottleneck:
            raise ValueError("sample requires a bottleneck (fsq or vae)")

        B, T, N, D = x.shape
        hidden = self.encoder_half(x)
        bn = self.bottleneck(hidden, stochastic=False)
        inferred = bn["latent"]  # (B, T, N, latent_dim)

        if latent is None:
            sampled = self.sample_prior((B, 1, N), device=x.device)
        else:
            sampled = latent

        latent_seq = torch.cat([inferred[:, 1:], sampled], dim=1)  # (B, T, N, latent_dim)
        x_full = hidden + self.latent_embed(latent_seq)
        return self.predictor_half(x_full)

    def forward(self, x):
        """Return a dict with 'pred' and VAE-only 'kl' / 'mu' / 'logvar'.

        bottleneck=none: pred is (B, T, N, D).
        bottleneck in {fsq, vae}: pred is (B, T-1, N, D).
        """
        B, T, N, D = x.shape

        if not self.has_bottleneck:
            mask = block_attention_mask(x)
            mask = repeat(mask, "m n -> b m n", b=B)
            for block in self.blocks:
                x = block(x, attn_mask=mask)
            return {"pred": self.projector(x)}

        hidden = self.encoder_half(x)
        bn = self.bottleneck(hidden, stochastic=True)
        x_context = hidden[:, :-1] + self.latent_embed(bn["latent"][:, 1:])
        pred = self.predictor_half(x_context)
        out = {"pred": pred}
        for k in ("kl", "mu", "logvar"):
            if k in bn:
                out[k] = bn[k]
        return out
