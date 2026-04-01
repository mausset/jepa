import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn
from torchvision.models import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
)

from jepa.models.modules import SwiGLUFFN


ENCODER_VARIANTS = {
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
    "convnext-t": {
        "backbone_dim": 768,
    },
    "convnext-s": {
        "backbone_dim": 768,
    },
    "convnext-b": {
        "backbone_dim": 1024,
    },
    "convnext-l": {
        "backbone_dim": 1536,
    },
}


def build_encoder_config(config: dict) -> dict:
    arch = config.get("arch", "vit-s")
    if arch not in ENCODER_VARIANTS:
        raise ValueError(f"Unknown encoder arch: {arch}")

    resolved = dict(ENCODER_VARIANTS[arch])
    resolved.update(config)
    resolved["arch"] = arch
    return resolved


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

        self.norm_attn = nn.LayerNorm(self.dim)
        self.norm_ffn = nn.LayerNorm(self.dim)

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

    def forward(self, x, freqs=None, n_registers=0):
        B, N, D = x.shape

        x_pre_attn = x
        x = self.norm_attn(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)

        if freqs is not None:
            q_reg, q_patch = q[:, :, :n_registers], q[:, :, n_registers:]
            k_reg, k_patch = k[:, :, :n_registers], k[:, :, n_registers:]
            q_patch = apply_rotary_emb(freqs, q_patch)
            k_patch = apply_rotary_emb(freqs, k_patch)
            q = torch.cat([q_reg, q_patch], dim=2)
            k = torch.cat([k_reg, k_patch], dim=2)

        with nn.attention.sdpa_kernel(self.sdpa_list):
            attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.attn_out(attn_output)

        x = x_pre_attn + attn_output
        x = x + self.ffn(self.norm_ffn(x))

        return x


class ViT(nn.Module):
    HEAD_DIM = 64

    def __init__(self, encoder_args) -> None:
        super().__init__()

        encoder_args = build_encoder_config(encoder_args)

        self.dim = encoder_args["dim"]
        self.heads = encoder_args["heads"]
        self.depth = encoder_args["depth"]

        self.n_registers = encoder_args["n_registers"]

        self.resolution = encoder_args["resolution"]
        self.patch_size = encoder_args["patch_size"]
        self.grid_size = self.resolution // self.patch_size
        self.n_patches = self.grid_size**2

        # Axial RoPE (DINOv3-style: D_head//4 per axis, tiled 2x, theta=100)
        self.rope_h = RotaryEmbedding(self.HEAD_DIM // 4, theta=100.0)
        self.rope_w = RotaryEmbedding(self.HEAD_DIM // 4, theta=100.0)

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
        self.projector = (
            MLPProjector(self.dim, norm=encoder_args.get("projector_norm", "none"))
            if encoder_args.get("projector", False)
            else nn.Identity()
        )

    def _axial_freqs(self, device):
        g = self.grid_size
        rows = torch.arange(g, device=device).float().repeat_interleave(g)
        cols = torch.arange(g, device=device).float().repeat(g)
        freqs_h = self.rope_h(rows)  # (g*g, HEAD_DIM//4)
        freqs_w = self.rope_w(cols)  # (g*g, HEAD_DIM//4)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # (g*g, HEAD_DIM//2)
        return freqs.tile(2)  # (g*g, HEAD_DIM)

    def forward_features(self, x):
        B, *_ = x.shape

        x = self.patch_embed(x)
        freqs = self._axial_freqs(x.device)

        r = repeat(self.registers, "n d -> b n d", b=B)
        x, ps = pack((r, x), "b * d")

        for layer in self.blocks[:-1]:
            x = layer(x, freqs=freqs, n_registers=self.n_registers)

        r, x = unpack(x, ps, "b * d")

        return {"register": r, "feature_map": x}

    def forward(self, x):
        B, *_ = x.shape

        x = self.patch_embed(x)
        freqs = self._axial_freqs(x.device)

        r = repeat(self.registers, "n d -> b n d", b=B)
        x, ps = pack((r, x), "b * d")

        for layer in self.blocks:
            x = layer(x, freqs=freqs, n_registers=self.n_registers)

        r, x = unpack(x, ps, "b * d")
        r = self.projector(r)
        x = self.projector(x)

        return {"register": r, "feature_map": x}


class ConvNeXtEncoder(nn.Module):
    _BACKBONES = {
        "convnext-t": convnext_tiny,
        "convnext-s": convnext_small,
        "convnext-b": convnext_base,
        "convnext-l": convnext_large,
    }

    def __init__(self, encoder_args) -> None:
        super().__init__()

        self.arch = encoder_args["arch"]
        self.dim = encoder_args["dim"]
        self.backbone_dim = encoder_args["backbone_dim"]

        backbone = self._BACKBONES[self.arch](weights=None)
        self.features = backbone.features
        self.pool = backbone.avgpool

        self.proj = nn.Identity()
        if self.backbone_dim != self.dim:
            self.proj = nn.Linear(self.backbone_dim, self.dim)

        self.projector = (
            MLPProjector(self.dim, norm=encoder_args.get("projector_norm", "none"))
            if encoder_args.get("projector", False)
            else nn.Identity()
        )

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        feature_grid = self.features(x)
        pooled = self.pool(feature_grid).flatten(1)
        pooled = self.proj(pooled)
        token = self.projector(pooled).unsqueeze(1)
        feature_map = rearrange(feature_grid, "b c h w -> b (h w) c")
        feature_map = self.projector(self.proj(feature_map))

        return {"register": token, "feature_map": feature_map}


def build_encoder(encoder_args):
    encoder_args = build_encoder_config(encoder_args)
    arch = encoder_args["arch"]
    if arch.startswith("vit"):
        return ViT(encoder_args)
    if arch.startswith("convnext"):
        return ConvNeXtEncoder(encoder_args)
    raise ValueError(f"Unknown encoder arch: {arch}")
