from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as tvf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 1, 1, 3
)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
    1, 1, 1, 3
)


@dataclass(frozen=True)
class SequenceSlice:
    episode: str
    start: int


def _batched_adjust_hue(frames: torch.Tensor, hue_factors: torch.Tensor) -> torch.Tensor:
    """Per-image hue shift. frames: (N, 3, H, W) float [0,1], hue_factors: (N,) in [-0.5, 0.5]."""
    # RGB → HSV
    maxc, argmax = frames.max(dim=1)
    minc = frames.min(dim=1).values
    cr = maxc - minc
    r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]

    h = torch.zeros_like(maxc)
    nz = cr > 0
    m0 = nz & (argmax == 0)
    m1 = nz & (argmax == 1)
    m2 = nz & (argmax == 2)
    h[m0] = ((g[m0] - b[m0]) / cr[m0]) % 6.0
    h[m1] = (b[m1] - r[m1]) / cr[m1] + 2.0
    h[m2] = (r[m2] - g[m2]) / cr[m2] + 4.0
    h = h / 6.0

    s = torch.where(maxc > 0, cr / maxc, torch.zeros_like(cr))
    v = maxc

    # shift hue per image
    h = (h + hue_factors[:, None, None]) % 1.0

    # HSV → RGB via sector lookup
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - (hp % 2.0 - 1.0).abs())
    m = v - c
    hi = hp.long() % 6

    zeros = torch.zeros_like(c)
    r_lut = torch.stack([c, x, zeros, zeros, x, c])  # (6, N, H, W)
    g_lut = torch.stack([x, c, c, x, zeros, zeros])
    b_lut = torch.stack([zeros, zeros, x, c, c, x])
    idx = hi.unsqueeze(0)                              # (1, N, H, W)
    return torch.stack([
        r_lut.gather(0, idx).squeeze(0) + m,
        g_lut.gather(0, idx).squeeze(0) + m,
        b_lut.gather(0, idx).squeeze(0) + m,
    ], dim=1)


class ToyEnvAugmentation:
    def __init__(self, photometric: str = "false", crop: bool = False):
        from torchvision.transforms import v2

        self.photometric = str(photometric).lower()
        self.crop = bool(crop)
        if self.photometric not in {"false", "true", "per_frame"}:
            raise ValueError(f"Unknown photometric augmentation mode: {photometric}")

        self.scale = (0.9, 1.0)
        self.ratio = (0.95, 1.05)
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.2
        self.hue = 0.1

        self._color_jitter = v2.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.crop and self.photometric == "false":
            return frames

        frames = frames.permute(0, 3, 1, 2).contiguous()
        if self.crop:
            frames = self._random_resized_crop(frames)
        if self.photometric == "true":
            # v2.ColorJitter applies same params to all frames — consistent jitter
            frames = self._color_jitter(frames)
        elif self.photometric == "per_frame":
            frames = self._photometric_jitter_per_frame(frames)
        return frames.permute(0, 2, 3, 1).contiguous()

    def _random_resized_crop(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        top, left, crop_h, crop_w = RandomResizedCrop.get_params(
            frames[0],
            scale=self.scale,
            ratio=self.ratio,
        )
        return tvf.resized_crop(
            frames,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _photometric_jitter_per_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """Vectorized per-frame photometric jitter. frames: (T, C, H, W)."""
        T = frames.shape[0]
        dev = frames.device

        # per-frame hue via batched RGB→HSV→RGB
        hues = (torch.rand(T, device=dev) * 2.0 - 1.0) * self.hue
        frames = _batched_adjust_hue(frames, hues)

        # per-frame brightness / contrast / saturation as (T, 1, 1, 1)
        shape = (T, 1, 1, 1)
        brightness = 1.0 + (torch.rand(shape, device=dev) * 2.0 - 1.0) * self.brightness
        contrast = 1.0 + (torch.rand(shape, device=dev) * 2.0 - 1.0) * self.contrast
        saturation = 1.0 + (torch.rand(shape, device=dev) * 2.0 - 1.0) * self.saturation

        frames = frames * brightness

        mean = frames.mean(dim=(1, 2, 3), keepdim=True)
        frames = (frames - mean) * contrast + mean

        gray = frames.mean(dim=1, keepdim=True)
        frames = gray + saturation * (frames - gray)

        return frames.clamp_(0.0, 1.0)

class ToyEnvSequenceDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        sequence_length: int,
        include_actions: bool,
        photometric: str = "false",
        crop: bool = False,
    ):
        self.path = str(path)
        self.sequence_length = int(sequence_length)
        self.include_actions = bool(include_actions)
        self.augmentation = ToyEnvAugmentation(photometric, crop=crop)
        self._file: h5py.File | None = None
        self._index = self._build_index()

    def _build_index(self) -> list[SequenceSlice]:
        index: list[SequenceSlice] = []
        with h5py.File(self.path, "r") as handle:
            for episode in handle.keys():
                episode_length = int(handle[episode].attrs["episode_length"])
                if episode_length < self.sequence_length:
                    continue

                n_slices = episode_length - self.sequence_length + 1
                index.extend(
                    SequenceSlice(episode=episode, start=start)
                    for start in range(n_slices)
                )
        return index

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._index[idx]
        episode = self.file[sample.episode]

        start = sample.start
        end = start + self.sequence_length

        frames = np.asarray(episode["frames"][start:end], dtype=np.float32) / 255.0
        frames = torch.from_numpy(frames)
        frames = self.augmentation(frames)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        result: dict[str, torch.Tensor] = {"data": frames}
        if self.include_actions:
            actions = np.asarray(episode["actions"][start : end - 1])
            result["actions"] = torch.from_numpy(actions)

        return result


def build_toy_env_iterators(
    config: dict,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    seed: int = -1,
):
    include_actions = bool(config.get("include_actions", False))
    sequence_length = int(config["sequence_length"])
    photometric = str(config.get("photometric", "false"))
    crop = bool(config.get("crop", False))

    train_dataset = ToyEnvSequenceDataset(
        config["train_path"],
        sequence_length=sequence_length,
        include_actions=include_actions,
        photometric=photometric,
        crop=crop,
    )
    val_dataset = ToyEnvSequenceDataset(
        config["val_path"],
        sequence_length=sequence_length,
        include_actions=include_actions,
    )

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=False,
        )

    generator = torch.Generator()
    if seed >= 0:
        generator.manual_seed(seed)

    num_workers = int(config.get("num_workers", config.get("num_threads", 4)))
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=val_sampler is None,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        generator=generator,
    )

    return train_loader, val_loader
