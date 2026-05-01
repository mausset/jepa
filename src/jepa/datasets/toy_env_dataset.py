from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  -- registers Zstd filter for read
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass(frozen=True)
class SequenceSlice:
    episode: str
    start: int


class ToyEnvGPUIterator:
    """Wraps a DataLoader to move frames to GPU, convert uint8→float, and ImageNet-normalize."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.mean is None:
            self.mean = IMAGENET_MEAN.view(1, 1, 1, 3).to(self.device)
            self.std = IMAGENET_STD.view(1, 1, 1, 3).to(self.device)
        for batch in self.loader:
            batch["data"] = self.process(batch["data"])
            if "actions" in batch:
                batch["actions"] = batch["actions"].to(self.device, non_blocking=True)
            yield batch

    def process(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True).float().div_(255.0)
        return (x - self.mean) / self.std


class ToyEnvSequenceDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        sequence_length: int,
        include_actions: bool,
    ):
        self.path = str(path)
        self.sequence_length = int(sequence_length)
        self.include_actions = bool(include_actions)
        self.file_handle: h5py.File | None = None
        self.index = self.build_index()

    def build_index(self) -> list[SequenceSlice]:
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
        if self.file_handle is None:
            self.file_handle = h5py.File(self.path, "r")
        return self.file_handle

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.index[idx]
        episode = self.file[sample.episode]

        start = sample.start
        end = start + self.sequence_length

        frames = torch.from_numpy(np.asarray(episode["frames"][start:end]))  # uint8
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

    train_dataset = ToyEnvSequenceDataset(
        config["train_path"],
        sequence_length=sequence_length,
        include_actions=include_actions,
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

    loader_kwargs: dict = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 8))

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
        **loader_kwargs,
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
        **loader_kwargs,
    )

    device = torch.device("cuda", local_rank)
    return ToyEnvGPUIterator(train_loader, device), ToyEnvGPUIterator(val_loader, device)
