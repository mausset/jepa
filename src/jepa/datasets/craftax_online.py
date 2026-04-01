"""
Online Craftax data iterator.

Episodes are generated on-the-fly via JAX vmap + lax.scan.  The scan runs
for exactly ``sequence_length - 1`` steps, so the output is directly
``(batch_size, sequence_length, H, W, 3)``.  Frames are resized on-device
and transferred to PyTorch via DLPack (zero-copy GPU→GPU).

Optionally, a buffer can be enabled (``buffer_size > 0``) to amortise
JAX call overhead: one JAX call collects ``buffer_size`` sequences,
which are then served across multiple ``__next__`` calls.

Constraints
-----------
* Use num_workers=0 — JAX is not fork-safe.
* JAX and PyTorch must share the same CUDA device.
"""
from __future__ import annotations

import functools
import os
from typing import Iterator

# Prevent JAX from preallocating most of the GPU — PyTorch needs it too.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch

from jepa.datasets.toy_env_dataset import IMAGENET_MEAN, IMAGENET_STD, ToyEnvAugmentation


class CraftaxOnlineBatchIterator:
    """Infinite iterator that yields JEPA-compatible batches from Craftax online.

    Each ``__next__`` runs one JIT-compiled JAX call that resets
    ``batch_size`` environments, steps them ``sequence_length - 1`` times,
    resizes on-device, and returns uint8 frames + int32 actions via DLPack.

    When ``buffer_size > 0``, a single JAX call collects ``buffer_size``
    sequences and subsequent ``__next__`` calls index into the GPU buffer
    until it is exhausted, then a fresh JAX call refills it.
    """

    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        frame_size: tuple[int, int] = (96, 96),
        include_actions: bool = True,
        photometric: str = "false",
        crop: bool = False,
        seed: int = 0,
        buffer_size: int = 0,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.include_actions = include_actions
        self.augmentation = ToyEnvAugmentation(photometric, crop=crop)
        self._jax = jax

        if buffer_size > 0 and buffer_size < batch_size:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be >= batch_size ({batch_size})"
            )
        self._buffer_size = buffer_size
        self._buf_frames: torch.Tensor | None = None
        self._buf_actions: torch.Tensor | None = None
        self._buf_pos: int = 0

        env = CraftaxClassicPixelsEnv()
        params = env.default_params
        num_actions = env.num_actions
        self.num_actions = num_actions

        v_reset = jax.vmap(env.reset, in_axes=(0, None))
        v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        target_h, target_w = frame_size

        @functools.partial(jax.jit, static_argnums=(1,))
        def _collect(key: jax.Array, n: int):
            key, rkey = jax.random.split(key)
            reset_keys = jax.random.split(rkey, n)
            obs, states = v_reset(reset_keys, params)

            def _scan_step(carry, _):
                key, states = carry
                key, skey, akey = jax.random.split(key, 3)
                acts = jax.random.randint(akey, (n,), 0, num_actions)
                step_keys = jax.random.split(skey, n)
                new_obs, new_states, _, _, _ = v_step(step_keys, states, acts, params)
                return (key, new_states), (new_obs, acts)

            _, (all_obs, all_acts) = jax.lax.scan(
                _scan_step, (key, states), None, length=sequence_length - 1
            )

            # (T-1, N, ...) → (N, T, ...)
            frames = jnp.concatenate([obs[None], all_obs], axis=0)  # (T, N, H, W, 3)
            frames = jnp.moveaxis(frames, 0, 1)                      # (N, T, H, W, 3)
            all_acts = jnp.moveaxis(all_acts, 0, 1)                  # (N, T-1)

            # resize on-device
            native_h, native_w = frames.shape[2:4]
            if native_h != target_h or native_w != target_w:
                N_b, T_b, H, W, C = frames.shape
                flat = frames.reshape(N_b * T_b, H, W, C)
                flat = jax.image.resize(
                    flat, (N_b * T_b, target_h, target_w, C), method="bilinear"
                )
                frames = flat.reshape(N_b, T_b, target_h, target_w, C)

            frames = (frames * 255).astype(jnp.uint8)
            return frames, all_acts

        self._collect = _collect
        self._key = jax.random.PRNGKey(seed)

        # normalisation tensors moved to device on first use
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _collect_batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run JAX to collect *n* sequences and return GPU tensors via DLPack."""
        self._key, subkey = self._jax.random.split(self._key)
        frames_jax, acts_jax = self._collect(subkey, n)
        return torch.from_dlpack(frames_jax), torch.from_dlpack(acts_jax)

    def _refill(self) -> None:
        """Fill the buffer with a single large JAX call."""
        self._buf_frames, self._buf_actions = self._collect_batch(self._buffer_size)
        self._buf_pos = 0

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._buffer_size <= 0:
            # unbuffered: one JAX call per batch
            frames, actions = self._collect_batch(self.batch_size)
        else:
            # buffered: index into pre-collected sequences
            if self._buf_frames is None or self._buf_pos + self.batch_size > self._buffer_size:
                self._refill()
            end = self._buf_pos + self.batch_size
            frames = self._buf_frames[self._buf_pos:end]
            actions = self._buf_actions[self._buf_pos:end]
            self._buf_pos = end

        frames = frames.float() / 255.0
        frames = torch.stack([self.augmentation(f) for f in frames])

        if self._mean is None:
            dev = frames.device
            self._mean = IMAGENET_MEAN.to(dev)
            self._std = IMAGENET_STD.to(dev)
        frames = (frames - self._mean) / self._std

        result: dict[str, torch.Tensor] = {"data": frames}
        if self.include_actions:
            result["actions"] = actions
        return result

    def __len__(self) -> int:
        return 2**31

    def reset(self) -> None:
        pass  # online — data is generated continuously


# ---------------------------------------------------------------------------

def build_craftax_online_iterators(
    config: dict,
    seed: int = 0,
) -> tuple[CraftaxOnlineBatchIterator, CraftaxOnlineBatchIterator]:
    seq_len = int(config["sequence_length"])
    batch_size = int(config["batch_size"])
    frame_size = tuple(config.get("frame_size", [96, 96]))
    include_actions = bool(config.get("include_actions", True))
    photometric = str(config.get("photometric", "false"))
    crop = bool(config.get("crop", False))
    buffer_size = int(config.get("buffer_size", 0))

    train_iter = CraftaxOnlineBatchIterator(
        sequence_length=seq_len,
        batch_size=batch_size,
        frame_size=frame_size,
        include_actions=include_actions,
        photometric=photometric,
        crop=crop,
        seed=seed,
        buffer_size=buffer_size,
    )
    val_iter = CraftaxOnlineBatchIterator(
        sequence_length=seq_len,
        batch_size=batch_size,
        frame_size=frame_size,
        include_actions=include_actions,
        seed=seed + 1,
        buffer_size=buffer_size,
    )
    return train_iter, val_iter
