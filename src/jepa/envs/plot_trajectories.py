"""Plot a grid of sample trajectories from each toy-env dataset.

Usage:
    python -m jepa.envs.plot_trajectories --data data/toy_envs --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa.datasets.toy_env_dataset import ToyEnvAugmentation

N_TRAJ = 8
N_STEPS = 6
SEED = 0

ENVS = ["pointmaze", "keydoor", "push", "pusht", "sokoban", "craftax"]
ENV_LABELS = {
    "pointmaze": "PointMaze",
    "keydoor":   "KeyDoor",
    "push":      "Push",
    "pusht":     "Push-T",
    "sokoban":   "Sokoban",
    "craftax":   "Craftax",
}


def pick_frames(frames: np.ndarray, n: int) -> np.ndarray:
    idx = np.round(np.linspace(0, len(frames) - 1, n)).astype(int)
    return frames[idx]


def plot_env(env: str, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> None:
    path = data_dir / f"{env}_train.h5"
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        chosen = rng.choice(len(keys), size=N_TRAJ, replace=False)
        trajectories = [
            pick_frames(np.asarray(f[keys[i]]["frames"]), N_STEPS)
            for i in chosen
        ]

    fig, axes = plt.subplots(
        N_TRAJ, N_STEPS,
        figsize=(N_STEPS * 1.4, N_TRAJ * 1.4),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )

    for row, frames in enumerate(trajectories):
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for col in range(N_STEPS):
        axes[0, col].set_title(f"$t_{col}$", fontsize=9, pad=4)

    fig.suptitle(ENV_LABELS[env], fontsize=11, y=1.01, fontweight="medium")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env}_trajectories.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


AUG_MODES = [
    ("Original",         dict(photometric="false", crop=False)),
    ("Crop",             dict(photometric="false", crop=True)),
    ("Photometric",      dict(photometric="true",  crop=False)),
    ("Crop + Photo",     dict(photometric="true",  crop=True)),
    ("Per-frame Photo",  dict(photometric="per_frame", crop=False)),
]


def apply_aug(frames_uint8: np.ndarray, aug: ToyEnvAugmentation) -> np.ndarray:
    frames = torch.from_numpy(frames_uint8.astype(np.float32) / 255.0)
    frames = aug(frames)
    return (frames.numpy() * 255).clip(0, 255).astype(np.uint8)


def plot_augmentations(env: str, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> None:
    path = data_dir / f"{env}_train.h5"
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        key = keys[rng.integers(len(keys))]
        frames_uint8 = pick_frames(np.asarray(f[key]["frames"]), N_STEPS)

    n_modes = len(AUG_MODES)
    fig, axes = plt.subplots(
        n_modes, N_STEPS,
        figsize=(N_STEPS * 1.4, n_modes * 1.4),
        gridspec_kw={"wspace": 0.04, "hspace": 0.08},
    )

    for row, (label, kwargs) in enumerate(AUG_MODES):
        aug = ToyEnvAugmentation(**kwargs)
        augmented = apply_aug(frames_uint8, aug)
        for col in range(N_STEPS):
            ax = axes[row, col]
            ax.imshow(augmented[col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(label, fontsize=8, labelpad=4)

    for col in range(N_STEPS):
        axes[0, col].set_title(f"$t_{col}$", fontsize=9, pad=4)

    fig.suptitle(f"{ENV_LABELS[env]} — Augmentations", fontsize=11, y=1.01, fontweight="medium")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env}_augmentations.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_full_episode(env: str, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> None:
    path = data_dir / f"{env}_train.h5"
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        key = keys[rng.integers(len(keys))]
        frames = np.asarray(f[key]["frames"])

    T = len(frames)
    cols = int(np.ceil(np.sqrt(T)))
    rows = int(np.ceil(T / cols))

    frame_h, frame_w = frames[0].shape[:2]
    cell = 1.5
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * cell, rows * cell),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    fig.patch.set_facecolor("black")

    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < T:
            ax.imshow(frames[i])
        else:
            ax.set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env}_episode.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="black", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_envs")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)

    for env in args.envs:
        path = data_dir / f"{env}_train.h5"
        if not path.exists():
            print(f"skipping {env} (no data at {path})")
            continue
        plot_env(env, data_dir, out_dir, rng)
        plot_augmentations(env, data_dir, out_dir, rng)
        plot_full_episode(env, data_dir, out_dir, rng)


if __name__ == "__main__":
    main()
