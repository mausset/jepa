"""Plot a grid of sample trajectories from each toy-env dataset.

Usage:
    python -m jepa.envs.plot_trajectories --data data/toy_envs --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  -- registers Zstd filter for compressed datasets
import matplotlib.pyplot as plt
import numpy as np

N_TRAJ = 8
N_STEPS = 6
SEED = 0

ENVS = ["pointmaze", "keydoor", "push", "pusht", "sokoban", "craftax"]


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

    cell = 1.5
    fig, axes = plt.subplots(
        N_TRAJ, N_STEPS,
        figsize=(N_STEPS * cell, N_TRAJ * cell),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    fig.patch.set_facecolor("black")

    for row, frames in enumerate(trajectories):
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env}_trajectories.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="black", pad_inches=0.02)
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
    out_path = out_dir / f"{env}_episode.png"
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
        plot_full_episode(env, data_dir, out_dir, rng)


if __name__ == "__main__":
    main()
