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

N_TRAJ = 4
N_STEPS = 6
SEED = 0

ENVS = ["pointmaze", "keydoor", "push", "sokoban", "craftax"]
ENV_LABELS = {
    "pointmaze": "PointMaze",
    "keydoor":   "KeyDoor",
    "push":      "Push",
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


if __name__ == "__main__":
    main()
