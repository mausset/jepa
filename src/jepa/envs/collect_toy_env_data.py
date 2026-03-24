from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from jepa.envs.toy_envs import build_toy_env, sanitize_name


def collect_episode(env, max_steps: int, frame_size: tuple[int, int], rng):
    env.reset(rng)

    frames = [env.render(frame_size)]
    states = [env.state_vector()]
    actions = []

    for _ in range(max_steps - 1):
        action = env.sample_action(rng)
        done = env.step(action)

        frames.append(env.render(frame_size))
        states.append(env.state_vector())
        actions.append(action)

        if done:
            break

    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions),
        "episode_length": len(frames),
    }


def collect_split(env_config: dict, split: str, frame_size: tuple[int, int], seed: int):
    env = build_toy_env(env_config)
    num_episodes = int(env_config[f"{split}_episodes"])
    max_steps = int(env_config["max_steps"])

    rng = np.random.default_rng(seed)
    episodes = []
    try:
        for _ in tqdm(range(num_episodes), desc=f"{env_config['kind']}:{split}"):
            episodes.append(
                collect_episode(env, max_steps=max_steps, frame_size=frame_size, rng=rng)
            )
        return env, episodes
    except Exception:
        env.close()
        raise


def save_split(path: Path, env, env_name: str, episodes, frame_size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["env_name"] = env_name
        handle.attrs["env_id"] = getattr(env, "env_id", env_name)
        handle.attrs["action_type"] = env.action_type
        handle.attrs["action_dim"] = env.action_dim
        handle.attrs["frame_height"] = frame_size[0]
        handle.attrs["frame_width"] = frame_size[1]
        handle.attrs["state_dim"] = episodes[0]["states"].shape[-1]

        for index, episode in enumerate(episodes):
            group = handle.create_group(f"{index:06d}")
            group.attrs["episode_length"] = int(episode["episode_length"])
            group.create_dataset("frames", data=episode["frames"])
            group.create_dataset("states", data=episode["states"])
            group.create_dataset("actions", data=episode["actions"])


def main():
    parser = argparse.ArgumentParser(description="Collect offline random toy-environment rollouts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", default=None, help="Collect only a single environment from the config.")
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    frame_size = tuple(config.get("frame_size", [96, 96]))
    output_dir = Path(config.get("output_dir", "data/toy_envs"))
    base_seed = int(config.get("seed", 0))

    envs = config["envs"]
    if args.env is not None:
        envs = {args.env: config["envs"][args.env]}

    for env_index, (env_name, env_config) in enumerate(envs.items()):
        train_seed = base_seed + env_index * 2
        val_seed = base_seed + env_index * 2 + 1

        env, train_episodes = collect_split(env_config, "train", frame_size, train_seed)
        try:
            _, val_episodes = collect_split(env_config, "val", frame_size, val_seed)

            env_path = sanitize_name(env_name)
            save_split(
                output_dir / f"{env_path}_train.h5",
                env,
                env_name,
                train_episodes,
                frame_size,
            )
            save_split(
                output_dir / f"{env_path}_val.h5",
                env,
                env_name,
                val_episodes,
                frame_size,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
