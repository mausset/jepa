import argparse
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import ale_py
import cv2
import gymnasium as gym
import mujoco
import gymnasium_robotics
import h5py
import numpy as np
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt

import imageio.v3 as iio


def resize_frame(frame, size=(224, 224)):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def save_gif(frames, path, fps: int = 10):
    processed = []
    for f in frames:
        processed.append(f)

    iio.imwrite(path, processed, format="GIF", duration=1 / fps)


def collect_episode(env, env_config, frame_size=(224, 224)):
    env_hz = env_config["env_hz"]
    capture_hz = env_config["capture_hz"]
    max_steps = env_config["max_steps"]
    include_prop = env_config.get("include_prop", False)

    assert (
        env_hz % capture_hz == 0
    ), f"Env hz ({env_hz}hz) not divisible by capture hz ({capture_hz}hz)"

    steps_per_capture = env_hz // capture_hz

    frames = []
    actions = []
    prop_data = []

    env.reset()
    step_count = 0
    current_action = None

    while step_count < max_steps:
        frame = env.render()
        frames.append(resize_frame(frame, frame_size))
        current_action = env.action_space.sample()
        actions.append(current_action)

        for _ in range(steps_per_capture):
            obs, _, terminated, truncated, _ = env.step(current_action)

        if terminated or truncated:  # type: ignore
            break

        if include_prop:
            prop_data.append(obs["observation"])

        step_count += 1

    actions.pop(-1)

    # iio.imwrite("plots/animation.gif", frames, format="GIF", duration=1 / capture_hz)
    # exit()

    result = {
        "frames": np.array(frames),
        "actions": np.array(actions),
        "episode_length": len(actions),
        "env_steps": step_count,
        "capture_frequency": capture_hz,
        "env_frequency": env_hz,
    }

    if include_prop:
        result["prop"] = np.array(prop_data)

    return result


def collect_paths(env_name, env_config, frame_size=(224, 224)):

    TOP_DOWN = dict(
        lookat=(0.0, 0.0, 0.0),  # centre of the maze
        distance=7.0,  # height above the floor
        elevation=-90,  # -90° = straight down
        azimuth=180,  # 0/90/180/270 pick the maze orientation you want
        trackbodyid=-1,  # -1 = don’t follow any body → stays static
    )

    if env_name == "PointMaze_UMaze-v3":
        env = gym.make(
            env_name,
            render_mode="rgb_array",
            height=frame_size[0],
            width=frame_size[1],
        )

        renderer = env.unwrapped.point_env.mujoco_renderer
        renderer.default_cam_config = TOP_DOWN
        renderer._viewers = {}  # forces a new viewer to be created with the new config
    else:
        env = gym.make(
            env_name,
            render_mode="rgb_array",
        )

    num_episodes = env_config["num_episodes"]

    episodes = []
    for _ in tqdm(range(num_episodes)):
        episode = collect_episode(env, env_config, frame_size)
        episodes.append(episode)

    env.close()
    return episodes


def save_paths_hdf5(episodes, output_dir, env_name, env_config, val=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_clean = env_name.lower().replace("-", "_").replace("/", "_")

    if val:
        path = output_dir / f"{env_clean}_val.h5"
    else:
        path = output_dir / f"{env_clean}.h5"

    with h5py.File(path, mode="w") as f:
        f.attrs["env_name"] = env_name
        f.attrs["env_hz"] = env_config["env_hz"]
        f.attrs["capture_hz"] = env_config["capture_hz"]
        for ep_idx, episode in enumerate(episodes):

            ep = f.create_group(f"{ep_idx:06d}")
            ep.attrs["episode_length"] = episode["frames"].shape[0]

            ep.create_dataset("frames", data=episode["frames"])
            ep.create_dataset("actions", data=episode["actions"])
            if "prop" in episode:
                ep.create_dataset("prop", data=episode["prop"])

    print(f"Saved {len(episodes)} episodes to h5 format in {output_dir}")

    episode_lengths = [ep["episode_length"] for ep in episodes]
    print(
        f"Episode length stats: mean={np.mean(episode_lengths):.1f}, "
        f"std={np.std(episode_lengths):.1f}, "
        f"min={np.min(episode_lengths)}, max={np.max(episode_lengths)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate random paths for environments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/envs/trajectory_config.yaml",
        help="Path to trajectory configuration YAML file",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Specific environment to generate (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    frame_size = tuple(config.get("frame_size", [224, 224]))
    output_dir = config.get("output_dir", "data/env_paths")

    for env_name, env_config in config["envs"].items():
        print(
            f"\nCollecting {env_config['num_episodes']} training episodes for {env_name}"
        )
        print(f"Env Hz: {env_config['env_hz']}, Capture Hz: {env_config['capture_hz']}")

        episodes = collect_paths(env_name, env_config, frame_size)
        save_paths_hdf5(episodes, output_dir, env_name, env_config)

        print(
            f"\nCollecting {env_config['num_episodes']} validation episodes for {env_name}"
        )
        episodes = collect_paths(env_name, env_config, frame_size)
        save_paths_hdf5(episodes, output_dir, env_name, env_config, val=True)


if __name__ == "__main__":
    main()
