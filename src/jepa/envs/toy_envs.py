from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def sanitize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace("/", "_")


@dataclass
class ToyEnv:
    action_type: str
    action_dim: int

    def reset(self, rng: np.random.Generator):
        raise NotImplementedError

    def sample_action(self, rng: np.random.Generator):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, frame_size: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

    def state_vector(self) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        return None


class GymToyEnv(ToyEnv):
    def __init__(self, config: dict):
        try:
            import gymnasium as gym
            import gymnasium_robotics  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Gymnasium Robotics is required for pointmaze and push toy environments."
            ) from exc

        self.frame_size = tuple(config.get("frame_size", (96, 96)))
        self.env_id = config["env_id"]
        self.action_repeat = int(config.get("action_repeat", 1))
        make_kwargs = {"render_mode": "rgb_array", "max_episode_steps": -1}
        make_kwargs.update(config.get("make_kwargs", {}))

        self.env = gym.make(self.env_id, **make_kwargs)
        self.last_obs = None
        self._configure_camera(config)

        if hasattr(self.env.action_space, "n"):
            action_type = "discrete"
            action_dim = int(self.env.action_space.n)
        else:
            action_type = "continuous"
            action_dim = int(np.prod(self.env.action_space.shape))

        super().__init__(action_type=action_type, action_dim=action_dim)

    def _configure_camera(self, config: dict):
        camera = config.get("camera")
        if camera is None:
            return

        u = self.env.unwrapped
        if hasattr(u, "point_env"):
            renderer = u.point_env.mujoco_renderer
        else:
            renderer = u.mujoco_renderer
        renderer.default_cam_config = camera
        renderer._viewers = {}

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            if "observation" in obs:
                return np.asarray(obs["observation"], dtype=np.float32)
            return np.concatenate(
                [np.asarray(value, dtype=np.float32).reshape(-1) for value in obs.values()],
                axis=0,
            )
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def reset(self, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        self.last_obs, _ = self.env.reset(seed=seed)

    def sample_action(self, rng: np.random.Generator):
        if self.action_type == "discrete":
            return np.int64(rng.integers(self.action_dim))

        low = np.asarray(self.env.action_space.low, dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        return rng.uniform(low, high).astype(np.float32)

    def step(self, action):
        done = False
        for _ in range(self.action_repeat):
            self.last_obs, _, terminated, truncated, _ = self.env.step(action)
            done = done or bool(terminated or truncated)
            if done:
                break
        return done

    def render(self, frame_size: tuple[int, int]) -> np.ndarray:
        frame = self.env.render()
        if frame.shape[:2] == frame_size:
            return np.asarray(frame, dtype=np.uint8)

        height, width = frame_size
        ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
        return np.asarray(frame[np.ix_(ys, xs)], dtype=np.uint8)

    def state_vector(self) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before reading state.")
        return self._flatten_obs(self.last_obs)

    def close(self):
        self.env.close()


class PointMazeEnv(GymToyEnv):
    def __init__(self, config: dict):
        merged = dict(config)
        merged.setdefault("env_id", "PointMaze_UMaze-v3")
        merged.setdefault(
            "camera",
            {
                "lookat": (0.0, 0.0, 0.0),
                "distance": 4.5,
                "elevation": -90,
                "azimuth": 180,
                "trackbodyid": -1,
            },
        )
        merged.setdefault(
            "make_kwargs",
            {
                "width": merged.get("frame_size", (96, 96))[1],
                "height": merged.get("frame_size", (96, 96))[0],
            },
        )
        super().__init__(merged)


class PushEnv(GymToyEnv):
    def __init__(self, config: dict):
        merged = dict(config)
        merged.setdefault("env_id", "FetchPush-v3")
        merged.setdefault(
            "camera",
            {
                "lookat": [1.3, 0.75, 0.5],
                "distance": 1.6,
                "elevation": -35,
                "azimuth": 180,
            },
        )
        super().__init__(merged)


class KeyDoorEnv(ToyEnv):
    def __init__(self, config: dict):
        try:
            import gymnasium as gym
            from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
        except ImportError as exc:
            raise ImportError(
                "Minigrid is required for the keydoor toy environment."
            ) from exc

        self.frame_size = tuple(config.get("frame_size", (96, 96)))
        self.env_id = config.get("env_id", "MiniGrid-DoorKey-16x16-v0")
        self.tile_size = int(config.get("tile_size", 8))

        env = gym.make(self.env_id, render_mode="rgb_array")
        env = RGBImgObsWrapper(env, tile_size=self.tile_size)
        env = ImgObsWrapper(env)
        self.env = env
        self.last_obs = None

        super().__init__(action_type="discrete", action_dim=int(self.env.action_space.n))

    def reset(self, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        self.last_obs, _ = self.env.reset(seed=seed)

    def sample_action(self, rng: np.random.Generator):
        return np.int64(rng.integers(self.action_dim))

    def step(self, action):
        self.last_obs, _, terminated, truncated, _ = self.env.step(int(action))
        return bool(terminated or truncated)

    def render(self, frame_size: tuple[int, int]) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before rendering.")

        frame = np.asarray(self.last_obs, dtype=np.uint8)
        if frame.shape[:2] == frame_size:
            return frame

        height, width = frame_size
        ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
        return np.asarray(frame[np.ix_(ys, xs)], dtype=np.uint8)

    def state_vector(self) -> np.ndarray:
        env = self.env.unwrapped
        key_pos = np.array([-1, -1], dtype=np.float32)
        door_pos = np.array([-1, -1], dtype=np.float32)
        door_open = 0.0
        door_locked = 0.0
        goal_pos = np.array([-1, -1], dtype=np.float32)

        for row in range(env.height):
            for col in range(env.width):
                cell = env.grid.get(col, row)
                if cell is None:
                    continue
                if cell.type == "key":
                    key_pos = np.array([col, row], dtype=np.float32)
                elif cell.type == "door":
                    door_pos = np.array([col, row], dtype=np.float32)
                    door_open = float(cell.is_open)
                    door_locked = float(cell.is_locked)
                elif cell.type == "goal":
                    goal_pos = np.array([col, row], dtype=np.float32)

        carrying_key = float(
            env.carrying is not None and getattr(env.carrying, "type", None) == "key"
        )
        return np.array(
            [
                float(env.agent_pos[0]),
                float(env.agent_pos[1]),
                float(env.agent_dir),
                carrying_key,
                key_pos[0],
                key_pos[1],
                door_pos[0],
                door_pos[1],
                door_open,
                door_locked,
                goal_pos[0],
                goal_pos[1],
            ],
            dtype=np.float32,
        )

    def close(self):
        self.env.close()


class SokobanLiteEnv(ToyEnv):
    def __init__(self, config: dict):
        try:
            import gym
            import gym_sokoban  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "gym-sokoban is required for the sokoban toy environment."
            ) from exc

        self.frame_size = tuple(config.get("frame_size", (96, 96)))
        self.env_id = config.get("env_id", "Sokoban-small-v0")
        self.render_mode = config.get("render_mode", "rgb_array")
        self.env = gym.make(self.env_id, disable_env_checker=True)
        self.last_obs = None

        super().__init__(action_type="discrete", action_dim=int(self.env.action_space.n))

    def reset(self, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            self.last_obs = self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            self.last_obs = self.env.reset()
        if isinstance(self.last_obs, tuple):
            self.last_obs = self.last_obs[0]

    def sample_action(self, rng: np.random.Generator):
        return np.int64(rng.integers(self.action_dim))

    def step(self, action):
        step_out = self.env.step(int(action))
        if len(step_out) == 5:
            self.last_obs, _, terminated, truncated, _ = step_out
            return bool(terminated or truncated)

        self.last_obs, _, done, _ = step_out
        return bool(done)

    def render(self, frame_size: tuple[int, int]) -> np.ndarray:
        frame = np.asarray(self.env.render(mode=self.render_mode), dtype=np.uint8)
        if frame.shape[:2] == frame_size:
            return frame

        height, width = frame_size
        ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
        return np.asarray(frame[np.ix_(ys, xs)], dtype=np.uint8)

    def state_vector(self) -> np.ndarray:
        if hasattr(self.env, "player_position") and hasattr(self.env, "room_state"):
            player = np.asarray(self.env.player_position, dtype=np.float32).reshape(-1)
            room = np.asarray(self.env.room_state, dtype=np.float32).reshape(-1)
            return np.concatenate((player, room), axis=0)

        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before reading state.")
        return np.asarray(self.last_obs, dtype=np.float32).reshape(-1)

    def close(self):
        self.env.close()


class CraftaxEnv(ToyEnv):
    def __init__(self, config: dict):
        try:
            import jax
            import jax.numpy as jnp
            from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
        except ImportError as exc:
            raise ImportError("Craftax and JAX are required for the craftax toy environment.") from exc

        self._jax = jax
        self._jnp = jnp

        self.frame_size = tuple(config.get("frame_size", (96, 96)))
        env = CraftaxClassicPixelsEnv()
        self._env = env
        self._params = env.default_params
        self._reset_fn = jax.jit(env.reset)
        self._step_fn = jax.jit(env.step)

        self._key = None
        self._state = None
        self._last_obs = None

        super().__init__(action_type="discrete", action_dim=env.num_actions)

    def reset(self, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        self._key = self._jax.random.PRNGKey(seed)
        self._key, subkey = self._jax.random.split(self._key)
        self._last_obs, self._state = self._reset_fn(subkey, self._params)

    def sample_action(self, rng: np.random.Generator):
        return np.int64(rng.integers(self.action_dim))

    def step(self, action):
        self._key, subkey = self._jax.random.split(self._key)
        act = self._jnp.int32(int(action))
        self._last_obs, self._state, _, done, _ = self._step_fn(
            subkey, self._state, act, self._params
        )
        return bool(done)

    def render(self, frame_size: tuple[int, int]) -> np.ndarray:
        if self._last_obs is None:
            raise RuntimeError("Environment must be reset before rendering.")
        frame = np.asarray(self._last_obs * 255, dtype=np.uint8)
        if frame.shape[:2] == frame_size:
            return frame
        height, width = frame_size
        ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
        return np.asarray(frame[np.ix_(ys, xs)], dtype=np.uint8)

    def state_vector(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment must be reset before reading state.")
        s = self._state
        pos = np.asarray(s.player_position, dtype=np.float32).reshape(-1)
        stats = np.asarray(
            [s.player_health, s.player_food, s.player_drink, s.player_energy],
            dtype=np.float32,
        )
        inv_leaves = self._jax.tree_util.tree_leaves(s.inventory)
        inv = np.concatenate([np.asarray(l, dtype=np.float32).reshape(-1) for l in inv_leaves])
        return np.concatenate([pos, stats, inv])


def build_toy_env(config: dict) -> ToyEnv:
    kind = config["kind"]
    if kind == "pointmaze":
        return PointMazeEnv(config)
    if kind == "push":
        return PushEnv(config)
    if kind == "keydoor":
        return KeyDoorEnv(config)
    if kind == "sokoban":
        return SokobanLiteEnv(config)
    if kind == "craftax":
        return CraftaxEnv(config)

    raise ValueError(f"Unknown toy environment kind: {kind}")
