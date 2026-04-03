from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from jepa.envs.toy_envs import build_toy_env, sanitize_name

PARALLEL_ENVS = {"pointmaze", "push", "pusht", "keydoor", "sokoban"}
JAX_ENVS = {"craftax"}


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


def _collect_chunk(args):
    """Worker function: collect a chunk of episodes in a single env instance."""
    env_config, n_episodes, max_steps, frame_size, seed = args
    env = build_toy_env(env_config)
    rng = np.random.default_rng(seed)
    episodes = []
    try:
        for _ in range(n_episodes):
            episodes.append(collect_episode(env, max_steps=max_steps, frame_size=frame_size, rng=rng))
    finally:
        env.close()
    return episodes


def _collect_craftax_batch(num_episodes, max_steps, frame_size, seed, batch_size=1024):
    """Collect Craftax episodes using vectorized JAX env stepping."""
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv

    env = CraftaxClassicPixelsEnv()
    params = env.default_params
    num_actions = env.num_actions

    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def _state_vec(state):
        pos = state.player_position.astype(jnp.float32)
        stats = jnp.array([
            state.player_health, state.player_food,
            state.player_drink, state.player_energy,
        ], dtype=jnp.float32)
        inv = jnp.array(jax.tree_util.tree_leaves(state.inventory), dtype=jnp.float32)
        return jnp.concatenate([pos, stats, inv])

    v_state_vec = jax.vmap(_state_vec)

    @functools.partial(jax.jit, static_argnums=(1,))
    def _collect_batch(key, n):
        key, rkey = jax.random.split(key)
        reset_keys = jax.random.split(rkey, n)
        obs, states = v_reset(reset_keys, params)

        def _scan_step(carry, _):
            key, states, done_mask = carry
            key, skey, akey = jax.random.split(key, 3)
            actions = jax.random.randint(akey, (n,), 0, num_actions)
            step_keys = jax.random.split(skey, n)
            new_obs, new_states, _, step_done, _ = v_step(step_keys, states, actions, params)
            new_done_mask = done_mask | step_done
            # keep old state if already done
            states = jax.tree.map(
                lambda old, new: jnp.where(
                    jnp.expand_dims(done_mask, tuple(range(1, new.ndim))), old, new
                ),
                states, new_states,
            )
            new_obs = jnp.where(
                jnp.expand_dims(done_mask, tuple(range(1, new_obs.ndim))), jnp.zeros_like(new_obs), new_obs
            )
            sv = v_state_vec(states)
            return (key, states, new_done_mask), (new_obs, actions, sv, step_done)

        init_sv = v_state_vec(states)
        init_carry = (key, states, jnp.zeros(n, dtype=bool))
        _, (all_obs, all_actions, all_svs, all_dones) = jax.lax.scan(
            _scan_step, init_carry, None, length=max_steps - 1
        )

        # all_obs: (T-1, N, H, W, 3), obs: (N, H, W, 3)
        frames = jnp.concatenate([obs[None], all_obs], axis=0)  # (T, N, H, W, 3)
        states_out = jnp.concatenate([init_sv[None], all_svs], axis=0)  # (T, N, state_dim)
        # frames: (T, N, ...) -> (N, T, ...)
        frames = jnp.moveaxis(frames, 0, 1)
        states_out = jnp.moveaxis(states_out, 0, 1)
        all_actions = jnp.moveaxis(all_actions, 0, 1)  # (N, T-1)
        all_dones = jnp.moveaxis(all_dones, 0, 1)  # (N, T-1)

        # compute episode lengths: first done + 1, or max_steps if no done
        any_done = all_dones.any(axis=1)
        first_done = jnp.argmax(all_dones.astype(jnp.int32), axis=1) + 2  # +1 for initial frame, +1 for 0-index
        ep_lengths = jnp.where(any_done, first_done, max_steps)

        return frames, states_out, all_actions, ep_lengths

    print(f"Craftax collection using JAX backend: {jax.default_backend()}")

    all_episodes = []
    remaining = num_episodes
    key = jax.random.PRNGKey(seed)

    pbar_outer = tqdm(total=num_episodes, desc="craftax (jax batched)")
    while remaining > 0:
        n = min(remaining, batch_size)
        key, subkey = jax.random.split(key)
        frames, states_out, actions, ep_lengths = _collect_batch(subkey, n)

        # transfer to CPU as numpy
        frames_np = np.asarray(frames)
        states_np = np.asarray(states_out)
        actions_np = np.asarray(actions)
        lengths_np = np.asarray(ep_lengths)

        for i in range(n):
            L = int(lengths_np[i])
            f = (frames_np[i, :L] * 255).astype(np.uint8)
            # resize if needed
            if f.shape[1:3] != frame_size:
                h, w = frame_size
                ys = np.linspace(0, f.shape[1] - 1, h).astype(np.int64)
                xs = np.linspace(0, f.shape[2] - 1, w).astype(np.int64)
                f = f[:, ys][:, :, xs]
            all_episodes.append({
                "frames": f,
                "states": states_np[i, :L],
                "actions": actions_np[i, :L - 1],
                "episode_length": L,
            })

        remaining -= n
        pbar_outer.update(n)

    pbar_outer.close()
    return all_episodes, num_actions


def _write_episodes(handle: h5py.File, episodes: list, offset: int):
    for i, episode in enumerate(episodes):
        group = handle.create_group(f"{offset + i:06d}")
        group.attrs["episode_length"] = int(episode["episode_length"])
        group.create_dataset("frames", data=episode["frames"])
        group.create_dataset("states", data=episode["states"])
        group.create_dataset("actions", data=episode["actions"])


def collect_and_save_split(
    env_config: dict,
    split: str,
    frame_size: tuple[int, int],
    seed: int,
    output_path: Path,
    env_name: str,
    num_workers: int = 1,
):
    env_kind = env_config["kind"]
    num_episodes = int(env_config[f"{split}_episodes"])
    max_steps = int(env_config["max_steps"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if env_kind in JAX_ENVS:
        episodes, num_actions = _collect_craftax_batch(
            num_episodes, max_steps, frame_size, seed,
        )
        state_dim = episodes[0]["states"].shape[-1]
        with h5py.File(output_path, "w") as handle:
            handle.attrs["env_name"] = env_name
            handle.attrs["env_id"] = env_name
            handle.attrs["action_type"] = "discrete"
            handle.attrs["action_dim"] = int(num_actions)
            handle.attrs["frame_height"] = frame_size[0]
            handle.attrs["frame_width"] = frame_size[1]
            handle.attrs["state_dim"] = state_dim
            _write_episodes(handle, episodes, offset=0)
        return None

    # Probe env for metadata
    probe_env = build_toy_env(env_config)
    probe_env.reset(np.random.default_rng(0))
    state_dim = probe_env.state_vector().shape[-1]

    use_pool = env_kind in PARALLEL_ENVS and num_workers > 1

    probe_env.close()
    with h5py.File(output_path, "w") as handle:
        handle.attrs["env_name"] = env_name
        handle.attrs["env_id"] = getattr(probe_env, "env_id", env_name)
        handle.attrs["action_type"] = probe_env.action_type
        handle.attrs["action_dim"] = probe_env.action_dim
        handle.attrs["frame_height"] = frame_size[0]
        handle.attrs["frame_width"] = frame_size[1]
        handle.attrs["state_dim"] = state_dim

        written = 0

        if use_pool:
            chunk_size = 128
            chunks = [chunk_size] * (num_episodes // chunk_size)
            remainder = num_episodes - sum(chunks)
            if remainder:
                chunks.append(remainder)

            rng = np.random.default_rng(seed)
            seeds = rng.integers(0, 2**31 - 1, size=len(chunks)).tolist()

            worker_args = [
                (env_config, n, max_steps, frame_size, int(s))
                for n, s in zip(chunks, seeds)
            ]

            ctx = mp.get_context("spawn")
            with ctx.Pool(num_workers) as pool:
                with tqdm(total=num_episodes, desc=f"{env_kind}:{split} ({num_workers} workers)") as pbar:
                    for chunk in pool.imap_unordered(_collect_chunk, worker_args):
                        _write_episodes(handle, chunk, offset=written)
                        written += len(chunk)
                        pbar.update(len(chunk))
        else:
            env = build_toy_env(env_config)
            rng = np.random.default_rng(seed)
            try:
                for _ in tqdm(range(num_episodes), desc=f"{env_kind}:{split}"):
                    episode = collect_episode(env, max_steps=max_steps, frame_size=frame_size, rng=rng)
                    _write_episodes(handle, [episode], offset=written)
                    written += 1
            finally:
                env.close()

    return probe_env


def main():
    parser = argparse.ArgumentParser(description="Collect offline random toy-environment rollouts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", default=None, help="Collect only a single environment from the config.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (ignored for craftax).")
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
        env_path = sanitize_name(env_name)

        collect_and_save_split(
            env_config, "train", frame_size, train_seed,
            output_dir / f"{env_path}_train.h5", env_name,
            num_workers=args.workers,
        )
        collect_and_save_split(
            env_config, "val", frame_size, val_seed,
            output_dir / f"{env_path}_val.h5", env_name,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
