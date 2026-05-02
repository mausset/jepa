import argparse
import json
import os
import secrets
import statistics
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from jepa.datasets.toy_env_dataset import IMAGENET_MEAN, IMAGENET_STD
from jepa.envs.toy_envs import build_toy_env
from jepa.eval.planning_plots import plot_distributions, plot_execution_trajectories
from jepa.models.jepa import JEPA
from jepa.planning.beam import BeamPlanner
from jepa.planning.cem import CEMPlanner
from jepa.planning.collocation import CollocationPlanner
from jepa.planning.mppi import MPPIPlanner
from jepa.planning.smc import SMCPlanner


N_TRAJ_SHOW = 3
EPISODE_ATTEMPTS_MULTIPLIER = 4

# Powers-of-two threshold ladder for goal-reaching success rate. Reported for
# both latent distance ‖z_exec_T - z_T‖_rms and state-vector distance
# ‖env.state_vector() - target_state‖_2 — we don't know a priori where these
# distributions sit, so report the full curve.
EPS_THRESHOLDS = tuple(0.001 * (2 ** i) for i in range(8))


@dataclass
class EpisodeData:
    z_0: torch.Tensor
    z_T: torch.Tensor
    traj_real: torch.Tensor
    frames: list[np.ndarray]
    rng_state_before_reset: dict[str, Any]
    target_state: np.ndarray


@dataclass
class EpisodeResult:
    exec_dist: float | None = None
    exec_dist_real: float | None = None
    state_dist: float | None = None
    state_dist_real: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("exec_dist", "exec_dist_real", "state_dist", "state_dist_real"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass(frozen=True)
class ExecutionResult:
    latent_dist: float
    state_dist: float
    frames: list[np.ndarray]


@dataclass(frozen=True)
class CollectedEpisodes:
    per_episode: list[dict[str, Any]]
    traj_frames: list[dict[str, list[np.ndarray]]]
    attempts: int


def current_git_hash():
    env_hash = os.environ.get("SOURCE_GIT_HASH")
    if env_hash:
        return env_hash
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="configs/envs/toy_envs.yaml")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--planner", type=str, default="mppi",
                        choices=["collocation", "cem", "mppi", "smc", "beam"])

    # Collocation-specific
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "ula"])
    parser.add_argument("--project", action="store_true", default=False)

    # CEM / MPPI shared. Defaults tuned for MPPI as the canonical eval planner:
    # smaller population/iterations than the planning literature usually uses,
    # to keep eval cost in the few-minutes-per-checkpoint range. Revisit via a
    # calibration sweep once we have a known-good baseline checkpoint.
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)

    # CEM-specific
    parser.add_argument("--elite-frac", type=float, default=0.1)

    # MPPI / SMC shared
    parser.add_argument("--temperature", type=float, default=1.0)

    # SMC-specific
    parser.add_argument("--ess-threshold", type=float, default=0.5)

    # Beam-specific
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branching", type=int, default=16)

    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    step = int(ckpt.get("step", -1))

    model = JEPA(
        config["encoder"],
        config["predictor"],
        config.get("action_decoder"),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, step, config


def build_planner(args, model):
    if args.planner == "collocation":
        planner = CollocationPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, steps=args.steps, lr=args.lr,
            optimizer=args.optimizer, project=args.project,
        )
        cfg = {"steps": args.steps, "lr": args.lr, "optimizer": args.optimizer,
               "project": args.project}
    elif args.planner == "cem":
        planner = CEMPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            elite_frac=args.elite_frac, iterations=args.iterations,
            alpha=args.alpha,
        )
        cfg = {"population": args.population, "elite_frac": args.elite_frac,
               "iterations": args.iterations, "alpha": args.alpha}
    elif args.planner == "mppi":
        planner = MPPIPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            temperature=args.temperature, iterations=args.iterations,
            alpha=args.alpha,
        )
        cfg = {"population": args.population, "temperature": args.temperature,
               "iterations": args.iterations, "alpha": args.alpha}
    elif args.planner == "smc":
        planner = SMCPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            temperature=args.temperature, ess_threshold=args.ess_threshold,
        )
        cfg = {"population": args.population, "temperature": args.temperature,
               "ess_threshold": args.ess_threshold}
    elif args.planner == "beam":
        planner = BeamPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, beam_width=args.beam_width,
            branching=args.branching, temperature=args.temperature,
        )
        cfg = {"beam_width": args.beam_width, "branching": args.branching,
               "temperature": args.temperature}
    else:
        raise ValueError(f"Unknown planner: {args.planner!r}")
    return planner, cfg


def build_env(env_config_path, env_name):
    raw = OmegaConf.to_container(OmegaConf.load(env_config_path), resolve=True)
    env_spec = dict(raw["envs"][env_name])
    env_spec.setdefault("frame_size", raw.get("frame_size", [96, 96]))
    frame_size = tuple(env_spec["frame_size"])
    return build_toy_env(env_spec), frame_size


def preprocess_frames(frames, device):
    """Normalize a list of HxWx3 uint8 frames to (1, T, H, W, 3) tensor."""
    arr = np.stack(frames)
    x = torch.from_numpy(arr).to(device).float().div_(255.0)
    mean = IMAGENET_MEAN.view(1, 1, 1, 3).to(device)
    std = IMAGENET_STD.view(1, 1, 1, 3).to(device)
    return ((x - mean) / std).unsqueeze(0)


@torch.inference_mode()
def collect_episode(model, env, rng, horizon, frame_size, device):
    """Run a random episode and encode frames.

    Returns None on early termination.
    """
    rng_state_before_reset = rng.bit_generator.state
    env.reset(rng)
    frames = [env.render(frame_size)]
    for _ in range(horizon):
        if env.step(env.sample_action(rng)):
            return None
        frames.append(env.render(frame_size))

    target_state = np.asarray(env.state_vector(), dtype=np.float64).flatten()

    with torch.amp.autocast("cuda"):
        x_seq = preprocess_frames(frames, device)
        traj_real = model.encode(x_seq)
        z_0 = traj_real[:, 0]
        z_T = traj_real[:, -1]

    return EpisodeData(
        z_0=z_0,
        z_T=z_T,
        traj_real=traj_real,
        frames=frames,
        rng_state_before_reset=rng_state_before_reset,
        target_state=target_state,
    )




@torch.inference_mode()
def execute_plan_in_env(model, env, ep_data, traj_opt, rng, frame_size, device):
    """Replay the env to the episode's initial state and execute decoded actions.

    Goal-reaching framing: we measure whether execution ends *near the target
    state* (the random rollout's final state, which gave us z_T), not whether
    the env's own task-completion flag fired. Returns:

      - `latent_dist`: ‖z_exec_T - z_T‖_rms — final-frame embedding distance
      - `state_dist`:  ‖env.state_vector() - target_state‖_2 — state-vector L2
      - `exec_frames`: rendered frames for plotting

    We still break on `env.step(...)` returning done because some envs are
    unsafe to step past terminal; the env's `state_vector()` at break time is
    treated as the achieved final state.
    """
    horizon = traj_opt.shape[1] - 1

    saved = rng.bit_generator.state
    rng.bit_generator.state = ep_data.rng_state_before_reset
    env.reset(rng)
    rng.bit_generator.state = saved

    exec_frames = [env.render(frame_size)]

    with torch.amp.autocast("cuda"):
        action_logits = model.decode_actions(traj_opt)[0]

    for t in range(horizon):
        if env.action_type == "discrete":
            action = int(action_logits[t].argmax().item())
        else:
            action = action_logits[t].cpu().numpy()
        done = env.step(action)
        exec_frames.append(env.render(frame_size))
        if done:
            break

    final_state = np.asarray(env.state_vector(), dtype=np.float64).flatten()
    state_dist = float(np.linalg.norm(final_state - ep_data.target_state))

    while len(exec_frames) < horizon + 1:
        exec_frames.append(exec_frames[-1])

    with torch.amp.autocast("cuda"):
        x_final = preprocess_frames([exec_frames[-1]], device)
        z_exec_T = model.encode(x_final)[:, 0]
        latent_dist = (z_exec_T - ep_data.z_T).pow(2).mean().sqrt().item()

    return ExecutionResult(latent_dist=latent_dist, state_dist=state_dist, frames=exec_frames)


def plan_batch(planner, pending: list[EpisodeData]) -> torch.Tensor:
    z_0_batch = torch.cat([ep.z_0 for ep in pending], dim=0)
    z_T_batch = torch.cat([ep.z_T for ep in pending], dim=0)
    return planner.plan(z_0_batch, z_T_batch)


def evaluate_episode(
    model,
    env,
    ep_data: EpisodeData,
    traj_opt: torch.Tensor,
    rng,
    frame_size,
    device,
    capture_frames: bool,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]] | None]:
    ep = EpisodeResult()
    traj_pair = None

    if model.action_decoder is not None:
        opt_exec = execute_plan_in_env(
            model, env, ep_data, traj_opt, rng, frame_size, device
        )
        real_exec = execute_plan_in_env(
            model, env, ep_data, ep_data.traj_real, rng, frame_size, device
        )
        ep.exec_dist = opt_exec.latent_dist
        ep.exec_dist_real = real_exec.latent_dist
        ep.state_dist = opt_exec.state_dist
        ep.state_dist_real = real_exec.state_dist
        if capture_frames:
            traj_pair = {"real": ep_data.frames, "exec": opt_exec.frames}

    return ep.to_json(), traj_pair


def collect_episodes(
    model,
    env,
    planner,
    rng,
    *,
    num_episodes: int,
    batch_size: int,
    horizon: int,
    frame_size,
    device,
    planner_name: str,
) -> CollectedEpisodes:
    episodes: list[dict[str, Any]] = []
    traj_frames: list[dict[str, list[np.ndarray]]] = []
    attempts = 0
    max_attempts = num_episodes * EPISODE_ATTEMPTS_MULTIPLIER

    with tqdm(total=num_episodes, desc=f"{planner_name} planning") as pbar:
        pending: list[EpisodeData] = []
        while len(episodes) < num_episodes and attempts < max_attempts:
            while len(pending) < batch_size and attempts < max_attempts:
                attempts += 1
                ep_data = collect_episode(
                    model, env, rng, horizon, frame_size, device
                )
                if ep_data is not None:
                    pending.append(ep_data)

            if not pending:
                break

            traj_opt_batch = plan_batch(planner, pending)
            for i, ep_data in enumerate(pending):
                ep, frames = evaluate_episode(
                    model=model,
                    env=env,
                    ep_data=ep_data,
                    traj_opt=traj_opt_batch[i : i + 1],
                    rng=rng,
                    frame_size=frame_size,
                    device=device,
                    capture_frames=len(traj_frames) < N_TRAJ_SHOW,
                )
                if frames is not None:
                    traj_frames.append(frames)
                episodes.append(ep)
                pbar.update(1)
                if len(episodes) >= num_episodes:
                    break
            pending = []

    return CollectedEpisodes(per_episode=episodes, traj_frames=traj_frames, attempts=attempts)


def aggregate(episodes):
    out = {}
    if not episodes:
        return out

    for key in ("exec_dist", "exec_dist_real", "state_dist", "state_dist_real"):
        if key not in episodes[0]:
            continue
        dists = [ep[key] for ep in episodes]
        out[key] = {
            "mean": statistics.mean(dists),
            "median": statistics.median(dists),
            "std": statistics.pstdev(dists),
        }

    for label, opt_key, real_key in (
        ("success_rate_latent", "exec_dist", "exec_dist_real"),
        ("success_rate_state",  "state_dist", "state_dist_real"),
    ):
        if opt_key not in episodes[0]:
            continue
        opt_dists = [ep[opt_key] for ep in episodes]
        real_dists = [ep[real_key] for ep in episodes]
        n = len(opt_dists)
        out[label] = {
            "thresholds": list(EPS_THRESHOLDS),
            "opt": success_curve(opt_dists, EPS_THRESHOLDS, n),
            "real": success_curve(real_dists, EPS_THRESHOLDS, n),
        }

    return out


def success_curve(distances, thresholds, n):
    rates, n_success = [], []
    for eps in thresholds:
        k = sum(1 for d in distances if d < eps)
        n_success.append(k)
        rates.append(k / n if n else 0.0)
    return {"rates": rates, "n_success": n_success, "n_total": n}


def resolve_output_path(args):
    """Resolve where to write the eval JSON.

    Default convention: `<sweep>/eval/<run_id>/planning_<planner>_<env>_<hex>.json`.
    The `<run_id>` here is the artifact run id (the checkpoint dir name); each
    checkpoint gets its own subdir so multi-eval outputs stay grouped.
    """
    if args.output is not None:
        return Path(args.output)
    ckpt = Path(args.checkpoint).resolve()
    run_id = ckpt.parent.name
    sweep_dir = ckpt.parent.parent.parent
    name = f"planning_{args.planner}_{args.env_name}_{secrets.token_hex(3)}.json"
    return sweep_dir / "eval" / run_id / name


def require_latent_model(model) -> None:
    if not model.has_bottleneck:
        raise ValueError(
            "planning eval requires a latent bottleneck predictor "
            f"(fsq or vae), got {model.bottleneck_type!r}"
        )


def build_eval_result(
    args,
    step: int,
    planner_cfg: dict[str, Any],
    episodes: list[dict[str, Any]],
    attempts: int,
) -> dict[str, Any]:
    planner_kwargs = {
        "num_episodes": args.num_episodes,
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        **planner_cfg,
    }
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": step,
        "env_name": args.env_name,
        "horizon": args.horizon,
        "planner_name": args.planner,
        "planner": planner_cfg,
        "planner_kwargs": planner_kwargs,
        "batch_size": args.batch_size,
        "num_episodes_requested": args.num_episodes,
        "num_episodes_used": len(episodes),
        "num_attempts": attempts,
        "seed": args.seed,
        "git_hash": current_git_hash(),
        "metrics": aggregate(episodes),
        "per_episode": episodes,
    }


def write_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def print_summary(result: dict[str, Any]) -> None:
    print(
        f"\n{result['env_name']} @ step {result['step']} "
        f"({result['num_episodes_used']} episodes, planner={result['planner_name']})"
    )
    metrics = result["metrics"]
    for key, label in (
        ("exec_dist",       "exec_dist        (latent, opt) "),
        ("exec_dist_real",  "exec_dist_real   (latent, real)"),
        ("state_dist",      "state_dist       (state,  opt) "),
        ("state_dist_real", "state_dist_real  (state,  real)"),
    ):
        if key not in metrics:
            continue
        d = metrics[key]
        print(f"  {label}  mean={d['mean']:+.4f}  median={d['median']:+.4f}  σ={d['std']:.4f}")

    if "success_rate_latent" not in metrics and "success_rate_state" not in metrics:
        return

    print("\n  success rate vs ε  (rate = fraction with distance < ε)")
    print(f"  {'ε':>9}   {'lat-opt':>8} {'lat-real':>9}   {'sta-opt':>8} {'sta-real':>9}")
    lat = metrics.get("success_rate_latent")
    sta = metrics.get("success_rate_state")
    for i, eps in enumerate(EPS_THRESHOLDS):
        lat_opt  = f"{lat['opt']['rates'][i] * 100:6.1f}%" if lat else "    -- "
        lat_real = f"{lat['real']['rates'][i] * 100:6.1f}%" if lat else "    -- "
        sta_opt  = f"{sta['opt']['rates'][i] * 100:6.1f}%" if sta else "    -- "
        sta_real = f"{sta['real']['rates'][i] * 100:6.1f}%" if sta else "    -- "
        print(f"  {eps:9.4f}   {lat_opt:>8} {lat_real:>9}   {sta_opt:>8} {sta_real:>9}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, step, _config = load_model(args.checkpoint, device)
    require_latent_model(model)

    env, frame_size = build_env(args.env_config, args.env_name)
    planner, planner_cfg = build_planner(args, model)
    rng = np.random.default_rng(args.seed)

    try:
        collected = collect_episodes(
            model=model,
            env=env,
            planner=planner,
            rng=rng,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            horizon=args.horizon,
            frame_size=frame_size,
            device=device,
            planner_name=args.planner,
        )
    finally:
        env.close()

    episodes = collected.per_episode
    traj_frames = collected.traj_frames
    attempts = collected.attempts
    if len(episodes) < args.num_episodes:
        warnings.warn(
            f"only collected {len(episodes)} / {args.num_episodes} episodes "
            f"in {attempts} attempts"
        )

    output_path = resolve_output_path(args)
    result = build_eval_result(args, step, planner_cfg, episodes, attempts)
    write_result(result, output_path)
    print_summary(result)

    print(f"Wrote {output_path}")

    if episodes:
        plot_distributions(result, output_path, args.planner, EPS_THRESHOLDS)
    if traj_frames:
        plot_execution_trajectories(
            traj_frames, args.horizon, args.env_name, step, output_path
        )


if __name__ == "__main__":
    main()
