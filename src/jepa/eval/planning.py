import argparse
import json
import statistics
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from jepa.datasets.toy_env_dataset import IMAGENET_MEAN, IMAGENET_STD
from jepa.envs.toy_envs import build_toy_env
from jepa.models.jepa import JEPA
from jepa.planning.beam import BeamPlanner
from jepa.planning.cem import CEMPlanner
from jepa.planning.collocation import CollocationPlanner
from jepa.planning.mppi import MPPIPlanner
from jepa.planning.smc import SMCPlanner


N_TRAJ_SHOW = 3
EPISODE_ATTEMPTS_MULTIPLIER = 4


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
    parser.add_argument("--use-latent", action="store_true", default=False)

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
            use_latent=args.use_latent,
        )
        cfg = {"steps": args.steps, "lr": args.lr, "optimizer": args.optimizer,
               "project": args.project, "use_latent": args.use_latent}
    elif args.planner == "cem":
        planner = CEMPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            elite_frac=args.elite_frac, iterations=args.iterations,
            alpha=args.alpha,
        )
        cfg = {"population": args.population, "elite_frac": args.elite_frac,
               "iterations": args.iterations, "alpha": args.alpha,
               "use_latent": args.use_latent}
    elif args.planner == "mppi":
        planner = MPPIPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            temperature=args.temperature, iterations=args.iterations,
            alpha=args.alpha,
        )
        cfg = {"population": args.population, "temperature": args.temperature,
               "iterations": args.iterations, "alpha": args.alpha,
               "use_latent": args.use_latent}
    elif args.planner == "smc":
        planner = SMCPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, population=args.population,
            temperature=args.temperature, ess_threshold=args.ess_threshold,
        )
        cfg = {"population": args.population, "temperature": args.temperature,
               "ess_threshold": args.ess_threshold,
               "use_latent": args.use_latent}
    elif args.planner == "beam":
        planner = BeamPlanner(
            wm=model, action_dim=0, pre_processor=None,
            horizon=args.horizon, beam_width=args.beam_width,
            branching=args.branching, temperature=args.temperature,
        )
        cfg = {"beam_width": args.beam_width, "branching": args.branching,
               "temperature": args.temperature,
               "use_latent": args.use_latent}
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


def token_rms_norm(mu):
    D = mu.shape[-1]
    return (mu.pow(2).sum(-1) / D).sqrt().mean().item()


def per_step_norms(mu):
    return [token_rms_norm(mu[:, t]) for t in range(mu.shape[1])]


def softmax_entropy(logits):
    p = F.softmax(logits.float(), dim=-1)
    log_p = F.log_softmax(logits.float(), dim=-1)
    return -(p * log_p).sum(-1).mean().item()


def per_step_entropies(logits, start, stop):
    return [softmax_entropy(logits[:, t]) for t in range(start, stop)]


def traj_metrics(model, traj, horizon, use_latent, use_entropy):
    """Norms + optional entropies for a (1, T+1, N, D) trajectory."""
    mu = model.predict_all(traj) if use_latent else model.predict_all(traj[:, :-1])
    out = {"norms": per_step_norms(mu)}
    if use_entropy:
        logits = model.decode_actions(traj)
        out["entropies"] = per_step_entropies(logits, 0, horizon)
    return out


@torch.inference_mode()
def collect_episode(model, env, rng, horizon, frame_size, device,
                    use_entropy, use_latent):
    """Run a random episode, encode frames, compute real-trajectory metrics.

    Returns None on early termination.
    """
    rng_state_before_reset = rng.bit_generator.state
    env.reset(rng)
    frames = [env.render(frame_size)]
    for _ in range(horizon):
        if env.step(env.sample_action(rng)):
            return None
        frames.append(env.render(frame_size))

    with torch.amp.autocast("cuda"):
        x_seq = preprocess_frames(frames, device)
        traj_real = model.encode(x_seq)
        z_0 = traj_real[:, 0]
        z_T = traj_real[:, -1]

        real_metrics = traj_metrics(model, traj_real, horizon, use_latent, use_entropy)

    ep = {
        "z_0": z_0,
        "z_T": z_T,
        "traj_real": traj_real,
        "frames": frames,
        "rng_state_before_reset": rng_state_before_reset,
        "norms_real": real_metrics["norms"],
    }
    if use_entropy:
        ep["entropies_real"] = real_metrics["entropies"]
    return ep


@torch.inference_mode()
def finalize_episode(model, ep_data, traj_opt, horizon, use_entropy, use_latent):
    """Compute opt-trajectory metrics and strip tensors for JSON storage."""
    with torch.amp.autocast("cuda"):
        opt_metrics = traj_metrics(model, traj_opt, horizon, use_latent, use_entropy)

    result = {
        "norms_real": ep_data["norms_real"],
        "norms_opt": opt_metrics["norms"],
    }
    if use_entropy:
        result["entropies_real"] = ep_data["entropies_real"]
        result["entropies_opt"] = opt_metrics["entropies"]
    return result


@torch.inference_mode()
def execute_plan_in_env(model, env, ep_data, traj_opt, rng, frame_size, device):
    """Replay the env to the episode's initial state and execute decoded actions.

    Returns (rms distance of final encoded frame to z_T, frames of execution,
    success). `success` is True iff `env.step(...)` returned done=True at any
    point during execution — i.e. the env terminated during the planned run.
    For most toy envs (pointmaze, keydoor, sokoban, pusht, push) this is the
    "goal reached" signal. Note that random pre-rollouts in `collect_episode`
    discard episodes that go done, so success here means the executed plan
    accomplished something the random policy did not.
    """
    horizon = traj_opt.shape[1] - 1

    saved = rng.bit_generator.state
    rng.bit_generator.state = ep_data["rng_state_before_reset"]
    env.reset(rng)
    rng.bit_generator.state = saved

    exec_frames = [env.render(frame_size)]
    success = False

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
            success = True
            break

    while len(exec_frames) < horizon + 1:
        exec_frames.append(exec_frames[-1])

    with torch.amp.autocast("cuda"):
        x_final = preprocess_frames([exec_frames[-1]], device)
        z_exec_T = model.encode(x_final)[:, 0]
        dist = (z_exec_T - ep_data["z_T"]).pow(2).mean().sqrt().item()

    return dist, exec_frames, success


def aggregate(episodes):
    out = {}
    horizon = len(episodes[0]["norms_real"])
    for key in ("norms_real", "norms_opt",
                "entropies_real", "entropies_opt"):
        if key not in episodes[0]:
            continue
        per_t = [[ep[key][t] for ep in episodes] for t in range(horizon)]
        out[key] = {
            "mean": [statistics.mean(per_t[t]) for t in range(horizon)],
            "std":  [statistics.pstdev(per_t[t]) if len(per_t[t]) > 1 else 0.0
                     for t in range(horizon)],
        }

    for key in ("exec_dist", "exec_dist_real"):
        if key not in episodes[0]:
            continue
        dists = [ep[key] for ep in episodes]
        out[key] = {
            "mean": statistics.mean(dists),
            "median": statistics.median(dists),
            "std": statistics.pstdev(dists),
        }

    for key in ("success", "success_real"):
        if key not in episodes[0]:
            continue
        flags = [bool(ep[key]) for ep in episodes]
        n = len(flags)
        k = sum(flags)
        out[f"{key}_rate"] = {
            "rate": k / n if n else 0.0,
            "n_success": k,
            "n_total": n,
        }

    return out


def style_ax(ax):
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="-", linewidth=0.4, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbb")
    ax.tick_params(labelsize=7, colors="#555", length=3)


def plot_metric_section(fig, gs, row_line, row_hist, metrics, episodes,
                        keys, labels, base_colors, cmaps,
                        steps, horizon, t_intensities,
                        line_ylabel, hist_xlabel, title):
    from matplotlib.patches import Patch

    ax_line = fig.add_subplot(gs[row_line, :])
    style_ax(ax_line)
    ax_line.tick_params(labelsize=8)
    for key in keys:
        means = np.array(metrics[key]["mean"])
        stds = np.array(metrics[key]["std"])
        ax_line.plot(steps, means, color=base_colors[key], linewidth=1.8,
                     label=labels[key], zorder=3)
        ax_line.fill_between(steps, means - stds, means + stds,
                             color=base_colors[key], alpha=0.18, zorder=2)
    ax_line.set_xlabel("Trajectory step $t$", fontsize=9)
    ax_line.set_ylabel(line_ylabel, fontsize=9)
    ax_line.legend(fontsize=8, framealpha=0.9)
    ax_line.set_title(title, fontsize=10, color="#222")

    all_vals = [ep[key][t] for ep in episodes for key in keys for t in range(horizon)]
    lo, hi = min(all_vals) * 0.97, max(all_vals) * 1.03
    bins = np.linspace(lo, hi, 30)

    for col_i, key in enumerate(keys):
        ax = fig.add_subplot(gs[row_hist, col_i])
        style_ax(ax)
        ax.tick_params(labelsize=7)
        cmap = cmaps[key]
        for t in range(horizon):
            vals = [ep[key][t] for ep in episodes]
            color = cmap(t_intensities[t])
            ax.hist(vals, bins=bins, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6,
                    histtype="stepfilled", zorder=2 + t)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(hist_xlabel, fontsize=8)
        if col_i == 0:
            ax.set_ylabel("Count (log)", fontsize=8)
        ax.set_title(labels[key], fontsize=9, color=base_colors[key], pad=5)
        legend_handles = [
            Patch(facecolor=cmap(t_intensities[0]),  label="t=0",
                  edgecolor=cmap(t_intensities[0]),  linewidth=0.5),
            Patch(facecolor=cmap(t_intensities[-1]), label=f"t={horizon - 1}",
                  edgecolor=cmap(t_intensities[-1]), linewidth=0.5),
        ]
        ax.legend(handles=legend_handles, fontsize=7, framealpha=0.85,
                  loc="upper right", handlelength=1.2)


def plot_execution_trajectories(traj_frames, horizon, env_name, step_num, output_path):
    N = len(traj_frames)
    T = horizon + 1
    n_cols = min(T, 6)
    t_indices = np.round(np.linspace(0, T - 1, n_cols)).astype(int)

    cell = 1.3
    real_color = "#4878d0"
    exec_color = "#ee854a"

    fig = plt.figure(figsize=(n_cols * cell + 0.7, N * 2 * cell + 0.8))
    fig.patch.set_facecolor("white")

    outer = fig.add_gridspec(N, 1, hspace=0.18, left=0.06, right=0.99,
                             top=0.93, bottom=0.02)

    for ep_i, pair in enumerate(traj_frames):
        inner = outer[ep_i].subgridspec(2, n_cols, wspace=0.04, hspace=0.04)
        for row_off, frames_key, color, row_label in (
            (0, "real", real_color, "real"),
            (1, "exec", exec_color, "exec"),
        ):
            frames = pair[frames_key]
            for col_i, t in enumerate(t_indices):
                ax = fig.add_subplot(inner[row_off, col_i])
                ax.imshow(frames[t], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col_i == 0:
                    ax.text(
                        -0.08, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=9, color=color,
                        ha="right", va="center", fontweight="semibold",
                    )
                if ep_i == 0 and row_off == 0:
                    ax.set_title(f"$t={t}$", fontsize=10, color="#333", pad=6)

    fig.suptitle(
        f"Real vs. Executed — {env_name} @ step {step_num}",
        fontsize=12, color="#222", y=0.985,
    )
    plot_path = output_path.with_name(output_path.stem + "_trajectories.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {plot_path}")


def plot_distributions(result, output_path, planner_name):
    episodes = result["per_episode"]
    metrics = result["metrics"]
    horizon = result["horizon"]
    steps = np.arange(horizon)
    has_entropy = "entropies_real" in metrics

    NORM_KEYS = ("norms_real", "norms_opt")
    ENTROPY_KEYS = ("entropies_real", "entropies_opt")
    BASE_COLORS = {
        "norms_real": "#6acc65", "entropies_real": "#6acc65",
        "norms_opt":  "#ee854a", "entropies_opt":  "#ee854a",
    }
    CMAPS = {
        "norms_real": plt.cm.Greens,  "entropies_real": plt.cm.Greens,
        "norms_opt":  plt.cm.Oranges, "entropies_opt":  plt.cm.Oranges,
    }
    LABELS = {
        "norms_real": "Real trajectory", "entropies_real": "Real trajectory",
        "norms_opt":  "Opt plan",        "entropies_opt":  "Opt plan",
    }

    t_intensities = np.linspace(0.30, 0.90, horizon)

    has_exec = "exec_dist" in metrics
    n_metric_sections = 1 + (1 if has_entropy else 0)
    n_sections = n_metric_sections + (1 if has_exec else 0)
    n_rows = 2 * n_metric_sections + (1 if has_exec else 0)
    height_ratios = [1.5, 1.0] * n_metric_sections + ([1.0] if has_exec else [])

    fig = plt.figure(figsize=(12, 3.5 * n_sections + 1.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(n_rows, 2, height_ratios=height_ratios,
                          hspace=0.55, wspace=0.28)

    n_ep = result.get("num_episodes_used", len(episodes))
    env_name = result.get("env_name", "")
    step_num = result.get("step", -1)

    plot_metric_section(
        fig, gs, row_line=0, row_hist=1,
        metrics=metrics, episodes=episodes,
        keys=NORM_KEYS, labels=LABELS,
        base_colors=BASE_COLORS, cmaps=CMAPS,
        steps=steps, horizon=horizon, t_intensities=t_intensities,
        line_ylabel=r"$\|\mu\| / \sqrt{D}$  (RMS norm per dim)",
        hist_xlabel=r"$\|\mu\| / \sqrt{D}$",
        title=f"{planner_name.capitalize()} Planning — {env_name} @ step {step_num}  ({n_ep} episodes)  —  Prediction norms",
    )

    if has_entropy:
        plot_metric_section(
            fig, gs, row_line=2, row_hist=3,
            metrics=metrics, episodes=episodes,
            keys=ENTROPY_KEYS, labels=LABELS,
            base_colors=BASE_COLORS, cmaps=CMAPS,
            steps=steps, horizon=horizon, t_intensities=t_intensities,
            line_ylabel="Action decoder entropy (nats)",
            hist_xlabel="Entropy (nats)",
            title=f"Action decoder entropy — {env_name} @ step {step_num}",
        )

    if has_exec:
        exec_row = 2 * n_metric_sections
        ax_exec = fig.add_subplot(gs[exec_row, :])
        style_ax(ax_exec)
        ax_exec.tick_params(labelsize=8)

        dists_opt = [ep["exec_dist"] for ep in episodes]
        dists_real = [ep["exec_dist_real"] for ep in episodes]
        lo_e = min(min(dists_opt), min(dists_real)) * 0.97
        hi_e = max(max(dists_opt), max(dists_real)) * 1.03
        bins_e = np.linspace(lo_e, hi_e, 40)

        ax_exec.hist(dists_real, bins=bins_e, color="#6acc65", alpha=0.7,
                     edgecolor="white", linewidth=0.4, zorder=2, label="real traj (baseline)")
        ax_exec.hist(dists_opt,  bins=bins_e, color="#8172b2", alpha=0.7,
                     edgecolor="white", linewidth=0.4, zorder=3, label="opt plan")

        mean_opt = metrics["exec_dist"]["mean"]
        mean_real = metrics["exec_dist_real"]["mean"]
        ax_exec.axvline(mean_opt,  color="#8172b2", linewidth=1.2, linestyle="--",
                        zorder=4, label=f"opt mean={mean_opt:.3f}")
        ax_exec.axvline(mean_real, color="#6acc65", linewidth=1.2, linestyle="--",
                        zorder=4, label=f"real mean={mean_real:.3f}")

        median_opt = metrics["exec_dist"]["median"]
        median_real = metrics["exec_dist_real"]["median"]
        ax_exec.set_title(
            f"Execution distance  —  opt: mean={mean_opt:.3f} median={median_opt:.3f}"
            f"  |  real baseline: mean={mean_real:.3f} median={median_real:.3f}",
            fontsize=10, color="#222",
        )
        ax_exec.set_xlabel(r"$\|\hat{z}_T - z_T\|_{\mathrm{RMS}}$  (executed vs target embedding)", fontsize=9)
        ax_exec.set_ylabel("Count", fontsize=9)
        ax_exec.legend(fontsize=8)

    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {plot_path}")


def resolve_output_path(args):
    """Resolve where to write the eval JSON.

    Default convention: `<sweep>/eval/<run_id>/planning_<planner>.json`. The
    `<run_id>` here is the wandb run id (the checkpoint dir name); each
    checkpoint gets its own subdir so multi-eval outputs stay grouped.
    """
    if args.output is not None:
        return Path(args.output)
    ckpt = Path(args.checkpoint).resolve()
    run_id = ckpt.parent.name
    sweep_dir = ckpt.parent.parent.parent
    return sweep_dir / "eval" / run_id / f"planning_{args.planner}.json"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, step, config = load_model(args.checkpoint, device)

    bottleneck = model.bottleneck_type
    if bottleneck != "none" and not args.use_latent:
        print(f"[planning eval] bottleneck={bottleneck!r}; forcing --use-latent")
        args.use_latent = True

    decoder_cfg = config.get("action_decoder") or {}
    use_entropy = (
        model.action_decoder is not None
        and decoder_cfg.get("enabled", False)
        and decoder_cfg.get("action_type") == "discrete"
    )

    env, frame_size = build_env(args.env_config, args.env_name)
    planner, planner_cfg = build_planner(args, model)
    rng = np.random.default_rng(args.seed)

    episodes = []
    traj_frames = []
    attempts = 0
    max_attempts = args.num_episodes * EPISODE_ATTEMPTS_MULTIPLIER
    try:
        pbar = tqdm(total=args.num_episodes, desc=f"{args.planner} planning")
        pending = []
        while len(episodes) < args.num_episodes and attempts < max_attempts:
            while len(pending) < args.batch_size and attempts < max_attempts:
                attempts += 1
                ep_data = collect_episode(model, env, rng, args.horizon,
                                          frame_size, device, use_entropy,
                                          args.use_latent)
                if ep_data is not None:
                    pending.append(ep_data)

            if not pending:
                break

            z_0_batch = torch.cat([ep["z_0"] for ep in pending], dim=0)
            z_T_batch = torch.cat([ep["z_T"] for ep in pending], dim=0)
            traj_opt_batch = planner.plan(z_0_batch, z_T_batch)

            for i, ep_data in enumerate(pending):
                ep = finalize_episode(model, ep_data, traj_opt_batch[i : i + 1],
                                      args.horizon, use_entropy, args.use_latent)
                if model.action_decoder is not None:
                    exec_dist, exec_frames, success = execute_plan_in_env(
                        model, env, ep_data, traj_opt_batch[i : i + 1],
                        rng, frame_size, device,
                    )
                    exec_dist_real, _, success_real = execute_plan_in_env(
                        model, env, ep_data, ep_data["traj_real"],
                        rng, frame_size, device,
                    )
                    ep["exec_dist"] = exec_dist
                    ep["exec_dist_real"] = exec_dist_real
                    ep["success"] = success
                    ep["success_real"] = success_real
                    if len(traj_frames) < N_TRAJ_SHOW:
                        traj_frames.append({
                            "real": ep_data["frames"],
                            "exec": exec_frames,
                        })
                episodes.append(ep)
                pbar.update(1)
                if len(episodes) >= args.num_episodes:
                    break
            pending = []
        pbar.close()
    finally:
        env.close()

    if len(episodes) < args.num_episodes:
        warnings.warn(
            f"only collected {len(episodes)} / {args.num_episodes} episodes "
            f"in {attempts} attempts"
        )

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": step,
        "env_name": args.env_name,
        "horizon": args.horizon,
        "planner_name": args.planner,
        "planner": planner_cfg,
        "batch_size": args.batch_size,
        "num_episodes_requested": args.num_episodes,
        "num_episodes_used": len(episodes),
        "num_attempts": attempts,
        "seed": args.seed,
        "git_hash": git_hash,
        "metrics": aggregate(episodes),
        "per_episode": episodes,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{args.env_name} @ step {step} ({len(episodes)} episodes, planner={args.planner})")
    m = result["metrics"]
    if "success_rate" in m:
        sr = m["success_rate"]
        print(f"  success_rate    {sr['rate'] * 100:5.1f}%  ({sr['n_success']}/{sr['n_total']})")
    if "success_real_rate" in m:
        sr = m["success_real_rate"]
        print(f"  success_rate_real {sr['rate'] * 100:5.1f}%  ({sr['n_success']}/{sr['n_total']})  (baseline: re-execute decoded real traj)")
    if "exec_dist" in m:
        print(f"  exec_dist       mean={m['exec_dist']['mean']:+.4f}  "
              f"median={m['exec_dist']['median']:+.4f}  "
              f"σ={m['exec_dist']['std']:.4f}")
        print(f"  exec_dist_real  mean={m['exec_dist_real']['mean']:+.4f}  "
              f"median={m['exec_dist_real']['median']:+.4f}  "
              f"σ={m['exec_dist_real']['std']:.4f}  (baseline: real traj)")
    print(f"Wrote {output_path}")

    if episodes:
        plot_distributions(result, output_path, args.planner)
    if traj_frames:
        plot_execution_trajectories(
            traj_frames, args.horizon, args.env_name, step, output_path
        )


if __name__ == "__main__":
    main()
