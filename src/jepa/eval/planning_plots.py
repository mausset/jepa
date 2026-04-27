import matplotlib.pyplot as plt
import numpy as np


def style_ax(ax):
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="-", linewidth=0.4, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbb")
    ax.tick_params(labelsize=7, colors="#555", length=3)


def plot_execution_trajectories(traj_frames, horizon, env_name, step_num, output_path):
    n_episodes = len(traj_frames)
    n_steps = horizon + 1
    n_cols = min(n_steps, 6)
    t_indices = np.round(np.linspace(0, n_steps - 1, n_cols)).astype(int)

    cell = 1.3
    real_color = "#4878d0"
    exec_color = "#ee854a"

    fig = plt.figure(figsize=(n_cols * cell + 0.7, n_episodes * 2 * cell + 0.8))
    fig.patch.set_facecolor("white")

    outer = fig.add_gridspec(n_episodes, 1, hspace=0.18, left=0.06, right=0.99,
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


def plot_dist_hist(ax, opt, real, mean_opt, mean_real, xlabel, title):
    style_ax(ax)
    ax.tick_params(labelsize=8)
    lo = min(min(opt), min(real)) * 0.97
    hi = max(max(opt), max(real)) * 1.03
    bins = np.linspace(lo, hi, 40)
    ax.hist(real, bins=bins, color="#6acc65", alpha=0.7, edgecolor="white",
            linewidth=0.4, zorder=2, label="real traj (baseline)")
    ax.hist(opt,  bins=bins, color="#8172b2", alpha=0.7, edgecolor="white",
            linewidth=0.4, zorder=3, label="opt plan")
    ax.axvline(mean_opt,  color="#8172b2", linewidth=1.2, linestyle="--",
               zorder=4, label=f"opt mean={mean_opt:.3f}")
    ax.axvline(mean_real, color="#6acc65", linewidth=1.2, linestyle="--",
               zorder=4, label=f"real mean={mean_real:.3f}")
    ax.set_title(title, fontsize=10, color="#222")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)


def plot_success_curves(ax, metrics, eps_thresholds):
    style_ax(ax)
    ax.tick_params(labelsize=8)
    eps = list(eps_thresholds)
    series = []
    if "success_rate_latent" in metrics:
        lat = metrics["success_rate_latent"]
        series.append(("latent (opt)",  lat["opt"]["rates"],  "#8172b2", "-"))
        series.append(("latent (real)", lat["real"]["rates"], "#8172b2", "--"))
    if "success_rate_state" in metrics:
        sta = metrics["success_rate_state"]
        series.append(("state (opt)",   sta["opt"]["rates"],  "#dd8452", "-"))
        series.append(("state (real)",  sta["real"]["rates"], "#dd8452", "--"))
    for label, rates, color, linestyle in series:
        ax.plot(eps, rates, color=color, linestyle=linestyle, linewidth=1.8,
                marker="o", markersize=3, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"threshold $\varepsilon$ (log)", fontsize=9)
    ax.set_ylabel("success rate (dist < ε)", fontsize=9)
    ax.set_title("Success rate vs threshold ε  (powers of two from 0.001)",
                 fontsize=10, color="#222")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")


def plot_distributions(result, output_path, planner_name, eps_thresholds):
    episodes = result["per_episode"]
    metrics = result["metrics"]

    if "exec_dist" not in metrics:
        return

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2],
                          hspace=0.55, wspace=0.28)

    plot_dist_hist(
        fig.add_subplot(gs[0, 0]),
        opt=[ep["exec_dist"] for ep in episodes],
        real=[ep["exec_dist_real"] for ep in episodes],
        mean_opt=metrics["exec_dist"]["mean"],
        mean_real=metrics["exec_dist_real"]["mean"],
        xlabel=r"$\|\hat{z}_T - z_T\|_{\mathrm{RMS}}$  (latent)",
        title="Latent distance",
    )
    plot_dist_hist(
        fig.add_subplot(gs[0, 1]),
        opt=[ep["state_dist"] for ep in episodes],
        real=[ep["state_dist_real"] for ep in episodes],
        mean_opt=metrics["state_dist"]["mean"],
        mean_real=metrics["state_dist_real"]["mean"],
        xlabel=r"$\|s_{\mathrm{exec}} - s_{\mathrm{target}}\|_2$  (state vector)",
        title="State-vector distance",
    )

    ax_curve = fig.add_subplot(gs[1, :])
    plot_success_curves(ax_curve, metrics, eps_thresholds)

    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {plot_path}")
