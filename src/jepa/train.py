import argparse
import itertools
import json
import os
import random
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from pytorch_optimizer import Muon
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from jepa.datasets.builder import build_iterators
from jepa.losses.sigreg import EppsPulley, SlicingUnivariateTest
from jepa.models.jepa import JEPA
from jepa.utils.distributed import setup_distributed
from jepa.utils.helpers import MeanMetric, rankme, spectrum
from jepa.utils.run_log import (
    MetricsLogger,
    RunStatus,
    slurm_job_id_from_env,
    to_scalar_dict,
)
from jepa.utils.scheduler import TrapezoidSchedule


ACTION_PLOT_METRIC_KEYS = {
    "action_pred_class",
    "action_target_class",
}


def unpack_batch(batch):
    if isinstance(batch, list):
        if len(batch) != 1:
            raise ValueError("Expected a single batch item from DALI.")
        batch = batch[0]

    if not isinstance(batch, dict):
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    return batch["data"], batch.get("actions")


def maybe_reset_loader(loader, epoch):
    if hasattr(loader, "reset"):
        loader.reset()

    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def current_git_hash():
    env_hash = os.environ.get("SOURCE_GIT_HASH")
    if env_hash:
        return env_hash
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


@torch.no_grad()
def plot_error_spectrum(result):
    """Log-log eigenvalue spectra of states, predictions, and prediction errors."""
    state = result["state"].float()
    pred = result.get("pred")
    if pred is None:
        return None
    pred = pred.float()
    target = state[:, 1:]
    # no-bottleneck predictor returns length T; drop the trailing position.
    if pred.shape[1] == state.shape[1]:
        pred = pred[:, :-1]

    state_flat = rearrange(state, "... d -> (...) d")
    pred_flat = rearrange(pred, "... d -> (...) d")
    error_flat = rearrange(pred - target, "... d -> (...) d")

    state_eigs = spectrum(state_flat)
    pred_eigs = spectrum(pred_flat)
    error_eigs = spectrum(error_flat)
    state_eigs = state_eigs[state_eigs > 0].cpu().numpy()
    pred_eigs = pred_eigs[pred_eigs > 0].cpu().numpy()
    error_eigs = error_eigs[error_eigs > 0].cpu().numpy()

    if len(state_eigs) < 2 or len(pred_eigs) < 2 or len(error_eigs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.loglog(range(1, len(state_eigs) + 1), state_eigs, linewidth=2.2, color="#a8d5ba", label="state")
    ax.loglog(range(1, len(pred_eigs) + 1), pred_eigs, linewidth=2.2, color="#aec6e4", label="prediction")
    ax.loglog(range(1, len(error_eigs) + 1), error_eigs, linewidth=2.2, color="#f5b394", label="error")
    ax.set_xlabel("Index", fontsize=10, color="#444")
    ax.set_ylabel("Eigenvalue", fontsize=10, color="#444")
    ax.set_title("Covariance spectrum", fontsize=11, color="#222", pad=10)
    ax.grid(True, which="both", linestyle="-", linewidth=0.4, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbb")
    ax.tick_params(labelsize=9, colors="#555", length=3)
    ax.legend(fontsize=9, frameon=False, loc="lower left", handlelength=1.8, labelcolor="#333")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def compute_training_metrics(result):
    """Compute all training metrics."""

    metrics = {}

    metrics["state_norm"] = result["state"].norm(dim=-1).mean()
    pred = result.get("pred")
    if pred is not None:
        metrics["pred_norm"] = pred.norm(dim=-1).mean()
    if "mu" in result:
        metrics["vae_mu_rms"] = result["mu"].pow(2).mean().sqrt()
    if "logvar" in result:
        metrics["vae_logvar_mean"] = result["logvar"].mean()

    state = rearrange(result["state"], "... d -> (...) d")
    eigvals = spectrum(state.float())
    metrics["rankme"] = rankme(eigvals)

    # temporal latent path straightness (Eq. 9, LeWorldModel)
    z = result["state"].float()  # (B, T, N, D)
    v = z[:, 1:] - z[:, :-1]  # (B, T-1, N, D)
    if v.shape[1] >= 2:
        cos = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)  # (B, T-2, N)
        metrics["path_straightness"] = cos.mean()

    return metrics


def update_metric_store(store, key, value):
    if key in ACTION_PLOT_METRIC_KEYS:
        value = value.detach().cpu()
        if key not in store:
            store[key] = value.clone()
        else:
            store[key] = torch.cat((store[key], value), dim=0)
        return

    if key not in store:
        store[key] = MeanMetric()
    store[key].update(value)


def finalize_metric_store(store):
    metrics = {}
    for key, value in store.items():
        if key in ACTION_PLOT_METRIC_KEYS:
            metrics[key] = value
        else:
            metrics[key] = value.avg
    return metrics


def split_scalar_and_plot_metrics(values):
    scalars = {}
    plot_data = {}
    for key, value in values.items():
        if key in ACTION_PLOT_METRIC_KEYS:
            plot_data[key] = value
        else:
            scalars[key] = value
    return scalars, plot_data


def build_action_confusion_matrix(stage, values):
    if not wandb.run or stage != "val":
        return {}

    pred = values.get("action_pred_class")
    target = values.get("action_target_class")
    if pred is None or target is None:
        return {}

    num_classes = int(torch.cat((pred, target)).max().item()) + 1
    class_names = [f"action_{idx}" for idx in range(num_classes)]
    return {
        f"{stage}/action_confusion": wandb.plot.confusion_matrix(
            preds=pred.tolist(),
            y_true=target.tolist(),
            class_names=class_names,
            title=f"{stage.capitalize()} Action Confusion Matrix",
        )
    }


def get_loss_fn(config):
    stats_test = EppsPulley()
    loss_fn = SlicingUnivariateTest(stats_test, 1024)
    action_config = config.get("action_decoder", {})
    action_enabled = bool(action_config.get("enabled", False))
    action_type = action_config.get("action_type", "continuous")
    sigreg_marginal = config.get("training", {}).get("sigreg_marginal", "full")
    kl_beta = float(config.get("predictor", {}).get("kl_beta", 1.0))
    detach_cond_target = config.get("training", {}).get("detach_cond_target", False)

    def compute_action_loss(result, actions):
        if not action_enabled:
            return {}
        if actions is None:
            raise ValueError(
                "Action decoder is enabled, but the batch does not contain actions."
            )

        if action_type == "continuous":
            log_prob = result.get("action_log_prob")
            if log_prob is None:
                raise ValueError(
                    "Continuous action decoder did not return a log-probability. "
                    "Ensure actions are passed into the forward pass."
                )
            nll = -log_prob.mean()
            return {"action": nll, "action_nll": nll}

        if action_type == "discrete":
            pred = result.get("action_pred")
            if pred is None:
                raise ValueError(
                    "Discrete action decoder did not return logits."
                )
            target = actions.long()
            logits = rearrange(pred, "b t c -> (b t) c")
            target = rearrange(target, "b t -> (b t)")
            pred_class = logits.argmax(dim=-1)
            is_correct = pred_class == target
            action_loss = F.cross_entropy(logits, target)
            action_acc = is_correct.float().mean()
            return {
                "action": action_loss,
                "action_acc": action_acc,
                "action_pred_class": pred_class,
                "action_target_class": target,
            }

        raise ValueError(f"Unknown action decoder type: {action_type}")

    def compute_rollout_diagnostic(result, actions):
        if not action_enabled or actions is None:
            return {}
        if action_type == "discrete":
            pred = result.get("rollout_action_pred")
            if pred is None:
                return {}
            target = actions.long()
            logits = rearrange(pred, "b t c -> (b t) c")
            target = rearrange(target, "b t -> (b t)")
            acc = (logits.argmax(dim=-1) == target).float().mean()
            return {"rollout_action_acc": acc}
        if action_type == "continuous":
            log_prob = result.get("rollout_action_log_prob")
            if log_prob is None:
                return {}
            return {"rollout_action_nll": -log_prob.mean()}
        return {}

    def compute_loss(result, actions, step):
        lam = config["training"]["lambda"]
        target = result["state"][:, 1:].float()
        pred = result["pred"].float()
        # no-bottleneck predictor returns length T; take [:, :-1] to align with target.
        if pred.shape[1] == result["state"].shape[1]:
            pred = pred[:, :-1]

        # sigreg on encoder states
        if sigreg_marginal == "time":
            state = rearrange(result["state"], "b t n d -> t n b d")
        elif sigreg_marginal == "shuffled_time":
            s = result["state"]
            B, T, N, D = s.shape
            perm = torch.rand(B, T, device=s.device).argsort(dim=1)[:, :, None, None].expand(-1, -1, N, D)
            state = rearrange(s.gather(1, perm), "b t n d -> t n b d")
        else:
            state = rearrange(result["state"], "... d -> (...) d").unsqueeze(0)
        state_sigreg_loss = loss_fn(state.float())
        total_loss = lam * state_sigreg_loss
        loss_dict = {"state_sigreg": state_sigreg_loss}

        mse_target = target.detach() if detach_cond_target else target
        mse_loss = F.mse_loss(pred, mse_target)
        total_loss = total_loss + mse_loss
        loss_dict["mse"] = mse_loss

        if "kl" in result:
            kl = result["kl"]
            total_loss = total_loss + kl_beta * kl
            loss_dict["kl"] = kl

        action_metrics = compute_action_loss(result, actions)
        if action_metrics:
            total_loss = total_loss + action_metrics["action"]

        loss_dict["total"] = total_loss
        loss_dict.update(action_metrics)
        loss_dict.update(compute_rollout_diagnostic(result, actions))
        return loss_dict

    return compute_loss


def log_progress(
    pbar,
    step,
    loss,
    metrics,
    stage="train",
):
    """Log training progress to progress bar and wandb."""
    scalar_loss, loss_plot_data = split_scalar_and_plot_metrics(loss)
    scalar_metrics, metric_plot_data = split_scalar_and_plot_metrics(metrics)

    # Update progress bar
    if pbar is not None:
        postfix = dict(
            state_sigreg=scalar_loss["state_sigreg"].item(),
        )
        if "mse" in scalar_loss:
            postfix["mse"] = scalar_loss["mse"].item()
        if "kl" in scalar_loss:
            postfix["kl"] = scalar_loss["kl"].item()
        if "action" in scalar_loss:
            postfix["action"] = scalar_loss["action"].item()
        if "action_acc" in scalar_loss:
            postfix["action_acc"] = scalar_loss["action_acc"].item()
        pbar.set_postfix(**postfix)
        pbar.update(1)

    metrics = {stage + "/" + k: v for k, v in scalar_metrics.items()}
    loss = {stage + "/" + k: v for k, v in scalar_loss.items()}

    # Log metrics to wandb
    if wandb.run:
        plot_data = {}
        plot_data.update(build_action_confusion_matrix(stage, loss_plot_data))
        plot_data.update(build_action_confusion_matrix(stage, metric_plot_data))
        wandb.log({**loss, **metrics, **plot_data}, step=step)


def save_checkpoint(config, model, run_id, step):
    module = model.module if hasattr(model, "module") else model

    if wandb.run:
        exp_id = wandb.run.group or os.getenv("WANDB_RUN_GROUP", "default")
        run_id = wandb.run.id  # ensure W&B run id
    else:
        exp_id = os.getenv("WANDB_RUN_GROUP", "default")
        run_id = str(run_id)

    workdir = os.environ.get("SOURCE_WORKDIR", os.getcwd())
    ckpt_dir = os.path.join(workdir, "research_results", exp_id, "checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    step_name = f"step_{step:08d}.pth"
    step_path = os.path.join(ckpt_dir, step_name)
    torch.save(
        {
            "model": module.state_dict(),
            "config": config,
            "step": step + 1,
            "wandb_run_id": run_id,
        },
        step_path,
    )

    # Atomically point checkpoint.pth at the newest step-stamped file.
    latest_path = os.path.join(ckpt_dir, "checkpoint.pth")
    tmp_link = latest_path + ".tmp"
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(step_name, tmp_link)
    os.replace(tmp_link, latest_path)


def init_opt(config, model):
    warmup_steps = int(config.get("warmup_steps", 0))
    total_steps = int(config["total_steps"])

    start_lr = float(config["lr_start"])
    lr = float(config["lr"])
    cooldown_frac = float(config["cooldown_frac"])
    final_lr_frac = float(config["final_lr_frac"])
    wd = float(config["wd"])
    clip_grad = float(config["clip_grad"])

    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))  # type: ignore
    elif config["optimizer"] == "muon":
        adamw_keys = [
            "pe",
            "registers",
            "token",
            "projector",
            "flow",
            "bottleneck_head",
        ]
        muon_group = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and not any(k in name for k in adamw_keys)
        ]
        adamw_group = [
            p
            for name, p in model.named_parameters()
            if p.ndim < 2 or any(k in name for k in adamw_keys)
        ]
        param_groups = [
            dict(params=muon_group, use_muon=True, lr=lr, weight_decay=wd),
            dict(
                params=adamw_group,
                use_muon=False,
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=wd,
            ),
        ]
        optimizer = Muon(param_groups, use_adjusted_lr=True)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    scaler = torch.amp.GradScaler()  # type: ignore
    lr_scheduler = TrapezoidSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=lr,
        total_steps=total_steps,
        cooldown_frac=cooldown_frac,
        final_lr_frac=final_lr_frac,
    )

    def optimization_step(loss, model, optimizer):
        """Perform optimization step with gradient scaling and clipping."""

        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        lr_scheduler.step()

    return optimizer, optimization_step


@torch.no_grad()
def val_epoch(model, loader, loss_fn, step, max_steps=500):
    model.eval()

    mean_metrics = {}
    pbar = tqdm(loader, desc="Validation", leave=False)
    for i, batch in enumerate(pbar):
        x, actions = unpack_batch(batch)
        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x, actions)

        loss = loss_fn(result, actions, step)
        metrics = compute_training_metrics(result)

        for k, v in itertools.chain(loss.items(), metrics.items()):
            update_metric_store(mean_metrics, k, v)

        if i > max_steps:
            break

    mean_metrics = finalize_metric_store(mean_metrics)
    log_progress(None, step, {}, mean_metrics, stage="val")
    if not dist.is_initialized() or dist.get_rank() == 0:
        summary = " ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(mean_metrics.items())
            if isinstance(v, (int, float))
        )
        print(f"[val step={step}] {summary}", flush=True)
    model.train()
    return mean_metrics


def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    opt_step,
    config,
    rank=0,
    status: RunStatus | None = None,
    metrics_logger: MetricsLogger | None = None,
):
    model.train()
    optimizer.zero_grad()

    total_steps = config["training"]["total_steps"]
    val_fraction = config["training"]["val_fraction"]
    checkpoint_fraction = config["training"]["ckpt_fraction"]
    val_max_steps = int(config["training"].get("val_max_steps", 500))
    final_val_max_steps = int(config["training"].get("final_val_max_steps", 2000))

    val_interval = int(total_steps * val_fraction)
    ckpt_interval = int(total_steps * checkpoint_fraction)

    print(f"Validation interval: {val_interval}")
    print(f"Validation steps: {len(val_loader)}")

    train_iter = iter(train_loader)
    epoch = 0
    pbar = tqdm(total=total_steps, initial=1, desc="Training", dynamic_ncols=True)
    for step in range(1, total_steps + 1):
        if status is not None:
            status.note_step(step)
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            maybe_reset_loader(train_loader, epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, actions = unpack_batch(batch)

        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x, actions)
        loss = loss_fn(result, actions, step)

        opt_step(loss["total"], model, optimizer)

        metrics = compute_training_metrics(result)
        log_progress(pbar, step, loss, metrics)

        if step % 100 == 0:
            if wandb.run:
                img = plot_error_spectrum(result)
                if img is not None:
                    wandb.log({"train/error_spectrum": img}, step=step)
            if metrics_logger is not None:
                scalar = to_scalar_dict(loss, prefix="loss/")
                scalar.update(to_scalar_dict(metrics, prefix="metric/"))
                metrics_logger.log(step=step, stage="train", metrics=scalar)
                if status is not None:
                    status.heartbeat(step, scalar)

        if val_interval > 0 and step % val_interval == 0 and step < total_steps:
            val_metrics = val_epoch(model, val_loader, loss_fn, step, max_steps=val_max_steps)
            if metrics_logger is not None:
                scalar = to_scalar_dict(val_metrics, prefix="val/")
                metrics_logger.log(step=step, stage="val", metrics=scalar)
                if status is not None:
                    status.heartbeat(step, scalar)

        if ckpt_interval > 0 and step % ckpt_interval == 0 and rank == 0:
            print("Saving checkpoint")
            save_checkpoint(config, model, wandb.run.id if wandb.run else "local", step)

    if rank == 0:
        save_checkpoint(config, model, wandb.run.id if wandb.run else "local", step)  # type: ignore

    final_metrics = val_epoch(model, val_loader, loss_fn, step, max_steps=final_val_max_steps)  # type: ignore
    if metrics_logger is not None:
        scalar = to_scalar_dict(final_metrics, prefix="val/")
        metrics_logger.log(step=step, stage="val", metrics=scalar)  # type: ignore
        if status is not None:
            status.heartbeat(step, scalar)  # type: ignore
    pbar.close()
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    train_conf = config["training"]
    data_conf = dict(config["data"])
    data_conf["sequence_length"] = config["context"] + 1

    seed = train_conf.get("seed", -1)
    wandb_mode = train_conf.get("wandb", "online")

    torch.manual_seed(seed)
    random.seed(seed)

    model = JEPA(
        config["encoder"],
        config["predictor"],
        config.get("action_decoder"),
    )

    local_rank, global_rank, world_size = setup_distributed()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)

    loss_fn = get_loss_fn(config)

    optimizer, opt_step = init_opt(train_conf, model)

    train_loader, val_loader = build_iterators(
        data_conf, local_rank, global_rank, world_size, seed=seed
    )

    print(f"Steps per epoch: {len(train_loader)}")

    cfg_path = os.path.abspath(args.config)
    sweep_dir = os.path.dirname(os.path.dirname(cfg_path))
    run_id = os.path.splitext(os.path.basename(cfg_path))[0]

    status: RunStatus | None = None
    metrics_logger: MetricsLogger | None = None
    if global_rank == 0:
        wandb_init_args = {
            "project": config["training"]["project"],
            "config": config,
            "mode": wandb_mode,
        }
        wandb.init(**wandb_init_args)
        status = RunStatus(
            sweep_dir,
            run_id,
            total_steps=int(train_conf["total_steps"]),
            slurm_job_id=slurm_job_id_from_env(),
            wandb_run_id=wandb.run.id if wandb.run else None,
        )
        metrics_logger = MetricsLogger(sweep_dir, run_id)

    t0 = time.time()
    last_step = 0
    try:
        final_metrics = train(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            opt_step,
            config,
            rank=global_rank,
            status=status,
            metrics_logger=metrics_logger,
        )
        last_step = int(train_conf["total_steps"])
    except BaseException as exc:
        if status is not None:
            status.crashed(None, exc)
        raise
    train_time = time.time() - t0

    if global_rank == 0 and final_metrics is not None:
        results_dir = os.path.join(sweep_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        metrics = to_scalar_dict(final_metrics)
        result = {
            "config": config,
            "metrics": metrics,
            "train_time_seconds": train_time,
            "git_hash": current_git_hash(),
        }
        with open(os.path.join(results_dir, f"{run_id}.json"), "w") as f:
            json.dump(result, f, indent=2)
        if status is not None:
            status.done(last_step)

    torch.cuda.empty_cache()
    dist.destroy_process_group()

    print("Finished training")


if __name__ == "__main__":
    main()
