import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from einops import rearrange
from pytorch_optimizer import Muon
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from jepa.datasets.video_dataset import build_dali_iterators
from jepa.losses.lejepa import EppsPulley, SlicingUnivariateTest
from jepa.losses.drifting import compute_V
from jepa.models.jepa import JEPA
from jepa.utils.distributed import all_gather, setup_distributed
from jepa.utils.helpers import MeanMetric, compute_alpha, rankme, spectrum
from jepa.utils.scheduler import TrapezoidSchedule


def compute_training_metrics(result, loss):
    """Compute all training metrics."""

    metrics = {}

    metrics["state_norm"] = result["state"].norm(dim=-1).mean()
    metrics["pred_norm"] = result["pred"].norm(dim=-1).mean()

    return metrics


def get_loss_fn(config):
    stats_test = EppsPulley()
    loss_fn = SlicingUnivariateTest(stats_test, 1024)

    # def drifting_loss(pred, state, temp=0.2):
    #     clamped_pred = [
    #         torch.cat((state[:, : t + 1], pred[:, t : t + 1]), dim=1)
    #         for t in range(pred.shape[1])
    #     ]
    #     clamped_pred = [rearrange(x, "b ... -> b (...)") for x in clamped_pred]
    #     target = [state[:, : t + 1] for t in range(1, state.shape[1])]
    #     target = [rearrange(x, "b ... -> b (...)") for x in target]
    #
    #     drifts = [compute_V(x, y, x, 0.2) for x, y in zip(clamped_pred, target)]
    #
    #     mse_loss = torch.stack(
    #         [(p - (d + p).detach()).pow(2).mean() for p, d in zip(clamped_pred, drifts)]
    #     ).mean()
    #
    #     mse_loss = (state[:, 1:] - pred).pow(2).mean()
    #     return mse_loss

    def find_temperature(dist, target_ess=0.5, T_lo=1e-5, T_hi=1.0, n_iter=20):
        K = dist.shape[0]
        lo = dist.new_full(dist.shape[1:], T_lo)
        hi = dist.new_full(dist.shape[1:], T_hi)

        for _ in range(n_iter):
            mid = (lo + hi) / 2
            w = torch.softmax(-dist / mid.unsqueeze(0), dim=0)
            ess = 1.0 / (w.pow(2).sum(dim=0) * K)
            lo = torch.where(ess < target_ess, mid, lo)
            hi = torch.where(ess >= target_ess, mid, hi)

        return (lo + hi) / 2

    def free_energy(pred, target, target_ess=0.5):
        dist = (target.unsqueeze(0) - pred).pow(2).mean(dim=-1)
        temp = find_temperature(dist, target_ess=target_ess)
        w = torch.softmax(-dist / temp.unsqueeze(0), dim=0).detach()
        fe = (w * dist).sum(dim=0).mean()
        return fe, w, temp

    def compute_loss(result, step):
        lam = config["training"]["lambda"]

        target = rearrange(result["state"][:, 1:], "... n d -> (...) n d")
        pred = rearrange(result["pred"], "k b t n d -> k (b t) n d")

        mse_loss, w, t = free_energy(pred, target)

        h = torch.special.entr(w).sum(dim=0) / np.log(config["predictor"]["k"])
        ess = 1.0 / (w.pow(2).sum(dim=0) * w.shape[0])

        state = rearrange(result["state"], "... d -> (...) d")
        state_sigreg_loss = loss_fn(state)

        total_loss = lam * state_sigreg_loss + (1 - lam) * mse_loss

        return {
            "total": total_loss,
            "state_sigreg": state_sigreg_loss,
            "mse": mse_loss,
            "h": h.mean(),
            "ess": ess.mean(),
            "t": t.mean(),
        }

    return compute_loss


def log_progress(
    pbar,
    step,
    loss,
    metrics,
    stage="train",
):
    """Log training progress to progress bar and wandb."""
    # Update progress bar
    if pbar is not None:
        pbar.set_postfix(
            state_sigreg=loss["state_sigreg"].item(),
            mse=loss["mse"].item(),
            h=loss["h"].item(),
            ess=loss["ess"].item(),
        )
        pbar.update(1)

    metrics = {stage + "/" + k: v for k, v in metrics.items()}
    loss = {stage + "/" + k: v for k, v in loss.items()}
    # metrics = {
    #     k: wandb.Histogram(v) if v.numel() > 1 else v for k, v in metrics.items()
    # }

    # Log metrics to wandb
    if wandb.run:
        wandb.log({**loss, **metrics}, step=step)


def save_checkpoint(config, model, run_id, step):
    exp_id = None
    if wandb.run:
        exp_id = wandb.run.group or os.getenv("WANDB_RUN_GROUP", "default")
        run_id = wandb.run.id  # ensure W&B run id
    else:
        exp_id = os.getenv("WANDB_RUN_GROUP", "default")
        run_id = str(run_id)

    ckpt_dir = os.path.join("experiments", exp_id, "checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    name = "checkpoint.pth"
    torch.save(
        {
            "model": model.module.state_dict(),
            "config": config,
            "step": step + 1,
            "wandb_run_id": run_id,
        },
        os.path.join(ckpt_dir, name),
    )


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
        skip = []
        filter = [
            "pe",
            "registers",
            "token",
        ]
        muon_group = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and not any(k in name for k in filter)
            and not any(k in name for k in skip)
        ]
        adamw_group = [
            p
            for name, p in model.named_parameters()
            if (p.ndim < 2 or any(k in name for k in filter))
            and not any(k in name for k in skip)
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
def val_epoch(model, loader, loss_fn, step, max_steps=100):
    model.eval()

    mean_metrics = {}
    pbar = tqdm(loader, desc="Validation", leave=False)
    for i, batch in enumerate(pbar):
        x = batch[0]["data"]
        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x)

        loss = loss_fn(result, step)
        metrics = compute_training_metrics(result, loss)

        for k, v in itertools.chain(loss.items(), metrics.items()):
            if k not in mean_metrics:
                mean_metrics[k] = MeanMetric()
            mean_metrics[k].update(v)

        if i > max_steps:
            break

    mean_metrics = {k: v.avg for k, v in mean_metrics.items()}
    log_progress(None, step, {}, mean_metrics, stage="val")


def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    opt_step,
    config,
    rank=0,
):
    model.train()
    optimizer.zero_grad()

    total_steps = config["training"]["total_steps"]
    val_fraction = config["training"]["val_fraction"]
    checkpoint_fraction = config["training"]["ckpt_fraction"]

    val_interval = int(total_steps * val_fraction)
    ckpt_interval = int(total_steps * checkpoint_fraction)

    print(f"Valdiation interval: {val_interval}")
    print(f"Validation steps: {len(val_loader)}")

    train_iter = iter(train_loader)
    pbar = tqdm(total=total_steps, initial=1, desc="Training", dynamic_ncols=True)
    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_loader.reset()
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch[0]["data"]
        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x)
        loss = loss_fn(result, step)

        opt_step(loss["total"], model, optimizer)

        metrics = compute_training_metrics(result, loss)
        log_progress(pbar, step, loss, metrics)

        if val_interval > 0 and step % val_interval == 0 and step < total_steps:
            val_epoch(model, val_loader, loss_fn, step)

        if ckpt_interval > 0 and step % ckpt_interval == 0 and rank == 0:
            print("Saving checkpoint")
            if wandb.run:
                save_checkpoint(config, model, wandb.run.id, step)

    if rank == 0 and wandb.run:
        save_checkpoint(config, model, wandb.run.id, step)  # type: ignore

    val_epoch(model, val_loader, loss_fn, step)  # type: ignore
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # torch.autograd.set_detect_anomaly(True)

    train_conf = config["training"]
    data_conf = config["data"]

    seed = train_conf.get("seed", -1)
    dist_enabled = train_conf.get("distributed", False)
    wandb_mode = train_conf.get("wandb", "online")

    torch.manual_seed(seed)
    random.seed(seed)

    model = JEPA(
        config["encoder"],
        config["predictor"],
    )

    local_rank, global_rank, world_size = 0, 0, 1
    if dist_enabled:
        local_rank, global_rank, world_size = setup_distributed()
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])  # , find_unused_parameters=True)
        model.compile(mode="default")

    loss_fn = get_loss_fn(config)

    optimizer, opt_step = init_opt(train_conf, model)

    train_loader, val_loader = build_dali_iterators(
        data_conf, local_rank, global_rank, world_size, seed=seed
    )

    print(f"Steps per epoch: {len(train_loader)}")

    if global_rank == 0:
        wandb_init_args = {
            "project": config["training"]["project"],
            "config": config,
            "mode": wandb_mode,
        }
        wandb.init(**wandb_init_args)

    train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        opt_step,
        config,
        rank=global_rank,
    )

    if dist_enabled:
        torch.cuda.empty_cache()
        dist.destroy_process_group()

    print("Finished training")


if __name__ == "__main__":
    main()
