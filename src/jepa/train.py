import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from einops import rearrange
from pytorch_optimizer import Muon
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from jepa.datasets.builder import build_iterators
from jepa.losses.lejepa import EppsPulley, SlicingUnivariateTest
from jepa.models.jepa import JEPA
from jepa.utils.distributed import setup_distributed
from jepa.utils.helpers import MeanMetric, rankme, spectrum
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


def move_to_device(x, device):
    if x is None:
        return None
    return x.to(device, non_blocking=True)


def get_model_device(model):
    return next(model.parameters()).device


def maybe_reset_loader(loader, epoch):
    if hasattr(loader, "reset"):
        loader.reset()

    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def compute_training_metrics(result):
    """Compute all training metrics."""

    metrics = {}

    metrics["state_norm"] = result["state"].norm(dim=-1).mean()
    metrics["pred_norm"] = result["pred"].norm(dim=-1).mean()

    state = rearrange(result["state"], "... d -> (...) d")
    eigvals = spectrum(state.float())
    metrics["rankme"] = rankme(eigvals)

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
    training_config = config["training"]
    action_enabled = bool(action_config.get("enabled", False))
    action_type = action_config.get("action_type", "continuous")
    loss_type = training_config.get("loss", "free_energy")
    temp_mode = training_config.get("temp_mode", "fixed")
    fixed_temp = float(training_config.get("temp", 0.01))
    target_ess = float(training_config.get("target_ess", 0.5))

    if loss_type not in {"free_energy", "imle"}:
        raise ValueError(f"Unknown loss type: {loss_type}")
    if temp_mode not in {"fixed", "ess", "profiled"}:
        raise ValueError(f"Unknown temp_mode: {temp_mode}")
    if fixed_temp <= 0:
        raise ValueError("training.temp must be positive.")

    def find_temperature_ess(dist, target_ess=0.5, T_lo=1e-5, T_hi=1.0, n_iter=20):
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

    def find_temperature_profiled(dist, n_iter=10):
        T = 2.0 * dist.mean(dim=0).clamp(min=1e-8)
        for _ in range(n_iter):
            w = torch.softmax(-dist / T.unsqueeze(0), dim=0)
            T = 2.0 * (w * dist).sum(dim=0).clamp(min=1e-8)
        return T.detach()

    def free_energy(pred, target):
        dist = (target.float().unsqueeze(0) - pred.float()).pow(2).mean(dim=(-1, -2))
        if temp_mode == "ess":
            temp = find_temperature_ess(dist, target_ess=target_ess)
        elif temp_mode == "profiled":
            temp = find_temperature_profiled(dist)
        else:
            temp = dist.new_full(dist.shape[1:], fixed_temp)
        w = torch.softmax(-dist / temp.unsqueeze(0), dim=0).detach()
        fe = (w * dist).sum(dim=0).mean()
        return fe, w, temp

    def imle(pred, target):
        dist = (target.unsqueeze(0) - pred).pow(2).mean(dim=-1)
        idx = dist.mean(dim=-1).argmin(dim=0)
        w = torch.zeros_like(dist)
        w[idx, torch.arange(dist.shape[1], device=dist.device)] = 1.0
        return (w * dist).sum(dim=0).mean(), w

    def compute_action_loss(result, actions):
        if not action_enabled:
            return {}
        if actions is None:
            raise ValueError(
                "Action decoder is enabled, but the batch does not contain actions."
            )

        pred = result.get("action_pred")
        if pred is None:
            raise ValueError(
                "Action decoder is enabled, but the model did not return action predictions."
            )

        if action_type == "continuous":
            target = actions.to(dtype=pred.dtype)
            action_loss = F.mse_loss(pred, target)
            action_l1 = F.l1_loss(pred, target)
            return {
                "action": action_loss,
                "action_l1": action_l1,
            }

        if action_type == "discrete":
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

    def compute_loss(result, actions, step):
        lam = config["training"]["lambda"]

        target = rearrange(result["state"][:, 1:], "... n d -> (...) n d").float()
        pred = rearrange(result["pred"], "k b t n d -> k (b t) n d").float()

        if loss_type == "free_energy":
            mse_loss, w, t = free_energy(pred, target)
        else:
            mse_loss, w = imle(pred, target)

        min_mse = (
            (target.unsqueeze(0) - pred)
            .pow(2)
            .mean(dim=(-1, -2))
            .min(dim=0)
            .values.mean()
        )

        h = torch.special.entr(w).sum(dim=0) / np.log(config["predictor"]["k"])
        ess = 1.0 / (w.pow(2).sum(dim=0) * w.shape[0])

        state = rearrange(result["state"], "... d -> (...) d")
        state_sigreg_loss = loss_fn(state.float())

        total_loss = lam * state_sigreg_loss + (1 - lam) * mse_loss
        action_metrics = compute_action_loss(result, actions)
        if action_metrics:
            total_loss = total_loss + action_metrics["action"]

        loss_dict = {
            "total": total_loss,
            "state_sigreg": state_sigreg_loss,
            "mse": mse_loss,
            "min_mse": min_mse,
            "h": h.mean(),
            "ess": ess.mean(),
        }
        if loss_type == "free_energy":
            loss_dict["t"] = t.mean()
        loss_dict.update(action_metrics)
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
            mse=scalar_loss["mse"].item(),
            h=scalar_loss["h"].item(),
            ess=scalar_loss["ess"].item(),
        )
        if "action" in scalar_loss:
            postfix["action"] = scalar_loss["action"].item()
        if "action_acc" in scalar_loss:
            postfix["action_acc"] = scalar_loss["action_acc"].item()
        pbar.set_postfix(**postfix)
        pbar.update(1)

    metrics = {stage + "/" + k: v for k, v in scalar_metrics.items()}
    loss = {stage + "/" + k: v for k, v in scalar_loss.items()}
    # metrics = {
    #     k: wandb.Histogram(v) if v.numel() > 1 else v for k, v in metrics.items()
    # }

    # Log metrics to wandb
    if wandb.run:
        plot_data = {}
        plot_data.update(build_action_confusion_matrix(stage, loss_plot_data))
        plot_data.update(build_action_confusion_matrix(stage, metric_plot_data))
        wandb.log({**loss, **metrics, **plot_data}, step=step)


def save_checkpoint(config, model, run_id, step):
    module = model.module if hasattr(model, "module") else model

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
            "model": module.state_dict(),
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
def val_epoch(model, loader, loss_fn, step, max_steps=500):
    model.eval()

    mean_metrics = {}
    pbar = tqdm(loader, desc="Validation", leave=False)
    for i, batch in enumerate(pbar):
        x, actions = unpack_batch(batch)
        device = get_model_device(model)
        x = move_to_device(x, device)
        actions = move_to_device(actions, device)
        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x)

        loss = loss_fn(result, actions, step)
        metrics = compute_training_metrics(result)

        for k, v in itertools.chain(loss.items(), metrics.items()):
            update_metric_store(mean_metrics, k, v)

        if i > max_steps:
            break

    mean_metrics = finalize_metric_store(mean_metrics)
    log_progress(None, step, {}, mean_metrics, stage="val")
    model.train()


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
    epoch = 0
    pbar = tqdm(total=total_steps, initial=1, desc="Training", dynamic_ncols=True)
    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            maybe_reset_loader(train_loader, epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, actions = unpack_batch(batch)
        device = get_model_device(model)
        x = move_to_device(x, device)
        actions = move_to_device(actions, device)

        with torch.amp.autocast("cuda"):  # type: ignore
            result = model(x)
        loss = loss_fn(result, actions, step)

        opt_step(loss["total"], model, optimizer)

        metrics = compute_training_metrics(result)
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
        config.get("action_decoder"),
    )

    local_rank, global_rank, world_size = 0, 0, 1
    if dist_enabled:
        local_rank, global_rank, world_size = setup_distributed()
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])  # , find_unused_parameters=True)
        model.compile(mode="default")
    else:
        model = model.cuda()

    loss_fn = get_loss_fn(config)

    optimizer, opt_step = init_opt(train_conf, model)

    train_loader, val_loader = build_iterators(
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
