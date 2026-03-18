import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import yaml
from tqdm import tqdm
from pytorch_optimizer import Muon

from jepa.datasets.image_dataset import build_dali_iterators
from jepa.models.attentive_probe import AttentiveProbe
from jepa.models.model import JEPA
from jepa.utils.distributed import setup_distributed
from jepa.utils.helpers import SmoothedValue
from jepa.utils.scheduler import WarmupCosineSchedule


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Attentive Probe with DDP")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file for training",
    )
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def setup_model(args, config, local_rank):
    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    pretrain_config = checkpoint["config"]

    state_dict = checkpoint["model"]

    # Initialize model
    model = JEPA(
        pretrain_config["encoder"],
        pretrain_config["predictor"],
        pretrain_config["transform_model"],
    ).to(local_rank)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)

    # Initialize Attentive Probe
    attentive_probe = AttentiveProbe(
        model.dim, config["heads"], config["num_classes"], config["features"]
    ).to(local_rank)
    attentive_probe = DDP(
        attentive_probe, device_ids=[local_rank], output_device=local_rank
    )

    return model, attentive_probe, pretrain_config


def setup_data_loaders(config, local_rank, world_size):
    return build_dali_iterators(config["data"], local_rank, local_rank, world_size)


def train_one_epoch(
    epoch,
    model,
    attentive_probe,
    train_loader,
    optimizer,
    lr_scheduler,
    loss_fn,
    config,
    global_rank,
):
    model.train()

    accumulation_steps = config.get("accumulation_steps", 1)
    train_smoothed_accuracy = SmoothedValue()

    if global_rank == 0:
        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Training Epoch {epoch + 1}/{config['epochs']}",
        )

    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        x = batch[0]["data"].cuda(non_blocking=True)
        y = batch[0]["label"].cuda(non_blocking=True).squeeze().long()

        with torch.inference_mode(), torch.amp.autocast("cuda"):  # type: ignore
            features = model.forward_features(x)[config["features"]]

        pred = attentive_probe(features.clone()).squeeze(1)
        pred = pred.float()
        loss = loss_fn(pred, y) / accumulation_steps  # Normalize loss

        acc = (pred.argmax(dim=1) == y).float().mean()
        train_smoothed_accuracy.update(acc.item())

        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        if global_rank == 0:
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                wandb.log(
                    {
                        "eval/train/loss": loss.item() * accumulation_steps,
                        "eval/train/acc": acc.item(),
                        "eval/train/smoothed_acc": train_smoothed_accuracy(),
                    }
                )
            progress_bar.set_postfix(  # type: ignore
                {
                    "loss": loss.item() * accumulation_steps,
                    "acc": train_smoothed_accuracy(),
                }
            )  # type: ignore
            progress_bar.update(1)  # type: ignore

    if global_rank == 0:
        progress_bar.close()  # type: ignore


def validate(model, attentive_probe, val_loader, loss_fn, config, local_rank):
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0

    if local_rank == 0:
        progress_bar = tqdm(total=len(val_loader), desc="Validation")

    with torch.inference_mode():
        for batch in val_loader:
            x = batch[0]["data"].cuda(non_blocking=True)
            y = batch[0]["label"].cuda(non_blocking=True).squeeze().long()

            with torch.amp.autocast("cuda"):  # type: ignore
                features = model.forward_features(x)[config["features"]]
                pred = attentive_probe(features).squeeze(1)

            pred = pred.float()
            loss = loss_fn(pred, y)
            acc = (pred.argmax(dim=1) == y).float().mean()

            total_val_loss += loss.item()
            total_val_acc += acc.item()

            if local_rank == 0:
                progress_bar.update(1)  # type: ignore

    # Gather metrics from all processes
    total_val_loss_tensor = torch.tensor(total_val_loss).cuda()
    total_val_acc_tensor = torch.tensor(total_val_acc).cuda()
    num_batches_tensor = torch.tensor(len(val_loader)).cuda()

    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_val_acc_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        avg_val_loss = total_val_loss_tensor.item() / num_batches_tensor.item()
        avg_val_acc = total_val_acc_tensor.item() / num_batches_tensor.item()

        wandb.log(
            {
                "eval/val/loss": avg_val_loss,
                "eval/val/acc": avg_val_acc,
            }
        )
        progress_bar.close()  # type: ignore


def main():
    args = parse_args()
    config = load_config(args)
    epochs = int(config.get("epochs", 10))
    lr = float(config.get("lr", 1e-3))

    # Distributed setup
    local_rank, global_rank, world_size = setup_distributed()
    torch.cuda.set_device(local_rank)

    # Setup model and optimizer
    model, attentive_probe, pretrain_config = setup_model(args, config, local_rank)

    # filter = ["projection"]
    # muon_group = [
    #     p
    #     for name, p in attentive_probe.named_parameters()
    #     if (p.ndim >= 2 and name not in filter)
    # ]
    # adamw_group = [
    #     p
    #     for name, p in attentive_probe.named_parameters()
    #     if (p.ndim < 2 or name in filter)
    # ]
    # param_groups = [
    #     dict(
    #         params=muon_group,
    #         use_muon=True,
    #         lr=lr,
    #         weight_decay=0.01,
    #     ),
    #     dict(
    #         params=adamw_group,
    #         use_muon=False,
    #         lr=lr,
    #         betas=(0.9, 0.95),
    #         weight_decay=0.01,
    #     ),
    # ]
    # optimizer = Muon(param_groups, use_adjusted_lr=True)

    optimizer = torch.optim.AdamW(attentive_probe.parameters(), lr=lr)

    # Setup data loaders
    train_loader, val_loader = build_dali_iterators(
        config["data"], local_rank, global_rank, world_size
    )

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=0,
        start_lr=lr,
        ref_lr=lr,
        total_steps=len(train_loader) * epochs,
        final_lr=0,
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(local_rank)

    # Initialize wandb only on the main process
    if global_rank == 0:
        wandb_mode = config.get("wandb", "online")

        wandb.init(
            project=config["project"],
            config={"probe_config": config, "pretrain_config": pretrain_config},
            mode=wandb_mode,
        )

    for epoch in range(epochs):
        train_one_epoch(
            epoch,
            model,
            attentive_probe,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_fn,
            config,
            global_rank,
        )
        validate(model, attentive_probe, val_loader, loss_fn, config, local_rank)


if __name__ == "__main__":
    main()
