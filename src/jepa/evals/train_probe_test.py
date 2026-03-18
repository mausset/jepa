import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import wandb
import yaml
from tqdm import tqdm
from einops import rearrange
from pytorch_optimizer import Muon

# Removed DALI import
# from jepa.datasets.image_dataset import build_dali_iterators
from jepa.models.attentive_probe import AttentiveProbe, FFNProbe
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
    ).to(local_rank)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)

    # Initialize Attentive Probe
    # attentive_probe = AttentiveProbe(
    #     model.dim, config["heads"], config["num_classes"], config["features"]
    # ).to(local_rank)
    attentive_probe = FFNProbe(model.dim, config["num_classes"]).to(local_rank)
    attentive_probe = DDP(
        attentive_probe, device_ids=[local_rank], output_device=local_rank
    )

    return model, attentive_probe, pretrain_config


def setup_dataloaders(config, global_rank, world_size):
    batch_size = config["data"].get("batch_size", 64)
    num_workers = config["data"].get("num_workers", 12)
    data_root = config["data"].get("root", "./data")

    # ImageNet statistics are standard for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            # Randomly crop the image and resize it to 224x224
            # This forces the model to learn features from parts of the object as well as the whole
            transforms.RandomResizedCrop(
                224, scale=(0.3, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Apply RandAugment (AutoAugment variant)
            # num_ops: number of augmentation transformations to apply sequentially
            # magnitude: strength of the augmentation
            transforms.RandAugment(
                num_ops=2, magnitude=9, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
            # Randomly erases a rectangle in the image to improve occlusion robustness
            # Must occur after ToTensor and Normalize
            # transforms.RandomErasing(p=0.25),
        ]
    )

    # Standard evaluation transform: Resize slightly larger then CenterCrop
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # Ensure data is downloaded only once in DDP
    if global_rank == 0:
        os.makedirs(data_root, exist_ok=True)
        datasets.CIFAR100(root=data_root, train=True, download=True)
        datasets.CIFAR100(root=data_root, train=False, download=True)

    # Wait for rank 0 to finish downloading
    dist.barrier()

    train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=val_transform
    )

    # Setup Distributed Samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


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
    # Important for DistributedSampler to shuffle correctly every epoch
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    accumulation_steps = config.get("accumulation_steps", 1)
    train_smoothed_accuracy = SmoothedValue()

    if global_rank == 0:
        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Training Epoch {epoch + 1}/{config['epochs']}",
        )

    optimizer.zero_grad()

    # Changed loop structure for standard PyTorch DataLoader
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        x = rearrange(x, "b c h w -> b h w c")

        with torch.inference_mode(), torch.amp.autocast("cuda"):  # type: ignore
            # features = model.encoder.forward_features(x)["register"]
            features = model.encoder(x)["register"]

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
        # Changed loop structure for standard PyTorch DataLoader
        for x, y in val_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            x = rearrange(x, "b c h w -> b h w c")

            with torch.amp.autocast("cuda"):  # type: ignore
                # features = model.encoder.forward_features(x)["register"]
                features = model.encoder(x)["register"]
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

    # optimizer = torch.optim.AdamW(attentive_probe.parameters(), lr=lr)

    muon_group = [p for _, p in attentive_probe.named_parameters() if p.ndim >= 2]
    adamw_group = [p for _, p in attentive_probe.named_parameters() if p.ndim < 2]
    param_groups = [
        dict(
            params=muon_group,
            use_muon=True,
            lr=float(config["lr"]),
            weight_decay=float(config["wd"]),
        ),
        dict(
            params=adamw_group,
            use_muon=False,
            lr=float(config["lr"]),
            betas=(0.9, 0.95),
            weight_decay=float(config["wd"]),
        ),
    ]
    optimizer = Muon(param_groups, use_adjusted_lr=True)

    # Setup data loaders (Replaced build_dali_iterators with setup_dataloaders)
    train_loader, val_loader = setup_dataloaders(config, global_rank, world_size)

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
