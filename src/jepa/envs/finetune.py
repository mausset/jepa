import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from jepa.datasets.path_dataset import build_path_dataloader
from jepa.models.model import JEPA
from jepa.utils.helpers import (
    SmoothedValue,
    categorical_kl,
    gaussian_kl_loss,
)
from jepa.utils.muon import SingleDeviceMuonWithAuxAdam


def init_opt(model, lr, wd):
    skip = ["encoder"]
    filter = [
        "context_pe",
        "image_pe",
        "registers",
        "token",
        "phi_head",
        "psi_head",
        "codebook",
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
        dict(
            params=muon_group,
            use_muon=True,
            lr=lr,
            weight_decay=wd,
        ),
        dict(
            params=adamw_group,
            use_muon=False,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=wd,
        ),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    scaler = torch.amp.GradScaler()  # type: ignore
    return optimizer, scaler


def train_step(model, data, optimizer, scaler, config):

    with torch.amp.autocast("cuda"):  # type: ignore
        phi_pred, psi_pred, target, psi, phi = model(data)

        phi_loss = F.mse_loss(phi_pred, target)
        psi_loss = F.mse_loss(psi_pred, target)

        phi_psi_kl = gaussian_kl_loss(*phi, *psi)
        psi_reg = gaussian_kl_loss(
            psi[0], psi[1], torch.zeros_like(psi[0]), torch.zeros_like(psi[1])
        )

    loss = (
        (phi_loss + psi_loss) / 2
        + config["beta"] * phi_psi_kl
        + config["gamma"] * psi_reg
    )

    scaler.scale(loss).backward()
    scaler.step(optimizer)

    scaler.update()
    optimizer.zero_grad()

    return phi_loss, psi_loss, phi_psi_kl, psi_reg


def val_step(model, data, optimizer, scaler, config):

    with torch.inference_mode(), torch.amp.autocast("cuda"):  # type: ignore
        phi_pred, psi_pred, target, psi, phi = model(data)

        phi_loss = F.mse_loss(phi_pred, target)
        psi_loss = F.mse_loss(psi_pred, target)

        phi_psi_kl = gaussian_kl_loss(*phi, *psi)
        psi_reg = gaussian_kl_loss(
            psi[0], psi[1], torch.zeros_like(psi[0]), torch.zeros_like(psi[1])
        )

    return phi_loss, psi_loss, phi_psi_kl, psi_reg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_config = config["data"]
    train_config = config["train"]

    checkpoint = torch.load(args.checkpoint)
    pretrain_config = checkpoint["config"]
    pretrain_data_config = pretrain_config["data"]

    train_config["beta"] = float(pretrain_config["beta"])
    train_config["gamma"] = float(pretrain_config["gamma"])

    world_model = JEPA(
        pretrain_config["encoder"],
        pretrain_config["predictor"],
        pretrain_config["transform_model"],
        resolution=pretrain_data_config["resolution"],
    ).cuda()
    world_model.load_state_dict(checkpoint["model"], strict=True)

    train_loader = build_path_dataloader(data_config)
    val_loader = build_path_dataloader(data_config, val=True)

    optimizer, scaler = init_opt(
        world_model, float(train_config["lr"]), float(train_config["wd"])
    )

    epochs = train_config["epochs"]

    for _ in range(epochs):

        phi_m = SmoothedValue(beta=0.98)
        psi_m = SmoothedValue(beta=0.98)
        phi_psi_kl_m = SmoothedValue(beta=0.98)
        psi_reg_m = SmoothedValue(beta=0.98)

        world_model.train()
        pbar = tqdm(train_loader, dynamic_ncols=True)
        for data in pbar:
            data = data[0]["frames"]

            phi_loss, psi_loss, phi_psi_kl, psi_reg = train_step(
                world_model, data, optimizer, scaler, train_config
            )

            phi_m.update(phi_loss)
            psi_m.update(psi_loss)
            phi_psi_kl_m.update(phi_psi_kl)
            psi_reg_m.update(psi_reg)
            pbar.set_postfix(
                {
                    "phi": phi_m(),
                    "psi": psi_m(),
                    "phi_psi_kl": phi_psi_kl_m(),
                    "psi_reg": psi_reg_m(),
                }
            )

        phi_m = SmoothedValue(beta=0.98)
        psi_m = SmoothedValue(beta=0.98)
        phi_psi_kl_m = SmoothedValue(beta=0.98)
        psi_reg_m = SmoothedValue(beta=0.98)
        pbar = tqdm(val_loader, dynamic_ncols=True)
        world_model.eval()
        for data in pbar:
            data = data[0]["frames"]

            phi_loss, psi_loss, phi_psi_kl, psi_reg = val_step(
                world_model, data, optimizer, scaler, train_config
            )

            phi_m.update(phi_loss)
            psi_m.update(psi_loss)
            phi_psi_kl_m.update(phi_psi_kl)
            psi_reg_m.update(psi_reg)
            pbar.set_postfix(
                {
                    "phi": phi_m(),
                    "psi": psi_m(),
                    "phi_psi_kl": phi_psi_kl_m(),
                    "psi_reg": psi_reg_m(),
                }
            )

    pth_path = Path(args.checkpoint)
    pth_path_finetune = pth_path.with_name(
        f"{pth_path.stem}_finetune{''.join(pth_path.suffixes)}"
    )
    checkpoint["model"] = world_model.state_dict()
    torch.save(checkpoint, pth_path_finetune)
