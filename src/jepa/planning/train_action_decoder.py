import argparse
from math import sqrt
import sys

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch.optim import AdamW
from tqdm import tqdm

from pytorch_optimizer import Muon

from jepa.datasets.path_dataset import build_path_dataloader
from jepa.models.action_decoder import TransformerActionDecoder
from jepa.models.jepa import JEPA
from jepa.utils.helpers import SmoothedValue, MeanMetric

if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dummy", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    checkpoint = torch.load(args.checkpoint)
    pretrain_config = checkpoint["config"]
    pretrain_data_config = pretrain_config["data"]

    world_model = (
        JEPA(
            pretrain_config["encoder"],
            pretrain_config["predictor"],
        )
        .requires_grad_(False)
        .eval()
        .cuda()
    )
    world_model.load_state_dict(checkpoint["model"], strict=True)

    model_config["in_dim"] = world_model.dim
    model = TransformerActionDecoder(model_config).cuda()

    dataloader = build_path_dataloader(data_config)
    valloader = build_path_dataloader(data_config, val=True)

    muon_group = [p for _, p in model.named_parameters() if p.ndim >= 2]
    adamw_group = [p for _, p in model.named_parameters() if p.ndim < 2]
    param_groups = [
        dict(
            params=muon_group,
            use_muon=True,
            lr=float(train_config["lr"]),
            weight_decay=float(train_config["wd"]),
        ),
        dict(
            params=adamw_group,
            use_muon=False,
            lr=float(train_config["lr"]),
            betas=(0.9, 0.95),
            weight_decay=float(train_config["wd"]),
        ),
    ]
    optimizer = Muon(param_groups, use_adjusted_lr=True)

    for ep in range(train_config["epochs"]):
        smoothed_mse_train = MeanMetric()
        pbar = tqdm(dataloader, dynamic_ncols=True)
        model.train()
        err = 0
        for i, x in enumerate(pbar):
            frames = x[0]["frames"]
            actions = x[0]["actions"]
            actions = actions.float()[..., :3]

            with torch.inference_mode(), torch.amp.autocast("cuda"):  # type: ignore
                B, T, *_ = frames.shape
                frames = rearrange(frames, "b t ... -> (b t) ...")
                states = world_model.forward_features(frames)["register"]
                states = rearrange(states, "(b t) ... -> b t ...", b=B)

            states = rearrange(states.float().clone(), "b t n d -> b (t n) d")

            action_pred = model(states)[:, 1:]

            acc = None
            H = None
            if config["data"]["type"] == "cls":
                action_pred = rearrange(action_pred, "b t c -> (b t) c")
                actions = rearrange(actions, "b t -> (b t)")
                loss = F.cross_entropy(action_pred, actions.long())
                acc = (
                    torch.argmax(action_pred, -1) == actions
                ).sum().item() / actions.shape[0]
                H = (
                    torch.special.entr(action_pred.softmax(dim=-1))
                    .sum(-1)
                    .mean()
                    .item()
                )
            else:
                loss = F.mse_loss(action_pred, actions)
                err += F.l1_loss(action_pred, actions, reduction="none").mean(
                    dim=(0, 2)
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            smoothed_mse_train.update(loss.item())
            pbar.set_postfix(
                {
                    "loss": smoothed_mse_train.avg,
                    "H": H,
                    "acc": acc,
                    # "err": err.cpu().detach() / i,
                }
            )

        if (ep + 1) % train_config["val_freq"] == 0:
            print("Validating...")
            smoothed_mse_val = MeanMetric()
            pbar = tqdm(valloader, dynamic_ncols=True)
            model.eval()
            for x in pbar:
                frames = x[0]["frames"]
                actions = x[0]["actions"]

                actions = actions.float()[..., :3]

                with torch.inference_mode(), torch.amp.autocast("cuda"):  # type: ignore
                    B, T, *_ = frames.shape
                    frames = rearrange(frames, "b t ... -> (b t) ...")
                    states = world_model.forward_features(frames)["register"]
                    states = rearrange(states, "(b t) ... -> b t ...", b=B)

                states = rearrange(states.float().clone(), "b t n d -> b (t n) d")

                with torch.no_grad():
                    action_pred = model(states)[:, 1:]

                acc = None
                H = None
                if config["data"]["type"] == "cls":
                    action_pred = rearrange(action_pred, "b t c -> (b t) c")
                    actions = rearrange(actions, "b t -> (b t)")
                    loss = F.cross_entropy(action_pred, actions.long())
                    acc = (
                        torch.argmax(action_pred, -1) == actions
                    ).sum().item() / actions.shape[0]
                    H = (
                        torch.special.entr(action_pred.softmax(dim=-1))
                        .sum(-1)
                        .mean()
                        .item()
                    )
                else:
                    loss = F.mse_loss(action_pred, actions)

                smoothed_mse_val.update(loss.item())
                pbar.set_postfix(
                    {
                        "loss": smoothed_mse_val.avg,
                        "H": H,
                        "acc": acc,
                    }
                )
