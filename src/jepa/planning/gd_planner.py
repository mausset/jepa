import math
import torch
import torch.nn.functional as F
from einops import rearrange
import argparse

from tqdm import tqdm

from jepa.datasets.video_dataset import build_dali_iterators
from jepa.models.model import JEPA
from jepa.planning.base_planner import BasePlanner


class GradientDescentPlanner(BasePlanner):
    def __init__(self, wm, action_dim, pre_processor, progress_bar=False):
        super().__init__(wm, action_dim, pre_processor)
        self.progress_bar = progress_bar

    def plan(self, x, iterations=100, beta=2e-7):
        B, T, *_ = x.shape

        with torch.no_grad():
            x = rearrange(x, "b t ... -> (b t) ...")
            state = self.wm.forward_features(x)["registers"]
            state = rearrange(state, "(b t) ... -> b t ...", b=B)

            start = state[:, :1]
            end = state[:, -1:]

            _, _, true_transform, _ = self.wm.post_hoc(state)
            print("true norm", true_transform.norm(2, dim=-1))
            _, transform_trajectory, *_ = self.wm.rollout(start, steps=T - 1)

            print((state - end).pow(2).mean(dim=-1).sum(dim=-1))

        print(transform_trajectory.norm(2, dim=-1).mean())
        transform_trajectory.requires_grad_(True)
        optimizer = torch.optim.Adam([transform_trajectory], lr=4e-2, betas=(0.9, 0.95))

        pbar = tqdm(total=iterations, dynamic_ncols=True)
        for i in range(iterations):
            state_trajectory, _, psi_mu, psi_std = self.wm.rollout(
                start, transform_trajectory, steps=T - 1
            )
            psi_var = psi_std.exp()

            nll = 0.5 * (
                (transform_trajectory - psi_mu) ** 2 / psi_var  # + psi_var.log()
            ).sum((-1, -2))
            goal_loss = (state_trajectory[:, -1] - end).pow(2).mean(dim=-1).sum()
            loss = goal_loss + beta * nll.mean()

            norm = transform_trajectory.norm(2, dim=-1).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(
                {
                    "nll": nll.mean().item(),
                    "goal_loss": goal_loss.item(),
                    "norm": norm.item(),
                }
            )
            pbar.update()

        print(transform_trajectory.norm(2, dim=-1))
        print(nll)
        print((state_trajectory - end).pow(2).mean(dim=-1).sum(dim=-1))


if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to resume"
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cuda"))
    config = checkpoint["config"]

    train_conf = config["training"]
    data_conf = config["data"]
    data_conf["data_root"] = "data/walking_tours/240p_hevc_ext/"
    data_conf["val_root"] = "data/walking_tours/240p_hevc_ext_val/"

    wm = (
        JEPA(
            config["encoder"],
            config["predictor"],
            n_registers=config["registers"],
            resolution=data_conf["resolution"],
            transform_dim=config["transform_dim"],
            context=data_conf["sequence_length"] - 1,
        )
        .cuda()
        .requires_grad_(False)
    )
    wm.load_state_dict(checkpoint["model"])
    planner = GradientDescentPlanner(wm, None, None)

    # data_conf["sequence_length"] = 12
    train_loader, val_loader = build_dali_iterators(data_conf, 0, 0, 1, seed=12312)
    obs = next(val_loader)[0]["data"][:1]

    planner.plan(obs, iterations=400, beta=1e-7)
