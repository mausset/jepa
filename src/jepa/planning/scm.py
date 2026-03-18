import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import argparse

from tqdm import tqdm

from jepa.datasets.video_dataset import build_dali_iterators
from jepa.models.model import JEPA
from jepa.planning.base_planner import BasePlanner


class Planner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.device = next(wm.parameters()).device

    @torch.no_grad()
    def _encode_sequence(self, x):
        B, T, *_ = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        state = self.wm.forward_features(x)["registers"]
        state = rearrange(state, "(b t) ... -> b t ...", b=B)
        print("true mse: ", (state - state[:, -1:]).pow(2).mean((-1, -2)))
        return state[:, :1].clone(), state[:, -1:].clone()

    @torch.no_grad
    def plan(self, x):  # type: ignore
        self.wm.eval()
        start, goal = self._encode_sequence(x)
        B, _, N, D = start.shape  # A state is N x D


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

    H = 8
    transform_dim = config["transform_dim"]
    data_conf["sequence_length"] = H
    planner = Planner(
        wm,
        transform_dim,
        None,
        H - 1,
    )

    train_loader, val_loader = build_dali_iterators(data_conf, 0, 0, 1, seed=1232)
    obs = next(val_loader)[0]["data"][:1]

    planner.plan(obs)
