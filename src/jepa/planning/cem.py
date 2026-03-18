import torch
from einops import rearrange, repeat
import argparse

from tqdm import tqdm

from jepa.datasets.video_dataset import build_dali_iterators
from jepa.models.model import JEPA
from jepa.planning.base_planner import BasePlanner


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        population=1024,
        elite_frac=0.1,
        iterations=6,
        alpha=0.1,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.population = population
        self.elite = max(1, int(population * elite_frac))
        self.iterations = iterations
        self.alpha = alpha
        self.progress_bar = progress_bar
        self.device = next(wm.parameters()).device

    @torch.no_grad
    def _encode_sequence(self, x):
        B, T, *_ = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        state = self.wm.forward_features(x)["register"]
        state = rearrange(state, "(b t) ... -> b t ...", b=B)

        print("true mse: ", (state - state[:, -1:]).pow(2).mean((-1, -2)))
        return state[:, :1], state[:, -1:]

    def distance(self, path, goal):
        return (path - goal).pow(2).mean((-1, -2))

    def correct_plan(self, start, plan, goal, samples=16):
        B, T, N, D = plan.shape

        print(start.shape, plan.shape)
        path = self.wm.simulate(start, plan)
        path = repeat(path, "b ... -> (b p) ...", p=samples)
        start = repeat(start, "b ... -> (b p) ...", p=samples)

        prop_plan = self.wm.sample_posterior(path)
        prop_path = self.wm.simulate(start, prop_plan)
        prop_plan = rearrange(prop_plan, "(b p) ... -> b p ...", p=samples)

        cost = self.distance(prop_path, goal)[:, -1]
        exit()
        cost = rearrange(cost, "(b p) ... -> b p ...", p=samples)
        idx = torch.argmin(cost, dim=1)
        idx = repeat(idx, "b p -> b p t n d", t=T, n=N, d=D)

        corrected_plan = torch.gather(prop_plan, 1, idx)
        print(corrected_plan.shape)
        exit()

        return corrected_plan

    def plan(self, x):  # type: ignore
        self.wm.eval()
        start, goal = self._encode_sequence(x)
        B, _, N, D = start.shape
        exit()

        mu = torch.zeros(B, self.horizon, N, self.action_dim, device=x.device)
        sigma = torch.ones(B, self.horizon, N, self.action_dim, device=x.device)

        bar = tqdm(range(self.iterations), dynamic_ncols=True)
        for _ in bar:

            eps = torch.randn(
                B,
                self.population,
                self.horizon,
                N,
                self.action_dim,
                device=x.device,
            )

            plan = mu + eps * sigma
            plan = rearrange(plan, "b p ... -> (b p) ...")
            path = repeat(start, "b ... -> (b p) ...", p=self.population)
            path = self.wm.simulate(path, plan)

            cost = self.distance(path, goal)[:, -1]
            cost = rearrange(cost, "(b p) ... -> b p ...", b=B)
            plan = rearrange(plan, "(b p) ... -> b p ...", b=B)

            (_, idx) = torch.topk(-cost, self.elite, dim=1)

            idx = repeat(
                idx, "b p -> b p t n d", t=self.horizon, n=N, d=self.action_dim
            )
            elite = torch.gather(plan, 1, idx)

            new_mu = elite.mean(dim=1)
            new_sigma = elite.var(dim=1).sqrt()

            mu = self.alpha * new_mu + (1 - self.alpha) * mu
            sigma = self.alpha * new_sigma + (1 - self.alpha) * sigma

            bar.set_postfix({"mean mse": cost.mean()})

        path = self.wm.simulate(start, mu)
        idk = self.correct_plan(start, mu, goal)
        idk = self.wm.sample_posterior(path)
        print(self.distance(path, goal))
        print(mu.norm(2, dim=-1).mean())
        print(idk.norm(2, dim=-1).mean())
        path_idk = self.wm.simulate(start, idk)
        print(self.distance(path_idk, goal))


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
            config["transform_model"],
        )
        .cuda()
        .requires_grad_(False)
    )
    H = 8
    transform_dim = config["transform_model"]["transform_dim"]
    data_conf["sequence_length"] = H
    wm.load_state_dict(checkpoint["model"])
    planner = CEMPlanner(wm, transform_dim, None, H - 1, population=32, iterations=30)

    train_loader, val_loader = build_dali_iterators(data_conf, 0, 0, 1, seed=213)
    obs = next(val_loader)[0]["data"][:1]

    with torch.amp.autocast("cuda"):  # type: ignore
        planner.plan(obs)
