import torch
from einops import rearrange, repeat
from tqdm import tqdm

from jepa.planning.base_planner import BasePlanner


class BeamPlanner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        beam_width=64,
        branching=16,
        temperature=1.0,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.beam_width = beam_width
        self.branching = branching
        self.temperature = temperature
        self.progress_bar = progress_bar
        self.device = next(wm.parameters()).device

        if not wm.has_bottleneck:
            raise ValueError(
                f"BeamPlanner requires a stochastic bottleneck (fsq or vae), got {wm.bottleneck_type!r}"
            )
        self.bottleneck_predictor = wm.predictor
        self.latent_dim = self.bottleneck_predictor.latent_dim
        self.context = self.bottleneck_predictor.context

    def expand(self, beam_hist):
        """Branch each of M beam members into `branching` children via the prior."""
        M, _, N, _ = beam_hist.shape
        K_branch = self.branching
        beam_rep = repeat(beam_hist, "m t n d -> (m k) t n d", k=K_branch).contiguous()
        latent = self.bottleneck_predictor.sample_prior(
            (M * K_branch, 1, N), device=beam_hist.device
        )
        window = beam_rep[:, -self.context :]
        z_next = self.wm.sample(window, latent=latent)[:, None]
        return torch.cat([beam_rep, z_next], dim=1)

    def select(self, cand_hist, z_T_flat_cand, B, W):
        """Pick W survivors per batch from W*branching candidates by quasimetric cost."""
        cost_flat = (cand_hist[:, -1] - z_T_flat_cand).pow(2).mean(dim=(-2, -1)).float()
        cost = rearrange(cost_flat, "(b wk) -> b wk", b=B)
        if self.temperature == 0:
            _, top_idx = (-cost).topk(W, dim=1)
        else:
            probs = torch.softmax(-cost / self.temperature, dim=1)
            top_idx = torch.multinomial(probs, W, replacement=False)
        offsets = (torch.arange(B, device=cand_hist.device) * (W * self.branching))[:, None]
        flat_idx = (top_idx + offsets).reshape(-1)
        return cand_hist[flat_idx], cost.gather(1, top_idx)

    def best_trajectory(self, beam_hist, z_T_flat_W, B, W):
        """Per-batch beam member with minimum final distance to the goal."""
        _, _, N, D = beam_hist.shape
        H = self.horizon
        cost_flat = (beam_hist[:, -1] - z_T_flat_W).pow(2).mean(dim=(-2, -1)).float()
        cost = rearrange(cost_flat, "(b w) -> b w", b=B)
        best_idx = cost.argmin(dim=1)
        hist_b = rearrange(beam_hist, "(b w) t n d -> b w t n d", b=B)
        idx_exp = best_idx[:, None, None, None, None].expand(B, 1, H, N, D)
        return hist_b.gather(1, idx_exp).squeeze(1)

    @torch.inference_mode()
    def plan(self, z_0, z_T):
        """Plan from z_0 to z_T by beam search over the bottleneck's prior."""
        B = z_0.shape[0]
        W = self.beam_width

        beam_hist = repeat(z_0, "b n d -> (b w) 1 n d", w=W).contiguous()
        z_T_flat_W = repeat(z_T, "b n d -> (b w) n d", w=W)
        z_T_flat_cand = repeat(z_T, "b n d -> (b w k) n d", w=W, k=self.branching)

        pbar = (
            tqdm(range(self.horizon - 1), desc="beam", leave=False)
            if self.progress_bar
            else range(self.horizon - 1)
        )

        with torch.amp.autocast("cuda"):
            for _ in pbar:
                cand_hist = self.expand(beam_hist)
                beam_hist, cost = self.select(cand_hist, z_T_flat_cand, B, W)

                if self.progress_bar:
                    pbar.set_postfix(
                        {"cost": f"{cost.min(dim=1).values.mean().item():.4f}"}
                    )

            best_traj = self.best_trajectory(beam_hist, z_T_flat_W, B, W)

        return torch.cat([best_traj, z_T[:, None]], dim=1)
