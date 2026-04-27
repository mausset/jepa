import torch
from einops import rearrange, repeat
from tqdm import tqdm

from jepa.planning.base_planner import BasePlanner


def stratified_resample(weights):
    """Batched stratified resampling.

    Args:
        weights: (B, P) normalized along dim 1
    Returns:
        ancestors: (B, P) long indices into the particle dim
    """
    B, P = weights.shape
    device = weights.device
    w = weights.float()
    cdf = w.cumsum(dim=1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
    strata = torch.arange(P, device=device, dtype=cdf.dtype)[None, :]
    u = (strata + torch.rand(B, P, device=device, dtype=cdf.dtype)) / P
    return torch.searchsorted(cdf, u).clamp_(max=P - 1)


class SMCPlanner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        population=1024,
        temperature=1.0,
        ess_threshold=0.5,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.population = population
        self.temperature = temperature
        self.ess_threshold = ess_threshold
        self.progress_bar = progress_bar
        self.device = next(wm.parameters()).device

        if not wm.has_bottleneck:
            raise ValueError(
                f"SMCPlanner requires a stochastic bottleneck (fsq or vae), got {wm.bottleneck_type!r}"
            )

    def weights(self, z_curr, z_T_flat, B, P):
        """Normalized weights from squared distance to the goal (per batch)."""
        cost_flat = (z_curr - z_T_flat).pow(2).mean(dim=(-2, -1)).float()
        cost = rearrange(cost_flat, "(b p) -> b p", b=B, p=P)
        min_cost = cost.min(dim=1, keepdim=True).values
        w = (-(cost - min_cost) / self.temperature).exp()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return cost, w

    def maybe_resample(self, traj, w, B, P):
        """Stratified resample per batch where ESS drops below threshold."""
        device = traj.device
        ess = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)
        needs_resample = ess < self.ess_threshold * P
        if not needs_resample.any():
            return traj, ess
        ancestors = stratified_resample(w)
        identity = torch.arange(P, device=device)[None, :].expand(B, P)
        ancestors = torch.where(needs_resample[:, None], ancestors, identity)
        offsets = (torch.arange(B, device=device) * P)[:, None]
        flat_ancestors = (ancestors + offsets).reshape(-1)
        return traj[flat_ancestors], ess

    def best_trajectory(self, traj, z_T_flat, B, P):
        """Per-batch particle with minimum final distance to the goal."""
        _, _, N, D = traj.shape
        H = self.horizon
        final_cost = (traj[:, -1] - z_T_flat).pow(2).mean(dim=(-2, -1)).float()
        final_cost = rearrange(final_cost, "(b p) -> b p", b=B, p=P)
        best_idx = final_cost.argmin(dim=1)
        traj_b = rearrange(traj, "(b p) t n d -> b p t n d", b=B, p=P)
        idx_exp = best_idx[:, None, None, None, None].expand(B, 1, H, N, D)
        return traj_b.gather(1, idx_exp).squeeze(1)

    @torch.inference_mode()
    def plan(self, z_0, z_T):
        """Plan from z_0 to z_T by SMC over the bottleneck's prior.

        Args:
            z_0: (B, N, D)
            z_T: (B, N, D)
        Returns:
            (B, H+1, N, D) best trajectory: [z_0, rollout states, z_T].
        """
        B = z_0.shape[0]
        P = self.population

        traj = repeat(z_0, "b n d -> (b p) 1 n d", p=P).contiguous()
        z_T_flat = repeat(z_T, "b n d -> (b p) n d", p=P)

        pbar = (
            tqdm(range(self.horizon - 1), desc="smc", leave=False)
            if self.progress_bar
            else range(self.horizon - 1)
        )

        with torch.amp.autocast("cuda"):
            for _ in pbar:
                traj = self.wm.rollout(traj, horizon=1)
                cost, w = self.weights(traj[:, -1], z_T_flat, B, P)
                traj, ess = self.maybe_resample(traj, w, B, P)

                if self.progress_bar:
                    pbar.set_postfix(
                        {
                            "cost": f"{cost.min(dim=1).values.mean().item():.4f}",
                            "ess": f"{(ess / P).mean().item():.2f}",
                        }
                    )

            best_traj = self.best_trajectory(traj, z_T_flat, B, P)

        return torch.cat([best_traj, z_T[:, None]], dim=1)
