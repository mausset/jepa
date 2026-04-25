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
        self.bottleneck_predictor = wm.predictor
        self.latent_dim = self.bottleneck_predictor.latent_dim
        self.context = self.bottleneck_predictor.context

    def propagate(self, state_hist):
        """Sample one latent from the bottleneck's prior and take one predictor step."""
        M, _, N, _ = state_hist.shape
        latent = self.bottleneck_predictor.sample_prior((M, 1, N), device=state_hist.device)
        window = state_hist[:, -self.context :]
        z_next = self.wm.sample(window, latent=latent)[:, None]
        return torch.cat([state_hist, z_next], dim=1)

    def weights(self, z_curr, z_T_flat, B, P):
        """Normalized weights from squared distance to the goal (per batch)."""
        cost_flat = (z_curr - z_T_flat).pow(2).mean(dim=(-2, -1)).float()
        cost = rearrange(cost_flat, "(b p) -> b p", b=B, p=P)
        min_cost = cost.min(dim=1, keepdim=True).values
        w = (-(cost - min_cost) / self.temperature).exp()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return cost, w

    def maybe_resample(self, state_hist, w, B, P):
        """Stratified resample per batch where ESS drops below threshold."""
        device = state_hist.device
        ess = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)
        needs_resample = ess < self.ess_threshold * P
        if not needs_resample.any():
            return state_hist, ess
        ancestors = stratified_resample(w)
        identity = torch.arange(P, device=device)[None, :].expand(B, P)
        ancestors = torch.where(needs_resample[:, None], ancestors, identity)
        offsets = (torch.arange(B, device=device) * P)[:, None]
        flat_ancestors = (ancestors + offsets).reshape(-1)
        return state_hist[flat_ancestors], ess

    def best_trajectory(self, state_hist, z_T_flat, B, P):
        """Per-batch particle with minimum final distance to the goal."""
        _, _, N, D = state_hist.shape
        H = self.horizon
        final_cost = (state_hist[:, -1] - z_T_flat).pow(2).mean(dim=(-2, -1)).float()
        final_cost = rearrange(final_cost, "(b p) -> b p", b=B, p=P)
        best_idx = final_cost.argmin(dim=1)
        state_hist_b = rearrange(state_hist, "(b p) t n d -> b p t n d", b=B, p=P)
        idx_exp = best_idx[:, None, None, None, None].expand(B, 1, H, N, D)
        return state_hist_b.gather(1, idx_exp).squeeze(1)

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

        state_hist = repeat(z_0, "b n d -> (b p) 1 n d", p=P).contiguous()
        z_T_flat = repeat(z_T, "b n d -> (b p) n d", p=P)

        pbar = (
            tqdm(range(self.horizon - 1), desc="smc", leave=False)
            if self.progress_bar
            else range(self.horizon - 1)
        )

        with torch.amp.autocast("cuda"):
            for _ in pbar:
                state_hist = self.propagate(state_hist)
                cost, w = self.weights(state_hist[:, -1], z_T_flat, B, P)
                state_hist, ess = self.maybe_resample(state_hist, w, B, P)

                if self.progress_bar:
                    pbar.set_postfix(
                        {
                            "cost": f"{cost.min(dim=1).values.mean().item():.4f}",
                            "ess": f"{(ess / P).mean().item():.2f}",
                        }
                    )

            best_traj = self.best_trajectory(state_hist, z_T_flat, B, P)

        return torch.cat([best_traj, z_T[:, None]], dim=1)
