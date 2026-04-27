import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from jepa.planning.base_planner import BasePlanner


class CEMPlanner(BasePlanner):
    """Cross-entropy method planner over the bottleneck's prior.

    - fsq: maintains per-(step, token) categorical logits over the FSQ codebook,
      refits to elite code counts.
    - vae: maintains per-(step, token) Gaussian (mu, logvar) over the VAE latent,
      refits to elite empirical mean/variance.
    """

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
        min_sigma=0.1,
        progress_bar=True,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.population = population
        self.elite = max(1, int(population * elite_frac))
        self.iterations = iterations
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.progress_bar = progress_bar
        self.device = next(wm.parameters()).device

        if not wm.has_bottleneck:
            raise ValueError(
                f"CEMPlanner requires a stochastic bottleneck (fsq or vae), got {wm.bottleneck_type!r}"
            )

    @property
    def bottleneck_type(self):
        return self.wm.bottleneck_type

    @property
    def latent_dim(self):
        return self.wm.predictor.latent_dim

    @property
    def K(self):
        return self.wm.predictor.fsq.codebook_size

    def init_search_state(self, B, N, device):
        """Initialize the per-(step, token) search distribution."""
        H = self.horizon
        if self.bottleneck_type == "fsq":
            return {"logits": torch.zeros(B, H - 1, N, self.K, device=device)}
        # vae: N(0, I) at start, matching the prior
        return {
            "mu": torch.zeros(B, H - 1, N, self.latent_dim, device=device),
            "logvar": torch.zeros(B, H - 1, N, self.latent_dim, device=device),
        }

    def sample_population(self, state, P, device):
        """Sample P populations from the search state.

        Returns:
            samples: search-space samples (B, P, H-1, N, ...) for bookkeeping/refit
            latents: latents to condition the predictor on, (B, P, H-1, N, latent_dim)
        """
        B = state["logits"].shape[0] if "logits" in state else state["mu"].shape[0]
        H = self.horizon

        if self.bottleneck_type == "fsq":
            logits = state["logits"]
            dist = torch.distributions.Categorical(
                logits=logits[:, None].expand(B, P, H - 1, -1, -1)
            )
            codes = dist.sample()  # (B, P, H-1, N)
            codebook = self.wm.predictor.fsq.implicit_codebook.to(device)
            latents = codebook[codes]  # (B, P, H-1, N, latent_dim)
            return codes, latents

        mu, logvar = state["mu"], state["logvar"]
        mu_e = mu[:, None].expand(B, P, -1, -1, -1)
        logvar_e = logvar[:, None].expand(B, P, -1, -1, -1)
        eps = torch.randn_like(mu_e)
        samples = mu_e + (0.5 * logvar_e).exp() * eps
        return samples, samples

    def refit(self, state, elite_samples):
        """Blend the search distribution toward the empirical elite distribution."""
        if self.bottleneck_type == "fsq":
            K = self.K
            one_hot = F.one_hot(elite_samples, num_classes=K).float()
            counts = one_hot.sum(dim=1)
            new_probs = (counts + 1e-6) / (counts.sum(dim=-1, keepdim=True) + K * 1e-6)
            old_probs = state["logits"].softmax(dim=-1)
            blended = (1.0 - self.alpha) * old_probs + self.alpha * new_probs
            state["logits"] = blended.clamp_min(1e-12).log()
            return

        new_mu = elite_samples.mean(dim=1)
        new_var = elite_samples.var(dim=1, unbiased=False).clamp_min(self.min_sigma ** 2)
        old_var = state["logvar"].exp()
        state["mu"] = (1.0 - self.alpha) * state["mu"] + self.alpha * new_mu
        blended_var = (1.0 - self.alpha) * old_var + self.alpha * new_var
        state["logvar"] = blended_var.clamp_min(self.min_sigma ** 2).log()

    @torch.inference_mode()
    def plan(self, z_0, z_T):
        """Plan a trajectory from z_0 to z_T by CEM over the bottleneck prior."""
        B, N, D = z_0.shape
        H = self.horizon
        P = self.population
        device = z_0.device

        search_state = self.init_search_state(B, N, device)

        pbar = tqdm(range(self.iterations), desc="cem", leave=False) if self.progress_bar else range(self.iterations)
        best_cost = None
        best_traj = None

        with torch.amp.autocast("cuda"):
            for _ in pbar:
                samples, latents = self.sample_population(search_state, P, device)

                latents_flat = rearrange(latents, "b p h n d -> (b p) h n d")
                z_0_flat = repeat(z_0, "b n d -> (b p) n d", p=P)
                traj_flat = self.wm.rollout(z_0_flat[:, None], latents=latents_flat)  # (B*P, H, N, D)

                z_T_rep = repeat(z_T, "b n d -> (b p) n d", p=P)
                cost_flat = (traj_flat[:, -1] - z_T_rep).pow(2).mean(dim=(-2, -1))
                cost = rearrange(cost_flat, "(b p) -> b p", b=B, p=P)

                _, elite_idx = cost.topk(self.elite, dim=1, largest=False)

                if self.bottleneck_type == "fsq":
                    # samples: (B, P, H-1, N)
                    idx_exp = elite_idx[:, :, None, None].expand(B, self.elite, H - 1, N)
                    elite_samples = samples.gather(1, idx_exp)
                else:
                    # samples: (B, P, H-1, N, latent_dim)
                    idx_exp = elite_idx[:, :, None, None, None].expand(
                        B, self.elite, H - 1, N, self.latent_dim
                    )
                    elite_samples = samples.gather(1, idx_exp)

                self.refit(search_state, elite_samples)

                iter_best_idx = cost.argmin(dim=1)
                traj = rearrange(traj_flat, "(b p) h n d -> b p h n d", b=B, p=P)
                iter_best_traj = traj.gather(
                    1,
                    iter_best_idx[:, None, None, None, None].expand(B, 1, H, N, D),
                ).squeeze(1)
                iter_best_cost = cost.gather(1, iter_best_idx[:, None]).squeeze(1)

                if best_cost is None:
                    best_cost = iter_best_cost
                    best_traj = iter_best_traj
                else:
                    improved = iter_best_cost < best_cost
                    best_cost = torch.where(improved, iter_best_cost, best_cost)
                    best_traj = torch.where(
                        improved[:, None, None, None], iter_best_traj, best_traj
                    )

                if self.progress_bar:
                    pbar.set_postfix({"cost": f"{best_cost.mean().item():.4f}"})

        return torch.cat([best_traj, z_T[:, None]], dim=1)
