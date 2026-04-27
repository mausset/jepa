import torch
import torch.nn.functional as F
from tqdm import tqdm

from jepa.planning.base_planner import BasePlanner


class CollocationPlanner(BasePlanner):
    def __init__(
        self,
        wm,
        action_dim,
        pre_processor,
        horizon,
        steps=200,
        lr=1e-2,
        optimizer="adam",
        project=False,
    ):
        super().__init__(wm, action_dim, pre_processor)
        self.horizon = horizon
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer  # "adam" | "sgd" | "ula"
        self.project = project

    def project_to_typical_set(self, z):
        """Project each token onto the sphere ||z_token||^2 / D = 1."""
        D = z.shape[-1]
        scale = (z.pow(2).sum(-1, keepdim=True) / D).sqrt().clamp_min(1e-8)
        return z / scale

    def nll(self, trajectory):
        """MSE of a bottlenecked trajectory under the generative model.

        Args:
            trajectory: (B, T+1, N, D)
        Returns:
            scalar loss
        """
        pred = self.wm.predict_all(trajectory)  # (B, T, N, D)
        return F.mse_loss(pred, trajectory[:, 1:])

    def postfix(self, loss, z_mid, z_T, trajectory):
        """Compute tqdm postfix metrics from the current trajectory."""
        D = z_mid.shape[-1]
        state_norm = (z_mid.pow(2).sum(-1) / D).sqrt().mean().item()

        pred_all = self.wm.predict_all(trajectory)  # (B, T, N, D)
        norm = (pred_all.pow(2).sum(-1) / D).sqrt().mean().item()
        nll_T = F.mse_loss(pred_all[:, -1], z_T).item()

        return {
            "loss": f"{loss:.4f}",
            "nll_T": f"{nll_T:.4f}",
            "pred_norm": f"{norm:.4f}",
            "state_norm": f"{state_norm:.4f}",
        }

    def plan(self, z_0, z_T):
        """Optimize intermediate latents to connect z_0 to z_T.

        Args:
            z_0: (B, N, D) initial encoded state
            z_T: (B, N, D) target encoded state
        Returns:
            trajectory: (B, T+1, N, D) optimized latent trajectory
        """
        z_0, z_T = z_0.detach(), z_T.detach()

        traj0 = self.wm.rollout(z_0[:, None], horizon=self.horizon - 1)
        z_mid = traj0[:, 1:].detach().requires_grad_(True)

        if self.optimizer_type == "ula":
            pbar = tqdm(range(self.steps), desc="plan", leave=False)
            for _ in pbar:
                if z_mid.grad is not None:
                    z_mid.grad.zero_()
                trajectory = torch.cat([z_0[:, None], z_mid, z_T[:, None]], dim=1)
                loss = self.nll(trajectory)
                loss.backward()
                with torch.no_grad():
                    noise_scale = (self.lr / z_mid.shape[-1]) ** 0.5
                    z_mid = z_mid - (self.lr / 2) * z_mid.grad + noise_scale * torch.randn_like(z_mid)
                    if self.project:
                        z_mid = self.project_to_typical_set(z_mid)
                z_mid = z_mid.detach().requires_grad_(True)

                with torch.no_grad():
                    trajectory = torch.cat([z_0[:, None], z_mid, z_T[:, None]], dim=1)
                    pbar.set_postfix(self.postfix(loss.item(), z_mid, z_T, trajectory))
        else:
            if self.optimizer_type == "adam":
                opt = torch.optim.Adam([z_mid], lr=self.lr)
            else:
                opt = torch.optim.SGD([z_mid], lr=self.lr)

            pbar = tqdm(range(self.steps), desc="plan", leave=False)
            for _ in pbar:
                opt.zero_grad()
                trajectory = torch.cat([z_0[:, None], z_mid, z_T[:, None]], dim=1)
                loss = self.nll(trajectory)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    if self.project:
                        z_mid.data.copy_(self.project_to_typical_set(z_mid.data))
                    trajectory = torch.cat([z_0[:, None], z_mid, z_T[:, None]], dim=1)
                    pbar.set_postfix(self.postfix(loss.item(), z_mid, z_T, trajectory))

        return torch.cat([z_0[:, None], z_mid.detach(), z_T[:, None]], dim=1)
