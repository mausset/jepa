import torch
from einops import rearrange
from torch import nn

from jepa.models.action_decoder import ActionDecoder
from jepa.models.encoder import build_encoder
from jepa.models.encoder import build_encoder_config
from jepa.models.predictor import Predictor


class JEPA(nn.Module):
    def __init__(
        self,
        encoder_args,
        predictor_args,
        action_decoder_args=None,
    ):
        super().__init__()
        self.predictor = Predictor(predictor_args)
        self.bottleneck_type = self.predictor.bottleneck_type

        pred_dim = self.predictor.dim

        encoder_args = dict(encoder_args)
        if encoder_args.get("arch", "vit-s").startswith("convnext"):
            encoder_args.setdefault("dim", pred_dim)
        encoder_args = build_encoder_config(encoder_args)

        self.enc_dim = encoder_args["dim"]
        self.pred_dim = pred_dim
        self.dim = self.enc_dim

        self.encoder = build_encoder(encoder_args)

        if self.enc_dim != self.pred_dim:
            self.proj_in = nn.Linear(self.enc_dim, self.pred_dim, bias=False)
            self.proj_out = nn.Linear(self.pred_dim, self.enc_dim, bias=False)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()

        self.action_decoder = None
        if action_decoder_args and action_decoder_args.get("enabled", False):
            decoder_args = dict(action_decoder_args)
            decoder_args.setdefault("in_dim", self.dim)
            self.action_decoder = ActionDecoder(decoder_args)

        self.context = self.predictor.context

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def has_bottleneck(self):
        return self.predictor.has_bottleneck

    def encode(self, x):
        B, T, *_ = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        state = self.encoder(x)["register"]
        return rearrange(state, "(b t) n d -> b t n d", b=B, t=T)

    def predict_all(self, state):
        """Prediction for every position in the sequence.

        With no bottleneck, returns (B, T, N, D) where output[:, t] predicts state[:, t+1].
        With fsq/vae, returns (B, T-1, N, D) conditioned on the inferred latent.
        """
        proj_state = self.proj_in(state)
        pred = self.predictor(proj_state)["pred"]
        return self.proj_out(pred)

    def predict(self, state):
        """Mean-style one-step prediction of frame following `state`.

        For bottleneck=none, uses the unconditional predictor output at [:, -1].
        For fsq/vae, this is not defined (use `sample` instead).
        """
        if self.has_bottleneck:
            raise ValueError("predict() is unconditional; use sample() with a bottleneck")
        proj_state = self.proj_in(state)
        pred = self.predictor(proj_state)["pred"]
        return self.proj_out(pred[:, -1])

    def sample_mean(self, state):
        mu = self.predict(state)
        D = mu.shape[-1]
        sq_norm = (mu * mu).sum(dim=-1, keepdim=True) / D
        sigma = (1.0 - sq_norm).clamp_min(0.0).sqrt()
        return mu + sigma * torch.randn_like(mu)

    def sample(self, state, latent=None):
        """Sample next frame. Requires a bottleneck; uses the prior over latents."""
        if not self.has_bottleneck:
            if latent is not None:
                raise ValueError("sample(latent=...) not supported without a bottleneck")
            return self.sample_mean(state)
        return self.proj_out(self.predictor.sample(self.proj_in(state), latent=latent))[:, -1]

    def decode_actions(self, traj):
        """Per-step action predictions, windowed to training context."""
        T = traj.shape[1]
        if T <= self.context:
            return self.action_decoder(traj)["pred"]
        out = []
        for t in range(1, T):
            start = max(0, t + 1 - self.context)
            out.append(self.action_decoder(traj[:, start : t + 1])["pred"][:, -1])
        return torch.stack(out, dim=1)

    def rollout(self, z_0, horizon, use_latent):
        """Autoregressive rollout from z_0 under the model, no planning."""
        sample_fn = self.sample if use_latent else self.sample_mean
        state_hist = z_0[:, None]
        out = []
        for _ in range(horizon):
            z_next = sample_fn(state_hist)[:, None]
            out.append(z_next)
            state_hist = torch.cat([state_hist, z_next], dim=1)
        return torch.cat(out, dim=1)

    def residuals(self, trajectory, use_latent):
        """Per-step prediction error of a trajectory under the transition model."""
        if use_latent:
            pred = self.predict_all(trajectory)
            return (pred - trajectory[:, 1:]).pow(2).mean(dim=-1).mean(dim=2)

        mu_all = self.predict_all(trajectory[:, :-1])
        targets = trajectory[:, 1:]
        D = mu_all.shape[-1]
        alpha = (mu_all * mu_all).sum(dim=-1, keepdim=True) / D
        sigma_sq = (1.0 - alpha).clamp_min(1e-8)
        res = (targets - mu_all).pow(2).sum(dim=-1, keepdim=True) / (D * sigma_sq)
        return res.squeeze(-1).mean(dim=2)

    def forward(self, x, actions=None):
        state = self.encode(x)
        proj_state = self.proj_in(state)

        pred_out = self.predictor(proj_state)
        pred = self.proj_out(pred_out["pred"])

        result = {
            "state": state,
            "pred": pred,
            "action_pred": None,
            "action_log_prob": None,
            "rollout_action_pred": None,
            "rollout_action_log_prob": None,
        }
        for k in ("kl", "mu", "logvar"):
            if k in pred_out:
                result[k] = pred_out[k]

        if self.action_decoder is not None:
            if actions is None:
                raise ValueError("action decoder enabled but actions not provided")
            out = self.action_decoder(state.detach(), actions)
            result["action_pred"] = out.get("pred")
            result["action_log_prob"] = out.get("log_prob")

            if self.has_bottleneck:
                imagined = torch.cat([state[:, :1], pred], dim=1).detach()
                rollout_out = self.action_decoder(imagined, actions)
                rout_pred = rollout_out.get("pred")
                rout_lp = rollout_out.get("log_prob")
                result["rollout_action_pred"] = rout_pred.detach() if rout_pred is not None else None
                result["rollout_action_log_prob"] = rout_lp.detach() if rout_lp is not None else None

        return result
