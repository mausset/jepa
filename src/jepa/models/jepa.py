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
        """One-step-ahead predictions at every position; output[t] predicts state[t+1].

        Chunks the input non-overlappingly: each chunk takes context+1 input
        frames (the predictor's training shape) and yields context predictions.
        Adjacent chunks share one boundary frame as input but never produce
        overlapping output predictions.

        Returns (B, T-1, N, D).
        """
        T = state.shape[1]
        C = self.context
        out = []
        for start in range(0, T - 1, C):
            end = min(start + C + 1, T)
            win = state[:, start:end]
            win_pred = self.proj_out(self.predictor(self.proj_in(win))["pred"])
            if not self.has_bottleneck:
                win_pred = win_pred[:, :-1]
            out.append(win_pred)
        return torch.cat(out, dim=1)

    def predict(self, state):
        """Mean-style one-step prediction of frame following `state`.

        For bottleneck=none, uses the unconditional predictor output at [:, -1].
        For fsq/vae, this is not defined (use `sample` instead).
        """
        if self.has_bottleneck:
            raise ValueError("predict() is unconditional; use sample() with a bottleneck")
        return self.predict_all(state)[:, -1]

    def sample(self, state, latent=None):
        """Sample next frame. Requires a bottleneck; uses the prior over latents."""
        if not self.has_bottleneck:
            raise ValueError("sample() requires a bottleneck (fsq or vae)")
        return self.proj_out(self.predictor.sample(self.proj_in(state), latent=latent))[:, -1]

    def decode_actions(self, traj):
        """Per-step action predictions, windowed to the action decoder's
        training input length (context + 1, matching data.sequence_length)."""
        T = traj.shape[1]
        L = self.context + 1
        if T <= L:
            return self.action_decoder(traj)["pred"]
        out = []
        for t in range(1, T):
            start = max(0, t + 1 - L)
            out.append(self.action_decoder(traj[:, start : t + 1])["pred"][:, -1])
        return torch.stack(out, dim=1)

    def rollout(self, traj, horizon=None, latents=None):
        """Append autoregressive steps to `traj`. Provide exactly one of
        `horizon` or `latents`.

        Each step samples a window of length `context` from the trajectory tail
        so RoPE positions stay within the predictor's training distribution.

        Args:
            traj:    (B, T, N, D) prefix; for a fresh rollout pass z_0[:, None].
            horizon: int number of steps to roll out from the prior.
            latents: (B, H, N, latent_dim) caller-provided latents; H steps.

        Returns:
            (B, T + H, N, D) — prefix included.
        """
        if (horizon is None) == (latents is None):
            raise ValueError("provide exactly one of `horizon`, `latents`")
        if latents is not None and not self.has_bottleneck:
            raise ValueError("`latents` requires a bottleneck (fsq or vae)")
        H = horizon if horizon is not None else latents.shape[1]
        for t in range(H):
            window = traj[:, -self.context :]
            latent_t = None if latents is None else latents[:, t : t + 1]
            z_next = self.sample(window, latent=latent_t)[:, None]
            traj = torch.cat([traj, z_next], dim=1)
        return traj

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
