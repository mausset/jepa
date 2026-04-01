import torch
from einops import rearrange, repeat
from torch import nn

from jepa.models.action_decoder import TransformerActionDecoder
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
        self.predictor_mode = predictor_args.get("mode", "mean")

        # build predictor based on mode
        if self.predictor_mode == "mean":
            mean_args = dict(predictor_args)
            mean_args["fsq_levels"] = None
            self.predictor = Predictor(mean_args)
        elif self.predictor_mode in ("latent", "residual"):
            self.predictor = Predictor(predictor_args)
        else:
            raise ValueError(f"Unknown predictor mode: {self.predictor_mode}")

        encoder_args = dict(encoder_args)
        if encoder_args.get("arch", "vit-s").startswith("convnext"):
            encoder_args.setdefault("dim", self.predictor.dim)
        encoder_args = build_encoder_config(encoder_args)

        self.enc_dim = encoder_args["dim"]
        self.pred_dim = self.predictor.dim
        self.dim = self.enc_dim

        self.encoder = build_encoder(encoder_args)

        # project between encoder and predictor dims if they differ
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
            self.action_decoder = TransformerActionDecoder(decoder_args)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        b, t, *_ = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        state = self.encoder(x)["register"]
        state = rearrange(state, "(b t) ... -> b t ...", b=b, t=t)

        pred = None
        pred_cond = None
        proj_state = self.proj_in(state)

        if self.predictor_mode == "mean":
            pred = self.proj_out(self.predictor(proj_state))
        elif self.predictor_mode == "latent":
            pred_cond = self.proj_out(self.predictor(proj_state))
        elif self.predictor_mode == "residual":
            p, pc = self.predictor.residual_forward(proj_state)
            pred = self.proj_out(p)
            pred_cond = self.proj_out(pc)

        # rollout source: residual mode sums both, otherwise prefer latent
        if self.predictor_mode == "residual":
            rollout_source = pred_cond
        elif pred_cond is not None:
            rollout_source = pred_cond
        else:
            rollout_source = pred

        action_pred = None
        rollout_action_pred = None
        if self.action_decoder is not None:
            pooled_state = state.mean(dim=2).detach()
            action_pred = self.action_decoder(pooled_state)[:, 1:]

            with torch.no_grad():
                imagined = torch.cat([state[:, :1], rollout_source], dim=1).detach()
                rollout_action_pred = self.action_decoder(imagined.mean(dim=2))[:, 1:]

        return {
            "pred": pred,
            "pred_cond": pred_cond,
            "state": state,
            "action_pred": action_pred,
            "rollout_action_pred": rollout_action_pred,
        }
