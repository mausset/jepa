from einops import rearrange, repeat
from torch import nn

from jepa.models.action_decoder import TransformerActionDecoder
from jepa.models.encoder import build_encoder
from jepa.models.encoder import build_encoder_config
from jepa.models.predictor import build_predictor_config
from jepa.models.predictor import Predictor


class JEPA(nn.Module):
    def __init__(
        self,
        encoder_args,
        predictor_args,
        action_decoder_args=None,
    ):
        super().__init__()
        predictor_args = build_predictor_config(predictor_args)
        encoder_args = dict(encoder_args)
        if encoder_args.get("arch", "vit-s").startswith("convnext"):
            encoder_args.setdefault("dim", predictor_args["dim"])
        encoder_args = build_encoder_config(encoder_args)

        self.dim = encoder_args["dim"]
        self.k = predictor_args["k"]

        self.encoder = build_encoder(encoder_args)
        self.predictor = Predictor(predictor_args)

        self.action_decoder = None
        if action_decoder_args and action_decoder_args.get("enabled", False):
            decoder_args = dict(action_decoder_args)
            decoder_args.setdefault("in_dim", self.dim)
            self.action_decoder = TransformerActionDecoder(decoder_args)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_features(self, x):
        return self.encoder(x)  # type: ignore

    def forward(self, x):
        b, t, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")
        state = self.encoder(x)["register"]
        state = rearrange(state, "(b t) ... -> b t ...", b=b, t=t)

        s = repeat(state[:, :-1], "b ... -> (k b) ...", k=self.k)
        pred = self.predictor(s)
        pred = rearrange(pred, "(k b) ... -> k b ...", k=self.k)

        action_pred = None
        if self.action_decoder is not None:
            pooled_state = state.mean(dim=2).detach()
            action_pred = self.action_decoder(pooled_state)[:, 1:]

        return {
            "pred": pred,
            "state": state,
            "action_pred": action_pred,
        }
