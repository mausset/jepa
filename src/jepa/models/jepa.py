from einops import rearrange, repeat
from torch import nn

from jepa.models.encoder import ViT
from jepa.models.predictor import Predictor


class JEPA(nn.Module):
    def __init__(
        self,
        encoder_args,
        predictor_args,
    ):
        super().__init__()
        self.dim = encoder_args["dim"]
        self.k = predictor_args["k"]

        self.encoder = ViT(encoder_args)
        self.predictor = Predictor(predictor_args)

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

        return {
            "pred": pred,
            "state": state,
        }
