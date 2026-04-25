import zuko
from einops import repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from jepa.models.predictor import PredictorBlock
from jepa.utils.helpers import block_attention_mask


class FlowHead(nn.Module):
    """Conditional neural spline flow for continuous actions.

    Samples are continuous and multi-modal; training uses exact log-likelihood.
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        transforms: int = 4,
        bins: int = 8,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.flow = zuko.flows.NSF(
            features=action_dim,
            context=context_dim,
            transforms=transforms,
            bins=bins,
            hidden_features=tuple(hidden),
        )

    def log_prob(self, actions, context):
        return self.flow(context).log_prob(actions)

    def sample(self, context):
        return self.flow(context).rsample()


class ActionDecoder(nn.Module):
    """Causal transformer mapping tokenized states (B, T, N, D) to per-step actions.

    Output depends on `action_type`:
      - "discrete": forward returns {"pred": (B, T-1, num_classes)} logits.
      - "continuous": flow head. forward(x, actions) returns {"log_prob": (B, T-1)};
        forward(x) returns {"pred": (B, T-1, action_dim)} via a sample.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.dim = int(config["in_dim"])
        self.heads = int(config.get("heads", 8))
        self.depth = int(config.get("depth", 2))
        self.action_dim = int(config["action_dim"])
        self.action_type = str(config.get("action_type", "continuous"))
        self.dropout = float(config.get("dropout", 0.0))

        self.rope = RotaryEmbedding(PredictorBlock.HEAD_DIM, theta=100.0)
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(self.dim, self.heads, self.rope, dropout=self.dropout)
                for _ in range(self.depth)
            ]
        )

        if self.action_type == "discrete":
            self.head = nn.Linear(self.dim, self.action_dim)
        elif self.action_type == "continuous":
            self.head = FlowHead(
                action_dim=self.action_dim,
                context_dim=self.dim,
                transforms=int(config.get("flow_transforms", 4)),
                bins=int(config.get("flow_bins", 8)),
                hidden=tuple(config.get("flow_hidden", (128, 128))),
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type!r}")

    def context_features(self, x):
        B = x.shape[0]
        mask = repeat(block_attention_mask(x), "m n -> b m n", b=B)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        return x.mean(dim=2)[:, 1:]

    def forward(self, x, actions=None):
        ctx = self.context_features(x)

        if self.action_type == "discrete":
            return {"pred": self.head(ctx)}

        if actions is None:
            return {"pred": self.head.sample(ctx)}

        return {"log_prob": self.head.log_prob(actions, ctx)}
