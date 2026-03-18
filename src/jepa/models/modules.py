import torch.nn.functional as F
from torch import nn


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, expansion=6) -> None:
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(((dim * expansion * 2 / 3) // 8) * 8)

        self.w1 = nn.Linear(self.dim, self.hidden_dim)
        self.w2 = nn.Linear(self.dim, self.hidden_dim)
        self.w3 = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(x)
