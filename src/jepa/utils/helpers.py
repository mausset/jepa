import math
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.distributed as dist


def bicubic_interpolate_pos_embed(
    pe: torch.Tensor,
    old_grid,
    new_grid,
) -> torch.Tensor:
    """
    Bicubic-interpolate ViT patch positional embeddings.
    """
    h0, w0 = old_grid
    h1, w1 = new_grid

    if (h0, w0) == (h1, w1):
        return pe

    pe_grid = rearrange(pe, "b (h w) d -> b d h w", h=h0, w=w0)
    pe_grid = F.interpolate(
        pe_grid,
        size=(h1, w1),
        mode="bicubic",
        align_corners=False,
    )
    return rearrange(pe_grid, "b d h w -> b (h w) d")


def block_attention_mask(x):
    """Create block attention mask for causal attention."""
    _, T, N, _ = x.shape
    mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).bool()
    mask = mask.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)
    return mask


def block_cross_attention_mask(x, context, causal=False):
    """Create block attention mask for causal attention."""
    _, T, N, _ = x.shape
    _, T, M, _ = context.shape
    if causal:
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).bool()
    else:
        mask = torch.eye(T, T, device=x.device).bool()
    mask = mask.repeat_interleave(N, dim=0).repeat_interleave(M, dim=1)

    return mask


def linear_warmup(step, total_steps, base_value, start_ratio=0.0, ratio=0.1):
    if step < total_steps * start_ratio:
        return 0.0

    if ratio == 0.0:
        return base_value

    start_step = total_steps * start_ratio
    return min(base_value, base_value * (step - start_step) / (total_steps * ratio))


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def spectrum(x):
    """
    Computes the eigenvalues of the covariance matrix of x.
    Handles distributed gathering to ensure sufficient sample size.
    """

    B, D = x.shape

    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    cov = (x_centered.T @ x_centered) / (B - 1)

    eigs = torch.linalg.eigvalsh(cov)

    eigs = torch.clamp(eigs, min=0)
    eigs = eigs.sort(descending=True).values
    return eigs


@torch.no_grad()
def participation_ratio(eigvals):
    """
    Computes the Participation Ratio (Effective Rank)
    Input: Eigenvalues of the covariance matrix (NOT the embeddings)
    """
    sum_eigs = eigvals.sum()
    if sum_eigs == 0:
        return torch.tensor(0.0, device=eigvals.device)

    # Formula: (Sum \lambda)^2 / Sum (\lambda^2)
    return sum_eigs.pow(2) / eigvals.pow(2).sum()


@torch.no_grad()
def rankme(eigvals, epsilon=1e-7):
    """
    Computes the RankMe metric (Entropy-based Rank)
    Input: Eigenvalues of the covariance matrix
    """
    # L1 normalization to create a probability distribution
    p = eigvals / (eigvals.sum() + epsilon)

    # Entropy: - sum(p * log(p))
    entropy = -torch.sum(p * torch.log(p + epsilon))

    # RankMe = exp(Entropy)
    return torch.exp(entropy)


@torch.no_grad()
def compute_alpha(eigvals, min_idx=0, max_idx=None):
    """
    Fits a power law to the spectrum: lambda_i approx C * i^(-alpha)
    Input: Sorted Eigenvalues
    """
    if max_idx is None:
        max_idx = len(eigvals)

    # We typically ignore the very tail (numerical noise) and the very head (outliers)
    # Taking the top 90% effective eigenvalues is a common heuristic
    valid_eigs = eigvals[min_idx:max_idx]
    valid_eigs = valid_eigs[valid_eigs > 1e-6]  # Filter zeros

    if len(valid_eigs) < 3:
        return float("nan")

    idxs = torch.arange(
        1, len(valid_eigs) + 1, device=eigvals.device, dtype=torch.float32
    )

    log_i = torch.log(idxs)
    log_v = torch.log(valid_eigs)

    # Linear regression in log-log space
    # y = -alpha * x + b
    x_mean = log_i.mean()
    y_mean = log_v.mean()

    num = ((log_i - x_mean) * (log_v - y_mean)).sum()
    den = ((log_i - x_mean) ** 2).sum()

    slope = num / (den + 1e-8)
    return -slope.item()  # Alpha is negative slope


def mi_proxy(lambdas, eps=1e-6, base2=False):
    """
    Gaussian MI proxy from correlation eigenvalues.

    I = -1/2 * sum_i log(1 - lambda_i^2)

    Args:
        lambdas: 1D tensor (d,) or anything broadcastable to that.
        eps: small number to keep |lambda| < 1 for numerical stability.
        base2: if True, return MI in bits (otherwise in nats).

    Returns:
        Scalar tensor: mutual information proxy.
    """
    lambdas = torch.as_tensor(lambdas)

    # Clamp to avoid log(0) and invalid values when |lambda| >= 1
    lambdas = torch.clamp(lambdas, min=-1 + eps, max=1 - eps)

    mi_nats = -0.5 * torch.log(1.0 - lambdas**2).sum()

    if base2:
        mi_nats = mi_nats / torch.log(
            torch.tensor(2.0, device=mi_nats.device, dtype=mi_nats.dtype)
        )

    return mi_nats


class MeanMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.total += value
        self.count += 1

    @property
    def avg(self):
        if self.total is None:
            return None
        return self.total / self.count if self.count else 0.0


class SmoothedValue:
    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self.last = None
        self.value = None

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self.last is None:
            self.value = value
        else:
            self.value = self.beta * self.last + (1 - self.beta) * value
        self.last = self.value

    def __call__(self):
        return self.value
