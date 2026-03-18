import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
) -> torch.Tensor:
    """
    Args:
        x: Generated samples in feature space, shape (N, D)
        y_pos: Positive (real data) samples, shape (N_pos, D)
        y_neg: Negative (generated) samples, shape (N_neg, D)
        temperature: Temperature for softmax (smaller = sharper)
        mask_self: Whether to mask self-distances (when y_neg == x)

    Returns:
        V: Drifting field, shape (N, D)
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    # 1. Compute pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    # 2. Mask self-distances (when y_neg contains x)
    if mask_self and N == N_neg:
        mask = torch.eye(N, device=device) * 1e6
        dist_neg = dist_neg + mask

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 4. Concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Normalize along BOTH dimensions (key insight from paper)
    A_row = torch.softmax(logit, dim=1)  # softmax over y (columns)
    A_col = torch.softmax(logit, dim=0)  # softmax over x (rows)
    A = torch.sqrt(A_row * A_col)  # geometric mean

    # 6. Split back to pos and neg
    A_pos = A[:, :N_pos]
    A_neg = A[:, N_pos:]

    # 7. Compute weights (cross-weighting from paper)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)

    # 8. Compute drift
    drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
    drift_neg = torch.mm(W_neg, y_neg)  # (N, D)

    V = drift_pos - drift_neg

    return V
