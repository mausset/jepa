import torch
from torch import nn
from torch import distributed as dist

from torch.distributed.nn.functional import all_reduce
from torch.distributed.nn import ReduceOp


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def maybe_all_reduce(x, op=ReduceOp.AVG):
    if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
        return all_reduce(x, op)
    else:
        return x


class UnivariateTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = torch.distributions.normal.Normal(0, 1)

    @property
    def world_size(self):
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1


class EppsPulley(UnivariateTest):
    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32, device="cuda")
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32, device="cuda")
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())  # type: ignore
        self.register_buffer("weights", weights * self.phi)  # type: ignore

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = maybe_all_reduce(cos_mean)
        sin_mean = maybe_all_reduce(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()  # type: ignore

        # Weighted integration
        return (err @ self.weights) * N * self.world_size  # type: ignore


class SlicingUnivariateTest(nn.Module):
    def __init__(
        self,
        univariate_test,
        num_slices: int,
        sampler: str = "gaussian",
    ):
        super().__init__()
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.register_buffer(
            "global_step", torch.zeros((), dtype=torch.long, device="cuda")
        )

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = maybe_all_reduce(
                self.global_step.clone(), op=ReduceOp.MAX
            )  # type: ignore
            seed = global_step_sync.item()  # type: ignore
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)  # type: ignore

        stats = self.univariate_test(x @ A)
        return stats.mean()
