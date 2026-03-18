import torch
import os
from torch import distributed as dist

from einops import rearrange


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, global_rank, world_size


def all_gather(x):
    x_list = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(x_list, x, async_op=False)
    x_list[dist.get_rank()] = x

    return torch.cat(x_list, dim=0)


@torch.no_grad()
def all_reduce_cov(x):

    x = rearrange(x, "... d -> (...) d")
    B = torch.tensor(x.shape[0], device=x.device)
    mu = x.sum(dim=0)
    gram = x.T @ x

    dist.all_reduce(B, dist.ReduceOp.SUM)
    dist.all_reduce(mu, dist.ReduceOp.SUM)
    dist.all_reduce(gram, dist.ReduceOp.SUM)

    mu = mu / B

    return (gram - B * mu[:, None] @ mu[None, :]) / (B - 1)
