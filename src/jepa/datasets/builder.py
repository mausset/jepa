from jepa.datasets.image_dataset import build_dali_iterators as build_image_iterators
from jepa.datasets.toy_env_dataset import build_toy_env_iterators
from jepa.datasets.video_dataset import build_dali_iterators as build_video_iterators


def build_iterators(
    config: dict,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    seed: int = -1,
):
    kind = config.get("kind", "video")
    if kind == "video":
        return build_video_iterators(
            config,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            seed=seed,
        )
    if kind == "toy_env":
        return build_toy_env_iterators(
            config,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            seed=seed,
        )
    if kind == "image":
        return build_image_iterators(
            config,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
        )

    raise ValueError(f"Unknown dataset kind: {kind}")
