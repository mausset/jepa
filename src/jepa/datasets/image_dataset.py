import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import trivial_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator


@pipeline_def(enable_conditionals=True)  # type: ignore
def image_pipe(file_root, resolution=224, shard_id=0, num_shards=1, val=False):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=not val,
        name="Reader",
    )
    images = fn.decoders.image(
        jpegs,
        device="cpu",
    )

    if val:
        mirror = False
    else:
        shapes = fn.peek_image_shape(jpegs)
        images = trivial_augment.trivial_augment_wide(images, shape=shapes)
        mirror = fn.random.coin_flip(probability=0.5)

    images = images.gpu()
    images = fn.resize(
        images,
        device="gpu",
        resize_shorter=resolution,
        interp_type=types.DALIInterpType.INTERP_LINEAR,
        antialias=True,
    )

    images = fn.crop_mirror_normalize(
        images,  # type: ignore
        dtype=types.FLOAT,  # type: ignore
        output_layout="HWC",
        crop=(resolution, resolution),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    return images, labels


def build_dali_iterators(args, local_rank=0, global_rank=0, world_size=1):
    train_root = args["data_root"]
    val_root = args["val_root"]
    train_pipe = image_pipe(
        file_root=train_root,
        resolution=args["resolution"],
        batch_size=args["batch_size"],
        device_id=local_rank,
        shard_id=global_rank,
        num_shards=world_size,
        num_threads=args["num_threads"],
    )
    val_pipe = image_pipe(
        file_root=val_root,
        resolution=args["resolution"],
        batch_size=args["batch_size"],
        device_id=local_rank,
        shard_id=global_rank,
        num_shards=world_size,
        num_threads=args["num_threads"],
        val=True,
    )
    train_loader = DALIGenericIterator(
        train_pipe,
        ["data", "label"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )
    val_loader = DALIGenericIterator(
        val_pipe,
        ["data", "label"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )
    return train_loader, val_loader


class LeJEPAMultiCropTransform:
    """
    LeJEPA / DINO-style multi-view SSL
    """

    def __init__(
        self,
        *,
        global_crop_size=224,
        local_crop_size=96,
        num_global_views=2,
        num_local_views=6,
        enable_photometric=True,
    ):
        self.num_global_views = int(num_global_views)
        self.num_local_views = int(num_local_views)
        assert self.num_global_views > 0
        assert self.num_local_views > 0

        # LeJEPA / DINO-style crop params
        self.global_crop = v2.RandomResizedCrop(
            size=(global_crop_size, global_crop_size),
            scale=(0.30, 1.00),
            antialias=True,
        )
        self.local_crop = v2.RandomResizedCrop(
            size=(local_crop_size, local_crop_size),
            scale=(0.05, 0.30),
            antialias=True,
        )

        if enable_photometric:
            photometric = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                v2.RandomSolarize(threshold=128.0, p=0.2),
            ]
        else:
            photometric = []

        to_tensor_norm = [
            v2.ToImage(),  # uint8, CxHxW
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            v2.Lambda(lambda x: x.permute(1, 2, 0).contiguous()),
        ]

        if enable_photometric:
            photometric_local = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                v2.RandomSolarize(threshold=128.0, p=0.2),
            ]
        else:
            photometric_local = []

        self.global_transform = v2.Compose(
            [self.global_crop] + photometric + to_tensor_norm
        )
        self.local_transform = v2.Compose(
            [self.local_crop] + photometric_local + to_tensor_norm
        )

    def __call__(self, img):
        global_views = [
            self.global_transform(img) for _ in range(self.num_global_views)
        ]
        local_views = [self.local_transform(img) for _ in range(self.num_local_views)]

        global_views = torch.stack(global_views, dim=0)
        local_views = torch.stack(local_views, dim=0)
        return {"global": global_views, "local": local_views}


def build_torch_iterators(args, local_rank=0, global_rank=0, world_size=1):
    train_root = args["data_root"]
    val_root = args["val_root"]

    batch_size = args["batch_size"]
    num_workers = args.get("num_workers", args.get("num_threads", 8))

    num_global_views = args.get("num_global_views", 1)
    num_local_views = args.get("num_local_views", 1)
    enable_photometric = args.get("photometric", True)

    global_crop_size = args.get("resolution", 224)

    transform = LeJEPAMultiCropTransform(
        global_crop_size=global_crop_size,
        local_crop_size=98,
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        enable_photometric=enable_photometric,
    )

    train_ds = ImageFolder(train_root, transform=transform)
    val_ds = ImageFolder(val_root, transform=transform)

    sampler = None
    val_sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
        )

        val_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
