import glob
import os
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def(enable_conditionals=True)  # type: ignore
def video_pipe(
    filenames=[],
    sequence_length=1,
    stride=1,
    step=-1,
    resolution=224,
    shard_id=0,
    num_shards=1,
    min_area=0.3,
    max_area=1.0,
    min_aspect_ratio=0.75,
    max_aspect_ratio=1.35,
    photometric=True,
    crop_aspect=True,
    brightness=0.4,
    contrast=0.4,
    saturation=0.2,
    hue=0.1,
    val=False,
):
    # Read video files
    videos = fn.readers.video_resize(
        device="gpu",
        filenames=filenames,
        resize_shorter=resolution,
        sequence_length=sequence_length,
        stride=stride,
        step=step,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,  # Always shuffle, otherwise val perf is not indicative
        initial_fill=16,
        prefetch_queue_depth=1,
        name="Reader",
    )

    if isinstance(videos, list):
        videos = videos[0]

    if val:
        mirror = False
    else:
        if crop_aspect:
            videos = fn.random_resized_crop(  # type: ignore
                videos,  # type: ignore
                device="gpu",
                size=(resolution, resolution),
                random_area=[min_area, max_area],
                random_aspect_ratio=[min_aspect_ratio, max_aspect_ratio],
                interp_type=types.DALIInterpType.INTERP_LINEAR,  # type: ignore
                antialias=True,
            )

        if photometric:
            if photometric == "per_frame":
                shape = (sequence_length,)

                def apply(x):  # type: ignore
                    return fn.per_frame(x)

            else:
                shape = (1,)

                def apply(x):
                    return x

            # Color jitter
            brightness = fn.random.uniform(
                range=[1 - brightness, 1 + brightness], shape=shape
            )
            contrast = fn.random.uniform(
                range=[1 - contrast, 1 + contrast], shape=shape
            )
            saturation = fn.random.uniform(
                range=[1 - saturation, 1 + saturation], shape=shape
            )
            hue_delta = fn.random.uniform(range=[-hue, hue], shape=shape)

            videos = fn.color_twist(  # type: ignore
                videos,
                brightness=apply(brightness),  # type: ignore
                contrast=apply(contrast),  # type: ignore
                saturation=apply(saturation),  # type: ignore
                hue=apply(hue_delta),  # type: ignore
            )

            gray_coin = fn.random.coin_flip(probability=0.2, shape=shape)
            gray_coin = fn.cast(gray_coin, dtype=types.DALIDataType.FLOAT)
            videos = fn.hsv(videos, saturation=apply(gray_coin))  # type: ignore

        # Flip
        mirror = fn.random.coin_flip(probability=0.5)

    videos = fn.crop_mirror_normalize(
        videos,  # type: ignore
        dtype=types.FLOAT,  # type: ignore
        output_layout="FHWC",
        crop=(resolution, resolution),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )

    return videos


def build_dali_iterators(args, local_rank=0, global_rank=0, world_size=1, seed=-1):
    print("Building DALI iterators...")

    assert "data_root" in args, "need to provide data root"
    assert "val_root" in args, "need to provide val root"

    root = args["data_root"]
    filenames = glob.glob("**/*.mp4", root_dir=root, recursive=True)
    filenames = [os.path.join(root, f) for f in filenames]
    print(f"Found training videos: {len(filenames)}")
    val_root = args["val_root"]
    filenames_val = glob.glob("**/*.mp4", root_dir=val_root, recursive=True)
    filenames_val = [os.path.join(val_root, f) for f in filenames_val]
    print(f"Found validation videos: {len(filenames_val)}")

    train_pipe = video_pipe(
        filenames=filenames,
        sequence_length=args["sequence_length"],
        resolution=args["resolution"],
        num_threads=args["num_threads"],
        batch_size=args["batch_size"],
        photometric=args["photometric"],
        crop_aspect=args["crop_aspect"],
        stride=args["stride"],
        step=args["step"],
        device_id=local_rank,
        shard_id=global_rank,
        num_shards=world_size,
        seed=seed,
    )

    val_pipe = video_pipe(
        filenames=filenames_val,
        sequence_length=args["sequence_length"],
        resolution=args["resolution"],
        num_threads=args["num_threads"],
        batch_size=args["batch_size"],
        photometric=args["photometric"],
        crop_aspect=args["crop_aspect"],
        stride=args["stride"],
        step=-1,
        device_id=local_rank,
        shard_id=global_rank,
        num_shards=world_size,
        # val=True,
        seed=seed,
    )

    train_pipe.build()
    train_iter = DALIGenericIterator(
        pipelines=train_pipe,
        output_map=["data"],
        last_batch_policy=LastBatchPolicy.DROP,
        reader_name="Reader",
        auto_reset=False,
    )

    val_pipe.build()
    val_iter = DALIGenericIterator(
        pipelines=val_pipe,
        output_map=["data"],
        last_batch_policy=LastBatchPolicy.DROP,
        reader_name="Reader",
    )

    print("Finished building DALI iterators.")

    return train_iter, val_iter
