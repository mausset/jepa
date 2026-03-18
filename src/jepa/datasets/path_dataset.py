import random

import h5py
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def, types  # type: ignore
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class PathDataset:

    def __init__(self, data_path, seq_len=8):
        self.data = h5py.File(data_path, "r")
        self.seq_len = seq_len

        self.index = []
        for ep in self.data.keys():
            ep_len = self.data[ep].attrs["episode_length"]
            if ep_len <= self.seq_len:  # type: ignore
                continue
            s = ep_len - seq_len  # type: ignore
            self.index.extend([(ep, i) for i in range(s)])
        random.shuffle(self.index)
        self.ptr = 0

    def _next_epoch(self):
        random.shuffle(self.index)
        self.ptr = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.index)

    def __next__(self):
        if self.ptr >= len(self.index):
            self._next_epoch()
            raise StopIteration

        (ep, i) = self.index[self.ptr]
        frames = self.data[ep]["frames"][i : i + self.seq_len]  # type: ignore
        actions = self.data[ep]["actions"][i : i + self.seq_len - 1]  # type: ignore
        # prop = np.zeros((0,))
        # if "prop" in self.data[ep]:  # type: ignore
        #     prop = np.array(self.data[ep]["prop"][i : i + self.seq_len - 1])  # type: ignore

        self.ptr += 1

        return np.array(frames), np.array(actions)


@pipeline_def()
def pipeline(source):

    frames, actions = fn.external_source(
        source,
        num_outputs=2,
        device="gpu",
        batch=False,
        layout=["FHWC"],
    )

    frames = fn.crop_mirror_normalize(
        frames,
        dtype=types.FLOAT,  # type: ignore
        output_layout="FHWC",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    return frames, actions


def build_path_dataloader(config, local_rank=0, global_rank=0, world_size=1, val=False):

    path = config["data_path"] if not val else config["val_path"]

    source = PathDataset(
        data_path=path,
        seq_len=config["seq_len"],
    )

    pipe = pipeline(
        source,
        num_threads=config["num_threads"],
        batch_size=config["batch_size"],
        device_id=local_rank,
        # shard_id=global_rank,
        # num_shards=world_size,
    )

    dataloader = DALIGenericIterator(
        pipe,
        output_map=["frames", "actions"],
        last_batch_policy=LastBatchPolicy.DROP,
        size=-1,
        # reader_name="Reader",
    )

    return dataloader


if __name__ == "__main__":

    path_dataset = PathDataset("data/env_paths/pointmaze_umaze_v3.h5")

    pipe = pipeline(path_dataset, num_threads=1, batch_size=8, device_id=0)

    dataloader = DALIGenericIterator(
        pipe,
        output_map=["frames", "actions"],
        last_batch_policy=LastBatchPolicy.DROP,
    )

    for x in dataloader:
        print(x[0]["frames"].shape)
        exit()
