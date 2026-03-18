import argparse

import io
from PIL import Image

import math
import os
import shutil

import matplotlib.pyplot as plt
import torch
import yaml
from einops import rearrange, reduce

from jepa.datasets.video_dataset import build_dali_iterators
from jepa.models.model import JEPA

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3).cuda()
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3).cuda()


def parse_args():
    parser = argparse.ArgumentParser("PCA script")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--num_images", type=int, default=8, help="Number of images to process"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model(checkpoint_path):
    ckp = torch.load(checkpoint_path, map_location="cpu")
    model_config = ckp["config"]
    model_data_config = model_config["data"]
    state_dict = {k.replace("module.", ""): v for k, v in ckp["model"].items()}

    model = JEPA(
        model_config["encoder"],
        model_config["predictor"],
        n_registers=model_config["registers"],
        resolution=model_data_config["resolution"],
        context=model_data_config["sequence_length"] - 1,
    ).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


def unnormalize_images(imgs):
    return torch.clamp(imgs * IMAGENET_STD + IMAGENET_MEAN, min=0, max=1)


def plot_attn(img, attn, res, idx):
    heads = attn.shape[0]
    r = math.ceil(math.sqrt(heads + 1))
    c = math.ceil(math.sqrt(heads + 1))
    plt.figure(figsize=(30, 30))
    for i in range(attn.shape[0]):
        plt.subplot(r, c, 1 + i)
        attn_img = rearrange(attn[i], "(h w) -> h w", h=res, w=res)
        plt.imshow(attn_img.cpu())
        plt.axis("off")
    plt.subplot(r, c, heads + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(
        f"plots/attn_{idx}.png", format="png", bbox_inches="tight", pad_inches=0
    )
    plt.close()
    # image = Image.open(buf).convert("RGBA")
    # buf.close()
    # return image


def save_gif(image_list, filename, duration=500, loop=0):
    if not image_list:
        raise ValueError("The image list is empty.")
    first_image = image_list[0]
    additional_images = image_list[1:]
    first_image.save(
        os.path.join("plots", filename),
        save_all=True,
        append_images=additional_images,
        duration=duration,
        loop=loop,
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    # Clear and recreate the plots directory
    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots", exist_ok=True)

    model = load_model(args.checkpoint)
    n_registers = model.n_registers

    train_loader, val_loader = build_dali_iterators(config["data"])
    data_loader = val_loader
    res = int(math.sqrt(model.n_patches))

    img = next(data_loader)[0]["data"][0]

    with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
        attn_all = model.forward_features(img, register_attn=True)["attn"]

    attn_all = reduce(attn_all, "b h n d -> b n d", "mean")
    attn_all = torch.clamp(attn_all, min=0, max=1)
    print(attn_all.shape)

    plots = []
    img = unnormalize_images(img).cpu()
    for i in range(attn_all.shape[0]):
        plots.append(plot_attn(img[i], attn_all[i], res, i))

    # save_gif(plots, "attn.gif", 8)


if __name__ == "__main__":
    main()
