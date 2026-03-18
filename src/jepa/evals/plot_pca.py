import argparse, math, os, shutil, torch, yaml, matplotlib.pyplot as plt
from jepa.datasets.video_dataset import build_dali_iterators
from jepa.models.model import JEPA

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3).cuda()
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3).cuda()


def parse_args():
    p = argparse.ArgumentParser("PCA visualisation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument(
        "--n_pcs",
        type=int,
        default=3,
        help="Number of PCs to plot (>=3). First three feed the RGB composite.",
    )
    return p.parse_args()


def load_config(path):
    return yaml.load(open(path), Loader=yaml.FullLoader)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    dcfg = cfg["data"]
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    m = JEPA(
        cfg["encoder"],
        cfg["predictor"],
        n_registers=cfg["registers"],
        resolution=dcfg["resolution"],
        context=dcfg["sequence_length"] - 1,
    ).cuda()
    m.load_state_dict(state)
    m.eval().requires_grad_(False)
    return m


def unnorm(x):
    return torch.clamp(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


# ────────── plotting ──────────
def plot_frame(img_np, heats, idx, n_pcs):
    """
    img_np (H,W,3), heats (n_pcs,res,res) in [0,1]
    """
    comp = heats[:3].permute(1, 2, 0).cpu().numpy()  # RGB map with PCs 1‑3
    cols = n_pcs + 2  # composite + n_pcs + original
    plt.figure(figsize=(4 * cols, 4))

    plt.subplot(1, cols, 1)
    plt.imshow(comp)
    plt.axis("off")
    plt.title("RGB (1‑3)")
    for k in range(n_pcs):
        plt.subplot(1, cols, k + 2)
        plt.imshow(heats[k].cpu(), cmap="viridis")
        plt.axis("off")
        plt.title(f"PC{k + 1}")
    plt.subplot(1, cols, cols)
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Frame")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/pca_{idx}.png", bbox_inches="tight")
    plt.close()


# ────────── main ──────────
def main():
    args = parse_args()
    n_pcs = max(args.n_pcs, 3)  # enforce minimum 3

    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots")

    cfg = load_config(args.config)
    model = load_model(args.checkpoint)
    _, val_loader = build_dali_iterators(cfg["data"])

    seq = next(val_loader)[0]["data"][0][: args.num_images].cuda()  # (T,H,W,3)
    res = int(math.sqrt(model.n_patches))

    # forward once
    with torch.no_grad(), torch.amp.autocast("cuda"):
        feats = model.forward_features(seq, register_attn=False)["features"]  # (T,P,D)

    # joint PCA
    flat = feats.reshape(-1, feats.shape[-1]).float()
    flat -= flat.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    pcs = vh[:n_pcs]  # (n_pcs,D)

    imgs = unnorm(seq).cpu().numpy()  # (T,H,W,3)

    for i in range(feats.shape[0]):
        scores = feats[i] @ pcs.T  # (P,n_pcs)
        smin, smax = scores.min(0)[0], scores.max(0)[0]
        scores = (scores - smin) / (smax - smin + 1e-6)
        heats = scores.T.view(n_pcs, res, res)
        plot_frame(imgs[i], heats, i, n_pcs)


if __name__ == "__main__":
    main()
