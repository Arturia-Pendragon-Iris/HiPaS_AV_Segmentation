"""
Train a single STS stage.

Usage:
    python train.py --config config.yaml

The stage index and all paths are read from config.yaml.
To train stages sequentially:
    1. Edit config.yaml: set stage=0, pretrained_stage_paths=[], checkpoint_dir=checkpoints/stage_0
    2. Run python train.py
    3. Edit config.yaml: set stage=1, pretrained_stage_paths=[checkpoints/stage_0/best.pth], ...
    4. Run precompute_prior.py to precompute stage-0 priors for all training samples
    5. Run python train.py
    6. Repeat for stages 2 and 3.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from HiPaS.model import HiPaSNet
from av_training.dataset import AVDataset, collate_fn
from av_training.loss import STSLoss


# ── helpers ──────────────────────────────────────────────────────────────────

def dice_metric(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    p = (pred > thresh).float()
    t = target.float().to(p.device)
    intersection = (p * t).sum()
    return (2.0 * intersection / (p.sum() + t.sum() + 1e-6)).item()


def load_stage_models(paths: list[str], cfg: dict, device: torch.device) -> list[nn.Module]:
    models = []
    for path in paths:
        m = HiPaSNet(
            in_channels=cfg["model"]["in_channels"],
            mid_channels=cfg["model"]["mid_channels"],
            out_channels=cfg["model"]["out_channels"],
            r=cfg["model"]["r"],
        )
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models


@torch.no_grad()
def cascade_prior(
    x: torch.Tensor,
    prior_models: list[nn.Module],
) -> torch.Tensor | None:
    """Run the chain of frozen prior models to get the prior for the current stage."""
    prior = None
    for model in prior_models:
        prior = model(x, prior)
    return prior


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: STSLoss,
    stage: int,
    prior_models: list[nn.Module],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train(is_train)
    total_loss = total_dice_a = total_dice_v = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["input"].to(device)
            masks = [m.to(device) for m in batch["masks"]]
            prior = batch["prior"]

            # If no precomputed prior, fall back to live cascade inference
            if prior is None and stage > 0:
                prior = cascade_prior(x, prior_models)
            elif prior is not None:
                prior = prior.to(device)

            pred = model(x, prior)
            loss, components = criterion(pred, masks, stage, prior)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            target = masks[stage]
            total_loss += loss.item()
            total_dice_a += dice_metric(pred[:, 0], target[:, 0])
            total_dice_v += dice_metric(pred[:, 1], target[:, 1])
            n += 1

    return {
        "loss": total_loss / n,
        "dice_artery": total_dice_a / n,
        "dice_vein": total_dice_v / n,
        "dice_mean": (total_dice_a + total_dice_v) / (2 * n),
    }


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = cfg["training"]["stage"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    # Datasets and loaders
    train_ds = AVDataset(
        dcfg["train_list"],
        stage=stage,
        crop_size=tcfg["crop_size"],
        npz_key=dcfg["npz_key"],
        normalize_hu=dcfg["normalize_hu"],
        hu_min=dcfg["hu_min"],
        hu_max=dcfg["hu_max"],
        augment=True,
    )
    val_ds = AVDataset(
        dcfg["val_list"],
        stage=stage,
        crop_size=tcfg["crop_size"],
        npz_key=dcfg["npz_key"],
        normalize_hu=dcfg["normalize_hu"],
        hu_min=dcfg["hu_min"],
        hu_max=dcfg["hu_max"],
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=tcfg["pin_memory"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=tcfg["pin_memory"],
        collate_fn=collate_fn,
    )

    # Prior models (frozen stages 0 .. stage-1)
    # Only used if no precomputed prior is available in the dataset
    prior_models = load_stage_models(tcfg["pretrained_stage_paths"], cfg, device)

    # Current stage model
    model = HiPaSNet(
        in_channels=cfg["model"]["in_channels"],
        mid_channels=cfg["model"]["mid_channels"],
        out_channels=cfg["model"]["out_channels"],
        r=cfg["model"]["r"],
    ).to(device)

    criterion = STSLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"]
    )

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    for epoch in range(1, tcfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, stage,
                                  prior_models, optimizer, device, is_train=True)
        scheduler.step()

        msg = (f"Epoch {epoch:03d} | "
               f"loss={train_metrics['loss']:.4f}  "
               f"dice_A={train_metrics['dice_artery']:.4f}  "
               f"dice_V={train_metrics['dice_vein']:.4f}")

        if epoch % tcfg["val_interval"] == 0:
            val_metrics = run_epoch(model, val_loader, criterion, stage,
                                    prior_models, None, device, is_train=False)
            msg += (f"  | val_loss={val_metrics['loss']:.4f}  "
                    f"val_dice={val_metrics['dice_mean']:.4f}")

            if tcfg["save_best"] and val_metrics["dice_mean"] > best_dice:
                best_dice = val_metrics["dice_mean"]
                torch.save(model.state_dict(), ckpt_dir / "best.pth")

        print(msg)

        if epoch % tcfg["save_interval"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

    torch.save(model.state_dict(), ckpt_dir / "last.pth")
    print(f"Done. Best val dice: {best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
