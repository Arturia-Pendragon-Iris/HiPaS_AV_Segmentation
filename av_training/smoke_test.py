"""
Smoke test — verifies the full training pipeline with synthetic in-memory data.

Run with:
    python smoke_test.py

Tests:
  1. Dataset → DataLoader (synthetic data, no disk I/O)
  2. Model forward pass (stage 0 and stage 1 with prior)
  3. Loss computation
  4. Backward pass
  5. Two training steps confirm loss is finite and the loop runs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader

from HiPaS.model import HiPaSNet
from av_training.dataset import AVDataset, collate_fn
from av_training.loss import STSLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP = (64, 96, 96)   # smaller than training for speed
BATCH = 1
MID_CH = 24
R = 4


# ── 1. Synthetic dataset ───────────────────────────────────────────────────────

class _SyntheticAVDataset(AVDataset):
    """Overrides __getitem__ to return random tensors without touching disk."""

    def __init__(self, n: int, stage: int, crop_size):
        self.samples = [{}] * n  # dummy entries
        self.stage = stage
        self.crop_size = crop_size
        self.augment = False
        # These won't be used (we override __getitem__)
        self.npz_key = "arr_0"
        self.normalize_hu = False
        self.hu_min = -1000.0
        self.hu_max = 600.0

    def __getitem__(self, idx):
        D, H, W = self.crop_size
        inp = torch.rand(2, D, H, W)

        # Cumulative masks: each level adds more vessels
        base = (torch.rand(2, D, H, W) > 0.95).float()
        masks = []
        acc = base.clone()
        for _ in range(4):
            acc = torch.clamp(acc + (torch.rand(2, D, H, W) > 0.97).float(), 0, 1)
            masks.append(acc.clone())

        prior = torch.rand(2, D, H, W) * 0.3 if self.stage > 0 else None
        return {"input": inp, "masks": masks, "prior": prior}


def _make_loader(stage: int, n: int = 4) -> DataLoader:
    ds = _SyntheticAVDataset(n, stage=stage, crop_size=CROP)
    return DataLoader(ds, batch_size=BATCH, collate_fn=collate_fn)


# ── 2. Model ───────────────────────────────────────────────────────────────────

def _build_model() -> HiPaSNet:
    return HiPaSNet(in_channels=2, mid_channels=MID_CH, out_channels=2, r=R).to(DEVICE)


# ── 3. Tests ───────────────────────────────────────────────────────────────────

def test_forward_stage0():
    print("  [1] Stage-0 forward (no prior)...")
    model = _build_model()
    loader = _make_loader(stage=0)
    batch = next(iter(loader))
    x = batch["input"].to(DEVICE)
    pred = model(x, p=None)
    assert pred.shape == (BATCH, 2, *CROP), f"Unexpected shape: {pred.shape}"
    assert pred.min() >= 0 and pred.max() <= 1, "Output not in [0,1]"
    print(f"     output shape: {pred.shape}  min={pred.min():.3f}  max={pred.max():.3f}  OK")


def test_forward_stage1():
    print("  [2] Stage-1 forward (with prior)...")
    model = _build_model()
    loader = _make_loader(stage=1)
    batch = next(iter(loader))
    x = batch["input"].to(DEVICE)
    prior = batch["prior"].to(DEVICE)
    pred = model(x, p=prior)
    assert pred.shape == (BATCH, 2, *CROP)
    print(f"     output shape: {pred.shape}  OK")


def test_loss():
    print("  [3] Loss computation...")
    criterion = STSLoss()
    model = _build_model()
    for stage in range(4):
        loader = _make_loader(stage=stage)
        batch = next(iter(loader))
        x = batch["input"].to(DEVICE)
        masks = [m.to(DEVICE) for m in batch["masks"]]
        prior = batch["prior"].to(DEVICE) if batch["prior"] is not None else None

        pred = model(x, p=prior)
        loss, components = criterion(pred, masks, stage, prior)

        assert torch.isfinite(loss), f"Loss is not finite at stage {stage}: {loss}"
        print(f"     stage={stage}  loss={loss:.4f}  dice={components['dice']:.4f}"
              f"  overlap={components['overlap']:.4f}  OK")


def test_backward():
    print("  [4] Backward pass (stage 0, 2 steps)...")
    model = _build_model()
    criterion = STSLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = _make_loader(stage=0, n=4)

    losses = []
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        x = batch["input"].to(DEVICE)
        masks = [m.to(DEVICE) for m in batch["masks"]]
        pred = model(x, p=None)
        loss, _ = criterion(pred, masks, stage=0, prior=None)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"     step {i+1}: loss={loss.item():.4f}")

    assert all(np.isfinite(l) for l in losses), "Non-finite loss during backward"
    print("     OK")


def test_stage1_with_frozen_prior_model():
    print("  [5] Stage-1 training step with frozen prior model (cascade)...")
    prior_model = _build_model()
    prior_model.eval()
    for p in prior_model.parameters():
        p.requires_grad_(False)

    model = _build_model()
    criterion = STSLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    loader = _make_loader(stage=1, n=2)
    batch = next(iter(loader))
    x = batch["input"].to(DEVICE)
    masks = [m.to(DEVICE) for m in batch["masks"]]

    with torch.no_grad():
        prior = prior_model(x, p=None)

    pred = model(x, p=prior)
    loss, _ = criterion(pred, masks, stage=1, prior=prior)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"     loss={loss.item():.4f}  OK")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nSmoke test — device: {DEVICE}")
    print(f"Crop size: {CROP}, batch: {BATCH}, mid_channels: {MID_CH}\n")

    test_forward_stage0()
    test_forward_stage1()
    test_loss()
    test_backward()
    test_stage1_with_frozen_prior_model()

    print("\nAll smoke tests passed.")
