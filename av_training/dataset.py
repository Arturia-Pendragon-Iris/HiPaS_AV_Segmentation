"""
AV segmentation dataset for STS multi-stage training.

Each sample JSON entry must have:
  ct, filter, mask_0, mask_1, mask_2, mask_3  — paths to .npz files

Optionally (filled by precompute_prior.py):
  prior_0, prior_1, prior_2  — precomputed prior probability maps from the
                                previous stage model (.npz with shape (2,H,W,D))
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npz(path: str, key: str = "arr_0") -> np.ndarray:
    return np.load(path)[key].astype(np.float32)


def _normalize_hu(arr: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    arr = np.clip(arr, hu_min, hu_max)
    return (arr - hu_min) / (hu_max - hu_min)


def _random_crop_3d(arrays: list[np.ndarray], crop_size: tuple[int, int, int]) -> list[np.ndarray]:
    """Uniformly random crop all arrays to crop_size. Pads if necessary."""
    # arrays: list of (C, D, H, W) or (D, H, W)
    ref = arrays[0]
    ndim = ref.ndim
    spatial = ref.shape[-3:]  # D, H, W (last 3 dims regardless of channels)

    # Pad if volume smaller than crop
    pad = [(0, max(0, crop_size[i] - spatial[i])) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        pad_full = [(0, 0)] * (ndim - 3) + pad
        arrays = [np.pad(a, pad_full, mode="constant") for a in arrays]
        spatial = arrays[0].shape[-3:]

    starts = [random.randint(0, spatial[i] - crop_size[i]) for i in range(3)]
    slices = tuple(slice(s, s + crop_size[i]) for i, s in enumerate(starts))

    def crop(a):
        if a.ndim == 3:
            return a[slices]
        return a[(...,) + slices]

    return [crop(a) for a in arrays]


def _augment(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Random flip along each of the 3 spatial axes."""
    for axis in range(-3, 0):
        if random.random() < 0.5:
            arrays = [np.flip(a, axis=axis).copy() for a in arrays]
    return arrays


class AVDataset(Dataset):
    def __init__(
        self,
        sample_list: str | list[dict],
        stage: int,
        crop_size: tuple[int, int, int] = (128, 192, 192),
        npz_key: str = "arr_0",
        normalize_hu: bool = True,
        hu_min: float = -1000.0,
        hu_max: float = 600.0,
        augment: bool = True,
    ):
        """
        stage: which STS stage is being trained (0-3).
               Determines which prior key to look for in each sample dict.
        """
        if isinstance(sample_list, str):
            with open(sample_list) as f:
                self.samples = json.load(f)
        else:
            self.samples = sample_list

        self.stage = stage
        self.crop_size = tuple(crop_size)
        self.npz_key = npz_key
        self.normalize_hu = normalize_hu
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        ct = _load_npz(s["ct"], self.npz_key)            # (D, H, W)
        filt = _load_npz(s["filter"], self.npz_key)      # (D, H, W)

        if self.normalize_hu:
            ct = _normalize_hu(ct, self.hu_min, self.hu_max)
        # filter is assumed to already be in [0, 1]
        filt = np.clip(filt, 0.0, 1.0)

        # Load all 4 cumulative masks: each (2, D, H, W), ch0=artery ch1=vein
        masks = [_load_npz(s[f"mask_{i}"], self.npz_key) for i in range(4)]

        # Load precomputed prior from stage-1 (None for stage 0)
        prior_key = f"prior_{self.stage - 1}"
        if self.stage > 0 and prior_key in s:
            prior = _load_npz(s[prior_key], self.npz_key)   # (2, D, H, W)
        else:
            prior = None

        # Stack CT and filter into 2-channel input
        inp = np.stack([ct, filt], axis=0)   # (2, D, H, W)

        # Collect all arrays for joint crop/augmentation
        all_arrays = [inp] + masks + ([prior] if prior is not None else [])
        all_arrays = _random_crop_3d(all_arrays, self.crop_size)
        if self.augment:
            all_arrays = _augment(all_arrays)

        inp = all_arrays[0]
        masks = all_arrays[1:5]
        prior = all_arrays[5] if prior is not None else None

        result = {
            "input": torch.from_numpy(inp),                             # (2, D, H, W)
            "masks": [torch.from_numpy(m) for m in masks],             # 4 x (2, D, H, W)
            "prior": torch.from_numpy(prior) if prior is not None else None,
        }
        return result


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items; handles None prior by using zeros."""
    inputs = torch.stack([b["input"] for b in batch])
    masks = [torch.stack([b["masks"][i] for b in batch]) for i in range(4)]

    if batch[0]["prior"] is not None:
        prior = torch.stack([b["prior"] for b in batch])
    else:
        prior = None

    return {"input": inputs, "masks": masks, "prior": prior}
