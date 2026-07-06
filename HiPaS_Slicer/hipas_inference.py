"""Command-line inference runner for the HiPaS Slicer demo.

This script intentionally runs outside Slicer's Python environment. 3D Slicer
loads/saves medical images well, while the deep-learning stack is usually much
easier to manage in a separate conda environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = MODULE_DIR
DEFAULT_MODEL_DIR = REPO_ROOT


@dataclass(frozen=True)
class LayoutContext:
    mode: str
    original_axcodes: tuple[str, str, str]
    original_shape: tuple[int, int, int]
    inference_shape: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HiPaS artery/vein segmentation on a NIfTI volume.")
    parser.add_argument("--input", required=True, type=Path, help="Input CT volume (.nii or .nii.gz).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output masks.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path, help="Directory containing .pth weights.")
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        type=Path,
        help="Directory containing models.py and frangi_gpu.py. Defaults to this Slicer module directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device. CPU is only intended for smoke tests on tiny volumes.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution when CUDA is not available. This is very slow for real CT volumes.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Threshold for the final artery and vein probabilities.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight NIfTI I/O smoke test and write synthetic masks instead of loading the models.",
    )
    parser.add_argument(
        "--layout",
        default="simple-av",
        choices=("simple-av", "native"),
        help="Use the Simple_AV_seg NIfTI layout, transpose(1,0,2), or keep native voxel layout.",
    )
    parser.add_argument("--hu-offset", default=1000.0, type=float, help="HU offset used for CT normalization.")
    parser.add_argument("--hu-scale", default=1400.0, type=float, help="HU scale used for CT normalization.")
    return parser.parse_args()


def add_source_dir(source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    if not (source_dir / "models.py").exists():
        raise FileNotFoundError(f"models.py was not found in source directory: {source_dir}")
    if not (source_dir / "frangi_gpu.py").exists():
        raise FileNotFoundError(f"frangi_gpu.py was not found in source directory: {source_dir}")
    sys.path.insert(0, str(source_dir))


def select_device(requested: str, allow_cpu: bool) -> torch.device:
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if allow_cpu or requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. Re-run with --allow-cpu only for tiny smoke tests.")


def require_weights(model_dir: Path) -> dict[str, Path]:
    weights = {
        "lung": model_dir / "lung.pth",
        "main_av": model_dir / "main_AV.pth",
        "stage_1": model_dir / "AV_stage_1.pth",
    }
    missing = [str(path) for path in weights.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required model weights:\n" + "\n".join(missing))
    return weights


def normalize_ct(data: np.ndarray, hu_offset: float, hu_scale: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Input volume contains no finite voxels.")

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if low < -10.0 or high > 2.0:
        data = (data + hu_offset) / hu_scale
    return np.clip(data, 0.0, 1.0).astype(np.float32, copy=False)


def apply_input_layout(data: np.ndarray, reference: nib.Nifti1Image, mode: str) -> tuple[np.ndarray, LayoutContext]:
    context = LayoutContext(
        mode=mode,
        original_axcodes=nib.aff2axcodes(reference.affine),
        original_shape=tuple(int(value) for value in data.shape),
        inference_shape=tuple(int(value) for value in data.shape),
    )
    if mode == "native":
        return np.ascontiguousarray(data, dtype=np.float32), context
    if mode == "simple-av":
        transposed = np.transpose(data, (1, 0, 2))
        context = LayoutContext(
            mode=mode,
            original_axcodes=context.original_axcodes,
            original_shape=context.original_shape,
            inference_shape=tuple(int(value) for value in transposed.shape),
        )
        return np.ascontiguousarray(transposed, dtype=np.float32), context
    raise ValueError(f"Unsupported input layout: {mode}")


def restore_output_layout(mask: np.ndarray, layout: LayoutContext) -> np.ndarray:
    if layout.mode == "native":
        return mask
    if layout.mode == "simple-av":
        return np.ascontiguousarray(np.transpose(mask, (1, 0, 2)), dtype=mask.dtype)
    raise ValueError(f"Unsupported output layout: {layout.mode}")


def layout_metadata(layout: LayoutContext) -> dict[str, object]:
    return {
        "original_orientation": "".join(layout.original_axcodes),
        "layout_mode": layout.mode,
        "original_shape": list(layout.original_shape),
        "inference_shape": list(layout.inference_shape),
    }


def maybe_crop_even_z(volume: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    original_shape = volume.shape
    if original_shape[-1] % 2 == 1:
        return volume[:, :, :-1], original_shape
    return volume, original_shape


def restore_shape(mask: np.ndarray, original_shape: tuple[int, int, int]) -> np.ndarray:
    if mask.shape == original_shape:
        return mask
    restored = np.zeros(original_shape, dtype=mask.dtype)
    slices = tuple(slice(0, min(mask.shape[index], original_shape[index])) for index in range(3))
    restored[slices] = mask[slices]
    return restored


def load_state(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def to_model_dtype(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        tensor = tensor.half()
    return tensor


def prepare_model(model: torch.nn.Module, weights_path: Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(load_state(weights_path, device), strict=True)
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.half()
    return model


def predict_zoomed(ct_array: np.ndarray, weights: dict[str, Path], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from models import UNet

    ct_tensor = torch.from_numpy(ct_array[np.newaxis, np.newaxis]).to(torch.float32)
    ct_tensor = to_model_dtype(ct_tensor, device)
    zoomed_ct = F.interpolate(ct_tensor, scale_factor=0.5, mode="trilinear", align_corners=False)

    lung_model = prepare_model(UNet(in_channel=1, num_classes=2, active="softmax"), weights["lung"], device)
    lung_ct = torch.clamp((zoomed_ct * 1600 - 1000 + 600) / 800, 0, 1)
    with torch.no_grad():
        zoom_lung = sliding_window_inference(
            inputs=lung_ct,
            predictor=lung_model,
            roi_size=(256, 256, 256),
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
            sigma_scale=0.125,
            progress=False,
            sw_device=device,
            device=torch.device("cpu"),
        )
    lung = F.interpolate(zoom_lung.float(), size=ct_array.shape, mode="trilinear", align_corners=False)
    lung = lung.detach().cpu().numpy()[0, 0] > 0.51
    del lung_model, zoom_lung
    if device.type == "cuda":
        torch.cuda.empty_cache()

    av_model = prepare_model(UNet(in_channel=1, num_classes=3, active="softmax"), weights["main_av"], device)
    with torch.no_grad():
        pre_av = sliding_window_inference(
            inputs=zoomed_ct,
            predictor=av_model,
            roi_size=(256, 256, 256),
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
            sigma_scale=0.125,
            progress=False,
            sw_device=device,
            device=torch.device("cpu"),
        )
    pre_av = F.interpolate(pre_av.float(), size=ct_array.shape, mode="trilinear", align_corners=False)
    pre_av = pre_av.detach().cpu().numpy()[0]
    artery_prior = pre_av[0] > 0.52
    vein_prior = pre_av[1] > 0.52
    del av_model, pre_av, zoomed_ct, ct_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return artery_prior.astype(np.float32), vein_prior.astype(np.float32), lung.astype(np.float32)


def predict_intra_av(scan: np.ndarray, weights: dict[str, Path], device: torch.device) -> np.ndarray:
    from models import MedNext

    model = MedNext(
        in_channels=3,
        n_classes=2,
        n_channels=24,
        kernel_size=3,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        do_res=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        deep_supervision=True,
    )
    model = prepare_model(model, weights["stage_1"], device)
    input_ct = torch.from_numpy(scan[np.newaxis]).to(torch.float32)
    input_ct = to_model_dtype(input_ct, device)

    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=input_ct,
            predictor=model,
            roi_size=(192, 192, 160),
            sw_batch_size=2,
            overlap=0.25,
            mode="gaussian",
            sigma_scale=0.125,
            progress=False,
            sw_device=device,
            device=torch.device("cpu"),
        )

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    result = prediction.detach().cpu().numpy()[0]
    del model, input_ct, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def predict_whole_av(ct_array: np.ndarray, weights: dict[str, Path], device: torch.device, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from frangi_gpu import frangi_filter_gpu

    cropped_ct, original_shape = maybe_crop_even_z(ct_array)
    artery_prior, vein_prior, lung = predict_zoomed(cropped_ct, weights, device)

    loc = np.array(np.where(lung > 0))
    if loc.size == 0:
        raise RuntimeError("The lung model produced an empty mask; cannot crop the AV inference region.")

    x_min, x_max = int(np.min(loc[0])), int(np.max(loc[0]))
    y_min, y_max = int(np.min(loc[1])), int(np.max(loc[1]))
    z_min, z_max = int(np.min(loc[2])), int(np.max(loc[2]))

    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise RuntimeError("The lung crop is degenerate; cannot run AV inference.")

    crop = cropped_ct[x_min:x_max, y_min:y_max, z_min:z_max]
    filtered = frangi_filter_gpu(crop, sigma=[0.5, 1, 1.5], transfer_device=True, device=device)
    prior = artery_prior * 0.25 + vein_prior * 0.75
    input_set = np.concatenate(
        (
            crop[np.newaxis],
            prior[x_min:x_max, y_min:y_max, z_min:z_max][np.newaxis],
            filtered[np.newaxis],
        ),
        axis=0,
    ).astype(np.float32, copy=False)

    probabilities = predict_intra_av(input_set, weights, device)
    prediction = np.zeros((2,) + cropped_ct.shape, dtype=np.float32)
    prediction[0, x_min:x_max, y_min:y_max, z_min:z_max] = probabilities[0]
    prediction[1, x_min:x_max, y_min:y_max, z_min:z_max] = probabilities[1]

    artery = restore_shape((prediction[0] > threshold).astype(np.uint8), original_shape)
    vein = restore_shape((prediction[1] > threshold).astype(np.uint8), original_shape)
    lung = restore_shape((lung > 0.5).astype(np.uint8), original_shape)
    return artery, vein, lung


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, output_path: Path) -> None:
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, str(output_path))


def write_outputs(
    artery: np.ndarray,
    vein: np.ndarray,
    lung: np.ndarray,
    reference: nib.Nifti1Image,
    output_dir: Path,
    metadata: dict[str, object],
) -> Path:
    outputs = {
        "artery": output_dir / "hipas_artery.nii.gz",
        "vein": output_dir / "hipas_vein.nii.gz",
        "lung": output_dir / "hipas_lung.nii.gz",
    }
    save_mask(artery, reference, outputs["artery"])
    save_mask(vein, reference, outputs["vein"])
    save_mask(lung, reference, outputs["lung"])

    metadata.update(
        {
            "outputs": {name: str(path) for name, path in outputs.items()},
            "voxel_counts": {
                "artery": int(artery.sum()),
                "vein": int(vein.sum()),
                "lung": int(lung.sum()),
            },
        }
    )
    metadata_path = output_dir / "hipas_outputs.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata_path


def run_smoke_test(
    ct_array: np.ndarray,
    reference: nib.Nifti1Image,
    output_dir: Path,
    input_path: Path,
    layout: LayoutContext,
) -> int:
    artery = (ct_array > 0.66).astype(np.uint8)
    vein = ((ct_array > 0.33) & (ct_array <= 0.66)).astype(np.uint8)
    lung = (ct_array > 0.05).astype(np.uint8)
    artery = restore_output_layout(artery, layout)
    vein = restore_output_layout(vein, layout)
    lung = restore_output_layout(lung, layout)
    metadata = {
        "input": str(input_path),
        "shape": list(ct_array.shape),
        "device": "none",
        "smoke_test": True,
        **layout_metadata(layout),
    }
    write_outputs(artery, vein, lung, reference, output_dir, metadata)
    return 0


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"Input: {args.input}", flush=True)

    reference = nib.load(str(args.input))
    raw_ct = reference.get_fdata(dtype=np.float32)
    ct_array, layout = apply_input_layout(raw_ct, reference, args.layout)
    ct_array = normalize_ct(ct_array, args.hu_offset, args.hu_scale)
    print(
        f"Layout: {layout.mode}; original shape {layout.original_shape}; inference shape {layout.inference_shape}; "
        f"normalization: (ct + {args.hu_offset:g}) / {args.hu_scale:g}",
        flush=True,
    )

    if args.smoke_test:
        print("Smoke test mode: writing synthetic masks without loading models.", flush=True)
        return run_smoke_test(ct_array, reference, args.output_dir, args.input, layout)

    add_source_dir(args.source_dir)
    device = select_device(args.device, args.allow_cpu)
    weights = require_weights(args.model_dir)

    print(f"Model directory: {args.model_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    artery, vein, lung = predict_whole_av(ct_array, weights, device, args.threshold)
    artery = restore_output_layout(artery, layout)
    vein = restore_output_layout(vein, layout)
    lung = restore_output_layout(lung, layout)

    metadata = {
        "input": str(args.input),
        "shape": list(ct_array.shape),
        "device": str(device),
        "smoke_test": False,
        "hu_offset": float(args.hu_offset),
        "hu_scale": float(args.hu_scale),
        **layout_metadata(layout),
    }
    write_outputs(artery, vein, lung, reference, args.output_dir, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
