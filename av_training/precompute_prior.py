"""
Precompute prior probability maps for all samples in a JSON list using
the trained model from the previous stage.

This must be run before training stage i (i > 0):

    python precompute_prior.py \\
        --config config.yaml \\
        --model checkpoints/stage_0/best.pth \\
        --source-stage 0 \\
        --data-list data/train.json \\
        --data-list data/val.json

Each sample's JSON entry gets a new key  "prior_<source_stage>"  pointing
to the saved .npz file (written next to the CT file as  prior_<stage>.npz).
The JSON files are updated in place.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from monai.inferers import SlidingWindowInferer

from HiPaS.model import HiPaSNet


def build_prior_chain(
    stage: int,
    sample: dict,
    all_models: list,
    inferer: SlidingWindowInferer,
    device: torch.device,
    npz_key: str,
    normalize_hu: bool,
    hu_min: float,
    hu_max: float,
) -> np.ndarray:
    """Run stage-0 → ... → stage-<stage> inference chain, return final prior."""
    ct = np.load(sample["ct"])[npz_key].astype(np.float32)
    filt = np.load(sample["filter"])[npz_key].astype(np.float32)

    if normalize_hu:
        ct = np.clip(ct, hu_min, hu_max)
        ct = (ct - hu_min) / (hu_max - hu_min)
    filt = np.clip(filt, 0.0, 1.0)

    x = torch.from_numpy(np.stack([ct, filt], axis=0)[np.newaxis]).float().to(device)

    prior = None
    for model in all_models[: stage + 1]:
        if prior is None:
            prior = inferer(inputs=x, network=model)
        else:
            prior = inferer(inputs=(x, prior), network=model)

    return prior[0].detach().cpu().numpy().astype(np.float32)  # (2, D, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True,
                        help="Path to the checkpoint of stage <source-stage>")
    parser.add_argument("--source-stage", type=int, required=True,
                        help="Which stage this model corresponds to (0, 1, 2, or 3)")
    parser.add_argument("--data-list", nargs="+", required=True,
                        help="JSON list files to process (train, val, etc.)")
    parser.add_argument("--prior-dir", default=None,
                        help="Directory to save prior .npz files. "
                             "Default: next to each CT file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    pcfg = cfg["precompute"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all models from stage 0 up to source_stage (need full chain)
    stage_paths = cfg["training"]["pretrained_stage_paths"] + [args.model]
    if len(stage_paths) != args.source_stage + 1:
        raise ValueError(
            f"Expected {args.source_stage + 1} model paths "
            f"(stages 0..{args.source_stage}), got {len(stage_paths)}.\n"
            "Set pretrained_stage_paths in config.yaml for stages 0 .. source_stage-1."
        )

    models = []
    for path in stage_paths:
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

    roi = tuple(pcfg["sw_roi_size"])
    inferer = SlidingWindowInferer(
        roi_size=roi,
        sw_batch_size=pcfg["sw_batch_size"],
        overlap=pcfg["sw_overlap"],
        mode="gaussian",
        progress=True,
        sw_device=device,
        device="cpu",
    )

    prior_key = f"prior_{args.source_stage}"

    for list_path in args.data_list:
        with open(list_path) as f:
            samples = json.load(f)

        for i, s in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {s['ct']}")
            prior = build_prior_chain(
                stage=args.source_stage,
                sample=s,
                all_models=models,
                inferer=inferer,
                device=device,
                npz_key=dcfg["npz_key"],
                normalize_hu=dcfg["normalize_hu"],
                hu_min=dcfg["hu_min"],
                hu_max=dcfg["hu_max"],
            )

            if args.prior_dir:
                out_dir = Path(args.prior_dir)
            else:
                out_dir = Path(s["ct"]).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            ct_stem = Path(s["ct"]).stem.replace(".npz", "")
            out_path = out_dir / f"{ct_stem}_{prior_key}.npz"
            np.savez_compressed(str(out_path), **{pcfg["output_key"]: prior})
            samples[i][prior_key] = str(out_path)

        with open(list_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Updated {list_path}")


if __name__ == "__main__":
    main()
