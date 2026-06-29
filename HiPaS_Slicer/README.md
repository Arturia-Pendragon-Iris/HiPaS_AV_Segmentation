# HiPaS Slicer Demo

This folder contains the 3D Slicer-facing part of the HiPaS artery-vein segmentation demo.

- `HiPaSAVSeg.py` is the Slicer Scripted Module.
- `hipas_inference.py` is the external Python runner used by the module.

The module calls a separate conda Python environment instead of installing PyTorch into Slicer's bundled Python.

## Quick Setup

1. Install 3D Slicer 5.10.0 or newer.
2. Create an environment such as `arturia_v1` with PyTorch, MONAI, nibabel, numpy, and scikit-image.
3. Add this folder to Slicer's additional module paths:

   ```text
   <repo-root>\HiPaSAVSeg
   ```

4. Restart Slicer and open `HiPaS AV Seg`.
5. Set `External Python` to your environment's `python.exe`, for example:

   ```text
   %USERPROFILE%\miniconda3\envs\arturia_v1\python.exe
   ```

## Required Weights

The full model demo expects these files in the repository root:

```text
lung.pth
main_AV.pth
AV_stage_1.pth
```
