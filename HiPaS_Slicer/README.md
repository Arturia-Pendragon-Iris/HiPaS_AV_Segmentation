# HiPaS AV Seg Slicer Module

This folder contains the 3D Slicer scripted module used to run HiPaS artery-vein segmentation from Slicer while using an external conda Python environment for PyTorch inference.

## Files

```text
HiPaSAVSeg.py       Slicer scripted module UI and Slicer scene integration
hipas_inference.py  External Python inference runner
models.py           Local UNet and MedNext model definitions
frangi_gpu.py       Local vesselness filter implementation
```

The module no longer requires an external `Simple_AV_seg-main` source directory at runtime. The model architecture code needed by inference is included in this folder.

## Runtime Assumptions

The current tested setup uses:

```text
External Python: /home/chuy/anaconda3/envs/Arturia_v2/bin/python
Checkpoint dir: /home/chuy/Slicer/code/Simple_HiPaS
```

The checkpoint directory must contain:

```text
lung.pth
main_AV.pth
AV_stage_1.pth
```

The external Python environment must provide:

```text
torch
monai
nibabel
numpy
scikit-image
```

## Important Preprocessing Details

This Slicer runner is aligned with the working `Simple_AV_seg-main/prediction.py` workflow:

```python
ct_array = nib.load(path).get_fdata()
ct_array = np.transpose(ct_array, (1, 0, 2))
ct_array = np.clip((ct_array + 1000) / 1400, 0, 1)
```

After inference, output masks are transposed back with:

```python
mask = np.transpose(mask, (1, 0, 2))
```

The output NIfTI masks are saved with the original input affine/header so they overlay correctly in Slicer.

The log should show:

```text
Layout: simple-av
normalization: (ct + 1000) / 1400
```

## Install The Module In Slicer

1. Start Slicer:

   ```bash
   /home/chuy/Slicer/Slicer
   ```

2. Open:

   ```text
   Edit -> Application Settings -> Modules
   ```

3. Add this folder to `Additional module paths`:

   ```text
   /home/chuy/Slicer/code/HiPaS_AV_Segmentation-main/HiPaS_Slicer
   ```

4. Restart Slicer.

5. Open the module:

   ```text
   Segmentation -> HiPaS AV Seg
   ```

In the current local setup this path has already been added to:

```text
/home/chuy/Slicer/slicer.org/Slicer-33241.ini
```

## Run Segmentation

1. Load the CT volume.

   For NIfTI:

   ```text
   Add Data -> select .nii or .nii.gz -> OK
   ```

   For DICOM:

   ```text
   DICOM -> Import -> select DICOM folder -> Load
   ```

2. Open:

   ```text
   Segmentation -> HiPaS AV Seg
   ```

3. Select parameters:

   ```text
   Input volume:     the loaded CT volume
   External Python:  /home/chuy/anaconda3/envs/Arturia_v2/bin/python
   Model directory:  /home/chuy/Slicer/code/Simple_HiPaS
   Output directory: any writable folder
   Load outputs:     checked
   ```

4. Click `Run`.

5. On success, the module loads three label volumes:

   ```text
   <input-name>_artery
   <input-name>_vein
   <input-name>_lung
   ```

The output directory also contains:

```text
hipas_artery.nii.gz
hipas_vein.nii.gz
hipas_lung.nii.gz
hipas_outputs.json
```

## 3D Visualization In Slicer

1. Open the `Segmentations` module.

2. In `Import/export nodes`, import the artery label volume:

   ```text
   Operation: Import
   Input node: <input-name>_artery
   Output segmentation: create AV_Segmentation
   ```

3. Import the vein label volume into the same `AV_Segmentation`.

4. In the segment list, set colors:

   ```text
   artery: red
   vein: blue
   ```

5. Enable 3D display for the segmentation.

6. Optional: use `Volume Rendering` to show the CT volume semi-transparently behind the vessels.

## Debugging

### External Python cannot import packages

If the log shows an error such as:

```text
ModuleNotFoundError: No module named 'nibabel'
```

make sure `External Python` points to:

```text
/home/chuy/anaconda3/envs/Arturia_v2/bin/python
```

The module clears Slicer's `PYTHONHOME`, `PYTHONPATH`, and `PYTHONNOUSERSITE` before launching the external runner, because those variables can otherwise break conda package resolution.

### CUDA is not available

If the log shows:

```text
CUDA is not available
```

start Slicer from a terminal where the conda/PyTorch environment can see the GPU. Full CT inference is expected to run on CUDA. CPU mode is only intended for tiny smoke tests.

### Segmentation orientation looks wrong

Check the log. The current expected layout line is:

```text
Layout: simple-av
```

This means the Slicer-exported NIfTI is loaded with `get_fdata()`, transposed as `(1, 0, 2)` before inference, then transposed back before saving. This matches the verified standalone `read_nii()` behavior.

### Smoke test from the command line

This only tests NIfTI I/O, layout handling, and output writing. It does not load the neural networks:

```bash
/home/chuy/anaconda3/envs/Arturia_v2/bin/python \
  /home/chuy/Slicer/code/HiPaS_AV_Segmentation-main/HiPaS_Slicer/hipas_inference.py \
  --input /path/to/input.nii.gz \
  --output-dir /tmp/hipas_smoke \
  --model-dir /home/chuy/Slicer/code/Simple_HiPaS \
  --smoke-test
```

### Full command-line inference

```bash
/home/chuy/anaconda3/envs/Arturia_v2/bin/python \
  /home/chuy/Slicer/code/HiPaS_AV_Segmentation-main/HiPaS_Slicer/hipas_inference.py \
  --input /path/to/input.nii.gz \
  --output-dir /path/to/output_dir \
  --model-dir /home/chuy/Slicer/code/Simple_HiPaS
```
