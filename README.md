# An end-to-end workflow for robust pulmonary artery-vein segmentation on thick-slice chest CT
## Overview
This repository provides the method described in the paper:
“An end-to-end workflow for robust pulmonary artery-vein segmentation on thick-slice chest CT”, which has been submitted to IEEE Transaction of Medical Imaging.

## Here You Can Find:
1) A super-resolution model along z-axis to reconstruct thick slice CT scans (inter-slice thickness>=5.00mm) into thin slice CT scans (inter-slice thickness=1.00mm).
2) A pulmonary arteries and veins segmentation model.  

## Use PuAV for:
### Thick-slice CT scans super-resolution
### Extra- and intra-pulmonary arteries and veins on thin-slice CT scans
### Extra- and intra-pulmonary arteries and veins on thick-slice CT scans
### Accurate and robust segmentations for Lung, Airway, Heart and Blood Vessels repectively

## Description
PuAV is is a computer-aided detection (CADe) method for achieveing well-performed 3D-visualized whole arteries and veins on chest computerized tomography (CT). Using deep-learning, PuAV first reconstruct thick-slice CT scans into a normzalied space with 1.00 mm inter-slice thickness. Basic semantics such as lung, airway, heart and blood vessels can be achieved from the reconstructed scans accurately. Then a two-stage segmentation algorithms performed on the reconstructed scans to get the extra- and intra-pulmonary arteries and veins respectively. The accurate artery-vein segmentation can be helpful for pulmonary diseases diagnosis and surgery planning.
### Workflow

## Reproduce Our Follow-up Results
1) All CT scans are normalized from [-1000, 400] to [0, 1]. 
2) Prepare the chest segmentation for lung&heart, airway&blood vessels. Lesions and lung nodules can be also annotated if you want to get more accyrate reconstructed scans.
3) Prepare the reconstrution datasets with preprececc\datasets_RE.py. 
4) Run training_RE.py to train your datasets.
5) For segmentation, you can cut the arrayies into [128, 128, 128] cubes firstly or use the trainging structure shown in https://github.com/wolny/pytorch-3dunet.
6) Run training_seg.py to train your segmentation models.
