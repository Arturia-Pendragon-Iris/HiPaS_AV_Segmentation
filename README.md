# An end-to-end workflow for robust pulmonary artery-vein segmentation on thick-slice chest CT
## Overview
This repository provides the method described in the paper:
“An end-to-end workflow for robust pulmonary artery-vein segmentation on thick-slice chest CT”

## Here You Can Find:
1) A super-resolution model along z-axis to reconstruct thick slice CT scans (inter-slice thickness>=5.00mm) into thin slice CT scans (inter-slice thickness=1.00mm).
2) A pulmonary arteries and veins segmentation model.  

## Use PuAV for:
### Thick-slice CT scans super-resolution
![image](https://github.com/Arturia-Pendragon-Iris/PuAV-Segmentation/blob/main/img/slice_renders.png)
### Extra- and intra-pulmonary arteries and veins on thin-slice CT scans
![image](https://github.com/Arturia-Pendragon-Iris/PuAV-Segmentation/blob/main/img/seg_thin.png)
### Extra- and intra-pulmonary arteries and veins on thick-slice CT scans
![image](https://github.com/Arturia-Pendragon-Iris/PuAV-Segmentation/blob/main/img/seg_thick.png)
### Accurate and robust segmentations for Lung, Airway, Heart and Blood Vessels repectively
![image](https://github.com/Arturia-Pendragon-Iris/PuAV-Segmentation/blob/main/img/semantics_seg.png)

## Description
PuAV is is a computer-aided detection (CADe) method for achieveing well-performed 3D-visualized whole arteries and veins on chest computerized tomography (CT). Using deep-learning, PuAV first reconstruct thick-slice CT scans into a normzalied space with 1.00 mm inter-slice thickness. Basic semantics such as lung, airway, heart and blood vessels can be achieved from the reconstructed scans accurately. Then a two-stage segmentation algorithms performed on the reconstructed scans to get the extra- and intra-pulmonary arteries and veins respectively. The accurate artery-vein segmentation can be helpful for pulmonary diseases diagnosis and surgery planning.
### Workflow
![image](https://github.com/Arturia-Pendragon-Iris/PuAV-Segmentation/blob/main/img/overview.jpg)

## Reproduce Our Follow-up Results
1) All CT scans are normalized from [-1000, 400] to [0, 1]. 
2) Prepare the chest segmentation for lung&heart, airway&blood vessels. Lesions and lung nodules can be also annotated if you want to get more accyrate reconstructed scans.
3) Prepare the reconstrution datasets with preprececc\datasets_RE.py. 
4) Run training_RE.py to train your datasets.
5) For segmentation, you can cut the arrayies into [128, 128, 128] cubes firstly or use the trainging structure shown in https://github.com/wolny/pytorch-3dunet.
6) Run training_seg.py to train your segmentation models.

## Supplementary Details
The voxel intensity of all scans was truncated within the Hounsfield Unit (HU) window of $[-1000, 400]$ and normalization to $[0, 1]$. For the super-resolution tasks, we grouped every five consecutive frames from 5.0 mm slice-thickness scans as an input patch $512\times512\times5$ in sliding windows, while the corresponding frame from 1.0 mm and the weights are concatenated together as the loss-term patch $512\times512\times10$. It was observed that this input size could capture the detailed tissue-level information for upsampling training. Similarly, due to the GPU memory limit, CT scans were cropped into sub-volume cubes of the size $128\times128\times128$ for segmentation tasks. We use PyTorch to implement the proposed method. The Adam optimizer was used for the segmentation network with an initial learning rate of $1\times10^{-4}$. The decay of the first-order momentum gradient is 0.9, and the decay of the second-order momentum gradient is 0.999.In the experiments, model training was executed on a Linux workstation with NVIDIA RTX A6000. Current parameters performed well for our tasks but are not necessarily optimum. Adjustment may be conducted in specific tasks.
