#

# PuAV-Segmentation
The supplementary code for the pulmonary artery-vein segmentation with various inter-slice thickness
1) Model:
The model for thick-slice reconstruction is published in "Reconstruction" model, and the artery-vein segmentation is published in "Artery-Vein" model.
Here we offer two models: 2.5DU-Net and hybrid U2Net for extra- and intra-pulmonary artery-vein segmentation respectively.

2) Datasets
All CT scans are normalized from [-1000, 400] to [0, 1].  For reconstruction, the data preparation is in preprececc\datasets_RE.py. 
A prior segmentation for lung, airway, blood vessels and heart are needed. 
For segmentation, you can cut the arrayies into [128, 128, 128] cubes firstly or use the trainging structure shown in https://github.com/wolny/pytorch-3dunet
