# Deep learning-driven pulmonary artery and vein segmentation reveals demography-associated vasculature anatomical differences
## Overview
This repository provides the method described in the paper:
“Deep learning-driven pulmonary artery and vein segmentation reveals demography-associated vasculature anatomical differences”

King Abdullah University of Science and Technology, KAUST

Our manuscript has been accepted by Nature Communication and is now in publication. Congratulation!!!!! We will update any information timely. 

If you have any questions about the code, paper, or datasets, please email yuetan.chu@kaust.edu.sa. 

## Installation
```
conda create -n HiPaS python=3.8
conda activate HiPaS
pip install -r requirements.txt
```
Here we include all the packages used in our whole platform. However, some packages are not used in this project. You can install some of these packages according to your situation.

<!--
## Sample data
A part of the accessible data and the predicted results achieved by HiPaS can be downloaded [here](https://drive.google.com/drive/folders/1Bvq4hvkdKZZOivoh0RwlNZNkP5wkejX2?usp=sharing). All CT scans here are normalized from [-1000, 600] to [0, 1] and resampled to a normalized spatial resolution with the scan shape of [512, 512, 512]. The results are presented across two channels, with the first being the outcomes of artery segmentation and the second being vein segmentation. These examples are intended to demonstrate temporarily the segmentation performance of HiPaS for external data, and any other application or exploitation of the results would not be permissible without seeking proper approval. If you want to access more data, please do not hesitate to contact yuetan.chu@kaust.edu.sa. 
-->

## Datasets


We have released ~250 cases of chest CT scans with artery-vein annotations. You can download the datasets using this [Google drive link](https://drive.google.com/drive/folders/1_cmGR_HbrzomaqoWZYqX8D36bmxHp2PL?usp=drive_link) or the [Zenodo link](https://zenodo.org/records/14879605). To open the CT data and annotation, you can use the following code


```
ct = np.load(".\ct_scan\001.npz", allow_pickle=True)["data"]
artery = np.load(".\annotation\artery\001.npz", allow_pickle=True)["data"]
vein = np.load(".\annotation\vein\001.npz", allow_pickle=True)["data"]
```

Due to the consideration of the project commercialization, the released annotations keep the same as the segmentation standard in the [PARSE22 challenge](https://grand-challenge.org/forums/forum/parse2022-623/), as shown in Supplementary Figure 6 (Stage 2) in our Supplementary Information. This can satisfy most clinical requirements.


## Train
You can use the [3D UNet](https://github.com/wolny/pytorch-3dunet) as the training process for the segmentation model and replace the default 3DUNet with our proposed network. We also provide our training framework in ```HiPaS```. The input data should be stored in HDF5 files. The HDF5 files for training should contain two datasets: raw and label. The "raw" dataset contains CT scans, while the "label" dataset is the artery-vein segmentation. The segmentation of different vessel levels should be trained separately. In order to train on your own data, you can provide the paths to your HDF5 training and validation datasets in the YAML file, and run ```HiPaS/train.py```.

## Predict
To predict on your own data, you can provide the checkpoint path as well as paths of the CT volume, and run ```HiPaS/predict_av.py```. It may take about 2 minutes to achieve the prediction result for one CT volume. To run the program on your own data, you can just replace the default path too your own file path.

### Workflow
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-1-4.png)

### Performance evaluation
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-2_1.png)

### Clinical evaluation
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-3-3.png)

### Anatomical study
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/stat.png)

## Cite
```
Chu, Y., Luo, G., Zhou, L. et al. Deep learning-driven pulmonary artery and vein segmentation reveals demography-associated vasculature anatomical differences. Nat Commun 16, 2262 (2025). https://doi.org/10.1038/s41467-025-56505-6
```



