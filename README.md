# Deep learning-driven pulmonary arteries and veins segmentation reveals demography-associated pulmonary vasculature anatomy
## Overview
This repository provides the method described in the paper:
“Deep learning-driven pulmonary arteries and veins segmentation reveals demography-associated pulmonary vasculature anatomy”

## Installation
```
conda create -n HiPaS python==3.8
pip install -r requirements.txt
```
Here we include all the packages used in our whole platform. However, some packages are not used in this project. You can install some of these packages according to your situation.

## Sample data
A part of the accessible data and the predicted results achieved by HiPaS can be downloaded [here](https://drive.google.com/drive/folders/1ogEe3Q5bqjmgN4SQCLX_kBATO3sQWT1B?usp=drive_link). All CT scans here are normalized from [-1000, 600] to [0, 1] and resampled to a normalized spatial resolution with the scan shape of [512, 512, 512]. The results are presented across two channels, with the first being the outcomes of artery segmentation and the second being vein segmentation. These examples are intended to demonstrate temporarily the segmentation performance of HiPaS for external data, and any other application or exploitation of the results would not be permissible without seeking proper approval. If you want to access more data, please do not hesitate to contact yuetan.chu@kaust.edu.sa. 


### Workflow
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-1-3.png)

### Performance evaluation
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-2_1.png)

### Clinical evaluation
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/fig-3-3.png)

### Anatomical study
![image](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/blob/main/img/stat.png)

## Acknowledgement
We use the [3D UNet](https://github.com/wolny/pytorch-3dunet) as the training process for the segmentation model.




