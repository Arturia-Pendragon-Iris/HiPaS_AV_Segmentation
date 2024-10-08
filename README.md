# Deep learning-driven pulmonary arteries and veins segmentation reveals demography-associated pulmonary vasculature anatomy
## Overview
This repository provides the method described in the paper:
“Deep learning-driven pulmonary arteries and veins segmentation reveals demography-associated pulmonary vasculature anatomy”

Yuetan Chu, Gongning Luo, Longxi Zhou, Shaodong Cao, Guolin Ma, Xianglin Meng, Juexiao Zhou, Changchun Yang, Dexuan Xie, Ricardo Henao, Xigang Xiao, Lianming Wu, Zhaowen Qiu, Xin Gao

King Abdullah University of Science and Technology, KAUST

## Installation
```
conda create -n HiPaS python=3.8
conda activate HiPaS
pip install -r requirements.txt
```
Here we include all the packages used in our whole platform. However, some packages are not used in this project. You can install some of these packages according to your situation.

## Sample data
A part of the accessible data and the predicted results achieved by HiPaS can be downloaded [here](https://drive.google.com/drive/folders/1Bvq4hvkdKZZOivoh0RwlNZNkP5wkejX2?usp=sharing). All CT scans here are normalized from [-1000, 600] to [0, 1] and resampled to a normalized spatial resolution with the scan shape of [512, 512, 512]. The results are presented across two channels, with the first being the outcomes of artery segmentation and the second being vein segmentation. These examples are intended to demonstrate temporarily the segmentation performance of HiPaS for external data, and any other application or exploitation of the results would not be permissible without seeking proper approval. If you want to access more data, please do not hesitate to contact yuetan.chu@kaust.edu.sa. 

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
@misc{chu2024deep,
      title={Deep learning-driven pulmonary arteries and veins segmentation reveals demography-associated pulmonary vasculature anatomy}, 
      author={Yuetan Chu and Gongning Luo and Longxi Zhou and Shaodong Cao and Guolin Ma and Xianglin Meng and Juexiao Zhou and Changchun Yang and Dexuan Xie and Ricardo Henao and Xigang Xiao and Lianming Wu and Zhaowen Qiu and Xin Gao},
      year={2024},
      eprint={2404.07671},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



