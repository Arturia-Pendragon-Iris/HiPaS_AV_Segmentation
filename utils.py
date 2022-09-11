import glob
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


def sigmoid(img, alpha, beta):

    return 1 / (1 + np.exp((beta - img) / alpha))


def augment(img, alpha=20, beta=80):
    img = (img * 1600 - 600 + 1000) / 1400
    img[img < 0] = 0
    img[img > 1] = 1
    return 1 / (1 + np.exp((beta - img * 255) / alpha))


class TrainSetLoader_Upsample(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader_Upsample, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = np.sort(os.listdir(dataset_dir))
        # print(self.file_list)
        self.device = device

    def __getitem__(self, index):
        # print(os.path.join(self.dataset_dir, self.file_list[index]))
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]), allow_pickle=True)["array"].item()
        lr_img = torch.tensor(np_array["model_input"]).to(self.device).to(torch.float)
        hr_img = torch.tensor(np_array["model_gt"]).to(self.device).to(torch.float)
        weight = torch.tensor(np_array["loss_weight"]).to(self.device).to(torch.float)

        return lr_img, hr_img, weight

    def __len__(self):
        return len(self.file_list)


class TrainSetLoader_Seg(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader_Seg, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = glob.glob((os.path.join(dataset_dir, "*.npz")))
        self.device = device

    def __getitem__(self, index):
        # print(self.file_list[index])
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]), allow_pickle=True)["array"].item()
        raw = np_array["model_gt"][np.newaxis, :]
        raw_aug = sigmoid(raw * 255, 20, 80)

        lung = np_array["lung"][np.newaxis, :]
        airway = np_array["airway_mask"][np.newaxis, :]
        heart = np_array["heart_mask"][np.newaxis, :]
        artery_main = np_array["artery_main"][np.newaxis, :]
        artery_detail = np_array["artery_detail"][np.newaxis, :]
        vein_main = np_array["vein_main"][np.newaxis, :]
        vein_detail = np_array["vein_detail"][np.newaxis, :]

        model_input = np.concatenate((raw, raw_aug), axis=0)
        model_input = torch.tensor(model_input).to(self.device).to(torch.float)

        mask = np.concatenate((lung, heart, airway, artery_main, vein_main, artery_detail, vein_detail), axis=0)
        mask = torch.tensor(mask).to(self.device).to(torch.float)

        return model_input, mask

    def __len__(self):
        return len(self.file_list)


def dice_loss(array_1, array_2):
    inter = torch.sum(array_1 * array_2)
    norm = torch.sum(array_1 * array_1) + torch.sum(array_2 * array_2)
    if norm <= 25:
        return 0
    return 1 - 2 * inter / norm


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg


class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):

        total_loss = []
        predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1, 2, 3])

                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


def sad_loss(pred, target, encoder_flag=True):
    target = target.detach()
    if (target.size(-1) == pred.size(-1)) and (target.size(-2) == pred.size(-2)):
        # target and prediction have the same spatial resolution
        pass
    else:
        if encoder_flag == True:
            # target is smaller than prediction
            # use consecutive layers with scale factor = 2
            target = F.interpolate(target, scale_factor=(1, 2, 2), mode='trilinear')
        else:
            # prediction is smaller than target
            # use consecutive layers with scale factor = 2
            pred = F.interpolate(pred, scale_factor=(1, 2, 2), mode='trilinear')

    num_batch = pred.size(0)
    pred = pred.view(num_batch, -1)
    target = target.view(num_batch, -1)
    pred = F.softmax(pred, dim=1)
    target = F.softmax(target, dim=1)
    return F.mse_loss(pred, target)