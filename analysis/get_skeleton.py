import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=3):

    new_img = torch.from_numpy(img[np.newaxis, np.newaxis, ...])

    img1 = soft_open(new_img)
    skel = F.relu(new_img-img1)
    for j in range(iter_):
        new_img = soft_erode(new_img)
        img1 = soft_open(new_img)
        delta = F.relu(new_img - img1)
        skel = skel + F.relu(delta-skel*delta)
    return np.squeeze(skel.numpy())