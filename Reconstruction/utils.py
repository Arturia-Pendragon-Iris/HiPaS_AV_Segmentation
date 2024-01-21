import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandHistogramShift,
    ImageFilter,
    Rand2DElastic,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAffine)


train_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.4, 0.5)),
     RandFlip(prob=0.5),
     RandRotate90(prob=0.5)
    ]
)


def refine_ct(ct_array):
    k = np.random.randint(low=-1000, high=-600)
    mu = np.random.randint(low=800, high=1800)
    ct_array = np.clip((ct_array - k) / mu, 0, 1)

    return ct_array


class TrainSetLoader(Dataset):
    def __init__(self, device):
        super(TrainSetLoader, self).__init__()
        total_list = []
        with open("/data/chest_CT/file_list.txt", "r") as file:
            sub_list = [line.strip() for line in file]
            total_list.extend(sub_list)

        self.file_list = total_list
        print("data number is", len(self.file_list))
        self.device = device

    def __getitem__(self, index):
        raw_array = np.load(self.file_list[index])["arr_0"]
        while len(raw_array.shape) != 3 or \
                raw_array.shape[-1] < 45 or \
                raw_array.shape[0] != 512:
            # print(self.file_list[index])
            index = np.random.randint(low=0, high=len(self.file_list))
            raw_array = np.load(self.file_list[index])["arr_0"]

        raw_array = refine_ct(raw_array)
        slice_index = np.random.randint(low=20, high=raw_array.shape[-1] - 20)

        # gt = raw_array[:, :, slice_index - 2:slice_index + 3]
        # raw = np.zeros([512, 512, 5])
        # for i in range(5):
        #     upper = slice_index + (i * 5 - 12)
        #     raw[:, :, i] = np.mean(raw_array[:, :, upper:upper + 5], axis=-1)

        gt = np.transpose(raw_array[:, :, slice_index - 2:slice_index + 3],
                          (2, 0, 1))
        raw = np.zeros([5, 512, 512])
        for i in range(5):
            upper = slice_index + (i * 5 - 12)
            raw[i] = np.mean(raw_array[:, :, upper:upper + 5], axis=-1)

        raw_tensor = torch.tensor(raw).clone()\
            .to(torch.float).cuda()
        gt_tensor = torch.tensor(gt).clone() \
            .to(torch.float).cuda()
        return raw_tensor, gt_tensor

    def __len__(self):
        return len(self.file_list)