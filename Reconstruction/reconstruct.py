import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from model import I2SR


def reconstruct_ct_5(raw_ct):
    model = I2SR(nc=32).cuda()
    model.load_state_dict(
        torch.load("/data/models/reconstruction/I2SR_1.pth"))
    # model = Full_CNN().cuda()
    # model.load_state_dict(torch.load(
    #     "/data/Train_and_Test/Artery_Vein/Upsmapling/model/Full_CNN/model_epoch_1.pth"))
    model.half()
    model.eval()

    new_ct = np.zeros([512, 512, 5 * raw_ct.shape[-1] + 2])
    raw_ct = np.transpose(raw_ct, (2, 0, 1))
    input_set = torch.from_numpy(raw_ct[np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()

    with torch.no_grad():
        # inferer = SliceInferer(spatial_dim=2,
        #                        roi_size=(5, 512, 512),
        #                        sw_batch_size=1,
        #                        progress=False)
        for i in range(1, raw_ct.shape[0] - 1):
            pre = model(input_set[:, i - 2:i + 3]).detach().cpu().numpy()[0]
            new_ct[:, :, 5 * i - 2:5 * i + 3] = np.transpose(pre, (1, 2, 0))

    return np.array(new_ct, "float32")

