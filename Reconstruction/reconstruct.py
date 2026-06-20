import numpy as np
import torch
from model import I2SR


def reconstruct_ct_5(raw_ct):
    model = I2SR(nc=32).cuda()
    model.load_state_dict(torch.load("/data/models/reconstruction/I2SR_1.pth"))
    model.half().eval()

    raw_ct = np.transpose(raw_ct, (2, 0, 1))
    input_tensor = torch.from_numpy(raw_ct[np.newaxis]).float().cuda().half()

    new_ct = np.zeros([512, 512, 5 * raw_ct.shape[0] + 2])
    with torch.no_grad():
        for i in range(1, raw_ct.shape[0] - 1):
            pred = model(input_tensor[:, i - 2:i + 3]).detach().cpu().numpy()[0]
            new_ct[:, :, 5 * i - 2:5 * i + 3] = np.transpose(pred, (1, 2, 0))

    return np.array(new_ct, "float32")
