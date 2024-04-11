import argparse, os
import torch
from analysis.connectivity import select_region, get_region
import visualization.view_3D as view
from monai.inferers import SlidingWindowInferer, sliding_window_inference
import numpy as np
from HiPaS.model import HiPaSNet
from analysis.filter.filter_2D import jerman_filter_scan, jerman_filter_xyz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


def predict_lung(ct_array):
    model = HiPaSNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("/Artery_Vein/lung.pth"))
    model.half()
    model.eval()
    model = model.to('cuda')

    ct_array = torch.tensor(ct_array[np.newaxis, np.newaxis]).to(torch.float).to(device).half()
    with torch.no_grad():
        pre = sliding_window_inference(inputs=ct_array,
                                       predictor=model,
                                       roi_size=(192, 192, 128),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")
    pre = pre.detach().cpu().numpy()[0, 0]
    lung = select_region(np.array(pre > 0.52, "float32"), num=2)
    return lung


def predict_av(ct_array, lung=None):
    if lung is None:
        lung = predict_lung(ct_array)

    loc = np.array(np.where(lung > 0))
    x_min, x_max = np.min(loc[0]), np.max(loc[0])
    y_min, y_max = np.min(loc[1]), np.max(loc[1])
    z_min, z_max = np.min(loc[2]), np.max(loc[2])
    filtered = jerman_filter_scan(ct_array[x_min:x_max, y_min:y_max, z_min:z_max], enhance=True)

    input_set = np.stack((ct_array[x_min:x_max, y_min:y_max, z_min:z_max], filtered), axis=0)
    input_set = torch.from_numpy(input_set[np.newaxis]).float()
    input_set = input_set.half()

    model_0 = HiPaSNet(in_channels=2, out_channels=2)
    model_0.load_state_dict(torch.load("/Artery_Vein/stage_0.pth"))
    model_0.half()
    model_0.eval()
    model_0 = model_0.to('cuda')

    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(192, 192, 128),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")
        pre_0 = inferer(inputs=input_set, network=model_0)

    model_1 = HiPaSNet(in_channels=2, out_channels=2)
    model_1.load_state_dict(torch.load("/Artery_Vein/stage_1.pth"))
    model_1.half()
    model_1.eval()
    model_1 = model_0.to('cuda')

    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(192, 192, 128),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")
        pre_1 = inferer(inputs=(input_set, pre_0), network=model_1)

    model_2 = HiPaSNet(in_channels=2, out_channels=2)
    model_2.load_state_dict(torch.load("/Artery_Vein/stage_2.pth"))
    model_2.half()
    model_2.eval()
    model_2 = model_0.to('cuda')

    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(192, 192, 128),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")
        pre_2 = inferer(inputs=(input_set, pre_1), network=model_2)

    model_3 = HiPaSNet(in_channels=2, out_channels=2)
    model_3.load_state_dict(torch.load("/Artery_Vein/stage_3.pth"))
    model_3.half()
    model_3.eval()
    model_3 = model_0.to('cuda')

    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(192, 192, 128),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=True,
                                       sw_device="cuda",
                                       device="cpu")
        pre_3 = inferer(inputs=(input_set, pre_2), network=model_3)

    final_pre = np.zeros([2, ct_array.shape[0], ct_array.shape[1], ct_array.shape[-1]])
    final_pre[0, x_min:x_max, y_min:y_max, z_min:z_max] = pre_3[0]
    final_pre[1, x_min:x_max, y_min:y_max, z_min:z_max] = pre_3[1]
    return (np.array(final_pre[0] > 0.52, "float32"),
            np.array(final_pre[1] > 0.52, "float32"))


if __name__ == "__main__":
    ct = np.load("/data/chest_CT/rescaled_ct/xwzc/ct_scan/xwzc000022.npz")["arr_0"]
    ct = np.clip((ct + 1000) / 1400, 0, 1)
    a, v = predict_av(ct)
    view.visualize_two_numpy(a, v)



