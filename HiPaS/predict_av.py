import os
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer

from analysis.connectivity import select_region
import view_3D as view
from HiPaS.model import HiPaSNet
from filter import jerman_filter_scan

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0")

_INFERER = SlidingWindowInferer(
    roi_size=(192, 192, 128),
    sw_batch_size=1,
    overlap=0.25,
    mode="gaussian",
    sigma_scale=0.25,
    progress=True,
    sw_device="cuda",
    device="cpu",
)

MODEL_DIR = "/Artery_Vein"
NUM_STAGES = 4


def _load_model(path, in_channels, out_channels, mid_channels=16):
    model = HiPaSNet(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
    model.load_state_dict(torch.load(path))
    model.half().eval()
    return model.to(DEVICE)


def predict_lung(ct_array):
    model = _load_model(os.path.join(MODEL_DIR, "lung.pth"), in_channels=1, out_channels=1)
    ct_tensor = torch.tensor(ct_array[np.newaxis, np.newaxis]).float().to(DEVICE).half()
    with torch.no_grad():
        pred = _INFERER(inputs=ct_tensor, network=model)
    pred = pred.detach().cpu().numpy()[0, 0]
    return select_region(np.array(pred > 0.52, "float32"), num=2)


def predict_av(ct_array, lung=None):
    if lung is None:
        lung = predict_lung(ct_array)

    loc = np.where(lung > 0)
    x_min, x_max = loc[0].min(), loc[0].max()
    y_min, y_max = loc[1].min(), loc[1].max()
    z_min, z_max = loc[2].min(), loc[2].max()

    ct_crop = ct_array[x_min:x_max, y_min:y_max, z_min:z_max]
    filtered = jerman_filter_scan(ct_crop, enhance=True)

    input_tensor = torch.from_numpy(
        np.stack([ct_crop, filtered], axis=0)[np.newaxis]
    ).float().half()

    pred = None
    for stage in range(NUM_STAGES):
        model = _load_model(
            os.path.join(MODEL_DIR, f"stage_{stage}.pth"),
            in_channels=2, out_channels=2, mid_channels=24,
        )
        with torch.no_grad():
            pred = _INFERER(inputs=(input_tensor, pred), network=model)

    final = np.zeros([2, *ct_array.shape])
    final[0, x_min:x_max, y_min:y_max, z_min:z_max] = pred[0, 0].cpu().numpy()
    final[1, x_min:x_max, y_min:y_max, z_min:z_max] = pred[0, 1].cpu().numpy()
    return (
        np.array(final[0] > 0.52, "float32"),
        np.array(final[1] > 0.52, "float32"),
    )


if __name__ == "__main__":
    ct = np.load("/data/chest_CT/rescaled_ct/xwzc/ct_scan/xwzc000022.npz")["arr_0"]
    ct = np.clip((ct + 1000) / 1400, 0, 1)
    artery, vein = predict_av(ct)
    view.visualize_two_numpy(artery, vein)
