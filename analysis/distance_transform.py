from scipy.ndimage import distance_transform_edt
# import cv2
# import numpy as np
# from Tool_Functions.Functions import plot_imgs_para


def perform_distance_trans(np_array):
    assert len(np_array.shape) in [2, 3]
    return distance_transform_edt(1 - np_array)

