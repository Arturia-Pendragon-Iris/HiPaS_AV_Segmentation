import numpy as np
from skimage.measure import label as connect
import skimage.measure as measure


def get_key(d, value):
    return [k for k, v in d.items() if v == value]


def select_region(mask, num, thre=50):
    center = np.array(mask.shape) / 2
    # print(center)
    labels, nums = connect(mask, connectivity=1, return_num=True)
    prop = measure.regionprops(labels)
    label_sum = {}
    # print(nums)
    new_mask = mask * 0
    for label in range(nums):
        if prop[label].area > thre:
            # if prop[label].area < 150:
            label_sum[label + 1] = prop[label].area
    # print(label_sum)
    area_list = []
    for value in label_sum.values():
        area_list.append(value)
    area_list = np.sort(area_list)
    area_list = area_list[::-1]
    # print(area_list)
    if num >= len(area_list):
        num = len(area_list)
    for i in range(num):
        area = area_list[i]
        label = get_key(label_sum, area)[0]
        # visualize_numpy_as_stl(np.array(labels == label, "float32"))
        section = np.array(labels == label, "float32")

        new_mask += section

    return new_mask


def get_region(mask, num, thre=50):
    center = np.array(mask.shape) / 2
    # print(center)
    labels, nums = connect(mask, connectivity=1, return_num=True)
    prop = measure.regionprops(labels)
    label_sum = {}
    # print(nums)
    new_mask = mask * 0
    for label in range(nums):
        if prop[label].area > thre:
            # if prop[label].area < 150:
            label_sum[label + 1] = prop[label].area
    # print(label_sum)
    area_list = []
    for value in label_sum.values():
        area_list.append(value)
    area_list = np.sort(area_list)
    area_list = area_list[::-1]
    # print(area_list)
    if num >= len(area_list):
        num = len(area_list)

    seq = []

    for i in range(num):
        area = area_list[i]
        label = get_key(label_sum, area)[0]
        # visualize_numpy_as_stl(np.array(labels == label, "float32"))
        section = np.array(labels == label, "float32")

        seq.append(section)

    return seq