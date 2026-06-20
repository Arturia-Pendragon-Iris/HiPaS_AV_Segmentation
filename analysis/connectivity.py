import numpy as np
from skimage.measure import label as connect
import skimage.measure as measure


def _get_sorted_regions(mask, thre):
    labels, num_labels = connect(mask, connectivity=1, return_num=True)
    props = measure.regionprops(labels)
    region_map = {
        label + 1: props[label].area
        for label in range(num_labels)
        if props[label].area > thre
    }
    sorted_areas = sorted(region_map.values(), reverse=True)
    return labels, region_map, sorted_areas


def select_region(mask, num, thre=50):
    labels, region_map, sorted_areas = _get_sorted_regions(mask, thre)
    num = min(num, len(sorted_areas))
    new_mask = np.zeros_like(mask)
    for area in sorted_areas[:num]:
        label = next(k for k, v in region_map.items() if v == area)
        new_mask += np.array(labels == label, "float32")
    return new_mask


def get_region(mask, num, thre=50):
    labels, region_map, sorted_areas = _get_sorted_regions(mask, thre)
    num = min(num, len(sorted_areas))
    regions = []
    for area in sorted_areas[:num]:
        label = next(k for k, v in region_map.items() if v == area)
        regions.append(np.array(labels == label, "float32"))
    return regions
