import torch, os
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# get 3D surface
class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


class GetRimLoose(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetRimLoose, self).__init__()
        super().__init__()
        kernel = [[[[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv2d(x, self.weight, padding=1)
        return x


class GetRimStrict(nn.Module):
    # if the adjacency is strictly defined, use this to get rim
    def __init__(self):
        super(GetRimStrict, self).__init__()
        super().__init__()
        kernel = [[[[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv2d(x, self.weight, padding=1)
        return x


class GetSurfaceLoose(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetSurfaceLoose, self).__init__()
        super().__init__()
        kernel = [[[[[-1, -1, -1],
                     [-1, -1, -1],
                     [-1, -1, -1]],
                    [[-1, -1, -1],
                     [-1, 26, -1],
                     [-1, -1, -1]],
                    [[-1, -1, -1],
                     [-1, -1, -1],
                     [-1, -1, -1]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


class GetSurfaceStrict(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetSurfaceStrict, self).__init__()
        super().__init__()
        kernel = [[[[[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]],
                    [[0, -1, 0],
                     [-1, 6, -1],
                     [0, -1, 0]],
                    [[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


class GroupAdjacentValues(nn.Module):
    """
    central pixel take weight 1, the six most adjacent pixels take weight 1/6, the 12 next most adjacent pixels take
    weight 1/12 (distance sqrt(2), weight 1 / 6 / 2), the 8 pixels with distance sqrt(3) with weight 1 / 6 / 3.
    """
    def __init__(self):
        super(GroupAdjacentValues, self).__init__()
        super().__init__()
        kernel = [[[[[1/18, 1/12, 1/18],
                     [1/12, 1/6, 1/12],
                     [1/18, 1/12, 1/18]],
                    [[1/12, 1/6, 1/12],
                     [1/6, 1, 1/6],
                     [1/12, 1/6, 1/12]],
                    [[1/18, 1/12, 1/18],
                     [1/12, 1/6, 1/12],
                     [1/18, 1/12, 1/18]]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = func.conv3d(x, self.weight, padding=1)
        return x


def get_rim(binary_mask, outer=False, strict=True):
    """
    :param strict: see kernel in GetRimStrict and GetRimStrict
    :param binary_mask: with shape [batch size, x, y] or [x, y]
    :param outer: outer rim?
    :return: binary mask, same shape with input, with 1 means rim
    """
    binary_mask = np.array(binary_mask, 'float32')
    if not strict:
        convolution_layer = GetRimLoose().cuda()
    else:
        convolution_layer = GetRimStrict().cuda()
    if torch.cuda.device_count() > 1:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(binary_mask)
    if len(shape) == 2:
        array = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
    elif len(shape) == 3:
        array = torch.from_numpy(binary_mask).unsqueeze(1)
    else:
        raise DimensionError(binary_mask)

    # now the array in shape [batch_size, 1, x, y]
    rim = convolution_layer(array.cuda())
    rim = rim.to('cpu')
    rim = rim.data.numpy()
    if outer:
        rim = np.array(rim < -0.1, 'float32')
    else:
        rim = np.array(rim > 0.1, 'float32')
    if len(shape) == 2:
        return rim[0, 0, :, :]  # [x, y]
    else:
        return rim[:, 0, :, :]  # [batch_size, x, y]


def get_surface_3D(binary_mask, outer=False, strict=False, parallel=False):
    """
    :param parallel: use more than one GPU ?
    :param strict: see kernel in GetSurfaceStrict and GetSurfaceStrict
    :param binary_mask: with shape [batch size, x, y, z] or [x, y, z]
    :param outer: outer surface?
    :return: binary mask, same shape with input, with 1 means surface
    """

    if not strict:
        convolution_layer = GetSurfaceLoose().to('cuda').to(torch.float)
    else:
        convolution_layer = GetSurfaceStrict().to('cuda').to(torch.float)
    if torch.cuda.device_count() > 1 and parallel:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(binary_mask)
    if len(shape) == 3:
        array = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0).to(torch.float)
    elif len(shape) == 4:
        array = torch.from_numpy(binary_mask).unsqueeze(1).to(torch.float)
    else:
        raise DimensionError(binary_mask)

    # now the array in shape [batch_size, 1, x, y, z]
    surface = convolution_layer(array.cuda())
    surface = surface.to('cpu')
    surface = surface.data.numpy()
    if outer:
        surface = np.array(surface < -0.1, 'float32')
    else:
        surface = np.array(surface > 0.1, 'float32')
    if len(shape) == 3:
        return surface[0, 0, :, :, :]  # [x, y, z]
    else:
        return surface[:, 0, :, :, :]  # [batch_size, x, y, z]


def soft_cast_to_binary(probability_mask, threshold=0.25):
    """

    :param probability_mask:
    :param threshold:
    :return:
    """
    convolution_layer = GroupAdjacentValues().cuda()
    if torch.cuda.device_count() > 1:
        convolution_layer = nn.DataParallel(convolution_layer)
    shape = np.shape(probability_mask)
    assert len(shape) == 3
    array = torch.from_numpy(probability_mask).unsqueeze(0).unsqueeze(0)

    # now the array in shape [1, 1, x, y, z]
    grouped_adjacent_array = convolution_layer(array.cuda())
    grouped_adjacent_array = grouped_adjacent_array.to('cpu')
    grouped_adjacent_array = grouped_adjacent_array.data.numpy()[0, 0, :, :, :]  # [x, y, z]

    max_value = np.max(grouped_adjacent_array)
    binary_mask = np.array(grouped_adjacent_array > max_value * threshold, 'float32')

    return binary_mask


def get_contour(bin_img):
    # get contour
    contour_img = np.zeros(shape=(bin_img.shape),dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(bin_img[i][j]==0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum == 0:
                    contour_img[i][j] = 255

    return contour_img


