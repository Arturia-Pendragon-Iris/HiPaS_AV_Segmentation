
from skimage.filters import frangi
import torch
import torch.nn.functional as F
import numpy as np


def apply_gaussian_filter(input_tensor, sigma, truncate=4.0):
    # Calculate the kernel size
    # print(input_tensor.shape)
    kernel_size = int(truncate * sigma) * 2 + 1
    # print(kernel_size)

    # Create a 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gaussian_1d = torch.exp(-(x ** 2) / (sigma ** 2))
    gaussian_1d /= gaussian_1d.sum()

    # Expand to create a 2D Gaussian kernel
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0).to(input_tensor.device)  # Add batch and channel dimensions

    padding = kernel_size // 2
    input_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')
    filtered_tensor = F.conv2d(input_tensor, gaussian_2d, groups=input_tensor.size(1))

    # Apply the Gaussian filter using F.conv2d
    # input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # filtered_tensor = F.conv2d(input_tensor, gaussian_2d, padding=kernel_size // 2)
    return filtered_tensor


def compure_gradient(image: torch.tensor, option):
    x_size = image.shape[-2]
    y_size = image.shape[-1]
    gradient = torch.zeros(image.shape).to(image.device)
    if option == "x":
        gradient[:, :, 0, :] = image[:, :, 1, :] - image[:, :, 0, :]
        gradient[:, :, x_size - 1, :] = image[:, :, x_size - 1, :] - image[:, :, x_size - 2, :]
        gradient[:, :, 1:x_size - 2, :] = (image[:, :, 2:x_size - 1, :] - image[:, :, 0:x_size - 3, :]) / 2
    else:
        gradient[:, :, :, 0] = image[:, :, :, 1] - image[:, :, :, 0]
        gradient[:, :, :, y_size - 1] = image[:, :, :, y_size - 1] - image[:, :, :, y_size - 2]
        gradient[:, :, :, 1:y_size - 2] = (image[:, :, :, 2:y_size - 1] - image[:, :, :, 0:y_size - 3]) / 2
    return gradient


def compute_hessian(image: torch.tensor):
    gy = compure_gradient(image, "y")
    gyy = compure_gradient(gy, "y")
    # print(gy.shape, gyy.shape)

    gx = compure_gradient(image, "x")
    gxx = compure_gradient(gx, "x")
    gxy = compure_gradient(gx, 'y')
    return gxx, gxy, gyy


def compute_hessian_eigenvalues(Gxx, Gxy, Gyy):
    """Compute the eigenvalues of the Hessian matrix."""
    # Trace and determinant of the Hessian

    trace = Gxx + Gyy
    # determinant = Gxx * Gyy - Gxy * Gxy

    # Discriminant of the characteristic polynomial
    discriminant = torch.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2)

    # Eigenvalues
    mu1 = (trace + discriminant) / 2
    mu2 = (trace - discriminant) / 2

    indices = (torch.abs(mu1) > torch.abs(mu2))
    lambda1 = mu1.clone()
    lambda1[indices] = mu2[indices]

    lambda2 = mu2.clone()
    lambda2[indices] = mu1[indices]

    return lambda1, lambda2


def image_eigenvalues(img, sigma, frangi=False):
    # truncate = 8 if sigma > 1 else 100
    truncate = 4.0
    img = apply_gaussian_filter(img, sigma, truncate)
    # print(img.shape)
    gxx, gxy, gyy = compute_hessian(img)
    # print(gxx.max(), gxy.max(), gyy.max())
    # plot_parallel(
    #     a=gxx[0, 0].cpu().numpy(),
    #     b=gxy[0, 0].cpu().numpy(),
    #     c=gyy[0, 0].cpu().numpy()
    # )

    c = sigma ** 2
    if frangi:
        hxx = c * gxx
        hyy = c * gyy
        hxy = c * gxy
    else:
        hxx = -c * gxx
        hyy = -c * gyy
        hxy = -c * gxy

    # B1 = -(hxx + hyy)
    # B2 = hxx * hyy - hxy ** 2
    # T = torch.ones(B1.shape).to(img.device)
    # T[(B1 < 0)] = 0
    # T[(B1 == 0) & (B2 == 0)] = 0
    # T = T.flatten()
    # print(hxx.shape, 1)
    lambda1, lambda2 = compute_hessian_eigenvalues(hxx, hxy, hyy)
    # lambda1 *= T
    # lambda2 *= T
    lambda1 = torch.nan_to_num(lambda1, nan=0.0, posinf=0.0, neginf=0.0)
    lambda2 = torch.nan_to_num(lambda2, nan=0.0, posinf=0.0, neginf=0.0)

    if frangi:
        lambda2[lambda2 < 1e-10] = 1e-10
    else:
        lambda1[torch.abs(lambda1) < 1e-10] = 1e-10
        lambda2[torch.abs(lambda2) < 1e-10] = 1e-10

    return lambda1, lambda2


def jerman_filter_gpu(img, sigma=(0.5, 1.0, 1.5), tau=1,
                      convert=True, device=None, enhance=False,
                      transfer_device=False):
    raw_shape = img.shape

    if transfer_device:
        if len(raw_shape) == 2:
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img).unsqueeze(1).to(torch.float32).to(device)
    else:
        device = img.device

    if enhance:
        img = torch.sqrt(1 - (1 - img) ** 2)

    if convert:
        img = 1 - img

    # print(img.shape)

    for j in range(len(sigma)):
        # print(img.shape)
        lambda1, lambda2 = image_eigenvalues(img, sigma[j])

        lambda3 = lambda2.clone()
        new_tau = tau * torch.min(lambda3)
        lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
        different = lambda3 - lambda2
        response = ((torch.abs(lambda2) ** 2) * torch.abs(different)) * 27 / (
                (2 * torch.abs(lambda2) + torch.abs(different)) ** 3)
        response[(lambda2 < lambda3 / 2)] = 1
        response[(lambda2 >= 0)] = 0

        response = torch.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
        if j == 0:
            output = response
        else:
            output = torch.max(output, response)

    output[(output < 1e-2)] = 0

    if transfer_device:
        if len(raw_shape) == 2:
            output = output.squeeze().cpu().detach().numpy()
        else:
            output = output.squeeze(1).cpu().detach().numpy()
            output = np.transpose(output, (1, 2, 0))
    # print(output)
    return output


def frangi_filter_gpu(
        img,
        sigma=(0.5, 1.0),
        beta=0.5,
        device=None,
        enhance=False,
        transfer_device=False,
):
    """
    Apply the Frangi filter to an image using GPU acceleration with PyTorch.
    Parameters:
    - img (torch.Tensor or np.ndarray): Input image (2D or 3D).
    - sigma (tuple): Sigma values for the Gaussian filter.
    - beta (float): Sensitivity to blob-like structures.
    - device (torch.device): Target device for computation (CPU/GPU).
    - enhance (bool): Whether to enhance the image before processing.
    - transfer_device (bool): Automatically transfer image to the target device.

    Returns:
    - output (torch.Tensor or np.ndarray): Processed image after applying the Frangi filter.
    """

    # Prepare the input image shape and transfer to the correct device if needed.
    raw_shape = img.shape
    if transfer_device:
        if len(raw_shape) == 2:  # 2D image
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        else:  # 3D image
            img = torch.tensor(img).unsqueeze(1).to(torch.float32).to(device)
    else:
        if len(raw_shape) == 2:  # 2D image
            img = img.unsqueeze(0).unsqueeze(0)
        else:  # 3D image
            img = img.unsqueeze(1)

    # Transfer to the specified device if it's not already there.
    if device is not None:
        raw_device = img.device
        if img.device != device:
            img = img.to(device)

    # Optionally enhance the image.
    if enhance:
        img = torch.sqrt(1 - (1 - img) ** 2)

    # Optionally invert the image.
    img = 1 - img

    gamma = None

    # Process the image for each sigma value.
    for j in range(len(sigma)):
        # Compute eigenvalues of the Hessian matrix at the current scale.
        lambda1, lambda2 = image_eigenvalues(img, sigma[j], frangi=True)

        # Calculate the ratio of the eigenvalues (blobness measure).
        r_b = torch.abs(lambda1) / lambda2  # Eq. (15).

        # Calculate the structure tensor magnitude (s).
        s = torch.sqrt(lambda1 ** 2 + lambda2 ** 2)  # Eq. (12).

        # Initialize gamma parameter based on the maximum value of s.
        if gamma is None:
            gamma = s.amax(dim=(2, 3)) / 4
            gamma[abs(gamma < 1e-10)] = 1

        # Compute the Frangi response.
        response = torch.exp(-r_b ** 2 / (2 * beta ** 2))  # Blobness term.
        # print((s ** 2).shape, (2 * gamma ** 2).shape)
        gamma_expanded = gamma.unsqueeze(-1).unsqueeze(-1)
        texture_control = 1 - torch.exp(-s ** 2 / (2 * gamma_expanded ** 2))  # Texture control term.
        response *= texture_control

        # Ensure no invalid values (NaN/Inf) are present in the response.
        response = torch.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine the responses across scales by taking the maximum.
        if j == 0:
            output = response
        else:
            output = torch.max(output, response)

    # Threshold the output to remove small values.
    output[output < 1e-4] = 0

    # Convert the output back to its original format if needed.
    if transfer_device:
        if len(raw_shape) == 2:  # 2D image
            output = output.squeeze().cpu().detach().numpy()
        else:  # 3D image
            output = output.squeeze(1).cpu().detach().numpy()
    else:
        if len(raw_shape) == 2:  # 2D image
            output = output.squeeze()
        else:  # 3D image
            output = output.squeeze(1)

    return output
