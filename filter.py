import numpy as np
from skimage.filters import frangi
from scipy import ndimage
from scipy.ndimage import gaussian_filter as gaussian


# frangi filter
def frangi_filter(np_array, scale_range=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=15, enhance=False):
    new_array = np_array.copy()
    if enhance:
        new_array = np.sqrt(1 - (1 - new_array) ** 2)

    return frangi(new_array, sigmas=scale_range,
                  alpha=alpha, beta=beta, gamma=gamma, black_ridges=False)


# jerman filter
class vesselness2d:
    def __init__(self, image, sigma, tau):
        super(vesselness2d, self).__init__()

        self.image = image
        self.sigma = sigma
        self.tau = tau
        self.size = image.shape

    def gaussian_filter(self, image, sigma):
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        return image

    def gradient_2d(self, np_array, option):
        x_size = self.size[0]
        y_size = self.size[1]
        gradient = np.zeros(np_array.shape)
        if option == "x":
            gradient[0, :] = np_array[1, :] - np_array[0, :]
            gradient[x_size - 1, :] = np_array[x_size - 1, :] - np_array[x_size - 2, :]
            gradient[1:x_size - 2, :] = \
                (np_array[2:x_size - 1, :] - np_array[0:x_size - 3, :]) / 2
        else:
            gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
            gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
            gradient[:, 1:y_size - 2] = \
                (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
        return gradient

    def Hessian2d(self, image, sigma):
        # print(sigma)
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        Dy = self.gradient_2d(image, "y")
        Dyy = self.gradient_2d(Dy, "y")

        Dx = self.gradient_2d(image, "x")
        Dxx = self.gradient_2d(Dx, "x")
        Dxy = self.gradient_2d(Dx, 'y')
        return Dxx, Dyy, Dxy

    def eigval_Hessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        # hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        hyy = -c * hyy
        hxy = -c * hxy

        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]
        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()
        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]
        #     lambda1i, lambda2i = hessian_matrix_eigvals([hxx, hyy, hxy])
        lambda1i, lambda2i = self.eigval_Hessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # removing noise
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2

    def vesselness2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            # return lambda1, lambda2
            # plot_parallel(
            #     a=lambda1,
            #     n=lambda2
            # )

            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                        (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        #     vesselness = vesselness / np.max(vesselness)
        vesselness[(vesselness < 1e-2)] = 0
        #         vesselness = vesselness.reshape(self.size)
        return vesselness


def jerman_filter(np_array, tau=1, enhance=False):
    new_array = np_array.copy()
    if enhance:
        new_array = np.sqrt(1 - (1 - new_array) ** 2)
    new_array = 255 * (1 - new_array)

    sigma = [0.5, 1, 1.5]
    output = vesselness2d(new_array, sigma, tau)
    output = output.vesselness2d()

    return output

# Gaussian filter
def gaussian_filter(np_array, sigma=2, order=0):
    return gaussian(np_array, sigma=sigma, order=order)


def sobel_filter(np_array):
    sobel_h = ndimage.sobel(np_array, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(np_array, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)

    return magnitude / np.max(magnitude)


# kernel filter
def do_highpass_filter(np_array, order=3, average=True):
    # cv2.imshow("org", image)
    if order == 3:
        kernel = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])
    elif order == 5:
        kernel = np.array([[-1, -1, -1, -1, -1],
                             [-1, 1, 2, 1, -1],
                             [-1, 2, 4, 2, -1],
                             [-1, 1, 2, 1, -1],
                             [-1, -1, -1, -1, -1]])
    else:
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    filtered = ndimage.convolve(np_array, kernel)
    if average:
        filtered *= np.mean(np_array) / np.mean(filtered)

    return filtered


def butterworth_highpass_filter(image, d0=30, n=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fft_image = np.fft.fft2(image)

    mask = 1 / (1 + ((np.sqrt((np.arange(rows)[:, np.newaxis] - crow)**2 +
                              (np.arange(cols)[np.newaxis, :] - ccol)**2)) / d0) ** (2 * n))

    # 应用滤波器

    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * (1 - mask)
    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    return result_image


def exponential_highpass_filter(image, d0=30):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建指数高通滤波器
    mask = 1 - np.exp(-((np.arange(rows)[:, np.newaxis] - crow) ** 2 +
                        (np.arange(cols)[np.newaxis, :] - ccol) ** 2) / (2 * d0 ** 2))

    # 应用滤波器
    fft_image = np.fft.fft2(image)
    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * mask
    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    # result_image *= np.mean(image) / np.mean(result_image)
    return result_image




