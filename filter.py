import numpy as np
from skimage.filters import frangi
from scipy import ndimage
from scipy.ndimage import gaussian_filter as gaussian


def frangi_filter(np_array, scale_range=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=15, enhance=False):
    arr = np_array.copy()
    if enhance:
        arr = np.sqrt(1 - (1 - arr) ** 2)
    return frangi(arr, sigmas=scale_range, alpha=alpha, beta=beta, gamma=gamma, black_ridges=False)


class vesselness2d:
    def __init__(self, image, sigma, tau):
        self.image = image
        self.sigma = sigma
        self.tau = tau
        self.size = image.shape

    def _gradient(self, arr, axis):
        n = self.size[0] if axis == "x" else self.size[1]
        grad = np.zeros_like(arr)
        if axis == "x":
            grad[0, :] = arr[1, :] - arr[0, :]
            grad[n - 1, :] = arr[n - 1, :] - arr[n - 2, :]
            grad[1:n - 2, :] = (arr[2:n - 1, :] - arr[0:n - 3, :]) / 2
        else:
            grad[:, 0] = arr[:, 1] - arr[:, 0]
            grad[:, n - 1] = arr[:, n - 1] - arr[:, n - 2]
            grad[:, 1:n - 2] = (arr[:, 2:n - 1] - arr[:, 0:n - 3]) / 2
        return grad

    def _hessian2d(self, image, sigma):
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        Dx = self._gradient(image, "x")
        Dy = self._gradient(image, "y")
        return self._gradient(Dx, "x"), self._gradient(Dy, "y"), self._gradient(Dx, "y")

    def _eigvals(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        swap = np.abs(mu1) > np.abs(mu2)
        lam1, lam2 = mu1.copy(), mu2.copy()
        lam1[swap], lam2[swap] = mu2[swap], mu1[swap]
        return lam1, lam2

    def _image_eigenvalues(self, image, sigma):
        hxx, hyy, hxy = self._hessian2d(image, sigma)
        c = sigma ** 2
        hxx, hyy, hxy = -c * hxx, -c * hyy, -c * hxy

        B1, B2 = -(hxx + hyy), hxx * hyy - hxy ** 2
        valid = ((B1 > 0) | ((B1 == 0) & (B2 != 0))).flatten()
        idx = np.where(valid)[0]

        hxx_f, hyy_f, hxy_f = hxx.flatten()[idx], hyy.flatten()[idx], hxy.flatten()[idx]
        lam1i, lam2i = self._eigvals(hxx_f, hyy_f, hxy_f)

        n = self.size[0] * self.size[1]
        lam1, lam2 = np.zeros(n), np.zeros(n)
        lam1[idx], lam2[idx] = lam1i, lam2i

        for lam in (lam1, lam2):
            lam[np.isinf(lam)] = 0
            lam[np.abs(lam) < 1e-4] = 0

        return lam1.reshape(self.size), lam2.reshape(self.size)

    def vesselness2d(self):
        vesselness = None
        for sigma in self.sigma:
            lam1, lam2 = self._image_eigenvalues(self.image, sigma)
            lam3 = lam2.copy()
            new_tau = self.tau * lam3.min()
            lam3[(lam3 < 0) & (lam3 >= new_tau)] = new_tau

            diff = lam3 - lam2
            response = (np.abs(lam2) ** 2 * np.abs(diff)) * 27 / (2 * np.abs(lam2) + np.abs(diff)) ** 3
            response[lam2 < lam3 / 2] = 1
            response[lam2 >= 0] = 0
            response[np.isinf(response)] = 0

            vesselness = response if vesselness is None else np.maximum(vesselness, response)

        vesselness[vesselness < 1e-2] = 0
        return vesselness


def jerman_filter(np_array, tau=1, enhance=False):
    arr = np_array.copy()
    if enhance:
        arr = np.sqrt(1 - (1 - arr) ** 2)
    arr = 255 * (1 - arr)
    return vesselness2d(arr, sigma=[0.5, 1, 1.5], tau=tau).vesselness2d()


def jerman_filter_scan(raw_array, tau=1, enhance=False):
    output = np.zeros(raw_array.shape)
    for j in range(raw_array.shape[-1]):
        slice_ = raw_array[:, :, j]
        if slice_.sum() == 0:
            continue
        output[:, :, j] = jerman_filter(slice_, enhance=enhance, tau=tau)
    return output


def gaussian_filter(np_array, sigma=2, order=0):
    return gaussian(np_array, sigma=sigma, order=order)


def sobel_filter(np_array):
    sobel_h = ndimage.sobel(np_array, 0)
    sobel_v = ndimage.sobel(np_array, 1)
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    return magnitude / magnitude.max()
