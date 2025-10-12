"""
oa_helpers.py
Common helper utilities extracted from other_author.py for reuse by per-method modules.
"""
from typing import Tuple
import numpy as np
from scipy import fftpack, ndimage
from skimage.util import img_as_float32
import cv2
import warnings


def normalize01(img: np.ndarray) -> np.ndarray:
    img = np.array(img, dtype=np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return img - mn


def fit_2d_poly(img: np.ndarray, order: int = 2) -> np.ndarray:
    h, w = img.shape
    Y, X = np.mgrid[0:h, 0:w]
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((X ** i) * (Y ** j))
    A = np.stack([t.ravel() for t in terms], axis=1)
    b = img.ravel()
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    fitted = (A @ coeffs).reshape(h, w)
    return fitted


def butterworth_lowpass(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    P, Q = shape
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U ** 2 + V ** 2)
    H = 1.0 / (1.0 + (D / (cutoff + 1e-8)) ** (2 * order))
    return fftpack.ifftshift(H)


def butterworth_filter(img: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    f = fftpack.fft2(img)
    H = butterworth_lowpass(img.shape, cutoff, order)
    return np.real(fftpack.ifft2(f * H))


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma=sigma)


# Export names
__all__ = [
    'normalize01', 'fit_2d_poly', 'butterworth_lowpass', 'butterworth_filter', 'gaussian_blur', 'img_as_float32'
]
