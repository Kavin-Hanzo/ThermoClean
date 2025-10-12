"""
Evaluator.py

Utilities to compute image-quality metrics between a reference (original) image
and a test (denoised) image.

Functions:
 - compute_image_metrics(ref, test, resize_ref=True)
   ref: path to RGB/gray image or numpy array (H,W) or (H,W,3)
   test: numpy array (H_t, W_t) or (H_t, W_t, 3) — expected float in [0,1]
   resize_ref: if True, ref will be resized to test shape when needed.

Returns dict with keys: mse, mae, psnr, ssim, cv (coefficient of variation).
"""

from typing import Union, Dict
import numpy as np
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings

ArrayLike = Union[str, np.ndarray]

def _load_and_gray(path_or_arr: ArrayLike, target_shape=None, resize_ref: bool = True) -> np.ndarray:
    """
    Load an image from path or accept an ndarray; return grayscale float image in [0,1]
    Optionally resize to target_shape (h,w) if provided and resize_ref True.
    """
    if isinstance(path_or_arr, str):
        img = img_as_float(io.imread(path_or_arr)).astype(np.float32)
    elif isinstance(path_or_arr, np.ndarray):
        img = img_as_float(path_or_arr).astype(np.float32)
    else:
        raise ValueError("ref must be file path or numpy array")

    # Convert to grayscale if multi-channel
    if img.ndim == 3:
        # handle alpha channel if present
        if img.shape[2] == 4:
            img = img[..., :3]
        imgg = rgb2gray(img)
    else:
        imgg = img

    imgg = np.clip(imgg, 0.0, 1.0)

    if target_shape is not None and resize_ref:
        th, tw = target_shape
        if (int(imgg.shape[0]) != int(th)) or (int(imgg.shape[1]) != int(tw)):
            # resize with anti_aliasing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imgg = resize(imgg, (th, tw), anti_aliasing=True, preserve_range=True)
            # ensure in [0,1]
            imgg = np.clip(imgg.astype(np.float32), 0.0, 1.0)
    return imgg.astype(np.float32)

def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean(np.abs(a - b)))

def cv_np(a: np.ndarray) -> float:
    """Coefficient of variation = std / mean (useful non-reference quality stat)"""
    a = a.astype(np.float64)
    mean = np.mean(a)
    std = np.std(a)
    if abs(mean) < 1e-12:
        return float(np.inf)
    return float(std / mean)

def compute_image_metrics(ref: ArrayLike, test: np.ndarray, resize_ref: bool = True) -> Dict[str, float]:
    """
    Compute MSE, MAE, PSNR, SSIM, CV between reference image (ref) and test image (test).
    - ref: path or array (RGB or grayscale) — will be converted to grayscale
    - test: array (grayscale or RGB) — if RGB converted to grayscale prior to metric
    - resize_ref: if True, ref will be resized to test shape if shapes mismatch

    Returns dictionary:
      {'mse':..., 'mae':..., 'psnr':..., 'ssim':..., 'cv_ref':...}
    """
    if not isinstance(test, np.ndarray):
        raise ValueError("test must be numpy array")

    # ensure test is grayscale float in [0,1]
    if test.ndim == 3:
        if test.shape[2] == 4:
            test = test[..., :3]
        test_gray = rgb2gray(test) if test.ndim == 3 else test
    else:
        test_gray = test
    test_gray = np.clip(test_gray.astype(np.float32), 0.0, 1.0)

    ref_gray = _load_and_gray(ref, target_shape=test_gray.shape, resize_ref=resize_ref)

    # metrics
    m = {}
    m['mse'] = mse_np(ref_gray, test_gray)
    m['mae'] = mae_np(ref_gray, test_gray)
    # PSNR: use data_range=1.0 (we ensure both in [0,1])
    m['psnr'] = float(peak_signal_noise_ratio(ref_gray, test_gray, data_range=1.0))
    # SSIM: structural_similarity; for 2D images multichannel=False
    try:
        m['ssim'] = float(structural_similarity(ref_gray, test_gray, data_range=1.0))
    except Exception:
        # fallback if structural_similarity fails
        m['ssim'] = float(structural_similarity(ref_gray, test_gray, data_range=1.0, gaussian_weights=True))
    m['cv'] = float(cv_np(ref_gray))
    return m
