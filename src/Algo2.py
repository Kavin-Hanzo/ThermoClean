#!/usr/bin/env python3
"""
Algo2.py

Corrected implementation of BFBSF+ and BFBSF (progressive bilateral-filter + Bézier-surface fitting)
with the following important fixes:
 - use skimage.restoration.denoise_bilateral to avoid unit mismatch for sigma_color
 - stabilize Bézier least-squares with Tikhonov (ridge) regularization
 - ensure enough Bézier sampling or lower degree to avoid underdetermined fits
 - improved HOG image regularizer: compute mean HOG vector across 4 blocks and sum squared deviations
 - weight orientation votes in R_bias by gradient magnitude
 - avoid confusing variable names and minor logic bugs

Functions:
 - bfbsf_plus(z, ...)  # full algorithm (research: re-search BF params each iter)
 - bfbsf(z, ...)       # faster algorithm (search once)
 - utility functions: bilateral_filter_img, R_bias, R_image_hog, fit_bezier_surface, etc.

Author: Sivakavin
"""

from typing import Tuple, List, Optional
import numpy as np
from skimage.restoration import denoise_bilateral
from skimage.feature import hog
import cv2
from tqdm import trange
import math

# -------------------------
# Utility functions
# -------------------------
def bilateral_filter_img(img: np.ndarray, sigma_space: float, sigma_color: float) -> np.ndarray:
    """
    Bilateral filtering using skimage (works on float images in [0,1]).
    - img: 2D float image in [0,1] (if not, we convert)
    - sigma_space: spatial sigma in pixels
    - sigma_color: radiometric sigma in intensity units (0..1)
    Returns filtered image (float in [0,1]).
    """
    # convert to float [0,1] if necessary
    if img.dtype == np.uint8:
        imgf = img.astype(np.float32) / 255.0
    else:
        imgf = img.astype(np.float32)
        # if values outside [0,1], normalize conservatively
        if imgf.min() < 0.0 or imgf.max() > 1.0:
            imgf = (imgf - imgf.min()) / (imgf.max() - imgf.min() + 1e-12)
    # skimage denoise_bilateral expects sigma_color in the same units as img (0..1)
    # sigma_spatial is in pixels
    try:
        res = denoise_bilateral(imgf, sigma_color=float(sigma_color), sigma_spatial=float(sigma_space),
                                multichannel=False)
    except Exception:
        # fallback: if denoise_bilateral fails for some reason, use OpenCV scaled approach
        img8 = np.clip(imgf * 255.0, 0, 255).astype(np.uint8)
        sigma_color_cv = float(sigma_color) * 255.0
        diameter = int(max(3, 2 * math.ceil(2 * sigma_space) + 1))
        res8 = cv2.bilateralFilter(img8, diameter, sigma_color_cv, sigma_space)
        res = res8.astype(np.float32) / 255.0
    return res.astype(np.float32)

def gradient_and_orientation(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation.
    - img: 2D float
    Returns (magnitude, orientation) where orientation in radians in [0,2pi).
    """
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx)  # [-pi,pi]
    ori = np.mod(ori, 2*np.pi)
    return mag, ori

# -------------------------
# Regularizers
# -------------------------
def R_bias(bias: np.ndarray, radiation_center: Optional[tuple]=None) -> float:
    """
    Bias-field regularizer:
     - L1 norm of gradient magnitudes (encourage smoothness)
     - orientation consistency: gradients pointing toward radiation_center are rewarded (weighted by magnitude)
    Returns scalar regularizer (lower is better in practice, but we return sum; caller weights it).
    """
    h, w = bias.shape
    mag, ori = gradient_and_orientation(bias)
    l1_grad = np.sum(np.abs(mag))

    if radiation_center is None:
        cx, cy = (w-1)/2.0, (h-1)/2.0
    else:
        cx, cy = radiation_center

    ys, xs = np.indices(bias.shape)
    vec_x = cx - xs
    vec_y = cy - ys
    theta_center = np.mod(np.arctan2(vec_y, vec_x), 2*np.pi)

    # angular difference in [0, pi]
    diff = np.abs(np.angle(np.exp(1j*(ori - theta_center))))
    aligned = (diff <= (np.pi/4.0)).astype(np.float32)  # within +/- 45 degrees
    # weight by gradient magnitude (so strong edges count more)
    orientation_score = np.sum(aligned * mag)

    # combine: L1 smoothness penalizes large gradients, orientation_score rewards center-pointing gradients.
    # We want a regularizer that is small for bias that points toward center and is smooth.
    # So return l1_grad - orientation_score (smaller is better). Caller will weight with beta (positive).
    reg = float(l1_grad - orientation_score)
    return reg

def R_image_hog(img: np.ndarray) -> float:
    """
    HOG-based image regularizer (improved):
    - split image into 4 quadrants, compute HOG vector for each
    - compute mean HOG vector across the 4 blocks
    - return sum over blocks of squared L2 norm of (hog_block - mean_hog)
    This encourages isotropy (blocks have similar gradient orientation histograms).
    """
    h, w = img.shape
    cx, cy = w // 2, h // 2
    # Define blocks: 4 corners and center
    blocks = [
        (0, cy, 0, cx),    # top-left
        (0, cy, cx, w),    # top-right
        (cy, h, cx, w),    # bottom-right
        (cy, h, 0, cx),    # bottom-left
        # center block
        (h//4, 3*h//4, w//4, 3*w//4),  # center (covers middle half)
    ]
    hogs = []
    for (y0, y1, x0, x1) in blocks:
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            hogs.append(np.zeros(9, dtype=np.float32))
            continue
        ph, pw = patch.shape
        cell_h = max(1, ph // 3)
        cell_w = max(1, pw // 3)
        try:
            hog_vec = hog(patch, pixels_per_cell=(cell_h, cell_w), cells_per_block=(1,1), orientations=9, feature_vector=True)
            hogs.append(hog_vec.astype(np.float32))
        except Exception:
            gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            ang = np.mod(np.arctan2(gy, gx), 2*np.pi).flatten()
            bins = np.linspace(0, 2*np.pi, 10)
            hist, _ = np.histogram(ang, bins=bins)
            hogs.append(hist.astype(np.float32))

    # stack and compute mean vector
    H = np.vstack([hvec.reshape(1, -1) for hvec in hogs])  # shape (5, M)
    mean_vec = np.mean(H, axis=0, keepdims=True)  # shape (1, M)
    diffs = H - mean_vec
    total = float(np.sum(diffs**2))
    return total

# -------------------------
# Bézier surface fitting (with ridge regularization)
# -------------------------
def bernstein(n: int, i: int, t: np.ndarray) -> np.ndarray:
    """Bernstein polynomial B_i^n(t) for vector t"""
    from math import comb
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_basis_matrix(m: int, n: int, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    U, V = np.meshgrid(us, vs, indexing='xy')
    U = U.flatten()
    V = V.flatten()
    nsamples = U.size
    nbasis = (m+1) * (n+1)
    A = np.zeros((nsamples, nbasis), dtype=np.float64)
    idx = 0
    for i in range(m+1):
        Bi = bernstein(m, i, U)
        for j in range(n+1):
            Bj = bernstein(n, j, V)
            A[:, idx] = Bi * Bj
            idx += 1
    return A

def fit_bezier_surface(img: np.ndarray, degree_u: int=6, degree_v: int=6, downsample: int=8, ridge: float=1e-6) -> np.ndarray:
    """
    Fit Bézier surface to img with ridge regularization.
    - Ensures stable solution even if sample count is near nbasis.
    - downsample controls the sampling density for LS fit.
    """
    h, w = img.shape
    # ensure we sample at least nbasis points
    nbasis = (degree_u + 1) * (degree_v + 1)
    # choose sampling count heuristically
    samp_h = max(2, h // downsample)
    samp_w = max(2, w // downsample)
    # if too few samples, increase sampling grid
    if samp_h * samp_w < nbasis:
        # increase samp_h/samp_w to roughly meet nbasis
        # try square-ish sampling
        side = int(math.ceil(math.sqrt(nbasis)))
        samp_h = min(h, side)
        samp_w = min(w, side)
    ys = np.linspace(0, h-1, samp_h).astype(int)
    xs = np.linspace(0, w-1, samp_w).astype(int)
    sample_vals = img[np.ix_(ys, xs)].astype(np.float64).flatten()
    us = xs.astype(np.float64) / (w - 1) if w > 1 else np.array([0.0])
    vs = ys.astype(np.float64) / (h - 1) if h > 1 else np.array([0.0])
    A = bezier_basis_matrix(degree_u, degree_v, us, vs)  # (nsamples, nbasis)
    # regularized least squares: (A^T A + ridge I) p = A^T s
    ATA = A.T @ A
    ATA += ridge * np.eye(ATA.shape[0])
    rhs = A.T @ sample_vals
    # solve
    p = np.linalg.solve(ATA, rhs)
    # evaluate at full resolution
    us_full = np.linspace(0.0, 1.0, w)
    vs_full = np.linspace(0.0, 1.0, h)
    A_full = bezier_basis_matrix(degree_u, degree_v, us_full, vs_full)
    S_full = (A_full @ p).reshape((h, w))
    return S_full.astype(np.float32)

# -------------------------
# BF parameter search (multi-scale grid)
# -------------------------
def select_best_bf_candidate(z: np.ndarray, f_curr: np.ndarray,
                             sigma_space_grid: List[float], sigma_color_grid: List[float],
                             eta: float, alpha: float, beta: float, radiation_center=None) -> Tuple[np.ndarray, float]:
    """
    Search BF parameter grid and return best candidate bias (bilateral output) and objective J.
    J = eta * || z - f - BF ||_2^2  + alpha * R(z - BF) + beta * R(BF)
    Assumes z and f_curr are float in [0,1].
    """
    best_J = float('inf')
    best_b = None
    # iterate
    for ss in sigma_space_grid:
        for sc in sigma_color_grid:
            bf = bilateral_filter_img(z, sigma_space=float(ss), sigma_color=float(sc))
            residual = z - bf
            data_term = eta * float(np.sum((z - f_curr - bf)**2))
            reg_img = alpha * float(R_image_hog(residual))
            reg_bias = beta * float(R_bias(bf, radiation_center=radiation_center))
            J = data_term + reg_img + reg_bias
            if J < best_J:
                best_J = J
                best_b = bf
    return best_b, best_J

# -------------------------
# Progressive BFBSF algorithms (fixed)
# -------------------------
def _build_sigma_grids(space_scales: List[float], color_scales: List[float]) -> Tuple[List[float], List[float]]:
    """
    Given base scales, build coarse->fine candidate lists (keeps values in natural float units).
    space_scales expected in pixels; color_scales expected in [0,1] radiometric units.
    """
    sigma_space_grid = sorted(set([max(0.5, s * factor) for s in space_scales for factor in (0.5, 1.0, 1.5)]))
    sigma_color_grid = sorted(set([max(1e-4, c * factor) for c in color_scales for factor in (0.5, 1.0, 2.0)]))
    return sigma_space_grid, sigma_color_grid

def bfbsf_plus(z: np.ndarray, n_iter: int=20, eta: float=0.15, alpha: float=0.6, beta: float=0.95, gamma: float=0.3,
               sigma_space_scales: List[float]=[4,8,16], sigma_color_scales: List[float]=[0.02,0.05,0.12],
               bezier_degree: Tuple[int,int]=(3,3), downsample: int=8, radiation_center=None, verbose: bool=True):
    """
    BFBSF+ (re-search BF params each iteration).
    z: degraded input (float [0,1])
    Returns final latent image estimate f_final and list of biases (each is Bézier-smoothed bias)
    """
    z = z.astype(np.float32)
    f_i = z.copy()  # initial latent estimate (paper uses current estimate; check paper semantics)
    biases = []
    sigma_space_grid, sigma_color_grid = _build_sigma_grids(sigma_space_scales, sigma_color_scales)
    it_range = trange(n_iter, desc='BFBSF+') if verbose else range(n_iter)
    for i in it_range:
        best_bf, bestJ = select_best_bf_candidate(z, f_i, sigma_space_grid, sigma_color_grid, eta, alpha, beta, radiation_center)
        # fit Bézier surface to best_bf
        b_bezier = fit_bezier_surface(best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1], downsample=downsample, ridge=1e-4)
        biases.append(b_bezier)
        f_i = f_i - gamma * b_bezier
        if verbose:
            it_range.set_postfix({'iter': i+1, 'J': float(bestJ)})
    return f_i, biases

def bfbsf(z: np.ndarray, n_iter: int=20, eta: float=0.2, alpha: float=0.6, beta: float=0.9, gamma: float=0.3,
          sigma_space_scales: List[float]=[8,16,32], sigma_color_scales: List[float]=[0.02,0.05,0.12],
          bezier_degree: Tuple[int,int]=(4,4), downsample: int=8, radiation_center=None, verbose: bool=True):
    """
    BFBSF (search BF params once, reuse).
    """
    z = z.astype(np.float32)
    f_i = z.copy()
    biases = []
    sigma_space_grid, sigma_color_grid = _build_sigma_grids(sigma_space_scales, sigma_color_scales)

    best_bf, bestJ = select_best_bf_candidate(z, f_i, sigma_space_grid, sigma_color_grid, eta, alpha, beta, radiation_center)
    if verbose:
        print("Initial BF selection done, J=", bestJ)

    for i in trange(n_iter, desc='BFBSF' if verbose else None):
        b_bezier = fit_bezier_surface(best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1], downsample=downsample, ridge=1e-4)
        biases.append(b_bezier)
        f_i = f_i - gamma * b_bezier
    return f_i, biases

# -------------------------
# Example usage / quick smoke test
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float
    import os

    # Put path to a grayscale degraded image (float range will be normalized to [0,1])
    path = "output_degraded.png"
    if not os.path.exists(path):
        # if example file not present, create a tiny synthetic test
        print("No 'output_degraded.png' found — creating a synthetic test image.")
        I = np.zeros((256, 256), dtype=np.float32)
        # add synthetic bright hotspot (for quick visual)
        cx, cy = 64, 64
        xv, yv = np.meshgrid(np.arange(256), np.arange(256))
        r2 = (xv - cx)**2 + (yv - cy)**2
        I += 0.5 * np.exp(-r2 / (2 * (24**2)))
        # add texture / content
        I += 0.3 * (np.sin(xv / 10.0) * np.cos(yv / 14.0)).astype(np.float32)
        I = np.clip(I, 0.0, 1.0)
    else:
        I = img_as_float(io.imread(path, as_gray=True)).astype(np.float32)
        I = np.clip(I, 0.0, 1.0)

    # Run BFBSF (fast)
    corrected_fast, biases_fast = bfbsf(I, n_iter=3, verbose=True)

    # Run BFBSF+ (full)
    corrected_plus, biases_plus = bfbsf_plus(I, n_iter=2, verbose=True)

    # Show results
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Original degraded")
    plt.imshow(I, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("BFBSF (fast)")
    plt.imshow(np.clip(corrected_fast,0,1), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("BFBSF+ (full)")
    plt.imshow(np.clip(corrected_plus,0,1), cmap='gray')
    plt.axis('off')

    plt.show()
