"""
Generate synthetic degraded images following z = f + b + n where:
 - b contains a smooth circular hotspot ("white surface") placed in one of four quadrants
   (top-left, top-right, bottom-left, bottom-right) or randomly among them.
 - n is sensor noise (Gaussian + optional Poisson).
 - f is the input image intensity (grayscale or RGB - same hotspot applied to all channels).

Includes:
 - test_single_image(image_path, out_path=None, **kwargs)  # tester for one image
 - degrade_train_folder(train_dir, out_dir, variants_per_image=1, **kwargs)  # batch folder

Dependencies:
    numpy, opencv-python-headless (or opencv-python), scikit-image, pillow, tqdm

Install:
    pip install numpy opencv-python-headless scikit-image pillow tqdm
"""

from pathlib import Path
import numpy as np
import math
import cv2
from skimage import io, color, util
from PIL import Image
from tqdm import tqdm
import os
import random

# ---------------------------
# Helper: create circular Gaussian hotspot bias
# ---------------------------
def generate_circular_hotspot(shape,
                              quadrant='random',
                              radius_frac=0.25,
                              sigma_frac=None,
                              amplitude=0.18,
                              edge_softness=1.0,
                              margin_frac=0.06,
                              rng=None):
    """
    Generate a circular Gaussian hotspot bias with center placed in one of four quadrants.

    Params:
      - shape: (h, w)
      - quadrant: 'top-left','top-right','bottom-left','bottom-right','center','random'
          If 'random', a quadrant is chosen uniformly among the 4 corners.
      - radius_frac: radius of the hotspot as fraction of min(h,w) (peak radius)
      - sigma_frac: Gaussian sigma as fraction of radius (if None uses radius*0.5)
      - amplitude: peak additive intensity (positive -> bright/white hotspot)
      - edge_softness: multiplier for sigma (higher -> smoother edges)
      - margin_frac: how far the hotspot center is from the edges as fraction of width/height
      - rng: RandomState or seed (optional)

    Returns: 2D numpy float array bias (shape h x w), zero-mean is NOT forced (we keep positive blob)
    """
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(int(rng))

    h, w = shape
    min_hw = min(h, w)
    radius = max(2.0, radius_frac * min_hw)  # radius in pixels
    sigma = (sigma_frac * radius) if sigma_frac is not None else (radius * 0.5 * edge_softness)

    # Determine center coordinates by quadrant
    cx_norm = 0.5
    cy_norm = 0.5
    margin_x = margin_frac
    margin_y = margin_frac

    q = quadrant
    if quadrant == 'random':
        q = random.choice(['top-left', 'top-right','center', 'bottom-left', 'bottom-right'])
    if q == 'top-left':
        cx_norm = margin_x + radius_frac
        cy_norm = margin_y + radius_frac
    elif q == 'top-right':
        cx_norm = 1.0 - (margin_x + radius_frac)
        cy_norm = margin_y + radius_frac
    elif q == 'bottom-left':
        cx_norm = margin_x + radius_frac
        cy_norm = 1.0 - (margin_y + radius_frac)
    elif q == 'bottom-right':
        cx_norm = 1.0 - (margin_x + radius_frac)
        cy_norm = 1.0 - (margin_y + radius_frac)
    elif q == 'center':
        cx_norm = 0.5
        cy_norm = 0.5
    else:
        # allow user-specified tuple center via quadrant parameter as (cx_norm, cy_norm)
        if isinstance(q, (tuple, list)) and len(q) == 2:
            cx_norm, cy_norm = float(q[0]), float(q[1])

    cx = cx_norm * (w - 1)
    cy = cy_norm * (h - 1)

    # Create meshgrid and Gaussian blob
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    blob = np.exp(-0.5 * dist2 / (sigma ** 2))

    # Normalize blob to peak = 1 and scale by amplitude
    blob = blob / (np.max(blob) + 1e-12)
    bias = amplitude * blob

    return bias, q, (cx, cy, radius, sigma)

# ---------------------------
# Optional: low-frequency background to combine with hotspot
# ---------------------------
def generate_smooth_background(shape, amp=0.03, blur_sigma_frac=0.2, rng=None):
    """
    Generate a low-frequency smooth background (blurred noise) to combine with hotspot.
    - amp: amplitude scale of the background
    - blur_sigma_frac: blur sigma relative to min(h,w)
    """
    if rng is None:
        rng = np.random.RandomState()
    h, w = shape
    min_hw = min(h, w)
    blur_sigma = max(1.0, blur_sigma_frac * min_hw)

    noise = rng.normal(size=(h, w)).astype(np.float32)
    # normalize then blur using OpenCV GaussianBlur
    nmin, nmax = noise.min(), noise.max()
    if abs(nmax - nmin) < 1e-9:
        norm = np.zeros_like(noise)
    else:
        norm = (noise - nmin) / (nmax - nmin)
    img8 = (norm * 255.0).astype(np.uint8)
    k = int(max(3, 2 * int(blur_sigma) + 1))
    bg = cv2.GaussianBlur(img8, (k, k), blur_sigma).astype(np.float32) / 255.0
    bg = (bg - 0.5) * 2.0  # center about 0
    return bg * amp

# ---------------------------
# Sensor noise
# ---------------------------
def add_sensor_noise(img, gauss_sigma=0.01, poisson_scale=0.0, rng=None):
    """
    Add Gaussian (additive) + optional Poisson shot noise.
    img expected float in [0,1].
    """
    if rng is None:
        rng = np.random.RandomState()
    out = img.astype(np.float32).copy()
    if poisson_scale and poisson_scale > 0:
        scaled = np.clip(out * poisson_scale, 0.0, None)
        lam = (scaled * 255).flatten()
        sampled = rng.poisson(lam).astype(np.float32) / 255.0
        out = sampled.reshape(out.shape)
    if gauss_sigma and gauss_sigma > 0:
        out = out + rng.normal(scale=gauss_sigma, size=out.shape)
    out = np.clip(out, 0.0, 1.0)
    return out

# ---------------------------
# Main degrade function (with quadrant hotspot)
# ---------------------------
def degrade_image_with_hotspot(image,
                               output_mode='grayscale',
                               quadrant='random',    # 'top-left','top-right','bottom-left','bottom-right','center','random'
                               radius_frac=0.85,     # YAML default
                               amplitude=0.5,        # YAML default
                               sigma_frac=None,      # sigma relative to radius; if None sigma=0.5*radius
                               add_background=True,  # add smooth low-frequency background
                               bg_amp=0.02,          # YAML default
                               gauss_noise=0.01,     # YAML default
                               poisson_scale=0.0,    # YAML default
                               bias_mean_subtract=False,
                               rng_seed=42):         # YAML default
    """
    Apply a circular hotspot bias and noise to an image.

    Returns:
      z (degraded image float32 in [0,1]),
      b (2D bias map float32),
      f (input intensity float32)
    """
    # RNG
    if rng_seed is None:
        rng = np.random.RandomState()
    elif isinstance(rng_seed, (int, np.integer)):
        rng = np.random.RandomState(int(rng_seed))
    else:
        rng = rng_seed

    # Load and convert to float
    if isinstance(image, (str, Path)):
        img = io.imread(str(image))
    else:
        img = image
    if img.dtype not in (np.float32, np.float64):
        imgf = util.img_as_float(img).astype(np.float32)
    else:
        imgf = np.clip(img.astype(np.float32), 0.0, 1.0)

    # Get f (grayscale intensity or RGB)
    if output_mode == 'grayscale':
        if imgf.ndim == 3 and imgf.shape[2] >= 3:
            f = color.rgb2gray(imgf).astype(np.float32)
        elif imgf.ndim == 2:
            f = imgf.astype(np.float32)
        else:
            f = np.mean(imgf[..., :3], axis=2).astype(np.float32)
    else:
        # RGB output; ensure 3 channels
        if imgf.ndim == 2:
            f = np.stack([imgf, imgf, imgf], axis=-1).astype(np.float32)
        else:
            if imgf.shape[2] > 3:
                f = imgf[..., :3].astype(np.float32)
            else:
                f = imgf.astype(np.float32)

    h, w = f.shape[:2]
    min_hw = min(h, w)
    radius = max(2.0, radius_frac * min_hw)
    if sigma_frac is None:
        sigma = radius * 0.5
    else:
        sigma = radius * sigma_frac

    # Create hotspot bias (2D)
    b_hotspot, used_quadrant, center_info = generate_circular_hotspot(
        (h, w),
        quadrant=quadrant,
        radius_frac=radius_frac,
        sigma_frac=(sigma / radius) if radius > 0 else 0.5,
        amplitude=amplitude,
        rng=rng
    )

    # Optional smooth background
    if add_background:
        b_bg = generate_smooth_background((h, w), amp=bg_amp, rng=rng)
    else:
        b_bg = np.zeros((h, w), dtype=np.float32)

    # Combine bias components
    b = b_hotspot + b_bg

    # Optionally subtract mean (paper sometimes treats bias as smooth offset; keep both options)
    if bias_mean_subtract:
        b = b - np.mean(b)

    # Apply bias (same to each channel if RGB)
    if output_mode == 'grayscale':
        z_pre = f + b
    else:
        b3 = np.stack([b, b, b], axis=-1)
        z_pre = f + b3

    # Add sensor noise
    z = add_sensor_noise(np.clip(z_pre, 0.0, 1.0), gauss_sigma=gauss_noise, poisson_scale=poisson_scale, rng=rng)

    return z.astype(np.float32), b.astype(np.float32), f.astype(np.float32), used_quadrant, center_info

# ---------------------------
# Tester function for single image
# ---------------------------
def test_single_image(input_path, out_path=None, show=False, **kwargs):
    """
    Degrade a single image and save it.
    Returns (z, b, f, quadrant, center_info)
    """
    z, b, f, used_quadrant, center_info = degrade_image_with_hotspot(input_path, **kwargs)
    if out_path is None and isinstance(input_path, (str, Path)):
        p = Path(input_path)
        out_path = p.with_name(p.stem + "_degraded.png")
    if out_path is not None:
        out_path = Path(out_path)
        # save as 8-bit PNG
        if z.ndim == 2:
            io.imsave(str(out_path), (np.clip(z, 0, 1) * 255).astype(np.uint8))
        else:
            io.imsave(str(out_path), (np.clip(z, 0, 1) * 255).astype(np.uint8))
    if show and out_path is not None:
        try:
            Image.open(str(out_path)).show()
        except Exception:
            pass
    return z, b, f, used_quadrant, center_info

# ---------------------------
# Batch degrade preserving folder structure (optionally create quadrant variants)
# ---------------------------
def degrade_train_folder_with_quadrants(train_dir, out_dir, variants_per_image=1, quadrants=('top-left','top-right','bottom-left','bottom-right','center'), overwrite=False, **degrade_kwargs):
    """
    Walk train_dir and create degraded variants for each image.
     - variants_per_image: if equals len(quadrants), create exactly one per quadrant (recommended)
       otherwise, variants are created with quadrant chosen at random per variant.
     - degrade_kwargs forwarded to degrade_image_with_hotspot
    """
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"{train_dir} does not exist")
    for root, dirs, files in os.walk(train_dir):
        root_path = Path(root)
        rel = root_path.relative_to(train_dir)
        out_sub = out_dir.joinpath(rel)
        out_sub.mkdir(parents=True, exist_ok=True)
        image_files = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
        for fname in tqdm(image_files, desc=f"Processing {rel}" if str(rel) != '.' else "Processing root"):
            inp = root_path.joinpath(fname)
            try:
                img = io.imread(str(inp))
            except Exception as e:
                print(f"Skipping {inp}: read error {e}")
                continue
            # Decide quadrants for this image's variants
            if variants_per_image <= 0:
                continue
            if variants_per_image >= len(quadrants):
                chosen_quads = quadrants[:variants_per_image]
            else:
                # sample randomly
                chosen_quads = [random.choice(quadrants) for _ in range(variants_per_image)]
            for v_idx, q in enumerate(chosen_quads):
                out_name = f"{Path(fname).stem}_deg_q{q.replace('-', '')}_v{v_idx}{Path(fname).suffix}"
                out_path = out_sub.joinpath(out_name)
                if out_path.exists() and not overwrite:
                    continue
                z, b, f, used_quad, center_info = degrade_image_with_hotspot(img, quadrant=q, **degrade_kwargs)
                io.imsave(str(out_path), (np.clip(z, 0, 1) * 255).astype(np.uint8))

# ---------------------------
# CLI
# ---------------------------

