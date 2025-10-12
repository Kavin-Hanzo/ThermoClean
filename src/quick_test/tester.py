"""
tester.py

Standalone tester for the degrading pipeline (single-image).
Requires `Datacreator.py` (must be in same folder) which provides:
    - test_single_image(image_path, out_path=None, show=False, **kwargs)
    - degrade_image_with_hotspot(...)

This script:
 - loads an input image
 - calls the degrade routine to create z (degraded), b (bias), f (original intensity)
 - saves the degraded image (and bias as vis), and displays them side-by-side

Usage:
    python tester.py --input path/to/img.jpg --out path/to/out.png --show --seed 42

If Datacreator.py is missing, the script prints a helpful message.
"""

import argparse
from pathlib import Path
import os
import sys


# Try to import functions from Datacreator.py
try:
    from Datacreator import test_single_image, degrade_image_with_hotspot
except Exception as e:
    test_single_image = None
    degrade_image_with_hotspot = None
    _import_err = e

# visualization tools
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, util

def show_and_save_results(z, b, f, out_prefix):
    """
    Show (inline) and save z (degraded) + b (bias visual) + f (original intensity).
    Saves:
      - {out_prefix}_degraded.png
      - {out_prefix}_bias.png
      - {out_prefix}_original.png
    """
    # prepare outputs as uint8
    def to_uint8(img):
        img_clipped = np.clip(img, 0.0, 1.0)
        return (img_clipped * 255.0).astype(np.uint8)

    z_u8 = to_uint8(z) if isinstance(z, np.ndarray) else z
    b_vis = b - b.min()
    if b_vis.max() > 0:
        b_vis = b_vis / b_vis.max()
    b_u8 = to_uint8(b_vis)
    f_u8 = to_uint8(f) if isinstance(f, np.ndarray) else f

    # Save files
    io.imsave(out_prefix + "_degraded.png", z_u8)
    # io.imsave(out_prefix + "_bias.png", b_u8)
    io.imsave(out_prefix + "_original.png", f_u8)

    # Display side-by-side
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Original (f)"); plt.axis('off')
    if f.ndim == 2: plt.imshow(f_u8, cmap='gray')
    else: plt.imshow(f_u8)
    plt.subplot(1,3,2); plt.title("Bias (b)"); plt.axis('off'); plt.imshow(b_u8, cmap='jet')
    plt.subplot(1,3,3); plt.title("Degraded (z)"); plt.axis('off')
    if z.ndim == 2: plt.imshow(z_u8, cmap='gray')
    else: plt.imshow(z_u8)
    plt.tight_layout()
    plt.show()

def run_test(image_path, out_prefix=None, show=True, seed=None, **degrade_kwargs):
    """
    High-level runner:
      - calls test_single_image from synthesize_degraded.py if available
      - otherwise tries to call degrade_image_with_hotspot directly
    Returns: (z, b, f, used_quadrant, center_info)
    """
    if test_single_image is None and degrade_image_with_hotspot is None:
        print("ERROR: synthesize_degraded.py not found or failed to import.")
        print("Make sure synthesize_degraded.py is in the same folder and provides test_single_image()")
        print("Import error was:", _import_err)
        sys.exit(1)

    if out_prefix is None:
        p = Path(image_path)
        out_prefix = str(p.with_name(p.stem + "_degraded"))

    # Prefer the high-level tester if available
    if test_single_image is not None:
        z, b, f, used_quadrant, center_info = test_single_image(image_path, out_path=out_prefix + "_degraded.png", show=False, rng_seed=seed, **degrade_kwargs)
    else:
        # call lower-level generator
        z, b, f, used_quadrant, center_info = degrade_image_with_hotspot(image_path, rng_seed=seed, **degrade_kwargs)
        # save degraded image
        io.imsave(out_prefix + "_degraded.png", (np.clip(z,0,1) * 255).astype(np.uint8))

    # show and save images
    show_and_save_results(z, b, f, out_prefix)
    print(f"Saved files: {out_prefix}_degraded.png, {out_prefix}_bias.png, {out_prefix}_original.png")
    print(f"Hotspot quadrant: {used_quadrant}, center/radius/sigma: {center_info}")
    return z, b, f, used_quadrant, center_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick tester for the degrading process (single image). Shows and saves results with informative plots.")
    parser.add_argument("--input", "-i", required=True, help="Input image path (RGB or grayscale).")
    parser.add_argument("--out-prefix", "-o", default=None, help="Output filename prefix (default: <input>_degraded).")
    parser.add_argument("--show", action='store_true', help="Show images after generation.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--quadrant", type=str, default='random', help="Hotspot quadrant: top-left/top-right/bottom-left/bottom-right/center/random")
    parser.add_argument("--radius", type=float, default=0.85, help="Hotspot radius fraction of min(h,w)")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Hotspot amplitude (additive)")
    parser.add_argument("--bg_amp", type=float, default=0.02, help="Background amplitude")
    parser.add_argument("--gauss", type=float, default=0.01, help="Gaussian noise sigma")
    parser.add_argument("--poisson", type=float, default=0.0, help="Poisson scale (0 disabled)")
    args = parser.parse_args()

    degrade_kwargs = dict(
        quadrant=args.quadrant,
        radius_frac=args.radius,
        amplitude=args.amplitude,
        add_background=True,
        bg_amp=args.bg_amp,
        gauss_noise=args.gauss,
        poisson_scale=args.poisson,
    )

    print("\n--- Quick Degradation Test ---")
    print(f"Input image: {args.input}")
    print(f"Quadrant: {args.quadrant}, Radius: {args.radius}, Amplitude: {args.amplitude}, BG Amp: {args.bg_amp}, Gauss: {args.gauss}, Poisson: {args.poisson}, Seed: {args.seed}")
    z, b, f, used_quadrant, center_info = run_test(args.input, out_prefix=args.out_prefix, show=args.show, seed=args.seed, **degrade_kwargs)
    print(f"\nSummary: Degraded image created with hotspot in quadrant '{used_quadrant}'. Center/radius/sigma: {center_info}")
