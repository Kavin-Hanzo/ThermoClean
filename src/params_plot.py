#!/usr/bin/env python3
"""
Gamma parameter analysis for BFBSF and BFBSF+ algorithms.
Evaluates PSNR and SSIM over a range of gamma values,
producing a plot similar to Figure 10 (gamma sensitivity only).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from Algo2 import bfbsf, bfbsf_plus 

# -------------------- Defaults --------------------
DEFAULT_GAMMA_RANGE = np.linspace(0.0, 0.05, 10)  # from 0.0 to 0.6 as in paper

# -------------------- Helpers --------------------
def normalize_image(img):
    img = img_as_float(img)
    return np.clip(img, 0.0, 1.0)

def compute_metrics(gt, rec):
    """Compute PSNR and SSIM between ground truth and reconstruction."""
    psnr_val = psnr(gt, rec, data_range=1.0)
    ssim_val = ssim(gt, rec, data_range=1.0, channel_axis=None)
    return float(psnr_val), float(ssim_val)

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze gamma parameter in BFBSF/BFBSF+")
    parser.add_argument("--degraded", required=True, help="Path to degraded image")
    parser.add_argument("--original", required=True, help="Path to original (clean) image")
    parser.add_argument("--outdir", default="gamma_analysis", help="Output folder")
    parser.add_argument("--n_iter", type=int, default=9, help="Number of iterations")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    degraded = normalize_image(imread(args.degraded, as_gray=True))
    original = normalize_image(imread(args.original, as_gray=True))

    gammas = DEFAULT_GAMMA_RANGE
    results = {"gamma": [], "psnr_bfbsf": [], "ssim_bfbsf": [],
               "psnr_bfbsf_plus": [], "ssim_bfbsf_plus": []}

    print(f"[+] Starting gamma sweep over {len(gammas)} values...")

    for g in gammas:
        print(f"  -> gamma = {g:.3f}")
        # BFBSF
        rec_bfbsf, _ = bfbsf(degraded, n_iter=args.n_iter, gamma=g,
                             verbose=False)
        psnr_b, ssim_b = compute_metrics(original, rec_bfbsf)

        # BFBSF+
        rec_bfbsf_plus, _ = bfbsf_plus(degraded, n_iter=args.n_iter, gamma=g,
                                       verbose=False)
        psnr_bp, ssim_bp = compute_metrics(original, rec_bfbsf_plus)

        # record
        results["gamma"].append(g)
        results["psnr_bfbsf"].append(psnr_b)
        results["ssim_bfbsf"].append(ssim_b)
        results["psnr_bfbsf_plus"].append(psnr_bp)
        results["ssim_bfbsf_plus"].append(ssim_bp)

        print(f"     BFBSF: PSNR={psnr_b:.3f}, SSIM={ssim_b:.3f} | BFBSF+: PSNR={psnr_bp:.3f}, SSIM={ssim_bp:.3f}")

    # -------------------- Plot --------------------
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(gammas, results["psnr_bfbsf"], 'o-', label="BFBSF")
    plt.plot(gammas, results["psnr_bfbsf_plus"], 's-', label="BFBSF+")
    plt.title("PSNR vs Gamma")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(gammas, results["ssim_bfbsf"], 'o-', label="BFBSF")
    plt.plot(gammas, results["ssim_bfbsf_plus"], 's-', label="BFBSF+")
    plt.title("SSIM vs Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("SSIM")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    outpath = os.path.join(args.outdir, "gamma_analysis.png")
    plt.savefig(outpath, dpi=150)
    plt.show()

    print(f"[+] Analysis complete. Saved plot: {outpath}")

if __name__ == "__main__":
    main()
