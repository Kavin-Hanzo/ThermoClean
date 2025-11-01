#!/usr/bin/env python3
"""
Adaptive Aero-Optical Restoration Controller

Simulates dynamic switching between BFBSF and BFBSF+ algorithms based on
altitude and speed thresholds during aerial imaging conditions.
"""
from Algo2 import bfbsf,bfbsf_plus
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ---------------- Helper Functions ----------------
def normalize_image(img):
    img = img_as_float(img)
    return np.clip(img, 0.0, 1.0)

def compute_metrics(original, restored):
    psnr_val = psnr(original, restored, data_range=1.0)
    ssim_val = ssim(original, restored, data_range=1.0, channel_axis=None)
    return float(psnr_val), float(ssim_val)

# ---------------- Wrapper Class ----------------
class AdaptiveBFBSFController:
    """
    Wrapper that switches between BFBSF and BFBSF+ depending on speed and altitude.
    """
    def __init__(self, altitude_thresh=4350, speed_thresh=120,
                 eta=0.2, alpha=0.6, beta=0.9, gamma=0.045, n_iter=9):
        self.altitude_thresh = altitude_thresh
        self.speed_thresh = speed_thresh
        self.params = dict(eta=eta, alpha=alpha, beta=beta, gamma=gamma, n_iter=n_iter)

    def process(self, img, speed_arr, altitude_arr, mode="altitude"):
        """
        Process image dynamically based on altitude or speed mode.
        - mode='altitude': start with bfbsf, switch to plus when altitude >= threshold
        - mode='speed': start with plus, switch to bfbsf when speed >= threshold
        """
        assert len(speed_arr) == len(altitude_arr), "speed and altitude arrays must match in length"
        outputs = []
        alg_used = []
        book = ""
        for i, (spd, alt) in enumerate(zip(speed_arr, altitude_arr)):
            if mode == "altitude":
                if alt < self.altitude_thresh:
                    alg = "BFBSF"
                    out, _ = bfbsf(img, **self.params)
                else:
                    alg = "BFBSF+"
                    out, _ = bfbsf_plus(img, **self.params)
            else:  # speed mode
                if spd < self.speed_thresh:
                    alg = "BFBSF+"
                    out, _ = bfbsf_plus(img, **self.params)
                else:
                    alg = "BFBSF"
                    out, _ = bfbsf(img, **self.params)

            outputs.append(out)
            alg_used.append(alg)
            string = f"[Frame {i+1}] Speed={spd:.1f} Altitude={alt:.1f} -> Using {alg}"
            print(string)
            book += str(string+"\n")
            filname = "adaptive_output/"+mode+".txt"

        with open(filname,"w") as f:
            f.write(book)


        return outputs, alg_used

# ---------------- Visualization ----------------
def plot_algorithm_switch(speed_arr, altitude_arr, alg_used, outdir="adaptive_output"):
    """
    Scatter plot (Speed vs Altitude) with vertical lines marking
    where algorithm switches between BFBSF and BFBSF+.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    x = np.arange(len(speed_arr))
    colors = ["tab:blue" if a == "BFBSF" else "tab:orange" for a in alg_used]

    # Detect switching points
    switch_indices = [i for i in range(1, len(alg_used)) if alg_used[i] != alg_used[i - 1]]

    plt.figure(figsize=(10, 6))
    plt.scatter(speed_arr, altitude_arr, c=colors, s=100, edgecolors='k', alpha=0.8)

    # Add vertical lines at switch points (x = speed at switch)
    for idx in switch_indices:
        plt.axhline(y=altitude_arr[idx], color='red', linestyle='--', linewidth=1.5,
                    label="Switch Point" if idx == switch_indices[0] else None)
        # Label at switch
        plt.text(speed_arr[idx] + 10, max(altitude_arr) * 0.95,
                 f"→ {alg_used[idx]}", color='red', fontsize=9)

    # Add text labels for each point
    for i, label in enumerate(alg_used):
        plt.text(speed_arr[i] + 5, altitude_arr[i], label, fontsize=8, color='black')

    plt.title("Algorithm Switching based on Speed and Altitude")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Altitude (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/algorithm_switch.png", dpi=150)
    plt.show()
    print(f"[+] Saved plot to {outdir}/algorithm_switch.png")


# ---------------- Main Script ----------------
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive BFBSF Controller")
    parser.add_argument("--degraded", required=True, help="Path to degraded image")
    parser.add_argument("--original", help="Path to original image (optional)")
    parser.add_argument("--outdir", default="adaptive_output", help="Output directory")
    parser.add_argument("--mode", choices=["speed", "altitude"], default="altitude")
    parser.add_argument("--n_points", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    degraded = normalize_image(imread(args.degraded, as_gray=True))

    if args.original:
        original = normalize_image(imread(args.original, as_gray=True))
    else:
        original = None

    # Simulated linear increase of speed and altitude
    speed_arr = np.linspace(50, 150, args.n_points)
    altitude_arr = np.linspace(3500, 5000, args.n_points)

    controller = AdaptiveBFBSFController(
        altitude_thresh=4350,
        speed_thresh=120,
        n_iter=9
    )

    outputs, alg_used = controller.process(degraded, speed_arr, altitude_arr, mode=args.mode)

    # Compute metrics (optional)
    if original is not None:
        psnrs, ssims = [], []
        for rec in outputs:
            p, s = compute_metrics(original, rec)
            psnrs.append(p)
            ssims.append(s)
        plt.figure(figsize=(8,5))
        plt.plot(psnrs, label="PSNR (dB)")
        plt.plot(ssims, label="SSIM")
        plt.xlabel("Frame index")
        plt.legend()
        plt.title("Performance over frames")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(f"{args.outdir}/performance_metrics.png", dpi=150)
        plt.show()

    # Plot switching behavior
    plot_algorithm_switch(speed_arr, altitude_arr, alg_used, args.outdir)
