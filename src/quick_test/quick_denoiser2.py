"""
quick_denoiser2.py

Visualize BFBSF iterations and compute image-quality metrics (PSNR, SSIM, MSE, MAE, CV)
between the final denoised image and the original RGB image (original is converted to grayscale).

Usage example:
    python quick_denoiser2.py --input degraded.png --original original_rgb.png --algorithm plus --n_iter 6 --gamma 0.3 --save_fig out.png

Note:
 - Original image path must point to the ground-truth RGB (or grayscale) image.
 - If original and denoised sizes differ, original is resized to match denoised before metric computation.
"""

import argparse
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.color import rgb2gray

# import algorithms implemented in bfbsf_fixed.py
try:
    from src.Algo2 import bfbsf, bfbsf_plus
except Exception as e:
    raise ImportError("Could not import bfbsf_fixed.py. Make sure bfbsf_fixed.py is in same folder or on PYTHONPATH.") from e

# import evaluation helper
try:
    from src.Evaluator import compute_image_metrics
except Exception:
    # fallback: metric functions inline minimal (but recommend having eval_metrics.py present)
    def compute_image_metrics(ref, test, resize_ref=True):
        raise ImportError("eval_metrics.py not found. Please place eval_metrics.py in same folder.")

def compute_iteration_sequence(z: np.ndarray, biases: list, gamma: float) -> list:
    seq = []
    f = z.copy().astype(np.float32)
    seq.append(f.copy())
    for b in biases:
        f = f - gamma * b
        f = np.clip(f, 0.0, 1.0)
        seq.append(f.copy())
    return seq

def plot_iteration_sequence(z: np.ndarray, seq: list, metrics: dict = None, save_path: Path = None, show: bool = True, cmap='gray', title_prefix='', orig_img: np.ndarray = None):
    n_plots = len(seq)
    n_display = n_plots + (1 if orig_img is not None else 0)
    cols = min(6, n_display)
    rows = math.ceil(n_display / cols)
    fig_w = cols * 3
    fig_h = rows * 2.6
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for i, img in enumerate(seq):
        ax = axes[i]
        ax.imshow(img, cmap=cmap)
        if i == 0:
            ax.set_title("Degraded (input)")
        else:
            ax.set_title(f"Iter {i}")
        ax.axis('off')

    idx = len(seq)
    if orig_img is not None and idx < len(axes):
        axo = axes[idx]
        axo.imshow(orig_img, cmap=cmap)
        axo.set_title("Original")
        axo.axis('off')
    elif idx < len(axes):
        axf = axes[idx]
        axf.imshow(seq[-1], cmap=cmap)
        axf.set_title("Final")
        axf.axis('off')

    # add metrics text in the figure suptitle or subtitle
    subtitle = title_prefix
    if metrics is not None:
        subtitle += ("  | PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  MSE={mse:.6f}  MAE={mae:.6f}  cv={cv:.4f}".format(
            psnr=metrics.get('psnr', float('nan')),
            ssim=metrics.get('ssim', float('nan')),
            mse=metrics.get('mse', float('nan')),
            mae=metrics.get('mae', float('nan')),
            cv=metrics.get('cv', float('nan'))
        ))
    plt.suptitle(subtitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        try:
            fig.savefig(str(save_path), dpi=200, bbox_inches='tight')
            print(f"Saved iteration figure to: {save_path}")
        except Exception as e:
            print("Warning: could not save figure:", e)
    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Quick denoising tester: visualize BFBSF iterations, compute metrics, and plot results.")
    parser.add_argument("--input", "-i", required=True, help="Path to degraded input image (grayscale or RGB).")
    parser.add_argument("--original", "-r", required=False, help="Path to ground-truth original RGB image (optional, for metrics).")
    parser.add_argument("--algorithm", "-a", choices=['fast','plus'], default='plus', help="Algorithm: 'fast' (bfbsf) or 'plus' (bfbsf_plus).")
    parser.add_argument("--n_iter", "-n", type=int, default=9, help="Number of iterations to run / visualize.")
    parser.add_argument("--gamma", type=float, default=0.045, help="Gamma (subtraction factor) used in algorithm.")
    parser.add_argument("--save_fig", type=str, default=None, help="If provided, save the iteration figure to this path.")
    parser.add_argument("--no_show", action='store_true', help="Do not show interactive figure.")
    parser.add_argument("--normalize_input", action='store_true', help="Normalize input image to [0,1] before processing.")
    args = parser.parse_args()

    print("\n--- Quick Denoising Test ---")
    print(f"Input image: {args.input}")
    print(f"Algorithm: {args.algorithm}, n_iter: {args.n_iter}, gamma: {args.gamma}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image {input_path} not found")

    # load degraded image as grayscale float
    z = img_as_float(io.imread(str(input_path), as_gray=True)).astype(np.float32)
    if args.normalize_input:
        z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    z = np.clip(z, 0.0, 1.0)

    print(f"Running algorithm '{args.algorithm}' for {args.n_iter} iterations (gamma={args.gamma}) ...")
    if args.algorithm == 'fast':
        f_final, biases = bfbsf(z, n_iter=args.n_iter, gamma=args.gamma, verbose=False)
    else:
        f_final, biases = bfbsf_plus(z, n_iter=args.n_iter, gamma=args.gamma, verbose=False)

    seq = compute_iteration_sequence(z, biases, args.gamma)
    # pad if fewer iterations than requested
    if len(seq) < args.n_iter + 1:
        last = seq[-1]
        while len(seq) < args.n_iter + 1:
            seq.append(last.copy())

    metrics = None
    orig_img = None
    orig_path = None
    if args.original:
        orig_path = Path(args.original)
    else:
        orig_path = Path("output_original.png")
    if orig_path.exists():
        try:
            orig_img = img_as_float(io.imread(str(orig_path), as_gray=True)).astype(np.float32)
        except Exception as e:
            print(f"Warning: could not load original image {orig_path}: {e}")
    if args.original and orig_img is not None:
        try:
            metrics = compute_image_metrics(str(orig_path), seq[-1], resize_ref=True)
            print("\nEvaluation Metrics (final vs original):")
            print("  PSNR (dB): {:.4f}".format(metrics['psnr']))
            print("  SSIM     : {:.6f}".format(metrics['ssim']))
            print("  MSE      : {:.8f}".format(metrics['mse']))
            print("  MAE      : {:.8f}".format(metrics['mae']))
            print("  CV(ref)  : {:.6f}".format(metrics['cv']))
        except Exception as e:
            print("Warning: failed to compute metrics:", e)

    title_prefix = f"{input_path.name}  | alg={args.algorithm}  | n_iter={args.n_iter}  | gamma={args.gamma}"
    save_path = Path(args.save_fig) if args.save_fig is not None else None
    plot_iteration_sequence(z, seq, metrics=metrics, save_path=save_path, show=(not args.no_show), cmap='gray', title_prefix=title_prefix, orig_img=orig_img)
    print("\nSummary: Denoising complete. Plots and metrics displayed above.")

if __name__ == "__main__":
    main()
