from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
from scipy import ndimage
from oa_helpers import fit_2d_poly, normalize01


def apply_cao_tisse(z: np.ndarray, poly_order=3, gaussian_sigma=15) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    bias_lp = ndimage.gaussian_filter(z, sigma=gaussian_sigma)
    poly_fit = fit_2d_poly(bias_lp, order=poly_order)
    estimated_bias = poly_fit
    corrected = z - estimated_bias
    corrected = normalize01(corrected)
    return corrected, estimated_bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    """Quick test CLI: run on single degraded image and optionally evaluate against original."""
    from skimage import io
    from skimage import img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_cao_tisse(z)
    print("Cao-Tisse quick test results:")
    if original_path:
        test_arr = corrected
        metrics = compute_image_metrics(original_path, test_arr, resize_ref=True)
        print(metrics)
    # show
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.title('Input'); plt.imshow(z, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title('Corrected'); plt.imshow(corrected, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title('Bias'); plt.imshow(normalize01(bias), cmap='magma'); plt.axis('off')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quick test for Cao & Tisse method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
