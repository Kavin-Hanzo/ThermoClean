from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
from scipy import ndimage
from oa_helpers import normalize01


def apply_liu_zhang(z: np.ndarray, p=0.8, lam=0.1, iterations=20) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    f = z.copy()
    for it in range(iterations):
        gx = np.roll(f, -1, axis=1) - f
        gy = np.roll(f, -1, axis=0) - f
        gmag = np.sqrt(gx*gx + gy*gy) + 1e-8
        w = gmag**(p - 2.0)
        div = (np.roll(w*gx, 1, axis=1) - w*gx) + (np.roll(w*gy, 1, axis=0) - w*gy)
        f = f + 0.2 * (z - f + lam * div)
    bias = ndimage.gaussian_filter(z - f, sigma=15)
    corrected = normalize01(f)
    return corrected, bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    from skimage import io, img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_liu_zhang(z)
    print("Liu-Zhang quick test results:")
    if original_path:
        metrics = compute_image_metrics(original_path, corrected, resize_ref=True)
        print(metrics)
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.title('Input'); plt.imshow(z, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title('Corrected'); plt.imshow(corrected, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title('Bias'); plt.imshow(normalize01(bias), cmap='magma'); plt.axis('off')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quick test for Liu-Zhang method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
