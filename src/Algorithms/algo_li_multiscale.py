from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
from skimage.transform import pyramid_gaussian
import cv2
from oa_helpers import fit_2d_poly, normalize01


def apply_li(z: np.ndarray, levels=3, poly_order=2) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    pyramid = tuple(pyramid_gaussian(z, max_layer=levels-1, channel_axis=None))
    bias_accum = np.zeros_like(z)
    for i, level_img in enumerate(reversed(pyramid)):
        fitted = fit_2d_poly(level_img, order=poly_order)
        fitted_up = cv2.resize(fitted, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_CUBIC)
        weight = 1.0 / (1 + i)
        bias_accum += weight * fitted_up
    estimated_bias = bias_accum / (bias_accum.max() + 1e-12) * z.max()
    corrected = normalize01(z - estimated_bias)
    return corrected, estimated_bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    from skimage import io, img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_li(z)
    print("Li multiscale quick test results:")
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
    parser = argparse.ArgumentParser(description='Quick test for Li multiscale method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
