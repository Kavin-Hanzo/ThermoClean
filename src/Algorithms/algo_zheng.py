from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
from skimage.restoration import denoise_tv_chambolle
from oa_helpers import gaussian_blur, normalize01


def apply_zheng(z: np.ndarray, tv_weight=0.1, bf_sigma=10) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    reflectance = denoise_tv_chambolle(z, weight=tv_weight, channel_axis=None)
    bias = gaussian_blur(z - reflectance, sigma=bf_sigma)
    corrected = z - bias
    corrected = normalize01(corrected)
    return corrected, bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    from skimage import io, img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_zheng(z)
    print("Zheng quick test results:")
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
    parser = argparse.ArgumentParser(description='Quick test for Zheng method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
