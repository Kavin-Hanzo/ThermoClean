from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
import cv2
from numpy.polynomial import Chebyshev
from oa_helpers import normalize01


def apply_shi_2022(z: np.ndarray, cheb_degree=6, downsample=8) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    h, w = z.shape
    small = cv2.resize(z, (max(8, w//downsample), max(8, h//downsample)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    coeffs_rows = np.zeros((sh, cheb_degree+1))
    x_norm = np.linspace(-1, 1, sw)
    for i in range(sh):
        c = Chebyshev.fit(x_norm, small[i, :], cheb_degree).coef
        coeffs_rows[i, :len(c)] = c
    y_norm = np.linspace(-1, 1, sh)
    coeffs2 = np.zeros((cheb_degree+1, cheb_degree+1))
    for k in range(cheb_degree+1):
        c = Chebyshev.fit(y_norm, coeffs_rows[:, k], cheb_degree).coef
        coeffs2[k, :len(c)] = c
    X = np.linspace(-1,1,sw)
    Y = np.linspace(-1,1,sh)
    XX, YY = np.meshgrid(X, Y)
    surface = np.zeros_like(small)
    for i in range(cheb_degree+1):
        for j in range(cheb_degree+1):
            tx = np.polynomial.chebyshev.chebval(XX, [0]*i + [1])
            ty = np.polynomial.chebyshev.chebval(YY, [0]*j + [1])
            surface += coeffs2[i, j] * tx * ty
    estimated_bias = cv2.resize(surface, (w, h), interpolation=cv2.INTER_CUBIC)
    corrected = normalize01(z - estimated_bias)
    return corrected, estimated_bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    from skimage import io, img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_shi_2022(z)
    print("Shi 2022 quick test results:")
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
    parser = argparse.ArgumentParser(description='Quick test for Shi 2022 method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
