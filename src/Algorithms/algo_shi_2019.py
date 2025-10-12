from typing import Tuple
import numpy as np
from skimage.util import img_as_float32
from scipy import ndimage, fftpack
from oa_helpers import butterworth_filter, normalize01


def apply_shi_2019(z: np.ndarray, butter_cutoff=30, butter_order=2, sparse_thresh=0.02) -> Tuple[np.ndarray, np.ndarray]:
    z = img_as_float32(z)
    bias_butter = butterworth_filter(z, cutoff=butter_cutoff, order=butter_order)
    residual = z - bias_butter
    gx = ndimage.sobel(residual, axis=1)
    gy = ndimage.sobel(residual, axis=0)
    gx_thresh = np.sign(gx) * np.maximum(np.abs(gx) - sparse_thresh, 0)
    gy_thresh = np.sign(gy) * np.maximum(np.abs(gy) - sparse_thresh, 0)
    div = (gx_thresh - np.roll(gx_thresh, 1, axis=1)) + (gy_thresh - np.roll(gy_thresh, 1, axis=0))
    h, w = z.shape
    ky = fftpack.fftfreq(h).reshape(-1,1) * 2*np.pi
    kx = fftpack.fftfreq(w).reshape(1,-1) * 2*np.pi
    denom = (2 - 2*np.cos(kx)) + (2 - 2*np.cos(ky))
    denom[0,0] = 1.0
    div_f = fftpack.fft2(div)
    recon = np.real(fftpack.ifft2(div_f / (denom + 1e-12)))
    reflectance = residual - recon
    estimated_bias = bias_butter + ndimage.gaussian_filter(recon, sigma=3)
    corrected = normalize01(z - estimated_bias)
    return corrected, estimated_bias.astype(np.float32)


def quick_test(degraded_path: str, original_path: str = None):
    from skimage import io, img_as_float
    from Evaluator import compute_image_metrics
    import matplotlib.pyplot as plt

    z = img_as_float(io.imread(degraded_path, as_gray=True)).astype(np.float32)
    corrected, bias = apply_shi_2019(z)
    print("Shi 2019 quick test results:")
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
    parser = argparse.ArgumentParser(description='Quick test for Shi 2019 method')
    parser.add_argument('degraded')
    parser.add_argument('--orig', default=None)
    args = parser.parse_args()
    quick_test(args.degraded, args.orig)
