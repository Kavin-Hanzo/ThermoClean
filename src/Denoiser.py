"""
Denoiser.py

Batch-denoise images in a folder-tree (e.g. 'degraded_test/<category>/*.png') using BFBSF algorithms.

Requires: Algo2.py in same folder or importable; that module must provide:
    - bfbsf(z, n_iter=..., ...) -> (f_final, biases)
    - bfbsf_plus(z, n_iter=..., ...) -> (f_final, biases)

For each input image we save:
 - <name>_corrected.png        <-- corrected image
 - <name>_estimated_bias.png   <-- visualization of estimated bias removed
 - optionally <name>_original.png

CLI options:
 - --num_subfolders   : number of category subfolders to process (default: all)
 - --num_images       : number of images per subfolder to process (default: all)
 - --output_dir       : root directory to save processed images; subfolders preserved
 - --algorithm        : 'fast' (bfbsf) or 'plus' (bfbsf_plus)
 - --random_subfolders / --random_images : optionally choose random subset instead of sorted first-N
 - --seed             : RNG seed for reproducible sampling

Author: Sivakavin
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from skimage import io, img_as_float
from tqdm import tqdm
import shutil
import importlib
from pathlib import Path as _Path

try:
    from src.Algo2 import bfbsf, bfbsf_plus
except Exception as e:
    raise ImportError("Could not import bfbsf_fixed.py. Make sure it is in the same folder or on PYTHONPATH.") from e

# Discover additional algorithm modules in a subfolder 'Algorithms'
OTHER_ALGOS = {}
alg_dir = _Path(os.path.dirname(__file__)) / 'Algorithms'
if alg_dir.exists() and alg_dir.is_dir():
    # add to sys.path so modules can be imported by filename
    if str(alg_dir) not in sys.path:
        sys.path.insert(0, str(alg_dir))
    for p in alg_dir.iterdir():
        if p.is_file() and p.suffix == '.py' and p.name.startswith('algo_'):
            mod_name = p.stem  # algo_cao_tisse
            try:
                mod = importlib.import_module(mod_name)
                # find an apply_ function in module
                apply_fn = None
                for attr in dir(mod):
                    if attr.startswith('apply_') and callable(getattr(mod, attr)):
                        apply_fn = getattr(mod, attr)
                        break
                if apply_fn is not None:
                    key = mod_name[len('algo_'):]
                    OTHER_ALGOS[key.lower()] = apply_fn
            except Exception as e:
                print(f"Warning: failed to import algorithm module {mod_name}: {e}")
    if OTHER_ALGOS:
        print(f"Discovered additional algorithms: {sorted(list(OTHER_ALGOS.keys()))}")

# -------------------------
# Utility helpers
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_subfolders_sorted(root: Path):
    # return only directories (not files) sorted alphabetically
    return sorted([p for p in root.iterdir() if p.is_dir()])

def list_image_files_sorted(folder: Path, exts=('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

def compute_estimated_bias_from_bias_list(biases_list, gamma: float):
    """
    biases_list: list of 2D arrays (each b_bezier)
    gamma: subtraction weight used in algorithm (f_i = f_i - gamma * b_bezier)
    Returns estimated_bias = sum_i (gamma * b_bezier_i)
    """
    if not biases_list:
        return None
    # stack and sum
    total = np.zeros_like(biases_list[0], dtype=np.float32)
    for b in biases_list:
        total += (gamma * b).astype(np.float32)
    return total

def save_image_uint8(arr_float, out_path: Path):
    """
    Save float image in [0,1] to uint8 PNG via skimage.io.imsave (works for 2D or 3D).
    """
    arr = np.clip(arr_float, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    io.imsave(str(out_path), arr_u8)

def save_bias_vis(bias, out_path: Path, cmap='jet'):
    """
    Visualize 2D bias as normalized grayscale (or color) image and save.
    We'll normalize bias to [0,1] by shifting min->0 and max->1.
    """
    if bias is None:
        return
    b = bias.astype(np.float32)
    bmin, bmax = float(b.min()), float(b.max())
    if (bmax - bmin) < 1e-9:
        b_vis = np.zeros_like(b)
    else:
        b_vis = (b - bmin) / (bmax - bmin)
    save_image_uint8(b_vis, out_path)

# -------------------------
# Batch processing
# -------------------------
def process_batch(input_dir: Path, output_dir: Path,
                  num_subfolders: int = 3, num_images: int = 10,
                  algorithm: str = 'fast', n_iter: int = 9, gamma: float = 0.045,
                  random_subfolders: bool = False, random_images: bool = False,
                  seed: int = 42, save_original: bool = False,
                  verbose: bool = True):
    """
    Main entrypoint to process images.

    - input_dir: root folder with category subfolders
    - output_dir: root folder where to write processed images (preserve categories)
    - num_subfolders: limit on number of subfolders to process (None => all)
    - num_images: limit on number of images per subfolder (None => all)
    - algorithm: 'fast' or 'plus'
    - n_iter: number of BFBSF iterations
    - gamma: subtraction factor (must match algorithm invocation)
    - random_subfolders / random_images: whether to pick randomly rather than sorted-first
    - seed: RNG seed for reproducible selection
    """
    rng = np.random.default_rng(seed)

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    subfolders = list_subfolders_sorted(input_dir)
    if not subfolders:
        raise ValueError(f"No subfolders found in {input_dir}")

    # optionally randomize subfolders
    if random_subfolders:
        idxs = rng.choice(len(subfolders), size=min(num_subfolders or len(subfolders), len(subfolders)), replace=False)
        chosen_subfolders = [subfolders[i] for i in idxs]
    else:
        chosen_subfolders = subfolders[:num_subfolders] if num_subfolders is not None else subfolders

    total_processed = 0
    for sub in (tqdm(chosen_subfolders, desc="Subfolders") if verbose else chosen_subfolders):
        rel = sub.name
        out_sub = output_dir / rel
        ensure_dir(out_sub)

        img_files = list_image_files_sorted(sub)
        if random_images:
            if not img_files:
                continue
            k = min(num_images or len(img_files), len(img_files))
            chosen_idxs = rng.choice(len(img_files), size=k, replace=False)
            chosen_files = [img_files[i] for i in chosen_idxs]
        else:
            chosen_files = img_files[:num_images] if num_images is not None else img_files

        if verbose:
            print(f"Processing subfolder '{rel}' : {len(chosen_files)} images")

        for img_path in (tqdm(chosen_files, desc=f"Images in {rel}", leave=False) if verbose else chosen_files):
            try:
                # load grayscale image (single-channel) and normalize to [0,1]
                img = img_as_float(io.imread(str(img_path), as_gray=True)).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
            except Exception as e:
                print(f"  Skipping {img_path.name}: cannot read ({e})")
                continue

            # call the chosen algorithm
            alg_key = algorithm.lower()
            if alg_key in ('fast', 'bfbsf'):
                f_final, biases = bfbsf(img, n_iter=n_iter, gamma=gamma, verbose=False)
            elif alg_key in ('plus', 'bfbsf_plus', 'bfbsf+'):
                f_final, biases = bfbsf_plus(img, n_iter=n_iter, gamma=gamma, verbose=False)
            elif alg_key in OTHER_ALGOS:
                # call other author algorithm: they return (corrected, estimated_bias)
                try:
                    f_final, est_bias = OTHER_ALGOS[alg_key](img)
                    # for consistency with BFBSF, set biases to a single-element list
                    biases = [est_bias]
                except Exception as e:
                    print(f"  Algorithm {algorithm} failed on {img_path.name}: {e}")
                    continue
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # compute estimated bias removed (sum gamma*bias_i)
            estimated_bias = compute_estimated_bias_from_bias_list(biases, gamma=gamma)

            # prepare output paths
            stem = img_path.stem
            corrected_path = out_sub / f"{stem}_corrected.png"
            bias_path = out_sub / f"{stem}_estimated_bias.png"
            orig_copy_path = out_sub / f"{stem}_original.png"

            # save images
            save_image_uint8(f_final, corrected_path)
            if estimated_bias is not None:
                save_bias_vis(estimated_bias, bias_path)
            if save_original:
                save_image_uint8(img, orig_copy_path)

            total_processed += 1

    if verbose:
        print(f"Done. Total processed images: {total_processed}")
        print(f"Outputs saved to: {output_dir}")

# -------------------------
# CLI
# -------------------------

