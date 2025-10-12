import os
import sys
import yaml
import shutil
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
sys.path.append(os.path.dirname(__file__))
from src.Datacreator import degrade_train_folder_with_quadrants
from src.Denoiser import process_batch
from src.Evaluator import compute_image_metrics
from skimage import io, img_as_float

# ----------------------
# Utility Functions
# ----------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_categories(root, max_categories=None):
    cats = [d for d in Path(root).iterdir() if d.is_dir()]
    cats = sorted(cats)
    if max_categories is not None:
        cats = cats[:max_categories]
    return cats

def list_images(folder, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    return sorted([f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() in exts])

def save_metrics_csv(metrics_dict, out_path):
    import csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "image", "algo", "psnr", "ssim", "cv"])
        for row in metrics_dict:
            writer.writerow(row)

def plot_category_metrics(category_metrics, out_dir, algos):
    # category_metrics: {algo: {category: [metric_dict, ...]}}
    for metric in ["psnr", "ssim", "cv"]:
        # collect categories (sorted)
        cats = []
        # find union of categories across algos (preserve order)
        for algo in algos:
            for c in category_metrics.get(algo, {}).keys():
                if c not in cats:
                    cats.append(c)
        cats = sorted(cats)

        # build value matrix: rows = algos, cols = categories
        vals_matrix = []
        for algo in algos:
            vals = []
            for cat in cats:
                entries = category_metrics.get(algo, {}).get(cat, [])
                if entries:
                    vals.append(np.mean([m[metric] for m in entries]))
                else:
                    vals.append(np.nan)
            vals_matrix.append(vals)

        # grouped bar chart
        x = np.arange(len(cats))
        n_alg = len(algos)
        total_width = 0.8
        if n_alg > 0:
            width = total_width / n_alg
        else:
            width = total_width

        plt.figure(figsize=(max(8, len(cats) * 0.8), 5))
        for i, algo in enumerate(algos):
            vals = vals_matrix[i]
            # offset so groups are centered
            offset = (i - (n_alg - 1) / 2.0) * width
            plt.bar(x + offset, vals, width=width, label=algo)

        plt.title(f"Category-wise {metric.upper()} Comparison")
        plt.xlabel("Category")
        plt.ylabel(metric.upper())
        plt.xticks(x, cats, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"categorywise_{metric}.png"))
        plt.close()

def plot_overall_comparison(category_metrics, out_dir, algos):
    # Overall average for each algo
    for metric in ["psnr", "ssim", "cv"]:
        plt.figure(figsize=(6,4))
        vals = []
        for algo in algos:
            all_vals = []
            for cat in category_metrics[algo]:
                all_vals.extend([m[metric] for m in category_metrics[algo][cat]])
            vals.append(np.mean(all_vals))
        plt.bar(algos, vals)
        plt.title(f"Overall {metric.upper()} Comparison")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overall_{metric}.png"))
        plt.close()

# ----------------------
# Main Pipeline
# ----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full pipeline: degrade, denoise, evaluate, plot.")
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Degrade dataset
    degrade_cfg = config['datacreator']
    degrade_in = degrade_cfg['input_dir']
    degrade_out = degrade_cfg['output_dir']
    ensure_dir(degrade_out)
    logging.info(f"Degrading dataset: {degrade_in} -> {degrade_out}")
    degrade_train_folder_with_quadrants(
        train_dir=degrade_in,
        out_dir=degrade_out,
        variants_per_image=degrade_cfg.get('variants', 1),
        quadrants=degrade_cfg.get('quadrants', ('top-left','top-right','bottom-left','bottom-right')),
        overwrite=degrade_cfg.get('overwrite', False),
        radius_frac=degrade_cfg.get('radius', 0.85),
        amplitude=degrade_cfg.get('amplitude', 0.5),
        bg_amp=degrade_cfg.get('bg_amp', 0.03),
        gauss_noise=degrade_cfg.get('gauss', 0.01),
        poisson_scale=degrade_cfg.get('poisson', 0.0),
        rng_seed=degrade_cfg.get('seed', 42)
    )
    
    # 2. Denoise with each algorithm
    denoise_cfg = config['denoiser']
    eval_cfg = config['evaluator']
    compare = config.get('compare', False)
    requested_algos = denoise_cfg.get('algos', ['fast'])

    # Auto-discover additional algorithms in Codes/Algorithms (optional).
    alg_dir = Path(os.path.dirname(__file__)) / 'Algorithms'
    discovered = []
    if alg_dir.exists() and alg_dir.is_dir():
        for p in alg_dir.iterdir():
            if p.is_file() and p.suffix == '.py' and p.name.startswith('algo_'):
                discovered.append(p.stem[len('algo_'):])

    # Build final algos list:
    # - If config lists '*' -> run base algos plus discovered
    # - Otherwise use the requested list but keep order and include discovered ones if explicitly named
    if isinstance(requested_algos, list) and '*' in requested_algos:
        algos = ['fast']                                                  # need plus add here
        # append discovered in sorted order
        for d in sorted(discovered):
            if d not in algos:
                algos.append(d)
    else:
        # keep requested order, normalize to strings
        algos = [str(a) for a in requested_algos]
    denoise_out_root = denoise_cfg['output_root']
    ensure_dir(denoise_out_root)
    degraded_root = degrade_out
    categories = list_categories(degraded_root, max_categories=denoise_cfg.get('num_categories', 3))
    metrics_all = []
    category_metrics = {algo: {cat.name: [] for cat in categories} for algo in algos}

    for algo in algos:
        logging.info(f"Denoising with algorithm: {algo}")
        algo_out = os.path.join(denoise_out_root, algo)
        ensure_dir(algo_out)
        process_batch(
            input_dir=degraded_root,
            output_dir=algo_out,
            num_subfolders=len(categories),
            num_images=denoise_cfg.get('num_images', None),
            algorithm=algo,
            n_iter=denoise_cfg.get('n_iter', 9),
            gamma=denoise_cfg.get('gamma', 0.045),
            random_subfolders=False,
            random_images=False,
            seed=denoise_cfg.get('seed', 42),
            save_original=False,
            verbose=True
        )

        # 3. Evaluate denoised images
        for cat in tqdm(categories, desc=f"Evaluating {algo}"):
            orig_cat_dir = Path(degrade_in) / cat.name
            den_cat_dir = Path(algo_out) / cat.name
            for img_path in list_images(den_cat_dir):
                if not img_path.name.endswith('_corrected.png'):
                    continue

                # degraded_stem is the stem used when creating the corrected output
                degraded_stem = img_path.name[:-len('_corrected.png')]

                # Try to recover original stem by removing typical degradation suffixes
                import re
                s = degraded_stem
                # remove anything starting with _deg or _degraded
                s = re.sub(r'(_deg|_degraded).*$', '', s)
                # remove quadrant markers like _qtopleft or _qtop_left
                s = re.sub(r'_q[0-9A-Za-z_]+', '', s)
                # remove variant suffixes like _v0, _v1
                s = re.sub(r'_v\d+$', '', s)
                # final cleaned candidate stem
                orig_stem_candidate = s

                # try common extensions and a few fallbacks
                found = False
                for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                    cand = orig_cat_dir / (orig_stem_candidate + ext)
                    if cand.exists():
                        orig_img_path = cand
                        found = True
                        break

                # fallback: try exact degraded_stem without cleaning (in case variants were off)
                if not found:
                    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                        cand = orig_cat_dir / (degraded_stem + ext)
                        if cand.exists():
                            orig_img_path = cand
                            found = True
                            break

                # final fallback: search for files containing the cleaned stem
                if not found:
                    for f in orig_cat_dir.iterdir():
                        if f.is_file() and orig_stem_candidate in f.stem:
                            orig_img_path = f
                            found = True
                            break

                if not found:
                    logging.warning(f"Original image not found for {img_path.name} (tried stem '{orig_stem_candidate}')")
                    continue
                try:
                    # load denoised (test) image as numpy array (grayscale or RGB) in [0,1]
                    test_arr = img_as_float(io.imread(str(img_path))).astype(np.float32)
                    metrics = compute_image_metrics(str(orig_img_path), test_arr, resize_ref=True)
                    row = [cat.name, img_path.name, algo, metrics['psnr'], metrics['ssim'], metrics['cv']]
                    metrics_all.append(row)
                    category_metrics[algo][cat.name].append(metrics)
                except Exception as e:
                    logging.warning(f"Metric computation failed for {img_path.name}: {e}")

    # 4. Save metrics and plot
    metrics_csv = os.path.join(denoise_out_root, 'metrics_all.csv')
    save_metrics_csv(metrics_all, metrics_csv)
    plot_category_metrics(category_metrics, denoise_out_root, algos)
    plot_overall_comparison(category_metrics, denoise_out_root, algos)
    logging.info(f"Pipeline complete. Metrics and plots saved in {denoise_out_root}")

if __name__ == "__main__":
    main()
