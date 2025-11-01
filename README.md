# Aero-Optical Thermal Radiation Degradation & Denoising Pipeline

<img width="1920" height="1080" alt="GUI" src="https://github.com/user-attachments/assets/1878a771-ceca-4842-bc5d-a8d5e2f9a304" />

## Overview
This project provides a complete pipeline for generating, degrading, denoising, and evaluating remote sensing images affected by synthetic aero-optical thermal radiation bias. The workflow is modular, reproducible, and configurable via YAML, supporting robust experimentation and benchmarking of denoising algorithms.

## Features
- **Synthetic Degradation**: Add configurable circular hotspot bias and sensor noise to images, preserving folder structure.
- **Batch Denoising**: Apply advanced denoising algorithms (BFBSF, BFBSF+) to entire datasets, saving corrected and bias images.
- **Evaluation**: Compute image quality metrics (PSNR, SSIM, CV, etc.) for quantitative assessment.
- **Visualization**: Plot denoising iterations, compare algorithms, and visualize results interactively or as saved figures.
- **Configurable Pipeline**: All parameters and paths are controlled via a YAML config file for reproducibility.
- **Quick Testing**: Scripts for rapid single-image testing of both degradation and denoising steps.

## Getting Started

### 1. Install Dependencies
Create a virtual environment (recommended) and install required packages:
```sh
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r Codes/requirements.txt
```

### 2. Configure the Pipeline
Edit `Codes/Config.yaml` to set input/output paths, algorithm parameters, and other options.

### 3. Run the Full Pipeline
```sh
python Codes/main.py --config Codes/Config.yaml
```
This will:
- Degrade images in the specified input folder
- Denoise them using selected algorithms
- Evaluate and plot results

### 4. Quick Single-Image Testing
- **Degradation:**
  ```sh
  python Codes/tester.py --input path/to/image.jpg --show
  ```
- **Denoising:**
  ```sh
  python Codes/quick_denoiser2.py --input path/to/degraded.png --original path/to/original.jpg --algorithm plus --n_iter 9 --gamma 0.045 --show
  ```

## Key Scripts
- `Datacreator.py`: Batch and single-image synthetic degradation
- `Denoiser.py`: Batch denoising with BFBSF/BFBSF+
- `Denoiser.py`: Batch denoising with BFBSF/BFBSF+ and support for additional `algo_*` modules placed in `Codes/Algorithms/`
- `Algo2.py`: Denoising algorithm implementations
- `Evaluator.py`: Image quality metrics
- `quick_denoiser2.py`: Quick denoising test and visualization (single image)
- `tester.py`: Quick degradation test and visualization (single image)
- `main.py`: Orchestrates the full pipeline using YAML config
 - `main.py`: Orchestrates the full pipeline using YAML config
 - `Codes/Algorithms/`: Directory containing additional third-party or reimplemented methods as `algo_<name>.py` modules. Each module should expose an `apply_<name>(z)` function returning `(corrected, estimated_bias)` and may optionally provide a `quick_test()` CLI.
 - `compare_other_methods.py`: Run a collection of `algo_*` modules on a single degraded image and show/save results and metrics.

## Configuration
All main parameters (paths, algorithm settings, number of variants, etc.) are set in `Codes/Config.yaml`. Example:
```yaml
datacreator:
  input_dir: Datasets/test
  output_dir: Degraded/
  variants: 1
  radius: 0.85
  amplitude: 0.5
  bg_amp: 0.02
  gauss: 0.01
  poisson: 0.0
  seed: 42

denoiser:
  output_root: Denosied/
  # algos can list any of: 'fast', 'plus', or additional algorithm keys derived from
  # filenames in Codes/Algorithms/ (e.g. 'cao_tisse', 'zheng'). Use ['*'] to run built-ins
  # plus all discovered algo_*.py modules automatically.
  algos: ['fast', 'plus']
  num_categories: 3
  num_images: 10
  n_iter: 9
  gamma: 0.045
  seed: 42

evaluator:
  # (add any evaluator-specific options here if needed)

compare: False
```

## Notes
- The pipeline preserves the original folder/category structure for all outputs.
- All scripts are robust to errors and will skip/log problematic files without stopping the pipeline.
- For best results, ensure all dependencies are installed and paths in the config are correct.
- To auto-include all additional algorithms placed in `Codes/Algorithms/`, set:
  ```yaml
  denoiser:
    algos: ['*']
  ```
  This runs `fast` and `plus` followed by all discovered `algo_*.py` modules.

- Each additional algorithm module must:
  - live in `Codes/Algorithms/` and be named `algo_<key>.py`.
  - export an `apply_<...>(z)` function taking a single-channel numpy array and returning `(corrected, estimated_bias)`.
  - (optional) provide a `quick_test()` CLI for single-image debugging.

## License
This project is for academic and research use. See individual scripts for author credits.

## Acknowledgments
- Developed by Sivakavin and contributors.
- Uses open-source libraries: numpy, scipy, opencv-python-headless, scikit-image, pillow, tqdm, matplotlib, pyyaml, etc.
