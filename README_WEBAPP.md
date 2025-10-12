Flask Web App for BFBSF Demo

This small Flask app allows uploading a grayscale image, applying the provided degrading function, selecting a denoiser algorithm, running denoising, and evaluating results with PSNR/SSIM using Evaluator.py.

Files:
 - web_app.py            : Flask application
 - templates/index.html  : web UI
 - uploads/              : uploaded grayscale images (created at runtime)
 - degraded/             : degraded images (created at runtime)
 - denoised/             : denoised outputs (created at runtime)

Requirements:
 - Install project requirements in your Python environment (prefer a virtualenv):

   python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt

 - If running on a separate env, ensure Flask and Pillow are installed.

Run:
 - From the `Codes` folder:

   python web_app.py

 - Open http://127.0.0.1:5000/ in your browser.

Notes:
 - Uploaded images are converted to grayscale PNGs.
 - The degrade step calls `degrade_image_with_hotspot` from `Datacreator.py` (see that file for parameters). The call in `web_app.py` uses default parameters.
 - The denoise step uses `bfbsf` / `bfbsf_plus` from `Algo2.py` via `Denoiser.py`. Other algorithms discovered in `Algorithms/` are also available if present.
 - If original upload cannot be matched when computing metrics, no metrics will be shown.

If you want changes (e.g., different default parameters or more UI features), tell me what to adjust.