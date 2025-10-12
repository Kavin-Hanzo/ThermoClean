from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
from skimage import io, img_as_float
from skimage.color import rgb2gray
from PIL import Image
import io as _io
import yaml

# local imports
from src.Datacreator import degrade_image_with_hotspot
from src.Denoiser import bfbsf, bfbsf_plus, OTHER_ALGOS
from src.Evaluator import compute_image_metrics

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DEG_FOLDER = BASE_DIR / 'Web_degrade'
DENOISED_FOLDER = BASE_DIR / 'Web_denoise'
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

for d in (UPLOAD_FOLDER, DEG_FOLDER, DENOISED_FOLDER):
    d.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dev-secret'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load Config.yaml if present
CONFIG_PATH = BASE_DIR / 'Config.yaml'
CONFIG = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as fh:
            CONFIG = yaml.safe_load(fh) or {}
    except Exception:
        CONFIG = {}

# Helpers

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


def save_gray_image(file_storage, out_path: Path):
    # read into PIL, convert to grayscale, save as PNG
    img = Image.open(file_storage.stream).convert('L')
    img.save(str(out_path))


def load_image_as_float_gray(path: Path) -> np.ndarray:
    arr = img_as_float(io.imread(str(path))).astype(np.float32)
    if arr.ndim == 3:
        arr = rgb2gray(arr)
    return np.clip(arr, 0.0, 1.0)


# Routes
@app.route('/')
def index():
    # list available algorithms: fast, plus, plus other discovered
    algos = ['fast', 'plus'] + sorted(list(OTHER_ALGOS.keys()))
    # existing images
    uploads = sorted(UPLOAD_FOLDER.iterdir()) if UPLOAD_FOLDER.exists() else []
    degraded = sorted(DEG_FOLDER.iterdir()) if DEG_FOLDER.exists() else []
    denoised = sorted(DENOISED_FOLDER.iterdir()) if DENOISED_FOLDER.exists() else []
    # read last-processed names from session (set after processing)
    last_degraded = session.get('last_degraded')
    last_denoised = session.get('last_denoised')
    return render_template('index.html', algos=algos, uploads=uploads, degraded=degraded,
                           denoised=denoised, last_degraded=last_degraded, last_denoised=last_denoised)


@app.route('/uploads', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        out_path = UPLOAD_FOLDER / filename
        save_gray_image(file, out_path)
        flash(f'Uploaded and saved as grayscale: {filename}')
    else:
        flash('Unsupported file type')
    return redirect(url_for('index'))


@app.route('/Web_degrade', methods=['POST'])
def degrade():
    # expects form field 'upload_filename'
    fname = request.form.get('upload_filename')
    if not fname:
        flash('No uploaded file selected')
        return redirect(url_for('index'))
    in_path = UPLOAD_FOLDER / fname
    if not in_path.exists():
        flash('Uploaded file not found')
        return redirect(url_for('index'))

    # load grayscale image and convert to float
    img = load_image_as_float_gray(in_path)

    # call degrading function (use defaults from Config.yaml datacreator if present)
    dc = CONFIG.get('datacreator', {})
    radius = float(dc.get('radius', 0.85))
    amplitude = float(dc.get('amplitude', 0.5))
    bg_amp = float(dc.get('bg_amp', 0.02))
    gauss_noise = float(dc.get('gauss', 0.01))
    poisson_scale = float(dc.get('poisson', 0.0))
    quadrant = dc.get('quadrants', 'random')
    # if quadrants is a list in yaml, we'll leave quadrant as 'random' (degrade fn can handle)
    try:
        z, b, f, used_quad, center_info = degrade_image_with_hotspot(
            img, output_mode='grayscale', quadrant='random', radius_frac=radius,
            amplitude=amplitude, bg_amp=bg_amp, gauss_noise=gauss_noise, poisson_scale=poisson_scale)
    except TypeError:
        # fallback if param names differ, call with minimal args
        z, b, f, used_quad, center_info = degrade_image_with_hotspot(img, output_mode='grayscale')

    # save degraded image
    out_name = f"{Path(fname).stem}_degraded.png"
    out_path = DEG_FOLDER / out_name
    io.imsave(str(out_path), (np.clip(z,0,1)*255).astype(np.uint8))

    # remember last degraded to show in UI
    session['last_degraded'] = out_name
    # clear last denoised since new degradation occurred
    session.pop('last_denoised', None)

    flash('Image degraded and saved')
    return redirect(url_for('index'))


@app.route('/Web_denoise', methods=['POST'])
def denoise():
    # expects form fields: degraded_filename, algorithm
    dname = request.form.get('degraded_filename')
    algo = request.form.get('algorithm')
    if not dname:
        flash('No degraded image selected')
        return redirect(url_for('index'))
    deg_path = DEG_FOLDER / dname
    if not deg_path.exists():
        flash('Degraded image not found')
        return redirect(url_for('index'))

    img = load_image_as_float_gray(deg_path)

    # call selected algorithm
    alg_key = algo.lower() if algo else 'fast'
    # if fast/plus, use parameters from CONFIG.den oiser if available
    den_cfg = CONFIG.get('denoiser', {})
    if alg_key in ('fast', 'bfbsf'):
        n_iter = int(den_cfg.get('n_iter', 3))
        gamma = float(den_cfg.get('gamma', 0.045))
        f_final, biases = bfbsf(img, n_iter=n_iter, gamma=gamma, verbose=False)
    elif alg_key in ('plus', 'bfbsf_plus', 'bfbsf+'):
        n_iter = int(den_cfg.get('n_iter', 3))
        gamma = float(den_cfg.get('gamma', 0.045))
        f_final, biases = bfbsf_plus(img, n_iter=n_iter, gamma=gamma, verbose=False)
    elif alg_key in OTHER_ALGOS:
        # other algs expect grayscale image and return (corrected, estimated_bias)
        f_final, est_bias = OTHER_ALGOS[alg_key](img)
        biases = [est_bias]
    else:
        flash(f'Unsupported algorithm: {algo}')
        return redirect(url_for('index'))

    out_name = Path("Denoised.png")
    out_path = DENOISED_FOLDER / out_name
    io.imsave(str(out_path), (np.clip(f_final,0,1)*255).astype(np.uint8))

    # evaluation: we need original reference. Try to find matching uploaded original by stem
    # if not found, we cannot compute metrics
    ref_candidates = list(UPLOAD_FOLDER.glob(f"{Path(dname).stem.split('_degraded')[0]}*"))
    metrics = None
    if ref_candidates:
        ref_path = str(ref_candidates[0])
        Y_denoised = img_as_float(io.imread(str(out_path))).astype(np.float32)
        metrics = compute_image_metrics(ref_path, Y_denoised, resize_ref=True)

    # save metrics as simple text next to file (optional)
    if metrics is not None:
        txt_path = DENOISED_FOLDER/Path("metrics.txt")
        with open(txt_path, 'w') as fh:
            for k,v in metrics.items():
                fh.write(f"{k}: {v}\n")

    flash('Denoising complete')
    return redirect(url_for('index'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/Web_degrade/<path:filename>')
def degraded_file(filename):
    return send_from_directory(str(DEG_FOLDER), filename)


@app.route('/Web_denoise/<path:filename>')
def denoised_file(filename):
    return send_from_directory(str(DENOISED_FOLDER), filename)


if __name__ == '__main__':
    app.run(debug=True)
