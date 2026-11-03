import io
import os
import time
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError

from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

from labels import CLASS_LABELS, display_crop_name, display_disease_name, is_healthy
from severity import STATUS_ERROR, STATUS_UNRELIABLE, build_visualizations, estimate_severity
from treatment import TreatmentEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_MB', '16')) * 1024 * 1024

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'Model.hdf5'))
INPUT_SIZE = (224, 224)
# Predictions below this confidence (percent) are flagged as uncertain and
# disease-specific / chemical advice is withheld (value from the project proposal).
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get('LOW_CONFIDENCE_THRESHOLD', '60'))
ALLOWED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP'}
MIN_IMAGE_SIDE = 32

HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'
LFS_POINTER_PREFIX = b'version https://git-lfs.github.com/spec'

model = None
model_load_error = None


def check_model_file(path):
    """Return a human-readable problem description, or None if the file looks like an HDF5 model."""
    if not os.path.exists(path):
        return 'Model file not found at %s. Set MODEL_PATH or place Model.hdf5 in the project root.' % path
    with open(path, 'rb') as fh:
        head = fh.read(64)
    if head.startswith(LFS_POINTER_PREFIX):
        return ('%s is a Git LFS pointer (%d bytes), not the trained weights. Clone the repository with '
                'git-lfs installed and run "git lfs pull", or download the real Model.hdf5 (~129 MB) from '
                'GitHub, or set MODEL_PATH to its location.' % (path, os.path.getsize(path)))
    if not head.startswith(HDF5_SIGNATURE):
        return '%s is not an HDF5 file.' % path
    return None


def load_model(path=MODEL_PATH):
    """Load the trained Keras model. On failure the app still starts and /predict returns 503."""
    global model, model_load_error
    problem = check_model_file(path)
    if problem:
        model, model_load_error = None, problem
        print(' ** Model NOT loaded: %s' % problem)
        return None
    import tensorflow as tf  # imported lazily so tests and tooling can run without loading TF
    print(' ** Model Loading **')
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model_load_error = None
        print(' ** Model Loaded **')
    except Exception as exc:
        model, model_load_error = None, 'Failed to load %s: %s' % (path, exc)
        print(' ** Model NOT loaded: %s' % model_load_error)
    return model


if os.environ.get('SKIP_MODEL_LOAD') != '1':
    load_model()

treatment_engine = TreatmentEngine(low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD)


class UploadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def preprocess_image(img_path):
    """Pre-processing used since the minor project: RGB -> 224x224 (PIL default resampling) -> [0, 1]."""
    img = Image.open(img_path).convert('RGB').resize(INPUT_SIZE)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def model_predict(img_path, model):
    x = preprocess_image(img_path)

    preds = model.predict(x, verbose=0)
    scores = preds[0]
    class_index = int(np.argmax(scores))
    class_name = CLASS_LABELS[class_index].split('___')

    crop_scores = {}
    for index, score in enumerate(scores):
        crop = CLASS_LABELS[index].split('___')[0]
        crop_scores[crop] = crop_scores.get(crop, 0.0) + float(score)

    top3_crops = sorted(crop_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    top3_crops_result = [
        {'crop': crop, 'confidence': round(score * 100, 2)}
        for crop, score in top3_crops
    ]

    top3_diseases = [
        {
            'class_label': CLASS_LABELS[i],
            'crop': display_crop_name(CLASS_LABELS[i]),
            'disease': display_disease_name(CLASS_LABELS[i]),
            'confidence': round(float(scores[i]) * 100, 2),
        }
        for i in np.argsort(scores)[::-1][:3]
    ]

    return {
        'predicted_crop': class_name[0],
        'predicted_disease': class_name[1].title().replace('_', ' ') if len(class_name) > 1 else 'Unknown',
        'confidence': round(float(scores[class_index]) * 100, 2),
        'top3_crops': top3_crops_result,
        'class_index': class_index,
        'class_label': CLASS_LABELS[class_index],
        'top3_diseases': top3_diseases,
    }


def read_validated_image(file_storage):
    """Read an uploaded file and make sure it is a decodable image in an allowed format."""
    data = file_storage.read()
    if not data:
        raise UploadError('Uploaded file is empty')
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
        with Image.open(io.BytesIO(data)) as img:  # verify() does not decode pixels; load() does
            img.load()
            width, height = img.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UploadError('Uploaded file is not a valid image (%s)' % exc.__class__.__name__)
    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise UploadError('Unsupported image format %s; use JPG or PNG' % fmt)
    if min(width, height) < MIN_IMAGE_SIDE:
        raise UploadError('Image is too small (%dx%d px)' % (width, height))
    return data, fmt


def save_upload(data, filename, fmt):
    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    name = secure_filename(filename) or 'upload.%s' % fmt.lower()
    file_path = os.path.join(upload_dir, '%s_%s' % (uuid.uuid4().hex[:8], name))
    with open(file_path, 'wb') as fh:
        fh.write(data)
    return file_path


def analyse_image(file_path, include_visualization=True):
    """Classifier -> severity estimator -> treatment engine, returning the enriched response dict."""
    timings = {}
    warnings = []

    t0 = time.perf_counter()
    prediction = model_predict(file_path, model)
    timings['classification'] = round((time.perf_counter() - t0) * 1000, 1)

    label = prediction['class_label']
    confidence = prediction['confidence']
    healthy = is_healthy(label)
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    if low_confidence:
        warnings.append('Low confidence (%.1f%%, threshold %.0f%%): the identification is uncertain. Upload a '
                        'clearer, well-lit close-up of a single leaf, or ask an expert to confirm.'
                        % (confidence, LOW_CONFIDENCE_THRESHOLD))

    t1 = time.perf_counter()
    visualization = None
    if healthy:
        severity = {
            'status': 'skipped', 'category': 'Not applicable', 'percentage': None,
            'method': None, 'experimental': True,
            'warnings': ['Leaf classified as healthy; severity estimation skipped.'],
        }
    else:
        result = estimate_severity(file_path, class_label=label)
        severity = result.to_dict()
        if low_confidence:
            severity['warnings'].append('The colour profile was selected from a low-confidence prediction.')
        if result.status in (STATUS_UNRELIABLE, STATUS_ERROR):
            warnings.append('Severity could not be estimated reliably (%s); no severity percentage is reported.'
                            % result.status)
        if include_visualization:
            visualization = build_visualizations(result)
    timings['severity'] = round((time.perf_counter() - t1) * 1000, 1)

    t2 = time.perf_counter()
    treatment = treatment_engine.recommend(label, confidence=confidence,
                                           severity_category=severity['category'],
                                           severity_status=severity['status'])
    timings['treatment'] = round((time.perf_counter() - t2) * 1000, 1)
    timings['total'] = round((time.perf_counter() - t0) * 1000, 1)

    response = dict(prediction)  # legacy top-level keys used by the original frontend
    response.update({
        'success': True,
        'prediction': {
            'class_index': prediction['class_index'],
            'class_label': label,
            'crop': display_crop_name(label),
            'disease': display_disease_name(label),
            'is_healthy': healthy,
            'confidence': confidence,
            'low_confidence': low_confidence,
            'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
            'top3_diseases': prediction['top3_diseases'],
            'top3_crops': prediction['top3_crops'],
        },
        'severity': severity,
        'visualization': visualization,
        'treatment': treatment,
        'warnings': warnings,
        'timing_ms': timings,
        'model': {'name': 'AlexNet-style CNN (Model.hdf5)', 'input_size': list(INPUT_SIZE),
                  'num_classes': len(CLASS_LABELS)},
    })
    return response


def error_response(message, status):
    return jsonify({'success': False, 'error': message}), status


@app.errorhandler(413)
def too_large(_exc):
    return error_response('File too large (limit %d MB)' % (app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)), 413)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    covered, missing = treatment_engine.coverage(CLASS_LABELS)
    return jsonify({
        'status': 'ok' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_error': model_load_error,
        'knowledge_base_coverage': '%d/%d' % (len(covered), len(CLASS_LABELS)),
        'knowledge_base_missing': missing,
        'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
    })


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed'}), 405

    if 'file' not in request.files:
        return error_response('No file part in request', 400)

    f = request.files['file']
    if f.filename == '':
        return error_response('No file selected', 400)

    if model is None:
        return error_response('Model is not loaded: %s' % model_load_error, 503)

    try:
        data, fmt = read_validated_image(f)
    except UploadError as exc:
        return error_response(str(exc), exc.status)

    include_visualization = request.values.get('visualize', '1').lower() not in ('0', 'false', 'no')
    try:
        file_path = save_upload(data, f.filename, fmt)
        return jsonify(analyse_image(file_path, include_visualization=include_visualization))
    except Exception as exc:
        app.logger.exception('Prediction failed')
        return error_response(str(exc), 500)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
