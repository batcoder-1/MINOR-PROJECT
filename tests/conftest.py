"""
Shared fixtures. Synthetic images are drawn with known geometry so that leaf and
lesion areas are known exactly; they test the *algorithms*, not real-world accuracy.
"""

import io
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The Flask app must not try to load Model.hdf5 during tests (it is a Git LFS
# pointer in a ZIP checkout and loading TensorFlow is slow).
os.environ['SKIP_MODEL_LOAD'] = '1'

from labels import CLASS_LABELS  # noqa: E402

BACKGROUND_RGB = (128, 128, 128)   # neutral grey: saturation 0
LEAF_RGB = (60, 140, 50)           # healthy green: H ~ 53 (OpenCV), S ~ 164
BROWN_RGB = (120, 70, 30)          # necrotic brown: H ~ 13, S ~ 191, V 120
DARK_RGB = (30, 30, 30)            # black necrosis: V 30


def make_leaf_image(size=400, lesions=(), leaf_axes=(150, 110), background=BACKGROUND_RGB,
                    leaf_color=LEAF_RGB, extra=None):
    """Draw an elliptical leaf with circular lesions.

    lesions: iterable of (cx, cy, radius, rgb)
    extra:   optional callable(img) for additional drawing (e.g. background blobs)
    Returns (rgb_image, leaf_mask_truth, lesion_mask_truth) as uint8 arrays.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = background
    leaf = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(leaf, (size // 2, size // 2), leaf_axes, 0, 0, 360, 255, -1)
    img[leaf > 0] = leaf_color
    lesion = np.zeros_like(leaf)
    for cx, cy, r, rgb in lesions:
        m = np.zeros_like(leaf)
        cv2.circle(m, (cx, cy), r, 255, -1)
        m &= leaf
        img[m > 0] = rgb
        lesion |= m
    if extra is not None:
        extra(img)
    return img, leaf, lesion


def true_percentage(leaf, lesion):
    return 100.0 * np.count_nonzero(lesion) / np.count_nonzero(leaf)


def to_png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def healthy_leaf():
    return make_leaf_image()


@pytest.fixture
def diseased_leaf():
    # ~15 % lesion area: three brown circles inside a 150x110 ellipse
    lesions = [(170, 180, 30, BROWN_RGB), (240, 220, 25, BROWN_RGB), (200, 250, 18, BROWN_RGB)]
    return make_leaf_image(lesions=lesions)


class FakeModel:
    """Stands in for the Keras model: returns a fixed probability vector."""

    def __init__(self, class_label='Tomato___Late_blight', confidence=0.9, runner_up='Tomato___Early_blight'):
        self.calls = 0
        self.last_input = None
        probs = np.zeros(len(CLASS_LABELS), dtype=np.float32)
        idx = CLASS_LABELS.index(class_label)
        probs[idx] = confidence
        rest = 1.0 - confidence
        if runner_up == class_label:
            runner_up = 'Apple___healthy' if class_label != 'Apple___healthy' else 'Tomato___healthy'
        if runner_up and rest > 0:
            probs[CLASS_LABELS.index(runner_up)] = rest
        self.probs = probs

    def predict(self, x, verbose=0):
        self.calls += 1
        self.last_input = x
        return np.repeat(self.probs[None, :], x.shape[0], axis=0)


@pytest.fixture
def app_module():
    import app as app_module
    original_model, original_error = app_module.model, app_module.model_load_error
    yield app_module
    app_module.model, app_module.model_load_error = original_model, original_error


@pytest.fixture
def client(app_module, tmp_path, monkeypatch):
    """Flask test client with uploads redirected to a temp dir."""
    monkeypatch.setattr(app_module, 'BASE_DIR', str(tmp_path))
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()
