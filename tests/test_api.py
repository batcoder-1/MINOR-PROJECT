"""Flask API integration tests using a fake classifier (deterministic, no TensorFlow needed)."""

import io
import os

import numpy as np
import pytest
from PIL import Image

from tests.conftest import BROWN_RGB, FakeModel, make_leaf_image, to_png_bytes

LEGACY_KEYS = {'predicted_crop', 'predicted_disease', 'confidence', 'top3_crops'}


def _post(client, data_bytes, filename='leaf.png', **extra):
    data = {'file': (io.BytesIO(data_bytes), filename)}
    data.update(extra)
    return client.post('/predict', data=data, content_type='multipart/form-data')


@pytest.fixture
def leaf_png():
    img, _, _ = make_leaf_image(lesions=[(170, 180, 30, BROWN_RGB), (240, 215, 25, BROWN_RGB)])
    return to_png_bytes(img)


def test_index_page_renders(client):
    resp = client.get('/')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'id="imageUpload"' in html and 'id="result"' in html and 'id="analysis"' in html


def test_get_predict_not_allowed(client):
    resp = client.get('/predict')
    assert resp.status_code == 405 and resp.get_json()['error'] == 'Method not allowed'


def test_missing_file_part(client, app_module):
    app_module.model = FakeModel()
    resp = client.post('/predict', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400 and 'No file part' in resp.get_json()['error']


def test_empty_filename(client, app_module):
    app_module.model = FakeModel()
    resp = _post(client, b'', filename='')
    assert resp.status_code == 400


@pytest.mark.parametrize('payload,fragment', [
    (b'definitely not an image', 'not a valid image'),
    (b'', 'empty'),
    (b'\x89PNG\r\n\x1a\n' + b'\x00' * 40, 'not a valid image'),   # truncated PNG header
])
def test_invalid_uploads_rejected(client, app_module, payload, fragment):
    app_module.model = FakeModel()
    resp = _post(client, payload, filename='leaf.jpg')
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['success'] is False and fragment in body['error']
    assert app_module.model.calls == 0


def test_unsupported_format_rejected(client, app_module):
    app_module.model = FakeModel()
    buf = io.BytesIO()
    Image.new('RGB', (100, 100), (0, 128, 0)).save(buf, format='GIF')
    resp = _post(client, buf.getvalue(), filename='leaf.gif')
    assert resp.status_code == 400 and 'Unsupported image format' in resp.get_json()['error']


def test_too_small_image_rejected(client, app_module):
    app_module.model = FakeModel()
    resp = _post(client, to_png_bytes(np.zeros((10, 10, 3), np.uint8)))
    assert resp.status_code == 400 and 'too small' in resp.get_json()['error']


def test_oversized_upload_returns_413(client, app_module, monkeypatch):
    app_module.model = FakeModel()
    monkeypatch.setitem(app_module.app.config, 'MAX_CONTENT_LENGTH', 1024)
    resp = _post(client, b'x' * 5000)
    assert resp.status_code == 413 and 'too large' in resp.get_json()['error']


def test_model_not_loaded_returns_503(client, app_module, leaf_png):
    app_module.model = None
    app_module.model_load_error = 'Model.hdf5 is a Git LFS pointer'
    resp = _post(client, leaf_png)
    assert resp.status_code == 503 and 'LFS' in resp.get_json()['error']


def test_diseased_prediction_full_response(client, app_module, leaf_png):
    app_module.model = FakeModel('Tomato___Early_blight', 0.93)
    resp = _post(client, leaf_png)
    assert resp.status_code == 200
    body = resp.get_json()

    # legacy contract used by the original frontend is unchanged
    assert LEGACY_KEYS <= set(body)
    assert body['predicted_crop'] == 'Tomato' and body['predicted_disease'] == 'Early Blight'
    assert body['confidence'] == pytest.approx(93.0, abs=0.01)
    assert body['top3_crops'][0]['crop'] == 'Tomato'

    assert body['success'] is True
    pred = body['prediction']
    assert pred['class_label'] == 'Tomato___Early_blight' and pred['class_index'] == 29
    assert pred['low_confidence'] is False and len(pred['top3_diseases']) == 3

    sev = body['severity']
    assert sev['status'] == 'estimated' and sev['category'] in ('Mild', 'Moderate', 'Severe')
    assert 0 < sev['percentage'] < 100 and sev['experimental'] is True

    for key in ('original', 'leaf_mask', 'lesion_mask', 'overlay'):
        assert body['visualization'][key].startswith('data:image/')

    treat = body['treatment']
    assert treat['class_label'] == 'Tomato___Early_blight'
    assert treat['decision']['severity_category'] == sev['category']
    assert treat['sources'] and treat['disclaimer']
    assert set(body['timing_ms']) == {'classification', 'severity', 'treatment', 'total'}
    assert app_module.model.last_input.shape == (1, 224, 224, 3)


def test_upload_saved_with_unique_name(client, app_module, leaf_png, tmp_path):
    app_module.model = FakeModel()
    _post(client, leaf_png, filename='same.png')
    _post(client, leaf_png, filename='same.png')
    saved = os.listdir(os.path.join(str(tmp_path), 'uploads'))
    assert len(saved) == 2 and all(name.endswith('_same.png') for name in saved)


def test_visualization_can_be_disabled(client, app_module, leaf_png):
    app_module.model = FakeModel()
    body = _post(client, leaf_png, visualize='0').get_json()
    assert body['visualization'] is None and body['severity']['status'] == 'estimated'


def test_healthy_prediction_skips_severity(client, app_module, leaf_png):
    app_module.model = FakeModel('Potato___healthy', 0.98, runner_up='Potato___Early_blight')
    body = _post(client, leaf_png).get_json()
    assert body['prediction']['is_healthy'] is True
    assert body['severity']['status'] == 'skipped' and body['severity']['percentage'] is None
    assert body['visualization'] is None
    assert body['treatment']['decision']['rules_applied'] == ['healthy_preventive_only']
    assert body['treatment']['chemical_guidance']['status'] == 'not_indicated'


def test_low_confidence_prediction_suppresses_treatment(client, app_module, leaf_png):
    app_module.model = FakeModel('Tomato___Late_blight', 0.45, runner_up='Tomato___Early_blight')
    body = _post(client, leaf_png).get_json()
    assert body['prediction']['low_confidence'] is True
    assert any('Low confidence' in w for w in body['warnings'])
    t = body['treatment']
    assert t['chemical_guidance']['status'] == 'suppressed_low_confidence'
    assert t['management_guidance'] == [] and t['expert_consultation']['recommended'] is True


def test_viral_prediction(client, app_module, leaf_png):
    app_module.model = FakeModel('Tomato___Tomato_Yellow_Leaf_Curl_Virus', 0.99)
    body = _post(client, leaf_png).get_json()
    assert body['severity']['status'] == 'not_applicable'
    assert body['treatment']['chemical_guidance']['status'] == 'no_curative_chemical'
    assert body['treatment']['vector_management']


def test_unreliable_segmentation_is_surfaced(client, app_module):
    app_module.model = FakeModel('Tomato___Early_blight', 0.95)
    flat = to_png_bytes(np.full((300, 300, 3), (128, 128, 128), np.uint8))
    body = _post(client, flat).get_json()
    assert body['severity']['status'] == 'unreliable' and body['severity']['percentage'] is None
    assert any('could not be estimated' in w for w in body['warnings'])
    assert body['treatment']['decision']['rules_applied'] == ['severity_unknown']


def test_health_endpoint(client, app_module):
    app_module.model = None
    app_module.model_load_error = 'missing'
    body = client.get('/health').get_json()
    assert body['model_loaded'] is False and body['status'] == 'degraded'
    assert body['knowledge_base_coverage'] == '38/38'


# ---------------------------------------------------------------- model-file checks

def test_lfs_pointer_detected(app_module, tmp_path):
    pointer = tmp_path / 'Model.hdf5'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 129105944\n')
    problem = app_module.check_model_file(str(pointer))
    assert problem and 'Git LFS pointer' in problem


def test_non_hdf5_and_missing_file_detected(app_module, tmp_path):
    junk = tmp_path / 'junk.hdf5'
    junk.write_bytes(b'hello world')
    assert 'not an HDF5' in app_module.check_model_file(str(junk))
    assert 'not found' in app_module.check_model_file(str(tmp_path / 'nope.hdf5'))


def test_load_model_failure_does_not_crash(app_module, tmp_path):
    pointer = tmp_path / 'Model.hdf5'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\n')
    assert app_module.load_model(str(pointer)) is None
    assert app_module.model is None and 'LFS' in app_module.model_load_error
