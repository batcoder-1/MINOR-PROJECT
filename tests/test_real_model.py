"""
Regression check against the real trained model. Skipped unless MODEL_PATH points
to the real Model.hdf5 (not the Git LFS pointer):

    MODEL_PATH=/path/to/Model.hdf5 pytest -m requires_model

This only confirms the preserved baseline still produces the same predictions on
the sample images shipped in uploads/; it is NOT an accuracy evaluation.
"""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get('MODEL_PATH', '')

pytestmark = pytest.mark.requires_model

# sample file -> class the baseline model predicted for it on 2026-11-03
EXPECTED = {
    'AppleCedarRust1.JPG': 'Apple___Cedar_apple_rust',
    'CornCommonRust1.JPG': 'Corn_(maize)___Common_rust_',
    'PotatoEarlyBlight1.JPG': 'Potato___Early_blight',
    'PotatoHealthy2.JPG': 'Potato___healthy',
    'TomatoEarlyBlight5.JPG': 'Tomato___Early_blight',
    'TomatoYellowCurlVirus1.JPG': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
}


@pytest.fixture(scope='module')
def real_model(app_module_real):
    return app_module_real.model


@pytest.fixture(scope='module')
def app_module_real():
    import app as app_module
    if not MODEL_PATH or app_module.check_model_file(MODEL_PATH):
        pytest.skip('MODEL_PATH does not point to the real Model.hdf5')
    model = app_module.load_model(MODEL_PATH)
    assert model is not None, app_module.model_load_error
    return app_module


def test_architecture_unchanged(real_model):
    assert real_model.input_shape == (None, 224, 224, 3)
    assert real_model.output_shape == (None, 38)
    assert real_model.count_params() == 28117790


@pytest.mark.parametrize('filename,expected', sorted(EXPECTED.items()))
def test_baseline_predictions_preserved(app_module_real, filename, expected):
    result = app_module_real.model_predict(os.path.join(ROOT, 'uploads', filename), app_module_real.model)
    assert result['class_label'] == expected
    assert result['confidence'] > 60
