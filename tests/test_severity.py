import math

import numpy as np
import pytest

import severity.estimator as estimator_module
from severity import (CATEGORY_MILD, CATEGORY_MODERATE, CATEGORY_NOT_APPLICABLE, CATEGORY_SEVERE, CATEGORY_UNKNOWN,
                      STATUS_ERROR, STATUS_ESTIMATED, STATUS_NOT_APPLICABLE, STATUS_UNRELIABLE, build_visualizations,
                      classify_severity, compute_severity_percentage, estimate_severity, get_profile, load_config,
                      render_panel)
from severity.config import DISEASE_PROFILES, LESION_COLOR_RANGES
from labels import CLASS_LABELS, is_healthy
from tests.conftest import BROWN_RGB, make_leaf_image, true_percentage


# ---------------------------------------------------------------- severity calculation

def test_percentage_formula():
    assert compute_severity_percentage(25, 100) == 25.0
    assert compute_severity_percentage(0, 10) == 0.0
    assert compute_severity_percentage(10, 10) == 100.0


@pytest.mark.parametrize('lesion,leaf', [(1, 0), (-1, 10), (11, 10)])
def test_percentage_rejects_invalid_areas(lesion, leaf):
    with pytest.raises(ValueError):
        compute_severity_percentage(lesion, leaf)


# ---------------------------------------------------------------- band boundaries

@pytest.mark.parametrize('pct,expected', [
    (0.0, CATEGORY_MILD), (9.99, CATEGORY_MILD), (10.0, CATEGORY_MODERATE), (17.5, CATEGORY_MODERATE),
    (25.0, CATEGORY_MODERATE), (25.01, CATEGORY_SEVERE), (100.0, CATEGORY_SEVERE),
    (None, CATEGORY_UNKNOWN), (float('nan'), CATEGORY_UNKNOWN),
])
def test_band_boundaries(pct, expected):
    assert classify_severity(pct) == expected


@pytest.mark.parametrize('pct', [-0.1, 100.1])
def test_band_rejects_out_of_range(pct):
    with pytest.raises(ValueError):
        classify_severity(pct)


def test_band_thresholds_are_configurable():
    bands = {'mild_upper': 5.0, 'moderate_upper': 15.0}
    assert classify_severity(6, bands) == CATEGORY_MODERATE
    assert classify_severity(16, bands) == CATEGORY_SEVERE


# ---------------------------------------------------------------- end-to-end estimator on synthetic leaves

@pytest.mark.parametrize('radii,expected_band', [
    ([18], CATEGORY_MILD),               # ~2 %
    ([30, 25, 18], CATEGORY_MODERATE),   # ~13-15 %
    ([45, 40, 38, 30], CATEGORY_SEVERE), # ~30 %+
])
def test_estimate_matches_ground_truth(radii, expected_band):
    centres = [(170, 180), (240, 215), (195, 250), (230, 160)]
    lesions = [(cx, cy, r, BROWN_RGB) for (cx, cy), r in zip(centres, radii)]
    img, leaf, lesion = make_leaf_image(lesions=lesions)
    truth = true_percentage(leaf, lesion)
    result = estimate_severity(img, class_label='Tomato___Early_blight')
    assert result.status == STATUS_ESTIMATED, result.warnings
    assert result.percentage == pytest.approx(truth, abs=1.5)
    assert result.category == expected_band == classify_severity(round(truth, 2))
    assert result.leaf_mask.shape == result.lesion_mask.shape == result.image.shape[:2]


def test_estimate_on_healthy_leaf_is_near_zero(healthy_leaf):
    img, _, _ = healthy_leaf
    result = estimate_severity(img, class_label='Tomato___Early_blight')
    assert result.status == STATUS_ESTIMATED
    assert result.percentage == 0.0
    assert any('No lesion pixels' in w for w in result.warnings)


def test_estimate_is_resolution_independent():
    lesions = [(170, 180, 30, BROWN_RGB), (240, 215, 25, BROWN_RGB)]
    small, _, _ = make_leaf_image(lesions=lesions)
    import cv2
    big = cv2.resize(small, (1200, 1200), interpolation=cv2.INTER_NEAREST)
    a = estimate_severity(small, 'Potato___Early_blight').percentage
    b = estimate_severity(big, 'Potato___Early_blight').percentage
    assert a == pytest.approx(b, abs=0.5)


def test_healthy_label_returns_not_applicable(diseased_leaf):
    img, _, _ = diseased_leaf
    result = estimate_severity(img, class_label='Tomato___healthy')
    assert result.status == STATUS_NOT_APPLICABLE
    assert result.category == CATEGORY_NOT_APPLICABLE and result.percentage is None


@pytest.mark.parametrize('label', ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                   'Orange___Haunglongbing_(Citrus_greening)'])
def test_systemic_diseases_not_applicable(label, diseased_leaf):
    img, _, _ = diseased_leaf
    result = estimate_severity(img, class_label=label)
    assert result.status == STATUS_NOT_APPLICABLE and result.percentage is None


def test_unreliable_segmentation_returns_no_number():
    img = np.full((300, 300, 3), (128, 128, 128), np.uint8)
    result = estimate_severity(img, class_label='Tomato___Early_blight')
    assert result.status == STATUS_UNRELIABLE
    assert result.category == CATEGORY_UNKNOWN and result.percentage is None
    assert result.warnings
    assert result.to_dict()['percentage'] is None


def test_too_small_image_is_unreliable():
    img, _, _ = make_leaf_image(size=40, leaf_axes=(15, 10))
    assert estimate_severity(img, 'Tomato___Early_blight').status == STATUS_UNRELIABLE


def test_errors_are_reported_not_hidden():
    result = estimate_severity(b'this is not an image', class_label='Tomato___Early_blight')
    assert result.status == STATUS_ERROR
    assert result.category == CATEGORY_UNKNOWN and result.percentage is None
    assert any('failed' in w for w in result.warnings)


def test_band_uses_rounded_percentage(monkeypatch, diseased_leaf):
    img, _, _ = diseased_leaf
    monkeypatch.setattr(estimator_module, 'compute_severity_percentage', lambda lesion, leaf: 9.996)
    result = estimate_severity(img, 'Tomato___Early_blight')
    assert result.percentage == 10.0 and result.category == CATEGORY_MODERATE


def test_to_dict_is_json_serialisable(diseased_leaf):
    import json
    img, _, _ = diseased_leaf
    d = estimate_severity(img, 'Tomato___Early_blight').to_dict()
    json.dumps(d)
    assert d['method'] == 'hsv_lesion_area_ratio' and d['experimental'] is True


# ---------------------------------------------------------------- configuration

def test_every_disease_class_has_a_profile_and_valid_colours():
    for label in CLASS_LABELS:
        if is_healthy(label):
            continue
        assert label in DISEASE_PROFILES, label
        profile = get_profile(label)
        for colour in profile['colors']:
            assert colour in LESION_COLOR_RANGES
        assert profile['supported'] == bool(profile['colors'])


def test_unknown_label_uses_default_profile():
    assert get_profile('Banana___Sigatoka')['name'] == 'default'


def test_json_override(tmp_path, diseased_leaf):
    import json
    override = tmp_path / 'override.json'
    override.write_text(json.dumps({'severity_bands': {'mild_upper': 1.0, 'moderate_upper': 5.0}}))
    config = load_config(str(override))
    assert config['severity_bands']['moderate_upper'] == 5.0
    assert config['leaf_segmentation']['min_saturation'] == 35   # untouched keys preserved
    img, _, _ = diseased_leaf
    assert estimate_severity(img, 'Tomato___Early_blight', config=config).category == CATEGORY_SEVERE


# ---------------------------------------------------------------- visualisation

def test_visualizations_are_data_uris(diseased_leaf):
    img, _, _ = diseased_leaf
    result = estimate_severity(img, 'Tomato___Early_blight')
    viz = build_visualizations(result)
    for key in ('original', 'leaf_mask', 'lesion_mask', 'overlay'):
        assert viz[key].startswith('data:image/')
    panel = render_panel(result, title='synthetic')
    assert panel.ndim == 3 and panel.shape[1] > 4 * result.image.shape[1]


def test_visualization_for_unreliable_result_has_no_lesion_mask():
    img = np.full((300, 300, 3), (128, 128, 128), np.uint8)
    viz = build_visualizations(estimate_severity(img, 'Tomato___Early_blight'))
    assert viz['lesion_mask'] is None and viz['original'].startswith('data:image/jpeg')


def test_overlay_marks_lesions_red(diseased_leaf):
    from severity.visualization import make_overlay
    img, _, _ = diseased_leaf
    result = estimate_severity(img, 'Tomato___Early_blight')
    overlay = make_overlay(result.image, result.leaf_mask, result.lesion_mask)
    lesion_pixels = overlay[result.lesion_mask > 0].astype(int)
    assert np.mean(lesion_pixels[:, 0] - lesion_pixels[:, 1]) > 60   # red channel dominates
    assert not math.isnan(float(overlay.mean()))


# ---------------------------------------------------------------- validation tool (Cohen's kappa)

def test_agreement_statistics():
    from severity.validate import agreement
    perfect = agreement(['Mild', 'Moderate', 'Severe'], ['Mild', 'Moderate', 'Severe'])
    assert perfect['exact_agreement'] == 1.0 and perfect['cohen_kappa'] == pytest.approx(1.0)
    mixed = agreement(['Mild', 'Mild', 'Severe', 'Moderate'], ['Mild', 'Severe', 'Severe', 'Mild'])
    assert mixed['exact_agreement'] == 0.5 and mixed['mild_severe_confusions'] == 1
    assert mixed['cohen_kappa_linear_weighted'] > mixed['cohen_kappa']   # partial credit for adjacent bands


def test_validation_template_and_score(tmp_path):
    from PIL import Image
    from severity.validate import score, write_template
    import csv
    folder = tmp_path / 'imgs'
    folder.mkdir()
    for i, radii in enumerate([[18], [30, 25, 18], [45, 40, 38, 30]]):
        centres = [(170, 180), (240, 215), (195, 250), (230, 160)]
        img, _, _ = make_leaf_image(lesions=[(cx, cy, r, BROWN_RGB) for (cx, cy), r in zip(centres, radii)])
        Image.fromarray(img).save(folder / ('leaf%d.png' % i))
    sheet = tmp_path / 'ann.csv'
    assert write_template(str(folder), 'Tomato___Early_blight', str(sheet), n=10) == 3
    rows = list(csv.DictReader(open(sheet)))
    for row, band in zip(rows, ['Mild', 'Moderate', 'Severe']):
        row['manual_category'] = band
    with open(sheet, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = score(str(sheet))
    assert stats['graded_by_annotator'] == 3 and stats['heuristic_coverage'] == 1.0
    assert stats['exact_agreement'] == 1.0
