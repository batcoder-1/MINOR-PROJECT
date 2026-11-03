import io

import cv2
import numpy as np
import pytest
from PIL import Image

from severity import segmentation as seg
from tests.conftest import make_leaf_image


# ---------------------------------------------------------------- classifier pre-processing (app.py)

def test_classifier_preprocessing_shape_and_range(app_module, tmp_path, diseased_leaf):
    img, _, _ = diseased_leaf
    path = tmp_path / 'leaf.png'
    Image.fromarray(img).save(path)
    x = app_module.preprocess_image(str(path))
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


@pytest.mark.parametrize('mode', ['RGBA', 'L', 'P'])
def test_classifier_preprocessing_converts_modes(app_module, tmp_path, mode):
    img = Image.new('RGB', (300, 200), (10, 200, 30)).convert(mode)
    path = tmp_path / ('img_%s.png' % mode)
    img.save(path)
    assert app_module.preprocess_image(str(path)).shape == (1, 224, 224, 3)


def test_classifier_preprocessing_matches_original_formula(app_module, tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / 'noise.png'
    Image.fromarray(arr).save(path)
    expected = np.asarray(Image.open(path).convert('RGB').resize((224, 224)), dtype=np.float32) / 255.0
    np.testing.assert_array_equal(app_module.preprocess_image(str(path))[0], expected)


# ---------------------------------------------------------------- segmentation pre-processing

def test_load_image_rgb_from_various_sources(tmp_path):
    arr = np.full((50, 60, 3), (200, 10, 10), dtype=np.uint8)
    path = tmp_path / 'red.png'
    Image.fromarray(arr).save(path)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    for src in (str(path), buf.getvalue(), Image.fromarray(arr), arr):
        out = seg.load_image_rgb(src)
        assert out.shape == (50, 60, 3) and out.dtype == np.uint8
        assert tuple(out[0, 0]) == (200, 10, 10)


def test_load_image_rgb_drops_alpha_and_expands_gray():
    assert seg.load_image_rgb(np.zeros((10, 10, 4), np.uint8)).shape == (10, 10, 3)
    assert seg.load_image_rgb(np.zeros((10, 10), np.uint8)).shape == (10, 10, 3)
    with pytest.raises(ValueError):
        seg.load_image_rgb(np.zeros((10, 10, 2), np.uint8))


def test_load_image_rgb_applies_exif_orientation():
    img = Image.new('RGB', (40, 20), (0, 255, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 deg clockwise on display
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif)
    assert seg.load_image_rgb(buf.getvalue()).shape[:2] == (40, 20)  # (H, W) swapped


@pytest.mark.parametrize('shape,expected', [((256, 256), (512, 512)), ((1000, 2000), (256, 512)), ((512, 300), (512, 300))])
def test_resize_to_working_side(shape, expected):
    out, scale = seg.resize_for_segmentation(np.zeros(shape + (3,), np.uint8), 512)
    assert out.shape[:2] == expected
    assert scale == pytest.approx(512 / max(shape))


@pytest.mark.parametrize('rgb,hue', [((255, 0, 0), 0), ((0, 255, 0), 60), ((0, 0, 255), 120), ((255, 255, 0), 30)])
def test_rgb_to_hsv_opencv_convention(rgb, hue):
    hsv = seg.rgb_to_hsv(np.full((2, 2, 3), rgb, np.uint8))
    assert hsv[0, 0, 0] == hue and hsv[0, 0, 1] == 255 and hsv[0, 0, 2] == 255


def test_otsu_matches_opencv_and_reports_separability():
    img, _, _ = make_leaf_image()
    sat = seg.rgb_to_hsv(img)[..., 1]
    t, eta = seg.otsu_threshold(sat)
    t_cv, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert t == int(t_cv)
    assert eta > 0.99        # two flat regions: perfectly bimodal
    assert seg.otsu_threshold(np.full((20, 20), 7, np.uint8)) == (0, 0.0)


def test_hue_wraparound_range():
    hue = np.array([[0, 5, 90, 170, 179]], dtype=np.uint8)
    assert seg.hue_in_ranges(hue, [(165, 4)]).tolist() == [[True, False, False, True, True]]
