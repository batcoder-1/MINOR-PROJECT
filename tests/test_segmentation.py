"""Deterministic leaf / lesion segmentation tests on synthetic images with known areas."""

import cv2
import numpy as np

from severity import segmentation as seg
from severity.config import load_config
from tests.conftest import BROWN_RGB, DARK_RGB, make_leaf_image

CFG = load_config()


def _hsv(img):
    return seg.preprocess(img, CFG['preprocessing']['blur_kernel'])


def _iou(a, b):
    a, b = a > 0, b > 0
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def test_leaf_mask_matches_drawn_leaf(healthy_leaf):
    img, leaf_truth, _ = healthy_leaf
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert leaf.reliable, leaf.warnings
    assert _iou(leaf.mask, leaf_truth) > 0.98
    assert 0.30 < leaf.quality['leaf_fraction'] < 0.35


def test_leaf_mask_includes_lesions_inside_leaf(diseased_leaf):
    img, leaf_truth, _ = diseased_leaf
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert _iou(leaf.mask, leaf_truth) > 0.98


def test_hole_filling_recovers_low_saturation_lesions():
    # black and grey spots have ~0 saturation, so they are holes in the Otsu seed
    img, leaf_truth, _ = make_leaf_image(lesions=[(200, 200, 35, DARK_RGB), (150, 170, 20, (150, 150, 150))])
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert _iou(leaf.mask, leaf_truth) > 0.98


def test_morphological_opening_removes_salt_noise(healthy_leaf):
    img, leaf_truth, _ = healthy_leaf
    noisy = img.copy()
    rng = np.random.default_rng(1)
    ys, xs = rng.integers(0, 400, 600), rng.integers(0, 400, 600)
    noisy[ys, xs] = (0, 255, 0)   # isolated saturated pixels in the background
    leaf = seg.segment_leaf(_hsv(noisy), CFG['leaf_segmentation'])
    assert leaf.reliable
    assert _iou(leaf.mask, leaf_truth) > 0.97


def test_largest_contour_selected_and_background_blob_excluded():
    def blob(img):
        cv2.circle(img, (40, 40), 20, BROWN_RGB, -1)   # saturated brown blob on the background
    img, leaf_truth, _ = make_leaf_image(extra=blob)
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert leaf.quality['candidate_regions'] >= 2
    assert leaf.mask[40, 40] == 0
    assert _iou(leaf.mask, leaf_truth) > 0.98


def test_uniform_image_is_unreliable():
    img = np.full((300, 300, 3), (128, 128, 128), np.uint8)
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert not leaf.reliable
    assert leaf.warnings


def test_tiny_leaf_is_unreliable():
    img, _, _ = make_leaf_image(leaf_axes=(20, 15))
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert not leaf.reliable
    assert any('only' in w for w in leaf.warnings)


def test_full_frame_foliage_is_unreliable():
    # leaf colour everywhere (like a field photo full of foliage): background cannot be separated
    img = np.full((300, 300, 3), (60, 140, 50), np.uint8)
    img[100:200, 100:200] = (150, 160, 60)
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert not leaf.reliable


def test_blue_background_hue_excluded():
    img, leaf_truth, _ = make_leaf_image(background=(70, 90, 160))   # saturated blue backdrop
    leaf = seg.segment_leaf(_hsv(img), CFG['leaf_segmentation'])
    assert leaf.reliable
    assert _iou(leaf.mask, leaf_truth) > 0.97


def test_lesion_mask_area_matches_truth(diseased_leaf):
    img, _, lesion_truth = diseased_leaf
    hsv = _hsv(img)
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    lesions = seg.segment_lesions(hsv, leaf.mask, ['brown', 'dark', 'yellow'], CFG['lesion_color_ranges'],
                                  CFG['lesion_segmentation'])
    assert _iou(lesions.mask, lesion_truth) > 0.9
    assert lesions.components == 3
    # blur mixes brown lesion edges with green leaf into yellow-ish hues; those pixels lie on the lesion
    # boundary (IoU above already passed) and must stay a small fraction of the brown pixels
    assert lesions.color_breakdown_px['brown'] > 0
    assert lesions.color_breakdown_px['yellow'] < 0.1 * lesions.color_breakdown_px['brown']


def test_lesion_mask_never_leaves_leaf_and_excludes_background():
    def blob(img):
        cv2.circle(img, (40, 40), 20, BROWN_RGB, -1)
    img, _, _ = make_leaf_image(lesions=[(200, 200, 25, BROWN_RGB)], extra=blob)
    hsv = _hsv(img)
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    lesions = seg.segment_lesions(hsv, leaf.mask, ['brown'], CFG['lesion_color_ranges'], CFG['lesion_segmentation'])
    assert np.all(lesions.mask[leaf.mask == 0] == 0)
    assert lesions.mask[40, 40] == 0


def test_healthy_leaf_has_no_lesions(healthy_leaf):
    img, _, _ = healthy_leaf
    hsv = _hsv(img)
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    lesions = seg.segment_lesions(hsv, leaf.mask, ['brown', 'dark', 'yellow'], CFG['lesion_color_ranges'],
                                  CFG['lesion_segmentation'])
    assert np.count_nonzero(lesions.mask) == 0


def test_small_noise_components_removed(healthy_leaf):
    img, _, _ = healthy_leaf
    speckled = img.copy()
    for x, y in [(180, 190), (220, 210), (200, 230)]:
        speckled[y, x] = BROWN_RGB   # single-pixel "lesions"
    hsv = seg.rgb_to_hsv(speckled)   # no blur, so only morphology/size filtering can remove them
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    lesions = seg.segment_lesions(hsv, leaf.mask, ['brown'], CFG['lesion_color_ranges'], CFG['lesion_segmentation'])
    assert np.count_nonzero(lesions.mask) == 0


def test_disease_specific_colours_change_result():
    yellow = (210, 190, 40)   # chlorotic yellow: H ~ 27
    img, _, _ = make_leaf_image(lesions=[(200, 200, 30, yellow)])
    hsv = _hsv(img)
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    with_yellow = seg.segment_lesions(hsv, leaf.mask, ['yellow'], CFG['lesion_color_ranges'], CFG['lesion_segmentation'])
    without = seg.segment_lesions(hsv, leaf.mask, ['brown', 'dark'], CFG['lesion_color_ranges'], CFG['lesion_segmentation'])
    assert np.count_nonzero(with_yellow.mask) > 2000
    assert np.count_nonzero(without.mask) == 0


def test_unknown_colour_name_raises(healthy_leaf):
    import pytest
    img, _, _ = healthy_leaf
    hsv = _hsv(img)
    leaf = seg.segment_leaf(hsv, CFG['leaf_segmentation'])
    with pytest.raises(KeyError):
        seg.segment_lesions(hsv, leaf.mask, ['magenta'], CFG['lesion_color_ranges'], CFG['lesion_segmentation'])
