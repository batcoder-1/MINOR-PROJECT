"""
Visualisations of the severity pipeline, generated from the actual uploaded
image and the masks computed for it (nothing here is a stored or sample image):

    original      working-resolution RGB image the masks were computed on
    leaf_mask     white = leaf pixels (denominator of the severity ratio)
    lesion_mask   white = lesion pixels (numerator)
    overlay       lesions tinted red, leaf outline drawn in yellow

``build_visualizations`` returns base64 data URIs for the web UI;
``render_panel`` / ``save_visualizations`` write PNGs for reports.
"""

import base64
import os

import cv2
import numpy as np

LESION_RGB = (230, 30, 30)
LEAF_OUTLINE_RGB = (255, 215, 0)


def mask_to_rgb(mask):
    if mask is None:
        return None
    return np.repeat((mask > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)


def make_overlay(rgb, leaf_mask=None, lesion_mask=None, alpha=0.55):
    """Blend lesion pixels with red and outline the leaf contour."""
    out = rgb.copy()
    if lesion_mask is not None and np.any(lesion_mask):
        sel = lesion_mask > 0
        tint = np.array(LESION_RGB, dtype=np.float32)
        out[sel] = (out[sel].astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)
    if leaf_mask is not None and np.any(leaf_mask):
        contours, _ = cv2.findContours((leaf_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(1, int(round(max(rgb.shape[:2]) / 256)))
        cv2.drawContours(out, contours, -1, LEAF_OUTLINE_RGB, thickness)
    return out


def encode_data_uri(rgb, fmt='png', jpeg_quality=85):
    """Encode an RGB image as a data URI string."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == 'jpeg':
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        mime = 'image/jpeg'
    else:
        ok, buf = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        mime = 'image/png'
    if not ok:
        raise RuntimeError('Image encoding failed')
    return 'data:%s;base64,%s' % (mime, base64.b64encode(buf.tobytes()).decode('ascii'))


def build_visualizations(result):
    """Data URIs for the four views of a SeverityResult (missing views are None)."""
    if result.image is None:
        return None
    overlay = make_overlay(result.image, result.leaf_mask, result.lesion_mask)
    return {
        'original': encode_data_uri(result.image, 'jpeg'),
        'leaf_mask': encode_data_uri(mask_to_rgb(result.leaf_mask)) if result.leaf_mask is not None else None,
        'lesion_mask': encode_data_uri(mask_to_rgb(result.lesion_mask)) if result.lesion_mask is not None else None,
        'overlay': encode_data_uri(overlay, 'jpeg'),
        'legend': {'lesion': 'red tint', 'leaf_outline': 'yellow line'},
        'source': 'Generated from the uploaded image by HSV segmentation (experimental); not a CNN output.',
    }


def _caption(img, text):
    bar_h = 26
    bar = np.full((bar_h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def render_panel(result, title=None):
    """1x4 panel: original | leaf mask | lesion mask | overlay, with captions."""
    if result.image is None:
        raise ValueError('SeverityResult has no image to visualise (status=%s)' % result.status)
    h, w = result.image.shape[:2]
    blank = np.full((h, w, 3), 235, dtype=np.uint8)
    tiles = [
        _caption(result.image, '1. Original'),
        _caption(mask_to_rgb(result.leaf_mask) if result.leaf_mask is not None else blank, '2. Leaf mask'),
        _caption(mask_to_rgb(result.lesion_mask) if result.lesion_mask is not None else blank, '3. Lesion mask'),
        _caption(make_overlay(result.image, result.leaf_mask, result.lesion_mask), '4. Overlay'),
    ]
    gap = np.full((tiles[0].shape[0], 6, 3), 255, dtype=np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.hstack([row, gap, tile])
    if result.percentage is not None:
        summary = '%s: %.2f%% lesion / leaf area' % (result.category, result.percentage)
    else:
        summary = 'Severity %s (status: %s)' % (result.category, result.status)
    header = (title + '  |  ' if title else '') + summary + '  [experimental HSV heuristic]'
    return _caption(row, header)


def save_visualizations(result, out_dir, prefix='severity'):
    """Write panel + individual views as PNG files; returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    def _write(name, rgb):
        path = os.path.join(out_dir, '%s_%s.png' % (prefix, name))
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        written[name] = path

    if result.image is None:
        return written
    _write('original', result.image)
    if result.leaf_mask is not None:
        _write('leaf_mask', mask_to_rgb(result.leaf_mask))
    if result.lesion_mask is not None:
        _write('lesion_mask', mask_to_rgb(result.lesion_mask))
    _write('overlay', make_overlay(result.image, result.leaf_mask, result.lesion_mask))
    _write('panel', render_panel(result, title=prefix))
    return written
