"""
Leaf and lesion segmentation in HSV colour space (OpenCV).

Pipeline
--------
1. load_image_rgb / resize_for_segmentation / preprocess
       RGB uint8 -> optional Gaussian blur -> HSV (OpenCV ranges H 0-179, S/V 0-255)
2. segment_leaf
       Otsu threshold on S (clamped) AND V >= min_value -> binary seed
       -> morphological opening (remove specks) -> closing (bridge gaps)
       -> largest external contour -> filled leaf mask -> reliability checks
3. segment_lesions
       union of disease-specific HSV boxes, restricted to the (slightly eroded)
       leaf mask -> opening / closing -> drop tiny connected components

All functions are pure (no global state) so they can be unit-tested with
synthetic images.
"""

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Step 1: pre-processing
# ---------------------------------------------------------------------------

def load_image_rgb(source):
    """Load an image as an RGB uint8 array of shape (H, W, 3).

    ``source`` may be a file path, raw bytes, a file-like object, a PIL image
    or a NumPy array (assumed RGB / RGBA / grayscale). EXIF orientation is
    applied so masks line up with what the browser shows.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError('Expected an image array of shape (H, W, 3), got %r' % (arr.shape,))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert('RGB'), dtype=np.uint8)


def resize_for_segmentation(rgb, working_side):
    """Rescale so the longest side equals ``working_side``. Returns (image, scale).

    Area interpolation is used for down-scaling and bilinear for up-scaling.
    Severity is a ratio of pixel counts, so it is (approximately) unaffected
    by the rescaling; the point is to make kernel sizes resolution-independent.
    """
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if not working_side or longest == working_side:
        return rgb, 1.0
    scale = working_side / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgb, size, interpolation=interp), scale


def rgb_to_hsv(rgb):
    """RGB uint8 -> OpenCV HSV uint8 (H in [0, 179])."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def preprocess(rgb, blur_kernel=5):
    """Optional Gaussian blur (noise suppression) followed by RGB -> HSV."""
    if blur_kernel and blur_kernel > 1:
        k = int(blur_kernel) | 1  # force odd
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)
    return rgb_to_hsv(rgb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def otsu_threshold(channel):
    """Otsu's method on an 8-bit channel.

    Returns (threshold, separability) where pixels ``> threshold`` form the
    foreground class (same convention as cv2.THRESH_OTSU) and

        separability (eta) = sigma_B^2(t*) / sigma_T^2   in [0, 1]

    is Otsu's own goodness measure: close to 1 for a cleanly bimodal
    histogram, close to 0 when there is no meaningful split.
    """
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0, 0.0
    p = hist / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)                 # P(class 0) for threshold t (values <= t)
    mu = np.cumsum(p * levels)           # cumulative mean
    mu_total = mu[-1]
    sigma_total = float(((levels - mu_total) ** 2 * p).sum())
    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_between = np.where(denom > 0, (mu_total * omega - mu) ** 2 / denom, 0.0)
    threshold = int(np.argmax(sigma_between))
    separability = float(sigma_between[threshold] / sigma_total) if sigma_total > 0 else 0.0
    return threshold, separability


def _kernel(size):
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morph_open_close(mask, open_kernel, open_iter, close_kernel, close_iter):
    """Opening removes isolated foreground specks; closing fills small gaps/holes."""
    if open_kernel and open_kernel > 1 and open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(open_kernel), iterations=int(open_iter))
    if close_kernel and close_kernel > 1 and close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(close_kernel), iterations=int(close_iter))
    return mask


def remove_small_components(mask, min_area):
    """Drop 8-connected components with fewer than ``min_area`` pixels."""
    if not min_area or min_area <= 1:
        return mask, int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    cleaned = np.where(keep[labels], 255, 0).astype(np.uint8)
    return cleaned, int(keep.sum())


def hue_in_ranges(hue, ranges):
    """Boolean mask of hue values inside any inclusive (lo, hi) range (wrap-around if lo > hi)."""
    out = np.zeros(hue.shape, dtype=bool)
    for lo, hi in ranges:
        if lo <= hi:
            out |= (hue >= lo) & (hue <= hi)
        else:
            out |= (hue >= lo) | (hue <= hi)
    return out


def hsv_range_mask(hsv, boxes):
    """Union of HSV boxes. A box whose h_min > h_max wraps around hue 179 -> 0."""
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for box in boxes:
        h_lo, h_hi = box['h']
        s_lo, s_hi = box['s']
        v_lo, v_hi = box['v']
        if h_lo <= h_hi:
            out |= cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
        else:
            out |= cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, s_hi, v_hi))
            out |= cv2.inRange(hsv, (0, s_lo, v_lo), (h_hi, s_hi, v_hi))
    return out


# ---------------------------------------------------------------------------
# Step 2: leaf segmentation
# ---------------------------------------------------------------------------

@dataclass
class LeafSegmentation:
    mask: np.ndarray                      # uint8 {0, 255}
    reliable: bool
    warnings: List[str] = field(default_factory=list)
    quality: Dict[str, float] = field(default_factory=dict)
    contour: Optional[np.ndarray] = None


def segment_leaf(hsv, cfg):
    """Segment the dominant leaf. ``cfg`` is config['leaf_segmentation']."""
    h, w = hsv.shape[:2]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    warnings = []
    quality = {}

    otsu_t, separability = otsu_threshold(sat)
    threshold = int(np.clip(otsu_t, cfg['min_saturation'], cfg['max_saturation']))
    quality.update({
        'otsu_threshold_raw': int(otsu_t),
        'saturation_threshold_used': threshold,
        'otsu_separability': round(separability, 4),
        'saturation_std': round(float(sat.std()), 3),
    })
    if threshold != otsu_t:
        warnings.append('Otsu saturation threshold %d was clamped to %d.' % (otsu_t, threshold))

    hue = hsv[..., 0]
    not_background_hue = ~hue_in_ranges(hue, cfg.get('exclude_hue_ranges', []))
    strong = (sat > threshold) & (val >= cfg['min_value']) & not_background_hue
    seed = strong
    if cfg.get('weak_seed_enabled', False):
        weak = ((sat > cfg['weak_min_saturation']) & (val >= cfg['min_value'])
                & (val <= cfg['weak_max_value']) & hue_in_ranges(hue, cfg['weak_hue_ranges']))
        seed = strong | weak
        quality['weak_seed_added_fraction'] = round(float(np.count_nonzero(weak & ~strong)) / weak.size, 4)
    quality['strong_seed_fraction'] = round(float(np.count_nonzero(strong)) / strong.size, 4)
    seed = seed.astype(np.uint8) * 255
    seed = morph_open_close(seed, cfg['open_kernel'], cfg['open_iterations'],
                            cfg['close_kernel'], cfg['close_iterations'])

    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros((h, w), dtype=np.uint8)
    if not contours:
        warnings.append('No leaf-like (saturated) region was found in the image.')
        quality['leaf_fraction'] = 0.0
        return LeafSegmentation(empty, False, warnings, quality)

    largest = max(contours, key=cv2.contourArea)
    leaf = empty.copy()
    cv2.drawContours(leaf, [largest], -1, 255, thickness=cv2.FILLED)
    if not cfg.get('fill_holes', True):
        leaf = cv2.bitwise_and(leaf, seed)

    leaf_px = int(np.count_nonzero(leaf))
    leaf_fraction = leaf_px / float(h * w)
    contour_area = float(cv2.contourArea(largest))
    hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    border = np.concatenate([leaf[0, :], leaf[-1, :], leaf[:, 0], leaf[:, -1]])
    border_contact = float(np.count_nonzero(border)) / border.size

    quality.update({
        'leaf_fraction': round(leaf_fraction, 4),
        'leaf_solidity': round(solidity, 4),
        'border_contact': round(border_contact, 4),
        'candidate_regions': len(contours),
    })

    reliable = True
    if quality['saturation_std'] < cfg['min_saturation_std']:
        reliable = False
        warnings.append('Image has almost no saturation contrast; leaf and background cannot be separated.')
    if separability < cfg['min_otsu_separability']:
        reliable = False
        warnings.append('Saturation histogram is not clearly bimodal (Otsu separability %.2f < %.2f).'
                        % (separability, cfg['min_otsu_separability']))
    if leaf_fraction < cfg['min_leaf_fraction']:
        reliable = False
        warnings.append('Detected leaf covers only %.1f%% of the image (minimum %.0f%%).'
                        % (leaf_fraction * 100, cfg['min_leaf_fraction'] * 100))
    if leaf_fraction > cfg['max_leaf_fraction']:
        reliable = False
        warnings.append('Leaf mask covers %.1f%% of the image; the background was probably not separated.'
                        % (leaf_fraction * 100))
    if border_contact > cfg.get('max_border_contact', 1.0):
        reliable = False
        warnings.append('Leaf mask touches %.0f%% of the image border; the leaf could not be separated from '
                        'surrounding foliage or background (typical of field photos).' % (border_contact * 100))
    elif border_contact > cfg['warn_border_contact_above']:
        warnings.append('Leaf touches %.0f%% of the image border; the leaf may be cropped or merged with background.'
                        % (border_contact * 100))
    if solidity < cfg['warn_solidity_below']:
        warnings.append('Leaf outline is irregular (solidity %.2f); overlapping leaves or background may be included.'
                        % solidity)

    return LeafSegmentation(leaf, reliable, warnings, quality, largest)


# ---------------------------------------------------------------------------
# Step 3: lesion segmentation
# ---------------------------------------------------------------------------

@dataclass
class LesionSegmentation:
    mask: np.ndarray                      # uint8 {0, 255}, subset of the leaf mask
    components: int
    color_breakdown_px: Dict[str, int] = field(default_factory=dict)


def segment_lesions(hsv, leaf_mask, colors, color_ranges, cfg):
    """Lesion mask = union of the selected HSV colour ranges inside the leaf.

    ``colors``       : names from config['lesion_color_ranges'] for this disease
    ``color_ranges`` : config['lesion_color_ranges']
    ``cfg``          : config['lesion_segmentation'] (with per-disease overrides applied)
    """
    unknown = [c for c in colors if c not in color_ranges]
    if unknown:
        raise KeyError('Unknown lesion colour range(s): %s' % ', '.join(unknown))

    search = leaf_mask
    erosion = int(cfg.get('boundary_erosion_px', 0))
    if erosion > 0:
        search = cv2.erode(leaf_mask, _kernel(2 * erosion + 1), iterations=1)

    per_color = {}
    raw = np.zeros(leaf_mask.shape, dtype=np.uint8)
    for name in colors:
        m = hsv_range_mask(hsv, color_ranges[name])
        per_color[name] = m
        raw |= m
    raw = cv2.bitwise_and(raw, search)

    cleaned = morph_open_close(raw, cfg['open_kernel'], cfg['open_iterations'],
                               cfg['close_kernel'], cfg['close_iterations'])
    cleaned, components = remove_small_components(cleaned, cfg['min_component_area_px'])
    cleaned = cv2.bitwise_and(cleaned, leaf_mask)   # closing must never leave the leaf

    breakdown = {name: int(np.count_nonzero(cv2.bitwise_and(m, cleaned))) for name, m in per_color.items()}
    return LesionSegmentation(cleaned, components, breakdown)
