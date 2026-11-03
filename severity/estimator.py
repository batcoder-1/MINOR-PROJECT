"""
Lesion-area-based severity estimation.

    severity_percentage = lesion_area / leaf_area * 100

where both areas are pixel counts of the binary masks produced by
``segmentation.py``. The percentage is mapped to Mild / Moderate / Severe using
configurable bands (``config.SEVERITY_BANDS``).

The estimator never invents a number: if the leaf cannot be segmented
reliably, or lesion-area ratio is not meaningful for the disease (viral /
systemic symptoms), the result carries an explicit status and ``percentage``
is ``None``.

Status values
-------------
estimated       percentage and category computed (still experimental)
unreliable      leaf segmentation failed its quality checks -> category 'Unknown'
not_applicable  healthy leaf, or disease whose severity is not lesion-area based
error           unexpected failure (exception message in warnings)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from labels import is_healthy
from . import segmentation as seg
from .config import DEFAULT_CONFIG, get_profile

logger = logging.getLogger(__name__)

METHOD = 'hsv_lesion_area_ratio'

STATUS_ESTIMATED = 'estimated'
STATUS_UNRELIABLE = 'unreliable'
STATUS_NOT_APPLICABLE = 'not_applicable'
STATUS_ERROR = 'error'

CATEGORY_MILD = 'Mild'
CATEGORY_MODERATE = 'Moderate'
CATEGORY_SEVERE = 'Severe'
CATEGORY_UNKNOWN = 'Unknown'
CATEGORY_NOT_APPLICABLE = 'Not applicable'


def compute_severity_percentage(lesion_area, leaf_area):
    """lesion_area / leaf_area * 100. Raises ValueError for an empty or inconsistent leaf."""
    if leaf_area <= 0:
        raise ValueError('leaf_area must be positive')
    if lesion_area < 0 or lesion_area > leaf_area:
        raise ValueError('lesion_area must be within [0, leaf_area]')
    return 100.0 * float(lesion_area) / float(leaf_area)


def classify_severity(percentage, bands=None):
    """Map a lesion percentage to a band.

    Mild: p < mild_upper | Moderate: mild_upper <= p <= moderate_upper | Severe: p > moderate_upper
    """
    bands = bands or DEFAULT_CONFIG['severity_bands']
    if percentage is None or not isinstance(percentage, (int, float)) or math.isnan(percentage):
        return CATEGORY_UNKNOWN
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be within [0, 100], got %r' % (percentage,))
    if percentage < bands['mild_upper']:
        return CATEGORY_MILD
    if percentage <= bands['moderate_upper']:
        return CATEGORY_MODERATE
    return CATEGORY_SEVERE


@dataclass
class SeverityResult:
    status: str
    category: str
    percentage: Optional[float] = None
    lesion_area_px: Optional[int] = None
    leaf_area_px: Optional[int] = None
    profile: str = 'default'
    lesion_colors: List[str] = field(default_factory=list)
    validity: str = 'heuristic'
    note: str = ''
    warnings: List[str] = field(default_factory=list)
    quality: Dict[str, float] = field(default_factory=dict)
    color_breakdown_px: Dict[str, int] = field(default_factory=dict)
    lesion_components: Optional[int] = None
    bands: Dict[str, float] = field(default_factory=dict)
    # arrays (not serialised): working-resolution RGB image and masks (uint8 0/255)
    image: Optional[np.ndarray] = field(default=None, repr=False)
    leaf_mask: Optional[np.ndarray] = field(default=None, repr=False)
    lesion_mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def reliable(self):
        return self.status == STATUS_ESTIMATED

    def to_dict(self):
        return {
            'status': self.status,
            'category': self.category,
            'percentage': self.percentage,
            'lesion_area_px': self.lesion_area_px,
            'leaf_area_px': self.leaf_area_px,
            'method': METHOD,
            'experimental': True,
            'profile': self.profile,
            'lesion_colors': list(self.lesion_colors),
            'validity': self.validity,
            'note': self.note,
            'bands': dict(self.bands),
            'quality': dict(self.quality),
            'color_breakdown_px': dict(self.color_breakdown_px),
            'lesion_components': self.lesion_components,
            'warnings': list(self.warnings),
        }


def estimate_severity(image, class_label=None, config=None):
    """Estimate disease severity for one leaf image.

    ``image``       : anything ``segmentation.load_image_rgb`` accepts (path, bytes, PIL, ndarray)
    ``class_label`` : predicted PlantVillage label; selects the disease-specific colour profile
    ``config``      : full config dict (``config.load_config()``); defaults to DEFAULT_CONFIG
    """
    config = config or DEFAULT_CONFIG
    profile = get_profile(class_label, config)
    base = dict(profile=profile['name'], lesion_colors=list(profile['colors']),
                validity=profile['validity'], note=profile.get('note', ''),
                bands=dict(config['severity_bands']))

    if class_label and is_healthy(class_label):
        return SeverityResult(status=STATUS_NOT_APPLICABLE, category=CATEGORY_NOT_APPLICABLE,
                              warnings=['Leaf classified as healthy; severity estimation skipped.'], **base)
    if not profile['supported']:
        return SeverityResult(status=STATUS_NOT_APPLICABLE, category=CATEGORY_NOT_APPLICABLE,
                              warnings=[profile.get('note') or 'Severity estimation is not supported for this disease.'],
                              **base)
    try:
        return _estimate(image, profile, config, base)
    except Exception as exc:  # explicit failure, never a fabricated number
        logger.exception('Severity estimation failed')
        return SeverityResult(status=STATUS_ERROR, category=CATEGORY_UNKNOWN,
                              warnings=['Severity estimation failed: %s' % exc], **base)


def _estimate(image, profile, config, base):
    pre = config['preprocessing']
    rgb = seg.load_image_rgb(image)
    h0, w0 = rgb.shape[:2]
    if min(h0, w0) < pre['min_image_side']:
        return SeverityResult(status=STATUS_UNRELIABLE, category=CATEGORY_UNKNOWN, image=rgb,
                              warnings=['Image is too small (%dx%d px) for reliable segmentation.' % (w0, h0)],
                              **base)
    rgb, scale = seg.resize_for_segmentation(rgb, pre['working_side'])
    h, w = rgb.shape[:2]

    hsv = seg.preprocess(rgb, pre['blur_kernel'])
    leaf = seg.segment_leaf(hsv, config['leaf_segmentation'])
    quality = dict(leaf.quality, input_size=[w0, h0], working_size=[w, h], resize_scale=round(scale, 4))
    warnings = list(leaf.warnings)
    leaf_px = int(np.count_nonzero(leaf.mask))

    if not leaf.reliable or leaf_px == 0:
        warnings.append('Leaf segmentation is unreliable, so no severity percentage is reported.')
        return SeverityResult(status=STATUS_UNRELIABLE, category=CATEGORY_UNKNOWN, leaf_area_px=leaf_px or None,
                              quality=quality, warnings=warnings, image=rgb, leaf_mask=leaf.mask, **base)

    lesion_cfg = dict(config['lesion_segmentation'])
    lesion_cfg.update(profile.get('overrides') or {})
    lesions = seg.segment_lesions(hsv, leaf.mask, profile['colors'], config['lesion_color_ranges'], lesion_cfg)
    lesion_px = int(np.count_nonzero(lesions.mask))

    # Round first and classify the rounded value, so the displayed number and
    # the band can never disagree (e.g. 9.996 shown as 10.0 but labelled Mild).
    percentage = round(compute_severity_percentage(lesion_px, leaf_px), 2)
    category = classify_severity(percentage, config['severity_bands'])

    if lesion_px == 0 and lesion_cfg.get('warn_zero_lesion_for_diseased', True):
        warnings.append('No lesion pixels matched the colour profile. The infection may be very early, '
                        'or the lesion colours fall outside the configured ranges.')
    if percentage > lesion_cfg.get('warn_severity_above', 100):
        warnings.append('Very high lesion fraction (%.1f%%); check the overlay before relying on it.' % percentage)
    if profile['validity'] == 'low':
        warnings.append('Low-validity profile: %s' % profile.get('note', ''))

    return SeverityResult(status=STATUS_ESTIMATED, category=category, percentage=percentage,
                          lesion_area_px=lesion_px, leaf_area_px=leaf_px, quality=quality,
                          color_breakdown_px=lesions.color_breakdown_px, lesion_components=lesions.components,
                          warnings=warnings, image=rgb, leaf_mask=leaf.mask, lesion_mask=lesions.mask, **base)
