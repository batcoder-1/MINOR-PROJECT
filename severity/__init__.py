"""
HSV / OpenCV based disease-severity estimation (experimental).

    from severity import estimate_severity
    result = estimate_severity('leaf.jpg', class_label='Tomato___Late_blight')
    result.status, result.category, result.percentage
"""

from .config import DEFAULT_CONFIG, get_profile, load_config
from .estimator import (
    CATEGORY_MILD, CATEGORY_MODERATE, CATEGORY_NOT_APPLICABLE, CATEGORY_SEVERE, CATEGORY_UNKNOWN,
    METHOD, STATUS_ERROR, STATUS_ESTIMATED, STATUS_NOT_APPLICABLE, STATUS_UNRELIABLE,
    SeverityResult, classify_severity, compute_severity_percentage, estimate_severity,
)
from .visualization import build_visualizations, render_panel, save_visualizations

__all__ = [
    'DEFAULT_CONFIG', 'get_profile', 'load_config',
    'SeverityResult', 'classify_severity', 'compute_severity_percentage', 'estimate_severity',
    'build_visualizations', 'render_panel', 'save_visualizations', 'METHOD',
    'STATUS_ESTIMATED', 'STATUS_UNRELIABLE', 'STATUS_NOT_APPLICABLE', 'STATUS_ERROR',
    'CATEGORY_MILD', 'CATEGORY_MODERATE', 'CATEGORY_SEVERE', 'CATEGORY_UNKNOWN', 'CATEGORY_NOT_APPLICABLE',
]
