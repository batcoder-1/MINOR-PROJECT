"""
Configuration for the HSV-based severity estimator.

Everything that is a tunable number lives here, so thresholds can be adjusted
without touching the algorithm code. All values are PRELIMINARY and
EXPERIMENTAL: they were chosen by inspecting PlantVillage-style images (single
leaf, plain background) and have not been validated against expert severity
ratings. See EXPERIMENTAL_METHODOLOGY.md for the validation plan.

Colour convention: OpenCV 8-bit HSV, i.e. H in [0, 179] (degrees / 2),
S in [0, 255], V in [0, 255].

An optional JSON file can override any value without editing this file:
set the environment variable SEVERITY_CONFIG=/path/to/override.json, where the
JSON has the same top-level keys as ``DEFAULT_CONFIG`` (nested dicts are merged).
"""

import copy
import json
import os

# ---------------------------------------------------------------------------
# Severity bands (percentage of leaf area covered by lesions)
#   Mild      : 0  <= p < 10
#   Moderate  : 10 <= p <= 25
#   Severe    : 25 <  p <= 100
# These cut-offs come from the project proposal; they are NOT a clinically
# validated universal standard and should be re-tuned per disease once
# expert-graded images are available.
# ---------------------------------------------------------------------------
SEVERITY_BANDS = {
    'mild_upper': 10.0,       # p < mild_upper            -> Mild
    'moderate_upper': 25.0,   # mild_upper <= p <= this   -> Moderate, above -> Severe
}

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
PREPROCESSING = {
    # The longest side is rescaled to exactly this many px (up- or down-scaled)
    # so that morphology kernel sizes below mean the same thing for a 256 px
    # PlantVillage image and a 4000 px phone photo.
    'working_side': 512,
    'min_image_side': 64,      # smaller inputs (before rescaling) are rejected
    'blur_kernel': 5,          # Gaussian blur (odd int, 0/1 disables)
}

# ---------------------------------------------------------------------------
# Leaf (foreground) segmentation
# ---------------------------------------------------------------------------
LEAF_SEGMENTATION = {
    # Otsu's threshold is computed on the saturation channel; it is then
    # clamped to [min_saturation, max_saturation] so a badly-shaped histogram
    # cannot produce an absurd threshold.
    'min_saturation': 35,
    'max_saturation': 140,
    # Pixels darker than this are not used as leaf seeds (deep shadow / black
    # backgrounds have unstable hue and saturation).
    'min_value': 35,
    # Leaves are never blue/violet, whereas PlantVillage backgrounds and cast
    # shadows usually are; these hues are excluded from the leaf seed. Purple
    # lesions *inside* the leaf are recovered by hole filling.
    'exclude_hue_ranges': [(96, 159)],
    # Weak seed: washed-out or necrotic leaf tissue has low saturation but
    # still a leaf-like hue (brown -> yellow -> green). Such pixels are added
    # to the Otsu seed when S > weak_min_saturation and V <= weak_max_value.
    'weak_seed_enabled': True,
    'weak_min_saturation': 25,
    'weak_max_value': 250,
    'weak_hue_ranges': [(5, 95)],
    'open_kernel': 5,
    'open_iterations': 1,
    'close_kernel': 11,
    'close_iterations': 2,
    # Fill the interior of the selected leaf contour so that pale / grey
    # lesions inside the leaf (low saturation) are counted in the leaf area.
    'fill_holes': True,
    # --- reliability checks ---
    'min_leaf_fraction': 0.05,        # leaf must cover >= 5 % of the image
    'max_leaf_fraction': 0.92,        # ...and must not be (almost) the whole frame
    'min_otsu_separability': 0.45,    # eta = between-class var / total var
    'min_saturation_std': 8.0,        # flat saturation histogram => no contrast
    # Fraction of the image border covered by the leaf mask. Single-leaf
    # PlantVillage photos stay well below 0.35; field photos where the "leaf"
    # merges with surrounding foliage/soil exceed 0.6.
    'max_border_contact': 0.60,       # above => unreliable
    'warn_border_contact_above': 0.35,
    'warn_solidity_below': 0.70,      # irregular contour => warn only
}

# ---------------------------------------------------------------------------
# Lesion colour ranges. Each entry is a list of HSV boxes that are OR-ed.
# A box with h_min > h_max wraps around the hue circle (e.g. reds).
# ---------------------------------------------------------------------------
LESION_COLOR_RANGES = {
    # necrotic brown tissue (orange hue, medium/low value)
    'brown': [{'h': (5, 20), 's': (40, 255), 'v': (40, 175)}],
    # bright orange (rust pustules, cedar-apple rust spots)
    'orange': [{'h': (5, 20), 's': (100, 255), 'v': (176, 255)}],
    # chlorosis / yellowing (healthy leaves are typically H 35-60)
    'yellow': [{'h': (17, 29), 's': (80, 255), 'v': (100, 255)}],
    # very dark / black necrosis. Saturated dark *green* (a shadowed but
    # healthy leaf margin) is deliberately excluded.
    'dark': [
        {'h': (0, 33), 's': (0, 255), 'v': (0, 60)},
        {'h': (96, 179), 's': (0, 255), 'v': (0, 60)},
        {'h': (34, 95), 's': (0, 60), 'v': (0, 60)},
    ],
    # red / red-brown (wraps around hue 0)
    'red': [{'h': (165, 4), 's': (70, 255), 'v': (40, 255)}],
    # purple / magenta spots (e.g. strawberry leaf scorch)
    'purple': [{'h': (125, 164), 's': (50, 255), 'v': (30, 220)}],
    # white powdery fungal growth (low saturation, bright)
    'white_powder': [{'h': (0, 179), 's': (0, 40), 'v': (165, 255)}],
    # grey / tan necrotic centres (low saturation, mid value)
    'gray_necrotic': [{'h': (0, 179), 's': (0, 45), 'v': (56, 164)}],
    # dark olive lesions (apple scab). Overlaps dark healthy green, use sparingly.
    'olive_dark': [{'h': (18, 40), 's': (40, 255), 'v': (56, 100)}],
}

LESION_SEGMENTATION = {
    'open_kernel': 3,
    'open_iterations': 1,
    'close_kernel': 3,
    'close_iterations': 1,
    # connected components smaller than this (at working resolution) are noise
    'min_component_area_px': 12,
    # shrink the leaf mask by this many px before searching for lesions, to
    # ignore the dark anti-aliased rim at the leaf/background boundary
    'boundary_erosion_px': 3,
    # result sanity checks (warnings, not failures)
    'warn_zero_lesion_for_diseased': True,
    'warn_severity_above': 85.0,
}

# ---------------------------------------------------------------------------
# Disease-specific profiles. Keys are the exact class labels from labels.py.
#   colors     : which LESION_COLOR_RANGES are treated as lesion for this disease
#   supported  : False when lesion-area ratio is not a meaningful severity
#                measure for the disease (systemic / viral symptoms)
#   validity   : 'heuristic' (colour-separable lesions) or 'low' (symptom
#                colours overlap healthy tissue; expect under/over-estimation)
#   note       : shown to the user as a caveat
#   overrides  : optional per-disease overrides of LESION_SEGMENTATION keys
# Healthy classes are handled before the estimator is called.
# ---------------------------------------------------------------------------
DEFAULT_PROFILE = {
    'colors': ['brown', 'yellow', 'dark'],
    'supported': True,
    'validity': 'heuristic',
    'note': 'Generic brown / yellow / dark lesion profile.',
}

DISEASE_PROFILES = {
    'Apple___Apple_scab': {
        'colors': ['brown', 'dark', 'olive_dark'], 'validity': 'low',
        'note': 'Olive-green scab lesions overlap healthy leaf colour; severity is likely under-estimated.',
    },
    'Apple___Black_rot': {
        'colors': ['brown', 'dark', 'red', 'purple'],
        'note': 'Frog-eye spots: tan/brown centres with purple margins.',
    },
    'Apple___Cedar_apple_rust': {
        'colors': ['orange', 'yellow', 'brown', 'red'],
        'note': 'Yellow-orange rust spots.',
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'colors': ['white_powder', 'yellow'], 'validity': 'low',
        'note': 'White fungal growth; specular glare can be mistaken for mildew.',
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'colors': ['gray_necrotic', 'brown', 'yellow'],
        'note': 'Rectangular grey/tan lesions between veins.',
    },
    'Corn_(maize)___Common_rust_': {
        'colors': ['brown', 'red', 'orange', 'dark'],
        'note': 'Reddish-brown pustules; very small pustules may fall below the noise filter.',
        # pustules are only a few px wide: skip opening, keep smaller components
        'overrides': {'open_kernel': 1, 'min_component_area_px': 6},
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'colors': ['gray_necrotic', 'brown', 'dark'],
        'note': 'Long elliptical grey-green to tan lesions.',
    },
    'Grape___Black_rot': {
        'colors': ['brown', 'dark', 'red'],
        'note': 'Brown circular lesions with dark margins.',
    },
    'Grape___Esca_(Black_Measles)': {
        'colors': ['brown', 'yellow', 'red', 'dark'], 'validity': 'low',
        'note': 'Esca is a trunk disease; leaf "tiger-stripe" area reflects foliar symptoms only, not vine-level severity.',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'colors': ['brown', 'dark'],
        'note': 'Dark brown irregular spots.',
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'colors': [], 'supported': False,
        'note': 'HLB is a systemic, phloem-limited bacterial disease. Blotchy-mottle area on one leaf '
                'is not a meaningful severity measure; tree-level assessment is required.',
    },
    'Peach___Bacterial_spot': {
        'colors': ['dark', 'brown', 'yellow', 'purple'],
        'note': 'Small angular dark spots, sometimes with yellow halos.',
    },
    'Pepper,_bell___Bacterial_spot': {
        'colors': ['dark', 'brown', 'yellow'],
        'note': 'Small dark water-soaked spots; yellow halos are counted as affected area.',
    },
    'Potato___Early_blight': {
        'colors': ['brown', 'dark', 'yellow'],
        'note': 'Brown concentric "target" lesions with yellow halos.',
    },
    'Potato___Late_blight': {
        'colors': ['dark', 'brown', 'yellow'],
        'note': 'Dark water-soaked lesions; pale-green lesion margins are not detected.',
    },
    'Squash___Powdery_mildew': {
        'colors': ['white_powder', 'yellow'], 'validity': 'low',
        'note': 'White fungal growth; specular glare can be mistaken for mildew.',
    },
    'Strawberry___Leaf_scorch': {
        'colors': ['purple', 'red', 'dark', 'brown'],
        'note': 'Irregular dark purple spots that merge as the leaf scorches.',
    },
    'Tomato___Bacterial_spot': {
        'colors': ['dark', 'brown', 'yellow'],
        'note': 'Small dark spots; yellow halos are counted as affected area.',
    },
    'Tomato___Early_blight': {
        'colors': ['brown', 'dark', 'yellow'],
        'note': 'Brown concentric "target" lesions with yellow halos.',
    },
    'Tomato___Late_blight': {
        'colors': ['dark', 'brown', 'yellow'],
        'note': 'Dark water-soaked lesions; pale-green lesion margins are not detected.',
    },
    'Tomato___Leaf_Mold': {
        'colors': ['yellow', 'brown', 'olive_dark'], 'validity': 'low',
        'note': 'Yellow patches on the upper leaf surface; the olive mould is on the underside.',
    },
    'Tomato___Septoria_leaf_spot': {
        'colors': ['dark', 'brown', 'gray_necrotic', 'yellow'],
        'note': 'Small circular spots with dark margins and grey/tan centres.',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'colors': ['yellow', 'brown'], 'validity': 'low',
        'note': 'Mite damage is fine stippling/bronzing, often below segmentation resolution. '
                'Treat the value as a rough indicator of discoloured area only.',
    },
    'Tomato___Target_Spot': {
        'colors': ['brown', 'dark', 'yellow'],
        'note': 'Brown lesions with concentric rings.',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'colors': [], 'supported': False,
        'note': 'Viral disease with systemic symptoms (leaf curling, yellowing, stunting). '
                'Lesion-area ratio is not a valid severity measure.',
    },
    'Tomato___Tomato_mosaic_virus': {
        'colors': [], 'supported': False,
        'note': 'Viral mosaic/mottling is spread across the leaf; lesion-area ratio is not a valid severity measure.',
    },
}

DEFAULT_CONFIG = {
    'severity_bands': SEVERITY_BANDS,
    'preprocessing': PREPROCESSING,
    'leaf_segmentation': LEAF_SEGMENTATION,
    'lesion_color_ranges': LESION_COLOR_RANGES,
    'lesion_segmentation': LESION_SEGMENTATION,
    'default_profile': DEFAULT_PROFILE,
    'disease_profiles': DISEASE_PROFILES,
}


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(override_path=None):
    """Return a deep copy of DEFAULT_CONFIG, optionally merged with a JSON override file."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    override_path = override_path or os.environ.get('SEVERITY_CONFIG')
    if override_path:
        with open(override_path, 'r', encoding='utf-8') as fh:
            _deep_merge(config, json.load(fh))
    return config


def get_profile(class_label, config=None):
    """Return the merged segmentation profile for a class label (default profile if unknown)."""
    config = config or DEFAULT_CONFIG
    profile = dict(config['default_profile'])
    specific = config['disease_profiles'].get(class_label)
    if specific:
        profile.update(specific)
        profile['name'] = class_label
    else:
        profile['name'] = 'default'
    profile.setdefault('supported', True)
    profile.setdefault('validity', 'heuristic')
    profile.setdefault('overrides', {})
    return profile
