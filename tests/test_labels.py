import ast
import os

from labels import (CLASS_LABELS, NUM_CLASSES, class_index, display_crop_name, display_disease_name, is_healthy,
                    split_label)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _labels_from_source(filename):
    """Read the CLASS_LABELS literal from a Python file without importing it (evaluate.py imports TensorFlow)."""
    tree = ast.parse(open(os.path.join(ROOT, filename), encoding='utf-8').read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'CLASS_LABELS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('CLASS_LABELS not found in %s' % filename)


def test_38_classes():
    assert NUM_CLASSES == 38 == len(set(CLASS_LABELS))


def test_order_matches_evaluate_script():
    # evaluate.py keeps its own copy of the list; both must match the model's output order
    assert _labels_from_source('evaluate.py') == CLASS_LABELS


def test_order_is_alphabetical_like_flow_from_directory():
    # Keras flow_from_directory sorts class folder names; the model's output index follows that order
    assert CLASS_LABELS == sorted(CLASS_LABELS)


def test_split_and_display_helpers():
    assert split_label('Tomato___Late_blight') == ('Tomato', 'Late_blight')
    assert display_disease_name('Corn_(maize)___Common_rust_') == 'Common Rust'   # no trailing space
    assert display_crop_name('Pepper,_bell___Bacterial_spot') == 'Pepper, bell'
    assert is_healthy('Apple___healthy') and not is_healthy('Apple___Apple_scab')
    assert class_index('Tomato___healthy') == 37
    assert sum(is_healthy(c) for c in CLASS_LABELS) == 12
