"""
Benchmark framework tests. Fast tests cover metrics and splitting; tests marked
``slow`` build real Keras models (random weights, small inputs) and run a
2-step training loop on a synthetic dataset purely to prove the pipeline runs.
None of these numbers are experimental results.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from benchmarking.dataset import group_keys, load_split, make_split, save_split, scan_directory, split_fingerprint
from benchmarking.metrics import classification_metrics, wilson_interval
from labels import CLASS_LABELS

CLASSES = ['Apple___Apple_scab', 'Apple___healthy', 'Tomato___Late_blight', 'Tomato___healthy']


def _make_dataset(root, per_class=20, augmented_copies=0, size=48):
    rng = np.random.default_rng(0)
    for ci, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(per_class):
            base = np.full((size, size, 3), (40 * ci, 120, 60), np.uint8)
            base = np.clip(base + rng.integers(0, 40, base.shape), 0, 255).astype(np.uint8)
            Image.fromarray(base).save(folder / ('img%03d___X %d.JPG' % (i, i)))
            for suffix in ['_flipLR', '_180deg', '_new30degFlipLR'][:augmented_copies]:
                Image.fromarray(base[:, ::-1]).save(folder / ('img%03d___X %d%s.JPG' % (i, i, suffix)))
    return str(root)


# ---------------------------------------------------------------- metrics

def test_metrics_known_values():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 0]
    m = classification_metrics(y_true, y_pred)
    assert m['accuracy'] == pytest.approx(6 / 8)
    # per-class recall: 3/4, 2/2, 1/2 -> macro 0.75 ; precision: 3/4, 2/3, 1/1
    assert m['recall_macro'] == pytest.approx((0.75 + 1.0 + 0.5) / 3)
    assert m['precision_macro'] == pytest.approx((0.75 + 2 / 3 + 1.0) / 3)
    assert m['num_classes_in_test'] == 3
    cm = np.array(m['confusion_matrix'])
    assert cm.shape == (38, 38) and cm.sum() == 8 and cm[0, 1] == 1 and cm[2, 0] == 1


def test_macro_f1_exposes_minority_class_failure():
    # 95 majority-class samples right, 5 minority samples all wrong
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 100
    m = classification_metrics(y_true, y_pred)
    assert m['accuracy'] == pytest.approx(0.95)
    assert m['f1_macro'] < 0.5 < m['f1_weighted']


def test_wilson_interval_matches_minor_project_sample():
    lo, hi = wilson_interval(717, 760)   # evaluation_results.json: 717 / 760 correct
    assert lo == pytest.approx(0.925, abs=0.002) and hi == pytest.approx(0.958, abs=0.002)


def test_top_k_accuracy():
    probs = np.zeros((2, 38))
    probs[0, [5, 3, 1]] = [0.5, 0.3, 0.2]
    probs[1, [7, 2, 4]] = [0.6, 0.3, 0.1]
    m = classification_metrics([3, 4], [5, 7], probs)
    assert m['accuracy'] == 0 and m['top3_accuracy'] == 1.0


# ---------------------------------------------------------------- dataset / splits

def test_scan_directory_reports_missing_classes(tmp_path):
    data = _make_dataset(tmp_path / 'data', per_class=5)
    (tmp_path / 'data' / 'Not_a_class').mkdir()
    paths, labels, report = scan_directory(data)
    assert len(paths) == 20 and set(labels.tolist()) == {CLASS_LABELS.index(c) for c in CLASSES}
    assert len(report['missing_classes']) == 34 and report['unrecognised_folders'] == ['Not_a_class']


def test_stratified_split_is_disjoint_balanced_and_deterministic(tmp_path):
    data = _make_dataset(tmp_path / 'data', per_class=20)
    a = make_split(data, val_frac=0.15, test_frac=0.15, seed=7)
    b = make_split(data, val_frac=0.15, test_frac=0.15, seed=7)
    c = make_split(data, val_frac=0.15, test_frac=0.15, seed=8)
    assert a['meta']['fingerprint'] == b['meta']['fingerprint'] != c['meta']['fingerprint']
    sets = [set(p for p, _ in a[k]) for k in ('train', 'val', 'test')]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 80
    for cls in CLASSES:   # stratified: every class in every subset, 3 test images each
        idx = CLASS_LABELS.index(cls)
        assert sum(1 for _, l in a['test'] if l == idx) == 3
        assert sum(1 for _, l in a['val'] if l == idx) >= 2


def test_group_aware_split_keeps_augmented_copies_together(tmp_path):
    data = _make_dataset(tmp_path / 'data', per_class=20, augmented_copies=3)
    paths, _, _ = scan_directory(data)
    keys = group_keys(paths, 'kaggle_augmented')
    assert len(set(keys)) == 80   # 4 files per source photo collapse to one group
    naive = make_split(data, seed=1)
    grouped = make_split(data, seed=1, group_regex='kaggle_augmented')
    assert grouped['meta']['group_leakage'] == {'train_val': 0, 'train_test': 0, 'val_test': 0}

    def leaked(split):
        tr = set(group_keys([p for p, _ in split['train']], 'kaggle_augmented'))
        te = set(group_keys([p for p, _ in split['test']], 'kaggle_augmented'))
        return len(tr & te)
    assert leaked(naive) > 0 and leaked(grouped) == 0


def test_predefined_mode_uses_separate_test_dir(tmp_path):
    train_dir = _make_dataset(tmp_path / 'train', per_class=10)
    test_dir = _make_dataset(tmp_path / 'valid', per_class=4)
    split = make_split(train_dir, test_dir=test_dir, val_frac=0.2, seed=0)
    assert split['meta']['mode'] == 'predefined'
    assert len(split['test']) == 16 and len(split['train']) + len(split['val']) == 40
    assert all(p.startswith(test_dir) for p, _ in split['test'])


def test_split_roundtrip(tmp_path):
    data = _make_dataset(tmp_path / 'data', per_class=10)
    split = make_split(data, seed=3)
    path = tmp_path / 'splits' / 's.json'
    save_split(split, str(path))
    loaded = load_split(str(path))
    assert split_fingerprint(loaded) == split['meta']['fingerprint']


# ---------------------------------------------------------------- models (TensorFlow)

@pytest.mark.slow
def test_alexnet_rebuild_matches_existing_model_size():
    from benchmarking.models import build_model, param_counts
    model, _ = build_model('alexnet')
    assert model.output_shape == (None, 38)
    assert param_counts(model)['total'] == 28117790   # identical to Model.hdf5


@pytest.mark.slow
@pytest.mark.parametrize('name,size', [('resnet50', 64), ('efficientnet_b4', 64)])
def test_transfer_models_build_with_shared_head(name, size):
    from benchmarking.models import build_model, param_counts, set_backbone_trainable
    model, backbone = build_model(name, weights=None, image_size=size)
    assert model.output_shape == (None, 38)
    assert [l.name for l in model.layers[-3:]] == ['gap', 'head_dropout', 'predictions']
    set_backbone_trainable(backbone, 0)
    head_only = param_counts(model)['trainable']
    assert head_only == model.get_layer('predictions').count_params()
    set_backbone_trainable(backbone, -1)
    assert param_counts(model)['trainable'] > 10 * head_only


@pytest.mark.slow
def test_model_specific_preprocessing():
    import tensorflow as tf
    from benchmarking.models import get_spec
    x = tf.constant(np.full((1, 2, 2, 3), 255.0, np.float32))
    assert float(get_spec('alexnet').preprocess(x).numpy().max()) == pytest.approx(1.0)
    resnet = get_spec('resnet50').preprocess(x).numpy()[0, 0, 0]
    assert resnet == pytest.approx([255 - 103.939, 255 - 116.779, 255 - 123.68], abs=1e-3)   # BGR mean subtraction
    assert float(get_spec('efficientnet_b4').preprocess(x).numpy().max()) == pytest.approx(255.0)


@pytest.mark.slow
def test_end_to_end_smoke_train_evaluate_compare(tmp_path):
    from benchmarking.benchmark import main
    data = _make_dataset(tmp_path / 'data', per_class=12)
    split_path = str(tmp_path / 'split.json')
    runs = str(tmp_path / 'runs')
    assert main(['split', '--data_dir', data, '--out', split_path, '--seed', '1']) == 0
    assert main(['run', '--model', 'resnet50', '--split', split_path, '--runs_dir', runs, '--weights', 'none',
                 '--image_size', '64', '--batch_size', '8', '--epochs_head', '1', '--epochs_finetune', '1',
                 '--max_steps_per_epoch', '2', '--latency_runs', '5', '--seed', '1']) == 0
    run_dir = os.path.join(runs, 'resnet50_seed1')
    for name in ('model.keras', 'history.csv', 'training_summary.json', 'metrics.json', 'per_class.csv',
                 'confusion_matrix.csv', 'predictions.csv'):
        assert os.path.exists(os.path.join(run_dir, name)), name
    record = json.load(open(os.path.join(run_dir, 'metrics.json')))
    assert record['efficiency']['params']['total'] > 20_000_000
    assert record['efficiency']['cpu_latency']['timed_runs'] == 5
    assert record['training']['epochs_run_total'] == 2
    assert any('SMOKE TEST' in c for c in record['caveats'])
    out = str(tmp_path / 'results')
    assert main(['compare', '--runs_dir', runs, '--out_dir', out]) == 0
    summary = json.load(open(os.path.join(out, 'summary.json')))
    assert summary['models'][0]['model'] == 'resnet50' and any('Smoke-test' in w for w in summary['warnings'])
