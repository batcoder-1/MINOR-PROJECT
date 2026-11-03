"""
Evaluation of one model on the shared test split, writing machine-readable
results:

    metrics.json            all metrics + efficiency + provenance
    per_class.csv           precision / recall / F1 / support per class
    confusion_matrix.csv    38 x 38 counts (rows = true class)
    confusion_matrix.png    row-normalised heat map (if matplotlib is installed)
    predictions.csv         one row per test image (true, predicted, confidence)
"""

import csv
import json
import os

import numpy as np

from labels import CLASS_LABELS
from .dataset import make_tf_dataset, split_fingerprint
from .metrics import (classification_metrics, environment_info, estimate_flops, measure_cpu_latency,
                      plot_confusion_matrix, size_on_disk, write_confusion_matrix_csv, write_per_class_csv)
from .models import get_spec, load_trained_model, param_counts

CONTAMINATION_NOTE = ('The existing Model.hdf5 was trained on the Kaggle "New Plant Diseases Dataset (Augmented)" '
                      'train/ folder, which is derived from PlantVillage. Unless this test set is that dataset\'s '
                      'valid/ folder (split mode "predefined"), test images or augmented copies of them may have '
                      'been seen during training, which would inflate its scores.')


def evaluate(model_name, split, out_dir, model_path=None, model=None, batch_size=32, image_size=None,
             subset='test', latency_runs=100, latency_warmup=10, flops=True, training_summary=None,
             max_batches=None, seed=None):
    import tensorflow as tf

    spec = get_spec(model_name)
    size = int(image_size or spec.input_size)
    os.makedirs(out_dir, exist_ok=True)
    if model is None:
        model = load_trained_model(model_name, model_path)

    samples = split[subset]
    if max_batches:
        samples = samples[:max_batches * batch_size]
    ds = make_tf_dataset(samples, size, spec.preprocess, batch_size)
    probs = model.predict(ds, verbose=0)
    y_true = np.array([label for _, label in samples])
    y_pred = probs.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, probs)

    params = param_counts(model)
    efficiency = {
        'params': params,
        'weights_float32_mb': round(params['total'] * 4 / 1e6, 2),
        'size_on_disk_mb': round(size_on_disk(model_path) / 1e6, 2) if model_path and os.path.exists(model_path) else None,
        'input_size': size,
        'cpu_latency': measure_cpu_latency(model, size, n_warmup=latency_warmup, n_runs=latency_runs)
        if latency_runs else None,
        'flops_per_image': estimate_flops(model, size) if flops else None,
    }

    caveats = []
    if model_name == 'alexnet_existing' and split['meta'].get('mode') != 'predefined':
        caveats.append(CONTAMINATION_NOTE)
    if max_batches or (training_summary or {}).get('smoke_test'):
        caveats.append('SMOKE TEST: truncated data/training. These numbers are not experimental results.')
    if len(set(y_true.tolist())) < len(CLASS_LABELS):
        caveats.append('Only %d of %d classes are present in the evaluated subset.'
                       % (len(set(y_true.tolist())), len(CLASS_LABELS)))

    record = {
        'model': model_name, 'display_name': spec.display_name, 'seed': seed,
        'model_path': model_path, 'subset': subset,
        'split_mode': split['meta'].get('mode'),
        'split_fingerprint': split['meta'].get('fingerprint') or split_fingerprint(split),
        'metrics': {k: v for k, v in metrics.items() if k not in ('per_class', 'confusion_matrix')},
        'efficiency': efficiency,
        'training': training_summary,
        'environment': environment_info(),
        'caveats': caveats,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2)
    write_per_class_csv(metrics['per_class'], os.path.join(out_dir, 'per_class.csv'))
    write_confusion_matrix_csv(metrics['confusion_matrix'], os.path.join(out_dir, 'confusion_matrix.csv'))
    plot_confusion_matrix(metrics['confusion_matrix'], os.path.join(out_dir, 'confusion_matrix.png'),
                          title='%s - row-normalised confusion matrix' % spec.display_name)
    with open(os.path.join(out_dir, 'predictions.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['path', 'true_index', 'true_label', 'pred_index', 'pred_label', 'confidence'])
        for (path, label), pred, p in zip(samples, y_pred, probs):
            writer.writerow([path, label, CLASS_LABELS[label], int(pred), CLASS_LABELS[int(pred)],
                             round(float(p[int(pred)]), 6)])
    tf.keras.backend.clear_session()
    return record
