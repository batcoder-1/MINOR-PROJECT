"""
Classification metrics and efficiency measurements for the benchmark.

Classification: accuracy (+ Wilson 95 % CI), macro / weighted precision, recall,
F1, balanced accuracy, top-k accuracy, per-class report, confusion matrix.

Efficiency: parameter count, size on disk, CPU latency at batch size 1
(warm-up, then median of N timed runs), FLOPs via the TensorFlow profiler.
"""

import csv
import math
import os
import platform
import statistics
import time

import numpy as np

from labels import CLASS_LABELS


def wilson_interval(successes, n, z=1.96):
    """95 % Wilson score interval for a proportion (better than the normal approximation near 0 or 1)."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classification_metrics(y_true, y_pred, probs=None, class_names=CLASS_LABELS):
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support,
                                 top_k_accuracy_score)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    n = int(len(y_true))
    correct = int((y_true == y_pred).sum())
    out = {'num_samples': n, 'num_correct': correct, 'accuracy': correct / n if n else float('nan')}
    out['accuracy_ci95'] = list(wilson_interval(correct, n))
    present = sorted(set(y_true.tolist()))
    out['num_classes_in_test'] = len(present)

    for avg in ('macro', 'weighted'):
        # macro averages over the classes present in the test set (absent classes would add spurious zeros)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average=avg, zero_division=0)
        out['precision_%s' % avg], out['recall_%s' % avg], out['f1_%s' % avg] = float(p), float(r), float(f)
    out['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    if probs is not None and len(present) > 1:
        for k in (3, 5):
            out['top%d_accuracy' % k] = float(top_k_accuracy_score(y_true, probs, k=k, labels=labels))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out['per_class'] = [
        {'class_index': i, 'class_label': class_names[i], 'precision': float(p[i]), 'recall': float(r[i]),
         'f1': float(f[i]), 'support': int(s[i])}
        for i in labels if s[i] > 0 or (y_pred == i).any()
    ]
    out['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


def write_per_class_csv(per_class, path):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['class_index', 'class_label', 'precision', 'recall', 'f1', 'support'])
        writer.writeheader()
        writer.writerows(per_class)


def write_confusion_matrix_csv(matrix, path, class_names=CLASS_LABELS):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['true \\ predicted'] + list(class_names))
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + list(row))


def plot_confusion_matrix(matrix, path, class_names=CLASS_LABELS, title='Confusion matrix (row-normalised)'):
    """Row-normalised heat map (recall on the diagonal). Requires matplotlib; returns False if unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    m = np.asarray(matrix, dtype=np.float64)
    rows = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [c.replace('_(including_sour)', '').replace('_(maize)', '').replace('___', ': ')[:38] for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------

def size_on_disk(path):
    """Bytes of a model file or SavedModel directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def environment_info():
    import tensorflow as tf
    return {'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(),
            'python': platform.python_version(), 'tensorflow': tf.__version__, 'cpu_count': os.cpu_count(),
            'intra_op_threads': tf.config.threading.get_intra_op_parallelism_threads(),
            'inter_op_threads': tf.config.threading.get_inter_op_parallelism_threads()}


def measure_cpu_latency(model, input_size, n_warmup=10, n_runs=100, batch_size=1, seed=0):
    """Wall-clock latency of one forward pass on CPU (batch 1 by default).

    The model call is wrapped in a tf.function so graph tracing happens during
    warm-up and is not timed. Input values do not affect latency, so a random
    tensor in the model's expected range is used.
    """
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.uniform(0, 1, (batch_size, input_size, input_size, 3)).astype(np.float32))
    with tf.device('/CPU:0'):
        forward = tf.function(lambda t: model(t, training=False))
        for _ in range(n_warmup):
            forward(x).numpy()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            forward(x).numpy()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {'batch_size': batch_size, 'warmup_runs': n_warmup, 'timed_runs': n_runs,
            'median_ms': statistics.median(times), 'mean_ms': statistics.fmean(times),
            'std_ms': statistics.pstdev(times), 'p90_ms': times[int(0.90 * (n_runs - 1))],
            'p95_ms': times[int(0.95 * (n_runs - 1))], 'min_ms': times[0], 'max_ms': times[-1],
            'throughput_img_per_s': 1000.0 * batch_size / statistics.median(times)}


def estimate_flops(model, input_size):
    """Total float operations for one image, counted by the TF profiler on the frozen graph.

    The profiler counts a multiply-add as 2 FLOPs, so values are ~2x the 'GMACs'
    figures often quoted for ImageNet models. Returns None if profiling fails.
    """
    import tensorflow as tf
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        spec = tf.TensorSpec([1, input_size, input_size, 3], tf.float32)
        concrete = tf.function(lambda t: model(t, training=False)).get_concrete_function(spec)
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'none'
            prof = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op',
                                                 options=opts)
        return int(prof.total_float_ops)
    except Exception:  # profiler APIs are internal and may change between TF versions
        return None
