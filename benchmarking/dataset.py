"""
Dataset scanning, a shared stratified train/validation/test split, and tf.data
pipelines with model-specific preprocessing.

Split modes
-----------
stratified  one directory of class folders (e.g. original PlantVillage "color")
            -> stratified train / val / test with a fixed seed, saved to JSON so
            every model is trained and tested on exactly the same files.
predefined  an existing train directory + a held-out test directory (e.g. the
            Kaggle "New Plant Diseases Dataset (Augmented)" train/ and valid/).
            Validation is carved (stratified) out of train/; valid/ is the test
            set. Use this to evaluate the existing Model.hdf5 fairly, because it
            was trained on that train/ folder.

Group-aware splitting
---------------------
Augmented datasets contain several transformed copies of one source photo.
If copies land in both train and test, test accuracy is inflated. With
``group_regex`` the augmentation suffix is stripped from the file stem and all
files sharing the remaining stem are kept in the same subset
(StratifiedGroupKFold). The ``kaggle_augmented`` preset covers the suffixes
observed in that dataset's file names; verify it against your copy.
"""

import hashlib
import json
import os
import re
from collections import Counter

import numpy as np

from labels import CLASS_LABELS

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

GROUP_PRESETS = {
    # e.g. "0a5e...___FREC_Pwd.M 4921_flipLR.JPG", "..._180deg.JPG", "..._new30degFlipLR.JPG"
    'kaggle_augmented': r'(_(new)?\d*deg(FlipLR|FlipTB)?|_flipLR|_flipTB|_newGRR)+$',
}


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------

def scan_directory(data_dir, class_labels=CLASS_LABELS, max_per_class=0, seed=42):
    """Return (paths, labels, report). Class folders must be named exactly like ``class_labels``."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: %s' % data_dir)
    present = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    unknown = [d for d in present if d not in class_labels]
    missing = [c for c in class_labels if c not in present]
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for idx, name in enumerate(class_labels):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
        if max_per_class and len(files) > max_per_class:
            files = sorted(rng.choice(files, size=max_per_class, replace=False).tolist())
        paths.extend(os.path.join(folder, f) for f in files)
        labels.extend([idx] * len(files))
    report = {'data_dir': os.path.abspath(data_dir), 'num_images': len(paths),
              'missing_classes': missing, 'unrecognised_folders': unknown,
              'per_class': {class_labels[k]: v for k, v in sorted(Counter(labels).items())}}
    return paths, np.array(labels, dtype=np.int64), report


def group_keys(paths, group_regex):
    """Group id per file: file stem with augmentation suffixes removed (regex applied to the stem)."""
    pattern = re.compile(GROUP_PRESETS.get(group_regex, group_regex))
    keys = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        keys.append(os.path.basename(os.path.dirname(p)) + '/' + pattern.sub('', stem))
    return np.array(keys)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _stratified_split(indices, labels, frac, seed, groups=None):
    """Split ``indices`` into (rest, held_out) with held_out ~= frac, stratified by label."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    if frac <= 0:
        return indices, np.array([], dtype=indices.dtype)
    if groups is None:
        rest, held = train_test_split(indices, test_size=frac, stratify=labels[indices], random_state=seed)
        return np.sort(rest), np.sort(held)
    n_splits = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_pos, held_pos = next(sgkf.split(indices, labels[indices], groups[indices]))
    return np.sort(indices[rest_pos]), np.sort(indices[held_pos])


def make_split(data_dir, val_frac=0.15, test_frac=0.15, seed=42, test_dir=None, group_regex=None,
               max_per_class=0, class_labels=CLASS_LABELS):
    """Build a split dict {'meta', 'train', 'val', 'test'}; subsets are lists of [path, label]."""
    paths, labels, report = scan_directory(data_dir, class_labels, max_per_class, seed)
    if not paths:
        raise ValueError('No images found under %s' % data_dir)
    groups = group_keys(paths, group_regex) if group_regex else None
    idx = np.arange(len(paths))
    meta = {'mode': 'predefined' if test_dir else 'stratified', 'seed': seed, 'val_frac': val_frac,
            'test_frac': None if test_dir else test_frac, 'group_regex': group_regex,
            'max_per_class': max_per_class, 'train_source': report}

    if test_dir:
        test_paths, test_labels, test_report = scan_directory(test_dir, class_labels, max_per_class, seed)
        train_idx, val_idx = _stratified_split(idx, labels, val_frac, seed, groups)
        test = [[p, int(l)] for p, l in zip(test_paths, test_labels)]
        meta['test_source'] = test_report
    else:
        rest_idx, test_idx = _stratified_split(idx, labels, test_frac, seed, groups)
        val_rel = val_frac / (1.0 - test_frac)
        train_idx, val_idx = _stratified_split(rest_idx, labels, val_rel, seed, groups)
        test = [[paths[i], int(labels[i])] for i in test_idx]

    split = {'meta': meta,
             'train': [[paths[i], int(labels[i])] for i in train_idx],
             'val': [[paths[i], int(labels[i])] for i in val_idx],
             'test': test}
    if groups is not None:
        tr, va = set(groups[train_idx]), set(groups[val_idx])
        te = set(groups[test_idx]) if not test_dir else set()
        meta['group_leakage'] = {'train_val': len(tr & va), 'train_test': len(tr & te), 'val_test': len(va & te)}
    meta['counts'] = {k: len(split[k]) for k in ('train', 'val', 'test')}
    meta['fingerprint'] = split_fingerprint(split)
    return split


def split_fingerprint(split):
    """SHA-256 over the sorted (subset, path, label) triples: proves two runs used the same data."""
    h = hashlib.sha256()
    for subset in ('train', 'val', 'test'):
        for path, label in sorted(split[subset]):
            h.update(('%s|%s|%d\n' % (subset, os.path.basename(os.path.dirname(path)) + '/' +
                                      os.path.basename(path), label)).encode('utf-8'))
    return h.hexdigest()


def per_class_counts(split, class_labels=CLASS_LABELS):
    return {subset: {class_labels[k]: v for k, v in sorted(Counter(l for _, l in split[subset]).items())}
            for subset in ('train', 'val', 'test')}


def save_split(split, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(split, fh, indent=1)


def load_split(path):
    with open(path, 'r', encoding='utf-8') as fh:
        split = json.load(fh)
    for key in ('meta', 'train', 'val', 'test'):
        if key not in split:
            raise ValueError('Split file %s is missing %r' % (path, key))
    return split


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------

def load_resized_uint8(path, size):
    """Same resizing as the deployed app: PIL RGB conversion + default (bicubic) resize."""
    from PIL import Image
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    img = Image.open(path).convert('RGB').resize((int(size), int(size)))
    return np.asarray(img, dtype=np.uint8)


def build_augmenter(seed=42):
    """Identical augmentation for every architecture: flips, small rotation, brightness/contrast jitter.
    Operates on float32 images in [0, 255] *before* model-specific preprocessing."""
    import tensorflow as tf
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical', seed=seed),
        layers.RandomRotation(0.1, fill_mode='reflect', seed=seed),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.15, seed=seed),
    ], name='shared_augmentation')


def make_tf_dataset(samples, image_size, preprocess_fn, batch_size=32, training=False, augment=False,
                    seed=42, shuffle_buffer=2048):
    """samples: list of [path, label]. Yields (preprocessed float32 batch, int labels)."""
    import tensorflow as tf
    paths = [p for p, _ in samples]
    labels = [int(l) for _, l in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, max(1, len(paths))), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.numpy_function(load_resized_uint8, [path, image_size], tf.uint8)
        img.set_shape((image_size, image_size, 3))
        return tf.cast(img, tf.float32), label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
