"""
Agreement between the HSV severity heuristic and manual (expert) grading - the
evaluation plan in the project proposal.

1. Create an annotation sheet for a sample of images of one disease:
       python -m severity.validate template --images_dir data/valid/Tomato___Early_blight \
           --label Tomato___Early_blight --n 150 --out annotations/tomato_early_blight.csv
2. Grade each image by eye (Mild / Moderate / Severe; optionally an estimated %)
   WITHOUT looking at the heuristic output, ideally by two annotators.
3. Score:
       python -m severity.validate score --annotations annotations/tomato_early_blight.csv \
           --out annotations/tomato_early_blight_results.json

Reported: coverage (share of images the heuristic could grade), exact band
agreement, Cohen's kappa (unweighted and linearly weighted - the bands are
ordinal), the 3x3 confusion matrix, the share of Mild<->Severe disagreements,
and the mean absolute error in percentage points when manual percentages exist.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from .config import load_config
from .estimator import estimate_severity

BANDS = ['Mild', 'Moderate', 'Severe']
FIELDS = ['image_path', 'class_label', 'manual_category', 'manual_percentage', 'annotator', 'notes']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def write_template(images_dir, label, out, n=150, seed=42):
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    rng = np.random.default_rng(seed)
    if n and len(files) > n:
        files = sorted(rng.choice(files, size=n, replace=False).tolist())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for f in files:
            writer.writerow({'image_path': os.path.join(images_dir, f), 'class_label': label})
    return len(files)


def agreement(manual, predicted):
    """Agreement statistics for two equal-length lists of band names."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    manual, predicted = list(manual), list(predicted)
    n = len(manual)
    if n == 0:
        return {'n': 0}
    idx_m = [BANDS.index(m) for m in manual]
    idx_p = [BANDS.index(p) for p in predicted]
    both_single = len(set(idx_m) | set(idx_p)) < 2
    return {
        'n': n,
        'exact_agreement': float(np.mean(np.array(idx_m) == np.array(idx_p))),
        'cohen_kappa': None if both_single else float(cohen_kappa_score(idx_m, idx_p, labels=[0, 1, 2])),
        'cohen_kappa_linear_weighted': None if both_single else float(
            cohen_kappa_score(idx_m, idx_p, labels=[0, 1, 2], weights='linear')),
        'mild_severe_confusions': int(sum(abs(a - b) == 2 for a, b in zip(idx_m, idx_p))),
        'confusion_matrix': {'rows_manual_cols_heuristic': BANDS,
                             'counts': confusion_matrix(idx_m, idx_p, labels=[0, 1, 2]).tolist()},
    }


def score(annotations, config=None):
    config = config or load_config()
    rows = list(csv.DictReader(open(annotations, encoding='utf-8')))
    per_image, manual, predicted, abs_err = [], [], [], []
    skipped_unlabelled = 0
    for row in rows:
        m = (row.get('manual_category') or '').strip().title()
        if m not in BANDS:
            skipped_unlabelled += 1
            continue
        result = estimate_severity(row['image_path'], class_label=row.get('class_label') or None, config=config)
        entry = {'image_path': row['image_path'], 'manual_category': m, 'status': result.status,
                 'heuristic_category': result.category, 'heuristic_percentage': result.percentage}
        if result.status == 'estimated':
            manual.append(m)
            predicted.append(result.category)
            mp = (row.get('manual_percentage') or '').strip()
            if mp:
                abs_err.append(abs(float(mp) - result.percentage))
        per_image.append(entry)
    graded = len(per_image)
    stats = agreement(manual, predicted)
    stats.update({
        'annotation_file': annotations,
        'rows': len(rows), 'rows_without_manual_label': skipped_unlabelled,
        'graded_by_annotator': graded,
        'heuristic_coverage': (len(manual) / graded) if graded else None,
        'heuristic_unreliable_or_na': graded - len(manual),
        'percentage_mae': float(np.mean(abs_err)) if abs_err else None,
        'percentage_pairs': len(abs_err),
        'per_image': per_image,
    })
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate the severity heuristic against manual grades')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('template')
    t.add_argument('--images_dir', required=True)
    t.add_argument('--label', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--n', type=int, default=150)
    t.add_argument('--seed', type=int, default=42)
    s = sub.add_parser('score')
    s.add_argument('--annotations', required=True)
    s.add_argument('--out', default=None)
    s.add_argument('--config', default=None)
    args = parser.parse_args(argv)

    if args.command == 'template':
        count = write_template(args.images_dir, args.label, args.out, args.n, args.seed)
        print('Wrote %d rows to %s; fill manual_category (Mild/Moderate/Severe).' % (count, args.out))
        return 0
    stats = score(args.annotations, load_config(args.config))
    summary = {k: v for k, v in stats.items() if k != 'per_image'}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(stats, fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
