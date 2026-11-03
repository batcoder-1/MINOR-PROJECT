"""
Command-line severity estimation for one image or a folder of images.

    python -m severity --image uploads/PotatoEarlyBlight1.JPG --label Potato___Early_blight --out_dir severity_outputs
    python -m severity --image uploads/ --out_dir severity_outputs        # whole folder, default profile

Writes <name>_panel.png (+ individual masks) and prints/writes a JSON summary.
The class label selects the colour profile; pass the classifier's prediction
(or the ground-truth label when preparing a validation sample).
"""

import argparse
import json
import os
import sys

from .config import load_config
from .estimator import estimate_severity
from .visualization import save_visualizations

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def main(argv=None):
    parser = argparse.ArgumentParser(description='HSV lesion-area severity estimation (experimental)')
    parser.add_argument('--image', required=True, help='Image file or directory of images')
    parser.add_argument('--label', default=None,
                        help='PlantVillage class label selecting the colour profile (default profile if omitted)')
    parser.add_argument('--out_dir', default='severity_outputs', help='Directory for PNG visualisations')
    parser.add_argument('--config', default=None, help='Optional JSON override for severity/config.py values')
    parser.add_argument('--json_out', default=None, help='Write all results to this JSON file')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if os.path.isdir(args.image):
        paths = sorted(os.path.join(args.image, f) for f in os.listdir(args.image)
                       if f.lower().endswith(IMAGE_EXTENSIONS))
    else:
        paths = [args.image]
    if not paths:
        print('No images found at %s' % args.image, file=sys.stderr)
        return 1

    results = []
    for path in paths:
        result = estimate_severity(path, class_label=args.label, config=config)
        prefix = os.path.splitext(os.path.basename(path))[0]
        files = save_visualizations(result, args.out_dir, prefix=prefix)
        entry = dict(result.to_dict(), image=path, class_label=args.label, files=files)
        results.append(entry)
        pct = '%.2f%%' % result.percentage if result.percentage is not None else '-'
        print('%-40s %-15s %-14s %8s' % (os.path.basename(path), result.status, result.category, pct))
        for warning in result.warnings:
            print('    ! ' + warning)

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
        print('Saved %s' % args.json_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
