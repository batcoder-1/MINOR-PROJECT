"""
Command-line interface for the AlexNet vs ResNet-50 vs EfficientNet-B4 benchmark.

    # 0. list models and their input sizes / preprocessing
    python -m benchmarking.benchmark info

    # 1. one shared split for every model (save it, commit it, reuse it)
    python -m benchmarking.benchmark split --data_dir data/plantvillage/color --out splits/pv_seed42.json
    python -m benchmarking.benchmark split --data_dir data/kaggle/train --test_dir data/kaggle/valid \
        --group_regex kaggle_augmented --out splits/kaggle_seed42.json

    # 2. train + evaluate one architecture (repeat with --seed 1 2 3 for mean +- std)
    python -m benchmarking.benchmark run --model resnet50 --split splits/kaggle_seed42.json --seed 42

    # 3. evaluate the existing minor-project model on the same test set
    python -m benchmarking.benchmark evaluate --model alexnet_existing --model_path Model.hdf5 \
        --split splits/kaggle_seed42.json

    # 4. efficiency only (params / size / FLOPs / CPU latency), no dataset needed
    python -m benchmarking.benchmark latency --models alexnet_existing resnet50 efficientnet_b4

    # 5. aggregate all runs into CSV/JSON/Markdown and apply the selection rule
    python -m benchmarking.benchmark compare --runs_dir benchmark_results/runs
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_RESULTS = os.path.join(ROOT, 'benchmark_results')


def cmd_info(_args):
    from .models import MODEL_SPECS
    for spec in MODEL_SPECS.values():
        print('%-18s input %3d  pretrained=%-12s two_phase=%-5s %s' % (
            spec.name, spec.input_size, spec.pretrained.split(' ')[0], spec.two_phase, spec.notes))
    return 0


def cmd_split(args):
    from .dataset import make_split, per_class_counts, save_split
    split = make_split(args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
                       test_dir=args.test_dir, group_regex=args.group_regex, max_per_class=args.max_per_class)
    save_split(split, args.out)
    meta = split['meta']
    print('Saved %s  (%s mode)  train=%d val=%d test=%d  fingerprint=%s' % (
        args.out, meta['mode'], meta['counts']['train'], meta['counts']['val'], meta['counts']['test'],
        meta['fingerprint'][:12]))
    if meta['train_source']['missing_classes']:
        print('WARNING missing classes: %s' % meta['train_source']['missing_classes'])
    if 'group_leakage' in meta:
        print('Group leakage between subsets: %s' % meta['group_leakage'])
    counts = per_class_counts(split)
    smallest = min(counts['test'].items(), key=lambda kv: kv[1]) if counts['test'] else None
    if smallest:
        print('Smallest test class: %s (%d images)' % smallest)
    return 0


def _run_dir(args):
    return os.path.join(args.runs_dir, '%s_seed%d' % (args.model, args.seed))


def cmd_train(args, evaluate_after=False):
    from .dataset import load_split
    from .train import train
    split = load_split(args.split)
    out_dir = _run_dir(args)
    summary = train(args.model, split, out_dir, weights=args.weights, image_size=args.image_size,
                    batch_size=args.batch_size, epochs=args.epochs, epochs_head=args.epochs_head,
                    epochs_finetune=args.epochs_finetune, lr=args.lr, lr_head=args.lr_head,
                    lr_finetune=args.lr_finetune, unfreeze_last=args.unfreeze_last, patience=args.patience,
                    dropout=args.dropout, augment=not args.no_augment, seed=args.seed,
                    max_steps_per_epoch=args.max_steps_per_epoch, save_format=args.save_format)
    print('Trained %s in %.1fs, best val_loss %.4f at epoch %s -> %s' % (
        args.model, summary['training_time_s'], summary['best_val_loss'] or float('nan'),
        summary['epochs_to_converge'], summary['model_path']))
    if evaluate_after:
        args.model_path = summary['model_path']
        return cmd_evaluate(args, training_summary=summary)
    return 0


def cmd_evaluate(args, training_summary=None):
    from .dataset import load_split
    from .evaluate import evaluate
    split = load_split(args.split)
    out_dir = _run_dir(args)
    if training_summary is None:
        summary_path = os.path.join(out_dir, 'training_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, encoding='utf-8') as fh:
                training_summary = json.load(fh)
    model_path = args.model_path
    if not model_path and args.model == 'alexnet_existing':
        from .models import DEFAULT_ALEXNET_PATH
        model_path = DEFAULT_ALEXNET_PATH
    elif not model_path:
        model_path = (training_summary or {}).get('model_path') or os.path.join(out_dir, 'model.keras')
    record = evaluate(args.model, split, out_dir, model_path=model_path, batch_size=args.batch_size,
                      image_size=args.image_size, subset=args.subset, latency_runs=args.latency_runs,
                      flops=not args.no_flops, training_summary=training_summary, max_batches=args.max_batches,
                      seed=args.seed)
    m = record['metrics']
    lat = (record['efficiency']['cpu_latency'] or {}).get('median_ms')
    print('%s on %s (%d images): acc %.4f  macro-F1 %.4f  macro-P %.4f  macro-R %.4f  CPU median %s ms' % (
        args.model, args.subset, m['num_samples'], m['accuracy'], m['f1_macro'], m['precision_macro'],
        m['recall_macro'], '%.1f' % lat if lat else '-'))
    for caveat in record['caveats']:
        print('CAVEAT: ' + caveat)
    print('Results in %s' % out_dir)
    return 0


def cmd_latency(args):
    """Architecture efficiency without a dataset. Latency/params/FLOPs do not depend on weight values."""
    import tensorflow as tf
    from .metrics import environment_info, estimate_flops, measure_cpu_latency, size_on_disk
    from .models import DEFAULT_ALEXNET_PATH, build_model, get_spec, param_counts
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for name in args.models:
        spec = get_spec(name)
        size = int(args.image_size or spec.input_size)
        if name == 'alexnet_existing':
            path = args.model_path or DEFAULT_ALEXNET_PATH
            model = build_model(name, model_path=path)[0]
            disk_mb = size_on_disk(path) / 1e6
            weights_note = 'trained weights (%s)' % os.path.basename(path)
        else:
            model = build_model(name, weights=None, image_size=size)[0]
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, 'model.keras')
                model.save(tmp_path)
                disk_mb = size_on_disk(tmp_path) / 1e6
            weights_note = 'random init (architecture cost only)'
        params = param_counts(model)
        lat = measure_cpu_latency(model, size, n_warmup=args.warmup, n_runs=args.runs)
        flops = estimate_flops(model, size)
        row = {'model': name, 'display_name': spec.display_name, 'input_size': size, 'weights': weights_note,
               'params_total': params['total'], 'params_trainable_when_loaded': params['trainable'],
               'weights_float32_mb': round(params['total'] * 4 / 1e6, 2), 'size_on_disk_mb': round(disk_mb, 2),
               'gflops_per_image': round(flops / 1e9, 3) if flops else None,
               'cpu_latency_median_ms': round(lat['median_ms'], 2), 'cpu_latency_p95_ms': round(lat['p95_ms'], 2),
               'cpu_latency_mean_ms': round(lat['mean_ms'], 2), 'cpu_latency_std_ms': round(lat['std_ms'], 2),
               'timed_runs': lat['timed_runs'], 'warmup_runs': lat['warmup_runs']}
        rows.append(row)
        print('%-18s %4dpx  params %10d  disk %7.1f MB  GFLOPs %-7s  CPU median %7.1f ms  p95 %7.1f ms' % (
            name, size, row['params_total'], row['size_on_disk_mb'], row['gflops_per_image'],
            row['cpu_latency_median_ms'], row['cpu_latency_p95_ms']))
        del model
        tf.keras.backend.clear_session()
    record = {'environment': environment_info(), 'results': rows,
              'note': 'FLOPs counted by the TF profiler (multiply-add = 2 FLOPs). Latency: batch 1, CPU, '
                      'after warm-up; median of timed runs.'}
    with open(os.path.join(args.out_dir, 'efficiency.json'), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2)
    with open(os.path.join(args.out_dir, 'efficiency.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved %s' % os.path.join(args.out_dir, 'efficiency.{json,csv}'))
    return 0


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def cmd_compare(args):
    files = sorted(glob.glob(os.path.join(args.runs_dir, '**', 'metrics.json'), recursive=True))
    if not files:
        print('No metrics.json files under %s' % args.runs_dir)
        return 1
    records = [json.load(open(f, encoding='utf-8')) for f in files]
    fingerprints = {r['split_fingerprint'] for r in records}
    warnings = []
    if len(fingerprints) > 1:
        warnings.append('Runs were evaluated on %d different splits; results are NOT directly comparable.'
                        % len(fingerprints))
    smoke = [r['model'] for r in records if any('SMOKE TEST' in c for c in r.get('caveats', []))]
    if smoke:
        warnings.append('Smoke-test runs present (%s); they are not experimental results.' % ', '.join(smoke))

    by_model = {}
    for r in records:
        by_model.setdefault(r['model'], []).append(r)
    rows = []
    for model, runs in sorted(by_model.items()):
        def field(getter):
            return _mean_std([getter(r) for r in runs])
        row = {'model': model, 'display_name': runs[0]['display_name'], 'num_runs': len(runs),
               'seeds': ' '.join(str(r.get('seed')) for r in runs)}
        for key, getter in [
            ('accuracy', lambda r: r['metrics']['accuracy']),
            ('precision_macro', lambda r: r['metrics']['precision_macro']),
            ('recall_macro', lambda r: r['metrics']['recall_macro']),
            ('f1_macro', lambda r: r['metrics']['f1_macro']),
            ('cpu_latency_median_ms', lambda r: (r['efficiency'].get('cpu_latency') or {}).get('median_ms')),
            ('training_time_s', lambda r: (r.get('training') or {}).get('training_time_s')),
            ('epochs_to_converge', lambda r: (r.get('training') or {}).get('epochs_to_converge')),
        ]:
            row[key + '_mean'], row[key + '_std'] = field(getter)
        eff = runs[0]['efficiency']
        row.update({'params_total': eff['params']['total'], 'size_on_disk_mb': eff.get('size_on_disk_mb'),
                    'weights_float32_mb': eff.get('weights_float32_mb'),
                    'gflops_per_image': round(eff['flops_per_image'] / 1e9, 3) if eff.get('flops_per_image') else None,
                    'test_images': runs[0]['metrics']['num_samples'],
                    'caveats': ' | '.join(sorted({c for r in runs for c in r.get('caveats', [])}))})
        rows.append(row)

    # selection rule stated in the proposal before training:
    # highest macro-F1 among models meeting the latency and size budgets
    eligible = [r for r in rows
                if (r['cpu_latency_median_ms_mean'] or 0) <= args.latency_budget_ms
                and (r['size_on_disk_mb'] or r['weights_float32_mb'] or 0) <= args.size_budget_mb]
    selected = max(eligible, key=lambda r: r['f1_macro_mean']) if eligible else None
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'runs': files, 'split_fingerprints': sorted(fingerprints), 'warnings': warnings, 'models': rows,
               'selection_rule': {'latency_budget_ms': args.latency_budget_ms, 'size_budget_mb': args.size_budget_mb,
                                  'selected': selected['model'] if selected else None}}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    def fmt(mean, std, scale=1.0, digits=4):
        if mean is None:
            return '-'
        return ('%.*f' % (digits, mean * scale)) + (' ± %.*f' % (digits, std * scale) if std else '')
    lines = ['| Model | Runs | Accuracy | Macro-P | Macro-R | Macro-F1 | Params | Size (MB) | CPU median (ms) | Train time (s) | Epochs to converge |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['display_name'], r['num_runs'], fmt(r['accuracy_mean'], r['accuracy_std']),
            fmt(r['precision_macro_mean'], r['precision_macro_std']), fmt(r['recall_macro_mean'], r['recall_macro_std']),
            fmt(r['f1_macro_mean'], r['f1_macro_std']), '{:,}'.format(r['params_total']),
            r['size_on_disk_mb'] or '-', fmt(r['cpu_latency_median_ms_mean'], r['cpu_latency_median_ms_std'], digits=1),
            fmt(r['training_time_s_mean'], r['training_time_s_std'], digits=0),
            fmt(r['epochs_to_converge_mean'], r['epochs_to_converge_std'], digits=1)))
    lines.append('')
    lines.append('Selected by rule (latency <= %g ms, size <= %g MB, max macro-F1): **%s**' % (
        args.latency_budget_ms, args.size_budget_mb, selected['display_name'] if selected else 'none eligible'))
    lines.extend('> WARNING: ' + w for w in warnings)
    with open(os.path.join(args.out_dir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description='Crop-disease architecture benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('info', help='list available models')

    p = sub.add_parser('split', help='create a shared stratified split')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--test_dir', default=None, help='held-out test folder (predefined mode)')
    p.add_argument('--out', required=True)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--test_frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group_regex', default=None, help="regex or preset ('kaggle_augmented') for group-aware split")
    p.add_argument('--max_per_class', type=int, default=0, help='subsample for quick experiments (0 = all)')

    def common(q, need_model=True):
        if need_model:
            q.add_argument('--model', required=True,
                           choices=['alexnet_existing', 'alexnet', 'resnet50', 'efficientnet_b4'])
        q.add_argument('--split', required=True)
        q.add_argument('--seed', type=int, default=42)
        q.add_argument('--runs_dir', default=os.path.join(DEFAULT_RESULTS, 'runs'))
        q.add_argument('--batch_size', type=int, default=32)
        q.add_argument('--image_size', type=int, default=None, help='override the native input size')

    def eval_args(q):
        q.add_argument('--model_path', default=None)
        q.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
        q.add_argument('--latency_runs', type=int, default=100)
        q.add_argument('--no_flops', action='store_true')
        q.add_argument('--max_batches', type=int, default=None, help='smoke test: evaluate only N batches')

    def train_args(q):
        q.add_argument('--weights', default='imagenet', help="'imagenet' or 'none'")
        q.add_argument('--epochs', type=int, default=20, help='single-phase models (alexnet)')
        q.add_argument('--epochs_head', type=int, default=5)
        q.add_argument('--epochs_finetune', type=int, default=15)
        q.add_argument('--lr', type=float, default=1e-3)
        q.add_argument('--lr_head', type=float, default=1e-3)
        q.add_argument('--lr_finetune', type=float, default=1e-5)
        q.add_argument('--unfreeze_last', type=int, default=-1, help='-1 = whole backbone, N = last N layers')
        q.add_argument('--patience', type=int, default=3)
        q.add_argument('--dropout', type=float, default=0.3)
        q.add_argument('--no_augment', action='store_true')
        q.add_argument('--max_steps_per_epoch', type=int, default=None, help='smoke test only')
        q.add_argument('--save_format', default='keras', choices=['keras', 'h5'])

    p = sub.add_parser('train', help='train one model')
    common(p)
    train_args(p)
    p = sub.add_parser('evaluate', help='evaluate one model on the shared test split')
    common(p)
    eval_args(p)
    p = sub.add_parser('run', help='train then evaluate')
    common(p)
    train_args(p)
    eval_args(p)

    p = sub.add_parser('latency', help='params / size / FLOPs / CPU latency without a dataset')
    p.add_argument('--models', nargs='+', default=['alexnet_existing', 'resnet50', 'efficientnet_b4'])
    p.add_argument('--model_path', default=None, help='path to the real Model.hdf5 for alexnet_existing')
    p.add_argument('--image_size', type=int, default=None)
    p.add_argument('--runs', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--out_dir', default=os.path.join(DEFAULT_RESULTS, 'efficiency'))

    p = sub.add_parser('compare', help='aggregate runs into summary tables')
    p.add_argument('--runs_dir', default=os.path.join(DEFAULT_RESULTS, 'runs'))
    p.add_argument('--out_dir', default=DEFAULT_RESULTS)
    p.add_argument('--latency_budget_ms', type=float, default=2000.0)
    p.add_argument('--size_budget_mb', type=float, default=500.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, 'weights', None) == 'none':
        args.weights = None
    handlers = {'info': cmd_info, 'split': cmd_split, 'train': cmd_train, 'evaluate': cmd_evaluate,
                'run': lambda a: cmd_train(a, evaluate_after=True), 'latency': cmd_latency, 'compare': cmd_compare}
    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
