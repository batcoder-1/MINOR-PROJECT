"""
Training for the architecture benchmark.

Two-phase transfer learning (ResNet-50, EfficientNet-B4):
    phase 1  backbone frozen, train the new head      (Adam, lr_head)
    phase 2  unfreeze the backbone (or its last N layers, BatchNorm kept frozen),
             fine-tune with a small learning rate      (Adam, lr_finetune)
Single phase (AlexNet-style network trained from scratch): Adam, lr.

Every phase uses early stopping on validation loss; the weights with the lowest
validation loss over the whole run are restored at the end and saved. The
epoch of that minimum is reported as "epochs to converge".
"""

import json
import math
import os
import time

from .dataset import make_tf_dataset, split_fingerprint
from .models import build_model, get_spec, param_counts, set_backbone_trainable


def _callbacks(tf, out_dir, phase, patience, tracker):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'checkpoint_%s.weights.h5' % phase),
                                           monitor='val_loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, 'history.csv'), append=True),
        tracker,
    ]


def _make_tracker(tf):
    class BestWeights(tf.keras.callbacks.Callback):
        """Keeps the weights with the lowest val_loss seen across all phases."""

        def __init__(self):
            super().__init__()
            self.best = math.inf
            self.best_epoch = None
            self.best_logs = None
            self.weights = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            value = logs.get('val_loss')
            if value is not None and value < self.best:
                self.best, self.best_epoch, self.best_logs = float(value), int(epoch), dict(logs)
                self.weights = self.model.get_weights()

    return BestWeights()


def train(model_name, split, out_dir, weights='imagenet', image_size=None, batch_size=32, epochs=20,
          epochs_head=5, epochs_finetune=15, lr=1e-3, lr_head=1e-3, lr_finetune=1e-5, unfreeze_last=-1,
          patience=3, dropout=0.3, augment=True, seed=42, max_steps_per_epoch=None, save_format='keras'):
    """Train one architecture on ``split`` and save it to ``out_dir``. Returns the training summary dict."""
    import tensorflow as tf

    spec = get_spec(model_name)
    if not spec.trainable:
        raise ValueError('%s is evaluate-only' % model_name)
    tf.keras.utils.set_random_seed(seed)
    size = int(image_size or spec.input_size)
    os.makedirs(out_dir, exist_ok=True)
    history_csv = os.path.join(out_dir, 'history.csv')
    if os.path.exists(history_csv):
        os.remove(history_csv)

    train_ds = make_tf_dataset(split['train'], size, spec.preprocess, batch_size, training=True, augment=augment,
                               seed=seed)
    val_ds = make_tf_dataset(split['val'], size, spec.preprocess, batch_size)
    if max_steps_per_epoch:
        train_ds, val_ds = train_ds.take(max_steps_per_epoch), val_ds.take(max_steps_per_epoch)

    model, backbone = build_model(model_name, weights=weights if spec.pretrained == 'imagenet' else None,
                                  image_size=size, dropout=dropout)
    tracker = _make_tracker(tf)
    phases = []

    def run_phase(phase, learning_rate, n_epochs, initial_epoch):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        t0 = time.perf_counter()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch + n_epochs,
                         initial_epoch=initial_epoch, callbacks=_callbacks(tf, out_dir, phase, patience, tracker),
                         verbose=2)
        elapsed = time.perf_counter() - t0
        ran = len(hist.epoch)
        phases.append({'phase': phase, 'learning_rate': learning_rate, 'epochs_requested': n_epochs,
                       'epochs_run': ran, 'early_stopped': ran < n_epochs, 'time_s': round(elapsed, 2),
                       'trainable_params': param_counts(model)['trainable'],
                       'final_val_loss': float(hist.history['val_loss'][-1]),
                       'final_val_accuracy': float(hist.history['val_accuracy'][-1])})
        return initial_epoch + ran

    if spec.two_phase:
        set_backbone_trainable(backbone, 0)
        epoch = run_phase('head', lr_head, epochs_head, 0)
        if epochs_finetune > 0:
            unfrozen = set_backbone_trainable(backbone, unfreeze_last)
            phases_note = 'unfroze %d backbone layers (BatchNorm frozen)' % unfrozen
            run_phase('finetune', lr_finetune, epochs_finetune, epoch)
            phases[-1]['note'] = phases_note
    else:
        run_phase('full', lr, epochs, 0)

    if tracker.weights is not None:
        model.set_weights(tracker.weights)
    model_path = os.path.join(out_dir, 'model.%s' % ('h5' if save_format == 'h5' else 'keras'))
    model.save(model_path)

    summary = {
        'model': model_name, 'display_name': spec.display_name, 'seed': seed, 'image_size': size,
        'pretrained_weights': weights if spec.pretrained == 'imagenet' else spec.pretrained,
        'augment': augment, 'batch_size': batch_size, 'patience': patience, 'dropout': dropout,
        'split_fingerprint': split['meta'].get('fingerprint') or split_fingerprint(split),
        'split_counts': {k: len(split[k]) for k in ('train', 'val', 'test')},
        'max_steps_per_epoch': max_steps_per_epoch,
        'phases': phases,
        'training_time_s': round(sum(p['time_s'] for p in phases), 2),
        'epochs_run_total': sum(p['epochs_run'] for p in phases),
        'epochs_to_converge': None if tracker.best_epoch is None else tracker.best_epoch + 1,
        'best_val_loss': None if tracker.best_epoch is None else tracker.best,
        'best_val_accuracy': None if tracker.best_logs is None else float(tracker.best_logs.get('val_accuracy')),
        'params': param_counts(model),
        'model_path': model_path,
        'smoke_test': bool(max_steps_per_epoch),
    }
    with open(os.path.join(out_dir, 'training_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    return summary
