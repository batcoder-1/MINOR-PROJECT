"""
Model registry for the architecture benchmark.

  alexnet_existing  the trained Model.hdf5 from the minor project, evaluated as-is
  alexnet           the same AlexNet-style architecture as Crop-Diseases-Detection.ipynb,
                    trainable on the shared split. Keras ships no ImageNet AlexNet, so
                    this variant starts from random weights (a known confound vs. the
                    ImageNet-pretrained ResNet-50 / EfficientNet-B4 - report it).
  resnet50          ImageNet ResNet-50 backbone + shared head
  efficientnet_b4   ImageNet EfficientNet-B4 backbone + shared head

Shared head (project proposal): GlobalAveragePooling -> Dropout -> Dense(38, softmax).

Every spec carries its own input size and ``preprocess_input`` so each network
sees inputs in the range/format it was designed for. All preprocessing
functions take float32 RGB in [0, 255].
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from labels import NUM_CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ALEXNET_PATH = os.environ.get('MODEL_PATH', os.path.join(ROOT, 'Model.hdf5'))


def _scale_01(x):
    return x / 255.0


def _resnet_preprocess(x):
    import tensorflow as tf
    return tf.keras.applications.resnet50.preprocess_input(x)  # RGB->BGR, ImageNet mean subtraction (caffe)


def _efficientnet_preprocess(x):
    import tensorflow as tf
    return tf.keras.applications.efficientnet.preprocess_input(x)  # identity: model rescales/normalises inside


@dataclass
class ModelSpec:
    name: str
    display_name: str
    input_size: int
    preprocess: Callable
    trainable: bool
    pretrained: str
    two_phase: bool               # freeze backbone -> train head -> fine-tune
    notes: str = ''
    reference_params: Optional[str] = None
    extra: dict = field(default_factory=dict)


MODEL_SPECS = {
    'alexnet_existing': ModelSpec(
        'alexnet_existing', 'AlexNet (existing Model.hdf5)', 224, _scale_01, trainable=False,
        pretrained='weights from the minor project (see notebook)', two_phase=False,
        notes='Evaluate-only. Trained on the Kaggle augmented PlantVillage train/ split.'),
    'alexnet': ModelSpec(
        'alexnet', 'AlexNet-style (notebook architecture, retrained)', 224, _scale_01, trainable=True,
        pretrained='none (random init)', two_phase=False,
        notes='Same layers as Crop-Diseases-Detection.ipynb; no ImageNet weights exist in Keras.'),
    'resnet50': ModelSpec(
        'resnet50', 'ResNet-50', 224, _resnet_preprocess, trainable=True, pretrained='imagenet', two_phase=True,
        reference_params='25.6M with the 1000-class ImageNet head (Keras docs)'),
    'efficientnet_b4': ModelSpec(
        'efficientnet_b4', 'EfficientNet-B4', 380, _efficientnet_preprocess, trainable=True, pretrained='imagenet',
        two_phase=True, reference_params='19.5M with the 1000-class ImageNet head (Keras docs)'),
}


def get_spec(name):
    if name not in MODEL_SPECS:
        raise KeyError('Unknown model %r; choose from %s' % (name, ', '.join(MODEL_SPECS)))
    return MODEL_SPECS[name]


def build_alexnet_notebook(num_classes=NUM_CLASSES, input_size=224):
    """Layer-for-layer rebuild of the AlexNet-style network in Crop-Diseases-Detection.ipynb."""
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((input_size, input_size, 3)),
        L.Conv2D(96, 11, strides=4, padding='valid', activation='relu'),
        L.MaxPooling2D(2, 2, padding='valid'), L.BatchNormalization(),
        L.Conv2D(256, 11, strides=1, padding='valid', activation='relu'),
        L.MaxPooling2D(2, 2, padding='valid'), L.BatchNormalization(),
        L.Conv2D(384, 3, padding='valid', activation='relu'), L.BatchNormalization(),
        L.Conv2D(384, 3, padding='valid', activation='relu'), L.BatchNormalization(),
        L.Conv2D(256, 3, padding='valid', activation='relu'),
        L.MaxPooling2D(2, 2, padding='valid'), L.BatchNormalization(),
        L.Flatten(),
        L.Dense(4096, activation='relu'), L.Dropout(0.4), L.BatchNormalization(),
        L.Dense(4096, activation='relu'), L.Dropout(0.4), L.BatchNormalization(),
        L.Dense(1000, activation='relu'), L.Dropout(0.2), L.BatchNormalization(),
        L.Dense(num_classes, activation='softmax'),
    ], name='alexnet_notebook')


def _with_shared_head(base, input_size, num_classes, dropout, name):
    import tensorflow as tf
    inputs = tf.keras.Input((input_size, input_size, 3), name='image')
    # training=False keeps BatchNormalization in inference mode even after the
    # backbone is unfrozen for fine-tuning (standard Keras transfer-learning practice)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dropout(dropout, name='head_dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_model(name, num_classes=NUM_CLASSES, weights='imagenet', image_size=None, dropout=0.3, model_path=None):
    """Return (model, backbone). ``backbone`` is None for single-phase models."""
    import tensorflow as tf
    spec = get_spec(name)
    size = int(image_size or spec.input_size)
    if name == 'alexnet_existing':
        path = model_path or DEFAULT_ALEXNET_PATH
        return tf.keras.models.load_model(path, compile=False), None
    if name == 'alexnet':
        return build_alexnet_notebook(num_classes, size), None
    if name == 'resnet50':
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=(size, size, 3))
    elif name == 'efficientnet_b4':
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights=weights, input_shape=(size, size, 3))
    else:
        raise KeyError(name)
    return _with_shared_head(base, size, num_classes, dropout, name), base


def load_trained_model(name, model_path):
    import tensorflow as tf
    if name == 'alexnet_existing':
        return build_model(name, model_path=model_path)[0]
    return tf.keras.models.load_model(model_path, compile=False)


def set_backbone_trainable(backbone, unfreeze_last=-1, train_batchnorm=False):
    """Freeze/unfreeze backbone layers for fine-tuning.

    unfreeze_last = 0   -> whole backbone frozen (head-only training)
    unfreeze_last = N   -> only the last N backbone layers trainable
    unfreeze_last = -1  -> entire backbone trainable
    BatchNormalization layers stay frozen unless ``train_batchnorm`` is True.
    """
    import tensorflow as tf
    layers = backbone.layers
    if unfreeze_last == 0:
        backbone.trainable = False
        return 0
    backbone.trainable = True
    cutoff = 0 if unfreeze_last < 0 else max(0, len(layers) - unfreeze_last)
    count = 0
    for i, layer in enumerate(layers):
        trainable = i >= cutoff
        if isinstance(layer, tf.keras.layers.BatchNormalization) and not train_batchnorm:
            trainable = False
        layer.trainable = trainable
        count += int(trainable)
    return count


def param_counts(model):
    import numpy as np
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {'total': trainable + non_trainable, 'trainable': trainable, 'non_trainable': non_trainable}
