"""
Architecture benchmarking framework: AlexNet (existing + retrainable) vs
ResNet-50 vs EfficientNet-B4 on one shared, stratified PlantVillage split.

CLI: python -m benchmarking.benchmark --help
TensorFlow is imported lazily inside functions so `info` / `split` / `compare`
work without loading it.
"""
