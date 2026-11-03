# Experimental Methodology

This document defines the experiments the major project will report. It says what has already been measured and what is still pending. **No accuracy result for ResNet-50 or EfficientNet-B4 exists yet.**

---

## 1. Research questions

1. **RQ1 (architecture).** On one identical PlantVillage split, how do AlexNet, ResNet-50 and EfficientNet-B4 compare on macro-F1, accuracy, macro precision/recall, CPU latency, size and training cost?
2. **RQ2 (severity).** How well does the HSV lesion-area heuristic agree with manual severity grading (Mild / Moderate / Severe)?
3. **RQ3 (decision support).** Does every class resolve to a sourced rule set, and do the safety rules (low confidence, viral, no doses) always hold? This one is answered by tests and is **done**.

---

## 2. Datasets

| Dataset | Use | Notes |
|---|---|---|
| Kaggle "New Plant Diseases Dataset (Augmented)", `train/` + `valid/` | **Primary for RQ1** (`predefined` mode) | The existing `Model.hdf5` was trained on this `train/`, so `valid/` is the only fair test set for it. Offline augmentation happened before the 80/20 split, so leakage is possible *within* the dataset; it affects all models equally. |
| Original PlantVillage "color" (54,306 images) | Secondary RQ1 (`stratified` mode, 70/15/15) | Imbalanced. The existing model may have seen these images in augmented form, so its scores here carry an automatic contamination caveat. Use the retrained `alexnet` arm instead. |
| Field images (e.g. PlantDoc) | Future robustness test | Not used yet |

Folder names must match `labels.CLASS_LABELS` exactly. Missing or unrecognised folders are reported by `split`.

---

## 3. Splits (identical evaluation data)

```bash
# Primary (fair to the existing model). Validation is carved from train/ and grouped by source photo.
python -m benchmarking.benchmark split \
    --data_dir "data/New Plant Diseases Dataset(Augmented)/train" \
    --test_dir "data/New Plant Diseases Dataset(Augmented)/valid" \
    --group_regex kaggle_augmented --val_frac 0.15 --seed 42 --out splits/kaggle_seed42.json

# Secondary: stratified 70/15/15 on original PlantVillage
python -m benchmarking.benchmark split --data_dir data/plantvillage/color \
    --val_frac 0.15 --test_frac 0.15 --seed 42 --out splits/pv_seed42.json
```

- **Stratified:** every class keeps its proportion in each subset (scikit-learn `train_test_split(stratify=…)`).
- **Group-aware:** `--group_regex` removes augmentation suffixes (`_flipLR`, `_180deg`, `_new30degFlipLR`, …) from the file stem, so all copies of one source photo share a group (`StratifiedGroupKFold`). `meta.group_leakage` reports how many groups overlap between subsets; it should be 0. **Check the preset against your copy of the dataset's file names.**
- **Fingerprint:** a SHA-256 hash of (subset, class/file, label). Every `metrics.json` stores it, and `compare` warns if runs used different splits.
- The split file is created **once** and reused for every model and seed. Commit it.

---

## 4. Models and training protocol

| | `alexnet_existing` | `alexnet` | `resnet50` | `efficientnet_b4` |
|---|---|---|---|---|
| Weights | Model.hdf5 | random init (no Keras ImageNet AlexNet) | ImageNet | ImageNet |
| Input | 224 | 224 | 224 | 380 |
| Pre-processing | /255 | /255 | caffe (RGB→BGR, mean subtraction) | built into the model |
| Head | original | original (notebook) | GAP → Dropout(0.3) → Dense(38) | GAP → Dropout(0.3) → Dense(38) |
| Training | none (evaluate only) | 1 phase, Adam 1e-3 | phase 1: head only, Adam 1e-3, 5 epochs; phase 2: whole backbone (BN frozen), Adam 1e-5, ≤ 15 epochs | same as ResNet-50 |

- **Shared across all trained models:** the same split, the same augmentation (random flip horizontal + vertical, rotation ±0.1 turn, brightness ±15%, contrast ±15%, applied on [0,255] before preprocessing), sparse categorical cross-entropy, batch 32, early stopping on val_loss (patience 3), and restoring the weights with the minimum val_loss across both phases.
- **Seeds:** 3 per model (42, 1, 2), reported as mean ± std.
- **Image loading:** PIL RGB + bicubic resize, identical to the deployed app.
- **Confound (state it in the report):** the retrained AlexNet has no ImageNet pretraining, and the existing AlexNet was trained with a different recipe (SGD, frozen layers, Keras `ImageDataGenerator`).

**Pre-registered selection rule** (proposal slide 11): choose the model with the highest mean macro-F1 whose CPU median latency is ≤ 2000 ms and whose size is ≤ the budget (`--size_budget_mb`, default 500). `compare` applies the rule automatically.

---

## 5. Metrics

| Metric | Definition | Implementation |
|---|---|---|
| Accuracy | correct / N, with a Wilson 95% CI | `metrics.classification_metrics` |
| Macro precision / recall / F1 | unweighted mean over the classes present in the test set | scikit-learn, `zero_division=0` |
| Weighted P/R/F1, balanced accuracy, top-3/top-5 accuracy | standard definitions | |
| Confusion matrix | 38×38, rows = true class | CSV + row-normalised PNG |
| Per-class P/R/F1/support | | `per_class.csv` |
| Parameters | trainable / non-trainable / total | `param_counts` |
| Size | bytes on disk of the saved model, plus float32 weight size | `size_on_disk` |
| FLOPs | TF profiler on the frozen graph (1 MAC = 2 FLOPs) | `estimate_flops` |
| CPU latency | batch 1, `tf.device('/CPU:0')`, compiled `tf.function`, 10 warm-up + 100 timed runs; median, p90, p95, mean, std | `measure_cpu_latency` |
| Training time / epochs to converge | wall-clock per phase; epoch of minimum val_loss | `train.py` |

**The primary metric is macro-F1**, because every disease matters equally. Accuracy is reported alongside it for comparison with the literature.

---

## 6. How to run the training (GPU recommended)

CPU training of EfficientNet-B4 at 380 px over ~70K images is impractical. Use Google Colab or Kaggle with a GPU:

```bash
git clone https://github.com/batcoder-1/MINOR-PROJECT && cd MINOR-PROJECT && git lfs pull
pip install -r requirements-dev.txt          # on Linux this installs tensorflow-cpu ...
pip install "tensorflow[and-cuda]==2.15.1"   # ... so add the GPU build on a GPU machine
# make the dataset available under data/ (Kaggle: add the "new-plant-diseases-dataset" input)
python -m benchmarking.benchmark split ... --out splits/kaggle_seed42.json
for s in 42 1 2; do
  python -m benchmarking.benchmark run --model resnet50        --split splits/kaggle_seed42.json --seed $s
  python -m benchmarking.benchmark run --model efficientnet_b4 --split splits/kaggle_seed42.json --seed $s --batch_size 16
  python -m benchmarking.benchmark run --model alexnet         --split splits/kaggle_seed42.json --seed $s
done
python -m benchmarking.benchmark evaluate --model alexnet_existing --model_path Model.hdf5 --split splits/kaggle_seed42.json
python -m benchmarking.benchmark compare --runs_dir benchmark_results/runs
```

**Python version caveat.** TensorFlow 2.15 supports Python 3.9–3.11 and Keras 2. If the GPU environment ships Python 3.12+ (and TF ≥ 2.16 with Keras 3), there are two options:
1. Create a Python 3.11 environment (`uv venv --python 3.11`, conda, or a 3.11 runtime image). This is **recommended** and matches the tested setup.
2. Use the pre-installed TF with `pip install tf-keras` and `export TF_USE_LEGACY_KERAS=1`. This path is **untested** in this repository; run `pytest -m slow` first.

Then re-run `latency` on the **deployment CPU** (the server), because latency on a Colab CPU isn't comparable. Copy the `benchmark_results/runs/*/metrics.json` files back into the repository.

Quick pilot, if time is short: add `--max_per_class 200` to `split`. That gives about 6,500 train images (38 × 200, minus 15% for validation), and its results must be labelled "pilot (200/class)".

---

## 7. Severity validation (RQ2)

1. Pick 2–3 diseases with clear lesions (e.g. `Tomato___Early_blight`, `Potato___Late_blight`, `Apple___Black_rot`), and **one low-validity disease** (e.g. `Apple___Apple_scab`) to document the failure mode.
2. `python -m severity.validate template --images_dir <valid>/<class> --label <class> --n 150 --out annotations/<class>.csv`
3. Two annotators grade every image independently (Mild / Moderate / Severe, optionally an estimated %), **without seeing the heuristic output**. Use a standard area diagram as a reference where one exists.
4. `python -m severity.validate score --annotations annotations/<class>.csv --out annotations/<class>_results.json`
5. Report:
   - **coverage** (share of images graded as `estimated`)
   - **exact agreement**
   - **Cohen's κ** and **linear-weighted κ**
   - the 3×3 confusion matrix
   - the number of Mild↔Severe disagreements
   - **MAE** in percentage points
   - the **inter-annotator κ**, as a ceiling
6. If agreement is poor: first inspect the overlays to see whether the error is in the measurement or the cut-offs. Re-tune the per-disease HSV ranges or bands on **one half** of the annotations and report on the **other half**. Never tune and report on the same images.

---

## 8. Already measured (2026-11-03, Apple M1 Pro CPU, TF 2.15.1)

| Model | Params | Size | GFLOPs | CPU median (batch 1) |
|---|---|---|---|---|
| AlexNet (existing) | 28.1M | 129.1 MB | 2.08 | 11.3 ms |
| ResNet-50 (38-class head) | 23.7M | 95.2 MB | 7.73 | 38.9 ms |
| EfficientNet-B4 (38-class head, 380 px) | 17.7M | 72.4 MB | 8.92 | 82.8 ms |

End-to-end web request with the existing model: classification 30.9 ms + severity 24.2 ms + treatment < 0.05 ms = 54.8 ms (median of 45).

Minor-project evaluation (`evaluation_results.json`): accuracy 0.9434 on 760 images (20/class). The Wilson 95% CI is [0.925, 0.958], which `benchmarking.metrics.wilson_interval` reproduces.

## 9. Threats to validity

- **Dataset bias:** results on PlantVillage do not transfer to the field (Mohanty et al. 2016: ~31% on outside images; Noyan 2022: 8 background pixels → 49%).
- **Leakage:** augmented copies across subsets, and multiple photos of one leaf. The group-aware split mitigates the first; the second cannot be fully controlled without leaf IDs.
- **Unequal starting points:** ImageNet vs random vs a previously trained AlexNet.
- **Hyper-parameters** are not tuned per model. One shared recipe is used, so the comparison is "out-of-the-box with an equal budget", not "best achievable".
- **Latency** depends on hardware, thread settings and TF version. Always report the environment (it is stored in `metrics.json`).
- **Small differences:** on the full `valid/` folder (~17.5K images) the 95% CI half-width is only ~0.3 points, so seed-to-seed variance, not test-sample size, will dominate. Report mean ± std over 3 seeds before claiming a winner. (On the minor project's 760-image sample the CI half-width was ±1.6 points.)
