# Implementation Status

Last updated: 2026-11-03. Environment used for verification: Apple M1 Pro (8 cores), macOS, Python 3.11.16, TensorFlow 2.15.1, OpenCV 4.10, NumPy 1.26.4.

Legend: ✅ implemented and tested · 🟡 implemented, not validated on real data · ⏳ pending experiments / external input · ⚠️ known limitation

---

## 1. Baseline (minor project): preserved

| Item | Status | Evidence |
|---|---|---|
| `Model.hdf5` untouched | ✅ | The file in the ZIP checkout is still the 134-byte Git LFS pointer. The real file (129,105,944 bytes, SHA-256 `cfd13975…0f59`, matching the pointer) was downloaded from the GitHub repo to a temporary location for verification only. |
| Existing model loads in TF 2.15 | ✅ | It was saved with Keras 2.2.4 and loads with `tf.keras.models.load_model(..., compile=False)`. 28,117,790 parameters; layer-for-layer identical to the notebook. |
| Preprocessing and label order unchanged | ✅ | `preprocess_image` reproduces the original formula exactly (test). `CLASS_LABELS` equals the list in `evaluate.py` and is in `sorted()` order, as `flow_from_directory` uses (tests). |
| Baseline predictions unchanged | ✅ | `tests/test_real_model.py` (6 sample images, run with `MODEL_PATH`) |
| `/predict` legacy response keys | ✅ | `predicted_crop`, `predicted_disease`, `confidence`, `top3_crops` are unchanged; the original result box still renders (API test, browser check). |
| `evaluate.py`, notebook, `Procfile` | ✅ | Unmodified |
| App start-up when the model is missing or an LFS pointer | ✅ (new behaviour) | The app used to crash at import. Now it logs a clear message and `/predict` returns 503 explaining `git lfs pull`. `/health` reports the status. |

## 2. Extension 1: severity estimation (`severity/`)

| Item | Status | Notes |
|---|---|---|
| Pre-processing: EXIF fix, 512 px working size, Gaussian blur, HSV | ✅ | `tests/test_preprocessing.py` |
| Leaf mask: Otsu on S (clamped) + V floor + blue-violet hue exclusion + weak hue seed, opening/closing, largest contour, hole filling | ✅ | `tests/test_segmentation.py`: IoU > 0.97–0.98 with the drawn leaf on synthetic images, including salt noise, background blobs and blue backgrounds |
| Reliability checks: Otsu separability, saturation std, leaf fraction, border contact (unreliable); solidity (warning) | ✅ | Uniform, tiny-leaf and full-frame-foliage images all return `unreliable`; the real field photo returns `unreliable` |
| Lesion mask: disease-specific HSV ranges, boundary erosion, morphology, small-component removal | ✅ | Area matches ground truth within 1.5 points on synthetic leaves at 2%, 14% and 30% severity |
| Disease-specific profiles for all 26 disease classes; viral/HLB marked unsupported | ✅ | Test checks that every disease class has a valid profile |
| Severity % and bands (Mild < 10 ≤ Moderate ≤ 25 < Severe), configurable, JSON override | ✅ | Boundary tests at 9.99 / 10 / 25 / 25.01 |
| Explicit statuses (`estimated` / `unreliable` / `not_applicable` / `error`); no fabricated numbers | ✅ | Tests |
| Visualisations (original, leaf mask, lesion mask, overlay, 4-panel) | ✅ | Base64 in the API; PNG/JPG via `python -m severity`; real examples in `docs/severity_examples/` |
| Validation tool (Cohen's κ, weighted κ, MAE vs manual grades) | ✅ code / ⏳ data | `python -m severity.validate`. Needs expert-graded images |
| **Accuracy of the severity values on real leaves** | 🟡 ⏳ | HSV ranges were tuned by inspecting ~11 sample images; this is not a validation. Agreement with expert grading has not been measured. |
| Approach B (supervised severity head, proposal slide 15) | ⏳ | Not implemented; needs severity-annotated images |

## 3. Extension 2: treatment recommendation (`treatment/`)

| Item | Status | Notes |
|---|---|---|
| JSON knowledge base covering all 38 classes | ✅ | Test: 38/38, `class_id` = model index, required fields present |
| Decision rules: unknown, healthy, low confidence, systemic/viral, Mild, Moderate, Severe, severity unknown | ✅ | `tests/test_treatment.py`: each rule, the 60% boundary, a custom threshold, and every class × every severity |
| No doses, rates or intervals | ✅ | Regex test over the whole KB; `dosage` and `interval` are always `null` |
| Viral diseases never get curative fungicides; vector guidance given | ✅ | Tests (TYLCV, ToMV, HLB, esca) |
| Sources: 51 unique pages, each fetched and checked for the disease on 2026-11-03 | 🟡 | Raw research is in `treatment/research/`. Indian sources (TNAU, PAU) cover 7 diseases. ICAR sites were unreachable from the build environment. |
| **Expert review of the content** | ⏳ | Status in the KB and UI: "DRAFT - not yet reviewed by an agricultural expert" |
| Active ingredients' registration in India (CIB&RC) | ⏳ | Not verified; flagged in every chemical section |

## 4. Extension 3: Flask integration

| Item | Status |
|---|---|
| Pipeline: upload validation → classifier → severity (skipped for healthy) → treatment → enriched JSON | ✅ API tests + live run with the real model |
| Low-confidence suppression (< 60%, env `LOW_CONFIDENCE_THRESHOLD`) | ✅ |
| Invalid / empty / truncated / unsupported / too-small / too-large uploads → JSON 4xx | ✅ |
| Unique upload filenames (no overwriting) | ✅ |
| `/health` endpoint, `visualize=0` option, per-stage timings | ✅ |
| Measured latency: ~55 ms per request (classify 31 ms, severity 24 ms, rules < 0.05 ms) | ✅ `benchmark_results/efficiency/pipeline_timing.json` |
| Deployment to Heroku with the new dependencies | ⏳ Not re-tested. Check the slug size: TensorFlow + OpenCV + the 129 MB model are close to Heroku's 500 MB limit. |

## 5. Frontend

| Item | Status |
|---|---|
| Original summary box (crop, disease, confidence, top-3 crops) kept | ✅ |
| New panels: uploaded image, prediction + confidence bar + top-3 diseases, severity badge/bar/status/warnings, 4-panel segmentation, management guidance, chemical section (collapsible active ingredients), sources, disclaimer | ✅ Checked in Chrome with the real model; no console errors |
| All server text inserted via `.text()` / escaping (no HTML injection) | ✅ |
| Mobile layout | 🟡 Responsive CSS added; not tested on a phone |

## 6. Extension 4: architecture benchmark (`benchmarking/`)

| Item | Status | Notes |
|---|---|---|
| Model registry: `alexnet_existing`, `alexnet` (notebook rebuild, 28,117,790 params ✅), `resnet50`, `efficientnet_b4` with the shared head GAP → Dropout → Dense(38) | ✅ | `tests/test_benchmarking.py` |
| Model-specific preprocessing (/255; caffe BGR mean; EfficientNet internal) and native sizes 224/224/380 | ✅ | Tests |
| Shared split: stratified 70/15/15 or predefined train/test, fixed seed, SHA-256 fingerprint, group-aware (anti-leakage) | ✅ | Tests: disjoint, stratified, deterministic, zero group leakage |
| Training: two-phase (frozen head → fine-tune with BN frozen), Adam, early stopping, checkpoints, CSV history, best-weights restore, timing, epochs to converge | ✅ smoke-tested | 2-step synthetic run only |
| Evaluation: accuracy + Wilson CI, macro/weighted P/R/F1, balanced accuracy, top-3/5, per-class CSV, confusion matrix CSV/PNG, predictions CSV, params, size, FLOPs, CPU latency | ✅ smoke-tested | |
| `compare`: mean ± std across seeds, split-fingerprint check, pre-registered selection rule, CSV/JSON/Markdown | ✅ smoke-tested | |
| **Efficiency measurements (no training needed)** | ✅ measured | `benchmark_results/efficiency/efficiency.{json,csv}`, see table below |
| ImageNet weight download for ResNet-50 / EfficientNet-B4 | ⏳ | Not exercised here (tests use `weights=None`). The first real `train` downloads ~94 MB + ~71 MB from Keras' storage. |
| **Training ResNet-50 and EfficientNet-B4; accuracy / F1 of all three on the shared split** | ⏳ | Needs the dataset and a GPU (see EXPERIMENTAL_METHODOLOGY.md §6). **No accuracy results exist yet; none are claimed.** |
| 3 seeds per model | ⏳ | Supported via `--seed` + `compare` |

### Measured efficiency (Apple M1 Pro CPU, TF 2.15.1, batch 1, 10 warm-up + 100 timed runs)

| Model | Input | Params | Size on disk | GFLOPs (profiler) | CPU median | CPU p95 |
|---|---|---|---|---|---|---|
| AlexNet (existing `Model.hdf5`) | 224 | 28,117,790 | 129.1 MB | 2.08 | 11.3 ms | 12.4 ms |
| ResNet-50 + 38-class head | 224 | 23,665,574 | 95.2 MB | 7.73 | 38.9 ms | 40.2 ms |
| EfficientNet-B4 + 38-class head | 380 | 17,741,957 | 72.4 MB | 8.92 | 82.8 ms | 86.3 ms |

The ResNet and EfficientNet rows were measured with random weights. Latency, parameter count, size and FLOPs don't depend on the weight values. The latency is a compiled `model(x)` call. In the web app, the classification step (PIL load/resize plus `model.predict()`, which has per-call overhead) measures ~31 ms for the same AlexNet.

## 7. Tests

`pytest` runs 148 tests: 141 pass and 7 skip without the real model; all 148 pass with `MODEL_PATH=/path/to/real/Model.hdf5`.
Fast subset: `pytest -m "not slow"` runs 136 tests in ~2 s.
Synthetic images test the algorithms. **They say nothing about real-world accuracy.**

## 8. Known limitations ⚠️

1. The accuracy of the existing model (94.3%, n = 760) is probably optimistic: the validation folder was monitored during training, and augmentation happened before splitting.
2. The classifier has no reject option: any image is assigned one of 38 classes.
3. HSV severity fails on cast shadows, glare and complex backgrounds (detected and reported as unreliable where the quality checks catch it, but not always; the TomatoHealthy4 shadow is not caught), and on colours that overlap healthy tissue (apple scab, leaf mould). These are flagged `validity: low`.
4. Severity bands are not validated and not disease-specific.
5. The knowledge base is mostly US extension sources and has not been reviewed by an expert. Its active ingredients are not checked against Indian registrations.
6. Softmax confidence is uncalibrated, so the 60% threshold is a heuristic.
7. There is a train/serve resize difference: training used Keras' `nearest` interpolation, while serving uses PIL bicubic. It was preserved, not changed.
8. Tested only on PlantVillage-style images plus one web photo; there has been no field evaluation.

## 9. Files added / modified

Added: `labels.py`, `severity/` (config, segmentation, estimator, visualization, validate, `__main__`), `treatment/` (engine, knowledge_base.json, research/), `benchmarking/` (models, dataset, train, evaluate, metrics, benchmark), `tests/`, `pytest.ini`, `requirements-dev.txt`, `.gitignore`, `docs/severity_examples/`, `benchmark_results/efficiency/`, and the five documentation files.

Modified: `app.py` (lazy and robust model loading, upload validation, enrichment pipeline, `/health`; `model_predict` keeps its signature and original keys), `templates/index.html`, `static/js/main.js`, `static/css/main.css` (new panels; the original box is kept), `requirements.txt` (OpenCV; macOS TensorFlow marker), `README.md` (major-project section).
