# Major Project Architecture

**Crop Disease Detection, Severity Estimation & Treatment Recommendation using Deep Learning**

## 1. Overview

The minor project is a Flask web app that classifies a leaf image into one of 38 PlantVillage classes with an AlexNet-style CNN. The major project keeps that path unchanged and adds three decoupled components:

| Component | Package | Kind | Runs in the web request? |
|---|---|---|---|
| Disease classifier | `app.py` + `Model.hdf5` | learned (CNN) | yes |
| Severity estimator | `severity/` | deterministic image processing (OpenCV, HSV) | yes (skipped for healthy) |
| Treatment engine | `treatment/` | deterministic rules over a JSON knowledge base | yes |
| Architecture benchmark | `benchmarking/` | training / evaluation tooling | no (offline) |

The components only share plain values (a class label string, a confidence %, a severity category), so any one of them can be replaced without touching the others.

## 2. Repository layout

```
app.py                      Flask app: model loading, upload validation, pipeline, routes
labels.py                   38 class labels (model output order) + name helpers
Model.hdf5                  trained AlexNet-style model (Git LFS; unchanged)
severity/
  config.py                 every threshold: bands, Otsu clamps, HSV ranges, per-disease profiles
  segmentation.py           pre-processing, Otsu, leaf mask, lesion mask (pure functions)
  estimator.py              severity %, band, statuses, reliability handling
  visualization.py          original / leaf mask / lesion mask / overlay / panel
  validate.py               agreement with manual grading (Cohen's kappa)
  __main__.py               CLI: python -m severity --image ... --label ...
treatment/
  engine.py                 decision logic only
  knowledge_base.json       38 entries + decision-rule texts + disclaimer (no doses)
  research/                 raw source-verification records (provenance)
benchmarking/
  models.py                 AlexNet (existing / rebuilt), ResNet-50, EfficientNet-B4 + shared head
  dataset.py                scan, stratified / group-aware split, tf.data pipeline
  train.py                  two-phase transfer learning, early stopping, checkpoints
  evaluate.py               metrics + efficiency -> JSON/CSV/PNG
  metrics.py                classification metrics, Wilson CI, latency, FLOPs, size
  benchmark.py              CLI: info | split | train | evaluate | run | latency | compare
templates/, static/         UI (original summary box + new analysis panels)
tests/                      pytest suite (synthetic images, fake classifier, optional real model)
benchmark_results/          machine-readable results (efficiency measured; training pending)
docs/severity_examples/     real pipeline outputs on the sample images
```

## 3. Request data flow

```
POST /predict  (multipart field "file")
  │
  ├─ 400  no file / empty / not decodable / unsupported format / < 32 px
  ├─ 413  > MAX_UPLOAD_MB (16)
  ├─ 503  model not loaded (e.g. Model.hdf5 is a Git LFS pointer)
  ▼
save uploads/<uuid8>_<secure_filename>
  ▼
model_predict(path, model)                     [unchanged algorithm]
  PIL RGB → resize 224×224 → /255 → model.predict → softmax(38)
  → class_index, class_label, confidence, top3_crops (summed per crop), top3_diseases
  ▼
low_confidence = confidence < 60 %             (LOW_CONFIDENCE_THRESHOLD)
  ▼
healthy? ── yes ─► severity = {status: skipped}
  │ no
  ▼
estimate_severity(path, class_label)
  profile = DISEASE_PROFILES[class_label]
  supported? ── no ─► status not_applicable (viral, HLB)
  load + EXIF → rescale to 512 → blur → HSV
  segment_leaf → reliable? ── no ─► status unreliable, category Unknown, percentage null
  segment_lesions(profile colours) → % = lesion/leaf×100 → band
  ▼
build_visualizations (4 base64 images)        (skipped if visualize=0)
  ▼
treatment_engine.recommend(label, confidence, severity.category, severity.status)
  ▼
JSON response ──► static/js/main.js renders the summary box + analysis panels
```

## 4. API response schema (`POST /predict`, 200)

```jsonc
{
  "success": true,
  // legacy keys (minor-project frontend contract, unchanged)
  "predicted_crop": "Potato", "predicted_disease": "Early Blight", "confidence": 99.25,
  "top3_crops": [{"crop": "Potato", "confidence": 99.89}, ...],
  "class_index": 20, "class_label": "Potato___Early_blight", "top3_diseases": [...],

  "prediction": {
    "class_index": 20, "class_label": "Potato___Early_blight", "crop": "Potato", "disease": "Early Blight",
    "is_healthy": false, "confidence": 99.25, "low_confidence": false, "low_confidence_threshold": 60.0,
    "top3_diseases": [{"class_label": "...", "crop": "...", "disease": "...", "confidence": 0.64}, ...],
    "top3_crops": [...]
  },
  "severity": {
    "status": "estimated",               // estimated | unreliable | not_applicable | skipped | error
    "category": "Moderate",              // Mild | Moderate | Severe | Unknown | Not applicable
    "percentage": 15.89,                 // null unless status == estimated
    "lesion_area_px": 26389, "leaf_area_px": 166066,
    "method": "hsv_lesion_area_ratio", "experimental": true,
    "profile": "Potato___Early_blight", "lesion_colors": ["brown", "dark", "yellow"], "validity": "heuristic",
    "note": "...", "bands": {"mild_upper": 10.0, "moderate_upper": 25.0},
    "quality": {"otsu_threshold_raw": 82, "otsu_separability": 0.7104, "leaf_fraction": 0.6335, ...},
    "color_breakdown_px": {...}, "lesion_components": 36, "warnings": []
  },
  "visualization": {"original": "data:image/jpeg;base64,...", "leaf_mask": "data:image/png;...",
                    "lesion_mask": "...", "overlay": "...", "legend": {...}, "source": "..."},
  "treatment": {
    "class_label": "...", "crop": "Potato", "disease": "Early blight", "found_in_knowledge_base": true,
    "disease_info": {"pathogen": "Alternaria solani", "pathogen_type": "fungal", "symptoms": [...],
                     "vector": null, "tentative": false},
    "decision": {"confidence": 99.25, "low_confidence": false, "low_confidence_threshold": 60.0,
                 "severity_category": "Moderate", "severity_status": "estimated",
                 "rules_applied": ["severity_moderate"]},
    "cultural_practices": [...], "preventive_measures": [...], "monitoring": [...],
    "management_guidance": [...], "vector_management": [],
    "chemical_guidance": {"status": "advisory_requires_expert_review", "summary": "...",
                          "generic_guidance": [...], "active_ingredients_named_by_source": [...],
                          "dosage": null, "interval": null, "requires_expert_review": true,
                          "label_note": "...", "notes": [...]},
    "expert_consultation": {"recommended": false, "reason": null},
    "sources": [{"title": "...", "organization": "...", "url": "https://..."}],
    "review_status": "DRAFT - not yet reviewed by an agricultural expert",
    "disclaimer": "...", "warnings": []
  },
  "warnings": [],                        // request-level warnings (low confidence, unreliable severity)
  "timing_ms": {"classification": 30.9, "severity": 24.2, "treatment": 0.0, "total": 54.8},
  "model": {"name": "AlexNet-style CNN (Model.hdf5)", "input_size": [224, 224], "num_classes": 38}
}
```

`chemical_guidance.status` is one of `not_indicated`, `suppressed_low_confidence`, `no_curative_chemical` or `advisory_requires_expert_review`.

Errors: `{"success": false, "error": "<message>"}` with 400 / 413 / 503 / 500. `GET /predict` returns `{"error": "Method not allowed"}` with 405, as before.

`GET /health` → `{status, model_loaded, model_error, knowledge_base_coverage, knowledge_base_missing, low_confidence_threshold}`.

## 5. Design decisions

| Decision | Reason |
|---|---|
| Keep `model_predict(img_path, model)` and its output keys | Preserve baseline behaviour and the frontend contract; new keys are additive only |
| Load the model lazily and fail gracefully | A ZIP checkout contains only the LFS pointer; the old app crashed at import. Now it reports the problem through `/predict` (503) and `/health` |
| Severity as image processing, not a learned model | PlantVillage has no severity labels; HSV + Otsu is explainable, label-free and fast (~24 ms) |
| Explicit `unreliable` / `not_applicable` statuses | Never show a number the method can't support (e.g. field photos, viral mosaics) |
| Per-disease colour profiles in config | Lesion colours differ by disease (white mildew vs brown blight vs orange rust); tunable without code changes |
| 512 px working resolution | Kernel sizes mean the same thing at every input resolution; the ratio is scale-invariant |
| Rules + JSON KB, separated from ML | Auditable, editable by experts, testable safety constraints, classifier swappable |
| No doses / intervals | Must come from the product label and local authority; avoids unverifiable or harmful advice |
| Benchmark offline, with a shared split file + fingerprint | Reproducible, provably identical evaluation data across models and seeds |
| PIL bicubic resize for all benchmark models | Matches the deployed preprocessing; only the per-model normalisation differs |

## 6. Configuration

| Setting | Where | Default |
|---|---|---|
| Model path | env `MODEL_PATH` | `<repo>/Model.hdf5` |
| Low-confidence threshold | env `LOW_CONFIDENCE_THRESHOLD` (or `decision_rules.low_confidence_threshold` in the KB when the engine is used standalone) | 60 |
| Max upload size | env `MAX_UPLOAD_MB` | 16 |
| Severity thresholds / HSV ranges | `severity/config.py`, or env `SEVERITY_CONFIG=override.json` | see file |
| Skip model loading (tests) | env `SKIP_MODEL_LOAD=1` | off |

## 7. Running

```bash
# Python 3.9-3.11 (TensorFlow 2.15). With uv:  uv venv --python 3.11 .venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # requirements.txt for production only

# Model.hdf5 must be the real file, not the LFS pointer:
git lfs install && git lfs pull              # in a git clone
# or: export MODEL_PATH=/path/to/Model.hdf5

python app.py                                # http://127.0.0.1:5000
gunicorn app:app --bind 0.0.0.0:5000         # production
curl -s localhost:5000/health

pytest                                       # 141 pass + 7 skipped (real-model tests)
MODEL_PATH=/path/to/Model.hdf5 pytest        # all 148
python -m severity --image uploads/PotatoEarlyBlight1.JPG --label Potato___Early_blight --out_dir severity_outputs
python -m benchmarking.benchmark --help
```
