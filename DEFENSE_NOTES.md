# Defense Notes: Crop Disease Detection, Severity Estimation & Treatment Recommendation

Supervisor-meeting preparation. Every number here was either measured in this repository on 2026-11-03 or comes from a cited paper. Anything not yet done is labelled **pending**.

---

## 0. The 60-second summary

> "Our minor project classifies a leaf photo into one of 38 PlantVillage classes using an AlexNet-style CNN (94.3% accuracy on a 760-image sample). The major project adds three things: (1) an **OpenCV/HSV severity estimator**, which measures the share of the leaf covered by lesions and maps it to Mild/Moderate/Severe; (2) a **rule-based treatment engine** over a JSON knowledge base with verified extension-service sources and no product doses; (3) a **benchmarking framework** for comparing AlexNet with ResNet-50 and EfficientNet-B4 on one shared split. Classification, severity and advice are separate modules. The severity bands and the knowledge base are still experimental and have not been validated by an expert. The ResNet/EfficientNet training runs have not been done yet. The framework is tested, and the efficiency measurements (parameters, size, FLOPs, CPU latency) are complete."

---

## 1. Simple architecture diagram

```
 ┌──────────┐   POST /predict (multipart image)
 │ Browser  │ ───────────────────────────────────────────┐
 │ HTML/JS  │ ◄──────────── enriched JSON ─────────────┐ │
 └──────────┘                                          │ ▼
                                     ┌──────────────────────────────────────┐
                                     │ Flask app (app.py)                   │
                                     │  1 validate upload (PIL verify/load) │
                                     │  2 save to uploads/<uuid>_name       │
                                     └───────────────┬──────────────────────┘
                                                     ▼
                     ┌───────────────────────────────────────────────────┐
                     │ CLASSIFIER (unchanged from minor project)         │
                     │ RGB → 224×224 → /255 → AlexNet-style CNN → softmax│
                     │ → class, confidence, top-3 diseases, top-3 crops  │
                     └───────────────┬───────────────────────────────────┘
                                     │ class label + confidence
                 healthy? ── yes ──► skip severity
                                     │ no
                                     ▼
                     ┌───────────────────────────────────────────────────┐
                     │ SEVERITY  (severity/, OpenCV, no learning)        │
                     │ HSV → Otsu(S) + hue gate → morphology → largest   │
                     │ contour = leaf mask → disease-specific HSV ranges │
                     │ → lesion mask → lesion px / leaf px × 100 → band  │
                     │ + reliability checks (→ "unreliable", no number)  │
                     └───────────────┬───────────────────────────────────┘
                                     │ category / status
                                     ▼
                     ┌───────────────────────────────────────────────────┐
                     │ TREATMENT ENGINE (treatment/, rules + JSON KB)    │
                     │ unknown → healthy → low conf (<60%) → systemic    │
                     │ (viral/HLB/esca) → Mild / Moderate / Severe       │
                     │ → practices, guidance, sources, disclaimer        │
                     └───────────────────────────────────────────────────┘
```

Offline, outside the web request: `benchmarking/` trains and evaluates the architectures on one shared split and writes JSON/CSV results.

---

## 2. Step-by-step: one real prediction request

These are real values from `uploads/PotatoEarlyBlight1.JPG`, sent to the running app with the real `Model.hdf5`:

| # | Step | What happens | Real values |
|---|---|---|---|
| 1 | Upload | Browser sends `multipart/form-data` field `file` to `POST /predict` | 256×256 JPEG |
| 2 | Validation | `PIL.Image.verify()` and `load()`; format must be JPEG/PNG/WEBP/BMP; ≥32 px; ≤16 MB | passes |
| 3 | Save | `uploads/<8-hex-uuid>_PotatoEarlyBlight1.JPG` (uuid avoids name collisions) | |
| 4 | Pre-process | `convert('RGB')` → `resize((224,224))` (PIL bicubic) → `float32 / 255` → batch of 1 | tensor (1,224,224,3) in [0,1] |
| 5 | CNN | `model.predict` → 38 softmax probabilities | argmax = index 20 |
| 6 | Decode | `CLASS_LABELS[20]` = `Potato___Early_blight`; confidence = max prob | **99.25%**; 2nd Late blight 0.64%; 3rd Strawberry leaf scorch 0.10% |
| 7 | Crop top-3 | sum probabilities of the classes of each crop | Potato 99.89%, Strawberry 0.10%, Tomato 0.01% |
| 8 | Confidence rule | 99.25 ≥ 60 → not low-confidence | |
| 9 | Healthy? | no → run severity with profile `Potato___Early_blight` (colours: brown, dark, yellow) | |
| 10 | Seg. pre-process | EXIF orientation fix; rescale 256→512 px (scale 2.0); 5×5 Gaussian blur; RGB→HSV | |
| 11 | Leaf mask | Otsu on S gives t=82 (separability η=0.71); seed = (S>82 ∧ V≥35 ∧ hue not blue-violet) ∪ (weak: S>25, leaf-like hue); open 5×5, close 11×11 ×2; largest contour, filled | leaf = 166,066 px (63.4% of image), solidity 0.97, border contact 2.8% |
| 12 | Checks | η ≥ 0.45, 5% ≤ leaf ≤ 92%, border contact ≤ 60%, S std ≥ 8 | all pass → reliable |
| 13 | Lesion mask | union of HSV boxes inside a 3-px-eroded leaf mask; open 3×3, close 3×3; drop components < 12 px | 26,389 px in 36 components (brown 18,786, yellow 8,201, dark 1,668 px; the colours can overlap) |
| 14 | Severity | 26,389 / 166,066 × 100 | **15.89% → Moderate** (10 ≤ p ≤ 25) |
| 15 | Treatment | KB lookup → rule `severity_moderate` → cultural + preventive + moderate guidance + *advisory* chemical section (generic statements and 10 active ingredients named by the sources, no doses, "requires expert review") | sources: TNAU Agritech, UC IPM |
| 16 | Response | JSON with legacy keys and `prediction`, `severity`, `visualization` (4 base64 images), `treatment`, `warnings`, `timing_ms` | ~183 KB |
| 17 | UI | original summary box, then cards for prediction, severity bar, 4-panel segmentation, guidance, sources, disclaimer | |

Timing on an Apple M1 Pro CPU (median over 45 requests): classification 30.9 ms, severity 24.2 ms, treatment < 0.05 ms, **total 54.8 ms** (`benchmark_results/efficiency/pipeline_timing.json`).

---

## 3. The existing minor project (what we actually built in Semester 6)

**Data.** The Kaggle *"New Plant Diseases Dataset (Augmented)"*: about 87K images in 38 classes, **recreated from PlantVillage by offline augmentation** and split 80/20 into `train/` and `valid/`, plus a separate 33-image `test/` folder. The sample images in `uploads/` appear to come from that test folder (the notebook predicts `test/test/TomatoEarlyBlight1.JPG`).

**Model** (verified by reading `Model.hdf5` itself; it was saved with Keras 2.2.4):

| Layer | Output (224 input) | Params |
|---|---|---|
| Conv 96 × 11×11, stride 4, valid, ReLU | 54×54×96 | 34,944 |
| MaxPool 2×2/2 → BN | 27×27×96 | 384 |
| Conv 256 × **11×11**, stride 1, ReLU | 17×17×256 | 2,973,952 |
| MaxPool → BN | 8×8×256 | 1,024 |
| Conv 384 × 3×3 → BN | 6×6×384 | 886,656 |
| Conv 384 × 3×3 → BN | 4×4×384 | 1,329,024 |
| Conv 256 × 3×3 → MaxPool → BN | 1×1×256 | 886,016 |
| Flatten | 256 | 0 |
| Dense 4096 → Dropout 0.4 → BN | 4096 | 1,069,056 |
| Dense 4096 → Dropout 0.4 → BN | 4096 | 16,797,696 |
| Dense 1000 → Dropout 0.2 → BN | 1000 | 4,101,000 |
| Dense 38 softmax | 38 | 38,038 |
| **Total** | | **28,117,790** |

**Training** (`Crop-Diseases-Detection.ipynb`):
- Weights were loaded from `best_weights_9.hdf5`, and the **first 20 layers were frozen**. That is every conv layer *and* the first two 4096-unit dense layers. Only Dense(1000) → BN → Dense(38) were trained: 4,137,038 of the 28.1M parameters.
- Optimiser: SGD, lr 0.001, momentum 0.9, decay 0.005. Batch size 128, 25 epochs.
- `ImageDataGenerator` augmentation: shear, zoom and shift of 0.2, with rescale 1/255.

**Serving** (`app.py`): Flask with PIL preprocessing → argmax class, confidence, and top-3 *crops* (probabilities summed per crop). Deployed with Heroku + Gunicorn; the 129 MB model is stored in Git LFS.

**Result** (`evaluation_results.json`):
- Accuracy 94.34% on **760 images = 20 per class**, taken as the first 20 files of each folder (not a random sample), most likely from `valid/`.
- 95% Wilson confidence interval: **92.5–95.8%**.
- No F1 score was computed.

### Corrections to the PPT, and how to phrase each one

| PPT says | What the code and model show | How to say it |
|---|---|---|
| "AlexNet transfer learning, ImageNet-pretrained" | Keras has no ImageNet AlexNet. Our network differs from the 2012 AlexNet (conv2 is 11×11, not 5×5; BN instead of LRN), so original AlexNet weights would not fit. The notebook loads `best_weights_9.hdf5` from a Kaggle input named "plant-diseases-classification-using-alexnet". The weights' origin is not documented in the repo, but the evidence points to an earlier PlantVillage training run of the same architecture. | "AlexNet-style CNN, fine-tuned from previously trained weights with the first 20 layers frozen." Don't claim ImageNet pretraining. |
| "Early conv layers frozen; final layers fine-tuned" | *All* conv layers and the first two FC layers are frozen; only the last two dense layers are trained. The notebook comment "train the top 2 conv blocks" is copied text and is wrong. | "Only the final classifier layers (4.1M of 28.1M parameters) were trained." |
| AlexNet "~61M params, ~230 MB, ~0.7 GFLOPs" | Those figures describe the 2012 AlexNet. Ours is measured at **28.1M params, 129.1 MB file** (112.5 MB of weights plus SGD momentum state), **2.08 GFLOPs** (≈1.04 GMACs). | Use the measured numbers. |
| "Predictions below 60% confidence flagged uncertain" | This was not in the minor-project code. | "This is now implemented in the major project." |
| "~95% accuracy" | 94.34% on 760 images; CI 92.5–95.8%; probably inflated (see §10). | "94.3% on a 20-per-class sample of the validation folder." |
| "< 2 s response" | Measured at ~55 ms on the server CPU, excluding upload time. | |
| Per-crop accuracy on slide 23 (97/96/95/93/91%) | No file in the repo produces these numbers. | Regenerate them with `benchmarking evaluate`, which writes `per_class.csv`. |
| ResNet "~4.1 G", EffNet "~4.2 G" FLOPs | These are **MACs** (multiply-accumulates). The TF profiler counts 7.73 and 8.92 GFLOPs (1 MAC = 2 FLOPs). With 38-class heads: 23.7M / 17.7M params, 95.2 / 72.4 MB. | State the convention you use. |
| KB entry with "dosage (from label / ICAR PoP), interval" | Deliberately not implemented: no doses or intervals anywhere, and ICAR sites were unreachable when the KB was built. | "Doses come from the product label and the local officer; the system is advisory." |

---

## 4. How AlexNet transfer learning works

- **AlexNet** (Krizhevsky, Sutskever & Hinton, NIPS 2012): 5 conv + 3 FC layers, ReLU, overlapping max-pooling, dropout, about 60M parameters. It won ILSVRC-2012 with a 15.3% top-5 error.
- **Transfer learning** reuses features learned on a large source task. Early conv filters learn generic edges and colour blobs, middle layers learn textures, and late layers learn task-specific combinations. You **freeze** the generic layers (their gradients are not applied), replace or retrain the classifier head, and optionally **fine-tune** the upper layers with a small learning rate.
- Benefits: less data, faster convergence, and less overfitting, because fewer parameters are trained.
- **In our project:** the frozen part is everything up to the second 4096-unit layer. Only the 1000→38 head learns, using SGD with momentum on 224×224 inputs scaled to [0,1].
- **Softmax** turns the 38 logits zᵢ into probabilities pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}. The loss is categorical cross-entropy, −log p_true.
- The "confidence" we display is the largest pᵢ. It is **not a calibrated probability**: modern networks tend to be over-confident (Guo et al., ICML 2017). That is why the 60% threshold is a heuristic.

---

## 5. Why ResNet-50 and EfficientNet-B4

We picked three generations of CNN design that span the accuracy / size / latency trade-off. All three are available in Keras, two of them with ImageNet weights.

| | AlexNet (2012) | ResNet-50 (2016) | EfficientNet-B4 (2019) |
|---|---|---|---|
| Key idea | deep conv + big FC layers | **residual connections** y = F(x) + x; bottleneck 1×1-3×3-1×1 blocks | **compound scaling** of depth, width and resolution; MBConv (depthwise-separable inverted residual) + squeeze-and-excitation |
| Why include it | our baseline | de-facto standard backbone; skip connections avoid the degradation/vanishing-gradient problem in deep nets | best accuracy per parameter in its family; tests whether efficiency on paper means speed on CPU |
| ImageNet top-1 (Keras docs) | — | 74.9% | 82.9% |
| Params (Keras docs, 1000-class head) | 61M (original) | 25.6M | 19.5M |
| **Measured, 38-class head** | 28.1M | 23.7M | 17.7M |
| Native input | 224 | 224 | 380 |
| **Measured CPU latency (M1 Pro, batch 1, median of 100)** | 11.3 ms | 38.9 ms | 82.8 ms |

**Compound scaling** (Tan & Le, 2019): depth d = α^φ, width w = β^φ, resolution r = γ^φ, subject to α·β²·γ² ≈ 2, so FLOPs grow by about 2^φ. B4 is φ = 4 from the B0 baseline.

**The key observation for the defense:** B4 has the *fewest* parameters but is the *slowest* on CPU. Its 380×380 input and depthwise convolutions have low arithmetic intensity on CPUs. Parameter count is not latency, which is exactly why we measure latency.

---

## 6. How HSV colour segmentation works

- **RGB mixes colour and brightness**: a shadowed green pixel and a lit green pixel have very different RGB values. HSV separates them:
  - **V** = max(R,G,B), the brightness.
  - **S** = (max − min)/max, how vivid the colour is (gray = 0).
  - **H** = the colour angle (red 0°, yellow 60°, green 120°, blue 240°). OpenCV stores H as degrees/2, so H ∈ [0,179]: green ≈ 60, yellow ≈ 30, brown/orange ≈ 5–20.
- **Leaf vs background.** PlantVillage backgrounds are gray or blue-gray (low S, or blue hue), while leaves are saturated green, yellow or brown.
  - We threshold S using **Otsu's method**. It picks the threshold t* that maximises the between-class variance σ_B²(t) = ω₀ω₁(μ₀ − μ₁)², which is the same as minimising the within-class variance.
  - We also report Otsu's own quality measure, **η = σ_B²(t*)/σ_T² ∈ [0,1]**. A low η means the histogram has no clear two-class split, so we flag the result as unreliable.
  - Blue-violet hues (H 96–159) are excluded, because leaves are never blue but backgrounds and cast shadows often are.
  - A **weak seed** adds low-saturation pixels with a leaf-like hue (H 5–95), recovering washed-out or necrotic tissue.
- **Largest external contour, filled** gives a single leaf. Filling it also brings pale or gray lesions inside the leaf back into the leaf area.
- **Lesions** are pixels inside the leaf that fall into disease-specific HSV boxes: brown (H 5–20), yellow (H 17–29), dark (V ≤ 60, but not saturated green), red, purple, white-powder (S ≤ 40, V ≥ 165) and gray-necrotic. Each disease has its own colour list in `severity/config.py`. For example, powdery mildew uses white-powder + yellow, and cedar-apple rust uses orange/yellow/brown/red.

---

## 7. Why morphological operations are used

All of these use an elliptical structuring element B, which is isotropic.

- **Erosion** (minimum filter) shrinks objects. **Dilation** (maximum filter) grows them.
- **Opening = erosion then dilation.** It removes objects smaller than B, such as isolated noisy pixels and JPEG speckle, while larger shapes keep roughly their size. We open the leaf seed with a 5×5 kernel; the test `test_morphological_opening_removes_salt_noise` checks this.
- **Closing = dilation then erosion.** It fills holes and gaps smaller than B and bridges thin breaks along veins (11×11, two iterations, on the leaf mask).
- **Order:** opening first, so noise is removed before closing can merge it into the leaf.
- **Lesion mask:** a 3×3 opening and closing, then removal of connected components under 12 px (at 512-px working size). For common rust we skip the opening, because the pustules are only a few pixels across.
- **Boundary erosion** (3 px) keeps the dark anti-aliased rim of the leaf from counting as "dark lesion".
- Every image is rescaled to a 512 px working size first, so a "5×5 kernel" means the same thing for a 256 px PlantVillage image and a 4000 px phone photo.

---

## 8. How lesion-area severity estimation works

```
severity % = lesion_area / leaf_area × 100 = (# lesion pixels) / (# leaf pixels) × 100
Mild: p < 10      Moderate: 10 ≤ p ≤ 25      Severe: p > 25
```

- **Severity** in plant pathology is the proportion of a plant unit (here, one leaf) showing symptoms. **Incidence** is the proportion of plants infected. We measure leaf-level severity only.
- The value is a **ratio**, so it doesn't depend on image size. A test checks that a 400 px and a 1200 px version agree within 0.5 points.
- The band is computed from the **rounded** value, so the number shown and the band shown can never disagree (9.996 → shown as 10.00 → Moderate).
- **No fabricated numbers:**
  - If the leaf checks fail, the status is `unreliable` with category `Unknown` and no percentage. The strawberry field photo shows this.
  - Viruses and HLB return `not_applicable` (see Q22).
  - Healthy leaves are skipped.
  - Exceptions return `error`.
- **Real outputs** (`docs/severity_examples/`):

| Image | CNN prediction | Severity |
|---|---|---|
| PotatoEarlyBlight1 | Early blight, 99.25% | 15.89%, Moderate |
| TomatoEarlyBlight5 | Early blight, 100% | 6.71%, Mild |
| AppleCedarRust4 | Cedar apple rust, 100% | 23.53%, Moderate |
| PotatoEarlyBlight3 | Early blight, 84% | 0.68%, Mild |
| strawberry-plant-leaf-spot (field photo) | Leaf scorch, 74.7% | **unreliable**: the leaf mask covers 96% of the frame and touches 80% of the border |

---

## 9. Why severity is different from disease classification

- **Classification** answers *"which disease?"*. It is a categorical label learned from the 38 PlantVillage labels.
- **Severity** answers *"how much tissue is affected?"*. It is a continuous quantity.
- **PlantVillage has no severity labels**, so the CNN was never trained to measure severity.
- **Confidence is not severity.** TomatoEarlyBlight5 is classified with 100% confidence yet has only 6.7% lesion area. PotatoEarlyBlight3 (84% confidence) has 0.68%. A clear but small lesion can still be classified with high confidence.
- Management decisions depend on severity: monitor vs act vs remove. That is why the treatment engine takes both inputs.
- A supervised severity model ("Approach B" in the PPT, after Wang et al. 2017) needs annotated severity labels. That is future work. Our heuristic needs no labels but must be validated (§12).

---

## 10. Limitations of PlantVillage (know these cold)

1. **Lab conditions:** a single detached leaf on a uniform background, photographed at US university research sites (Hughes & Salathé 2015). Real field photos have soil, other leaves, shadows and several diseases at once.
2. **Out-of-distribution failure:** Mohanty et al. (2016) reached 99.35% on a held-out PlantVillage test set, but **just above 31%** on images collected from other sources (31.40% / 31.69%).
3. **Background bias:** Noyan (2022, arXiv:2206.04374) reached **49.0% accuracy using only 8 background pixels** (random chance is 2.6%). The labels correlate with capture conditions, and models can exploit this shortcut.
4. **Several photos of the same leaf:** Mohanty et al. kept images of the same leaf together for 41,112 of 54,306 images. The Kaggle copies don't include that mapping, so image-level splits can leak.
5. **Augmentation before splitting:** the Kaggle augmented set was augmented *before* its 80/20 split. Flipped or rotated copies of one photo may sit in both `train/` and `valid/`, which inflates validation accuracy. This is probably part of why our 94% is optimistic. We implemented group-aware splitting (`--group_regex kaggle_augmented`) to prevent it.
6. **Class imbalance in the original release:** roughly ~150 images (potato healthy) up to ~5,500 (orange HLB) per class. Accuracy is then dominated by the large classes, which is why we use macro-F1.
7. **Coverage:** 14 crops and 26 diseases, with one disease per image. There is no "unknown / not a leaf" class, so the model always picks one of 38 (see Q9).
8. **No severity labels**, and no metadata such as region, cultivar or crop stage.

---

## 11. Limitations of HSV segmentation

These are observed on our own sample images, not hypothetical.

| Limitation | Example / effect |
|---|---|
| Cast shadows | In TomatoHealthy4 the dark shadow next to the leaf has a reddish hue and V ≈ 50, so it is segmented as leaf and counted as "dark lesion". That is 14% "severity" on a healthy leaf (healthy predictions skip severity in the app). |
| Over-exposure / glare | Washed-out leaf regions (S ≈ 0, V = 255) carry no colour information and drop out of the leaf mask (AppleCedarRust1). Glare looks like powdery mildew. |
| Complex backgrounds | The strawberry field photo: the leaf merges with surrounding foliage and soil. Caught by the border-contact and leaf-fraction checks and reported as `unreliable`. |
| Symptom colour ≈ healthy colour | Olive apple-scab lesions are missed (AppleScab2: 1.2%). These profiles are marked `validity: low` and the user is warned. |
| Mixed edge pixels | Blur mixes brown lesion edges with green into "yellow" pixels. These count toward the lesion area at its boundary (seen in our synthetic tests). |
| Fixed thresholds | They were tuned by inspecting ~11 sample images, which is **not** a validation set, and they depend on camera white balance. |
| 2-D projection | Curled or tilted leaves change the apparent areas. |
| Systemic symptoms | Mosaic, leaf curl and HLB mottle have no discrete "lesion", so these return `not_applicable`. |

Better approach (future work): a learned lesion-segmentation model (e.g. U-Net) trained on pixel-annotated data. The cost is the annotation effort.

---

## 12. Why our severity thresholds require validation

- **10% / 25% came from the project proposal, not from a phytopathology standard.** Real severity scales are disease-specific: standard area diagrams, or the Horsfall–Barratt 12-grade logarithmic scale (1945), which itself has been criticised for biased estimates (see Bock et al. 2010).
- The **measurement** is biased per disease. It under-estimates scab and spider-mite stippling and can over-estimate when shadows are present. A correct band therefore needs both a correct measurement and correct cut-offs.
- Bands should match **action thresholds** (when spraying pays off), and those depend on the crop, its growth stage, the weather and the economics.
- **Validation plan (tool implemented: `python -m severity.validate`):**
  1. Sample 100–200 images per selected disease.
  2. Have 2 annotators grade each image Mild/Moderate/Severe (optionally with a %), **blind to the heuristic**.
  3. Report coverage, exact agreement, **Cohen's κ** and **linear-weighted κ** (the bands are ordinal, so Mild↔Severe errors should count more than adjacent errors), the confusion matrix, and the MAE in percentage points.
  4. Report the inter-annotator κ as the realistic ceiling.
- **κ = (p_o − p_e)/(1 − p_e)**: agreement corrected for chance. Rough guide (Landis & Koch): 0.41–0.60 moderate, 0.61–0.80 substantial.

---

## 13. Evaluation metrics, and why macro-F1 matters

For class c: **precision** P_c = TP/(TP+FP) ("when we say c, how often are we right?"), **recall** R_c = TP/(TP+FN) ("of the real c, how many did we find?"), and **F1** = 2·P·R/(P+R), the harmonic mean.

- **Macro** average: the mean over classes, with each disease counting equally. **Weighted** average: weighted by class support.
- **Why macro-F1:**
  - Accuracy and weighted scores are dominated by the large classes.
  - Our test `test_macro_f1_exposes_minority_class_failure` shows the gap: with 95 majority samples right and all 5 minority samples wrong, accuracy is **95%** but macro-F1 is **below 0.5**.
  - Missing a rare disease is exactly the error that matters, and macro-F1 exposes it.
- **Professor trap: "Why is your macro recall exactly equal to your accuracy?"**
  - Our 760-image evaluation had exactly 20 images per class.
  - With equal support, macro recall = (1/C)·Σ TP_c/20 = ΣTP/(20C) = accuracy.
  - For the same reason, weighted = macro in `evaluation_results.json`. It's a property of the balanced sample, not a bug.
- **Confusion matrix:** rows = true class, columns = predicted. It shows *which* diseases get confused, e.g. early vs late blight. The benchmark writes it as CSV and PNG.
- **Uncertainty:** we report a Wilson 95% CI for accuracy. With n = 760 it is ±1.6 points, so differences under about 2 points between models are not meaningful on a sample that size.

---

## 14. How the rule-based recommendation engine works

- Inputs: `(class_label, confidence %, severity category, severity status)`. Output: a structured advisory.
- **Rules apply in priority order**, and the names of the rules that fired are returned in `decision.rules_applied`, so every output can be traced:
  1. **Unknown class** (not in the KB) → generic advice, "consult an expert", no chemical section.
  2. **Healthy** → preventive practices only.
  3. **Confidence < 60%** → no disease-specific management and **no chemical guidance**. Only generic safe practices, plus "retake the photo / confirm with an expert". The disease info is marked *tentative*.
  4. **Systemic, no cure** (viral TYLCV/ToMV, HLB, esca) → `no_curative_chemical`: remove infected plants and manage the vector (whitefly, psyllid). **Never a curative fungicide.** ToMV has no insect vector, so the advice is sanitation.
  5. **Mild** → cultural practices + monitoring + prevention; chemical "not indicated".
  6. **Moderate** → plus disease-specific management, plus an *advisory* chemical section.
  7. **Severe** → plus removal of heavily infected material, plus expert consultation.
  8. **Severity unknown** (segmentation unreliable) → moderate-level guidance with "base your decision on a field inspection" and expert consultation.
- **Knowledge base** (`treatment/knowledge_base.json`):
  - **38/38 classes** covered (a unit test checks this).
  - 26 disease entries, each with: class_id, crop, disease, pathogen, pathogen type, symptoms, cultural and preventive practices, mild/moderate/severe recommendations (or severity-independent management for systemic diseases), a chemical-guidance section (`dosage: null`, `requires_expert_review`), and sources.
- **Sources:**
  - 51 unique source pages (87 references across entries, because healthy entries reuse their crop's sources), e.g. UC IPM, Ohio State, Penn State, UF/IFAS, Cornell, NC State and Crop Protection Network.
  - Indian sources (TNAU Agritech Portal, Punjab Agricultural University) are used for 7 disease entries (potato and tomato).
  - Every URL was fetched and checked for that disease on 2026-11-03. The raw research records are in `treatment/research/`.
- **Status:** a draft, *not yet reviewed by an agricultural expert*. That status is shown in the UI.

---

## 15. Why we separated ML inference from treatment logic

1. **Different kinds of knowledge.** The CNN learns statistical patterns from images. Treatment advice is expert, regulated knowledge: label claims, regional practices.
2. **Updatable without retraining.** Fixing a recommendation means editing JSON, not retraining for hours.
3. **Auditable and reviewable.** An agronomist can review JSON and rules, which they can't do with network weights. Every item cites a source.
4. **Swappable classifier.** If the benchmark selects ResNet-50, the severity and treatment modules don't change, because they only consume `(label, confidence)`.
5. **Errors can be analysed independently.** A wrong recommendation is either a classification error, a severity error, or a KB error, and each module has its own tests.
6. **Safety.** Deterministic, conservative rules (low-confidence suppression, no curative chemicals for viruses, no doses) are easy to guarantee and test in a rule engine. They would be hard to guarantee in a learned recommender.

---

## 16. Questions a professor may ask, with answers

**Q1. Where did the ImageNet AlexNet weights come from?**
Keras has no pretrained AlexNet. Our architecture also differs from the 2012 network (11×11 conv2, BatchNorm), so ImageNet AlexNet weights would not fit. The notebook loads `best_weights_9.hdf5` from an earlier run. The PPT's "ImageNet-pretrained" claim is incorrect, and we have corrected it to "fine-tuned from previously trained weights".

**Q2. What exactly was frozen?**
The first 20 layers: all conv layers and the first two 4096-unit dense layers. Only 4.14M of the 28.1M parameters were trained.

**Q3. Why is your accuracy (94.3%) lower than Mohanty's 99.35%?**
- They trained full networks (GoogLeNet/AlexNet) end to end with leaf-grouped splits. We trained only the classifier head on top of frozen features.
- Our number comes from just 760 images (CI 92.5–95.8%), chosen as the first 20 files per class.
- The training data differs (the augmented Kaggle copy).

**Q4. Is 94.3% trustworthy?**
- It is probably **optimistic**: `valid/` was monitored during training, and the offline augmentation was done before the split, so copies of one photo may appear in both.
- It is also a small, non-random sample.
- The benchmark's `predefined` mode re-evaluates on the full `valid/` folder. The group-aware `stratified` mode gives a leak-free split.

**Q5. Why 224 and not 227?**
With 'valid' padding and stride 4, 227 gives the original AlexNet's 55×55 conv1 output. Our network was built for 224 (54×54) and trained at 224, so we keep it.

**Q6. Why is the model file 129 MB?**
28.1M float32 weights take 112.5 MB. The remaining ~16 MB is SGD momentum state for the trainable layers, saved with `model.save`. Most parameters sit in the FC layers (Dense 4096×4096 alone has 16.8M), which is the classic AlexNet inefficiency.

**Q7. How do you make the architecture comparison fair?**
- One saved split file with a SHA-256 fingerprint: every result records it, and `compare` warns if they differ.
- Identical augmentation and the same optimiser (Adam) and early-stopping rule for all models.
- Each model gets its own native input size and `preprocess_input`.
- Same test images, same metrics code, and latency measured on the same CPU (batch 1, warm-up, median of 100).
- Multiple seeds, reported as mean ± std.
- The selection rule is fixed **before** training: the highest macro-F1 that meets the latency and size budgets.

**Q8. How can you evaluate the existing AlexNet fairly if it was trained on the Kaggle `train/`?**
Use `predefined` mode: train the other models on that same `train/` (with validation carved out of it) and test everyone on `valid/`. In `stratified` mode on raw PlantVillage the evaluation code attaches a contamination warning, because the old model may have seen those images. A retrainable `alexnet` (the notebook architecture) is provided for a like-for-like comparison. Note it has no ImageNet weights, which is a stated confound.

**Q9. What happens if someone uploads a photo of a cat or a field?**
- The softmax always picks one of the 38 classes; there is no reject class. That is a real limitation.
- Mitigations: the 60% low-confidence rule, and severity checks that flag non-leaf images as `unreliable`.
- A proper fix is open-set recognition, or an extra "not a leaf / unknown" class (future work).

**Q10. Is softmax confidence a probability? Why 60%?**
Not a calibrated one. Networks are often over-confident (Guo et al. 2017). The 60% comes from our proposal. A principled threshold would be chosen after temperature scaling, using a reliability diagram on the validation set.

**Q11. Why HSV and not RGB or LAB?**
HSV separates chromaticity (H, S) from brightness (V), so "brown", "yellow" and "gray background" become simple ranges and are less sensitive to illumination than RGB. LAB's b* channel (yellow–blue) would also separate leaf from background, and it is a reasonable alternative or addition. HSV was the method proposed, and it is easy to explain.

**Q12. How did you choose the HSV ranges?**
- From colour theory (brown/orange H 5–20, yellow 17–29, healthy green ~35–60).
- Then by inspecting HSV channel images and failure cases on our ~11 sample leaves. Two concrete fixes came from that: the hue gate for blue backgrounds, and excluding dark green from "dark".
- That is tuning, **not** validation. The ranges are in `severity/config.py` and can be overridden per disease or via a JSON file.

**Q13. What if Otsu fails?**
- Its threshold is clamped to [35, 140].
- We compute η, and if η < 0.45 the result is marked unreliable.
- Other checks: saturation std, leaf fraction (5–92%), and border contact (≤ 60%). A failed check means no number is reported.

**Q14. Why fill the leaf contour?**
Gray or white lesions (septoria centres, mildew) have low saturation and would be holes in the saturation mask. Excluding them would *shrink the denominator and drop the numerator*, which under-estimates severity. The side effect is that real shot-holes also count as leaf area.

**Q15. Why not train a segmentation network?**
It needs pixel-level lesion masks, which PlantVillage doesn't have. The HSV heuristic needs no labels, is explainable, and runs in ~24 ms. A U-Net is listed as future work, once annotation is possible.

**Q16. How will you validate severity?**
With expert-graded samples, Cohen's κ / weighted κ and MAE, using the implemented `severity.validate` tool (§12). Results are **pending annotation**.

**Q17. Where do the treatment recommendations come from, and are they verified?**
- Each source URL was fetched and confirmed to cover the disease; the raw records are in `treatment/research/`.
- The content is paraphrased and tagged with its source.
- ICAR pages couldn't be reached during the build, so most sources are US extension services, with TNAU/PAU used where available.
- It is a **draft pending expert review**, and the UI says so.

**Q18. Why no pesticide doses?**
- Doses depend on the product formulation, the crop, the region and the regulations. In India the authority is the CIB&RC-approved label claim.
- Wrong doses can harm crops, people and the environment.
- The system names only active ingredients that the cited sources themselves name, marks them "requires expert review", and points to the label and the local KVK.

**Q19. Why rule-based instead of ML for recommendations?**
- There is no training data mapping images to correct treatments.
- Rules are transparent, testable and editable.
- Safety constraints (low-confidence suppression, no fungicide for viruses) must hold every time, not merely be likely to hold.

**Q20. Why is EfficientNet-B4 slower than ResNet-50 even with fewer parameters?**
- Its 380×380 input means ~2.9× more pixels.
- It is deeper (258 vs 107 layers in Keras).
- Depthwise convolutions and squeeze-and-excitation do little computation per byte of memory moved, which CPUs handle poorly.
- Measured: 82.8 ms vs 38.9 ms. Keras' own table shows the same pattern (308 ms vs 58 ms per step on CPU).

**Q21. FLOPs vs MACs?**
A multiply-accumulate (MAC) is one multiply plus one add, i.e. 2 FLOPs. Papers often quote MACs and call them "FLOPs". The TF profiler counts both operations, so ResNet-50 shows 7.73 GFLOPs ≈ 3.9 GMACs ≈ the usual "4.1 G".

**Q22. Why don't viral diseases get a severity value?**
Their symptoms are systemic (mosaic, curling, stunting) rather than discrete lesions, so "lesion pixels ÷ leaf pixels" doesn't measure disease. Reporting a number would be fabricated precision. Treatment for viruses is the same at any visible level anyway: remove infected plants and control the vector.

**Q23. What is data leakage, and how do you prevent it?**
Information from the test set reaching training, e.g. augmented copies of the same photo in both sets. We prevent it with group-aware splits (`StratifiedGroupKFold` on file stems with augmentation suffixes stripped), report `group_leakage` counts, and never select models on the test set.

**Q24. Why keep BatchNorm frozen during fine-tuning?**
With small batches, updating BN statistics on the new data can destabilise the pretrained features. Calling `base(x, training=False)` keeps BN in inference mode, which is the Keras-recommended practice.

**Q25. How is overfitting controlled?**
Dropout (head 0.3; AlexNet 0.4/0.4/0.2), data augmentation, early stopping on validation loss with best-weight restoration, and frozen backbones during the first phase.

**Q26. What is your contribution beyond existing work?**
An integrated, deployed and tested pipeline: classification → measurable severity, with explicit reliability reporting → sourced, safety-constrained advice. Plus a reproducible benchmark protocol. We don't claim new state-of-the-art accuracy.

**Q27. How would you add a new disease?**
- The classifier needs new labelled images and retraining.
- The severity profile and KB entry are just configuration and JSON edits; no code changes.
- Tests then enforce that the new class has a profile and a KB entry with sources.

**Q28. Mobile deployment?**
Export to TFLite with post-training quantization (int8 cuts size ~4×), then re-measure accuracy and latency on the device. This is future work. The benchmark already reports size and CPU latency.

**Q29. Why do you report a confidence interval?**
760 test images give ±1.6 points. Without intervals, a "1% improvement" could be noise.

**Q30. What is still not done?**
- Training ResNet-50 and EfficientNet-B4 (needs the dataset and a GPU).
- The severity validation annotations.
- Expert review of the KB.
- Field-image testing.
- `IMPLEMENTATION_STATUS.md` lists everything.

---

## 17. Numbers to have on one card

| Item | Value |
|---|---|
| Classes / crops / diseases | 38 / 14 / 26 |
| Existing model | 28,117,790 params; 129.1 MB file; 224×224 input |
| Minor-project accuracy | 94.34% on 760 images (CI 92.5–95.8%) |
| CPU latency, batch 1 (M1 Pro, TF 2.15) | AlexNet 11.3 ms · ResNet-50 38.9 ms · EffNet-B4 82.8 ms |
| GFLOPs (profiler) | 2.08 · 7.73 · 8.92 |
| Params (38-class heads) | 28.1M · 23.7M · 17.7M |
| End-to-end request | ~55 ms (classify 31 + severity 24 + rules <0.05) |
| KB coverage / sources | 38/38 classes; 51 unique verified source pages (7 diseases with Indian sources); draft, not expert-reviewed |
| Tests | 148 pass with the real model; 141 pass + 7 skipped without it |
| Severity bands | Mild < 10% ≤ Moderate ≤ 25% < Severe (experimental) |
| Low-confidence threshold | 60% |

## References

- Krizhevsky, Sutskever & Hinton (2012). ImageNet classification with deep CNNs. NIPS.
- He, Zhang, Ren & Sun (2016). Deep residual learning for image recognition. CVPR. arXiv:1512.03385
- Tan & Le (2019). EfficientNet: Rethinking model scaling for CNNs. ICML. arXiv:1905.11946
- Hughes & Salathé (2015). An open access repository of images on plant health… arXiv:1511.08060
- Mohanty, Hughes & Salathé (2016). Using deep learning for image-based plant disease detection. Front. Plant Sci. 7:1419. doi:10.3389/fpls.2016.01419
- Noyan (2022). Uncovering bias in the PlantVillage dataset. arXiv:2206.04374
- Otsu (1979). A threshold selection method from gray-level histograms. IEEE Trans. SMC 9(1).
- Bock, Poole, Parker & Gottwald (2010). Plant disease severity estimated visually, by digital photography and image analysis, and by hyperspectral imaging. Crit. Rev. Plant Sci. 29(2).
- Horsfall & Barratt (1945). An improved grading system for measuring plant diseases. Phytopathology 35.
- Wang, Sun & Wang (2017). Automatic image-based plant disease severity estimation using deep learning. Comput. Intell. Neurosci.
- Guo, Pleiss, Sun & Weinberger (2017). On calibration of modern neural networks. ICML.
- Keras Applications documentation (parameter counts, ImageNet accuracy): https://keras.io/api/applications/
- Treatment sources: see `treatment/knowledge_base.json` (per entry) and `treatment/research/`.
