# Technical Explanation

The algorithms and the mathematics behind each module, with the actual parameter values used in the code.

---

## 1. Classification (preserved minor-project model)

### 1.1 Pre-processing (`app.preprocess_image`)

```
x = PIL.open(path).convert('RGB').resize((224, 224))     # PIL default: bicubic for RGB
x = float32(x) / 255                                     # range [0, 1]
x = x[None, ...]                                         # shape (1, 224, 224, 3)
```

This must match what the network saw in training (`ImageDataGenerator(rescale=1./255)`). The one difference: Keras resized training images with nearest-neighbour interpolation, while serving uses bicubic. It is left unchanged to preserve the baseline.

### 1.2 Network

AlexNet-style Sequential model; see DEFENSE_NOTES §3 for the layer/parameter table. Output size after each conv/pool layer, with 'valid' padding:

```
out = floor((in - k) / s) + 1
224 -(11,s4)-> 54 -(pool2)-> 27 -(11,s1)-> 17 -(pool2)-> 8 -(3)-> 6 -(3)-> 4 -(3)-> 2 -(pool2)-> 1
```

The flattened feature vector has only 256 values, so 60% of all parameters sit in the 4096×4096 dense layer.

### 1.3 Outputs

- Softmax: p_i = exp(z_i) / Σ_j exp(z_j). Prediction ŷ = argmax p. Confidence = max p × 100.
- **Top-3 crops:** P(crop) = Σ p_i over that crop's classes. This is the original feature.
- **Top-3 diseases:** the three largest p_i. This is new and matches the proposal slides.
- **Low-confidence flag:** confidence < 60. Softmax scores are not calibrated, so the threshold is a heuristic.

---

## 2. Severity estimation (`severity/`)

### 2.1 Colour space

OpenCV 8-bit HSV for an RGB pixel with M = max(R,G,B), m = min(R,G,B):

```
V = M
S = 255 · (M − m) / M            (0 if M = 0)
H = ½ · { 60·(G−B)/(M−m)          if M = R   (mod 360)
          60·(B−R)/(M−m) + 120    if M = G
          60·(R−G)/(M−m) + 240    if M = B }        → H ∈ [0, 179]
```

Pure red is H = 0, yellow 30, green 60, blue 120 (unit test `test_rgb_to_hsv_opencv_convention`).

### 2.2 Pre-processing

1. Load as RGB and apply the EXIF orientation.
2. Reject inputs smaller than 64 px.
3. Rescale so the longest side is **512 px** (INTER_AREA when shrinking, bilinear when enlarging).
4. Apply a 5×5 Gaussian blur.
5. Convert RGB → HSV.

### 2.3 Otsu's threshold on saturation

For the histogram p(k), k = 0..255, and a threshold t (class 0 is k ≤ t):

```
ω(t) = Σ_{k≤t} p(k)          μ(t) = Σ_{k≤t} k·p(k)          μ_T = μ(255)
σ_B²(t) = (μ_T·ω(t) − μ(t))² / (ω(t)·(1 − ω(t)))          t* = argmax σ_B²(t)
η = σ_B²(t*) / σ_T²      (separability, 0..1)
```

The implementation (`segmentation.otsu_threshold`) is tested against `cv2.THRESH_OTSU`. The threshold actually used is t = clip(t*, 35, 140).

### 2.4 Leaf mask

```
strong = (S > t) ∧ (V ≥ 35) ∧ ¬(H ∈ [96, 159])                       # not blue/violet
weak   = (S > 25) ∧ (35 ≤ V ≤ 250) ∧ (H ∈ [5, 95])                    # washed-out / necrotic leaf
seed   = strong ∪ weak
seed   = close(open(seed, E5), E11 ×2)                                 # E_k = k×k ellipse
leaf   = filled(largest external contour of seed)
```

**Reliability.** The result is *unreliable* if any of the following holds:
- std(S) < 8
- η < 0.45
- leaf fraction < 5% or > 92%
- border contact > 60%

A warning (not a failure) is added if solidity (contour area / convex hull area) < 0.70 or border contact > 35%.

### 2.5 Lesion mask

For the disease profile's colour list C (e.g. `Potato___Early_blight` → brown, dark, yellow):

```
search = erode(leaf, E7)                                   # 3-px boundary margin
raw    = ( ∪_{c∈C} inRange(HSV, box_c) ) ∧ search
lesion = close(open(raw, E3), E3)
lesion = remove connected components with < 12 px
lesion = lesion ∧ leaf
```

| Colour | H | S | V |
|---|---|---|---|
| brown | 5–20 | 40–255 | 40–175 |
| orange | 5–20 | 100–255 | 176–255 |
| yellow | 17–29 | 80–255 | 100–255 |
| dark | 0–33 or 96–179 (any S); 34–95 only if S ≤ 60 | | 0–60 |
| red | 165–4 (wraps around 0) | 70–255 | 40–255 |
| purple | 125–164 | 50–255 | 30–220 |
| white_powder | any | 0–40 | 165–255 |
| gray_necrotic | any | 0–45 | 56–164 |
| olive_dark | 18–40 | 40–255 | 56–100 |

Healthy leaves are typically H ≈ 35–60, which is why yellow stops at 29 and "dark" excludes saturated green.

### 2.6 Severity and bands

```
p = round(100 · |lesion| / |leaf|, 2)
band(p) = Mild if p < 10 ;  Moderate if 10 ≤ p ≤ 25 ;  Severe if p > 25
```

Because p is a ratio of pixel counts, it is invariant to uniform rescaling. The test compares 400 px and 1200 px versions, which agree within 0.5 points.

### 2.7 Statuses

| Status | When | Percentage |
|---|---|---|
| `estimated` | all checks pass | number |
| `unreliable` | leaf checks fail, or input too small | `null`, category `Unknown` |
| `not_applicable` | healthy, or profile `supported: false` (TYLCV, ToMV, HLB) | `null` |
| `skipped` | (app) healthy prediction; the estimator isn't called | `null` |
| `error` | exception (message in warnings) | `null` |

### 2.8 Visualisation

The overlay blends lesion pixels as 0.45 · original + 0.55 · red (230, 30, 30), and draws the leaf contour in yellow. All four views are generated from the working-resolution image and the same masks used for the calculation.

---

## 3. Treatment engine (`treatment/`)

Rules are evaluated in priority order; the first one that matches decides the branch:

```
entry = KB[class_label]
if entry is None                          → unknown_disease_fallback
elif healthy                              → healthy_preventive_only (+ low-confidence note)
elif confidence < 60                      → low_confidence_suppression
elif no_curative_chemical or viral        → systemic_no_curative_chemical
elif severity == Mild                     → severity_mild
elif severity == Moderate                 → severity_moderate
elif severity == Severe                   → severity_severe
else (Unknown / Not applicable)           → severity_unknown
```

| Branch | cultural | preventive | management | chemical status | expert |
|---|---|---|---|---|---|
| unknown | generic | – | – | (none) | yes |
| healthy | – | crop-level | – | not_indicated | only if low conf. |
| low confidence | generic safe practices | – | – | suppressed_low_confidence | yes |
| systemic | entry | entry | entry.management + vector | no_curative_chemical | yes |
| mild | entry | entry | entry.mild | not_indicated | no |
| moderate | entry | entry | entry.moderate | advisory_requires_expert_review | no |
| severe | entry | entry | entry.severe + generic removal | advisory_requires_expert_review | yes |
| severity unknown | entry | entry | entry.moderate | advisory + field-inspection note | yes |

The advisory chemical section contains the sources' generic statements and the active ingredients those sources name. It **never** includes a dose or an interval (`dosage: null`, `interval: null`), and it always carries the label-compliance note (CIB&RC label claim, local KVK).

---

## 4. Benchmark metrics

For class c with TP_c, FP_c, FN_c:

```
P_c = TP_c / (TP_c + FP_c)      R_c = TP_c / (TP_c + FN_c)      F1_c = 2 P_c R_c / (P_c + R_c)
macro-X  = (1/|C|) Σ_c X_c                       (C = classes present in the test set)
weighted-X = Σ_c (n_c / N) X_c
balanced accuracy = macro recall
```

With equal support n_c = n, macro recall = Σ TP_c / (n|C|) = accuracy. That is why the minor project's balanced 760-image evaluation shows recall_macro = accuracy.

Wilson 95% interval for k successes in n:

```
centre = (p̂ + z²/2n) / (1 + z²/n)
half   = z·sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n),   z = 1.96
```

For 717/760 this gives [0.925, 0.958].

Cohen's κ (severity agreement): κ = (p_o − p_e)/(1 − p_e), where p_e = Σ_k p_manual(k)·p_heuristic(k). The linear-weighted κ gives an adjacent-band disagreement half the penalty of a Mild↔Severe one.

**Latency:** each timed run executes `forward(x).numpy()`, so the call blocks until the result is ready. The median is reported because it is robust to OS scheduling spikes.

**FLOPs:** the Keras model is wrapped in a `tf.function`, frozen to constants, and passed through `tf.compat.v1.profiler` with `float_operation()`. Multiply and add are counted separately.

---

## 5. Transfer-learning protocol (`benchmarking/train.py`)

1. Build the backbone (`include_top=False`, ImageNet weights) and the shared head. The backbone is called with `training=False`, so its BatchNorm layers always use their moving statistics.
2. **Phase 1:** `backbone.trainable = False`; Adam(1e-3); train the head for ≤ 5 epochs.
3. **Phase 2:** unfreeze the backbone (all of it, or the last N layers), keep BN layers frozen, and use Adam(1e-5) for ≤ 15 more epochs. The 100× smaller learning rate avoids destroying the pretrained features.
4. EarlyStopping(val_loss, patience 3), a per-phase checkpoint, a CSV history, and a callback that keeps the weights with the global minimum val_loss. Those weights are restored before saving `model.keras`.
5. Recorded: per-phase epochs, time, trainable parameters, best val_loss/accuracy, and epochs to converge (= 1 + the index of the best epoch).
