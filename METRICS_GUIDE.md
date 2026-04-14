# Model Evaluation Metrics Guide

The three essential metrics used to evaluate ML classification models are:

## 1. **Accuracy**

**Formula:** `(TP + TN) / (TP + TN + FP + FN)`

**Definition:** Proportion of correct predictions among all predictions

**Meaning:**
- Overall correctness of the model
- How many predictions were right out of total predictions

**When to use:**
- Balanced datasets (similar number of samples per class)
- When all errors have equal importance

**Example:**
- If model makes 100 predictions and 90 are correct: Accuracy = 0.90 (90%)

---

## 2. **Precision**

**Formula:** `TP / (TP + FP)`

**Definition:** Of all positive predictions made, how many were actually correct?

**Meaning:**
- Measure of exactness/reliability
- When model predicts "disease X", how often is it right?

**When to use:**
- When false positives are costly (e.g., diagnosing healthy leaf as diseased)
- Medical/critical applications
- When you want to minimize false alarms

**Example:**
- Model predicts 50 instances of "Apple Scab"
- Actually 40 are correct, 10 are wrong
- Precision = 40/50 = 0.80 (80%)

---

## 3. **Recall (Sensitivity)**

**Formula:** `TP / (TP + FN)`

**Definition:** Of all actual positives, how many did the model find/predict?

**Meaning:**
- Measure of completeness/coverage
- Of all actual sick leaves, how many did model catch?

**When to use:**
- When false negatives are costly (e.g., missing a disease)
- Medical/agricultural applications
- When you want to catch all positives (better to over-predict than miss)

**Example:**
- There are 50 actual instances of "Potato Blight"
- Model finds 45 of them
- Recall = 45/50 = 0.90 (90%)

---

## All Terms Explained

| Term | Meaning |
|------|---------|
| **TP (True Positive)** | Model predicted positive, actually positive ✓ |
| **TN (True Negative)** | Model predicted negative, actually negative ✓ |
| **FP (False Positive)** | Model predicted positive, actually negative ✗ |
| **FN (False Negative)** | Model predicted negative, actually positive ✗ |

---

## Quick Comparison

| Metric | Focus | Question |
|--------|-------|----------|
| **Accuracy** | Overall | How many correct overall? |
| **Precision** | Predicted Positives | When I predict disease, am I right? |
| **Recall** | Actual Positives | Do I find all the diseases? |

---

## Usage in Your Project

Run the evaluation script:

```bash
python evaluate.py --dataset_path "./path/to/test/dataset"
```

Output will show:
- ✓ **Accuracy** - Overall performance
- ✓ **Precision (Weighted & Macro)** - How reliable predictions are
- ✓ **Recall (Weighted & Macro)** - How complete predictions are
- Detailed report per class

---

## For Your Teacher

**You can report these 3 metrics as:**

1. **Accuracy: [X]%** - Overall correctness
2. **Precision: [X]%** - Reliability of disease detection
3. **Recall: [X]%** - Completeness of disease detection

This shows a complete picture of your model's performance beyond just accuracy!
