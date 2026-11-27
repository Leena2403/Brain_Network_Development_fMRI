# Model evaluation

Task: classify each participant as **child (3–12 y)** or **adult (18–39 y)** from their brain connectivity.
Code: [`notebooks/03_model_development.ipynb`](../notebooks/03_model_development.ipynb), [`notebooks/04_error_analysis.ipynb`](../notebooks/04_error_analysis.ipynb), [`src/evaluation.py`](../src/evaluation.py).

## Setup

| | |
|---|---|
| Sample | 120 participants after motion exclusion (88 children, 32 adults) |
| Features | 741 connections (39 MSDL regions → 39 × 38 / 2 pairs) |
| Model | `StandardScaler` → `LogisticRegression(C=1, class_weight="balanced")` |
| Validation | Stratified 5-fold CV × 10 repeats = 50 test folds |
| Metric | Balanced accuracy (mean of the child and adult hit rates); chance = 0.50 |
| Leakage control | Connectivity measure (incl. tangent reference), scaler and motion regression are fitted on training folds only |

## Results

| Model | Balanced accuracy (mean ± SD) | Worst fold |
|---|---|---|
| Chance (always "child") | 0.500 ± 0.000 | 0.500 |
| Motion only (mean FD) | 0.635 ± 0.100 | 0.417 |
| Partial correlation | 0.875 ± 0.082 | 0.667 |
| Tangent | 0.937 ± 0.051 | 0.833 |
| **Correlation** | **0.945 ± 0.055** | 0.806 |
| Correlation, mean FD regressed out | 0.862 ± 0.085 | 0.611 |

**Permutation test** (correlation features, 5-fold CV, 1,000 label shuffles): real score 0.947; best shuffled score 0.702; **p ≈ 0.001** (no shuffle matched the real score).

## Errors (correlation model)

Confusion matrix, pooled over all 50 test folds (rows = true class):

| | predicted child | predicted adult |
|---|---|---|
| **child** (n = 880 test cases) | **98.6%** | 1.4% |
| **adult** (n = 320 test cases) | 9.4% | **90.6%** |

Averaged over the 10 repeats, **4 of 120** participants are misclassified:

| Participant | True group | Age | Mean FD (mm) | P(adult) |
|---|---|---|---|---|
| sub-pixar109 | child | 10.4 | 0.228 | 0.86 |
| sub-pixar150 | adult | 19.0 | 0.232 | 0.48 |
| sub-pixar144 | adult | 20.0 | 0.319 | 0.06 |
| sub-pixar123 | adult | 27.1 | 0.159 | 0.48 |

Errors cluster near the age boundary (an older child, young adults) rather than among high-motion participants.

## Maturity signal

Among children, the model's out-of-fold P(adult) increases with age: Spearman ρ = 0.42 (p = 5 × 10⁻⁵). It stays significant after controlling for motion (partial ρ = 0.40, p = 1 × 10⁻⁴), although the model never saw age.

## Most-relied-on connections

Frequency in the top 20 |weights| across the 50 folds. All listed connections keep the same sign in 100% of folds.

| Connection | Networks | Direction | Top-20 frequency |
|---|---|---|---|
| R insula – L insula | Cing-Ins | adult-like | 0.90 |
| Broca – ventral ACC | Language – Salience | child-like | 0.88 |
| Cerebellum – L parietal | Cereb – L V Att | adult-like | 0.86 |
| Broca – R anterior insula | Language – Salience | child-like | 0.78 |
| Cerebellum – posterior occipital | Cereb – Occ post | adult-like | 0.76 |
| R STS – L STS | Temporal | adult-like | 0.72 |

## Caveats

- The test folds are small (~6–7 adults each), so one adult error changes a fold's score by ~7 points. The SD above reflects that.
- Children were excluded for motion far more often (28% vs. 3%), so the kept children are a calmer-than-typical subset.
- Regressing out motion is conservative: motion is correlated with group, so it also removes some real developmental signal.
- All results are within a single dataset; no external test set.
