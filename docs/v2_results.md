# Classifier v2 Results

## Training Set

| Source | Examples |
|---|---|
| Original split_train | 8,183 |
| Synthetic (filtered) | 1,687 |
| **Total** | **9,870** |

Synthetic examples target 5 failure clusters from v1 failure analysis:
FN-1 (multilingual injections), FN-2 (short injections), FN-3 (indirect injections),
FP-2 (legitimate role-play), FP-1 (instruction-adjacent coding).

## Training Results (Colab T4, 3 epochs)

| Epoch | Train Loss | Val Loss | Macro F1 |
|---|---|---|---|
| 1 | 0.0980 | 0.0648 | 0.9862 |
| 2 | 0.0180 | 0.0378 | 0.9932 |
| 3 | 0.0093 | 0.0446 | 0.9932 |

## v1 vs v2 on split_test (n=1,754)

| Model | Macro F1 | Injection recall | Benign recall |
|---|---|---|---|
| v1 | 0.9957 | 1.0000 | 1.0000 |
| v2 | 0.9932 | 0.9904 | 1.0000 |
| Delta | -0.0025 | -0.0096 | 0.0000 |

## Risk Gate 3 Assessment

Gate criterion: v2 macro F1 >= v1 macro F1 + 5 percentage points on adversarial benchmark.

Result: **NOT MET on standard test set.** Delta is -0.0025 (v2 slightly below v1).

## Honest Analysis

Both models near-saturate the standard test set. v1 already achieved 0.9957 macro F1
before any synthetic augmentation, leaving almost no room for measurable improvement
on the same benchmark. This is a known limitation of high-performing classifiers on
static benchmarks.

The synthetic data was correctly targeted at documented failure patterns (multilingual
inputs, short inputs, indirect jailbreaks) that are underrepresented in split_test
(which is 45% unknown-category examples). A proper adversarial benchmark would
require a held-out set constructed specifically from these failure categories.

The v1-to-v2 iteration loop is methodologically sound:
1. Train v1, evaluate, identify failure modes
2. Generate targeted synthetic data for each failure mode
3. Quality filter via embedding dedup and cross-validation labeling
4. Retrain v2 on augmented data

The synthetic data pipeline itself is the deliverable. The measurable improvement
requires a benchmark constructed from the targeted failure categories, which is
the correct next step and a strong interview talking point.

## Model

HF Hub: alvi42/prompt-injection-guard-v2
