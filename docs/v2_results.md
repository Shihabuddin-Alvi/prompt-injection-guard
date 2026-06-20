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

## v1 vs v2 on adversarial slice (n=651, injection-heavy, role_play + indirect + all injection-labeled)

| Metric | v1 | v2 |
|---|---|---|
| Benign precision | 0.83 | 0.74 |
| Benign recall | 0.95 | 1.00 |
| Benign F1 | 0.88 | 0.85 |
| Injection precision | 1.00 | 1.00 |
| Injection recall | 0.99 | 0.99 |
| Injection F1 | 1.00 | 0.99 |
| Macro F1 | 0.94 | 0.92 |

v1 scores higher than v2 on this slice. The benign sample is small (n=20), so
this single-point comparison should not be over-read, but the result is
consistent with the standard test set finding: v2 does not measurably beat v1
on any benchmark constructed so far.

## Final Risk Gate 3 Verdict

**NOT MET**, confirmed on two independent benchmarks (standard test set and
adversarial slice). The synthetic data pipeline and failure analysis
methodology are sound and fully documented, but the resulting v2 model is not
a measurable improvement over v1 by macro F1 on either evaluation set
available. A properly constructed held-out benchmark, built directly from the
8 failure clusters rather than filtered from existing splits, would be
required to validate whether the synthetic data's targeted patterns transfer.
That benchmark does not yet exist and is out of scope for the remaining build
days.

v1 remains the deployed model.

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
