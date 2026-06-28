# Phase 0 Baseline

Date: 2026-06-28

Test set: split_test from unified.duckdb
Examples: 1754

Note: attack_category is 'unknown' for the majority of examples.
Per-category breakdown by taxonomy family is not available at this phase.
That breakdown is the Phase 4 deliverable.

## v1

Macro F1: 0.9957

```
              precision    recall  f1-score   support

      benign     0.9973    0.9965    0.9969      1127
   injection     0.9936    0.9952    0.9944       627

    accuracy                         0.9960      1754
   macro avg     0.9955    0.9958    0.9957      1754
weighted avg     0.9960    0.9960    0.9960      1754

```

Confusion matrix (rows=actual, cols=predicted):

```
              pred_benign  pred_injection
actual_benign       1123               4
actual_inject          3             624
```

## v2

Macro F1: 0.9932

```
              precision    recall  f1-score   support

      benign     0.9947    0.9956    0.9951      1127
   injection     0.9920    0.9904    0.9912       627

    accuracy                         0.9937      1754
   macro avg     0.9933    0.9930    0.9932      1754
weighted avg     0.9937    0.9937    0.9937      1754

```

Confusion matrix (rows=actual, cols=predicted):

```
              pred_benign  pred_injection
actual_benign       1122               5
actual_inject          6             621
```
