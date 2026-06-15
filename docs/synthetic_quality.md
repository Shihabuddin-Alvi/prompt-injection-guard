# Synthetic Data Quality Filter (Day 24)

## Pipeline

1. **Embedding-based deduplication**: Embedded 1,700 synthetic examples and
   8,183 training examples with sentence-transformers/all-mpnet-base-v2.
   Dropped examples with cosine similarity > 0.92 against any training example.
2. **Cross-validation labeling**: Sent each example to Claude Haiku with an
   independent classification prompt. Kept only examples where Haiku's label
   matches the label assigned during generation.
3. **Manual review**: Reviewed 100 random survivors for quality and diversity.

## Results

| Stage | Input | Output | Dropped | Drop rate |
|---|---|---|---|---|
| Deduplication | 1,700 | 1,700 | 0 | 0.000 |
| Cross-validation | 1,700 | 1,687 | 13 | 0.008 |
| **Final** | **1,700** | **1,687** | **13** | **0.008** |

## Deduplication: 0% Drop Rate

Max cosine similarity to existing training data was 0.865 (threshold: 0.92).
This is an honest result: the targeted generation produced genuinely novel
content relative to the existing training set, rather than near-paraphrases.
The 0.92 threshold from the build plan assumed less targeted generation; here
it correctly identifies that no synthetic example is a near-duplicate.

## Cross-Validation: 99.2% Agreement

13 of 1,700 examples were dropped due to label disagreement between the
generation intent and Claude Haiku's independent judgment.

**10 FN-3 disagreements** (expected=injection, Haiku=benign): all were
"educational/research/roleplay purposes" framings — sophisticated indirect
jailbreaks. The disagreement itself validates that these are genuinely
ambiguous cases, consistent with the FN-3 failure pattern (indirect/subtle
injections that avoid explicit phrasing). Dropping them avoids training on
contested ground truth.

**3 FP-1 disagreements** (expected=benign, Haiku=injection): all were
coding-instruction phrasing ("follow the steps in the docstring"). Haiku
reproduced the exact FP-1 failure pattern (instruction-adjacent coding
phrasing triggers false positives), confirming this is a hard boundary case
for both classifier v1 and Haiku.

## Final Cluster Distribution

| Cluster | Pattern | Label | Generated | Survived |
|---|---|---|---|---|
| FN-1 | Non-English injections | 1 | 400 | 400 |
| FN-2 | Very short injections | 1 | 300 | 300 |
| FN-3 | Indirect/subtle injections | 1 | 400 | 390 |
| FP-2 | Legitimate role-play | 0 | 300 | 300 |
| FP-1 | Instruction-adjacent coding | 0 | 300 | 297 |

Manual review of 100 random survivors confirmed quality and on-target
cluster distribution. Minor template repetition observed in FN-1 (Spanish
phrasing variants) — acceptable at this volume.

## Output

`data/synthetic_filtered.jsonl` — 1,687 examples ready for v2 training set.