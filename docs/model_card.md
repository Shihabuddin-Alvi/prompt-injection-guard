# Prompt Injection Guard — Classifier v1

## Model Description

DeBERTa-v3-base fine-tuned for binary prompt injection detection.
Classifies user inputs to LLM applications as benign or injection attempt.

Built as part of a 35-day portfolio project targeting the Anthropic Safeguards ML/Research Engineer role.

## Intended Use

- Real-time classification of user inputs before passing to an LLM
- Batch scoring of logged inputs for offline review
- Baseline for iterative synthetic data augmentation (v2 in progress)

## Training Data

Three public datasets unified into a single DuckDB schema:

| Source | Examples |
|--------|----------|
| jasperai/prompt-injections | 653 |
| lakera/gandalf | 999 |
| xTRam1/safe-guard-prompt-injection | 10,038 |

Total after deduplication (MinHash LSH): 11,690 examples.
Train/val/test split: 70/15/15, stratified by attack category.

## Training Details

- Base model: microsoft/deberta-v3-base
- Framework: HuggingFace Trainer
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Max sequence length: 256
- fp16: False (required for DeBERTa v3 on T4)
- transformers version: 4.47.0 (pinned — 5.0.0 breaks DeBERTa v3 LayerNorm loading)
- Training time: ~28 minutes on Colab T4

## Evaluation Results (Held-Out Test Set, n=1,754)

| Metric | Value |
|--------|-------|
| Macro F1 | 0.9957 |
| 95% Bootstrap CI | (0.9924, 0.9982) |
| Accuracy | 1.00 |
| Injection Recall | 1.00 |
| Injection Precision | 0.99 |

### Per-Category F1

| Category | F1 | n |
|----------|-----|---|
| direct | 1.0000 | 92 |
| jailbreak | 1.0000 | 26 |
| system_prompt_leak | 1.0000 | 12 |
| indirect | 1.0000 | 1 |
| unknown | 0.9962 | 1566 |
| role_play | 0.9644 | 57 |

### Known Failure Patterns

- Ambiguous role-play requests (legitimate persona vs. injection)
- Instruction-adjacent phrasing in benign coding contexts
- Very short inputs with no injection signal
- Non-English inputs (German false negative observed)

These patterns are the synthetic data targets for v2.

## Zero-Shot Baseline

Claude Haiku zero-shot on 200 test samples: macro F1 0.93, injection recall 0.85.
Classifier v1 outperforms the zero-shot baseline on both metrics.

## Limitations

- English-only training data
- Role-play category has the lowest F1 (0.9644) — targeted for v2 improvement
- Static model — does not adapt to new injection techniques without retraining

## Repository

https://github.com/Shihabuddin-Alvi/prompt-injection-guard
