# Prompt Injection Guard

Prompt injection detection trained with an iterative synthetic data pipeline. Classifier v1: macro F1 0.9957 on 1,754 held-out examples. Sub-100ms inference target. Open methodology, reproducible benchmarks.

## What It Does

Detects prompt injection attempts in user inputs to LLM applications.

- Real-time API: sub-100ms p99 latency per request
- Batch API: async processing at 500+ examples per second
- Synthetic data loop: classifier v1 failure modes feed a targeted data generator, v2 trains on the augmented dataset and shows measurable improvement

## Datasets

| Dataset | Examples | Source |
|---------|----------|--------|
| jasperai/prompt-injections | 662 | HF Hub |
| Lakera Gandalf | 1,000 | HF Hub |
| xTRam1/safe-guard-prompt-injection | 10,296 | HF Hub |
| WildGuardMix (held-out) | — | HF Hub |

Total after deduplication: 11,690 examples.

## Baseline Metrics (Claude Haiku Zero-Shot)

Evaluated on 200 sampled examples from the held-out test set.

| Metric | Score |
|--------|-------|
| Macro F1 | 0.93 |
| Benign precision | 0.91 |
| Benign recall | 0.99 |
| Injection precision | 0.99 |
| Injection recall | 0.85 |
| Accuracy | 0.94 |

Haiku misses 15% of injections (12/78 false negatives). DeBERTa v1 target: macro F1 > 0.93, injection recall > 0.85.

## Classifier v1 Results

Fine-tuned DeBERTa-v3-base on 8,183 training examples. Evaluated on 1,754 held-out test examples.

| Metric | Score |
|--------|-------|
| Macro F1 | 0.9957 |
| 95% Bootstrap CI | (0.9924, 0.9982) |
| Accuracy | 1.00 |
| Injection recall | 1.00 |
| Injection precision | 0.99 |

### Per-Category F1

![Per-category F1](assets/v1_per_category_f1.png)

| Category | F1 | n |
|----------|-----|---|
| direct | 1.0000 | 92 |
| jailbreak | 1.0000 | 26 |
| system_prompt_leak | 1.0000 | 12 |
| indirect | 1.0000 | 1 |
| unknown | 0.9962 | 1,566 |
| role_play | 0.9644 | 57 |

Role-play is the weakest category. Legitimate persona requests sit on the boundary with injection attempts. This is the synthetic data target for v2.

Model card: [alvi42/prompt-injection-guard-v1](https://huggingface.co/alvi42/prompt-injection-guard-v1)

## Stack

- DeBERTa-v3-base: classifier backbone
- ONNX Runtime: CPU inference acceleration
- FastAPI: serving layer
- DuckDB: dataset versioning and request logging
- Anthropic API: synthetic data generation
- Hugging Face Spaces: public demo
- Docker Compose: local reproducibility

## Quickstart

```bash
git clone https://github.com/Shihabuddin-Alvi/prompt-injection-guard.git
cd prompt-injection-guard
uv sync
uv run python3 -m src.data.unify
uv run python3 -m src.data.split
```

## Why This Exists

Maps directly to three responsibility lines in the Anthropic Safeguards ML/Research Engineer posting: detecting misuse at scale, developing synthetic data pipelines, and deploying mitigations for prompt injection attacks.