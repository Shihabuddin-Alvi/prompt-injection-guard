# Prompt Injection Guard

Prompt injection detection trained with an iterative synthetic data pipeline. Sub-100ms inference. Open methodology, reproducible benchmarks, public demo.

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
