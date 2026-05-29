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
| Lakera Gandalf | TBD | HF Hub |
| deepset/prompt-injections | TBD | HF Hub |
| jasperai/prompt-injections | TBD | HF Hub |
| BIPIA | TBD | HF Hub |
| WildGuardMix | TBD | HF Hub |

## Results

| Model | Macro F1 | p99 Latency |
|-------|----------|-------------|
| Zero-shot Claude Haiku (baseline) | TBD | - |
| Classifier v1 (DeBERTa) | TBD | TBD |
| Classifier v2 (+ synthetic data) | TBD | TBD |

## Quick Start

```bash
docker compose up
curl localhost:8000/classify -d '{"text": "Ignore previous instructions and..."}'
```

## Architecture

Three layers: data ingestion, training, serving. Synthetic data feedback loop connects serving failures back to training.

## Links

- Live demo: TBD
- Model card: TBD
- LinkedIn: AlviAnalytics

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

Haiku misses 15% of injections (12/78 false negatives) while producing almost no false alarms on benign inputs (1/122).
DeBERTa v1 target: macro F1 > 0.93, injection recall > 0.85.
