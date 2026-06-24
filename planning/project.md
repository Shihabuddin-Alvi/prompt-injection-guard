# Project 3: Prompt Injection Classifier with Synthetic Data Pipeline

A production-grade prompt injection detection service, trained with an iterative synthetic data pipeline, demonstrating measurable robustness gains across classifier versions.

---

## Why This Exists

This project maps directly to three responsibility lines in the Anthropic Safeguards ML/Research Engineer job posting:

1. **"Develop classifiers to detect misuse and anomalous behavior at scale"**
2. **"Developing synthetic data pipelines for training classifiers"**
3. **"Developing and deploying mitigations for prompt injection attacks"**

The Safeguards team builds systems that identify harmful use of Claude. Prompt injection is one of the highest-risk attack surfaces in agentic deployments. This project demonstrates the full workflow: detect, evaluate, iterate, deploy.

---

## What It Does

Detects prompt injection attempts in user inputs to LLM applications.

- **Real-time API**: sub-100ms latency, single classification per request.
- **Batch API**: async processing of large input sets for offline review.
- **Synthetic data loop**: failure modes from classifier v1 feed an LLM-driven synthetic data generator. Classifier v2 trains on the augmented dataset and shows measurable improvement on held-out adversarial benchmarks.

---

## Differentiation

| Existing tool | Limitation | This project |
|---------------|------------|--------------|
| Lakera Guard | Closed source, opaque training | Open methodology, reproducible benchmarks |
| Rebuff | Commercial, no public methodology | Public datasets, public model |
| Meta PromptGuard | Static model, no iteration loop | Iterative synthetic data pipeline |
| ProtectAI llm-guard | Rule-heavy, no adversarial training | Adversarial training built into the workflow |

The defensible edge: a transparent, reproducible synthetic data iteration loop. Most public prompt injection detectors are static. This one shows v1, finds failures, generates targeted synthetic adversarial examples, retrains, and ships v2 with documented improvement.

---

## Tech Stack

- **FastAPI** for serving
- **Hugging Face Transformers**, **DeBERTa-v3-base** as the classifier backbone
- **ONNX Runtime** for inference acceleration on CPU
- **DuckDB** for metrics, failure logs, and dataset versioning
- **Anthropic API** for synthetic data generation (free tier credits or low cost)
- **pytest** for the evaluation harness
- **GitHub Actions** for CI and automated benchmarking on every push
- **Docker Compose** for local reproducibility
- **Hugging Face Spaces** for free public demo hosting
- **Gradio** for the demo UI

---

## Architecture

Three layers:

1. **Data Layer**: ingests public prompt injection datasets, unifies schema in DuckDB. Tracks dataset versions, attack categories, sources.
2. **Training Layer**: fine-tunes DeBERTa, evaluates on held-out benchmarks, logs failure modes by attack category.
3. **Serving Layer**: FastAPI endpoints (real-time and batch), metrics dashboard, request logging.

**Iteration Loop**: serving layer logs uncertain predictions and confirmed failures. Training layer ingests these, generates synthetic adversarial examples covering failure patterns, retrains classifier, redeploys.

---

## Data Sources (All Public)

- **Lakera Gandalf** prompt collection
- **Microsoft PINT** (Prompt Injection Test) benchmark
- **BIPIA** (Benchmarking Indirect Prompt Injection Attacks)
- **deepset/prompt-injections** on Hugging Face
- **jasperai/prompt-injections** on Hugging Face
- **WildGuardMix** for adjacent misuse categories (held-out for generalization testing)

---

## Build Plan: 5 Weeks at 21 Hours per Week

### Week 1: Foundation
- Repo structure, pre-commit hooks (ruff, black, mypy), GitHub Actions skeleton
- Ingest all public datasets into unified DuckDB schema
- Schema: `input_text`, `label`, `attack_category`, `source`, `dataset_version`
- Baseline evaluation: zero-shot Claude Haiku as reference benchmark
- Initial train/val/test split with stratification by attack category
- **Deliverable**: data pipeline notebook, baseline metrics in README

### Week 2: Classifier v1
- Fine-tune DeBERTa-v3-base on unified dataset
- Training environment: Colab free tier T4 GPU, fallback to MacBook M5 with MPS
- Evaluation: macro F1, per-attack-category recall, confusion matrix
- ONNX export, latency profiling on CPU
- **Deliverable**: classifier v1 with metrics table in README, model card

### Week 3: Serving Layer
- FastAPI app: `/classify` (real-time), `/classify-batch` (async)
- Pydantic schemas, OpenAPI spec auto-generated
- Sub-100ms p99 latency target validated via load testing
- Docker Compose setup for one-command local reproduction
- Request logging to DuckDB for the iteration loop
- **Deliverable**: working API, Swagger docs, Docker setup, load test report

### Week 4: Synthetic Data Pipeline
- Failure mode analysis: cluster classifier v1 errors by attack pattern using sentence embeddings
- Synthetic data generator: Claude API prompts produce variants of each failure cluster
- Quality filter: deduplication via embedding similarity, cross-validation labeling
- Train classifier v2 on original + synthetic data
- A/B comparison: v1 vs v2 on held-out adversarial benchmark (WildGuardMix prompt injection subset)
- **Deliverable**: failure analysis report, synthetic data generation script, v2 model with improvement metrics

### Week 5: Demo and Public Launch
- Deploy to Hugging Face Spaces with Gradio UI
- Public demo features: paste prompt, see classification, confidence score, attack category
- README polish: architecture diagram, metrics dashboard image, before/after comparison
- LinkedIn write-up for AlviAnalytics
- **Deliverable**: live public demo URL, polished README, social post

---

## Success Criteria (Measurable)

Non-negotiable:

- Real-time p99 latency under 100ms on a single CPU
- Macro F1 above 0.88 on held-out test set
- Classifier v2 shows at least 5 percentage points F1 improvement over v1 on the held-out adversarial benchmark
- Batch endpoint processes 500 examples per second
- Live public demo accessible with zero authentication friction
- README readable in 10 minutes by a hiring manager

---

## Demo Strategy

- **Live demo**: Hugging Face Spaces with Gradio UI. Free tier sufficient. Public URL.
- **Repo**: GitHub with comprehensive README, architecture diagram, metrics dashboard image.
- **30-second recruiter story**: "Public demo at huggingface.co/spaces/alvi/prompt-injection-guard. Paste any prompt and see real-time classification with confidence scores."
- **10-minute hiring manager story**: README walks through dataset choices, model selection rationale, synthetic data pipeline architecture, before/after metrics, latency profiling, deployment notes.

---

## Portfolio Narrative

Three projects, three layers of LLM production reliability:

- **Project 1 (LLM Eval Platform)**: Is the model's output good?
- **Project 2 (Tool-Call Reliability)**: Did the agent behave correctly?
- **Project 3 (Prompt Injection Classifier)**: Was the input safe to act on?

This is a coherent input → behavior → output safety stack, not three disconnected tools.

---

## README Headline

"Prompt injection detection trained with an iterative synthetic data pipeline. Sub-100ms inference. Open methodology, reproducible benchmarks, public demo."

---

## Public Artifact Checklist

- [ ] GitHub repo (public, MIT license)
- [ ] Live Hugging Face Spaces demo
- [ ] LinkedIn post on AlviAnalytics with metrics screenshot
- [ ] Architecture diagram (PNG in repo)
- [ ] Metrics dashboard image (PNG in repo)
- [ ] Reproducible benchmark notebook
- [ ] Model card on Hugging Face Hub
- [ ] Docker image on Docker Hub (optional)

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Synthetic data quality is low and v2 does not beat v1 | Use Claude Sonnet (not Haiku) for generation, add human-eye spot check, fall back to documented honest result if v2 fails |
| Latency target missed on CPU | Switch to DistilBERT instead of DeBERTa-v3-base, accept small F1 trade-off |
| Public datasets are too easy and F1 saturates at 0.95+ | Add adversarial perturbations (typos, encoding tricks) to test set, source examples from recent prompt injection research papers |
| Hugging Face Spaces free tier insufficient for live demo | Use Modal or Render free tier as fallback |

---

## Interview Talking Points

When a hiring manager asks about this project, lead with:

1. **The iteration loop**: not the model, the methodology. The model is a means to the end.
2. **The failure mode analysis**: how you discovered v1's blind spots and chose which to patch.
3. **The synthetic data quality controls**: deduplication, cross-validation, why you trust the augmented training set.
4. **The deployment tradeoffs**: ONNX vs PyTorch serving, latency budgets, batching strategies.
5. **Connection to Anthropic Safeguards work**: the posting calls for synthetic data pipelines and prompt injection mitigation. This project is a working example of both.

---

## Out of Scope (Intentionally)

- Multi-turn conversation tracking (could be Project 4)
- Multilingual support (English only for v1)
- Output-side safety (covered by other safety classifiers)
- Adversarial robustness against gradient-based attacks (research territory, not production)
- Fine-tuning a custom LLM as the classifier (DeBERTa is faster, cheaper, sufficient)
