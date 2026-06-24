# Prompt Injection Guard: 35-Day Build Plan

**Daily commitment:** 3 hours
**Build days:** 25 (5 per week)
**Buffer days:** 5 (1 per week, absorbs slippage)
**Rest days:** 5 (1 per week, no code)

---

## Risk Gates (Non-Negotiable)

| Day | Gate | If missed |
|-----|------|-----------|
| 10 | Classifier v1 macro F1 ≥ 0.85 | Use Day 13 buffer to tune. Drop to F1 ≥ 0.82 only if dataset noise documented. |
| 17 | Real-time API p99 latency < 100ms | Use Day 20 buffer. Switch to DistilBERT if DeBERTa fails. |
| 25 | v2 F1 ≥ v1 F1 + 5 pts on adversarial benchmark | Use Day 27 buffer. Document honest result if v2 fails to beat v1. |

---

## LinkedIn Milestones (AlviAnalytics)

| Day | Post topic | Hook |
|-----|------------|------|
| 12 | Classifier v1 metrics, per-category breakdown | "Fine-tuned DeBERTa on five public prompt injection datasets. Here is what it catches and what it misses." |
| 26 | v1 vs v2 with synthetic data | "Used Claude to generate 2,000 targeted adversarial prompts. v2 macro F1 up X points. Methodology in the repo." |
| 33 | Live demo launch | "Public prompt injection classifier with sub-100ms inference. Try it yourself." |

---

## Week 1: Foundation and Data (Days 1–7)

### Day 1: Project Scaffold + Environment
**Goal:** Repo live on GitHub, CI green, all accounts ready.
- 60 min: Create GitHub repo `prompt-injection-guard` with MIT license. Init Python 3.11 with uv. Folder structure: `src/`, `data/`, `notebooks/`, `tests/`, `docker/`.
- 45 min: Pre-commit hooks (ruff, black, mypy). GitHub Actions skeleton (lint + pytest on push).
- 45 min: Anthropic API key in `.env`. Hugging Face account + access token. Colab linked to Google Drive.
- 30 min: README skeleton with project pitch and dataset table placeholder.
- **Commit:** `chore: project scaffold and CI`
- **Done when:** First push triggers CI, CI passes green.

### Day 2: Dataset Ingestion
**Goal:** Five public datasets loaded with one function call each.
- 30 min: Learn HF `datasets` library API.
- 75 min: Write `src/data/loaders.py`. One function per source: Lakera Gandalf, deepset/prompt-injections, jasperai/prompt-injections, BIPIA, WildGuardMix prompt injection subset.
- 60 min: Notebook inspecting each: label distribution, text length histograms, ten sample examples per dataset.
- 15 min: Document access issues in `docs/data.md`.
- **Commit:** `feat: dataset loaders for five public sources`
- **Done when:** Notebook runs end to end, all five datasets visible.

### Day 3: Unified Schema in DuckDB
**Goal:** All data in one table with consistent schema.
- 30 min: Learn DuckDB Python API.
- 60 min: Define schema (`id`, `input_text`, `label`, `attack_category`, `source`, `dataset_version`). Write `src/data/unify.py`.
- 45 min: Map each source into the schema. Handle label disagreements explicitly.
- 45 min: Define attack category taxonomy (direct injection, indirect, jailbreak, role-play, system prompt leak). Tag examples.
- **Commit:** `feat: unified dataset schema in DuckDB`
- **Done when:** `SELECT COUNT(*) FROM unified GROUP BY label, attack_category` returns sensible distribution.

### Day 4: Baseline + Stratified Split
**Goal:** Know what good looks like before training anything.
- 45 min: Train/val/test split (70/15/15) stratified by attack_category.
- 75 min: Zero-shot baseline. Claude Haiku classifies the test set via Anthropic API. Async batch with rate limiting.
- 45 min: Compute macro F1, per-category recall, confusion matrix.
- 15 min: Update README with baseline numbers and dataset table.
- **Commit:** `feat: zero-shot Claude Haiku baseline`
- **Done when:** Baseline metrics in README, README pushed publicly.

### Day 5: Data Quality + Week Wrap
**Goal:** Clean foundation, no surprises in Week 2.
- 60 min: Deduplication via MinHash or embedding similarity > 0.95.
- 45 min: Label consistency cross-check. Spot any source bias.
- 45 min: Add `tests/test_data.py`: schema validation, label distribution sanity checks. Wire into CI.
- 30 min: Write `notebooks/week1_summary.ipynb` for future LinkedIn reference.
- **Commit:** `test: data pipeline tests + week 1 wrap`
- **Done when:** All tests pass in CI, README has Week 1 summary section.

### Day 6: Buffer
Use for: catching up on Days 1–5, extra dataset exploration, or code cleanup. If fully on track, read two recent prompt injection papers (Greshake et al. on indirect injection, the BIPIA paper).

### Day 7: Rest
No code. Step away.

---

## Week 2: Classifier v1 (Days 8–14)

### Day 8: HF Trainer Smoke Test
**Goal:** Training loop runs on a tiny subset without errors.
- 60 min: Learn HF Trainer API. Read the Transformers fine-tuning tutorial.
- 60 min: Write `src/models/train.py`. Load DeBERTa-v3-base, tokenize, set up Trainer.
- 60 min: Train on 500 examples, 1 epoch, on Colab T4. Verify checkpoint saves correctly.
- **Commit:** `feat: training scaffold, smoke test passes`
- **Done when:** A checkpoint exists on Drive, loss decreased over the run.

### Day 9: Full Training Run
**Goal:** Classifier v1 trained on full data.
- 30 min: Hyperparameter starting point: lr 2e-5, batch 16, 3 epochs, weight decay 0.01.
- 120 min: Full training on Colab T4. Save tokenizer, model, training metrics to Drive. Push weights to HF Hub private repo.
- 30 min: Document training time, GPU cost, peak VRAM in `docs/training.md`.
- **Commit:** `feat: classifier v1 trained`
- **Done when:** Model loads from HF Hub, predicts on a sample input.

### Day 10: Evaluation — RISK GATE
**Goal:** Macro F1 ≥ 0.85 on held-out test set.
- 60 min: Run `src/eval/evaluate.py` on test set. Macro F1, per-category recall, confusion matrix.
- 60 min: Failure analysis. Pull 50 false positives and 50 false negatives. Categorize errors.
- 60 min: If F1 < 0.85: investigate (class imbalance, label noise, undertraining). Plan Day 11 retrain.
- **Commit:** `eval: classifier v1 results + failure analysis`
- **Done when:** Gate met OR clear remediation plan documented for Day 11.

### Day 11: Tuning Pass (or Polish)
**Goal:** Hit the gate if you missed it. Otherwise lock v1.
- If gate missed: rerun with adjustments (longer training, weight balancing, learning rate schedule).
- If gate met: tighten the eval harness. Add bootstrap confidence intervals to F1. Document v1 model card on HF Hub.
- **Commit:** `feat: classifier v1 final + model card`
- **Done when:** v1 metrics frozen and published.

### Day 12: LinkedIn Post 1 + Week Wrap
**Goal:** Public v1 metrics post.
- 90 min: Write LinkedIn post. Lead with one number that surprised you (e.g., recall on indirect injection). Include per-category bar chart.
- 60 min: Generate the bar chart with matplotlib. Save as PNG to `assets/`.
- 30 min: Update README with v1 final metrics, link to model card.
- **Commit:** `docs: README with v1 metrics, week 2 wrap`
- **Done when:** Post published, LinkedIn impressions tracked.

### Day 13: Buffer
Use for: Day 10 gate recovery if needed, or read papers on synthetic data augmentation (e.g., AutoEval, Self-Instruct).

### Day 14: Rest

---

## Week 3: Serving Layer (Days 15–21)

### Day 15: FastAPI Scaffold + /classify Endpoint
**Goal:** Real-time classification endpoint live on localhost.
- 45 min: Write `src/api/main.py`. FastAPI app with `/classify` POST endpoint. Pydantic request/response schemas.
- 60 min: Load classifier v1 at startup. Implement classification logic with confidence scores per category.
- 45 min: Request logging to DuckDB. Log `input_text`, `prediction`, `confidence`, `latency_ms`, `timestamp`.
- 30 min: Write `tests/test_api.py`. Basic request/response assertions.
- **Commit:** `feat: FastAPI /classify endpoint`
- **Done when:** `curl localhost:8000/classify -d '{"text":"..."}'` returns valid JSON.

### Day 16: ONNX Export + Runtime
**Goal:** Serve via ONNX, see latency drop.
- 45 min: Learn `optimum` library for HF-to-ONNX export.
- 60 min: Export classifier v1 to ONNX with dynamic quantization (int8).
- 60 min: Swap PyTorch inference for ONNX Runtime in `src/api/main.py`. Verify predictions match within tolerance.
- 15 min: Document the export pipeline in `docs/deployment.md`.
- **Commit:** `feat: ONNX runtime for inference`
- **Done when:** ONNX model loaded, predictions match PyTorch baseline.

### Day 17: Latency Profiling — RISK GATE
**Goal:** p99 latency < 100ms on single CPU core.
- 45 min: Write `scripts/load_test.py` using locust or wrk. Single-core throttling.
- 60 min: Run load test at 50, 100, 200 RPS. Capture p50, p95, p99.
- 60 min: If p99 > 100ms: profile with py-spy. Likely fixes: ONNX session reuse, tokenizer batching, removing logging from hot path.
- 15 min: Latency report in `docs/performance.md`.
- **Commit:** `perf: latency profiling + p99 under 100ms`
- **Done when:** Gate met OR remediation plan for Day 20 buffer documented.

### Day 18: /classify-batch Async Endpoint
**Goal:** Batch scoring at 500+ examples per second.
- 60 min: Implement `/classify-batch` POST endpoint. Accepts list of texts, returns list of predictions.
- 75 min: Async processing with `asyncio.gather` and dynamic batching (batch size 32, max wait 50ms).
- 30 min: Add throughput test to `scripts/load_test.py`.
- 15 min: Update README with API examples.
- **Commit:** `feat: batch endpoint with async processing`
- **Done when:** Batch endpoint hits 500+ examples per second.

### Day 19: Docker Compose + Week Wrap
**Goal:** One command brings up the full stack locally.
- 60 min: Write `Dockerfile` for the API. Multi-stage build, slim base.
- 60 min: `docker-compose.yml` with API + DuckDB volume. Healthcheck endpoint.
- 30 min: Test from a clean clone. Document in README quickstart section.
- 30 min: Write `notebooks/week3_summary.ipynb` covering latency story.
- **Commit:** `infra: docker compose for local reproduction`
- **Done when:** Fresh clone → `docker compose up` → working API in under 5 minutes.

### Day 20: Buffer
Use for: latency gate recovery (most common slip), Docker debugging, or stretch task of adding Prometheus metrics endpoint.

### Day 21: Rest

---

## Week 4: Synthetic Data Pipeline (Days 22–28)

### Day 22: Failure Mode Clustering
**Goal:** Group classifier v1 errors into actionable patterns.
- 30 min: Pull all v1 false positives and false negatives from DuckDB (around 200–500 examples).
- 60 min: Embed each error with `sentence-transformers/all-mpnet-base-v2`.
- 60 min: HDBSCAN clustering on embeddings. Inspect clusters manually.
- 30 min: Name 5–8 distinct failure patterns. Document in `docs/failures.md`.
- **Commit:** `analysis: failure mode clustering`
- **Done when:** Each cluster has a name and 3–5 representative examples.

### Day 23: Synthetic Data Generator
**Goal:** Claude generates 2,000+ targeted adversarial prompts.
- 60 min: Write `src/synth/generate.py`. Prompt template per failure cluster. Async generation via Anthropic API.
- 60 min: For each cluster, generate 250–400 variants. Use Claude Sonnet for quality.
- 45 min: Save raw outputs with metadata (`source_cluster`, `generation_prompt`, `model`, `timestamp`).
- 15 min: Spot-check 30 random samples. Are they actually adversarial?
- **Commit:** `feat: synthetic data generator`
- **Done when:** 2,000+ synthetic examples saved with full provenance.

### Day 24: Quality Filter
**Goal:** Keep only the high-signal synthetic examples.
- 60 min: Embedding-based deduplication. Drop synthetic examples too close to existing training data (cosine > 0.92).
- 45 min: Cross-validation labeling. Use a different model (GPT-4o-mini or Claude Haiku) to label each synthetic example. Keep only where labels agree.
- 45 min: Manual review of 100 random survivors. Reject the filter if quality is poor.
- 30 min: Final synthetic dataset stats. Document drop rate, agreement rate.
- **Commit:** `feat: synthetic data quality filter`
- **Done when:** Filtered synthetic dataset committed with rejection reasoning.

### Day 25: Train v2 + A/B — RISK GATE
**Goal:** v2 F1 beats v1 by at least 5 percentage points on adversarial benchmark.
- 30 min: Build v2 training set: original train + filtered synthetic.
- 90 min: Train v2 on Colab T4. Same hyperparameters as v1 for fair comparison.
- 45 min: Evaluate v2 on the same test set as v1. Plus held-out adversarial benchmark (WildGuardMix prompt injection subset, untouched until now).
- 15 min: A/B comparison table: v1 vs v2 per-category recall and macro F1.
- **Commit:** `feat: classifier v2 with synthetic augmentation`
- **Done when:** Gate met OR honest result documented for LinkedIn.

### Day 26: LinkedIn Post 2 + Week Wrap
**Goal:** v1 vs v2 comparison post.
- 60 min: Write LinkedIn post. Lead with the delta. Show the bar chart.
- 60 min: Generate side-by-side comparison charts. Save to `assets/`.
- 45 min: Update README with v2 metrics, methodology section, link to synthetic data card on HF Hub.
- 15 min: Update model card on HF Hub.
- **Commit:** `docs: v2 results + week 4 wrap`
- **Done when:** Post published, README reflects v2 as the deployed model.

### Day 27: Buffer
Use for: gate recovery if v2 did not beat v1, or stretch task of generating one more synthetic round if you have ideas.

### Day 28: Rest

---

## Week 5: Demo and Launch (Days 29–35)

### Day 29: Gradio UI
**Goal:** Interactive UI for the live demo.
- 30 min: Learn Gradio basics.
- 90 min: Build `src/demo/app.py`. Text input, classification output with confidence bars per category, example prompts (benign, direct injection, indirect injection, jailbreak).
- 45 min: Style the UI. Title, description, link to GitHub repo, link to model card.
- 15 min: Test locally with `python app.py`.
- **Commit:** `feat: gradio demo UI`
- **Done when:** Local Gradio UI looks recruiter-ready.

### Day 30: HF Spaces Deployment
**Goal:** Live public demo URL.
- 30 min: Learn HF Spaces deployment workflow.
- 60 min: Create Space. Push Gradio app + ONNX model.
- 60 min: Debug deployment issues (dependencies, memory limits on free tier).
- 30 min: Verify public URL works from a fresh browser. Test five example prompts.
- **Commit:** `deploy: HF Spaces public demo`
- **Done when:** Public URL accessible, classification working in browser.

### Day 31: Architecture Diagram + Metrics Dashboard
**Goal:** Two visuals a recruiter understands in 5 seconds each.
- 90 min: Architecture diagram in excalidraw or draw.io. Three layers (data, training, serving). Synthetic data feedback loop highlighted. Export PNG.
- 60 min: Metrics dashboard image. v1 vs v2 bars, latency percentiles, per-category recall. Single image.
- 30 min: Embed both in README near the top.
- **Commit:** `docs: architecture diagram + metrics dashboard`
- **Done when:** Repo top reads like a product page.

### Day 32: README Polish + Repo Presentation
**Goal:** A non-technical recruiter understands the project in 30 seconds, a hiring manager in 10 minutes.
- 60 min: Rewrite README top. Hero line, demo GIF, one-paragraph elevator pitch, link to live demo.
- 60 min: Quickstart section (Docker compose) verified one more time.
- 30 min: Add `Why this exists` section linking to the Anthropic Safeguards posting responsibility lines.
- 30 min: Add `Talking points` section for interview prep.
- **Commit:** `docs: README polish for portfolio`
- **Done when:** README passes the 30-second test on a non-technical friend.

### Day 33: LinkedIn Post 3 + Launch
**Goal:** Public launch.
- 90 min: Write LinkedIn launch post. Lead with the demo URL. Include the metrics dashboard image. Tag relevant people (Anthropic, Mistral followers).
- 30 min: Share to other channels (Twitter/X, personal site, Discord communities).
- 30 min: Tweet thread version. Three tweets covering the journey.
- 30 min: Add the live demo link to GitHub repo description and your LinkedIn featured section.
- **Commit:** `docs: launch announcement`
- **Done when:** Post live, demo URL pinned to your AlviAnalytics profile.

### Day 34: Buffer
Use for: post-launch fixes (people will find bugs), responding to comments, or final write-ups.

### Day 35: Rest
No code. Take stock.

---

## Daily Discipline

Every build day ends with:
- A green CI run
- A commit pushed to main
- A one-line note in `LOG.md` recording what you finished and what blocked you (if anything)

This makes the buffer day actually useful. You arrive on Day 6 knowing exactly what slipped.

---

## What Success Looks Like on Day 35

- GitHub repo public with 800+ stars-worthy README
- Live HF Spaces demo at a memorable URL
- Three LinkedIn posts with the AlviAnalytics audience
- A model card on HF Hub
- A measurable v1-to-v2 improvement documented end to end
- One sentence you have rehearsed for interviews:

"I built a prompt injection classifier with a synthetic data iteration loop that improved macro F1 by X points on adversarial benchmarks while serving at sub-100ms p99 latency. The full methodology is in the repo and the live demo is at [URL]."

---

## What to Do When Stuck

- Stuck for 30 minutes → ask Claude or GPT to debug with you
- Stuck for 60 minutes → write a question in the project's `BLOCKERS.md` and move to the next subtask
- Stuck on a buffer day → reassess scope. Cut features, not quality.

Cut order if scope must shrink:
1. Drop ONNX, serve PyTorch directly (Day 16, lose latency edge but ship)
2. Drop async batch endpoint (Day 18)
3. Drop Docker Compose (Day 19, run uvicorn directly)
4. Drop the third LinkedIn post (Day 33)
5. Never cut: classifier v1, synthetic loop, live demo
