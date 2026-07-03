# Audit and Roadmap

Date: 2026-07-02
Scope: full-repo audit (code, DuckDB, docs, git history, HF Hub) plus a phased roadmap continuing the constitutional-classifier extension from Phase 2.

This document is written for execution one phase at a time, in separate sessions, by a model with no memory of this audit. Every claim below was verified against the repository on the date above. Where a doc and the repo disagreed, the repo won.

---

## How to execute this document (session protocol)

1. One phase per session. Do not start a phase until the previous phase's gate is recorded as PASSED or FAILED in this file's "Gate log" section (bottom).
2. Within a phase, do the steps in order. Each step names its files and tables. Do not improvise new scope.
3. Every phase ends with a measurable gate. Run the gate check, record the number, mark PASSED or FAILED. A FAILED gate is an acceptable, documented outcome — the v2 Risk Gate 3 failure in `docs/v2_results.md` is the model for how to write one up.
4. End every session with: green `make test`, a commit, and a gate-log entry here.
5. Costs run against the Anthropic API. Estimate before generating: this project's entire v2 synthetic round (1,700 examples, Sonnet) cost ~$3.50. Stay in that order of magnitude per phase unless the phase says otherwise.

---

# PART 1: AUDIT

## 1.1 What is built and verified

All 16 tests pass locally (`make test`): 9 data tests in [tests/test_data.py](../tests/test_data.py), 7 API tests in [tests/test_api.py](../tests/test_api.py). CI (`.github/workflows`) runs ruff, black, pytest on push. Local `main` is in sync with `origin/main`.

**Data layer** — built, tested.
- `src/data/loaders.py`: 5 loaders (deepset, JasperLS, Lakera Gandalf, xTRam1 safe-guard, WildGuardMix).
- `src/data/unify.py`, `dedup.py`, `split.py`: unified schema, MinHash dedup (930 dropped), 70/15/15 stratified split.
- `data/unified.duckdb` tables (verified 2026-07-02):

| Table | Rows | Notes |
|---|---|---|
| `unified` | 11,690 | schema: `id, input_text, label, attack_category, source, split` |
| `split_train` | 8,183 | |
| `split_val` | 1,753 | |
| `split_test` | 1,754 | |
| `split_train_v2` | 9,870 | original train + 1,687 synthetic (`source='synthetic_v1'`) |
| `v1_errors` | 10 | val-set errors only (6 FP, 4 FN) |
| `baseline_results` | 200 | Haiku zero-shot |
| `request_log` | 1,700 | API request logging (mostly load-test traffic) |

**Training/eval layer** — built, results committed, no unit tests.
- v1 trained (macro F1 0.9957 on `split_test`, bootstrap CI in `src/eval/evaluate.py`), v2 trained (0.9932). Risk Gate 3 (v2 ≥ v1 + 5pts) honestly FAILED and documented in `docs/v2_results.md`. v1 is the deployed model.
- Zero-shot Haiku baseline: macro F1 0.93, injection recall 0.85 (`src/eval/baseline.py`, `baseline_results` table).

**Synthetic pipeline** — built, ran once, no unit tests.
- `src/synth/generate.py` (5 failure-cluster templates, Sonnet), `scripts/dedup_synthetic.py` (0/1,700 dropped, max cosine 0.865), `scripts/crossval_label.py` + `finalize_crossval.py` (Haiku agreement 99.2%, 13 dropped). Artifacts: `data/synthetic_raw.jsonl`, `synthetic_filtered.jsonl`.

**Serving layer** — built, tested, profiled.
- `src/api/main.py`: `/classify`, `/classify-batch`, thread-pool inference offload, DuckDB request logging, ONNX int8 (`checkpoints/v1-onnx-int8/`). p99 91ms single-user documented in `docs/performance.md`. Batch throughput on the M5 dev machine is ~9 examples/s (kernel doesn't vectorize on Apple Silicon) — the 500/s target was never demonstrated on target hardware.

**Demo** — deployed. HF Space `alvi42/prompt-injection-guard` (Gradio, cpu-basic, currently SLEEPING — normal free-tier behavior). Pins v1.

**Constitutional-classifier extension** — Phases 0 and 1 confirmed done:
- Phase 0: `docs/baseline.md` + `scripts/eval_baseline.py` (commit 6407a68) — v1/v2 side-by-side on `split_test` with confusion matrices.
- Phase 1: [constitution.md](../constitution.md) v1.1 (PR1–PR10, NR1–NR6, F2/H4 boundary rule) and [attack_taxonomy.md](../attack_taxonomy.md) v1.0 (train families F1–F6, held-out H1–H4, benign contrast set).

## 1.2 Real state of H1–H4

**Defined in taxonomy only. Zero data anywhere.**
- No DuckDB table contains a `family` column. No table named anything like `heldout`, `h1`–`h4`, or `family_*` exists.
- `attack_category` (the *old* taxonomy: direct/jailbreak/role_play/system_prompt_leak/indirect/unknown) is `'unknown'` for 10,441 of 11,690 rows (89%). The 5 rows tagged `indirect` are not H1 data — they're heuristic tags from `classify_attack_category()` in `src/data/unify.py`.
- The constitution's required label schema (`text, label, family, split`) exists in no table and no JSONL file.

## 1.3 Where the 35-day plan and the 9-phase extension disagree or overlap

- **The 9-phase plan is not in the repo.** Only two phase references exist: `docs/baseline.md` ("Phase 0", "Phase 4 deliverable" = per-family breakdown) and this file now. The plan lives outside git. Part 2 of this document fixes that by becoming the committed plan of record.
- **Two taxonomies coexist with no mapping.** The Day-3 `attack_category` taxonomy (in `unified`) and the F/H family taxonomy (in `attack_taxonomy.md`) are different vocabularies. Old categories stay as-is on old data; new data gets `family`. Do not retrofit families onto old rows — 89% are `unknown` and any mapping would be guesswork.
- **Version drift**: `constitution.md` is v1.1, `attack_taxonomy.md` is v1.0. The v1.1 change (F2/H4 boundary) affects both documents' subject matter. Acceptable now (no data generated yet), but Phase 2 must generate against constitution v1.1 + taxonomy v1.0 and record both versions in provenance.
- **Overlap that helps**: `docs/v2_results.md` explicitly calls for "a properly constructed held-out benchmark, built directly from the failure clusters." The H1–H4 held-out eval (Phases 2–4 below) is that benchmark, upgraded from failure-cluster patching to a generalization test. The old plan's dead end is the new plan's entry point.
- **Scope tension, must be resolved in Phase 2**: `constitution.md` scopes the classifier to "a single piece of content," but H3 (multi-turn crescendo) is only visible across turns. Resolution (decided here, execute in Phase 2): H3 examples are serialized single-document transcripts (`User: ...\nAssistant: ...\nUser: ...`), classified as one input. This tests whether the classifier reads aggregate intent in a transcript, which is the realistic deployment shape for a pre-LLM input filter sitting in front of a chat product. Document this in the H3 generation prompt and in the eval writeup. Note `MAX_LENGTH = 256` tokens in all inference code — H3 transcripts must fit or be truncated honestly; measure truncation rate.

## 1.4 Gaps (each is real, verified, and fixable)

**Portfolio-breaking:**
1. **`alvi42/prompt-injection-guard-v1` is PRIVATE on the Hub** (API returns 401). The README links its model card; that link 404s for everyone but the owner. Worse: `src/api/main.py:102` and `src/demo/app.py` load the tokenizer from this private repo, so the advertised Docker/local quickstart **cannot work for anyone else** — a fresh clone with a stranger's HF_TOKEN fails at model load. `docs/training.md` even records "(private)". v2 is public. Fix: flip v1 to public (Hub settings, 2 minutes) and verify `curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/alvi42/prompt-injection-guard-v1` returns 200 without auth.
2. **Two resume .docx files sit untracked at repo root** (`M Shihab Resume N*.docx`). Move them out of the repo before any `git add -A` ever happens.

**Stale/wrong docs:**
3. `docs/model_card.md` says training data includes "allenai/wildjailbreak (safeguard split)" — wrong; the source is `xTRam1/safe-guard-prompt-injection`. It also says "Four public datasets" over a three-row table.
4. README dataset table: "jasperai/prompt-injections — 662". Actual DB: source `jasperai` = 653 rows, and the correct Hub ID is `JasperLS/prompt-injections` (noted in `docs/data.md`). Also, `unified` contains **no deepset rows** even though the loader and unify mapping exist — deepset was evidently removed wholesale by MinHash dedup against its near-copy JasperLS. This is nowhere documented; README's "unified 4 public datasets" is only true pre-dedup.
5. `docs/failures.md` says 17 errors (10 FP / 7 FN) across val+test; the `v1_errors` table holds only the 10 val-set errors (6 FP / 4 FN). Test-set errors were never persisted. Not blocking, but the table and doc disagree with each other's framing.

**Dead code / never-run code:**
6. `scripts/push_model_card.py` is 0 bytes. Delete it (`scripts/push_v2_model_card.py` is the real one).
7. `scripts/eval_ood_wildguard.py` (and `make eval-ood`) was committed but never run — no `docs/ood_results.json` exists. Either run it once and commit results, or delete the Makefile target. Its docstring honestly says it's not an injection benchmark; H1–H4 supersedes it.
8. `make eval` and `make eval-ood` default to `--model-path checkpoints/v1`, which does not exist locally (only `v1-onnx/` and `v1-onnx-int8/`). Evaluation currently only reproduces by pointing at the Hub repo — which is private (gap 1).

**Missing tests:**
9. No tests for eval code, training argument parsing, synth generation JSON parsing, or ONNX-vs-PyTorch prediction parity. Test coverage is data-schema + API only. The parity check was done manually on Day 16 but never encoded.

**Known model weaknesses (documented, unquantified):**
10. Calibration: README talking points admit confidence runs 65–73% on textbook injections. No ECE, no reliability diagram, no calibration set exists. The API and demo surface these raw softmax scores to users.

---

# PART 2: DIRECTIONS CONSIDERED (Task 2)

Six directions were weighed against the three Safeguards responsibilities: (R1) classifiers to detect misuse at scale, (R2) synthetic data pipelines, (R3) prompt injection mitigations.

**Chosen:**

**A. Held-out family generalization (H1–H4 as true OOD eval).** Evidence produced: per-family recall on four never-trained attack families plus over-refusal rate on a benign contrast set — the memorization-vs-generalization question `attack_taxonomy.md` was written to answer. Teaches: eval-set construction discipline (train/eval contamination control is the #1 practical skill in classifier evals), and honest reporting when a family fails. Role mapping: R1 + R2 directly; this is a miniature of how constitutional-classifier evals are actually run. This is also the only direction the repo is already contractually committed to — the taxonomy and constitution exist solely to feed it.

**B. Calibration + false-positive cost modeling (one direction, not two).** Evidence: ECE and reliability diagram for v1 on `split_val`, temperature-scaled variant, then an operating-threshold analysis under explicit FP:FN cost ratios (a moderation deployment blocks real users on FPs; cost asymmetry is the whole game). Teaches: that a 0.99-F1 classifier can still be undeployable at a fixed threshold, and how production teams actually choose operating points. Role mapping: R1 ("at scale" means thresholds, not argmax). Cheap: no GPU, no API spend, ~2 sessions. The known 65–73% confidence gap (audit item 10) means there is a real finding here, not a checkbox.

**C. Automated red-team loop.** Evidence: attack success rate (ASR) of a generator model against the classifier, per family, before and after one hardening iteration. Teaches: adversarial dynamics — the defender's metric is not F1 on a static set but ASR against an adaptive attacker; this is the closest a portfolio can get to Safeguards' actual loop. Role mapping: R2 + R3, and it's the flashiest legitimate artifact for the posting. Runs last because it needs A's eval harness and per-family reporting to score anything.

**Rejected or folded in:**

- **Latency/throughput at real volumes**: rejected as a phase. `docs/performance.md` already documents the honest hardware story (Apple Silicon kernel doesn't vectorize; target was AVX-512). Re-benchmarking on a rented c5 instance produces a number, not a capability. Not worth a phase; a one-hour add-on at most.
- **Comparison to Anthropic's Constitutional Classifiers paper**: not a standalone phase because it can't pass or fail. Folded into Phase 7's writeup as a required section (what this project borrows: constitution-driven generation, held-out attack families, over-refusal budget; what it lacks: output classification, rubric-graded harmfulness, streaming prediction).
- **WildGuardMix OOD eval**: superseded by A (it measures general harmfulness transfer, not injection). Phase 2 pre-work either runs it once for the record or deletes the target (audit item 7).

---

# PART 3: ROADMAP (Task 3)

Continues the existing phase numbering. Phases 0–1 are done. Session ≈ 3 focused hours (this project's historical unit).

## Phase 2 — Constitution-driven training data for F1–F6 (+ repo hygiene)

**Pre-work checklist (30 minutes, do first, no debate):**
- Flip `alvi42/prompt-injection-guard-v1` to public on the Hub; verify unauthenticated 200 (audit gap 1).
- Move both resume .docx files out of the repo root (gap 2).
- Delete `scripts/push_model_card.py` (gap 6).
- Fix `docs/model_card.md` dataset section and README dataset table; document the deepset-deduped-away fact in `docs/data.md` (gaps 3–4).
- Decide `make eval-ood`: run once and commit `docs/ood_results.json`, or delete the target (gap 7).

**Deliverable:** a `family_data` table in `data/unified.duckdb` with schema `id, text, label, family, split, constitution_version, taxonomy_version, generator_model, created_at`, populated with generated examples for **train families only**: F1–F6 positives plus constitution NR1–NR6 benign hard negatives. Target: 200 positives per family (1,200) + 800 hard negatives = 2,000 examples. Generator: Claude Sonnet, prompts built from the family definitions in `attack_taxonomy.md` and the rules in `constitution.md` (reuse the loop structure of `src/synth/generate.py`; new module `src/synth/generate_families.py`). Apply the existing quality gauntlet: embedding dedup vs `split_train_v2` (threshold 0.92, `scripts/dedup_synthetic.py` pattern) + cross-model label agreement (Haiku, `scripts/crossval_label.py` pattern). Enforce the F2/H4 boundary rule from `constitution.md` §Boundary: any example the labeler flags as H4-ish is dropped from F2, not relabeled.

**Gate (pass/fail):** every family retains ≥150 surviving examples after filtering AND cross-model label agreement ≥95% overall AND `SELECT family, COUNT(*) FROM family_data GROUP BY family` is committed in a `docs/phase2_data.md` with drop rates per stage. Add 3+ tests to `tests/test_data.py`: `family_data` exists, family values ∈ {F1..F6, benign}, no text overlap with any `split_*` table.

**Estimate:** 2 sessions (~6h), ~$4–6 API.

## Phase 3 — Held-out eval set: H1–H4 + benign contrast

**Deliverable:** `heldout_eval` table, same schema, populated with H1–H4 positives (150 per family = 600) and 400 benign-contrast examples per `attack_taxonomy.md` §Benign contrast set (own-draft revisions, security-education content, quoted attacks in analysis, injection-vocabulary-in-benign-use). H3 examples are single-document serialized transcripts (see audit §1.3); record what fraction exceed 256 tokens. Generated by a **different model than Phase 2's generator** (e.g., generate with Opus, validate with Sonnet) so train and eval sets don't share one generator's fingerprint — say this in the writeup, it's the kind of contamination thinking the role wants.

**Contamination controls (hard requirements):** zero examples enter any training table, ever; max cosine similarity of any `heldout_eval` row against `family_data` ∪ `split_train_v2` < 0.85 (reuse the embedding pipeline); manual read of 50 random examples with notes.

**Gate:** all four H families and the contrast set at full target counts AND validator-model label agreement ≥95% AND the contamination similarity report committed in `docs/phase3_heldout.md`. A test in `tests/test_data.py` asserting `heldout_eval` ∩ training tables = ∅ by exact text match.

**Estimate:** 2 sessions (~6h), ~$4–8 API (Opus generation is pricier per token; volume is small).

## Phase 4 — Train v3, run the generalization eval (the "Phase 4 deliverable" promised in docs/baseline.md)

**Deliverable:** v3 = DeBERTa-v3-base trained on `split_train` + `family_data` (same hyperparameters as v1/v2, `docs/training.md`: lr 2e-5, batch 16, 3 epochs, fp16 off, transformers==4.47.0, Colab T4). New script `scripts/eval_heldout.py`: evaluates **v1 and v3** on `heldout_eval`, reporting per-family recall (H1, H2, H3, H4 separately — the taxonomy forbids averaging them) and false-positive rate on the benign contrast set. Also re-run `scripts/eval_baseline.py` on `split_test` for v3 to confirm no in-distribution regression. Results to `docs/phase4_generalization.md` with a per-family bar chart in `assets/`.

**Gate:** the eval runs and the following are recorded per model: recall on each of H1–H4, contrast-set FPR, `split_test` macro F1. **Pass** = v3 mean held-out recall exceeds v1's AND contrast FPR ≤ 10% AND `split_test` macro F1 ≥ 0.99. **Fail** = anything else, written up in the honest-result style of `docs/v2_results.md` (a v1-generalizes-fine result or a v3-overfits result are both publishable findings — the gate fails, the project doesn't).

**Estimate:** 2 sessions (~6h; one training run ~30 min on T4), ~$0 API.

## Phase 5 — Calibration and operating-point analysis

**Deliverable:** `scripts/calibration.py`: reliability diagram + ECE (15 bins) for the deployed model on `split_val` and on `heldout_eval`; temperature scaling fit on `split_val` only; then a cost-model section: expected cost per 10k requests at FP:FN cost ratios 1:1, 1:10, 1:50, with the optimal threshold at each ratio, for raw vs. temperature-scaled scores. Output: `docs/phase5_calibration.md` + two figures in `assets/`. If the ONNX int8 and PyTorch models diverge in score distribution, report both (the API serves int8 — `checkpoints/v1-onnx-int8/` — so int8 is the one that matters; this also retires audit gap 9's parity question with data).

**Gate:** ECE reported pre/post temperature scaling on both eval sets. **Pass** = post-scaling ECE ≤ 0.05 on `split_val` AND a recommended production threshold per cost ratio is stated in one summary table. **Fail** = scaling can't reach 0.05 (report why — likely the saturated training distribution).

**Estimate:** 2 sessions (~6h), $0 API, no GPU.

## Phase 6 — Automated red-team loop (one full iteration)

**Deliverable:** `src/redteam/loop.py`: (1) attacker model (Sonnet) generates candidate attacks per family — all ten F+H families now that H data exists independently — explicitly instructed to defeat a classifier whose constitution it is shown; (2) each candidate is scored against the deployed classifier via `_run_batch_inference` (import from `src/api/main.py` or load the ONNX session directly); (3) successes (classifier says benign, validator model confirms it's a real attack) land in a `redteam_pool` DuckDB table with family, generation, and score; (4) one hardening round: fold train-family successes into training data, retrain (v4), re-measure ASR on a fresh attack batch. Budget cap: 1,000 attacker generations per round.

**Gate:** ASR of the attacker against the pre-hardening model and post-hardening model, per family, in `docs/phase6_redteam.md`. **Pass** = post-hardening ASR strictly lower on train families with held-out family recall (Phase 4 eval, re-run) not degraded by more than 2 points. **Fail** = ASR doesn't move or held-out recall collapses (a whack-a-mole finding — again publishable).

**Estimate:** 3 sessions (~9h), ~$8–15 API. The expensive phase; do not start it before Phases 4–5 are logged.

## Phase 7 — Consolidation, model cards, and the constitutional-classifiers comparison

**Deliverable:** README rewritten around the completed arc (iteration loop → constitution → held-out generalization → calibration → red-team); v3/v4 model card published public on the Hub with per-family held-out metrics; a `docs/comparison_constitutional_classifiers.md` section comparing this design to Anthropic's published Constitutional Classifiers work (borrowed: constitution-driven synthetic data, held-out attack families, over-refusal budget; absent: output-side classification, streaming, rubric grading); all stale-doc fixes from the audit verified done.

**Gate:** a checklist audit of every quantitative claim in README against a repo artifact — each claim gets a file:line citation in the PR description. **Pass** = zero claims without a backing artifact AND all model links resolve publicly (unauthenticated curl 200). This is binary and 30 minutes to verify.

**Estimate:** 1–2 sessions (~4h), $0 API.

---

## Gate log

| Phase | Date | Gate | Result | Evidence |
|---|---|---|---|---|
| 0 | 2026-06-28 | Baseline metrics on split_test for v1+v2 | PASSED | `docs/baseline.md` |
| 1 | 2026-07-01 | Constitution + taxonomy committed, versioned | PASSED | `constitution.md` v1.1, `attack_taxonomy.md` v1.0 |
| 2 | | ≥150/family survive filters, agreement ≥95%, tests added | | |
| 3 | | H1–H4 + contrast at target counts, contamination report | | |
| 4 | | Per-family recall + contrast FPR recorded for v1 and v3 | | |
| 5 | | ECE pre/post scaling, threshold per cost ratio | | |
| 6 | | ASR pre/post hardening per family | | |
| 7 | | Every README claim artifact-backed, links public | | |
