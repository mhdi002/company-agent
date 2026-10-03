# Phase reports

## Phase 0 — Plan
* Hardware: no GPU, 4 CPU cores, 15 GiB RAM, ~30 GB disk; network limited to package registries.
* Stack chosen: PyTorch, HF tokenizers trainer, requests+bs4, docx-js (per docx skill) with python-docx fallback, FastAPI UI.
* `docs/ARCHITECTURE.md` written (diagram, to-do flow, data flow, budget: 1.000B params, 20B tokens, 76.3k steps, ≈6 days on 1×H100).
* Tests: `tests/test_config.py` (config + env overrides), model param-count sanity.

## Phase 1 — Public data collection
* `data/registry.py`: 13 datasets in three groups (general: FineWeb-Edu, Wikipedia, PG-19; business: EDGAR 10-K business sections, CORDIS H2020/Horizon project objectives, NSF & NIH award abstracts, EU TED tenders, arXiv abstracts, BIG-PATENT; agentic: Glaive & Hermes function-calling, plus our synthetic SRLM traces). License + source recorded for each.
* `data/download.py`: HTTP Range resume from `.part`, retry with exponential backoff, SHA-256 sidecars verified against HF LFS digests when available, conversion of parquet/jsonl/json/csv/zip-csv/zip-xml to `docs.jsonl`, `data/MANIFEST.md` generation.
* Network to dataset hosts is blocked in this container, so the bundled offline fixture corpus (`data/sample/`, 3,420 docs with deliberate defects) was installed with `python -m data.download --sample`.
* Full-run command: `python -m data.download --groups general business agentic --max-docs 20000000`.
* Tests: `tests/test_download.py` (8 passed): resume from partial file, resume after dropped connection, checksum mismatch, format conversion, manifest rendering.

## Phase 2 — Complete data cleaning
* `data/clean/`: `normalize` (mojibake repair, NFKC, control chars, HTML + boilerplate removal), `langid` (own char-trigram naive Bayes + stop-word prior; `data.languages` configurable), `quality` (length, symbol/digit/alpha ratios, duplicate lines, top 2/3-gram fraction, stop-word and word-length checks, **own Kneser-Ney trigram reference LM** perplexity filter), `dedup` (exact SHA-1 + MinHash LSH 128 perm × 32 bands, cross-dataset labels), `pii` (email, phone, SSN/IDs, IBAN, Luhn-checked cards, IPs, street addresses → placeholders; lexicon toxicity/NSFW), `decontam` (13-gram overlap vs. HellaSwag/ARC/MMLU/PIQA/WinoGrande/GSM8K/LAMBADA, offline fallback list), `shard` (stage 7: `<|bos|> … <|eos|>`, uint16/uint32 shards, deterministic hash-based train/val split).
* `python -m data.clean.pipeline` writes per-stage stats and 100 kept + 100 removed samples to `data/reports/`, plus `cleaning_report.md`.
* Sample run: 3,420 → 2,882 docs (84.3 % kept): 220 non-English, 80 low quality, 100 exact + 98 near duplicates, 30 toxic, 40 PII-scrubbed, 10 benchmark-contaminated.
* Tests: `tests/test_cleaning.py` (12 passed) incl. full pipeline end-to-end and shard determinism.

## Phase 3 — Tokenizer and model from scratch
* `tokenizer/`: byte-level BPE trained on our cleaned corpus (+ synthetic agent traces), 48k target vocab, 15 special tokens including `<|step|> <|thought|> <|tool|> <|args|> <|result|> <|todo|> <|end|>` (atomic, tested).
* `model/model.py`: decoder-only transformer from scratch — RoPE (θ=5e5, linear position interpolation), RMSNorm, SwiGLU, GQA, tied embeddings, KV cache (incremental + chunked prefill), gradient checkpointing, sampling with top-p and a logits-processor hook for constrained decoding. `1b` preset = **1,000,163,328** params.
* `training/pretrain.py`: bf16 autocast, SDPA/FlashAttention, grad checkpointing, accumulation, cosine LR + warmup, atomic checkpoints with optimizer + RNG, resume, val perplexity, samples, DDP via torchrun.
* Tiny smoke run (CPU, 6.9M params on the sample shards, 300 steps, seq 256): loss 7.91 → 0.49, val ppl 1.56 (sample corpus is highly repetitive), ≈20k tok/s, 35 s.
* Full run: `python -m tokenizer.train && python -m data.clean.shard && torchrun --nproc_per_node 8 -m training.pretrain --preset 1b --resume` (≈2.5 days on 8×A100, ≈6 days on 1×H100; see ARCHITECTURE §6).
* Tests: `tests/test_model.py` (12: param count incl. 1B on meta device, causal masking, KV cache equivalence, chunked prefill, grad-checkpoint gradient equality, RoPE scaling, loss mask, constrained generation, pretrain+resume), `tests/test_tokenizer.py` (2).

## SRLM engine (spec §5, `agent/srlm/`)
* `repl.py` + `sandbox_worker.py`: one persistent subprocess per candidate; context lives in REPL variables (`context`, `pages`, `evidence`, `facts`, `corpus`, …), never in the prompt. Limits: per-step wall-clock timeout (default 60 s, kill + restart), RLIMIT_AS (512 MB), RLIMIT_CPU, RLIMIT_FSIZE, no core dumps, private temp cwd, AST validation (allow-listed imports only; no dunder/frame attributes; no dunder strings), restricted builtins (no open/eval/exec/compile/getattr/type…), socket/subprocess/urllib blocked. Helper API: `search_context`, `bm25`, `slice`, `extract_fields`, plus `sentences`, `overlap`, `project_from_gap`, `describe`, `FINAL`. `sub_call` exists only when `use_subcalls: true` (default OFF).
* `protocol.py`: step grammar, the exact confidence instruction appended to every prompt, robust confidence parsing (fenced block → loose fallback; range (0,100]).
* `select.py`: `VC(p)=Σ log(ν_t/100)`, missing ν filled with the trajectory mean, `Len(p)=Σ l_t` (our tokenizer), `s(p)=VC·Len ≤ 0`, plurality + consistent set, argmax within S, tie-breaks (mean confidence → shorter → lower index), `low_agreement`, retry → "Not verified". Ablation switches for every signal + `direct_baseline`.
* `engine.py`: K candidates in parallel threads (each with its own sandbox); every decision unit logged to `srlm.jsonl` (query, K, code, per-step outputs and confidences, VC, Len, s(p), canonical output, prob(a), S, selection, low_agreement/retry, wall-clock, violations).
* `tasks.py`: the four decision units with grounding validators (facts must be quoted from their cited source; claims must be grounded in their source ids).
* Tests: `tests/test_srlm.py` (30): sign convention, consistent set, missing-confidence fill, tie-breaks, tied plurality, ablations, retry trigger, confidence parsing, prompt never contains context, clustering, sandbox state/violations/timeout/memory/truncation, parallel speed-up, Not-verified path, invalid FINAL rejection, direct baseline.

## Phase 4 — Agent-format training
* `agent/synth/generate.py`: programmatic episodes (fictional companies built from templates + real cleaned text as background pages; evidence with off-field distractors). For each decision unit, K=4 programs sampled at temperature from the program library (good, flawed: wrong slice / missed page / buggy call / paraphrased fact / unsourced claim / trends-only, and recoveries after errors), **executed in the sandbox**, scored against ground truth; per-step confidence targets set from actual correctness (graded). Fixture companies are held out for validation.
* `training/sft.py`: target-only loss (thought, program, confidence block, `<|end|>`), init from our pretrained checkpoint.
* `agent/policy.py` `ModelPolicy`: constrained decoding — thought → forced `<|program|>` + code fence → fenced JSON confidence with a number-only grammar → `<|end|>`; every step is valid by construction.
* Smoke results (CPU, tiny 6.9 M model; 60 synthetic companies → 3,466 train / 699 val steps; 600 SFT steps):
  * SFT val loss 0.105; **unconstrained step-format accuracy 99.3 %**, confidence parse rate 100 % (constrained: 100 % by construction).
  * **SRLM field accuracy with our model writing the programs: 9/12 = 75 %** on held-out fixture companies (K=2, 4 steps); template policy reference 12/12.
  * **Calibration: ECE 0.265, Brier 0.223** for the model's verbalized confidence (targets themselves: ECE 0.040). The tiny model is not yet calibrated; ECE is reported in `docs/eval/eval_model.json` with the reliability curve.
  * Caveat: the held-out companies share page templates with the synthetic training companies, so this is an in-distribution check of the pipeline, not a measure of real-web generalisation.

## Phase 5 — Tools
* `tools/http.py` (UA, robots.txt, per-host delay, retries with backoff + Retry-After, size cap, disk cache, offline transport), `search.py` (DuckDuckGo HTML / seed file / offline; aggregator + social filtering; dedup by domain; failing queries isolated), `fetch.py` + `extract.py` (home + about/services/products, boilerplate-free text), `field_classifier.py` (name, country, field, services, size-when-stated), `research.py` (Wikipedia API evidence with URLs / offline KB), `docx_writer.py` (docx-js following the docx skill; python-docx fallback; cover, TOC, headings, tables, bullets, page X of Y), `telegram.py` (chunking, retry incl. 429 retry_after, delivery receipts).
* LibreOffice cannot open any file in this container, so the visual render step of the docx skill could not run; the skill's `validate.py` passes on the docx-js output, and tests check TOC field, PAGE fields, headings, and tables.
* Tests: `tests/test_tools.py` (33) incl. failure injection: network down, bad HTML, Telegram 400/429.

## Phase 6 — Agent loop
* `agent/loop.py`: Start/Stop/Resume, persistent `state.json` (atomic), per-step outputs saved under `outputs/cache/work/<domain>/` and reused on resume, per-company isolation, dedup by domain across runs, daily limit, configurable concurrency; SRLM for items 2/3/4; to-do statuses pending/in_progress/done/failed(/skipped).
* Offline end-to-end run (template policy): 8 fixture companies → 8 `.docx` + `.json` records, gap-derived top projects, unverifiable sections marked "Not verified".
* Fixed during this phase: double-escaped gap regex in the project programs; correlated evidence errors (BM25 matched "contract logistics" to "contract manufacturers") → evidence programs now keep only documents whose classified field matches; subset answers merging into the consistent set → separate `set_agreement_threshold: 0.8`.
* Tests: `tests/test_agent.py` (7): full run with Telegram delivery and log sending, dedup + daily limit, per-company isolation, Telegram down, search network down, stop/resume, crash recovery without redoing completed steps.

## Phase 7 — Logging
* Channels `agent`, `tools`, `training`, `errors`, `srlm`: structured JSONL plus human lines `timestamp | run_id | company | step | tool | status | duration | details` (`srlm` is `srlm.jsonl` only). Error events (with stack traces) are mirrored to `errors.log`/`errors.jsonl`.
* Logged: tool inputs/outputs (truncated), every model prompt/output step (`srlm.step`, `srlm.log_prompts`), retries, REPL errors and sandbox violations, to-do status transitions, run lifecycle, every SRLM decision unit (`srlm.decision`).
* In-memory ring buffer with channel/status/company/text filters for the UI; seeded from log tails after a restart.
* Telegram: each proposal (.docx + .json) right after it is finished, a status message after every company, log files every `agent.send_logs_every` companies and at run end.
* Tests: `tests/test_logging.py` (4).

## Phase 8 — UI
* `python -m app` → FastAPI + a single static page (`app/static/index.html`, no build step): Start/Resume, Stop, Reset; live to-do list (run item + one expandable group per company with per-step status dots); current company; counters (found, processed, failed, sent); log viewer with channel/status/company/text filters and follow mode; proposals list with .docx/.json downloads; settings page (agent, SRLM incl. all ablation switches, providers, politeness, Telegram, docx backend) saved to `config.override.yaml`, Telegram secrets saved to `.env` and never echoed back. Works at phone width.
* Screenshots: `docs/ui_dashboard.png`, `docs/ui_logs.png` (offline run, 4 companies).
* Tests: `tests/test_app.py` (3): start → live status → done, log filters, file download + path-traversal rejection, settings validation and secret handling, stop + reset.

## Phase 9 — Evaluation and hardening
* **End-to-end, 10 companies:** `tests/test_e2e.py::test_e2e_offline_10` runs 10 fixture companies through the whole agent (180 SRLM decision units, 10 `.docx` + 10 records, every `[S#]` citation resolves). `test_e2e_real_10` (real DuckDuckGo + websites + Wikipedia) exists but is **skipped here**: the container's network allows package registries only. Run it with `PA_NETWORK_TESTS=1 python -m pytest tests/test_e2e.py`.
* **Failure injection:** network down (search, fetch), unreachable company site, bad HTML, Telegram 400/429/connection errors, REPL worker killed mid-program (now reported to the program, not silently reset — bug fixed), corrupt state file, timeouts, memory limit, sandbox violations.
* **Ablations** (`training/ablation.py`, results in `docs/eval/`):

Template policy with flawed programs at full weight (noisy programmatic policy), K=8, 30 held-out companies (12 fixtures + 18 synthetic):

| Variant | Field acc. | Evidence P | Evidence R | Project | Halluc. | Completeness | s/company |
|---|---|---|---|---|---|---|---|
| direct | 0.900 | 0.922 | 0.873 | 0.652 | 0.0 | 0.984 | 0.78 |
| +verb_conf | 0.933 | 0.633 | 0.529 | 0.267 | 0.0 | 0.980 | 2.06 |
| +trace_len | 0.933 | 0.633 | 0.529 | 0.267 | 0.0 | 0.980 | 2.28 |
| +self_consist | 0.933 | **1.000** | **0.891** | **0.712** | 0.0 | 0.984 | 2.34 |
| srlm (full) | 0.933 | 0.967 | 0.874 | 0.638 | 0.0 | 0.984 | 2.20 |

Our tiny SFT model (6.9 M params) writing the programs, field extraction, K=4, 12 held-out companies:

| Variant | Field acc. | Not verified | s/company |
|---|---|---|---|
| direct | 0.500 | 6 | 0.71 |
| +verb_conf / +trace_len / +self_consist / srlm | **0.917** | 1 | 6.0 |

* Reading: search + selection gives a large gain for the model (0.50 → 0.92). Each signal alone already recovers it here because K=4 candidates rarely disagree once one succeeds. For the template policy, VC or Len alone *hurts*: its confidences are heuristic, not calibrated, so short flawed programs (one-step paraphrases, trends-only) win. Self-consistency removes them. Inside the consistent set, VC·Len still slightly prefers concise subset answers, which costs a little recall and project score versus SC alone. The paper's VC term assumes calibrated confidence; our trained model's ECE is 0.265 at smoke scale, so calibration is the main lever for the full run.
* **Hallucination rate 0.0** by construction: every `[S#]` sentence is a verbatim quote validated against its cited source (grounding validators reject anything else). **Completeness 98.4 %** of sections verified; the rest are explicitly "Not verified".
* **Calibration (ECE)** of the model's verbalized confidence: 0.265 (Brier 0.223) on held-out steps; training targets themselves: 0.040. See `docs/eval/eval_model.json`.
* README written (setup, data download, training, running, troubleshooting). Full suite: **118 passed, 1 skipped (network)**.

## Follow-up — live internet, Telegram in the app, one-command launch
* Network became available, so the live path was tested: **`test_e2e_real_10` passes: 10/10 real companies → 10 proposals.** Search was Wikidata, crawling was polite with robots.txt respected, research was Wikipedia. 4 of 14 candidate sites refused the bot (HTTP 403) or had broken TLS; they were skipped and reserve candidates took their place.
* DuckDuckGo answers this server with a bot challenge. That is now detected (`SearchBlocked`), and Wikidata is the default provider.
* **Honest quality note:** with the deterministic fallback brain (no trained 1B checkpoint yet), general Wikipedia articles yield generic project ideas. Programs now keep only current, actionable statements (`is_current`, action-term filter), so definitions are no longer shown as projects; sections without such evidence say "Not verified". Field detection on real sites: 7 of 10 correct field labels, and country "unknown" when the site never states it.
* **No hardcoding:** endpoints, query templates, the industry/country maps, budget bands, timeline, team, Telegram limits, heartbeat and log sizes all moved to `config.yaml`.
* **Telegram wired into the app:** a dashboard card handles connect (token verified via getMe) → open the bot → press Start (chat paired automatically) → test/disconnect. The bot accepts `/run /stop /status /logs` from the paired chat only. Heartbeat status messages are sent every `telegram.heartbeat_minutes`.
* **Continuous mode + heartbeat:** after Start the agent cycles runs every `agent.run_interval_minutes` until Stop. Heartbeat and next run are shown in the UI.
* **One-command launch:** `run.sh` (Linux/macOS) and `run.bat`/`run.ps1` (Windows) install Python (if missing), the venv, PyTorch (CUDA or CPU) and requirements, plus docx-js; then they create `.env`, start the app and open the browser. Tested in a fresh clone on Linux (install ≈45 s with cached wheels; offline run → 8 proposals → waiting for next cycle). `run.ps1` passes the PowerShell 7 parser. It could not be executed here (Linux container), and the sandbox REPL was made Windows-compatible (reader thread instead of `select`; POSIX rlimits only where available).
* Tests: 128 passed, 1 network test (passes when `PA_NETWORK_TESTS=1`).
