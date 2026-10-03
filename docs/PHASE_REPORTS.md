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
