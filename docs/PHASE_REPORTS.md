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
