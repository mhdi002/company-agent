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
