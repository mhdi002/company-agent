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
