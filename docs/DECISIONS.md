# Decisions log

| # | Decision | Why |
|---|---|---|
| D1 | Repository root = `proposal-agent/` root of the spec. | The repo already exists as `company-agent`; nesting another folder adds nothing. |
| D2 | No GPU in the build container → implement the full pipeline, verify on the `tiny` preset, document the full 1B commands. | Hard rule in the spec. |
| D3 | Network is restricted to package registries here. Every network tool has an offline provider (`offline`/`seedfile`) and bundled fixtures in `data/sample/`; tests never need the internet. | Allows full end-to-end verification; real providers are the defaults in `config.yaml`. |
| D4 | Tokenizer: byte-level BPE trained with HF `tokenizers` (trainer only), 48k vocab. | We still train our own vocabulary from scratch on our corpus; the Rust trainer is just faster. |
| D5 | 1B preset: d=2048, L=20, H=16, KV=4, FFN=5632, vocab 48k → 1.000 B params (verified by test). | Matches "about 1B" with GQA 4:1. |
| D6 | Attention via `F.scaled_dot_product_attention` (FlashAttention kernels when available). | No extra dependency; same math. |
| D7 | `agent.policy: model` (our checkpoint) is the default runtime brain. A deterministic `template` policy exists — it is the same programmatic program generator that produces the synthetic SFT traces — and is used for tests and when no trained checkpoint exists. | No external LLM is ever called. Keeps the app usable before the multi-day training run completes and lets tests be deterministic. Clearly logged as `policy=template` in every SRLM record. |
| D8 | REPL sandbox = separate Python subprocess with `resource` limits (CPU time, address space), AST-checked imports, restricted builtins, no `open`/sockets, cwd in a temp dir. | Simple, robust on Linux, killable on timeout. |
| D9 | Free-text agreement: normalized key-fact (content-word set) Jaccard by default; optional hashed TF-IDF embedding cosine clustering. | Deterministic, local, no external models. |
| D10 | Word documents use docx-js following `/mnt/skills/public/docx/SKILL.md` (available here). python-docx fallback when Node/docx is missing. | Spec: follow the docx skill if available. |
| D11 | Language ID: our own character-trigram naive-Bayes classifier trained on the bundled multilingual seed sentences; pluggable for fastText `lid.176` if the user downloads it. | No pretrained models required; configurable target languages. |
| D12 | Perplexity filter: our own Kneser-Ney-smoothed word trigram model trained on the high-quality reference slice (Wikipedia + FineWeb-Edu score ≥ 4). | "Small reference model trained on high-quality text". |
| D13 | Toxicity/NSFW: lexicon + density thresholds (configurable list in `data/clean/lexicons.py`). | Transparent and reproducible; a classifier can be added later. |
| D14 | State persistence: JSON with atomic replace; per-company records keyed by domain. | Crash recovery without a DB server. |
| D15 | Proposal facts that cannot be traced to fetched text are written as "Not verified". | Spec requirement; also the SRLM fall-back when retries fail. |
