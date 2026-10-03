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
| D12 | Perplexity filter: our own Kneser-Ney-smoothed word trigram model trained on the high-quality reference slice: heuristic-passing docs from datasets flagged `quality_reference` (Wikipedia, FineWeb-Edu). | "Small reference model trained on high-quality text". |
| D13 | Toxicity/NSFW: lexicon + density thresholds (configurable list in `data/clean/lexicons.py`). | Transparent and reproducible; a classifier can be added later. |
| D14 | State persistence: JSON with atomic replace; per-company records keyed by domain. | Crash recovery without a DB server. |
| D15 | Proposal facts that cannot be traced to fetched text are written as "Not verified". | Spec requirement; also the SRLM fall-back when retries fail. |
| D16 | Field-extraction vote key = normalized `field | country`; services are carried with the winner but not voted on. | Exact match on free-form service lists would almost never agree; field and country are the structured decision. |
| D17 | Project-selection vote key = the set of top-3 normalized titles (order ignored); the winner's order is kept. | Exact match on the full ranked list is too strict; the set captures "which projects". |
| D18 | Lists (evidence facts, fact-sheet claims) agree when soft-Jaccard ≥ `set_agreement_threshold` (0.8). | At 0.6, short subset answers joined the full answer's cluster and then won on conciseness, losing recall (observed in the offline run). |
| D19 | Tied plurality → the cluster whose best member has the best s(p). No plurality (all K differ) → best s(p) over all K, flagged `low_agreement`. | Deterministic, and follows spec §5.5. |
| D20 | `direct_baseline` samples one program with the same decoding settings and no selection. | Isolates the effect of search and selection in ablations. |
| D21 | Calibration targets: final correctness for non-error steps (graded for partial answers), ≤10 for steps that errored, ≤12 when the program never produced a valid answer. | The realised outcome is an unbiased estimate of P(correct given the state), so the trained confidence is calibrated in expectation. |
| D22 | Selected projects are not citable sources; only fetched pages and evidence documents are. | Prevents a project rationale being cited as if it were an external fact. |
| D23 | The docx skill's visual check (LibreOffice → PDF → images) cannot run here because `soffice` fails to load any file in this container. The skill's `validate.py` passes on our output, and tests check the XML structure. | Environment limitation, documented. |
| D24 | Evidence programs keep only documents whose own classified field matches the company's field. | BM25 alone produced correlated errors ("contract logistics" matched "contract manufacturers") that self-consistency could not catch. |
| D25 | Default company search = Wikidata SPARQL (companies with country P17, industry P452, official website P856). DuckDuckGo remains selectable but answers datacenter IPs with a bot challenge, which is now detected and reported. | Live test: DuckDuckGo returned an "anomaly" page; Wikidata returned real companies in ~3 s, free and keyless. |
| D26 | No operational constants in code: endpoints, query templates, industry/country maps, budget bands, timeline, team roles, Telegram limits, heartbeat, log sizes are all in `config.yaml` (overridable via env/UI). | User requirement: no hardcoding. |
| D27 | Telegram is paired from the UI: token verified with getMe → user presses Start in the bot chat → that chat id is stored in .env. Commands are accepted only from the paired chat; 4xx Bot-API errors are not retried. | One-click connection, no manual chat-id lookup, safe against strangers messaging the bot. |
| D28 | Continuous mode (default on) + heartbeat: after a run, the agent waits `run_interval_minutes` and starts a new run (dedup + daily limit still apply) until Stop; a heartbeat is recorded every 30 s and a Telegram status sent every `heartbeat_minutes`. | "Press Start and the heartbeat starts and goes." |
| D29 | Unreachable sites (403, TLS errors, down) are `skipped` and replaced by reserve candidates (`reserve_companies`). | Live run: 3/10 real sites blocked bots or had broken certificates. |
| D30 | REPL sandbox reads worker output through a reader thread (portable timeouts); POSIX rlimits apply on Linux/macOS, Windows relies on the wall-clock timeout + kill. | Windows support for run.ps1. |
