# ProposalAgent — Architecture

## 1. Hardware inspected (Phase 0)

| Item | Found in the build container |
|---|---|
| GPU | **none** (`nvidia-smi` not present) |
| CPU | 4 cores |
| RAM | 15 GiB |
| Disk | ~30 GB free |
| Network | Package registries only (PyPI/npm); all other hosts denied by the sandbox proxy |

Consequence (per the hard rules): the complete pipeline is implemented and verified end-to-end on the
`tiny` model preset (6.9 M params) and on bundled offline fixtures. The full 1B run is documented
with exact commands and time estimates (§5) and runs unchanged on a GPU machine.

## 2. Stack

* Python 3.11, PyTorch 2.x (model, training, inference), HF `tokenizers` (BPE trainer only — the vocabulary is trained by us on our cleaned corpus).
* `requests` + `beautifulsoup4`/`lxml` for the polite crawler; `huggingface_hub` for dataset downloads.
* Word output: **docx-js** (Node, following `/mnt/skills/public/docx/SKILL.md`) with a `python-docx` fallback.
* UI: FastAPI + a single static HTML/JS page (no build step).
* State: JSON files with atomic writes (crash-safe), JSONL logs.

## 3. Component diagram

```
                ┌──────────────────────── app/ (FastAPI UI) ───────────────────────┐
                │ Start/Stop · live to-do · counters · log viewer · settings      │
                └───────────────┬─────────────────────────────────────────────────┘
                                │ start()/stop()
                ┌───────────────▼──────────────── agent/loop.py ───────────────────┐
                │ Orchestrator: per-company to-do, state.json (crash recovery),    │
                │ domain dedup, concurrency, daily limit, Telegram delivery         │
                └──┬───────────────┬──────────────────┬──────────────────┬─────────┘
                   │ tools/        │ decision units   │ prose            │ output
     ┌─────────────▼───┐   ┌───────▼────────────┐ ┌───▼──────────┐ ┌─────▼────────────┐
     │ search  fetch   │   │ agent/srlm/engine  │ │ agent/       │ │ tools/docx_writer│
     │ extract research│   │ K programs ∥ REPL  │ │ proposal.py  │ │ tools/telegram   │
     │ field_classifier│   │ SC + VC + Len      │ │ (fact sheet  │ └──────────────────┘
     └────────┬────────┘   └───────┬────────────┘ │  → prose)    │
              │ pages, evidence    │ prompts       └──────┬───────┘
              ▼                    ▼                      ▼
     ┌──────────────────┐  ┌────────────────────────────────────────┐
     │ agent/srlm/repl  │  │ agent/policy.py → OUR model (model/)    │
     │ sandbox vars:    │  │  tokenizer/ BPE  ·  constrained JSON    │
     │ context, pages,  │  │  decoding · confidence block enforced   │
     │ evidence, draft  │  └────────────────────────────────────────┘
     └──────────────────┘
```

Training side:

```
data/download.py → data/raw → data/clean/pipeline.py (7 stages, reports) → data/cleaned
   → tokenizer/train.py (BPE 48k + agent special tokens) → data/clean/shard.py (uint16/uint32 shards, train/val)
   → training/pretrain.py (1B, 8K ctx) → agent/synth/generate.py (SRLM program traces, calibrated confidence)
   → training/sft.py → training/eval.py (step format, field acc., ECE, proposal quality)
```

## 4. To-do flow (per run, per company)

```
Start ─▶ [1] Gather companies' websites   (search tool; dedup by domain; daily limit)
         for each company (isolated, concurrent up to agent.concurrency):
           [2] Fetch working field          fetch+extract → pages[domain] → SRLM(field extraction)
           [3] Research the field           research tool → evidence → SRLM(evidence extraction)
                                            → SRLM(project selection)
           [4] Find text for the proposal   SRLM(fact sheet) per section
           [5] Fetch & read docx skill      reads SKILL.md (once per run, cached)
           [6] Finalize the proposal        prose from verified fact sheets → .docx + .json
           [7] Save & send                  outputs/ + Telegram (doc, record, status msg)
         every N companies + at end: send log files
```

Each item moves pending → in_progress → done | failed; a failure fails only that company.

## 5. Data flow and SRLM

Long material never enters the prompt. It lives as variables inside a sandboxed REPL subprocess:
`context` (all text for the decision unit), `pages` (domain → url → text), `evidence` (list of
`{id,url,text}`), `draft`. The model sees only: the query, a short state summary (variable names,
sizes), and the printed output of its own previous steps. For each decision unit the engine samples
K programs (default 8) in parallel, executes each step, and selects with
`s(p) = VC(p) · Len(p)` inside the self-consistent set (§5.5 of the spec; see `agent/srlm/select.py`).

## 6. Training budget (full run)

| Quantity | Value |
|---|---|
| Parameters | 1.000 B (`1b` preset: d=2048, L=20, 16 heads, 4 KV heads, SwiGLU 5632, vocab 48k, tied) |
| Tokens (Chinchilla ≈ 20×N) | 20 B |
| Context | 8192 (2048 for first 90 % of steps is an allowed speed-up; RoPE θ=5e5) |
| Tokens / optimizer step | 4 × 8 × 8192 = 262 144 (1 GPU) — scale micro-batch/accum for more GPUs |
| Steps | 76 300 (warmup 2 000, cosine to 10 %) |
| Compute | 6·N·D ≈ 1.2e20 FLOPs + attention ≈ 2.0e20 FLOPs at 8K |
| Wall clock | 1×H100 (~400 TFLOP/s eff.) ≈ 6 days · 1×A100 ≈ 18 days · 8×A100 ≈ 2.5 days |
| SFT (agent traces) | ~300 M tokens, 20 000 steps, ≈ 3–5 % of pretraining time |
| This container (CPU) | tiny preset ≈ 13.7 k tok/s → smoke tests only |

Disk: 20 B tokens × 2 bytes ≈ 40 GB of shards; raw downloads ≈ 120 GB (FineWeb-Edu sample-100BT
subset is streamed and capped by `--max-docs`).
