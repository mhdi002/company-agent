# Final report

## What works (verified in this container)
* Full data pipeline: registry of 13 public datasets with licenses, resumable checksummed downloads, 7-stage cleaning with per-stage reports and samples, BPE tokenizer with agent tokens, binary shards.
* From-scratch transformer (1.000 B-param preset; RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, KV cache) and pretraining / SFT with bf16, grad checkpointing, accumulation, cosine LR, resume, DDP.
* SRLM engine exactly as specified: context in a sandboxed REPL, K parallel candidate programs, self-consistency + verbalized confidence + trace length, s(p)=VC·Len selection with tie-breaks, low-agreement flag, retries → "Not verified", full `srlm.jsonl` traces, ablation switches, optional sub_call (off).
* Agent loop with the 7-item to-do list, Stop/Resume, crash recovery, per-company isolation, domain dedup, concurrency, daily limit; polished `.docx` (docx-js per the docx skill) + JSON record; Telegram delivery with retries and receipts; complete logging; FastAPI UI.
* Whole training pipeline run on CPU at tiny scale; the tiny model writes valid program steps 99.3 % of the time unconstrained and reaches 91.7 % field accuracy with SRLM (vs 50 % for a single program).

## Metrics
See `docs/PHASE_REPORTS.md` (Phases 2, 3, 4, 9) and `docs/eval/`.

## Known limits
* **No GPU here.** The 1B model is not trained; the shipped checkpoint is the tiny smoke model (not committed; recreate with `./run.sh --smoke-train`). Until a trained checkpoint exists the agent uses the deterministic fallback policy, which gives generic project ideas on real Wikipedia evidence.
* Live internet was tested later in the session: 10/10 real companies processed (Wikidata → crawl → Wikipedia → .docx). Telegram delivery is tested against the real API for the error path only, because no bot token was available.
* Evaluation companies are fixtures/synthetic and share page templates with the synthetic training data: these are in-distribution pipeline checks, not real-web generalization numbers.
* The tiny model's confidence is poorly calibrated (ECE 0.265); the template policy's confidences are heuristic, which is why VC-only selection underperforms there.
* Field taxonomy (10 fields) and the field classifier are keyword based; services extraction is regex based.
* Planning sections (objectives, timeline, budget band) are templated proposal language derived from selected projects, clearly labelled as estimates, not facts.
* LibreOffice in this container cannot render documents, so the docx skill's visual check was replaced by its XML validator plus structural tests.

## Next steps
1. Allow network access (huggingface.co, dataset hosts, duckduckgo.com, wikipedia.org, api.telegram.org) and run `python -m data.download …` and the cleaning pipeline at scale.
2. Train the tokenizer and the 1B model on a GPU node (commands in README §3; ≈6 days on 1×H100), then generate about 20 k synthetic companies of SRLM traces and run SFT.
3. Re-run `training.eval` and `training.ablation --policy model --k 8` and track ECE; add a calibration-focused SFT round if ECE stays above 0.1.
4. Run `PA_NETWORK_TESTS=1 pytest tests/test_e2e.py` against 10 real companies and review the proposals by hand.
5. Broaden the field taxonomy and use the trained model (not keywords) for the services list.
