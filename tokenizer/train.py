"""Train the BPE tokenizer on the cleaned corpus (+ agent-format samples).

Usage:
    python -m tokenizer.train                       # vocab from config.yaml (48k)
    python -m tokenizer.train --vocab-size 4000     # tiny smoke test
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from core.config import load_config, resolve
from core.logging import get_logger
from tokenizer.tok import Tok, iter_texts, train_bpe

log = get_logger("training")


def agent_format_texts(n: int = 2000) -> list[str]:
    """A few synthetic agent traces so the agent format tokenizes compactly."""
    try:
        from agent.synth.generate import sample_texts
        return sample_texts(n)
    except Exception:  # synth module not built yet / unavailable
        return []


def main(argv: list[str] | None = None) -> Path:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(resolve(cfg["paths"]["data_cleaned"]) / "cleaned.jsonl"))
    ap.add_argument("--vocab-size", type=int, default=cfg["tokenizer"]["vocab_size"])
    ap.add_argument("--max-docs", type=int, default=2_000_000)
    ap.add_argument("--out", default=str(resolve(cfg["paths"]["tokenizer"])))
    a = ap.parse_args(argv)
    texts = itertools.chain(iter_texts(Path(a.input), a.max_docs), agent_format_texts())
    tk = train_bpe(texts, a.vocab_size, cfg["tokenizer"]["special_tokens"], cfg["tokenizer"]["min_frequency"])
    tok = Tok(tk)
    tok.save(a.out)
    sample = "Nordwind Solar develops solar parks. <|step|> <|thought|> read pages <|end|>"
    log.event("tokenizer.train", "done", details=f"vocab={tok.vocab_size} sample_ids={len(tok.encode(sample))}")
    print(json.dumps({"vocab_size": tok.vocab_size, "out": a.out, "special": tok.special}))
    return Path(a.out)


if __name__ == "__main__":
    main()
