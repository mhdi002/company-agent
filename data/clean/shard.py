"""Stage 7: tokenize cleaned docs and write binary shards with a train/validation split.

Each document is encoded as `<|bos|> tokens <|eos|>`. Shards are flat numpy
arrays (uint16 if vocab < 65536 else uint32) named `{split}_{i:05d}.bin`, with
`meta.json` describing dtype, token counts and the tokenizer used. The split is
deterministic by document-id hash, so reruns are reproducible.

Usage:
    python -m data.clean.shard [--input data/cleaned/cleaned.jsonl] [--extra traces.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.config import load_config, resolve
from core.logging import get_logger
from tokenizer.tok import Tok

log = get_logger("training")


def is_val(doc_id: str, frac: float) -> bool:
    h = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


class ShardWriter:
    def __init__(self, out: Path, split: str, dtype, shard_tokens: int):
        self.out, self.split, self.dtype, self.cap = out, split, dtype, shard_tokens
        self.buf: list[np.ndarray] = []
        self.n_buf = 0
        self.idx = 0
        self.total = 0
        self.files: list[str] = []

    def add(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=self.dtype)
        self.buf.append(arr)
        self.n_buf += len(arr)
        self.total += len(arr)
        if self.n_buf >= self.cap:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"{self.split}_{self.idx:05d}.bin"
        np.concatenate(self.buf).tofile(self.out / name)
        self.files.append(name)
        self.idx += 1
        self.buf, self.n_buf = [], 0


def write_shards(inputs: list[Path], tok: Tok, out: Path, val_fraction: float, shard_tokens: int,
                 batch: int = 512) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    writers = {s: ShardWriter(out, s, dtype, shard_tokens) for s in ("train", "val")}
    docs = {"train": 0, "val": 0}

    def flush_batch(items: list[dict]) -> None:
        for d, ids in zip(items, tok.encode_batch([x["text"] for x in items])):
            split = "val" if is_val(d["id"], val_fraction) else "train"
            writers[split].add([tok.bos_id, *ids, tok.eos_id])
            docs[split] += 1

    for path in inputs:
        pending: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pending.append(json.loads(line))
                if len(pending) >= batch:
                    flush_batch(pending)
                    pending = []
        if pending:
            flush_batch(pending)
    for w in writers.values():
        w.flush()
    meta = {"dtype": np.dtype(dtype).name, "vocab_size": tok.vocab_size,
            "train_tokens": writers["train"].total, "val_tokens": writers["val"].total,
            "train_docs": docs["train"], "val_docs": docs["val"],
            "train_files": writers["train"].files, "val_files": writers["val"].files}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.event("clean.stage7.shard", "done", details=meta)
    return meta


def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=[str(resolve(cfg["paths"]["data_cleaned"]) / "cleaned.jsonl")])
    ap.add_argument("--out", default=str(resolve(cfg["paths"]["shards"])))
    ap.add_argument("--tokenizer", default=str(resolve(cfg["paths"]["tokenizer"])))
    a = ap.parse_args(argv)
    meta = write_shards([Path(p) for p in a.input], Tok.load(a.tokenizer), Path(a.out),
                        cfg["data"]["val_fraction"], cfg["data"]["shard_tokens"])
    print(json.dumps({k: v for k, v in meta.items() if not k.endswith("files")}))
    return meta


if __name__ == "__main__":
    main()
