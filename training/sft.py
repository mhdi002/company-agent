"""Supervised agent-format fine-tuning of OUR pretrained checkpoint on synthetic SRLM traces.

Loss is computed only on the target step tokens (thought, program, confidence block, <|end|>).

Usage:
    python -m training.sft --init training/checkpoints/pretrain/latest.pt --data data/sft
    python -m training.sft --init training/checkpoints/tiny/latest.pt --data data/sft --max-steps 300 \
        --seq-len 1024 --micro-bs 8 --out training/checkpoints/sft-tiny
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from core.config import load_config, resolve
from core.logging import configure, get_logger
from model.config import ModelConfig
from model.model import Transformer
from tokenizer.tok import Tok
from training.common import (autocast_ctx, cosine_lr, load_checkpoint, make_optimizer, pick_device, save_checkpoint,
                             seed_all)

log = get_logger("training")


def encode_example(tok: Tok, ex: dict, seq_len: int, head_keep: int = 256) -> tuple[list[int], list[int]]:
    p = tok.encode(ex["prompt"])
    t = tok.encode(ex["target"])
    room = seq_len - len(t) - 2
    if room <= 0:
        t = t[: seq_len - 2 - min(64, len(p))]
        room = seq_len - len(t) - 2
    if len(p) > room:
        keep = min(head_keep, room // 2)
        p = p[:keep] + p[-(room - keep):]
    ids = [tok.bos_id] + p + t + [tok.eos_id]
    mask = [0] * (1 + len(p)) + [1] * (len(t) + 1)
    return ids, mask


class SFTData:
    def __init__(self, path: Path, tok: Tok, seq_len: int, seed: int = 0):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.items.append(encode_example(tok, ex, seq_len))
        self.pad = tok.pad_id
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def collate(self, batch: list[tuple[list[int], list[int]]], device) -> tuple[torch.Tensor, ...]:
        L = max(len(ids) for ids, _ in batch) - 1
        x = torch.full((len(batch), L), self.pad, dtype=torch.long)
        y = torch.full((len(batch), L), self.pad, dtype=torch.long)
        m = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1])
            y[i, :n] = torch.tensor(ids[1:])
            m[i, :n] = torch.tensor(mask[1:])
        return x.to(device), y.to(device), m.to(device)

    def batch(self, bs: int, device):
        return self.collate(self.rng.sample(self.items, min(bs, len(self.items))), device)

    def iter_batches(self, bs: int, device):
        for i in range(0, len(self.items), bs):
            yield self.collate(self.items[i:i + bs], device)


@torch.no_grad()
def eval_loss(model, data: SFTData, bs: int, device, dtype: str, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y, m) in enumerate(data.iter_batches(bs, device)):
        if i >= max_batches:
            break
        with autocast_ctx(device, dtype):
            _, loss = model(x, y, loss_mask=m)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main(argv=None) -> dict:
    cfg = load_config()
    s = cfg["training"]["sft"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=str(resolve(cfg["paths"]["checkpoints"]) / "pretrain" / "latest.pt"))
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--tokenizer", default=str(resolve(cfg["paths"]["tokenizer"])))
    ap.add_argument("--out", default=str(resolve(cfg["paths"]["checkpoints"]) / "sft"))
    ap.add_argument("--seq-len", type=int, default=s["seq_len"])
    ap.add_argument("--micro-bs", type=int, default=cfg["training"]["micro_batch_size"])
    ap.add_argument("--accum", type=int, default=cfg["training"]["grad_accum_steps"])
    ap.add_argument("--lr", type=float, default=s["lr"])
    ap.add_argument("--warmup", type=int, default=s["warmup_steps"])
    ap.add_argument("--max-steps", type=int, default=s["max_steps"])
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default=cfg["training"]["device"])
    ap.add_argument("--dtype", default=cfg["training"]["dtype"])
    ap.add_argument("--seed", type=int, default=cfg["training"]["seed"])
    a = ap.parse_args(argv)
    configure(cfg["paths"]["logs"])
    seed_all(a.seed)
    device = pick_device(a.device)
    tok = Tok.load(a.tokenizer)
    ck = load_checkpoint(Path(a.init), map_location=device)
    mcfg = ModelConfig.from_dict(ck["model_config"])
    if mcfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"tokenizer vocab {tok.vocab_size} != checkpoint vocab {mcfg.vocab_size}")
    if a.seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = a.seq_len   # RoPE tables are recomputed; scale with model.rope_scaling if needed
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    data_dir = resolve(a.data)
    train = SFTData(data_dir / "train.jsonl", tok, a.seq_len, a.seed)
    val = SFTData(data_dir / "val.jsonl", tok, a.seq_len, a.seed + 1)
    opt = make_optimizer(model, a.lr, cfg["training"]["weight_decay"], device)
    out = Path(a.out)
    log.event("sft.start", "info", details={"init": a.init, "train": len(train), "val": len(val),
                                           "params": model.num_params()})
    history = []
    t0 = time.time()
    model.train()
    for step in range(1, a.max_steps + 1):
        lr = cosine_lr(step - 1, a.lr, a.lr * 0.1, a.warmup, a.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for _ in range(a.accum):
            x, y, m = train.batch(a.micro_bs, device)
            with autocast_ctx(device, a.dtype):
                _, loss = model(x, y, loss_mask=m)
            (loss / a.accum).backward()
            tot += loss.item() / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": round(tot, 4), "lr": lr, "s": round(time.time() - t0, 1)}
            history.append(rec)
            log.event("sft.step", "info", details=rec)
            print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.max_steps:
            vl = eval_loss(model, val, a.micro_bs, device, a.dtype)
            history.append({"step": step, "val_loss": round(vl, 4)})
            log.event("sft.eval", "info", details=history[-1])
            print(json.dumps(history[-1]), flush=True)
            save_checkpoint(out / "latest.pt", model, opt, step, {"tokenizer": a.tokenizer, "init": a.init})
    (out / "history.json").write_text(json.dumps(history, indent=1))
    return {"history": history}


if __name__ == "__main__":
    main()
