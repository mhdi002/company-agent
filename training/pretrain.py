"""Pretrain the from-scratch transformer on tokenized shards.

Features: bf16 autocast, SDPA/FlashAttention, gradient checkpointing, gradient
accumulation, cosine LR with warmup, checkpoint + resume (model, optimizer,
step, RNG), loss/throughput logging, periodic sample generation, validation
perplexity, optional DDP via torchrun.

Tiny smoke test (CPU, minutes):
    python -m training.pretrain --preset tiny --seq-len 256 --micro-bs 8 --accum 1 \
        --max-steps 300 --warmup 30 --eval-every 100 --sample-every 150 --save-every 100 --out training/checkpoints/tiny

Full 1B run (1 GPU; for N GPUs: torchrun --nproc_per_node N -m training.pretrain ...):
    python -m training.pretrain --preset 1b --resume
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.config import load_config, resolve
from core.logging import configure, get_logger
from tokenizer.tok import Tok
from training.common import (autocast_ctx, build_model, cosine_lr, ddp_setup, load_checkpoint, make_optimizer,
                             pick_device, restore_rng, save_checkpoint, seed_all)

log = get_logger("training")


class ShardData:
    """Random windows of seq_len+1 tokens from memory-mapped shards."""

    def __init__(self, shard_dir: Path, split: str, seq_len: int, seed: int = 0):
        meta = json.loads((shard_dir / "meta.json").read_text())
        self.arrs = [np.memmap(shard_dir / f, dtype=meta["dtype"], mode="r") for f in meta[f"{split}_files"]]
        self.arrs = [a for a in self.arrs if len(a) > seq_len + 1]
        if not self.arrs:
            raise ValueError(f"no {split} shard longer than seq_len={seq_len}")
        self.sizes = np.array([len(a) for a in self.arrs], dtype=np.float64)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        which = self.rng.choice(len(self.arrs), size=bs, p=self.sizes / self.sizes.sum())
        xs = []
        for w in which:
            a = self.arrs[w]
            i = int(self.rng.integers(0, len(a) - self.seq_len - 1))
            xs.append(torch.from_numpy(a[i:i + self.seq_len + 1].astype(np.int64)))
        b = torch.stack(xs)
        x, y = b[:, :-1], b[:, 1:]
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data: ShardData, batches: int, bs: int, device, dtype: str) -> float:
    """Mean validation loss."""
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = data.batch(bs, device)
        with autocast_ctx(device, dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def sample_text(model, tok: Tok, device, prompt: str = "The company", n: int = 48) -> str:
    raw = model.module if hasattr(model, "module") else model
    ids = torch.tensor([[tok.bos_id, *tok.encode(prompt)]], device=device)
    out = raw.generate(ids, n, temperature=0.8, top_p=0.95, stop_ids={tok.eos_id})
    raw.train()
    return prompt + tok.decode(out, skip_special=True)


def parse(argv=None):
    cfg = load_config()
    t = cfg["training"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=cfg["model"]["preset"])
    ap.add_argument("--shards", default=str(resolve(cfg["paths"]["shards"])))
    ap.add_argument("--tokenizer", default=str(resolve(cfg["paths"]["tokenizer"])))
    ap.add_argument("--out", default=str(resolve(cfg["paths"]["checkpoints"]) / "pretrain"))
    ap.add_argument("--seq-len", type=int, default=t["seq_len"])
    ap.add_argument("--micro-bs", type=int, default=t["micro_batch_size"])
    ap.add_argument("--accum", type=int, default=t["grad_accum_steps"])
    ap.add_argument("--lr", type=float, default=t["lr"])
    ap.add_argument("--min-lr", type=float, default=t["min_lr"])
    ap.add_argument("--warmup", type=int, default=t["warmup_steps"])
    ap.add_argument("--max-steps", type=int, default=t["max_steps"])
    ap.add_argument("--eval-every", type=int, default=t["eval_every"])
    ap.add_argument("--eval-batches", type=int, default=t["eval_batches"])
    ap.add_argument("--sample-every", type=int, default=t["sample_every"])
    ap.add_argument("--save-every", type=int, default=t["save_every"])
    ap.add_argument("--log-every", type=int, default=t["log_every"])
    ap.add_argument("--device", default=t["device"])
    ap.add_argument("--dtype", default=t["dtype"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=t["seed"])
    return cfg, ap.parse_args(argv)


def main(argv=None) -> dict:
    cfg, a = parse(argv)
    rank, local, world = ddp_setup()
    master = rank == 0
    configure(cfg["paths"]["logs"])
    device = pick_device(a.device) if world == 1 or not torch.cuda.is_available() else torch.device("cuda", local)
    seed_all(a.seed + rank)
    tok = Tok.load(a.tokenizer)
    shards = Path(a.shards)
    seq_len = min(a.seq_len, 10**9)
    model = build_model(cfg, tok.vocab_size, a.preset, max_seq_len=max(seq_len, 16),
                        grad_checkpointing=not a.no_grad_ckpt and cfg["training"]["grad_checkpointing"]).to(device)
    opt = make_optimizer(model, a.lr, cfg["training"]["weight_decay"], device)
    out = Path(a.out)
    step = 0
    if a.resume and (out / "latest.pt").exists():
        ck = load_checkpoint(out / "latest.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        restore_rng(ck.get("rng"))
        if master:
            log.event("pretrain.resume", "info", details=f"resumed at step {step}")
    raw_model = model
    if a.compile:
        model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None)
    train = ShardData(shards, "train", seq_len, a.seed + rank + step)
    val = ShardData(shards, "val", seq_len, a.seed + 999)
    if master:
        log.event("pretrain.start", "info", details={"params": raw_model.num_params(), "preset": a.preset,
                                                    "device": str(device), "world": world, "seq_len": seq_len,
                                                    "tokens_per_step": a.micro_bs * a.accum * seq_len * world})
    history = []
    model.train()
    t0, tokens_since = time.time(), 0
    while step < a.max_steps:
        lr = cosine_lr(step, a.lr, a.min_lr, a.warmup, a.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        total = 0.0
        for micro in range(a.accum):
            x, y = train.batch(a.micro_bs, device)
            sync = world == 1 or micro == a.accum - 1
            ctx = model.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx, autocast_ctx(device, a.dtype):
                _, loss = model(x, y)
                (loss / a.accum).backward()
            total += loss.item() / a.accum
            tokens_since += x.numel() * world
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if master and (step % a.log_every == 0 or step == 1):
            dt = time.time() - t0
            rec = {"step": step, "loss": round(total, 4), "lr": lr, "grad_norm": round(float(gn), 3),
                   "tok_per_s": round(tokens_since / max(dt, 1e-9), 1)}
            history.append(rec)
            log.event("pretrain.step", "info", details=rec)
            print(json.dumps(rec), flush=True)
            t0, tokens_since = time.time(), 0
        if master and step % a.eval_every == 0:
            vl = evaluate(model, val, a.eval_batches, a.micro_bs, device, a.dtype)
            rec = {"step": step, "val_loss": round(vl, 4), "val_ppl": round(float(np.exp(vl)), 2)}
            history.append(rec)
            log.event("pretrain.eval", "info", details=rec)
            print(json.dumps(rec), flush=True)
        if master and step % a.sample_every == 0:
            s = sample_text(raw_model, tok, device)
            log.event("pretrain.sample", "info", details=s)
            print("sample:", s.replace("\n", " ")[:200], flush=True)
        if master and (step % a.save_every == 0 or step == a.max_steps):
            save_checkpoint(out / "latest.pt", raw_model, opt, step, {"tokenizer": a.tokenizer})
    if master:
        vl = evaluate(model, val, a.eval_batches, a.micro_bs, device, a.dtype)
        final = {"step": step, "final_val_loss": round(vl, 4), "final_val_ppl": round(float(np.exp(vl)), 2)}
        history.append(final)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        log.event("pretrain.done", "done", details=final)
        print(json.dumps(final))
    if world > 1:
        dist.destroy_process_group()
    return {"step": step, "history": history}


if __name__ == "__main__":
    main()
