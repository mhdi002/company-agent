"""Shared training utilities: device/dtype, model building, LR schedule, checkpoints, DDP."""
from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from model.config import ModelConfig, preset
from model.model import Transformer


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast_ctx(device: torch.device, dtype: str):
    if device.type == "cuda" and dtype in ("bf16", "fp16"):
        return torch.autocast("cuda", dtype=torch.bfloat16 if dtype == "bf16" else torch.float16)
    return nullcontext()  # CPU runs in fp32 (bf16 autocast on CPU is slow and only for testing)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_lr(step: int, base: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))


def build_model(cfg: dict, vocab_size: int, preset_name: str | None = None, **overrides) -> Transformer:
    mc = cfg["model"]
    p = preset_name or mc["preset"]
    kw = dict(rope_theta=mc.get("rope_theta", 500000.0), rope_scaling=mc.get("rope_scaling", 1.0))
    kw.update(overrides)
    return Transformer(preset(p, vocab_size, **kw))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: torch.device):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **extra)


def save_checkpoint(path: Path, model: torch.nn.Module, opt, step: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    state = {"model": raw.state_dict(), "optimizer": opt.state_dict() if opt else None, "step": step,
             "model_config": raw.cfg.to_dict(), "rng": {"torch": torch.get_rng_state(),
                                                        "numpy": np.random.get_state(), "python": random.getstate()},
             **(extra or {})}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: Path, device: torch.device | str = "cpu") -> Transformer:
    ck = load_checkpoint(Path(path), map_location=device)
    m = Transformer(ModelConfig.from_dict(ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.to(device)


def restore_rng(state: dict) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def ddp_setup() -> tuple[int, int, int]:
    """Initialise torch.distributed when launched with torchrun. Returns (rank, local_rank, world)."""
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) == 1:
        return 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, local, world = dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, local, world
