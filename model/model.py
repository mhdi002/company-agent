"""Decoder-only transformer, written from scratch.

RoPE (with linear position-interpolation scaling), RMSNorm, SwiGLU MLP,
grouped-query attention, tied input/output embeddings, KV cache, optional
gradient checkpointing. Attention uses torch's scaled_dot_product_attention,
which dispatches to FlashAttention kernels when available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float, scaling: float = 1.0,
                device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim/2). scaling>1 interpolates positions."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float() / scaling
    freqs = torch.outer(pos, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D/2). Rotates interleaved-as-halves pairs."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@dataclass
class KVCache:
    """Per-layer key/value cache. Tensors are (B, n_kv, T, D)."""
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return 0 if not self.keys or self.keys[0] is None else self.keys[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.keys) <= layer:
            self.keys.append(None)
            self.values.append(None)
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=2)
        return self.keys[layer], self.values[layer]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg, self.layer_idx = cfg, layer_idx
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: KVCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            past = cache.keys[self.layer_idx].shape[2] if len(cache.keys) > self.layer_idx and cache.keys[self.layer_idx] is not None else 0
            k, v = cache.update(self.layer_idx, k, v)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.cfg.dropout if self.training else 0.0
        if past == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, dropout_p=drop)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:
            S = past + T
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Scaled init for residual output projections (GPT-2 style).
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None, loss_mask: torch.Tensor | None = None):
        """Return (logits, loss). targets use -100 or loss_mask==0 to ignore positions."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.embed(idx)
        for blk in self.blocks:
            if self.cfg.grad_checkpointing and self.training and cache is None:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, cache)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:]) if cache is not None else self.lm_head(x), None
        logits = self.lm_head(x)
        if loss_mask is not None:
            targets = targets.masked_fill(loss_mask == 0, -100)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0, stop_ids: set[int] | None = None,
                 logits_processor: Callable[[list[int], torch.Tensor], torch.Tensor] | None = None,
                 generator: torch.Generator | None = None) -> list[int]:
        """Sample tokens for a single sequence (B=1) with KV cache.

        logits_processor(generated_ids, logits) may mask logits (constrained decoding).
        """
        self.eval()
        cache = KVCache()
        ctx = idx[:, -(self.cfg.max_seq_len - max_new_tokens):] if idx.shape[1] + max_new_tokens > self.cfg.max_seq_len else idx
        logits, _ = self.forward(ctx, cache=cache)
        out: list[int] = []
        for _ in range(max_new_tokens):
            lg = logits[0, -1].float()
            if logits_processor is not None:
                lg = logits_processor(out, lg)
            if temperature <= 0:
                nxt = int(torch.argmax(lg))
            else:
                probs = F.softmax(lg / temperature, dim=-1)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = sp.cumsum(0)
                    keep = cum - sp < top_p
                    sp = sp * keep
                    nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=generator)])
                else:
                    nxt = int(torch.multinomial(probs, 1, generator=generator))
            out.append(nxt)
            if stop_ids and nxt in stop_ids:
                break
            if cache.length >= self.cfg.max_seq_len:
                break
            logits, _ = self.forward(torch.tensor([[nxt]], device=idx.device), cache=cache)
        return out
