"""Model hyper-parameters and presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 48000
    d_model: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_hidden: int = 5632
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    rope_scaling: float = 1.0      # linear position interpolation factor
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    grad_checkpointing: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def param_count(self) -> int:
        """Analytic parameter count (matches the nn.Module count)."""
        d, h = self.d_model, self.ffn_hidden
        kv = self.n_kv_heads * self.head_dim
        attn = d * d + 2 * d * kv + d * d
        mlp = 3 * d * h
        norms = 2 * d
        layer = attn + mlp + norms
        emb = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return self.n_layers * layer + emb + head + d


PRESETS: dict[str, dict] = {
    # ~1.0B parameters with a 48k vocabulary.
    "1b": dict(d_model=2048, n_layers=20, n_heads=16, n_kv_heads=4, ffn_hidden=5632, max_seq_len=8192),
    # ~125M, a mid-size config for single consumer GPUs.
    "small": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, ffn_hidden=2048, max_seq_len=8192),
    # A few million parameters: CPU smoke tests.
    "tiny": dict(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_hidden=352, max_seq_len=1024),
}


def preset(name: str, vocab_size: int, **overrides) -> ModelConfig:
    """Build a ModelConfig from a named preset."""
    kw = dict(PRESETS[name])
    kw.update(overrides)
    return ModelConfig(vocab_size=vocab_size, **kw)
