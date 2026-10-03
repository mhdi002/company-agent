"""Byte-level BPE tokenizer trained from scratch on our cleaned corpus.

Uses the HF `tokenizers` library only as a fast BPE trainer/runtime; the
vocabulary and merges are learned from our data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

AGENT_TOKENS = ["<|step|>", "<|thought|>", "<|tool|>", "<|args|>", "<|result|>", "<|todo|>", "<|end|>"]
DEFAULT_SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", *AGENT_TOKENS,
                    "<|conf|>", "<|user|>", "<|program|>", "<|output|>"]


def train_bpe(texts: Iterable[str], vocab_size: int, special_tokens: list[str] | None = None,
              min_frequency: int = 2) -> Tokenizer:
    specials = special_tokens or DEFAULT_SPECIALS
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=specials, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


class Tok:
    """Thin wrapper with convenient special-token ids."""

    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.special = {t: tokenizer.token_to_id(t) for t in DEFAULT_SPECIALS if tokenizer.token_to_id(t) is not None}
        self.pad_id = self.special.get("<|pad|>", 0)
        self.bos_id = self.special.get("<|bos|>")
        self.eos_id = self.special.get("<|eos|>")
        self.end_id = self.special.get("<|end|>")

    @classmethod
    def load(cls, directory: str | Path) -> "Tok":
        return cls(Tokenizer.from_file(str(Path(directory) / "tokenizer.json")))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(d / "tokenizer.json"))
        (d / "meta.json").write_text(json.dumps({"vocab_size": self.vocab_size, "special": self.special}, indent=2))

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)

    def count(self, text: str) -> int:
        return len(self.tk.encode(text).ids)


def iter_texts(path: Path, limit: int = 0) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            yield json.loads(line)["text"]
