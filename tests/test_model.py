import json

import torch

from model.config import ModelConfig, preset
from model.model import KVCache, Transformer, apply_rope, rope_tables


def tiny(**kw):
    cfg = ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                      max_seq_len=64, **kw)
    torch.manual_seed(0)
    return Transformer(cfg)


def test_param_count_matches_analytic():
    for name in ("tiny", "small"):
        cfg = preset(name, 1000)
        assert Transformer(cfg).num_params() == cfg.param_count()


def test_1b_param_count_on_meta():
    cfg = preset("1b", 48000)
    with torch.device("meta"):
        m = Transformer(cfg)
    n = m.num_params()
    assert n == cfg.param_count()
    assert 0.95e9 <= n <= 1.05e9


def test_tied_embeddings():
    m = tiny()
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()


def test_forward_shapes_and_loss():
    m = tiny()
    x = torch.randint(0, 97, (2, 16))
    logits, loss = m(x, torch.randint(0, 97, (2, 16)))
    assert logits.shape == (2, 16, 97)
    assert torch.isfinite(loss) and abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 1.0


def test_causal_masking():
    """Changing a future token must not change earlier logits."""
    m = tiny().eval()
    x = torch.randint(0, 97, (1, 20))
    y = x.clone()
    y[0, 15] = (y[0, 15] + 1) % 97
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-5)
    assert not torch.allclose(a[0, 15:], b[0, 15:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    m = tiny().eval()
    x = torch.randint(0, 97, (1, 24))
    full, _ = m(x)
    cache = KVCache()
    out = []
    lg, _ = m(x[:, :10], cache=cache)          # prefill
    out.append(lg[:, -1])
    for t in range(10, 24):
        lg, _ = m(x[:, t:t + 1], cache=cache)  # incremental
        out.append(lg[:, -1])
    inc = torch.stack(out[:-1], dim=1)
    assert torch.allclose(full[:, 9:23], inc, atol=1e-4)
    assert cache.length == 24


def test_kv_cache_chunked_prefill():
    m = tiny().eval()
    x = torch.randint(0, 97, (1, 20))
    full, _ = m(x)
    cache = KVCache()
    m(x[:, :8], cache=cache)
    lg, _ = m(x[:, 8:20], cache=cache)   # multi-token step with past (explicit mask path)
    assert torch.allclose(full[:, -1], lg[:, -1], atol=1e-4)


def test_grad_checkpointing_same_grads():
    x = torch.randint(0, 97, (2, 12))
    m1, m2 = tiny(), tiny(grad_checkpointing=True)
    m2.load_state_dict(m1.state_dict())
    m1.train(); m2.train()
    m1(x, x)[1].backward()
    m2(x, x)[1].backward()
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), n


def test_rope_rotation_preserves_norm_and_scaling():
    cos, sin = rope_tables(16, 32, 10000.0)
    x = torch.randn(1, 2, 32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    c2, _ = rope_tables(16, 64, 10000.0, scaling=2.0)
    assert torch.allclose(c2[10], cos[5], atol=1e-6)   # position interpolation


def test_loss_mask_ignores_positions():
    m = tiny()
    x = torch.randint(0, 97, (1, 10))
    mask = torch.zeros(1, 10, dtype=torch.long)
    mask[0, 5:] = 1
    _, l1 = m(x, x, loss_mask=mask)
    t = x.clone()
    t[0, :5] = 3
    _, l2 = m(x, t, loss_mask=mask)
    assert torch.allclose(l1, l2)


def test_generate_with_processor():
    m = tiny()
    banned = set(range(50, 97))

    def proc(ids, logits):
        logits[list(banned)] = float("-inf")
        return logits
    out = m.generate(torch.tensor([[1, 2, 3]]), 15, temperature=1.0, top_p=0.9, logits_processor=proc)
    assert len(out) == 15 and not set(out) & banned


def test_pretrain_smoke_and_resume(tmp_path):
    from data.clean.shard import write_shards
    from tokenizer.tok import Tok, train_bpe
    from training.pretrain import main
    texts = ["The river flows through the old town before it reaches the sea. " * 8,
             "Solar parks connect batteries to the grid to store energy for the evening. " * 8] * 30
    tok = Tok(train_bpe(texts, 300, min_frequency=1))
    tok.save(tmp_path / "tok")
    inp = tmp_path / "c.jsonl"
    inp.write_text("".join(json.dumps({"id": f"d{i}", "text": t}) + "\n" for i, t in enumerate(texts)))
    write_shards([inp], tok, tmp_path / "sh", val_fraction=0.2, shard_tokens=10**6)
    args = ["--preset", "tiny", "--shards", str(tmp_path / "sh"), "--tokenizer", str(tmp_path / "tok"),
            "--out", str(tmp_path / "ck"), "--seq-len", "64", "--micro-bs", "4", "--accum", "2",
            "--warmup", "2", "--eval-every", "10", "--eval-batches", "2", "--sample-every", "10",
            "--save-every", "5", "--log-every", "5", "--lr", "3e-3"]
    r1 = main(args + ["--max-steps", "10"])
    assert r1["step"] == 10
    first = [h["loss"] for h in r1["history"] if "loss" in h]
    assert first[-1] < first[0]
    r2 = main(args + ["--max-steps", "15", "--resume"])
    assert r2["step"] == 15
    assert (tmp_path / "ck" / "latest.pt").exists()
