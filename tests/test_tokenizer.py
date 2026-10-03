from tokenizer.tok import AGENT_TOKENS, DEFAULT_SPECIALS, Tok, train_bpe

TEXTS = ["The company develops software for small businesses.", "Solar parks store energy in batteries."] * 50


def test_special_tokens_are_atomic(tmp_path):
    tok = Tok(train_bpe(TEXTS, 500, min_frequency=1))
    for t in AGENT_TOKENS:
        ids = tok.encode(t)
        assert len(ids) == 1 and ids[0] == tok.special[t]
    assert set(DEFAULT_SPECIALS) <= set(tok.special)
    tok.save(tmp_path)
    tok2 = Tok.load(tmp_path)
    s = "<|step|><|thought|> read the about page <|end|>"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.decode(tok2.encode(s)) == s


def test_roundtrip_unicode():
    tok = Tok(train_bpe(TEXTS, 400, min_frequency=1))
    s = "Café résumé — 東京 ✓"
    assert tok.decode(tok.encode(s)) == s
