import json

import numpy as np

from data.clean import decontam, langid, normalize, pii, quality
from data.clean.dedup import Deduplicator, MinHashLSH
from data.clean.pipeline import main as pipeline_main

CFG = {"min_chars": 200, "max_chars": 100000, "max_symbol_ratio": 0.1, "max_digit_ratio": 0.15,
       "max_dup_line_frac": 0.3, "max_top_ngram_frac": 0.2, "perplexity_max": 3000, "languages": ["en"],
       "minhash_perm": 128, "minhash_bands": 32, "near_dup_threshold": 0.8, "seed": 1}
GOOD = ("The company develops software for small businesses and public institutions. Our team has more "
        "than twenty years of experience in industrial engineering and in the design of reliable systems. "
        "Customers can track their orders online at any time of the day, and the support team answers "
        "questions within one working day.")


def test_normalize_html_and_mojibake():
    html = "<html><body><nav>Home | About | Contact</nav><p>CafÃ© rÃ©sumÃ© text</p><footer>All rights reserved</footer></body></html>"
    out = normalize.normalize(html)
    assert out == "Café résumé text"


def test_normalize_unicode_and_controls():
    assert normalize.normalize("ﬁne​  text\x07\n\n\n\nend") == "fine text\n\nend"


def test_langid():
    assert langid.detect(GOOD)[0] == "en"
    assert langid.detect("Das Unternehmen entwickelt Software für kleine Betriebe und die Verwaltung.")[0] == "de"
    assert langid.detect("L'entreprise développe des logiciels pour les petites entreprises de la région.")[0] == "fr"
    assert langid.stage({"text": "La empresa desarrolla programas para pequeñas empresas.", "id": "x", "source": "s"}, CFG)[0] is None


def test_quality_heuristics():
    assert quality.heuristics(GOOD, CFG) is None
    assert quality.heuristics("short", CFG) == "too_short"
    assert quality.heuristics("$$$ ### " * 50, CFG) == "symbol_ratio"
    assert quality.heuristics("click here to win\n" * 30, CFG) == "dup_lines"
    assert quality.heuristics(" ".join(str(i) for i in range(400)), CFG) == "digit_ratio"


def test_kneser_ney_perplexity_orders_text():
    lm = quality.KneserNeyLM().fit([GOOD] * 5 + ["The river flows through the old town before it reaches the sea."] * 5)
    assert lm.perplexity(GOOD) < lm.perplexity("zebra quantum marmalade oscillates purple sideways forever")


def test_minhash_near_dup():
    lsh = MinHashLSH(threshold=0.7)
    a = GOOD
    b = GOOD.replace("reliable", "robust")
    assert lsh.add_or_match("a", a) is None
    assert lsh.add_or_match("b", b) == "a"
    assert lsh.add_or_match("c", "Completely different text about rivers, mountains and the weather in spring.") is None


def test_dedup_exact_and_cross_dataset():
    d = Deduplicator(CFG)
    assert d.stage({"id": "ds1:1", "source": "ds1", "text": GOOD}, CFG)[1] == "kept"
    assert d.stage({"id": "ds2:1", "source": "ds2", "text": GOOD + "  "}, CFG)[1] == "exact_dup_cross"


def test_pii_scrub():
    text, counts = pii.scrub("Mail jane.roe@corp.example, call +44 20 7946 0958, SSN 078-05-1120, "
                             "card 4111 1111 1111 1111, ip 192.168.1.20, lives at 221 Baker Street. Founded in 2009.")
    assert "<EMAIL>" in text and "<PHONE>" in text and "<ID>" in text and "<CARD>" in text
    assert "<IP>" in text and "<ADDRESS>" in text and "2009" in text
    assert counts["EMAIL"] == 1


def test_toxicity():
    assert pii.stage({"id": "1", "source": "s", "text": "You stupid idiot moron, I hate you."}, CFG)[0] is None
    assert pii.stage({"id": "2", "source": "s", "text": GOOD}, CFG)[1] == "kept"


def test_decontam():
    bench = "A man is sitting on a roof. He is using wrap to wrap a pair of skis."
    long_bench = " ".join(f"w{i}" for i in range(30))
    dc = decontam.Decontaminator([bench, long_bench], n=13)
    assert dc.contaminated("intro " + bench + " outro")
    assert dc.contaminated("x " + " ".join(f"w{i}" for i in range(5, 20)) + " y")
    assert not dc.contaminated(GOOD)


def test_pipeline_end_to_end(tmp_path):
    from data.sample.build_sample import build_corpus
    raw = tmp_path / "docs.jsonl"
    with open(raw, "w") as f:
        for d in build_corpus(n_general=300):
            f.write(json.dumps(d) + "\n")
    stats = pipeline_main(["--inputs", str(raw), "--offline-benchmarks", "--out-dir", str(tmp_path / "clean"),
                           "--reports-dir", str(tmp_path / "rep")])
    assert [s["name"] for s in stats] == ["normalize", "langid", "quality", "dedup", "pii_toxicity", "decontam"]
    for s in stats:
        assert s["docs_out"] <= s["docs_in"]
        samples = (tmp_path / "rep" / f"stage{s['stage']}_{s['name']}_samples.jsonl").read_text().splitlines()
        assert 0 < len(samples) <= 200
    reasons = {k for s in stats for k in s["reasons"]}
    assert {"exact_dup", "toxic", "benchmark_overlap", "dup_lines"} <= reasons
    out = (tmp_path / "clean" / "cleaned.jsonl").read_text()
    assert "@mail.example" not in out and "<html" not in out and "Ã" not in out
    assert (tmp_path / "rep" / "cleaning_report.md").exists()


def test_shards(tmp_path):
    from data.clean.shard import write_shards
    from tokenizer.tok import Tok, train_bpe
    texts = [GOOD, "The river flows through the old town before it reaches the sea."] * 20
    tok = Tok(train_bpe(texts, 400, min_frequency=1))
    inp = tmp_path / "c.jsonl"
    inp.write_text("".join(json.dumps({"id": f"d{i}", "text": t}) + "\n" for i, t in enumerate(texts)))
    meta = write_shards([inp], tok, tmp_path / "sh", val_fraction=0.2, shard_tokens=500)
    assert meta["val_docs"] > 0 and meta["train_docs"] > 0
    arr = np.fromfile(tmp_path / "sh" / meta["train_files"][0], dtype=np.uint16)
    assert arr[0] == tok.bos_id and (arr == tok.eos_id).sum() >= 1
    # deterministic split
    meta2 = write_shards([inp], tok, tmp_path / "sh2", val_fraction=0.2, shard_tokens=500)
    assert meta2["val_docs"] == meta["val_docs"]
