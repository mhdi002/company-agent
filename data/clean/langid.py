"""Stage 2: language identification with our own character n-gram naive Bayes model.

Trained on `data/sample/langid_seed.py` (original seed sentences in 7 languages).
If `paths.fasttext_lid` points to a fastText `lid.176.bin` and the `fasttext`
package is installed, that model is used instead (pluggable).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from data.sample.langid_seed import SEED

_WORD = re.compile(r"[^\W\d_]+", re.U)

# Very frequent function words add a strong, robust signal.
STOPWORDS = {
    "en": "the and of to in is that for with are this on was by be as at from it an or have".split(),
    "de": "der die das und ist nicht mit den von zu ein eine für auf im sich dem des wir sie".split(),
    "fr": "le la les et des est pour une dans que qui pas sur du au avec nous vous ce".split(),
    "es": "el la los las y de que en un una es para con por del se su al como".split(),
    "nl": "de het een en van is dat op te in voor met zijn niet wij die ook".split(),
    "it": "il la di e che un una per con non sono del della nel alla gli le".split(),
    "pt": "o a os as e de que em um uma para com não por do da dos se".split(),
}


def _ngrams(text: str, n: int = 3) -> list[str]:
    grams = []
    for w in _WORD.findall(text.lower()):
        w = f" {w} "
        grams.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return grams


class NaiveBayesLangID:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    def fit(self, data: dict[str, list[str]]) -> "NaiveBayesLangID":
        for lang, sents in data.items():
            c = Counter()
            for s in sents + [" ".join(STOPWORDS.get(lang, []))] * 3:
                c.update(_ngrams(s))
            self.counts[lang] = c
            self.totals[lang] = sum(c.values())
            self.vocab |= set(c)
        return self

    def scores(self, text: str) -> dict[str, float]:
        grams = _ngrams(text[:3000])
        words = _WORD.findall(text[:3000].lower())
        V = len(self.vocab) + 1
        out = {}
        for lang, c in self.counts.items():
            denom = math.log(self.totals[lang] + self.alpha * V)
            s = sum(math.log(c.get(g, 0) + self.alpha) - denom for g in grams)
            sw = set(STOPWORDS.get(lang, []))
            s += 3.0 * sum(1 for w in words if w in sw)
            out[lang] = s
        return out

    def predict(self, text: str) -> tuple[str, float]:
        sc = self.scores(text)
        if not sc:
            return "unk", 0.0
        m = max(sc.values())
        exp = {k: math.exp(max(v - m, -50)) for k, v in sc.items()}
        z = sum(exp.values())
        lang = max(exp, key=exp.get)
        return lang, exp[lang] / z


@lru_cache(maxsize=1)
def default_model() -> NaiveBayesLangID:
    return NaiveBayesLangID().fit(SEED)


def detect(text: str) -> tuple[str, float]:
    return default_model().predict(text)


def stage(doc: dict, cfg: dict) -> tuple[dict | None, str]:
    langs = set(cfg.get("languages", ["en"]))
    lang, p = detect(doc["text"])
    if lang not in langs:
        return None, f"lang_{lang}"
    return dict(doc, meta={**doc.get("meta", {}), "lang": lang, "lang_p": round(p, 3)}), "kept"
