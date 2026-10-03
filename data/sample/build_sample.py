"""Deterministically build the offline fixture corpus used for smoke tests.

The corpus deliberately contains the defects each cleaning stage must remove:
HTML/boilerplate, mojibake, non-English text, low-quality spam, exact and near
duplicates, PII, toxic text, and benchmark contamination. Text is original and
written for this repository.
"""
from __future__ import annotations

import random

from data.sample.companies import COMPANIES, FIELD_KB
from data.sample.langid_seed import SEED

GENERAL = [
    "Rivers shape the landscape by carrying sediment from mountains to the coast.",
    "The printing press made books cheaper and helped spread literacy across Europe.",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "A balanced budget means that planned spending does not exceed expected income.",
    "Cities grew rapidly during the industrial revolution as people moved to work in factories.",
    "Vaccines train the immune system to recognise a pathogen without causing disease.",
    "The water cycle describes how water evaporates, condenses and falls as precipitation.",
    "Good project management starts with clear objectives and a realistic schedule.",
    "Electric motors convert electrical energy into mechanical motion using magnetic fields.",
    "Supply chains connect raw material producers, manufacturers, distributors and retailers.",
    "Soil health depends on organic matter, microbial life and careful use of fertiliser.",
    "Software testing finds defects early, when they are cheaper to fix.",
    "Renewable energy sources include solar, wind, hydro and geothermal power.",
    "Small businesses often lack the staff to analyse their own operational data.",
    "Public tenders describe the work a government body wants to buy and how bids are judged.",
    "A grant proposal explains the problem, the planned activities and the expected impact.",
    "Market analysis estimates the size of a market and the needs of its customers.",
    "Risk management identifies what could go wrong and how to reduce the likelihood or impact.",
    "Logistics companies plan routes to reduce fuel use and delivery times.",
    "Manufacturers measure overall equipment effectiveness to find lost production time.",
    "Hospitals use electronic health records to share patient information between departments.",
    "Precision agriculture uses sensors and maps to apply water and nutrients where needed.",
    "Retailers forecast demand so that shelves are stocked without creating waste.",
    "Construction projects coordinate many trades, materials and inspections over months or years.",
    "Universities and colleges offer vocational courses that prepare students for specific jobs.",
    "Credit risk models estimate the probability that a borrower will not repay a loan.",
    "Machine learning systems learn patterns from examples rather than explicit rules.",
    "The Romans built roads and aqueducts that connected distant parts of their empire.",
    "Mountains influence local weather by forcing air to rise and cool.",
    "Libraries preserve knowledge and give the public free access to books and archives.",
    "A milestone marks the completion of an important stage in a project plan.",
    "Deliverables are the concrete outputs a project hands over to its client.",
    "Interviews with users reveal problems that are not visible in usage statistics.",
    "Energy efficiency reduces costs and emissions at the same time.",
    "Open data allows researchers and companies to build new services on public information.",
    "Digital twins mirror a physical asset so that engineers can test changes safely.",
    "Teams that document decisions find it easier to onboard new members.",
    "Customer support tickets are a valuable source of product improvement ideas.",
    "Ports handle containers with cranes, straddle carriers and automated guided vehicles.",
    "Batteries store electricity chemically and release it when demand is high.",
]

TOXIC = ["You are a stupid idiot and I hate you, moron.", "This idiot is a worthless moron, stupid trash."]
BENCH = ["A man is sitting on a roof. He is using wrap to wrap a pair of skis.",
         "Which of the following statements best explains why magnets usually stick to a refrigerator door?"]


def _doc(i: int, text: str, source: str = "sample") -> dict:
    return {"id": f"{source}:{i}", "text": text, "source": source, "meta": {"group": "sample", "license": "repo"}}


def company_text(c: dict) -> str:
    return (f"{c['name']} is a {c['field']} company based in {c['city']}, {c['country']}, founded in {c['founded']}. "
            f"{c['about']} The company offers {', '.join(c['services'])}. "
            f"Its products include {', '.join(c['products'])}. Customers include {c['clients']}.")


def build_corpus(n_general: int = 2500, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    docs: list[dict] = []
    kb_sents = [s.strip() + "." for v in FIELD_KB.values() for a in v for s in a["text"].split(".") if s.strip()]
    pool = GENERAL + kb_sents
    for i in range(n_general):
        k = rng.randint(5, 12)
        paras = [" ".join(rng.sample(pool, min(k, len(pool)))) for _ in range(rng.randint(1, 3))]
        docs.append(_doc(len(docs), "\n\n".join(paras)))
    for c in COMPANIES:
        for _ in range(20):
            extra = " ".join(rng.sample(pool, 4))
            docs.append(_doc(len(docs), company_text(c) + " " + extra, "sample-business"))
    # --- defects ---
    for i in range(60):   # HTML + boilerplate
        body = " ".join(rng.sample(GENERAL, 6))
        docs.append(_doc(len(docs), f"<html><head><script>var a=1;</script></head><body><nav>Home | About | Contact</nav>"
                                    f"<p>{body}</p><footer>Cookie policy. All rights reserved.</footer></body></html>"))
    for i in range(40):   # mojibake
        docs.append(_doc(len(docs), " ".join(rng.sample(GENERAL, 6)).replace("e", "Ã©", 3) + " CafÃ© rÃ©sumÃ©."))
    for lang, sents in SEED.items():
        if lang == "en":
            continue
        for _ in range(30):
            docs.append(_doc(len(docs), " ".join(rng.sample(sents, min(8, len(sents))))))
    for i in range(40):   # symbol / digit spam, repetition
        docs.append(_doc(len(docs), "$$$ ### !!! >>> " * 40 + " buy now"))
        docs.append(_doc(len(docs), " ".join(str(rng.randint(0, 99999)) for _ in range(120))))
        docs.append(_doc(len(docs), "click here to win a prize\n" * 30))
    base = [d["text"] for d in docs[:200]]
    for t in base[:100]:  # exact duplicates
        docs.append(_doc(len(docs), t))
    for t in base[100:200]:   # near duplicates (one word changed)
        words = t.split()
        words[len(words) // 2] = "notably"
        docs.append(_doc(len(docs), " ".join(words)))
    for i in range(40):   # PII
        docs.append(_doc(len(docs), " ".join(rng.sample(GENERAL, 6)) +
                         f" Contact John at john.doe{i}@mail.example or call +44 20 7946 {1000 + i:04d}. "
                         f"His SSN is 078-05-{1120 + i:04d}."))
    for i in range(30):   # toxic
        docs.append(_doc(len(docs), " ".join(rng.sample(GENERAL, 3)) + " " + " ".join(TOXIC * 2)))
    for i in range(10):   # benchmark contamination
        docs.append(_doc(len(docs), " ".join(rng.sample(GENERAL, 4)) + " " + BENCH[i % 2] + " " + " ".join(rng.sample(GENERAL, 3))))
    rng.shuffle(docs)
    return docs


def benchmark_texts() -> list[str]:
    """Offline stand-in for benchmark eval sets used by decontamination tests."""
    return BENCH
