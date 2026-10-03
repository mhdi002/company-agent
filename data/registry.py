"""Registry of public datasets: source, license, and how to turn records into text.

Every dataset here is public and legally usable for model training under the
listed license. `data/download.py` reads this registry, downloads with resume
and checksum verification, converts to JSONL (`{"id","text","source","meta"}`)
and writes `data/MANIFEST.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dataset:
    name: str
    group: str                  # general | business | agentic
    kind: str                   # hf | url
    license: str
    source: str                 # human-readable URL
    repo: str = ""              # HF dataset repo id (kind=hf)
    patterns: list[str] = field(default_factory=list)  # file globs inside the HF repo
    urls: list[str] = field(default_factory=list)      # direct URLs (kind=url)
    text_fields: list[str] = field(default_factory=lambda: ["text"])
    fmt: str = "parquet"        # parquet | jsonl | csv | zip-xml | zip-csv
    max_files: int = 0          # 0 = all matched files
    notes: str = ""
    quality_reference: bool = False   # used to train the perplexity reference model


REGISTRY: list[Dataset] = [
    # ---------------- General language ----------------
    Dataset("fineweb-edu", "general", "hf", "ODC-By 1.0",
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            repo="HuggingFaceFW/fineweb-edu", patterns=["sample/10BT/*.parquet"], max_files=14,
            notes="Educational web text filtered by an edu classifier; 10BT sample.",
            quality_reference=True),
    Dataset("wikipedia-en", "general", "hf", "CC BY-SA 3.0 / GFDL",
            "https://huggingface.co/datasets/wikimedia/wikipedia",
            repo="wikimedia/wikipedia", patterns=["20231101.en/*.parquet"],
            quality_reference=True),
    Dataset("pg19", "general", "hf", "Apache-2.0 (texts are public domain)",
            "https://huggingface.co/datasets/deepmind/pg19",
            repo="deepmind/pg19", patterns=["data/train-*.parquet"], max_files=10,
            notes="Project Gutenberg books published before 1919."),
    # ---------------- Business / domain ----------------
    Dataset("edgar-corpus", "business", "hf", "Apache-2.0 (SEC filings are public domain)",
            "https://huggingface.co/datasets/eloukas/edgar-corpus",
            repo="eloukas/edgar-corpus", patterns=["*/2018*", "*/2019*", "*/2020*"], fmt="jsonl",
            text_fields=["section_1", "section_7"],
            notes="10-K 'Item 1. Business' and MD&A sections: company self-descriptions."),
    Dataset("cordis-h2020", "business", "url", "CC BY 4.0 (EU Open Data Portal)",
            "https://data.europa.eu/data/datasets/cordish2020projects",
            urls=["https://cordis.europa.eu/data/cordis-h2020projects-csv.zip",
                  "https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip"],
            fmt="zip-csv", text_fields=["title", "objective"],
            notes="EU-funded project objectives: proposal-style text."),
    Dataset("nsf-awards", "business", "url", "Public domain (US federal government work)",
            "https://www.nsf.gov/awardsearch/download.jsp",
            urls=[f"https://www.nsf.gov/awardsearch/download?DownloadFileName={y}&All=true" for y in range(2015, 2025)],
            fmt="zip-xml", text_fields=["AwardTitle", "AbstractNarration"],
            notes="Funded grant abstracts (project proposals)."),
    Dataset("nih-exporter", "business", "url", "Public domain (US federal government work)",
            "https://reporter.nih.gov/exporter",
            urls=[f"https://reporter.nih.gov/exporter/abstracts/download/{y}" for y in range(2018, 2024)],
            fmt="zip-csv", text_fields=["ABSTRACT_TEXT"],
            notes="NIH project abstracts."),
    Dataset("ted-tenders", "business", "url", "EU reuse policy (Commission Decision 2011/833/EU)",
            "https://data.europa.eu/data/datasets/ted-csv",
            urls=["https://data.europa.eu/api/hub/store/data/ted-contract-award-notices-2022.zip",
                  "https://data.europa.eu/api/hub/store/data/ted-contract-notices-2022.zip"],
            fmt="zip-csv", text_fields=["TITLE", "SHORT_DESCR"],
            notes="Public tender / RFP notices (EU TED)."),
    Dataset("arxiv-abstracts", "business", "hf", "CC0 1.0 (arXiv metadata)",
            "https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021",
            repo="gfissore/arxiv-abstracts-2021", patterns=["*.jsonl*", "*.parquet", "data/*"], fmt="auto",
            text_fields=["title", "abstract"], notes="Technical report abstracts."),
    Dataset("big-patent", "business", "hf", "CC BY 4.0",
            "https://huggingface.co/datasets/NortheasternUniversity/big_patent",
            repo="NortheasternUniversity/big_patent", patterns=["data/all/train-*"], max_files=4, fmt="auto",
            text_fields=["abstract"], notes="Patent abstracts: technical problem/solution language."),
    # ---------------- Agentic ----------------
    Dataset("glaive-function-calling-v2", "agentic", "hf", "Apache-2.0",
            "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
            repo="glaiveai/glaive-function-calling-v2", patterns=["*.json"], fmt="auto",
            text_fields=["system", "chat"], notes="Function-calling dialogues (format exposure only)."),
    Dataset("hermes-function-calling-v1", "agentic", "hf", "Apache-2.0",
            "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1",
            repo="NousResearch/hermes-function-calling-v1", patterns=["*.json"], fmt="auto",
            text_fields=["conversations"], notes="Structured tool-calling conversations."),
    Dataset("synthetic-srlm-traces", "agentic", "local", "Generated by this repo (agent/synth); same license as repo",
            "agent/synth/generate.py", notes="Programmatic SRLM program traces with calibrated confidence (Phase 4)."),
]

BY_NAME = {d.name: d for d in REGISTRY}

# Benchmarks used for decontamination (Phase 2 stage 6). Public eval sets.
BENCHMARKS = {
    "hellaswag": ("Rowan/hellaswag", "MIT"),
    "arc": ("allenai/ai2_arc", "CC BY-SA 4.0"),
    "mmlu": ("cais/mmlu", "MIT"),
    "piqa": ("ybisk/piqa", "AFL-3.0"),
    "winogrande": ("allenai/winogrande", "CC BY 4.0"),
    "gsm8k": ("openai/gsm8k", "MIT"),
    "lambada": ("EleutherAI/lambada_openai", "MIT"),
}
