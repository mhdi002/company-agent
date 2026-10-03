# Cleaning report

Input: **3,420 docs / 3,842,794 chars** → Output: **2,882 docs / 3,369,644 chars** (84.3% docs kept).

| Stage | Docs in | Docs out | Removed | Chars in | Chars out | Top reasons | Seconds |
|---|---|---|---|---|---|---|---|
| 1. normalize | 3,420 | 3,420 | 0 | 3,842,794 | 3,833,114 | kept: 3420 | 0.25 |
| 2. langid | 3,420 | 3,200 | 220 | 3,833,114 | 3,706,973 | kept: 3200, lang_it: 70, lang_es: 30, lang_fr: 30, lang_de: 30 | 1.26 |
| 3. quality | 3,200 | 3,120 | 80 | 3,706,973 | 3,647,560 | kept: 3120, dup_lines: 40, digit_ratio: 40 | 2.13 |
| 4. dedup | 3,120 | 2,922 | 198 | 3,647,560 | 3,390,274 | kept: 2922, exact_dup: 100, near_dup: 98 | 1.61 |
| 5. pii_toxicity | 2,922 | 2,892 | 30 | 3,390,274 | 3,376,099 | kept: 2852, kept_scrubbed: 40, toxic: 30 | 0.29 |
| 6. decontam | 2,892 | 2,882 | 10 | 3,376,099 | 3,369,644 | kept: 2882, benchmark_overlap: 10 | 0.24 |

## Docs per source (before → after)

| Source | Before | After |
|---|---|---|
| sample | 3,180 | 2,642 |
| sample-business | 240 | 240 |

Review samples: `data/reports/stage<N>_<name>_samples.jsonl` (100 random kept + 100 random removed per stage; `before` shows the pre-stage text when modified).
