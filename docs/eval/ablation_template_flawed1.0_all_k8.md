### Ablation — policy=template (flawed×1.0), K=8, 30 held-out companies

| Variant | Field acc. | Evidence P | Evidence R | Project | Halluc. rate | Completeness | s/company |
|---|---|---|---|---|---|---|---|
| direct | 0.9 | 0.9222 | 0.873 | 0.6517 | 0.0 | 0.9843 | 0.7783 |
| +verb_conf | 0.9333 | 0.6333 | 0.5289 | 0.2667 | 0.0 | 0.9804 | 2.0557 |
| +trace_len | 0.9333 | 0.6333 | 0.5289 | 0.2667 | 0.0 | 0.9804 | 2.2819 |
| +self_consist | 0.9333 | 1.0 | 0.8914 | 0.7117 | 0.0 | 0.9843 | 2.3431 |
| srlm | 0.9333 | 0.9667 | 0.874 | 0.6383 | 0.0 | 0.9843 | 2.203 |
