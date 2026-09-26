# ML Challenge 2026 — Business Entity Resolution

Matches every Source 1 business record to its Source 2 / Source 3 records.
Validation macro F0.5: **0.9749** (candidate ceiling 0.9917).

```text
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard file)
│   └── candidate_pairs.tsv         # blocking candidate set (in the submission zip; too large for GitHub)
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── model_artifacts/        # trained models, decision rules, token translations
│       ├── README.md               # how to reproduce end-to-end (data -> blocking -> matching -> output)
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # methodology write-up
```

Reproduction: see `code/business_entity_resolution/README.md`. The challenge data is not
included; place it under `challenge/dataset/{train,test}/`.
