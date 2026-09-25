# Business Entity Resolution

## Overview

This project solves the Business Entity Resolution challenge: given noisy
business records from three independent sources (Source 1 = deduplicated
reference, Source 2, Source 3), determine which records across sources refer
to the same real-world business entity.

The pipeline consists of:

1. Data loading
2. Data validation
3. Text normalization
4. Candidate generation / blocking
5. Feature engineering
6. Machine learning matching
7. Decision and singleton handling
8. Submission generation

## Scale

This is a large-data problem, not a toy one:

| File | Rows | Size |
| --- | --- | --- |
| train_source1.tsv | ~2.21M | ~210 MB |
| train_source2.tsv | ~5.03M | ~489 MB |
| train_source3.tsv | ~5.29M | ~504 MB |
| train_ground_truth.tsv | ~2.21M | ~127 MB |
| test_source1.tsv | ~1.73M | ~175 MB |
| test_source2.tsv | ~4.89M | ~509 MB |
| test_source3.tsv | ~5.08M | ~506 MB |

Naive full-file loads and O(n×m) comparisons are not viable. All data-check
and pipeline scripts read in chunks / with `usecols` where possible, and the
blocking stage is what keeps downstream matching tractable.

## Input

Training:

- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

Test:

- test_source1.tsv
- test_source2.tsv
- test_source3.tsv

## Output

The pipeline generates:

- matching_results.tsv (scored on the leaderboard)
- candidate_pairs.tsv (blocking candidate set, not scored)

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Stage 0 environment/data checks:

```bash
python src/data/environment_check.py
python src/data/data_check.py
python src/data/schema_check.py
python src/data/id_check.py
python src/data/country_check.py
python src/data/missing_check.py
python src/data/ground_truth_check.py
```

Full pipeline (once implemented):

```bash
python -m src.main
```

## Data Policy

The pipeline uses only the provided challenge data and does not perform
external business lookup or external data augmentation, per the challenge's
fair-play rules.
