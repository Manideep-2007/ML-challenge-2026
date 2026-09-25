# Business Entity Resolution Challenge

## Project Structure

```text
challenge/       # raw challenge data + provided tooling (untouched)
code/            # our pipeline (code/business_entity_resolution)
output/          # matching_results.tsv + candidate_pairs.tsv (generated later)
experiments/     # experiment log / run artifacts
```

## Goal

Resolve noisy business records across Source 1 (deduplicated reference),
Source 2, and Source 3, producing matches for every Source 1 entity. Scored
with macro-averaged F_0.5 (precision-heavy).

## Development Strategy

1. Environment setup
2. Data audit
3. Validation framework
4. Normalization
5. Blocking / candidate generation
6. Feature engineering
7. ML matching
8. F_0.5 threshold optimization
9. Error analysis
10. Final submission

## Notes

- Data is large: ~2.2M Source 1 records, ~5M each for Source 2/Source 3
  (train and test combined ~2.5 GB). See
  `code/business_entity_resolution/README.md` for exact sizes and the
  memory-safe (chunked) approach used throughout.
- `country` is an open string field — test data includes `France`, which
  never appears in training. Do not hardcode to `{US, India}`.
- No external data lookups/APIs allowed (fair-play rule).
