# Business Entity Resolution — ML Challenge 2026

Matches every Source 1 business record to its Source 2 / Source 3 records
(0, 1 or many) and writes the two submission files:

- `output/matching_results.tsv` — final matches (scored, macro F0.5)
- `output/candidate_pairs.tsv` — the candidate set the model scored

Validation (20% of training Source 1 entities, never used for fitting):
**macro F0.5 0.9749** (2-fold cross-fitted), candidate ceiling 0.9917.
Methodology: `Documentation_template.md` in the submission root.

## 1. Layout

The code expects this directory layout (paths are resolved from `src/`):

```
<root>/
├── challenge/dataset/{train,test}/*.tsv   # challenge data (not shipped)
├── challenge/utils/validate_submission.py
├── code/business_entity_resolution/      # this folder
│   ├── src/
│   ├── model_artifacts/                  # trained models, decision rules, token translations
│   ├── README.md
│   └── requirements.txt
├── artifacts/                            # generated: normalized data, candidates, features
├── experiments/                          # generated: reports for every stage
└── output/                               # generated: submission files
```

## 2. Environment

Python 3.13, CPU only (developed on 32 cores / 16 GB RAM, Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Main libraries: pandas, numpy, pyarrow, scipy, scikit-learn, rapidfuzz,
lightgbm, xgboost — all MIT / BSD / Apache-2.0. No external data, APIs or
pretrained models are used; the only learned artefacts come from the provided
training data.

## 3. Reproduce the submission (uses the shipped model artifacts)

```bash
cd code/business_entity_resolution
python -m src.main --split test --all-countries-in-subprocesses
```

Test inference runs one country at a time, each in a fresh process (no true link
crosses countries and every blocking channel is country-scoped, so this is exact;
IDF statistics still come from the whole test reference universe). Each finished
country leaves a `done.json` marker, so an interrupted run resumes where it stopped.
On machines with more than ~32 GB RAM, `python -m src.main --split test` runs all
countries in one process.

Steps performed:
1. Stage 3 normalization of the test files if `artifacts/normalized/` is missing.
2. Stage 4 blocking for every test S1 (country-scoped exact name / fingerprint
   channels + IDF-weighted name+address retrieval with the learned token translation).
3. Stage 5 features (102 used by the model) for every candidate pair.
4. Stage 6 ensemble probability (mean of LightGBM and XGBoost).
5. Stage 7 decisions (threshold 0.80, S2/S3 exclusivity, per-source caps).
6. Writes both TSVs to `<root>/output/` and runs the official validator.

Runtime on the reference machine (16 GB RAM, 32 cores): ~4-5 h for 1.73M test S1
(candidates are streamed in 20k-S1 chunks of ~2.8M pairs).

`python -m src.main --split validation --sample 20000` runs the same pipeline on
20,000 validation S1 and checks it reproduces the staged candidates exactly.

## 4. Rebuild everything from the raw data

Each stage writes a report to `experiments/stageN/`. Run from this folder.

```bash
# Stage 0-1: environment and data audit (optional, reports only)
python src/data/environment_check.py
python src/data/audit.py
python src/data/ground_truth_analysis.py

# Stage 2: deterministic validation split (20% of S1, seed 42) + metric tests
python src/evaluation/validation_split.py
python src/evaluation/run_validation.py

# Stage 3: multi-view normalization of all six files (+ audit)
python src/normalization/run_stage3.py

# Stage 4: candidates for validation and for a 200k-S1 training sample
python src/blocking/run_stage4.py --queries validation --write-candidates
python src/blocking/run_stage4.py --queries train_sample --size 200000 --write-candidates

# Stage 5: pair features
python src/features/run_stage5.py --sets train_sample,validation

# Stage 6: train, compare (validation subset), score the selected models, ensemble
python src/models/train.py --phase fit
python src/models/train.py --phase evaluate --scope subset
python src/models/train.py --phase evaluate --scope full --experiments M001_all_weighted,M008_xgboost_all_weighted
python src/models/ensemble.py --id E001_lgbm_xgb --members M001_all_weighted,M008_xgboost_all_weighted

# Stage 7: decision rules (cross-fitted on validation) + analyses
python src/decision/run_stage7.py --model E001_lgbm_xgb

# Stage 8: error analysis of the frozen baseline
python src/analysis/run_stage8.py

# Final: learn token translations from all training labels, run test, package
python -m src.main --split test --all-countries-in-subprocesses
python src/export_artifacts.py
python src/package_submission.py --team "<team name>"
```

Blocking parameter search (optional): `python src/blocking/run_stage4.py --queries tuning --channels all --tag iter1`
(see `experiments/stage4/tuning/tuning_log.md`).

## 5. Pipeline summary

| Stage | Module | What it does |
|---|---|---|
| 3 | `normalization/` | raw values kept; NFKC + casefold + Latin-only diacritic removal (Indic scripts untouched); views: basic, compact, fingerprint, content (legal forms / connectors / web tokens removed), numbers, script |
| 4 | `blocking/` | per country: exact `content_name`, `fingerprint_name`, `fingerprint_address`; `hybrid_translated` = IDF-weighted name+address token overlap, top-100, de-leet + learned S2/S3→S1 token translation |
| 5 | `features/` | string similarities (RapidFuzz), token/IDF evidence, legal-form and number conflicts, missingness, cross-field and within-S1 relative features, blocking provenance |
| 6 | `models/` | LightGBM (3,000 trees) + XGBoost (964 trees) on 3.96M sampled pairs from 200k training S1, weighted to the true candidate distribution |
| 7 | `decision/` | threshold 0.80, each S2/S3 record assigned to at most one S1, caps 5 (S2) / 6 (S3) |

## 6. Validation results

| | Value |
|---|---|
| Candidate pair recall | 0.9755 |
| Candidate F0.5 ceiling | 0.9917 |
| Ensemble, global threshold | 0.97437 |
| **Ensemble + decision engine (out-of-fold)** | **0.97492** |
| Macro precision / recall | 0.9875 / 0.9447 |
| US / India | 0.9791 / 0.9687 |

## 7. Data policy

Only the provided challenge data is used. Token translations, IDF weights and
the models are learned from the training files; nothing is looked up
externally. Test data is used only for inference (its own reference records
supply the IDF statistics); no parameter was tuned on test data.
