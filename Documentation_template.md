# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve each Source 1 record against ~10M Source 2/3 records with a four-part pipeline:
multi-view normalization, high-recall blocking driven by an IDF-weighted name+address
retriever with a **token translation learned from the training labels** (covering state
codes, abbreviations and Indian-script names), a 102-feature pairwise evidence engine,
and a LightGBM + XGBoost ensemble followed by a precision-first per-entity decision engine
with S2/S3 exclusivity. On a held-out 20% of training Source 1 entities the system scores
**macro F0.5 ≈ 0.975** (2-fold cross-fitted estimate 0.97492); the candidate set limits a
perfect matcher to 0.9917.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from a full audit of the training data (2.21M S1, 5.03M S2, 5.29M S3 records):

- **Ground-truth structure.** 5.58% of S1 entities have no match; matched S1 have 3.67 links on
  average (max 11; up to 5 in S2 and 6 in S3). **Every S2/S3 record belongs to at most one S1**,
  ~26% of S2/S3 records match nothing (distractors), and **all 7.6M links stay within one
  country**.
- **S1 is clean; S2/S3 are systematically corrupted copies.** Observed noise (with measured
  frequencies): leetspeak (`c0rp`, `5ervices`, `6roup`), injected accents, word shuffles, legal
  suffixes added/dropped/swapped, honorifics and filler tokens (`Smt`, `Center`, `Greater`),
  domain-style names (`fortunefinance.com`, ~4%), DBA/formerly aliases, acronyms, and **names
  replaced entirely** (only the address links them).
- **Native scripts.** 13–23% of India S2/S3 names are in Devanagari, Bengali, Tamil, Telugu,
  Kannada, Malayalam, Gujarati, etc.; S1 names are always Latin.
- **Addresses.** Reordered components, state as code / full name / native script, street
  abbreviations (and wrong expansions such as `Street → SAINT`), altered or zero-padded house
  numbers, literal `NULL`/`N/A` components (~2.7%), and ~3% empty addresses in S2/S3. Postal codes
  are essentially absent (5-digit US numbers are house numbers).
- **Generic chain names** repeat hundreds of times (`Primary Care Group` ×253 in S1), so a name
  alone cannot identify a business.
- **France** (~15% of test) has no training examples; country is treated as an open set.
- Two data traps: default CSV readers turn the literal business name `NA` into null (every such
  record is an acronym of its true match), and pyarrow's Unicode normalization does not compose
  characters, so all normalization uses Python `unicodedata`.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise gradient-boosted classifier + per-entity decision engine
**Core Innovation:** a token translation dictionary learned purely from training labels (for
each S2/S3-only token, the S1 token it co-occurs with in true pairs), used inside an
IDF-weighted name+address retriever. It recovers state-name/code, abbreviation, typo and
Indian-script transliteration correspondences (`टेक→tech`, `ಕರ್ನಾಟಕ→karnataka`,
`arizona→az`, `ave→avenue`) without any external data.

Pipeline: Stage 3 normalization → Stage 4 blocking → Stage 5 features → Stage 6 model →
Stage 7 decisions. Every stage was measured against a competition-exact macro F0.5 evaluator
on a fixed validation split (20% of training S1, stratified by country × match count); all
parameters were chosen on training data or with cross-fitting, never on test.

---

## 3. Candidate Generation (Blocking)

All channels operate **within one country** (no true link crosses countries). Final channels:

| Channel | Key | Role |
|---|---|---|
| `content_name` | name without legal forms/connectors/web tokens, spaces removed | exact name match despite suffix/punctuation noise |
| `fingerprint_name` | sorted unique name tokens | word-order shuffles |
| `fingerprint_address` | sorted unique address tokens | reordered addresses |
| `hybrid_translated` | IDF-weighted overlap of name + address tokens, top-100 per S1 | the main retriever: rare shared tokens in either field |

`hybrid_translated` de-leets names and appends the learned S2/S3→S1 token translations to
reference text; tokens with document frequency above 50,000 are dropped, the rest weighted by
IDF; retrieval is a sparse matrix product with vectorized top-K, multi-threaded.

- **Blocking keys used:** normalized name / name fingerprint / address fingerprint (exact), and
  IDF-weighted rare-token overlap of name + address with learned token translation (ranked).
- **Candidate pairs generated:** validation 61.4M (median 100, mean 139 per S1);
  test 222.4M over 1,732,544 S1 (mean 128 per S1; France 36.4M, India 101.2M, US 84.8M).
  Test output: 5,588,777 predicted matches; 1,627,229 S1 with at least one match, 105,315 predicted singletons (6.1%).
- **How you ensured true matches were not lost:** channels were chosen by measured recall on a
  50,000-S1 training sample (10 channels tried, leave-one-out contribution measured, top-K and
  `max_df` swept) and confirmed once on validation: **97.55% of true pairs retrieved; 99.77% of
  matched S1 have at least one true candidate; F0.5 ceiling 0.9917** (US 0.9960, India 0.9853).
  Per-candidate provenance and retrieval rank are kept as model features.

---

## 4. Matching Model

**Features used (102 non-constant):**
- Name features: exact matches on basic / compact / content / fingerprint views; RapidFuzz
  ratio, partial ratio, token-sort, token-set, Jaro-Winkler; token Jaccard and containment;
  IDF-weighted shared / unshared token evidence and rare-token counts (on the translated view);
  legal-form presence and conflict; acronym match; candidate script.
- Address features: exact matches on basic / compact / fingerprint; RapidFuzz similarities
  (including on the translated view); token Jaccard, containment and IDF evidence.
- Other: address-number overlap, conflict, first-number and long-number agreement; name-number
  conflict; missingness flags (missing ≠ mismatch; NaN for "no evidence"); name×address
  cross-field combinations; **within-S1 relative features** (rank, margin to best, lead over
  second on name, address and combined evidence); blocking provenance, retrieval rank and
  candidate-set size; source.

**Model type:** mean of a LightGBM classifier (3,000 trees, 31 leaves) and an XGBoost
classifier (964 trees, depth 8), trained on 3.96M pairs sampled from the candidates of 200,000
training S1 (all positives; hard/medium/easy negatives sampled at 0.20/0.06/0.04 with
inverse-rate weights); early stopping on a separate S1-grouped dev split.
Ablations (same validation subset): name only 0.714, + address 0.968, + numbers 0.973, all 0.974.

**Threshold selection method:** entity-level macro F0.5 on validation (never pair-level).
Decision rules searched greedily with 2-fold cross-fitting over validation S1: global threshold,
separate first/other thresholds, top-vs-second margin, relative-to-best cut, per-source caps,
S2/S3 exclusivity and per-country thresholds. Chosen: threshold 0.80, each S2/S3 record assigned
to at most one S1 (highest probability), caps of 5 (S2) and 6 (S3) matches. The score is flat
for ±0.02 around the threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.97492 on the validation split (2-fold out-of-fold estimate;
  precision 0.9875, recall 0.9447; US 0.9791, India 0.9687; empty-truth entities 0.968).
  Candidate ceiling 0.9917.
- **Common false positives (wrong merges):** 91% are an extra candidate added to an S1 that also
  received its true matches (typically a sibling record of the same chain at a nearby
  address); 9% are false matches for S1 with no true match. Pair-level precision is 0.993.
- **Common false negatives (missed matches):** 42% never retrieved by blocking (mostly
  empty-address records with corrupted generic names, and native-script names without a learned
  translation); 26% have an empty candidate address so only a noisy name is available; 18%
  have good similarities but fall just below the threshold; 10% carry heavy name noise.

---

## 6. Conclusion

A precision-first blocking + GBDT pipeline reaches macro F0.5 ≈ 0.975 on held-out training
entities. The largest gains came from data-driven retrieval (IDF weighting with common tokens
kept at low weight, and a label-learned token translation that handles Indian scripts and state
formats) and from within-entity relative features; decision rules added mainly through S2/S3
exclusivity. Remaining headroom is in retrieving records whose address is empty and in
native-script names.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

```
src/
  data/           Stage 0-1 environment and data audit scripts
  evaluation/     competition-exact macro F0.5, validation split
  normalization/  Stage 3 multi-view normalization (+ audit)
  blocking/       Stage 4 channels, learned token translation, candidate union, evaluation
  features/       Stage 5 pairwise features and quality gate
  models/         Stage 6 training, ensemble, diagnostics
  decision/       Stage 7 decision engine, threshold search, analyses
  inference/      end-to-end streaming pipeline
  main.py         entry point
  export_artifacts.py
model_artifacts/  trained LightGBM + XGBoost, decision parameters, token translations
README.md         full reproduction steps
requirements.txt  pinned environment (Python 3.13)
```

Reproduce the submission files (data placed under `challenge/dataset/`):

```bash
cd code/business_entity_resolution
python -m src.main --split test
```

This normalizes the test files if needed, runs blocking → features → model → decisions for all
test S1 and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the
official validator. The README lists the commands that rebuild every model artifact from the
training data.

### B. Additional Results

| Experiment (validation) | Macro F0.5 |
|---|---:|
| Predict no matches | 0.0558 |
| Name-only model, best threshold (subset) | 0.7136 |
| LightGBM, all features, best threshold | 0.97403 |
| XGBoost, all features, best threshold | 0.97395 |
| Ensemble, best threshold | 0.97437 |
| **Ensemble + decision engine (cross-fitted)** | **0.97492** |
| Candidate ceiling (perfect matcher on candidates) | 0.99168 |

Blocking tuning (50k training S1): 10-channel baseline ceiling 0.9686 → raising `max_df`
0.9857 → learned translation 0.9938 (K=200); final K=100 chosen for cost at ceiling 0.9920.
