"""
Stage 3 audit: information loss, collisions, identity conflicts (train split
only), script mix, Unicode stability, idempotence, raw preservation, and
token-frequency evidence for future rules.
"""

from collections import Counter
from pathlib import Path
import random
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .address_normalizer import normalize_addresses
from .name_normalizer import normalize_names
from .tokenization import canonical_numbers


NAME_VIEWS = [
    "name_nfkc", "name_basic", "name_compact",
    "name_content", "name_content_compact", "name_fingerprint",
]
ADDRESS_VIEWS = ["address_nfkc", "address_basic", "address_compact", "address_fingerprint"]

GT_NAME_VIEWS = ["name_raw"] + NAME_VIEWS
GT_ADDRESS_VIEWS = ["address_raw"] + ADDRESS_VIEWS + ["address_numbers"]
GT_COMBOS = [
    ("name_raw", "address_raw"),
    ("name_nfkc", "address_nfkc"),
    ("name_basic", "address_basic"),
    ("name_compact", "address_compact"),
    ("name_content_compact", "address_compact"),
    ("name_fingerprint", "address_fingerprint"),
]

TOP_COLLISIONS = 15
SAMPLE_VARIANTS = 5


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def sorted_numbers(series: pd.Series) -> pd.Series:
    return series.map(lambda s: " ".join(sorted(s.split())))


# ============================================================
# 1. INFORMATION LOSS + COLLISIONS (per file, per view)
# ============================================================

def view_audit(df: pd.DataFrame, file: str, raw_col: str, view_col: str, numbers_key: pd.Series):
    raw = df[raw_col]
    view = df[view_col]

    raw_present = raw.str.strip().ne("")
    view_present = view.ne("")
    raw_digit = raw.str.contains(r"[0-9]", regex=True)
    view_digit = view.str.contains(r"[0-9]", regex=True)

    keep = raw_present & view_present
    pairs = pd.DataFrame({
        "view": view[keep], "raw": raw[keep], "num": numbers_key[keep],
    }).drop_duplicates(["view", "raw"])

    variants = pairs.groupby("view", sort=False).agg(
        raw_variants=("raw", "size"), number_variants=("num", "nunique"),
    )
    colliding = variants[variants["raw_variants"] > 1]
    conflicting = colliding[colliding["number_variants"] > 1]
    rows_per_view = view[view_present].value_counts()

    metrics = {
        "file": file,
        "view": view_col,
        "rows": int(len(df)),
        "raw_present": int(raw_present.sum()),
        "changed_vs_raw": int((raw_present & view.ne(raw)).sum()),
        "became_empty": int((raw_present & ~view_present).sum()),
        "zero_token_rows": int((~view_present).sum()),
        "raw_has_digit": int(raw_digit.sum()),
        "digit_lost": int((raw_digit & ~view_digit).sum()),
        "raw_distinct": int(len(pairs)),
        "view_distinct": int(len(variants)),
        "collapse_ratio": round(len(pairs) / max(len(variants), 1), 4),
        "colliding_view_values": int(len(colliding)),
        "rows_in_collisions": int(rows_per_view.reindex(colliding.index).sum()),
        "digit_conflict_view_values": int(len(conflicting)),
        "rows_in_digit_conflicts": int(rows_per_view.reindex(conflicting.index).sum()),
    }

    picked = pd.concat([
        colliding.nlargest(TOP_COLLISIONS, "raw_variants").assign(kind="largest_collision"),
        conflicting.nlargest(TOP_COLLISIONS, "raw_variants").assign(kind="digit_conflict"),
    ])
    samples = (
        pairs[pairs["view"].isin(picked.index)]
        .groupby("view")["raw"]
        .agg(lambda s: " || ".join(s.head(SAMPLE_VARIANTS)))
    )
    examples = picked.assign(
        file=file, view_column=view_col,
        rows=rows_per_view.reindex(picked.index).values,
        raw_examples=samples.reindex(picked.index).values,
    ).rename_axis("view_value").reset_index()

    return metrics, examples


def file_loss_and_collisions(path: Path, file: str):
    metrics, examples = [], []

    for raw_col, views, numbers_col in [
        ("name_raw", NAME_VIEWS, "name_numbers"),
        ("address_raw", ADDRESS_VIEWS + ["address_numbers"], "address_numbers"),
    ]:
        df = read_columns(path, [raw_col, numbers_col] + [v for v in views if v != numbers_col])
        key = sorted_numbers(df[numbers_col])
        for view_col in views:
            m, e = view_audit(df, file, raw_col, view_col, key)
            metrics.append(m)
            examples.append(e)
        del df

    return metrics, pd.concat(examples, ignore_index=True)


# ============================================================
# 2. SCRIPT MIX
# ============================================================

def script_profile(path: Path) -> dict:
    df = read_columns(path, ["country_raw", "name_script", "name_script_mixed",
                             "address_script", "address_script_mixed"])
    out = {}
    for country, g in df.groupby("country_raw"):
        out[country] = {
            "rows": int(len(g)),
            "name_script_pct": (100 * g["name_script"].value_counts(normalize=True)).round(3).to_dict(),
            "name_mixed_script_pct": round(100 * float(g["name_script_mixed"].mean()), 3),
            "address_script_pct": (100 * g["address_script"].value_counts(normalize=True)).round(3).to_dict(),
            "address_mixed_script_pct": round(100 * float(g["address_script_mixed"].mean()), 3),
        }
    return out


# ============================================================
# 3. PROPERTY CHECKS ON REAL DATA
# ============================================================

def raw_preservation(path: Path, source_tsv: Path) -> dict:
    # Independent parser (pandas C engine, no NA conversion), one column at a time.
    result = {}
    for source_col, normalized_col, label in [
        ("entity_id", "entity_id", "entity_id_mismatches"),
        ("business_name", "name_raw", "name_raw_mismatches"),
        ("business_address", "address_raw", "address_raw_mismatches"),
        ("country", "country_raw", "country_mismatches"),
    ]:
        original = pd.read_csv(source_tsv, sep="\t", usecols=[source_col], dtype=str, na_filter=False)[source_col]
        normalized = pq.read_table(path, columns=[normalized_col])[normalized_col].to_pylist()
        result["rows_equal"] = len(original) == len(normalized)
        result[label] = sum(a != b for a, b in zip(original.tolist(), normalized)) + abs(len(original) - len(normalized))
        del original, normalized
    return result


def idempotence(path: Path, sample_rows: int, seed: int) -> dict:
    table = pq.read_table(path)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(table.num_rows), min(sample_rows, table.num_rows)))
    df = table.take(pa.array(idx)).to_pandas()
    del table

    country = df["country_raw"]
    checks = {}

    # normalize(view)[view] must equal view.
    for view in ["name_basic", "name_compact", "name_content", "name_fingerprint"]:
        again = normalize_names(df[view], country)
        checks[view] = int((pd.Series(_py(again[view])) != df[view].values).sum())

    checks["name_numbers"] = int((df["name_numbers"].map(canonical_numbers) != df["name_numbers"]).sum())

    for view in ["address_basic", "address_compact", "address_fingerprint"]:
        again = normalize_addresses(df[view])
        checks[view] = int((pd.Series(_py(again[view])) != df[view].values).sum())

    checks["address_numbers"] = int((df["address_numbers"].map(canonical_numbers) != df["address_numbers"]).sum())

    return {"sample_rows": len(df), "non_idempotent_rows": checks}


def _py(values):
    return values.to_pylist() if hasattr(values, "to_pylist") else list(values)


STABILITY_VIEWS = ["name_nfkc", "name_basic", "name_compact", "name_content",
                   "address_nfkc", "address_basic", "address_compact"]


def unicode_stability(path: Path) -> dict:
    """Every normalized view must already be valid NFC and NFKC (full file)."""
    out = {}
    for col in STABILITY_VIEWS:
        values = pq.read_table(path, columns=[col])[col].to_pylist()
        bad = [v for v in values if unicodedata.normalize("NFKC", v) != v]
        out[col] = {"not_nfkc": len(bad), "examples": bad[:3]}
        del values
    return out


# ============================================================
# 4. GROUND-TRUTH DIAGNOSTICS (TRAIN SPLIT ONLY)
# ============================================================

def train_pairs(gt_path: Path, validation_gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    validation_ids = set(
        pd.read_csv(validation_gt_path, sep="\t", dtype=str, keep_default_na=False)["source1_entity_id"]
    )
    gt = gt[~gt["source1_entity_id"].isin(validation_ids)]
    pairs = gt.assign(target_id=gt["matched_entity_ids"].str.split(",")).explode("target_id")
    pairs = pairs[pairs["target_id"].fillna("").ne("")]
    return pairs[["source1_entity_id", "target_id"]].reset_index(drop=True)


def gt_diagnostics(normalized: dict[str, Path], gt_path: Path, validation_gt_path: Path) -> list[dict]:
    pairs = train_pairs(gt_path, validation_gt_path)
    train_s1_ids = set(pairs["source1_entity_id"])

    s1_meta = read_columns(normalized["train_source1"], ["entity_id", "country_raw"])
    s1_meta = s1_meta[s1_meta["entity_id"].isin(train_s1_ids)]
    s1_country = s1_meta.set_index("entity_id")["country_raw"]
    pairs["country"] = pairs["source1_entity_id"].map(s1_country)
    del s1_meta

    specs = [(v,) for v in GT_NAME_VIEWS + GT_ADDRESS_VIEWS] + list(GT_COMBOS)
    rows = []

    for spec in specs:
        cols = list(spec)
        s1 = read_columns(normalized["train_source1"], ["entity_id"] + cols)
        s1 = s1[s1["entity_id"].isin(train_s1_ids)]
        targets = pd.concat(
            [read_columns(normalized[k], ["entity_id"] + cols) for k in ("train_source2", "train_source3")],
            ignore_index=True,
        )
        s1_key = _key(s1, cols)
        target_key = _key(targets, cols)
        s1_map = pd.Series(s1_key.values, index=s1["entity_id"].values)
        target_map = pd.Series(target_key.values, index=targets["entity_id"].values)
        del s1, targets

        left = pairs["source1_entity_id"].map(s1_map)
        right = pairs["target_id"].map(target_map)
        present = left.ne("") & right.ne("")
        equal = present & left.eq(right)

        s1_present = s1_map[s1_map.ne("")]
        s1_shared = s1_present.duplicated(keep=False)

        matched = pd.DataFrame({"key": right[right.ne("")], "s1": pairs.loc[right.ne(""), "source1_entity_id"]})
        s1_per_key = matched.groupby("key")["s1"].nunique()
        conflicting_keys = s1_per_key[s1_per_key > 1].index
        conflicted_targets = matched["key"].isin(conflicting_keys)

        row = {
            "view": " + ".join(cols),
            "true_pairs": int(len(pairs)),
            "pairs_both_present": int(present.sum()),
            "true_pair_exact_agreement_pct": round(100 * float(equal.mean()), 3),
            "s1_entities": int(len(s1_map)),
            "s1_sharing_key_with_other_s1": int(s1_shared.sum()),
            "s1_sharing_key_pct": round(100 * float(s1_shared.mean()), 3),
            "matched_targets": int(len(matched)),
            "targets_in_cross_entity_groups": int(conflicted_targets.sum()),
            "targets_in_cross_entity_groups_pct": round(100 * float(conflicted_targets.mean()), 3),
        }
        for country, g in equal.groupby(pairs["country"]):
            row[f"agreement_pct_{country}"] = round(100 * float(g.mean()), 3)
        rows.append(row)
        print(f"  GT diagnostics: {row['view']:<45} agreement {row['true_pair_exact_agreement_pct']:6.2f}%  "
              f"S1 sharing {row['s1_sharing_key_pct']:6.3f}%  cross-entity targets {row['targets_in_cross_entity_groups_pct']:6.3f}%")
        del s1_map, target_map, left, right, matched

    return rows


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].fillna("")
    parts = [df[c].fillna("") for c in cols]
    key = parts[0] + "\x1f" + parts[1]
    return key.where(parts[0].ne("") & parts[1].ne(""), "")


# ============================================================
# 5. TOKEN FREQUENCY (evidence for future rules; no labels used)
# ============================================================

def token_frequencies(normalized: dict[str, Path], min_count: int = 200, top_n: int = 300) -> pd.DataFrame:
    counts = {}
    totals = {}
    for source in ("source1", "source2", "source3"):
        df = read_columns(normalized[f"train_{source}"], ["country_raw", "name_basic"])
        for country, g in df.groupby("country_raw"):
            counter = Counter()
            for text in g["name_basic"]:
                counter.update(set(text.split()))
            counts[(source, country)] = counter
            totals[(source, country)] = len(g)
        del df

    rows = []
    for country in sorted({c for _, c in counts}):
        tokens = set()
        for source in ("source1", "source2", "source3"):
            tokens.update(t for t, n in counts[(source, country)].items() if n >= min_count)
        for token in tokens:
            row = {"country": country, "token": token}
            for source in ("source1", "source2", "source3"):
                n = counts[(source, country)][token]
                row[f"{source}_count"] = n
                row[f"{source}_pct"] = round(100 * n / totals[(source, country)], 4)
            s1 = row["source1_pct"]
            s23 = (row["source2_pct"] + row["source3_pct"]) / 2
            row["s2s3_vs_s1_ratio"] = round(s23 / s1, 3) if s1 > 0 else float("inf")
            rows.append(row)

    table = pd.DataFrame(rows)
    frequent = table.sort_values("source1_count", ascending=False).groupby("country").head(top_n)
    injected = table.sort_values("s2s3_vs_s1_ratio", ascending=False).groupby("country").head(top_n)
    return pd.concat([frequent.assign(list="most_frequent_in_s1"),
                      injected.assign(list="overrepresented_in_s2_s3")], ignore_index=True)


# ============================================================
# 6. EXAMPLES FOR MANUAL INSPECTION
# ============================================================

EXAMPLE_COLUMNS = [
    "entity_id", "country_raw",
    "name_raw", "name_basic", "name_compact", "name_content", "name_content_compact",
    "name_fingerprint", "name_numbers", "name_script", "name_script_mixed", "name_content_fallback",
    "address_raw", "address_basic", "address_compact", "address_fingerprint", "address_numbers",
    "address_placeholder_removed", "address_script",
]


def examples(path: Path, file: str, seed: int, per_group: int = 12, per_special: int = 8) -> pd.DataFrame:
    df = read_columns(path, EXAMPLE_COLUMNS + ["name_missing"])
    frames = []
    for country, g in df.groupby("country_raw"):
        frames.append(g.sample(min(per_group, len(g)), random_state=seed).assign(category=f"random_{country}"))

    special = {
        "non_latin_name": df["name_script"].ne("Latin") & df["name_raw"].ne(""),
        "mixed_script_name": df["name_script_mixed"],
        "address_placeholder_removed": df["address_placeholder_removed"],
        "name_content_fallback": df["name_content_fallback"],
        "name_became_missing": df["name_missing"] & df["name_raw"].str.strip().ne(""),
        "name_has_digits": df["name_numbers"].ne(""),
    }
    for category, mask in special.items():
        g = df[mask]
        if len(g):
            frames.append(g.sample(min(per_special, len(g)), random_state=seed).assign(category=category))

    out = pd.concat(frames, ignore_index=True).drop(columns=["name_missing"])
    out.insert(0, "file", file)
    return out
