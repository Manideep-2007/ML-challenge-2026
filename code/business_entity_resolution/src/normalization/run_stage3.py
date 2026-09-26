from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from normalization.normalization_pipeline import normalize_file, normalize_record  # noqa: E402
from normalization import normalization_audit as audit  # noqa: E402

DATASET_DIR = ROOT / "challenge" / "dataset"
NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
STAGE3_DIR = ROOT / "experiments" / "stage3"
GT_PATH = DATASET_DIR / "train" / "train_ground_truth.tsv"
VALIDATION_GT_PATH = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"

FILES = [f"{split}_{source}" for split in ("train", "test") for source in ("source1", "source2", "source3")]
SEED = 42
PROPERTY_SAMPLE = 100_000


def source_path(stem: str) -> Path:
    return DATASET_DIR / stem.split("_")[0] / f"{stem}.tsv"


def normalized_path(stem: str) -> Path:
    return NORMALIZED_DIR / f"{stem}.parquet"


# ============================================================
# UNIT TESTS
# ============================================================

def rec(name, address="", country="US"):
    return normalize_record(name, address, country)


def unit_tests() -> list[tuple[str, bool]]:
    results = []

    def check(name, condition):
        results.append((name, bool(condition)))
        print(f"  {'PASS' if condition else 'FAIL'}  {name}")

    hindi = "फॉर्च्यून फाइनेंस"
    malayalam = "പെർഫെക്റ്റ് കൺസൾട്ടന്റ്സ്"

    check("nfkc_fullwidth", rec("ＡＢＣ Ltd")["name_basic"] == "abc ltd")
    check("nfkc_ligature", rec("ﬁne Foods")["name_basic"] == "fine foods")
    check("casefold_sharp_s", rec("Straße Cafe")["name_basic"] == rec("STRASSE CAFE")["name_basic"])
    check("latin_diacritics_stripped", rec("Café Élan")["name_basic"] == "cafe elan")
    check("devanagari_preserved", rec(hindi, country="India")["name_basic"] == hindi)
    check("devanagari_tokens", rec(hindi, country="India")["name_tokens"] == hindi.split())
    check("record_keys_do_not_collide", rec("Alpha Beta", "12 Main Street")["name_tokens"] == ["alpha", "beta"]
          and rec("Alpha Beta", "12 Main Street")["address_tokens"] == ["12", "main", "street"])
    check("devanagari_script", rec(hindi, country="India")["name_script"] == "Devanagari")
    check("latin_decomposed_equals_composed", rec("Café")["name_basic"] == rec("Café")["name_basic"] == "cafe")
    tamil_composed, tamil_decomposed = "கோ", "கோ"
    check("indic_decomposed_equals_composed",
          rec(tamil_decomposed, country="India")["name_basic"] == rec(tamil_composed, country="India")["name_basic"] == tamil_composed)
    check("malayalam_preserved", rec(malayalam, country="India")["name_basic"] == malayalam)
    check("mixed_script_flag", rec("आनंद Ventures", country="India")["name_script_mixed"])

    pvt = rec("ABC Pvt. Ltd.", country="India")
    check("legal_forms_kept_in_basic", pvt["name_basic"] == "abc pvt ltd")
    check("legal_forms_removed_in_content", pvt["name_content"] == "abc")
    check("compact_view", pvt["name_compact"] == "abcpvtltd")

    acme, acme_2000 = rec("ACME LTD"), rec("ACME 2000 LTD")
    for view in ["name_basic", "name_compact", "name_content", "name_fingerprint"]:
        check(f"digits_distinguish_{view}", acme[view] != acme_2000[view])
    check("name_numbers_extracted", acme_2000["name_numbers"] == "2000")

    fallback = rec("Company Limited")
    check("content_fallback_never_empty", fallback["name_content"] == "company limited" and fallback["name_content_fallback"])
    check("connector_and_ampersand", rec("Keys & Co")["name_content"] == rec("KEYS and CO")["name_content"])
    check("web_domain_content_compact",
          rec("fortunefinance.com", country="India")["name_content_compact"]
          == rec("Fortune Finance Pvt Ltd", country="India")["name_content_compact"])
    check("fingerprint_order_invariant",
          rec("Coastal Co Avalanche")["name_fingerprint"] == rec("Coastal Avalanche Co")["name_fingerprint"]
          and rec("Coastal Co Avalanche")["name_basic"] != rec("Coastal Avalanche Co")["name_basic"])

    check("france_legal_form", rec("Nantes Sport SARL", country="France")["name_content"] == "nantes sport")
    check("unknown_country_uses_union", rec("Nantes Sport SARL", country="Narnia")["name_content"] == "nantes sport")
    check("country_specific_rules", rec("Nantes Sport SARL", country="US")["name_content"] == "nantes sport sarl")
    check("name_na_is_not_missing", rec("NA")["name_basic"] == "na" and not rec("NA")["name_missing"])
    check("punctuation_only_name_missing", rec("---")["name_missing"])

    addr = rec("X", "3800, NULL, CLEARFIELD, UT")
    check("address_placeholder_component_removed", addr["address_basic"] == "3800 clearfield ut" and addr["address_placeholder_removed"])
    check("address_placeholder_whole_value_missing", rec("X", "NULL")["address_missing"])
    check("address_placeholder_only_whole_component", rec("X", "Na Road, Indore")["address_basic"] == "na road indore")
    check("address_numbers_zero_padding", rec("X", "00930 PARK CENTRAL DR")["address_numbers"] == "930")
    check("address_numbers_non_ascii_digits", rec("X", "मकान १२३, Delhi", "India")["address_numbers"] == "123")
    check("address_numbers_split", rec("X", "76, Cathedral Road, Madras-86.76")["address_numbers"] == "76 86 76")
    check("address_no_abbreviation_expansion", rec("X", "12 Main Street")["address_basic"] == "12 main street")

    raw = rec("  ABC  Ltd ", " 1 Main St ", "  France ")
    check("raw_name_preserved", raw["name_raw"] == "  ABC  Ltd ")
    check("raw_address_preserved", raw["address_raw"] == " 1 Main St ")
    check("country_preserved", raw["country_raw"] == "  France ")
    none = normalize_record(None, None, None)
    check("none_inputs_safe", none["name_missing"] and none["address_missing"] and none["name_raw"] == "")

    once = rec("Smt. Al Media (Pvt) Ltd, #78A", "H.NO 78A, Raja Basant Roy Rd., KOLKATA, N/A", "India")
    twice = rec(once["name_basic"], once["address_basic"], "India")
    check("idempotent_basic", once["name_basic"] == twice["name_basic"] and once["address_basic"] == twice["address_basic"])

    return results


# ============================================================
# NORMALIZE
# ============================================================

def normalize_all(force: bool) -> list[dict]:
    runs = []
    for stem in FILES:
        out = normalized_path(stem)
        if out.exists() and not force:
            print(f"  {stem}: exists, skipping (use --force to rebuild)")
            continue
        run = normalize_file(source_path(stem), out)
        run["output"] = str(out.relative_to(ROOT))
        run["output_mb"] = round(out.stat().st_size / 2**20, 1)
        print(f"  {stem}: {run['rows']:,} rows in {run['seconds']}s -> {run['output_mb']} MB")
        runs.append(run)
    return runs


# ============================================================
# AUDIT
# ============================================================

def run_audit() -> dict:
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    profile = {"files": {}}
    view_metrics, collision_examples, example_frames = [], [], []

    for stem in FILES:
        path = normalized_path(stem)
        t = time.time()
        metrics, collisions = audit.file_loss_and_collisions(path, stem)
        view_metrics += metrics
        collision_examples.append(collisions)
        example_frames.append(audit.examples(path, stem, SEED))
        profile["files"][stem] = {
            "raw_preservation": audit.raw_preservation(path, source_path(stem)),
            "idempotence": audit.idempotence(path, PROPERTY_SAMPLE, SEED),
            "unicode_stability": audit.unicode_stability(path),
            "scripts": audit.script_profile(path),
        }
        print(f"  audited {stem} in {time.time() - t:.0f}s")

    metrics_df = pd.DataFrame(view_metrics)
    metrics_df[metrics_df["view"].str.startswith("name")].to_csv(STAGE3_DIR / "name_collision_report.csv", index=False)
    metrics_df[metrics_df["view"].str.startswith("address")].to_csv(STAGE3_DIR / "address_collision_report.csv", index=False)
    pd.concat(collision_examples, ignore_index=True).to_csv(STAGE3_DIR / "normalization_collisions.csv", index=False)
    pd.concat(example_frames, ignore_index=True).to_csv(STAGE3_DIR / "normalization_examples.csv", index=False)
    profile["view_metrics"] = view_metrics

    print("  ground-truth diagnostics (train split only):")
    gt_rows = audit.gt_diagnostics({s: normalized_path(s) for s in FILES}, GT_PATH, VALIDATION_GT_PATH)
    pd.DataFrame(gt_rows).to_csv(STAGE3_DIR / "gt_view_diagnostics.csv", index=False)
    profile["gt_diagnostics"] = gt_rows

    print("  token frequencies (train, unsupervised):")
    audit.token_frequencies({s: normalized_path(s) for s in FILES}).to_csv(
        STAGE3_DIR / "token_frequency_report.csv", index=False)

    return profile


# ============================================================
# REPORT
# ============================================================

def write_report(profile: dict, tests: list, runs: list):
    lines = ["STAGE 3 NORMALIZATION REPORT", "=" * 78, ""]

    lines.append(f"Unit tests: {sum(ok for _, ok in tests)}/{len(tests)} passed")
    lines += [f"  FAIL {name}" for name, ok in tests if not ok]

    if runs:
        lines += ["", "Normalization runs:"]
        lines += [f"  {r['file']:<22}{r['rows']:>11,} rows  {r['seconds']:>6}s  {r['output_mb']:>7} MB" for r in runs]

    lines += ["", "Raw preservation (full files; mismatches must be 0):"]
    for stem, f in profile["files"].items():
        rp = f["raw_preservation"]
        lines.append(f"  {stem:<16} rows_equal={rp['rows_equal']}  id={rp['entity_id_mismatches']}  "
                     f"name={rp['name_raw_mismatches']}  address={rp['address_raw_mismatches']}  "
                     f"country={rp['country_mismatches']}")

    lines += ["", f"Idempotence (normalize(view) == view, {PROPERTY_SAMPLE:,}-row sample per file; must be 0):"]
    for stem, f in profile["files"].items():
        bad = {k: v for k, v in f["idempotence"]["non_idempotent_rows"].items() if v}
        lines.append(f"  {stem:<16} {'all views idempotent' if not bad else bad}")

    lines += ["", "Unicode stability (views that are not already NFKC; full files; must be 0):"]
    for stem, f in profile["files"].items():
        bad = {k: v["not_nfkc"] for k, v in f["unicode_stability"].items() if v["not_nfkc"]}
        lines.append(f"  {stem:<16} {'all views stable' if not bad else bad}")

    lines += ["", "Information loss per view (raw non-empty -> view empty / digits lost):"]
    lines.append(f"  {'file':<16}{'view':<24}{'changed':>11}{'emptied':>9}{'digit_lost':>11}"
                 f"{'raw_dist':>11}{'view_dist':>11}{'collapse':>9}{'coll_rows':>11}{'digit_conf':>11}")
    for m in profile["view_metrics"]:
        lines.append(f"  {m['file']:<16}{m['view']:<24}{m['changed_vs_raw']:>11,}{m['became_empty']:>9,}"
                     f"{m['digit_lost']:>11,}{m['raw_distinct']:>11,}{m['view_distinct']:>11,}"
                     f"{m['collapse_ratio']:>9.4f}{m['rows_in_collisions']:>11,}{m['rows_in_digit_conflicts']:>11,}")

    lines += ["", "Ground-truth view diagnostics (TRAIN split only; validation untouched):",
              "  agreement = % of true S1<->S2/S3 pairs whose view is exactly equal (benefit)",
              "  S1 sharing = % of distinct S1 entities sharing the key with another S1 (ambiguity)",
              "  cross-entity = % of matched S2/S3 records whose key also belongs to another entity's record (false-merge risk)"]
    lines.append(f"  {'view':<46}{'agree%':>8}{'US%':>8}{'India%':>8}{'S1share%':>10}{'crossEnt%':>11}")
    for g in profile["gt_diagnostics"]:
        lines.append(f"  {g['view']:<46}{g['true_pair_exact_agreement_pct']:>8.2f}"
                     f"{g.get('agreement_pct_US', 0):>8.2f}{g.get('agreement_pct_India', 0):>8.2f}"
                     f"{g['s1_sharing_key_pct']:>10.3f}{g['targets_in_cross_entity_groups_pct']:>11.3f}")

    lines += ["", "Name script mix (% of rows):"]
    for stem, f in profile["files"].items():
        for country, s in f["scripts"].items():
            top = ", ".join(f"{k} {v}" for k, v in list(s["name_script_pct"].items())[:5])
            lines.append(f"  {stem:<16}{country:<8} {top}  | mixed {s['name_mixed_script_pct']}%")

    lines += ["", "See normalization_findings.md for the manual inspection and proposed rules."]

    text = "\n".join(lines)
    (STAGE3_DIR / "stage3_report.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="tests,normalize,audit")
    parser.add_argument("--force", action="store_true", help="rebuild existing normalized files")
    args = parser.parse_args()
    steps = set(args.steps.split(","))

    print("Unit tests:")
    tests = unit_tests()
    if not all(ok for _, ok in tests):
        raise SystemExit("Unit tests failed; not continuing.")

    runs = []
    if "normalize" in steps:
        print("\nNormalizing:")
        runs = normalize_all(args.force)
        STAGE3_DIR.mkdir(parents=True, exist_ok=True)
        if runs:
            (STAGE3_DIR / "normalization_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    if "audit" in steps:
        runs_file = STAGE3_DIR / "normalization_runs.json"
        if not runs and runs_file.exists():
            runs = json.loads(runs_file.read_text(encoding="utf-8"))
        print("\nAuditing:")
        profile = audit_and_save(tests, runs)
        write_report(profile, tests, runs)


def audit_and_save(tests, runs):
    profile = run_audit()
    profile["unit_tests"] = {name: ok for name, ok in tests}
    profile["normalization_runs"] = runs
    with open(STAGE3_DIR / "normalization_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
    return profile


if __name__ == "__main__":
    main()
