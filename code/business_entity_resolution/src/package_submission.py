"""
Build the final submission archive:

    <team_name>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, model_artifacts/, README.md, requirements.txt}
    └── Documentation_template.md

    python src/package_submission.py --team "<team name>"

Runs the official validator on the output files first and refuses to package
if it fails.
"""

from pathlib import Path
import argparse
import re
import subprocess
import sys
import zipfile

SRC = Path(__file__).resolve().parent
PACKAGE = SRC.parent
ROOT = SRC.parents[2]
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".log"}


def validate() -> None:
    """Official validator on matching_results.tsv (with --check-ids); candidate_pairs.tsv
    (~200M IDs, too large for the official validator's in-memory sets) is checked by
    main.check_candidate_tsv, which applies the same rules streamed."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "challenge" / "utils" / "validate_submission.py"),
         "--matching", str(ROOT / "output" / "matching_results.tsv"),
         "--candidate", str(ROOT / "output" / "__skip__.tsv"),
         "--test-dir", str(ROOT / "challenge" / "dataset" / "test"), "--check-ids"],
        capture_output=True, text=True)
    print(result.stdout, result.stderr)
    if result.returncode != 0:
        raise SystemExit("Validator failed; not packaging.")
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from main import check_candidate_tsv
    s1 = set(pd.read_csv(ROOT / "challenge" / "dataset" / "test" / "test_source1.tsv", sep="	", dtype=str,
                         usecols=["entity_id"], keep_default_na=False)["entity_id"])
    check = check_candidate_tsv(ROOT / "output" / "candidate_pairs.tsv", s1, ROOT / "output" / "matching_results.tsv")
    print("candidate_pairs.tsv check:", check)
    if not check["ok"]:
        raise SystemExit("candidate_pairs.tsv check failed; not packaging.")


def code_files():
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if path.is_file() and not (set(rel.parts) & EXCLUDE_DIRS) and path.suffix not in EXCLUDE_SUFFIXES:
            yield path, Path("code") / "business_entity_resolution" / rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    args = parser.parse_args()
    validate()

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.team).strip("_") or "team"
    target = ROOT / f"{name}_submission.zip"
    files = [
        (ROOT / "output" / "matching_results.tsv", Path("output") / "matching_results.tsv"),
        (ROOT / "output" / "candidate_pairs.tsv", Path("output") / "candidate_pairs.tsv"),
        (ROOT / "Documentation_template.md", Path("Documentation_template.md")),
        *code_files(),
    ]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arcname in files:
            archive.write(source, arcname.as_posix())
    print(f"{target}  ({target.stat().st_size / 2**20:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
