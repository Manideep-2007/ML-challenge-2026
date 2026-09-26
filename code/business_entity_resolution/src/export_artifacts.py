"""
Copy the selected model, decision rules and token translations into
code/business_entity_resolution/model_artifacts/ so the code folder alone can
regenerate the submission files (python -m src.main --split test).

    python src/export_artifacts.py
"""

from pathlib import Path
import json
import shutil

SRC = Path(__file__).resolve().parent
PACKAGE = SRC.parent
ROOT = SRC.parents[2]
TARGET = PACKAGE / "model_artifacts"
STAGE6 = ROOT / "experiments" / "stage6"
STAGE7 = ROOT / "experiments" / "stage7"
FINAL_TRANSLATIONS = ROOT / "artifacts" / "final" / "translations"

MODEL_FILES = ["config.json", "model.txt", "model.json", "calibrator.joblib", "metrics.json"]


def copy_model(model_id: str):
    source = STAGE6 / model_id
    target = TARGET / model_id
    target.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    config = json.loads((source / "config.json").read_text())
    for member in config.get("members", []):
        copy_model(member)


def main():
    decision = json.loads((STAGE7 / "decision_params.json").read_text())
    TARGET.mkdir(parents=True, exist_ok=True)
    copy_model(decision["model"])
    shutil.copy2(STAGE7 / "decision_params.json", TARGET / "decision_params.json")
    if (FINAL_TRANSLATIONS / "token_translation_name.csv").exists():
        shutil.copytree(FINAL_TRANSLATIONS, TARGET / "translations", dirs_exist_ok=True)
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(PACKAGE)}  {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
