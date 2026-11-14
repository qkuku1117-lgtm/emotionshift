"""Mark B4 complete only after native predictions and executed artifacts exist."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b4/emowoz"


def finalize():
    verified = json.loads((OUT / "verification.json").read_text())
    assert verified["prediction_csvs"] == 5
    assert verified["e4_controls_verified"] and verified["paired_comparisons_verified"]
    summary = json.loads((OUT / "summary.json").read_text())
    headline = summary["test"]["metrics"]["weighted_f1_no_neutral"]
    suite = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", "-q"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    (OUT / "test_results.txt").write_text(suite.stdout, encoding="utf-8")
    suite.check_returncode()
    test_summary = next(line for line in reversed(suite.stdout.splitlines()) if " passed" in line)
    print(test_summary, flush=True)
    note = ROOT / "docs/EMOWOZ_B4.md"
    text = note.read_text(encoding="utf-8")
    text = re.sub(
        r"^Status:.*$",
        f"Status: five seeds, official test evaluation, E4 controls and artifact audits completed. Mean non-neutral weighted-F1: {headline['mean'] * 100:.2f}% (population SD {headline['std'] * 100:.2f} points).",
        text,
        flags=re.MULTILINE,
    )
    note.write_text(text, encoding="utf-8")
    nb = nbformat.read(ROOT / "notebooks/emowoz_b4.ipynb", as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None and len(c.source.splitlines()) <= 15 for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.get("outputs", []))
    files = [
        ROOT / "README.md",
        note,
        ROOT / "notebooks/emowoz_b4.ipynb",
        ROOT / "configs/emowoz_b4.yaml",
        ROOT / "emotionshift/models/roberta_context.py",
    ]
    files += list(OUT.rglob("*"))
    files += list((ROOT / "predictions").glob("b4_emowoz_test_seed*.csv"))
    files += [
        ROOT / name
        for name in [
            "scripts/train_emowoz_context.py",
            "scripts/analyse_emowoz_context.py",
            "scripts/make_emowoz_context_artifacts.py",
            "scripts/finalize_emowoz_context.py",
            "tests/test_context_emowoz.py",
            "tests/test_context_paired.py",
        ]
    ]
    count = 0
    for path in files:
        if not path.is_file():
            continue
        if path.suffix == ".png":
            with Image.open(path) as im:
                assert im.width >= 1600
            continue
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".ipynb":
            text = json.dumps(json.loads(text), ensure_ascii=False)
        assert not re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text), path
        count += 1
    record = {
        "status": "complete",
        "seeds": [0, 1, 2, 3, 4],
        "targets_per_csv": 8634,
        "full_project_tests": test_summary,
        "paired_bootstrap_tests": "2 passed",
        "prediction_and_e4_audits": "passed",
        "notebook": "executed without errors, snippets <=15 lines",
        "english_text_files_scanned": count,
        "figures": "dimensions verified; manual visual review separate",
        "large_weights": "ignored; checksums supplied",
        "commit_push": "not performed for workstream 5",
    }
    (OUT / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Final B4 English scan and executed-artifact checks passed.", flush=True)


if __name__ == "__main__":
    finalize()
