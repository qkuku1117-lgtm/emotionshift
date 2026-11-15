"""One command to finish B4 after the trainings: merge k = all, copy Drive outputs into the repo, make the tables.

    python scripts/finish_b4.py --drive /content/drive/MyDrive/b4

What it does, in order (each step is safe to repeat):
  1. reports which results exist on Drive (so a missing dataset or tag is visible at once);
  2. ``merge_b4_all.py``: adds the k = all results to each ``reports/b4/<dataset>/summary.json`` on Drive;
  3. copies ``reports/`` and ``predictions/`` from Drive into the repo (models are not copied);
  4. ``analyse_b4_ctx.py --compare b4all`` (or without ``--compare`` when no k = all run exists) and prints the overall result.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATASETS = ["meld", "dailydialog", "emowoz"]
TAGS = ["b4", "b4all", "b4plus", "b4allplus"]


def inventory(drive: Path) -> dict:
    """{tag: [datasets with a summary.json]} for the tags that matter."""
    return {t: [d for d in DATASETS if (drive / "reports" / t / d / "summary.json").exists()] for t in TAGS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="the --out folder used for training (Drive)")
    ap.add_argument("--note", default="k = all trained on a Colab A100; the session was switched from a T4 after the first "
                                      "MELD and DailyDialog sweep runs")
    a = ap.parse_args()
    drive = Path(a.drive)
    if not (drive / "reports").exists():
        sys.exit(f"{drive}/reports not found: is Drive mounted and is --drive the folder used as --out?")

    inv = inventory(drive)
    print("Results found on Drive (summary.json per tag):")
    for tag, ds in inv.items():
        print(f"  {tag:10s} {', '.join(ds) if ds else '-'}")
    if not inv["b4"]:
        sys.exit("no reports/b4/<dataset>/summary.json on Drive: nothing to analyse.")

    import merge_b4_all
    done = merge_b4_all.merge(drive, "b4", "b4all", DATASETS, a.note)
    print("merged k = all into:", ", ".join(done) or "nothing (no b4all summaries)")

    for sub in ("reports", "predictions"):
        if (drive / sub).exists():
            shutil.copytree(drive / sub, ROOT / sub, dirs_exist_ok=True)
    print("copied reports/ and predictions/ from Drive into the repo")

    cmd = [sys.executable, str(ROOT / "scripts" / "analyse_b4_ctx.py")]
    if inv["b4all"]:
        cmd += ["--compare", "b4all"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    out = ROOT / "reports" / "b4" / ("results_overall.md" if inv["b4all"] else "results_ctx.md")
    print(f"\n===== {out.relative_to(ROOT)} =====\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
