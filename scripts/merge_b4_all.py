"""Add the k = all run to the existing B4 summaries (no new result files to hand in).

    python scripts/merge_b4_all.py --root /content/drive/MyDrive/b4 --note "k = all trained on a Colab A100 (the session was switched from a T4 after the main MELD / DailyDialog runs)"

For each dataset it reads ``reports/b4all/<dataset>/summary.json`` (written by ``train_b4.py --k all --tag b4all``)
and stores it inside ``reports/b4/<dataset>/summary.json`` under the key ``"k_all"``, together with the note; the
controls file becomes ``reports/b4/<dataset>/controls_k_all.csv``. Everything else in the summary is left as it is, and running
it twice gives the same file. The k = all predictions stay in ``predictions/b4all_*`` on purpose:
``scripts/summarize_all.py`` reads ``predictions/b4_*`` as the chosen-window results.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["chosen_k", "chosen_by", "config", "seeds", "dev", "test", "efficiency", "best_seed_on_dev"]


def merge(root: Path, tag: str, other: str, datasets: list[str], note: str) -> list[str]:
    done = []
    for ds in datasets:
        main_f = root / "reports" / tag / ds / "summary.json"
        src_f = root / "reports" / other / ds / "summary.json"
        if not main_f.exists() or not src_f.exists():
            continue
        main, src = json.loads(main_f.read_text()), json.loads(src_f.read_text())
        main["k_all"] = {**{k: src[k] for k in KEEP if k in src}, "tag": other, "note": note,
                         "predictions": f"predictions/{other}_{ds}_<split>_seed<k>.csv"}
        main_f.write_text(json.dumps(main, indent=1, default=float))
        ctl = root / "reports" / other / ds / "controls.csv"
        if ctl.exists():
            shutil.copy(ctl, root / "reports" / tag / ds / "controls_k_all.csv")
        done.append(ds)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b4")
    ap.add_argument("--other", default="b4all")
    ap.add_argument("--datasets", nargs="+", default=["meld", "dailydialog", "emowoz"])
    ap.add_argument("--root", default=str(ROOT), help="folder holding reports/ (the Drive --out folder on Colab)")
    ap.add_argument("--note", default="k = all trained on a Colab A100 (GPU switched after the main runs)")
    a = ap.parse_args()
    done = merge(Path(a.root), a.tag, a.other, a.datasets, a.note)
    print("merged k = all into:", ", ".join(done) or "nothing (no b4all summaries found)")


if __name__ == "__main__":
    main()
