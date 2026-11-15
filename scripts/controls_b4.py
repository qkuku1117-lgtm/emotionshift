"""E4 input controls for B4 (proposal E4): the trained model, a corrupted test input. GPU or CPU, minutes.

    python scripts/controls_b4.py --datasets meld dailydialog emowoz
    python scripts/controls_b4.py --datasets meld --tag b4 --out /content/drive/MyDrive/b4

Loads the final model of each dataset (``models/<tag>_<dataset>``, the best-on-dev seed at the chosen k)
and scores its headline test split with: the real history, no history, a shuffled history (same number of
turns from random other dialogues) and shuffled speaker names. If real context really matters (H2) the
shuffled history scores worse than no history and speaker shuffling changes little.

Writes ``reports/<tag>/<dataset>/controls.csv``. One model per dataset (the saved seed), so the scores are
single-seed: compare the rows with each other, not with the 5-seed means in summary.json.
Design of E4 and of the original MELD/DailyDialog controls: Lixian (docs/B4_CONTEXT.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.context import parse_k  # noqa: E402
from emotionshift.data.labels import emotions  # noqa: E402
from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.eval.controls import score_controls  # noqa: E402


def load_model(path):
    from emotionshift.models.roberta_ctx import RobertaContext
    return RobertaContext.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["meld", "dailydialog", "emowoz"])
    ap.add_argument("--tag", default="b4")
    ap.add_argument("--out", default=str(ROOT), help="output root used for train_b4.py (Drive on Colab)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the shuffles")
    a = ap.parse_args()
    out = Path(a.out)
    for ds in a.datasets:
        summary = json.loads((out / "reports" / a.tag / ds / "summary.json").read_text())
        k, split = parse_k(summary["chosen_k"]), next(iter(summary["test"]))
        frame = load(ds)
        frame = frame[frame.split == split].reset_index(drop=True)
        model = load_model(out / "models" / f"{a.tag}_{ds}")
        df = score_controls(model, frame, k, emotions(ds), M.HEADLINE[ds], a.seed)
        df.round(4).to_csv(out / "reports" / a.tag / ds / "controls.csv", index=False)
        print(f"[{ds}] k={summary['chosen_k']} {split}")
        print(df[df.group == "all"][["input", "metric", "score"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
