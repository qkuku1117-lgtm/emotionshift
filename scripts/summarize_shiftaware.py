"""Average per-seed ShiftAware reports into one summary per variant, next to B1.

    uv run scripts/summarize_shiftaware.py

Reads  reports/<variant>/<dataset>/seed<k>/ written by evaluate_shiftaware.py
Writes reports/<variant>/<dataset>/summary.json, stratified_<split>.csv, per_class_<split>.csv
       reports/shiftaware/results.md   one table with B1 and every ShiftAware variant
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
METRIC_KEYS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]


def mean_std(values) -> dict[str, float]:
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": values}


def summarize(variant_dir: Path, dataset: str) -> dict[str, Any] | None:
    seed_dirs = sorted(variant_dir.glob(f"{dataset}/seed*"), key=lambda p: int(p.name[4:]))
    runs = [json.loads((d / "summary.json").read_text()) for d in seed_dirs]
    if not runs:
        return None
    out = variant_dir / dataset
    summary: dict[str, Any] = {
        "dataset": dataset,
        "model": runs[0]["model"],
        "seeds": [r["seed"] for r in runs],
        "best_epochs": [r["best_epoch"] for r in runs],
        "commits": sorted({(r["git"] or {}).get("commit") or "unknown" for r in runs}),
        "dev": {k: mean_std(r["dev"][k] for r in runs) for k in METRIC_KEYS},
        "test": {},
    }
    for split in runs[0]["test"]:
        tests = [r["test"][split] for r in runs]
        result: dict[str, Any] = {k: mean_std(t[k] for t in tests) for k in METRIC_KEYS}
        result["n"] = tests[0]["n"]
        result["shift_head"] = {
            k: mean_std(t["shift_head"][k] for t in tests) for k in ("shift_f1", "auroc")
        }
        result["derived_shift_f1"] = mean_std(
            t["derived_shift_detection"]["shift_f1"] for t in tests
        )
        if all(t["vs_b1_seed0"] for t in tests):
            result["vs_b1_seed0"] = {
                scope: {
                    "diff": mean_std(t["vs_b1_seed0"][scope]["diff"] for t in tests),
                    "seeds_with_ci_above_zero": sum(
                        t["vs_b1_seed0"][scope]["ci_low"] > 0 for t in tests
                    ),
                }
                for scope in ("all", "shift_turns")
            }
        summary["test"][split] = result
        for table in ("stratified", "per_class"):
            key = "group" if table == "stratified" else "emotion"
            frames = [pd.read_csv(d / f"{table}_{split}.csv") for d in seed_dirs]
            pd.concat(frames).groupby(key, sort=False).mean().round(4).reset_index().to_csv(
                out / f"{table}_{split}.csv", index=False
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def cell(stat: dict) -> str:
    return f"{100 * stat['mean']:.2f} ± {100 * stat['std']:.2f}"


def shift_turn_f1(variant_dir: Path, dataset: str, split: str) -> str:
    table = pd.read_csv(variant_dir / dataset / f"stratified_{split}.csv").set_index("group")
    return f"{100 * table.loc['stratum:shift', 'weighted_f1']:.2f}"


def main():
    rows = []
    for dataset in ("meld", "dailydialog"):
        b1_path = REPORTS / "b1" / dataset / "summary.json"
        if b1_path.exists():
            b1 = json.loads(b1_path.read_text())
            for split, t in b1["test"].items():
                rows.append(
                    [dataset, split, "B1 TF-IDF", len(b1["seeds"])]
                    + [cell(t[k]) for k in ("weighted_f1", "macro_f1", "micro_f1_no_neutral")]
                    + [shift_turn_f1(REPORTS / "b1", dataset, split), "n/a"]
                )
        for variant_dir in sorted(REPORTS.glob("shiftaware*")):
            summary = summarize(variant_dir, dataset)
            if summary is None:
                continue
            for split, t in summary["test"].items():
                rows.append(
                    [dataset, split, summary["model"], len(summary["seeds"])]
                    + [cell(t[k]) for k in ("weighted_f1", "macro_f1", "micro_f1_no_neutral")]
                    + [shift_turn_f1(variant_dir, dataset, split), cell(t["shift_head"]["auroc"])]
                )
    header = [
        "dataset",
        "split",
        "model",
        "seeds",
        "weighted F1",
        "macro F1",
        "micro F1 (no neutral)",
        "weighted F1 on shift turns",
        "shift-head AUROC",
    ]
    lines = [
        "# ShiftAware results (test, mean ± std over seeds)",
        "",
        "Checkpoints and variants are selected on dev only. Shift-turn F1 is the seed mean.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]
    (REPORTS / "shiftaware").mkdir(exist_ok=True)
    (REPORTS / "shiftaware" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
