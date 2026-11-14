"""Native paired dialogue-bootstrap comparisons for B4, B3 and E4 controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emotionshift.data.labels import emotions
from emotionshift.eval import metrics as M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b4/emowoz"
LABELS = emotions("emowoz")
METRIC = M.HEADLINE["emowoz"]
KEYS = ["dataset", "split", "dialogue_id", "turn", "emotion", "stratum"]


def paired_delta(reference, candidate, group="all", n_boot=1000, seed=0):
    """CI conditional on the fixed seed pairs; resample shared whole dialogues."""
    assert len(reference) == len(candidate) > 0
    keys = reference[0][KEYS].astype(str).to_numpy()
    for frame in [*reference, *candidate]:
        if not np.array_equal(keys, frame[KEYS].astype(str).to_numpy()):
            raise ValueError("Paired targets are not aligned")
    mask = (
        np.ones(len(reference[0]), dtype=bool)
        if group == "all"
        else reference[0].stratum.eq(group).to_numpy()
    )
    if not mask.any():
        raise ValueError("Empty evaluation group")
    target = reference[0].loc[mask]
    codes, unique = pd.factorize(target.dialogue_id, sort=False)
    index = {label: i for i, label in enumerate(LABELS)}
    k = len(LABELS)
    gold = target.emotion.map(index).to_numpy()

    def matrices(frames):
        result = []
        for frame in frames:
            pred = frame.loc[mask, "pred"].map(index).to_numpy()
            assert not pd.isna(pred).any()
            result.append(
                np.bincount(codes * k * k + gold * k + pred, minlength=len(unique) * k * k).reshape(
                    len(unique), k, k
                )
            )
        return np.stack(result)

    left, right = matrices(reference), matrices(candidate)

    def scores(matrix):
        return np.array([M.scores_from_confusion(c, LABELS)[METRIC] for c in matrix])

    point_left = scores(left.sum(axis=1))
    point_right = scores(right.sum(axis=1))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        deltas.append(
            float((scores(right[:, pick].sum(axis=1)) - scores(left[:, pick].sum(axis=1))).mean())
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "group": group,
        "n": len(target),
        "dialogues": len(unique),
        "metric": METRIC,
        "reference_mean": float(point_left.mean()),
        "candidate_mean": float(point_right.mean()),
        "mean_delta": float((point_right - point_left).mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def analyse():
    b3 = [
        pd.read_csv(ROOT / f"predictions/b3_emowoz_test_seed{s}.csv", keep_default_na=False)
        for s in range(5)
    ]
    b4 = [
        pd.read_csv(ROOT / f"predictions/b4_emowoz_test_seed{s}.csv", keep_default_na=False)
        for s in range(5)
    ]
    cfg = json.loads((OUT / "manifest.json").read_text())["config"]
    controls = pd.read_csv(OUT / "controls_per_seed.csv").set_index(["seed", "mode"])
    assert len(controls) == 15
    for seed, frame in enumerate(b4):
        for mode in cfg["controls"]:
            assert frame[f"pred_{mode}"].isin(LABELS).all()
            for metric, value in M.scores(frame.emotion, frame[f"pred_{mode}"], LABELS).items():
                assert np.isclose(value, controls.loc[(seed, mode), metric]), (seed, mode, metric)
    comparisons = []
    for group in ["all", "first", "stable", "shift"]:
        comparisons.append(
            {
                "comparison": "b4_minus_b3",
                **paired_delta(b3, b4, group=group, n_boot=cfg["bootstrap"]),
            }
        )
    for mode in cfg["controls"]:
        altered = [frame.assign(pred=frame[f"pred_{mode}"]) for frame in b4]
        comparisons.append(
            {
                "comparison": f"real_minus_{mode}",
                **paired_delta(altered, b4, n_boot=cfg["bootstrap"]),
            }
        )
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    rows = []
    for mode in cfg["controls"]:
        table = controls.xs(mode, level="mode")
        rows.append(
            {
                "mode": mode,
                "n": int(table.n.iloc[0]),
                **{
                    f"{metric}_{stat}": float(
                        table[metric].mean() if stat == "mean" else table[metric].std(ddof=0)
                    )
                    for metric in M.METRICS
                    for stat in ["mean", "std"]
                },
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "controls_summary.csv", index=False)
    (OUT / "paired_protocol.json").write_text(
        json.dumps(
            {
                "unit": "whole dialogue, shared between models and seed pairs",
                "fixed_seeds": [0, 1, 2, 3, 4],
                "multiplicity": "exploratory unadjusted intervals",
                "resamples": cfg["bootstrap"],
                "seed": 0,
                "statistic": "mean of per-seed metric differences; no seed resampling",
                "window": "fixed from MELD dev; test does not select configurations",
                "limitations": "B4 vs B3 also changes speaker tags and token budget. E4 controls are inference-time perturbations, not retrained models.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Verified E4 controls and paired native-label dialogue comparisons.", flush=True)


if __name__ == "__main__":
    analyse()
