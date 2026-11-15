"""E4 input controls: one trained model scored with its test input changed (see ``data/context.py``)."""

from __future__ import annotations

import pandas as pd

from ..data.context import CONTROLS, build_inputs, control_inputs
from . import metrics as M

GROUPS = ["all", "stratum:first", "stratum:stable", "stratum:shift"]


def score_controls(model, frame: pd.DataFrame, k, labels, headline: str, seed: int = 0) -> pd.DataFrame:
    """One row per (input, group): the headline metric of ``model`` on ``frame`` under each control.

    ``frame`` is one split of a loader table (EmoWOZ: with its system turns), ``model.proba_inputs`` maps
    ready-made inputs to probabilities. Inputs: real history, then ``CONTROLS`` (none, shuffled_context,
    shuffled_speakers).
    """
    frame = frame.reset_index(drop=True)
    rows = []
    for name in ["real", *CONTROLS]:
        t = build_inputs(frame, k) if name == "real" else control_inputs(frame, k, name, seed)
        t = t.copy()
        t["pred"] = [labels[i] for i in model.proba_inputs(t["input"]).argmax(1)]
        s = M.stratified(t, labels=labels).set_index("group")
        rows += [{"input": name, "group": g, "n": int(s.loc[g, "n"]), "metric": headline, "score": float(s.loc[g, headline])}
                 for g in GROUPS if g in s.index]
    return pd.DataFrame(rows)


def aggregate(per_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean and sample SD over seeds of ``score`` for every (input, group)."""
    t = pd.concat([d.assign(seed=i) for i, d in enumerate(per_seed)])
    g = t.groupby(["input", "group"], sort=False)
    out = g.agg(n=("n", "first"), metric=("metric", "first"), mean=("score", "mean"),
                std=("score", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0), seeds=("score", "size"))
    return out.reset_index()
