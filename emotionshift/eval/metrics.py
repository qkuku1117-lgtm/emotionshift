"""Metrics shared by every model (proposal: "Evaluation" section).

* MELD headline: weighted-F1 over the 7 classes.
* DailyDialog headline: micro-F1 *excluding neutral* (standard in the literature).
* Always also: macro-F1, per-class F1, confusion matrix.
* Stratified: the same metrics per stratum (first / stable / shift) and per
  shift type, with ``n`` in every cell.
* 95% bootstrap CIs, resampling whole dialogues (utterances in one dialogue
  are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from ..data.labels import EMOTIONS
from ..data.shifts import SHIFT_TYPES, STRATA

NON_NEUTRAL = [e for e in EMOTIONS if e != "neutral"]
HEADLINE = {"meld": "weighted_f1", "dailydialog": "micro_f1_no_neutral"}


METRICS = ("accuracy", "weighted_f1", "macro_f1", "micro_f1_no_neutral")


def _confusion(y_true, y_pred, labels):
    index = {lab: i for i, lab in enumerate(labels)}
    t = np.array([index[v] for v in y_true], dtype=int)
    p = np.array([index[v] for v in y_pred], dtype=int)
    k = len(labels)
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def scores_from_confusion(cm, labels=EMOTIONS):
    """All METRICS from a confusion matrix (rows = gold, columns = predicted)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * tp / (support + predicted))
    emo = np.array([lab != "neutral" for lab in labels])
    tp_e, pred_e, sup_e = tp[emo].sum(), predicted[emo].sum(), support[emo].sum()
    micro = 2 * tp_e / (pred_e + sup_e) if pred_e + sup_e else 0.0
    return {
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else float("nan"),
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "micro_f1_no_neutral": float(micro),
    }


def scores(gold, pred, labels=EMOTIONS) -> dict:
    gold, pred = np.asarray(gold), np.asarray(pred)
    # Some strata have no rows; newer sklearn versions reject empty inputs.
    if len(gold) == 0 and len(pred) == 0:
        return {
            "n": 0,
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "micro_f1_no_neutral": 0.0,
            "accuracy": float("nan"),
        }
    return {
        "n": len(gold),
        "weighted_f1": f1_score(gold, pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "micro_f1_no_neutral": f1_score(gold, pred, labels=[e for e in labels if e != "neutral"], average="micro", zero_division=0),
        "accuracy": float((gold == pred).mean()) if len(gold) else float("nan"),
    }


def per_class_f1(gold, pred) -> pd.DataFrame:
    f1 = f1_score(gold, pred, labels=EMOTIONS, average=None, zero_division=0)
    support = pd.Series(gold).value_counts().reindex(EMOTIONS, fill_value=0)
    return pd.DataFrame({"emotion": EMOTIONS, "f1": f1, "support": support.values})


def confusion(gold, pred) -> pd.DataFrame:
    cm = confusion_matrix(gold, pred, labels=EMOTIONS)
    return pd.DataFrame(cm, index=[f"gold_{e}" for e in EMOTIONS], columns=[f"pred_{e}" for e in EMOTIONS])


def stratified(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Scores per stratum and per shift type. ``df`` needs emotion, stratum,
    shift_type and ``pred_col``."""
    rows = [{"group": "all", **scores(df["emotion"], df[pred_col])}]
    for s in STRATA:
        sub = df[df["stratum"] == s]
        rows.append({"group": f"stratum:{s}", **scores(sub["emotion"], sub[pred_col])})
    for t in SHIFT_TYPES:
        sub = df[df["shift_type"] == t]
        rows.append({"group": f"shift_type:{t}", **scores(sub["emotion"], sub[pred_col])})
    return pd.DataFrame(rows)


def derived_shift_detection(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Task T2, *derived* variant: a predicted shift is a predicted label that
    differs from the speaker's previous *predicted* label (causal, no gold
    labels used). Scored on turns that have a previous same-speaker turn."""
    d = df.sort_values(["split", "dialogue_id", "turn"])
    prev_pred = d.groupby(["split", "dialogue_id", "speaker"], sort=False)[pred_col].shift(1)
    elig = d["stratum"] != "first"
    gold = (d.loc[elig, "stratum"] == "shift").astype(int)
    pred = (d.loc[elig, pred_col] != prev_pred[elig]).astype(int)
    return {
        "n": int(elig.sum()),
        "base_rate": float(gold.mean()),
        "shift_f1": f1_score(gold, pred, zero_division=0),
        "predicted_shift_rate": float(pred.mean()),
    }


def bootstrap_ci(df: pd.DataFrame, metric: str, pred_col: str = "pred",
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of ``metric``, resampling dialogues with replacement."""
    rng = np.random.default_rng(seed)
    scope = [c for c in ("dataset", "split") if c in df.columns]
    idx_by_key = list(df.groupby([*scope, "dialogue_id"], sort=False).indices.values())
    gold, pred = df["emotion"].to_numpy(), df[pred_col].to_numpy()
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_key), len(idx_by_key))
        ix = np.concatenate([idx_by_key[p] for p in pick])
        vals.append(scores(gold[ix], pred[ix])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stratified_scores(df, pred_col, label_col="emotion", labels=EMOTIONS):
    """METRICS overall, per stratum and per shift type; needs `add_shift_columns` output."""
    groups = [("all", df)]
    groups += [(s, df[df["stratum"] == s]) for s in STRATA]
    groups += [(f"shift:{t}", df[df["shift_type"] == t]) for t in SHIFT_TYPES]
    rows = []
    for name, part in groups:
        if len(part):
            rows.append({"group": name, "n": len(part),
                         **scores(part[label_col], part[pred_col], labels)})
    return pd.DataFrame(rows).set_index("group")


def paired_bootstrap(df, pred_a, pred_b, metric="weighted_f1", mask=None, label_col="emotion",
                     labels=EMOTIONS, n_boot=1000, seed=0):
    """Difference metric(b) - metric(a) with a 95% CI, resampling whole dialogues.

    Resampling dialogues rather than utterances respects that turns of one
    conversation are not independent. `mask` (boolean Series aligned with `df`)
    restricts scoring to a subset such as shift turns.
    """
    keep = np.ones(len(df), bool) if mask is None else np.asarray(mask, bool)
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    gold = df[label_col].map(index).to_numpy()
    codes_a = df[pred_a].map(index).to_numpy()
    codes_b = df[pred_b].map(index).to_numpy()
    scope = [c for c in ("dataset", "split") if c in df.columns]
    dialogues = pd.factorize(pd.MultiIndex.from_frame(df[[*scope, "dialogue_id"]]))[0]
    rows_of = [np.flatnonzero((dialogues == d) & keep) for d in range(dialogues.max() + 1)]

    def metric_of(rows, codes):
        cm = np.bincount(gold[rows] * k + codes[rows], minlength=k * k).reshape(k, k)
        return scores_from_confusion(cm, labels)[metric]

    observed_rows = np.flatnonzero(keep)
    observed = metric_of(observed_rows, codes_b) - metric_of(observed_rows, codes_a)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(rows_of), len(rows_of))
        rows = np.concatenate([rows_of[d] for d in picked])
        diffs.append(metric_of(rows, codes_b) - metric_of(rows, codes_a))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": observed, "ci_low": low, "ci_high": high, "n": int(keep.sum())}
