"""ERC metrics, overall and per stratum, with dialogue-level bootstrap intervals.

Weighted-F1 is the standard MELD metric and micro-F1 without neutral the
standard DailyDialog metric; macro-F1 exposes the minority classes.
"""

import numpy as np
import pandas as pd

from emotionshift.data.datasets import EMOTIONS
from emotionshift.data.shifts import SHIFT_TYPES, STRATA

METRICS = ("accuracy", "weighted_f1", "macro_f1", "micro_f1_no_neutral")


def _confusion(y_true, y_pred, labels):
    index = {lab: i for i, lab in enumerate(labels)}
    t = np.array([index[v] for v in y_true])
    p = np.array([index[v] for v in y_pred])
    k = len(labels)
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def scores_from_confusion(cm, labels=EMOTIONS):
    """All METRICS from a confusion matrix (rows = gold, columns = predicted)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * tp / (support + predicted))
    present = support > 0
    emo = np.array([lab != "neutral" for lab in labels])
    tp_e, pred_e, sup_e = tp[emo].sum(), predicted[emo].sum(), support[emo].sum()
    micro = 2 * tp_e / (pred_e + sup_e) if pred_e + sup_e else 0.0
    return {
        "accuracy": tp.sum() / cm.sum(),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "macro_f1": float(f1[present].mean()),
        "micro_f1_no_neutral": float(micro),
    }


def scores(y_true, y_pred, labels=EMOTIONS):
    """METRICS for one set of predictions."""
    return scores_from_confusion(_confusion(list(y_true), list(y_pred), labels), labels)


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
    dialogues = pd.factorize(df["dialogue_id"])[0]
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
