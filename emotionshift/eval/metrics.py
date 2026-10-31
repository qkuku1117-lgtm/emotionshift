"""Metrics shared by every model (proposal: "Evaluation" section).

* MELD headline: weighted-F1 over the 7 classes.
* DailyDialog headline: micro-F1 *excluding neutral* (standard in the literature).
* EmoWOZ headline: weighted-F1 *excluding neutral* over its own seven labels,
  with macro-F1 excluding neutral alongside (Feng et al., 2022). Score only
  labelled (user) turns: pass ``labels=emotions("emowoz")``.
* Always also: macro-F1, per-class F1, confusion matrix.
* Stratified: the same metrics per stratum (first / stable / shift) and per
  shift type, with ``n`` in every cell.
* 95% bootstrap CIs, resampling whole dialogues (utterances in one dialogue
  are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from ..data.labels import EMOTIONS
from ..data.shifts import SHIFT_TYPES, STRATA

HEADLINE = {
    "meld": "weighted_f1",
    "dailydialog": "micro_f1_no_neutral",
    "emowoz": "weighted_f1_no_neutral",
}


METRICS = (
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "micro_f1_no_neutral",
    "weighted_f1_no_neutral",
    "macro_f1_no_neutral",
)


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
        "weighted_f1_no_neutral": (float((f1[emo] * support[emo]).sum() / sup_e) if sup_e else 0.0),
        "macro_f1_no_neutral": float(f1[emo].mean()),
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
            "weighted_f1_no_neutral": 0.0,
            "macro_f1_no_neutral": 0.0,
            "accuracy": float("nan"),
        }
    emotional = [e for e in labels if e != "neutral"]
    return {
        "n": len(gold),
        "weighted_f1": f1_score(gold, pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "micro_f1_no_neutral": f1_score(
            gold, pred, labels=emotional, average="micro", zero_division=0
        ),
        "weighted_f1_no_neutral": f1_score(
            gold, pred, labels=emotional, average="weighted", zero_division=0
        ),
        "macro_f1_no_neutral": f1_score(
            gold, pred, labels=emotional, average="macro", zero_division=0
        ),
        "accuracy": float((gold == pred).mean()) if len(gold) else float("nan"),
    }


def per_class_f1(gold, pred, labels=EMOTIONS) -> pd.DataFrame:
    f1 = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    support = pd.Series(gold).value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"emotion": labels, "f1": f1, "support": support.values})


def confusion(gold, pred, labels=EMOTIONS) -> pd.DataFrame:
    cm = confusion_matrix(gold, pred, labels=labels)
    return pd.DataFrame(
        cm, index=[f"gold_{e}" for e in labels], columns=[f"pred_{e}" for e in labels]
    )


def stratified(df: pd.DataFrame, pred_col: str = "pred", labels=EMOTIONS) -> pd.DataFrame:
    """Scores per stratum and per shift type. ``df`` needs emotion, stratum,
    shift_type and ``pred_col``."""
    rows = [{"group": "all", **scores(df["emotion"], df[pred_col], labels)}]
    for s in STRATA:
        sub = df[df["stratum"] == s]
        rows.append({"group": f"stratum:{s}", **scores(sub["emotion"], sub[pred_col], labels)})
    for t in SHIFT_TYPES:
        sub = df[df["shift_type"] == t]
        rows.append({"group": f"shift_type:{t}", **scores(sub["emotion"], sub[pred_col], labels)})
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


def _auroc(gold, score) -> float | None:
    gold = np.asarray(gold)
    return float(roc_auc_score(gold, score)) if len(np.unique(gold)) == 2 else None


def forecast_scores(df: pd.DataFrame, train: pd.DataFrame, labels=EMOTIONS) -> dict:
    """Task T3: the speaker's next emotion, and whether it shifts, predicted
    *before* their utterance is read. Scored on turns with a previous
    same-speaker labelled turn.

    ``df`` needs emotion, prev_emotion, stratum, pred (the model's recognition),
    forecast_pred, forecast_shift_prob and forecast_p_<label> per label.
    Baselines from label history alone:

    * persistence: repeat the previous turn's emotion, as recognized by the
      model (``_pred``, causal) or gold (``_gold``, an oracle);
    * transition: P(emotion | previous emotion) and P(shift | previous
      emotion), estimated on ``train`` and read with the previous predicted
      or gold emotion.
    """
    d = df.sort_values(["split", "dialogue_id", "turn"]).copy()
    d["prev_pred"] = d.groupby(["split", "dialogue_id", "speaker"], sort=False)["pred"].shift(1)
    d = d[d["stratum"].isin(("stable", "shift"))]
    t = train[train["stratum"].isin(("stable", "shift"))]
    counts = pd.crosstab(t["prev_emotion"], t["emotion"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    marginal = t["emotion"].value_counts(normalize=True).reindex(labels, fill_value=0)
    # One pseudo-count spread by the marginal: unseen previous emotions fall back to it.
    transition = counts.add(marginal, axis=1)
    transition = transition.div(transition.sum(axis=1), axis=0)
    stay = pd.Series(np.diag(transition), index=labels)

    shift = (d["stratum"] == "shift").astype(int)
    result: dict = {
        "n": len(d),
        "base_rate": float(shift.mean()),
        "shift_auroc": {
            "forecast": _auroc(shift, d["forecast_shift_prob"]),
            "transition_pred": _auroc(shift, 1 - d["prev_pred"].map(stay)),
            "transition_gold": _auroc(shift, 1 - d["prev_emotion"].map(stay)),
        },
        "emotion": {
            "forecast": scores(d["emotion"], d["forecast_pred"], labels),
            "persistence_pred": scores(d["emotion"], d["prev_pred"], labels),
            "persistence_gold": scores(d["emotion"], d["prev_emotion"], labels),
        },
        "class_auroc": {},
    }
    for label in labels:
        if label == "neutral":
            continue
        gold = (d["emotion"] == label).astype(int)
        result["class_auroc"][label] = {
            "n": int(gold.sum()),
            "forecast": _auroc(gold, d[f"forecast_p_{label}"]),
            "transition_pred": _auroc(gold, d["prev_pred"].map(transition[label])),
            "transition_gold": _auroc(gold, d["prev_emotion"].map(transition[label])),
        }
    return result


def bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    pred_col: str = "pred",
    n_boot: int = 1000,
    seed: int = 0,
    labels=EMOTIONS,
) -> tuple[float, float]:
    """95% percentile CI of ``metric``, resampling dialogues with replacement."""
    rng = np.random.default_rng(seed)
    scope = [c for c in ("dataset", "split") if c in df.columns]
    idx_by_key = list(df.groupby([*scope, "dialogue_id"], sort=False).indices.values())
    gold, pred = df["emotion"].to_numpy(), df[pred_col].to_numpy()
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_key), len(idx_by_key))
        ix = np.concatenate([idx_by_key[p] for p in pick])
        vals.append(scores(gold[ix], pred[ix], labels)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stratified_scores(df, pred_col, label_col="emotion", labels=EMOTIONS):
    """METRICS overall, per stratum and per shift type; needs `add_shift_columns` output."""
    groups = [("all", df)]
    groups += [(s, df[df["stratum"] == s]) for s in STRATA]
    groups += [(f"shift:{t}", df[df["shift_type"] == t]) for t in SHIFT_TYPES]
    rows = []
    for name, part in groups:
        if len(part):
            rows.append(
                {"group": name, "n": len(part), **scores(part[label_col], part[pred_col], labels)}
            )
    return pd.DataFrame(rows).set_index("group")


def paired_bootstrap(
    df,
    pred_a,
    pred_b,
    metric="weighted_f1",
    mask=None,
    label_col="emotion",
    labels=EMOTIONS,
    n_boot=1000,
    seed=0,
):
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
