"""Pilot statistics behind the EmotionShift proposal.

From the group-project folder, run `bash scripts/fetch_data.sh`, then
`uv run python pilot/pilot_stats.py`.

A *shift* (or *flip*) is an utterance whose gold emotion differs from the same
speaker's previous utterance in the same dialogue. A speaker's first utterance
has no previous turn and is neither a shift nor stable. This matches the flip
definition of MELD-FR (SemEval-2024 Task 10): the counts agree split by split.

Baselines marked "oracle" read the previous *gold* label, so they are reference
points for how much signal emotional inertia carries, not deployable models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
EKMAN = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
VALENCE = {"neutral": "neu", "joy": "pos", "surprise": "amb", "anger": "neg",
           "sadness": "neg", "disgust": "neg", "fear": "neg"}
DD_LABELS = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
MAIN_SIX = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]


def add_history(df, dialogue, speaker, emotion):
    """Add the same speaker's previous emotion and the shift flag."""
    df = df.copy()
    df["prev_same"] = df.groupby([dialogue, speaker])[emotion].shift(1)
    df["eligible"] = df["prev_same"].notna()
    df["shift"] = df["eligible"] & (df[emotion] != df["prev_same"])
    return df


def shift_type(prev, curr):
    """Coarse taxonomy of a shift between two different emotions."""
    if prev == "neutral":
        return "onset"
    if curr == "neutral":
        return "offset"
    pair = {VALENCE[prev], VALENCE[curr]}
    if pair == {"pos", "neg"}:
        return "valence_flip"
    if pair == {"neg"}:
        return "within_negative"
    return "surprise_involved"


def load_meld(split):
    df = pd.read_csv(DATA / "meld" / f"{split}_sent_emo.csv")
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    return add_history(df, "Dialogue_ID", "Speaker", "Emotion")


def load_dailydialog(split):
    raw = pd.read_parquet(DATA / "dailydialog" / f"{split}.parquet")
    rows = [
        (d, t, t % 2, text, DD_LABELS[e])  # speakers strictly alternate A/B
        for d, r in raw.iterrows()
        for t, (text, e) in enumerate(zip(r.utterances, r.emotions))
    ]
    df = pd.DataFrame(rows, columns=["dialogue", "turn", "speaker", "text", "emotion"])
    return raw, add_history(df, "dialogue", "speaker", "emotion")


def fmt_scores(y, pred, labels=None, average="weighted"):
    return f"F1({average})={f1_score(y, pred, labels=labels, average=average):.3f}"


def meld_report():
    splits = {s: load_meld(s) for s in ["train", "dev", "test"]}
    print("=== MELD ===")
    for name, df in splits.items():
        n_shift, n_elig = df["shift"].sum(), df["eligible"].sum()
        print(f"{name}: {len(df)} utts / {df.Dialogue_ID.nunique()} dialogues | "
              f"shift {n_shift} / stable {n_elig - n_shift} / first-turn {len(df) - n_elig} "
              f"| shift rate {n_shift / n_elig:.1%} | neutral {df.Emotion.eq('neutral').mean():.1%}")

    train, test = splits["train"], splits["test"]
    trans = pd.crosstab(train.prev_same, train.Emotion, normalize="index")
    print("\nP(current | previous), same speaker, train:")
    print(trans.reindex(index=EKMAN, columns=EKMAN).round(2).to_string())

    shifted = test[test["shift"]]
    types = shifted.apply(lambda r: shift_type(r.prev_same, r.Emotion), axis=1)
    print("\nTest shift taxonomy:", types.value_counts().to_dict())

    y = test.Emotion
    print("\nTest ERC baselines:")
    print("  majority (neutral):           ", fmt_scores(y, ["neutral"] * len(y)),
          fmt_scores(y, ["neutral"] * len(y), average="macro"))
    inertia = test.prev_same.fillna("neutral")
    print("  inertia, oracle prev label:   ", fmt_scores(y, inertia),
          fmt_scores(y, inertia, average="macro"))

    # Forecast "will this speaker's next turn shift?" from the previous emotion alone
    p_shift = 1 - pd.Series(np.diag(trans.reindex(index=EKMAN, columns=EKMAN)), index=EKMAN)
    elig = test[test.eligible]
    score = elig.prev_same.map(p_shift)
    print(f"\nShift-forecast baseline (Markov prior, oracle prev label): "
          f"AUROC={roc_auc_score(elig['shift'], score):.3f} "
          f"AUPRC={average_precision_score(elig['shift'], score):.3f} "
          f"(chance AUPRC = base rate {elig['shift'].mean():.3f})")

    allsplits = pd.concat(splits.values())
    main = allsplits[allsplits.Speaker.isin(MAIN_SIX) & allsplits.eligible]
    rate = main.groupby("Speaker")["shift"].agg(["mean", "size"]).sort_values("mean")
    print("\nShift rate by main character (all splits):")
    print(rate.rename(columns={"mean": "shift_rate", "size": "eligible_turns"}).round(3).to_string())
    return splits


def meld_fr_report():
    print("\n=== MELD-FR (emotion-flip triggers) ===")
    for split, key in [("train", "triggers"), ("val", "triggers"), ("test", "labels")]:
        data = json.loads((DATA / "meld" / f"MELD_{split}_efr.json").read_text())
        y, preds, dist = [], {"target": [], "previous": [], "target+previous": []}, []
        for inst in data:
            gold = [1 if str(v) == "1.0" else 0 for v in inst[key]]
            n = len(gold)
            y += gold
            dist += [n - 1 - i for i, g in enumerate(gold) if g]
            preds["target"] += [int(i == n - 1) for i in range(n)]
            preds["previous"] += [int(i == n - 2) for i in range(n)]
            preds["target+previous"] += [int(i >= n - 2) for i in range(n)]
        near = np.bincount(dist)[:4] / len(dist)
        print(f"{split}: {len(data)} flips, {sum(y)} triggers among {len(y)} candidates | "
              f"trigger distance 0/1/2/3 = {np.round(near, 2).tolist()}")
        for name, p in preds.items():
            prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary")
            print(f"  heuristic '{name}' as trigger: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")


def dailydialog_report():
    print("\n=== DailyDialog ===")
    raws, splits = {}, {}
    for s in ["train", "validation", "test"]:
        raws[s], df = load_dailydialog(s)
        splits[s] = df
        n_shift, n_elig = df["shift"].sum(), df["eligible"].sum()
        print(f"{s}: {len(raws[s])} dialogues / {len(df)} utts | shift rate {n_shift / n_elig:.1%} "
              f"| neutral {df.emotion.eq('neutral').mean():.1%}")

    test = splits["test"]
    emotional = [c for c in DD_LABELS if c != "neutral"]
    inertia = test.prev_same.fillna("neutral")
    print("Test inertia (oracle prev label):",
          fmt_scores(test.emotion, inertia, labels=emotional, average="micro"), "excl. neutral")

    def norm(t):
        return " ".join(str(t).lower().split())

    def full(df):
        return {"||".join(map(norm, r.utterances)) for _, r in df.iterrows()}

    def opening(df):
        return ["||".join(map(norm, r.utterances[:2])) for _, r in df.iterrows()]

    train_full, test_full = full(raws["train"]), full(raws["test"])
    train_open = set(opening(raws["train"]))
    shared_open = sum(o in train_open for o in opening(raws["test"]))
    print(f"Leakage: {len(test_full & train_full)}/{len(test_full)} unique test dialogues "
          f"appear verbatim in train; {shared_open}/{len(raws['test'])} share their first two turns")


if __name__ == "__main__":
    meld_report()
    meld_fr_report()
    dailydialog_report()
