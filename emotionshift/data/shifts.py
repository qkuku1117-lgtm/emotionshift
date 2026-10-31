"""Shift labelling (proposal: "Tasks" section).

A *shift* is an utterance whose emotion differs from the same speaker's
previous utterance in the same dialogue. A speaker's first turn is neither
shift nor stable (stratum ``first``).

Every shift gets exactly one type, assigned in this priority order (the first
five reproduce the MELD test counts in the proposal: 355 / 340 / 158 / 81 / 69):

1. ``onset``            neutral -> any emotion (including surprise)
2. ``offset``           any emotion (including surprise) -> neutral
3. ``surprise``         surprise <-> another non-neutral emotion
4. ``within_negative``  negative -> different negative (e.g. anger -> sadness)
5. ``valence_flip``     positive <-> negative
6. ``within_positive``  positive -> different positive; only EmoWOZ has two
                        positive labels (excited, satisfied)
"""

from __future__ import annotations

import pandas as pd

from .labels import NEGATIVE, POSITIVE

SHIFT_TYPES = [
    "onset",
    "offset",
    "surprise",
    "within_negative",
    "valence_flip",
    "within_positive",
]
STRATA = ["first", "stable", "shift"]


def shift_type(prev: str, cur: str) -> str | None:
    """Return the shift type for a speaker going from ``prev`` to ``cur``."""
    if prev == cur:
        return None
    if prev == "neutral":
        return "onset"
    if cur == "neutral":
        return "offset"
    if "surprise" in (prev, cur):
        return "surprise"
    if prev in NEGATIVE and cur in NEGATIVE:
        return "within_negative"
    if (prev in POSITIVE and cur in NEGATIVE) or (prev in NEGATIVE and cur in POSITIVE):
        return "valence_flip"
    if prev in POSITIVE and cur in POSITIVE:
        return "within_positive"
    raise ValueError(f"unclassified shift {prev} -> {cur}")


def add_shift_labels(df: pd.DataFrame, label_col: str = "emotion") -> pd.DataFrame:
    """Add ``prev_emotion``, ``stratum`` and ``shift_type`` columns.

    Uses *gold* labels by default (for stratifying evaluation). Pass a column
    of predicted labels to derive predicted shifts instead.

    Unlabelled rows (missing label, e.g. EmoWOZ system turns) get no stratum and
    are skipped when finding a speaker's previous label. ``turn_gap`` counts all
    turns, labelled or not.
    """
    scope = [col for col in ("dataset", "split") if col in df.columns]
    order = "source_dialogue_id" if "source_dialogue_id" in df.columns else "dialogue_id"
    df = df.sort_values([*scope, order, "turn"]).copy()
    keys = [*scope, "dialogue_id", "speaker"]
    df["turn_gap"] = df["turn"] - df.groupby(keys, sort=False)["turn"].shift(1)
    labelled = df[label_col].notna()
    prev = df[labelled].groupby(keys, sort=False)[label_col].shift(1).reindex(df.index)
    df["prev_emotion"] = prev
    is_first = labelled & prev.isna()
    is_shift = labelled & ~is_first & (prev != df[label_col])
    df["stratum"] = None
    df.loc[labelled, "stratum"] = "stable"
    df.loc[is_first, "stratum"] = "first"
    df.loc[is_shift, "stratum"] = "shift"
    df["shift_type"] = [
        shift_type(p, c) if s == "shift" else None
        for p, c, s in zip(df["prev_emotion"], df[label_col], df["stratum"], strict=True)
    ]
    return df


def add_shift_columns(df: pd.DataFrame, emotion_col: str = "emotion") -> pd.DataFrame:
    """Notebook-compatible name for the shared shift labelling implementation."""
    return add_shift_labels(df, label_col=emotion_col)
