"""Label each utterance as a speaker's first turn, a stable turn or an emotion shift.

A shift is an utterance whose emotion differs from the same speaker's previous
utterance in the dialogue. This matches the flip definition of MELD-FR.
"""

import numpy as np

from emotionshift.data.datasets import VALENCE

STRATA = ("first", "stable", "shift")
SHIFT_TYPES = ("onset", "offset", "valence_flip", "within_negative", "surprise_involved")


def shift_type(prev, curr):
    """Coarse type of a change from emotion `prev` to a different emotion `curr`."""
    if prev == "neutral":
        return "onset"
    if curr == "neutral":
        return "offset"
    pair = {VALENCE[prev], VALENCE[curr]}
    if pair == {"positive", "negative"}:
        return "valence_flip"
    if pair == {"negative"}:
        return "within_negative"
    return "surprise_involved"


def add_shift_columns(df, emotion_col="emotion"):
    """Return a copy of `df` with the speaker's history and the shift labels.

    Expects rows ordered by dialogue and turn. Adds:
        prev_emotion  the same speaker's previous emotion (NaN on their first turn)
        turn_gap      turns since that speaker last spoke (NaN on their first turn)
        stratum       "first", "stable" or "shift"
        shift_type    one of SHIFT_TYPES on shift rows, else NaN
    """
    out = df.copy()
    by_speaker = out.groupby(["dialogue_id", "speaker"], sort=False)
    out["prev_emotion"] = by_speaker[emotion_col].shift(1)
    out["turn_gap"] = out["turn"] - by_speaker["turn"].shift(1)
    first = out["prev_emotion"].isna()
    shift = ~first & (out[emotion_col] != out["prev_emotion"])
    out["stratum"] = np.select([first, shift], ["first", "shift"], default="stable")
    out["shift_type"] = [
        shift_type(p, c) if s else np.nan
        for p, c, s in zip(out["prev_emotion"], out[emotion_col], shift)
    ]
    return out
