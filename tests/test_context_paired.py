"""Paired native F1 checks with analytically known differences."""

import pandas as pd
import pytest

from scripts.analyse_emowoz_context import paired_delta


def frame():
    return pd.DataFrame(
        {
            "dataset": ["emowoz"] * 4,
            "split": ["test"] * 4,
            "dialogue_id": [0, 0, 1, 1],
            "turn": [0, 1, 0, 1],
            "emotion": ["neutral", "satisfied", "neutral", "dissatisfied"],
            "pred": ["neutral", "satisfied", "neutral", "dissatisfied"],
            "stratum": ["first", "shift", "first", "shift"],
        }
    )


def test_identical_predictions_have_zero_paired_interval():
    df = frame()
    row = paired_delta([df, df], [df.copy(), df.copy()], n_boot=100)
    assert row["mean_delta"] == row["ci95_low"] == row["ci95_high"] == 0
    bad = df.assign(pred="neutral")
    row = paired_delta([bad], [df], n_boot=100)
    assert row["mean_delta"] == 1


def test_misaligned_pairs_rejected():
    df = frame()
    with pytest.raises(ValueError, match="aligned"):
        paired_delta([df], [df.iloc[::-1]], n_boot=10)
