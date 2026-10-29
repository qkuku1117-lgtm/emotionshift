"""Verify notebook and B1 imports use the same shifts and evaluation rules."""

import pandas as pd
import pytest

from emotionshift.data import add_shift_columns
from emotionshift.data.shifts import add_shift_labels
from emotionshift.eval import metrics as b1_metrics
from emotionshift.eval import scores, scores_from_confusion, stratified_scores


def test_notebook_and_b1_helpers_use_the_shared_shift_schema():
    dialogue = pd.DataFrame(
        {
            "split": ["test"] * 3,
            "dialogue_id": ["d0"] * 3,
            "turn": [0, 1, 2],
            "speaker": ["A"] * 3,
            "emotion": ["joy", "surprise", "surprise"],
            "pred": ["joy", "surprise", "surprise"],
        }
    )
    original = add_shift_columns(dialogue)
    b1 = add_shift_labels(dialogue)

    assert original.stratum.tolist() == b1.stratum.tolist() == ["first", "shift", "stable"]
    assert original.loc[1, "shift_type"] == "surprise"
    assert b1.loc[1, "shift_type"] == "surprise"
    assert original.turn_gap.iloc[1:].tolist() == [1, 1]

    original_table = stratified_scores(original, "pred")
    b1_table = b1_metrics.stratified(b1).set_index("group")
    assert original_table.loc["shift:surprise", "n"] == 1
    assert b1_table.loc["shift_type:surprise", "n"] == 1
    assert original_table.loc["all", "weighted_f1"] == 1
    assert b1_table.loc["all", "weighted_f1"] == 1


def test_macro_f1_always_uses_all_seven_classes():
    gold = ["neutral", "joy"]
    assert scores(gold, gold)["macro_f1"] == pytest.approx(2 / 7)
    assert b1_metrics.scores(gold, gold)["macro_f1"] == pytest.approx(2 / 7)
    assert b1_metrics.scores(gold, gold)["n"] == 2
    matrix = b1_metrics.confusion(gold, gold).to_numpy()
    assert scores_from_confusion(matrix)["macro_f1"] == pytest.approx(2 / 7)
