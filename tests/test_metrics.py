import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from emotionshift.data import EMOTIONS, add_shift_columns
from emotionshift.eval import paired_bootstrap, scores, stratified_scores

rng = np.random.default_rng(0)
GOLD = rng.choice(EMOTIONS, 500).tolist()
PRED = [g if rng.random() < 0.6 else rng.choice(EMOTIONS) for g in GOLD]


def test_scores_match_sklearn():
    s = scores(GOLD, PRED)
    emotional = [e for e in EMOTIONS if e != "neutral"]
    assert s["accuracy"] == pytest.approx(accuracy_score(GOLD, PRED))
    assert s["weighted_f1"] == pytest.approx(f1_score(GOLD, PRED, average="weighted"))
    assert s["macro_f1"] == pytest.approx(f1_score(GOLD, PRED, average="macro"))
    assert s["micro_f1_no_neutral"] == pytest.approx(
        f1_score(GOLD, PRED, labels=emotional, average="micro")
    )


def dialogues():
    n = len(GOLD)
    return add_shift_columns(
        pd.DataFrame(
            {
                "dialogue_id": [f"d{i // 10}" for i in range(n)],
                "turn": [i % 10 for i in range(n)],
                "speaker": ["AB"[i % 2] for i in range(n)],
                "text": "x",
                "emotion": GOLD,
                "pred": PRED,
                "perfect": GOLD,
            }
        )
    )


def test_stratified_counts_add_up():
    table = stratified_scores(dialogues(), "pred")
    assert table.loc[["first", "stable", "shift"], "n"].sum() == table.loc["all", "n"]


def test_bootstrap_of_perfect_vs_noisy_is_positive():
    result = paired_bootstrap(dialogues(), "pred", "perfect", n_boot=200)
    assert result["ci_low"] > 0
    assert result["diff"] == pytest.approx(1 - scores(GOLD, PRED)["weighted_f1"])


def test_no_neutral_metrics_drop_neutral_from_the_averages():
    from emotionshift.data.labels import emotions

    labels = emotions("emowoz")
    gold = ["neutral", "neutral", "satisfied", "satisfied", "dissatisfied"]
    pred = ["neutral", "satisfied", "satisfied", "neutral", "dissatisfied"]
    s = scores(gold, pred, labels)
    emotional = [e for e in labels if e != "neutral"]
    assert s["weighted_f1_no_neutral"] == pytest.approx(
        f1_score(gold, pred, labels=emotional, average="weighted", zero_division=0)
    )
    assert s["macro_f1_no_neutral"] == pytest.approx(
        f1_score(gold, pred, labels=emotional, average="macro", zero_division=0)
    )
    # Satisfied F1 0.5 (support 2) and dissatisfied F1 1.0 (support 1).
    assert s["weighted_f1_no_neutral"] == pytest.approx((0.5 * 2 + 1.0) / 3)
