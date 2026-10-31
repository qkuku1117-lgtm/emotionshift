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


def forecast_frame(emotions, *, speakers="AB"):
    """One dialogue per row of `emotions`; turns alternate between `speakers`."""
    rows = [
        {
            "split": "test",
            "dialogue_id": f"d{i}",
            "turn": t,
            "speaker": speakers[t % 2],
            "text": "x",
            "emotion": e,
        }
        for i, dialogue in enumerate(emotions)
        for t, e in enumerate(dialogue)
    ]
    return add_shift_columns(pd.DataFrame(rows))


def test_forecast_scores_a_perfect_forecast_and_the_persistence_baseline():
    from emotionshift.eval.metrics import forecast_scores

    df = forecast_frame([GOLD[i : i + 10] for i in range(0, len(GOLD), 10)])
    df["pred"] = df["emotion"]
    df["forecast_pred"] = df["emotion"]
    df["forecast_shift_prob"] = (df["stratum"] == "shift").astype(float)
    for label in EMOTIONS:
        df[f"forecast_p_{label}"] = (df["emotion"] == label).astype(float)

    result = forecast_scores(df, df)
    scored = df[df["stratum"].isin(("stable", "shift"))]
    assert result["n"] == len(scored)
    assert result["shift_auroc"]["forecast"] == pytest.approx(1.0)
    assert result["emotion"]["forecast"]["weighted_f1"] == pytest.approx(1.0)
    # Recognition is perfect here, so the causal and oracle baselines agree.
    expected = scores(scored["emotion"], scored["prev_emotion"])
    assert result["emotion"]["persistence_gold"] == pytest.approx(expected)
    assert result["emotion"]["persistence_pred"] == pytest.approx(expected)
    assert result["class_auroc"]["joy"]["forecast"] == pytest.approx(1.0)


def test_forecast_transition_baseline_learns_which_emotions_persist():
    from emotionshift.eval.metrics import forecast_scores

    # Each speaker's joy is always followed by a shift; sadness always persists.
    df = forecast_frame([["joy", "sadness", "sadness", "sadness"]] * 5, speakers="AA")
    df["pred"] = df["emotion"]
    df["forecast_pred"] = "neutral"
    df["forecast_shift_prob"] = 0.5
    for label in EMOTIONS:
        df[f"forecast_p_{label}"] = 1 / len(EMOTIONS)

    result = forecast_scores(df, df)
    assert result["shift_auroc"]["forecast"] == pytest.approx(0.5)
    assert result["shift_auroc"]["transition_gold"] == pytest.approx(1.0)
