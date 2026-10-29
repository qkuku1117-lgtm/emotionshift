import pandas as pd
import pytest

from emotionshift.data import add_shift_columns, clean_text, shift_type


def toy_dialogue():
    #           A          B        A        B        A
    emotions = ["neutral", "joy", "anger", "joy", "anger"]
    return pd.DataFrame({
        "dialogue_id": "d0",
        "turn": range(5),
        "speaker": ["A", "B", "A", "B", "A"],
        "text": ["x"] * 5,
        "emotion": emotions,
    })


def test_strata_follow_the_same_speaker():
    out = add_shift_columns(toy_dialogue())
    assert out["stratum"].tolist() == ["first", "first", "shift", "stable", "stable"]
    assert out["prev_emotion"].tolist()[2:] == ["neutral", "joy", "anger"]
    assert out["turn_gap"].tolist()[2:] == [2, 2, 2]


def test_shift_type_only_on_shift_rows():
    out = add_shift_columns(toy_dialogue())
    assert out["shift_type"].isna().tolist() == [True, True, False, True, True]
    assert out.loc[2, "shift_type"] == "onset"


def test_speakers_in_different_dialogues_do_not_share_history():
    a, b = toy_dialogue(), toy_dialogue()
    b["dialogue_id"] = "d1"
    out = add_shift_columns(pd.concat([a, b], ignore_index=True))
    assert out.loc[5, "stratum"] == "first"


@pytest.mark.parametrize(("prev", "curr", "expected"), [
    ("neutral", "anger", "onset"),
    ("sadness", "neutral", "offset"),
    ("joy", "anger", "valence_flip"),
    ("fear", "joy", "valence_flip"),
    ("anger", "sadness", "within_negative"),
    ("surprise", "joy", "surprise"),
    ("anger", "surprise", "surprise"),
])
def test_shift_type(prev, curr, expected):
    assert shift_type(prev, curr) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("You’re a genius!", "You're a genius!"),
    ("You\x92re a genius!", "You're a genius!"),
    ("I ’ ll call you , okay ?", "I ' ll call you , okay ?"),
    ("Wait…", "Wait..."),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected
