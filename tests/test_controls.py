import numpy as np
import pandas as pd
import pytest

from emotionshift.data.context import build_inputs, control_inputs


def frame():
    rows = []
    for d in range(4):
        for t in range(5):
            rows.append(dict(dataset="meld", split="test", dialogue_id=f"d{d}", turn=t, speaker="AB"[t % 2] + str(d),
                             text=f"d{d}t{t}", emotion="neutral"))
    return pd.DataFrame(rows)


def test_none_is_bare_utterance():
    f = frame()
    assert list(control_inputs(f, 3, "none")["input"]) == list(f.text)


def test_shuffled_context_keeps_length_and_target_and_uses_other_dialogues():
    f = frame()
    real, sh = build_inputs(f, 3), control_inputs(f, 3, "shuffled_context", seed=1)
    for r, s, (_, row) in zip(real["input"], sh["input"], f.iterrows()):
        assert len(r.split("\n")) == len(s.split("\n"))
        assert s.split("\n")[-1] == r.split("\n")[-1]                       # target unchanged
        for line in s.split("\n")[:-1]:
            assert not line.endswith(f"d{row.dialogue_id[1:]}t{row.turn}")  # never the own turn
            assert f"{row.dialogue_id}t" not in line                        # nor any own-dialogue turn


def test_shuffled_speakers_changes_names_not_text():
    f = frame()
    real, sh = build_inputs(f, 2), control_inputs(f, 2, "shuffled_speakers", seed=2)
    strip = lambda s: [l.split(": ", 1)[1] for l in s.split("\n")]
    assert [strip(a) for a in real["input"]] == [strip(a) for a in sh["input"]]
    assert list(real["input"]) != list(sh["input"])


def test_deterministic_and_unknown_mode():
    f = frame()
    a = control_inputs(f, 3, "shuffled_context", seed=5)["input"].tolist()
    assert a == control_inputs(f, 3, "shuffled_context", seed=5)["input"].tolist()
    with pytest.raises(ValueError):
        control_inputs(f, 3, "nope")


def test_k_all_and_emowoz_system_turns_are_context_not_targets():
    f = frame()
    f.loc[f.turn % 2 == 1, "emotion"] = np.nan
    out = control_inputs(f, None, "shuffled_context", seed=0)
    assert len(out) == int(f.emotion.notna().sum())
