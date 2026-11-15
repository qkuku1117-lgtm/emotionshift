"""B4 input builder: the shared ``Speaker: text`` format, no leakage, EmoWOZ system turns as context."""

import numpy as np
import pandas as pd
import pytest

from emotionshift.data.context import build_inputs, dialogue_inputs, k_label, parse_k
from emotionshift.data.shifts import add_shift_labels

TURNS = [("A", "one"), ("B", "two"), ("A", "three"), ("B", "four")]


def frame(dataset="meld", split="train", n=3, system=False, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["neutral", "joy", "anger"] if dataset != "emowoz" else ["neutral", "fearful", "abusive"]
    rows = []
    for d in range(n):
        for t in range(int(rng.integers(3, 7))):
            sys_turn = system and t % 2 == 1
            rows.append((dataset, split, f"D{d}" if system else d, t, "system" if sys_turn else ("user" if system else "AB"[t % 2]),
                         f"text {d} {t}", None if sys_turn else labels[int(rng.integers(0, 3))]))
    df = pd.DataFrame(rows, columns=["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"])
    return add_shift_labels(df).reset_index(drop=True)


def test_k_zero_is_the_bare_utterance():
    assert dialogue_inputs(TURNS, 0) == ["one", "two", "three", "four"]


def test_window_is_oldest_first_target_last():
    assert dialogue_inputs(TURNS, 2)[3] == "B: two\nA: three\nB: four"
    assert dialogue_inputs(TURNS, 2)[0] == "A: one"


def test_all_keeps_the_whole_past_and_never_the_future():
    out = dialogue_inputs(TURNS, None)
    assert out[2] == "A: one\nB: two\nA: three" and "four" not in out[2]


def test_parse_and_label_k():
    assert parse_k("all") is None and parse_k(None) is None and parse_k("5") == 5 and parse_k(0) == 0
    assert k_label(None) == "all" and k_label(3) == "3"
    with pytest.raises(ValueError):
        parse_k(-1)


def test_history_stays_inside_the_dialogue_and_split():
    df = pd.concat([frame(split="train"), frame(split="dev")], ignore_index=True)
    out = build_inputs(df, None)
    assert len(out) == len(df)
    for _, r in out.iterrows():
        first = r["input"].split("\n")[0]
        assert first == f"{r['speaker']}: text {r['dialogue_id']} 0" or r["turn"] > 0
        assert all(f"text {r['dialogue_id']} " in line for line in r["input"].split("\n"))


def test_emowoz_system_turns_are_context_not_targets():
    df = frame("emowoz", system=True)
    out = build_inputs(df, 1)
    assert out.emotion.notna().all() and len(out) == df.emotion.notna().sum()
    assert all(i.startswith("system: ") for i, t in zip(out.input, out.turn) if t > 0)


def test_labels_never_enter_the_input():
    df = frame()
    a = build_inputs(df, 3)["input"]
    b = build_inputs(df.assign(emotion="anger"), 3)["input"]
    assert a.tolist() == b.tolist()


def test_rows_keep_their_order_when_the_frame_is_shuffled():
    df = frame(n=4)
    shuffled = df.sample(frac=1, random_state=1)
    a = build_inputs(df, 2).set_index(["dialogue_id", "turn"])["input"]
    b = build_inputs(shuffled, 2).set_index(["dialogue_id", "turn"])["input"]
    assert a.sort_index().equals(b.sort_index())


def test_bad_frames_are_refused():
    with pytest.raises(ValueError):
        build_inputs(frame().drop(columns="speaker"), 2)
    with pytest.raises(ValueError):
        build_inputs(pd.concat([frame(), frame()]), 2)           # repeated index
    with pytest.raises(ValueError):
        build_inputs(pd.concat([frame(), frame()], ignore_index=True), 2)   # same dialogue twice


@pytest.mark.parametrize("dataset,system", [("meld", False), ("emowoz", True)])
@pytest.mark.parametrize("k", [0, 1, 3, None])
def test_matches_the_shared_utterance_dataset(dataset, system, k):
    pytest.importorskip("torch")
    from emotionshift.data.torch_data import IGNORE_INDEX, UtteranceDataset
    df = frame(dataset, system=system)
    theirs = {(s["metadata"]["dialogue_id"], s["metadata"]["turn"]): s["text"]
              for s in UtteranceDataset(df, context_turns=k) if s["label"] != IGNORE_INDEX}
    ours = build_inputs(df, k).set_index(["dialogue_id", "turn"])["input"].to_dict()
    assert ours == theirs
