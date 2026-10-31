"""Loader contracts on small fixtures, independent of downloaded datasets."""

import json

import pandas as pd
import pytest

from emotionshift.data import load
from emotionshift.data.loaders import duplicate_dialogues


def write_dd(root, split, dialogues):
    folder = root / "dailydialog"
    folder.mkdir(exist_ok=True)
    (folder / f"{split}.json").write_text(
        "\n".join(
            json.dumps({"dialogue": [{"text": text, "emotion": emotion} for text, emotion in d]})
            for d in dialogues
        )
    )


def test_loads_only_requested_split_and_preserves_raw_text(tmp_path):
    text = "  I \u2019 m happy !  "
    write_dd(tmp_path, "test", [[(text, "happiness")]])
    frame = load("dailydialog", "test", root=tmp_path)
    assert frame.raw_text.tolist() == [text]
    assert frame.text.tolist() == ["I ' m happy !"]
    assert frame.emotion.tolist() == ["joy"]
    assert frame.dialogue_id.tolist() == ["dailydialog-test-0"]
    assert frame.source_dialogue_id.tolist() == [0]
    assert frame.source_split.tolist() == ["test"]


def test_dedup_uses_raw_case_sensitive_text_and_keeps_first_copy(tmp_path):
    write_dd(tmp_path, "train", [[("Hello", "no_emotion")]])
    write_dd(
        tmp_path,
        "test",
        [
            [("  Hello  ", "no_emotion")],
            [("hello", "no_emotion")],
            [("I \u2019 m fine", "happiness")],
            [("I ' m fine", "happiness")],
            [("hello", "no_emotion")],
        ],
    )
    frame = load("dailydialog", ("train", "test", "test_dedup"), root=tmp_path)
    dedup = frame[frame.split == "test_dedup"]
    assert dedup.source_dialogue_id.tolist() == [1, 2, 3]
    assert dedup.source_split.unique().tolist() == ["test"]
    # Normalized model inputs can coincide without becoming verbatim duplicates.
    assert dedup.text.iloc[1] == dedup.text.iloc[2]
    assert not frame.duplicated(["dialogue_id", "turn"]).any()
    assert duplicate_dialogues(frame[frame.split == "train"], frame[frame.split == "test"]) == {
        "dailydialog-test-0",
    }


def test_numeric_source_order_and_speaker_history(tmp_path):
    folder = tmp_path / "meld"
    folder.mkdir()
    pd.DataFrame(
        {
            "Dialogue_ID": [10, 2, 2, 2],
            "Utterance_ID": [0, 2, 0, 1],
            "Speaker": ["A", "A", "A", "B"],
            "Utterance": ["Okay", "Wow", "Okay", "Yeah"],
            "Emotion": ["neutral", "surprise", "neutral", "joy"],
        }
    ).to_csv(folder / "test_sent_emo.csv", index=False)
    frame = load("meld", "test", root=tmp_path)
    assert frame.source_dialogue_id.tolist() == [2, 2, 2, 10]
    assert frame.stratum.tolist() == ["first", "first", "shift", "first"]
    assert frame.turn_gap.iloc[2] == 2
    assert frame.prev_emotion.iloc[2] == "neutral"


def test_default_splits_do_not_duplicate_test_rows(tmp_path):
    for split in ("train", "valid", "test"):
        write_dd(tmp_path, split, [[("Okay", "no_emotion")]])
    frame = load("dailydialog", root=tmp_path)
    assert frame.split.tolist() == ["train", "dev", "test"]
    assert frame.stratum.tolist() == ["first"] * 3
    assert frame.dialogue_id.nunique() == 3


@pytest.mark.parametrize(
    "dataset,splits",
    [
        ("unknown", None),
        ("meld", "test_dedup"),
        ("meld", ()),
        ("dailydialog", ("train", "train")),
        ("dailydialog", "validation"),
    ],
)
def test_invalid_requests_fail_before_reading_files(tmp_path, dataset, splits):
    with pytest.raises(ValueError):
        load(dataset, splits, root=tmp_path)


def test_unknown_labels_are_not_silently_mapped(tmp_path):
    write_dd(tmp_path, "test", [[("Okay", "typo")]])
    with pytest.raises(ValueError, match="Unknown emotion"):
        load("dailydialog", "test", root=tmp_path)


def write_emowoz(root, split_ids, dialogues):
    """dialogues: id -> [(text, label)], label -1 for system turns."""
    folder = root / "emowoz"
    folder.mkdir(exist_ok=True)
    logs = {
        dialogue_id: {
            "log": [
                {
                    "text": text,
                    "emotion": []
                    if label == -1
                    else [{"annotation": label}] * 3 + [{"emotion": label, "sentiment": 0}],
                }
                for text, label in turns
            ]
        }
        for dialogue_id, turns in dialogues.items()
    }
    sources = {"multiwoz": {}, "dialmage": {}}
    for dialogue_id, log in logs.items():
        sources["dialmage" if dialogue_id.startswith("DMAGE") else "multiwoz"][dialogue_id] = log
    for source, content in sources.items():
        (folder / f"emowoz-{source}.json").write_text(json.dumps(content))
    (folder / "data-split.json").write_text(json.dumps(split_ids))


def test_emowoz_keeps_system_turns_as_unlabelled_context(tmp_path):
    write_emowoz(
        tmp_path,
        {
            "train": {"multiwoz": ["MUL1.json", "MUL1.json"], "dialmage": ["DMAGE2.json"]},
            "dev": {"multiwoz": [], "dialmage": []},
            "test": {"multiwoz": [], "dialmage": []},
        },
        {
            "MUL1.json": [("Find a hotel.", 0), ("Which area?", -1), ("Great, thanks!", 6)],
            "DMAGE2.json": [("Book it.", 0), ("", -1), ("Hello??", 2), ("Sorry.", -1)],
        },
    )
    frame = load("emowoz", "train", root=tmp_path)
    # The repeated train ID is loaded once; the ".json" suffix is dropped.
    assert sorted(frame.dialogue_id.unique()) == ["emowoz-train-DMAGE2", "emowoz-train-MUL1"]
    user = frame[frame.speaker == "user"]
    system = frame[frame.speaker == "system"]
    assert user.turn.tolist() == [0, 2, 0, 2]
    assert user.emotion.tolist() == ["neutral", "dissatisfied", "neutral", "satisfied"]
    assert system.emotion.isna().all() and system.stratum.isna().all()
    assert user.stratum.tolist() == ["first", "shift", "first", "shift"]
    assert user.prev_emotion.tolist()[1] == "neutral"
    assert set(user.turn_gap.dropna()) == {2}


def test_emowoz_rejects_unknown_labels(tmp_path):
    write_emowoz(
        tmp_path,
        {"train": {"multiwoz": ["MUL1.json"], "dialmage": []}},
        {"MUL1.json": [("Hi", 7)]},
    )
    with pytest.raises(ValueError, match="Unknown EmoWOZ label"):
        load("emowoz", "train", root=tmp_path)
