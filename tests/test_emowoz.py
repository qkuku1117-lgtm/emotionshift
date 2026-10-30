"""Offline fixtures cover the official format; release checks run if downloaded."""
import json

import pytest

from emotionshift.data.labels import EMOTIONS, EMOWOZ_LABELS, EMOWOZ_MAP
from emotionshift.data.loaders import RAW, load_emowoz


@pytest.fixture
def raw(tmp_path):
    def dialogue(ids):
        log = []
        for label in ids:
            log.extend([{"text": "  user  text ", "emotion": [
                {"annotation": 6}, {"annotation": 6}, {"annotation": 6}, {"emotion": label}]},
                {"text": "system reply"}])
        return {"log": log}
    multiwoz = {"a.json": dialogue([0, 2, 4, 6, 5, 1, 3, 0]),
                "b.json": dialogue([6]), "c.json": dialogue([0])}
    dialmage = {"DMAGE1.json": dialogue([0, 5])}
    splits = {"train": {"multiwoz": ["a.json", "a.json"], "dialmage": ["DMAGE1.json"]},
              "dev": {"multiwoz": ["b.json"], "dialmage": []},
              "test": {"multiwoz": ["c.json"], "dialmage": []}}
    for name, data in [("emowoz-multiwoz.json", multiwoz),
                       ("emowoz-dialmage.json", dialmage), ("data-split.json", splits)]:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_native_labels_schema_and_shifts(raw):
    df = load_emowoz(raw)
    assert df.columns.tolist() == ["dataset", "split", "dialogue_id", "turn", "speaker",
        "text", "emotion", "prev_emotion", "stratum", "shift_type",
        "emotion_original_id", "emotion_original", "source"]
    assert len(df) == 12
    assert set(df.emotion) <= set(EMOTIONS)
    assert not df.duplicated(["split", "dialogue_id", "turn"]).any()
    assert set(df.speaker) == {"user"}
    assert set(df.text) == {"user text"}
    a = df[df.dialogue_id == "a.json"]
    assert a.turn.tolist() == list(range(0, 16, 2))
    assert a.emotion_original_id.tolist() == [0, 2, 4, 6, 5, 1, 3, 0]
    assert a.emotion_original.tolist() == [EMOWOZ_LABELS[i] for i in a.emotion_original_id]
    assert a.emotion.tolist() == [EMOWOZ_MAP[e] for e in a.emotion_original]
    assert a.stratum.tolist() == ["first", "shift", "stable", "shift", "stable", "shift", "shift", "shift"]
    assert a.shift_type.fillna("none").tolist() == ["none", "onset", "none", "valence_flip", "none",
                                        "valence_flip", "within_negative", "offset"]
    first = df[df.stratum == "first"]
    assert len(first) == 4
    assert first.prev_emotion.isna().all()


def test_invalid_label_is_not_silently_mapped(raw):
    path = raw / "emowoz-multiwoz.json"
    data = json.loads(path.read_text())
    data["a.json"]["log"][0]["emotion"][3]["emotion"] = 7
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid EmoWOZ label"):
        load_emowoz(raw)


def test_partition_overlap_rejected(raw):
    path = raw / "data-split.json"
    data = json.loads(path.read_text())
    data["dev"]["multiwoz"].append("a.json")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="multiple partitions"):
        load_emowoz(raw)


@pytest.mark.skipif(not (RAW / "emowoz" / "emowoz-multiwoz.json").exists(),
                    reason="Run scripts/fetch_emowoz.py for official release checks")
def test_official_release():
    from emotionshift.data.loaders import load
    df = load("emowoz")
    assert df.groupby("split").size().to_dict() == {"train": 66474, "dev": 8509, "test": 8634}
    assert len(df) == 83617
    assert df.dialogue_id.nunique() == 11433
    assert df.groupby("dialogue_id").split.nunique().max() == 1
    assert not df.duplicated(["split", "dialogue_id", "turn"]).any()
    assert df.groupby("dialogue_id").apply(lambda g: (g.stratum == "first").sum(),
                                         include_groups=False).eq(1).all()
