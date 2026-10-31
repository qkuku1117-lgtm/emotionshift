"""One utterance table for MELD, the conv-emotion DailyDialog JSON release and EmoWOZ.

``load(dataset, splits=..., root=...)`` reads only requested official splits.
DailyDialog's ``test_dedup`` is an explicit view: exclude verbatim train overlap
and keep the first occurrence of each remaining test dialogue. Matching uses
case-sensitive raw text with whitespace collapsed, before model text cleanup.

Every row retains raw_text, source_split and source_dialogue_id alongside a
unique dialogue_id scoped to dataset and output split. Gold shift annotations
are evaluation metadata; models must not use them as inference features.
MELD-FR has its own instance/trigger schema via load_meld_fr.

EmoWOZ uses its own label space (``labels.emotions("emowoz")``). Its system
turns stay in the table as context with a missing ``emotion`` and no stratum;
drop them before scoring.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .labels import DAILYDIALOG_MAP, DATASETS, EMOWOZ_MAP, MELD_MAP, SPLITS
from .shifts import add_shift_labels

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
_MELD_FR_FILES = {"train": "train", "dev": "val", "test": "test"}
_DD_FILES = {"train": "train", "dev": "valid", "test": "test"}
_EMOWOZ_SOURCES = ("multiwoz", "dialmage")
_COLUMNS = [
    "dataset",
    "split",
    "dialogue_id",
    "source_split",
    "source_dialogue_id",
    "turn",
    "speaker",
    "raw_text",
    "text",
    "emotion",
]

# MELD's CSVs contain Windows-1252 mojibake for quotes and dashes.
_MOJIBAKE = {
    "\u0092": "'",
    "\u0091": "'",
    "\u0093": '"',
    "\u0094": '"',
    "\u0085": "...",
    "\u0096": "-",
    "\u0097": "-",
    "\u00a0": " ",
    "â\x80\x99": "'",
    "â\x80\x98": "'",
    "â\x80\x9c": '"',
    "â\x80\x9d": '"',
    "â\x80\x94": "-",
    "â\x80\x93": "-",
    "â\x80¦": "...",
}


def clean_text(s: str) -> str:
    for bad, good in _MOJIBAKE.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("\u2013", "-").replace("…", "...")
    return re.sub(r"\s+", " ", s).strip()


def _select_splits(dataset, splits):
    allowed = (*SPLITS, "test_dedup") if dataset == "dailydialog" else SPLITS
    selected = (
        SPLITS if splits is None else ((splits,) if isinstance(splits, str) else tuple(splits))
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(s not in allowed for s in selected)
    ):
        raise ValueError(f"Invalid splits {selected!r} for {dataset}; choose from {allowed}")
    return selected


def _read_dd(raw, split):
    path = raw / f"{_DD_FILES[split]}.json"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["dialogue"] for line in stream if line.strip()]


def _read_emowoz(raw, selected):
    """Rows per split from the Zenodo release: user turns are even, system turns odd.

    The label is the final (aggregated) annotation; -1 marks system turns.
    """
    dialogues = {}
    for source in _EMOWOZ_SOURCES:
        dialogues.update(json.loads((raw / f"emowoz-{source}.json").read_text(encoding="utf-8")))
    split_ids = json.loads((raw / "data-split.json").read_text(encoding="utf-8"))
    rows = {}
    for split in selected:
        # The train list repeats one dialogue ID; keep its first occurrence.
        ids = dict.fromkeys(i for source in _EMOWOZ_SOURCES for i in split_ids[split][source])
        rows[split] = []
        for dialogue_id in ids:
            for turn, utterance in enumerate(dialogues[dialogue_id]["log"]):
                if turn % 2:
                    speaker, emotion = "system", None
                else:
                    label = utterance["emotion"][3]["emotion"]
                    if label not in EMOWOZ_MAP:
                        raise ValueError(f"Unknown EmoWOZ label {label!r} in {dialogue_id}")
                    speaker, emotion = "user", EMOWOZ_MAP[label]
                rows[split].append(
                    (dialogue_id.removesuffix(".json"), turn, speaker, utterance["text"], emotion)
                )
    return rows


def _dialogue_key(texts):
    return tuple(re.sub(r"\s+", " ", text).strip() for text in texts)


def _finish(frame, dataset, split):
    frame = frame.copy()
    frame["dataset"] = dataset
    frame["split"] = split
    frame["source_split"] = "test" if split == "test_dedup" else split
    frame["dialogue_id"] = dataset + "-" + split + "-" + frame["source_dialogue_id"].astype(str)
    # Only EmoWOZ system turns are unlabelled; any other missing label is unknown.
    unlabelled = frame["speaker"].eq("system") if dataset == "emowoz" else False
    if (frame["emotion"].isna() != unlabelled).any():
        raise ValueError(f"Unknown emotion label in {dataset}/{split}")
    if frame[["raw_text", "speaker"]].isna().any().any():
        raise ValueError(f"Missing text or speaker in {dataset}/{split}")
    if frame.duplicated(["dialogue_id", "turn"]).any():
        raise ValueError(f"Duplicate utterance key in {dataset}/{split}")
    frame["text"] = frame["raw_text"].map(clean_text)
    return add_shift_labels(frame[_COLUMNS]).reset_index(drop=True)


def load(dataset: str, splits=None, *, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Return annotated utterances for the requested splits, in requested order.

    Defaults to train/dev/test; request DailyDialog test_dedup explicitly.
    ``root`` contains meld/, dailydialog/ and emowoz/. No downloads or writes occur.
    Within each split, rows follow source dialogue ID and turn order.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASETS}")
    selected = _select_splits(dataset, splits)
    raw = Path(root) / dataset
    frames = []
    dialogues = {}
    emowoz = _read_emowoz(raw, selected) if dataset == "emowoz" else {}
    if dataset == "dailydialog":
        needed = {"train", "test"} if "test_dedup" in selected else set()
        needed.update(s for s in selected if s != "test_dedup")
        dialogues = {s: _read_dd(raw, s) for s in SPLITS if s in needed}
    for split in selected:
        if dataset == "meld":
            source = pd.read_csv(raw / f"{split}_sent_emo.csv", encoding="utf-8")
            frame = pd.DataFrame(
                {
                    "source_dialogue_id": source["Dialogue_ID"].astype(int),
                    "turn": source["Utterance_ID"].astype(int),
                    "speaker": source["Speaker"],
                    "raw_text": source["Utterance"],
                    "emotion": source["Emotion"].map(MELD_MAP),
                }
            )
        elif dataset == "emowoz":
            frame = pd.DataFrame(
                emowoz[split],
                columns=["source_dialogue_id", "turn", "speaker", "raw_text", "emotion"],
            )
        else:
            source_split = "test" if split == "test_dedup" else split
            keep = None
            if split == "test_dedup":
                train_keys = {_dialogue_key([u["text"] for u in d]) for d in dialogues["train"]}
                seen, keep = set(), set()
                for i, d in enumerate(dialogues["test"]):
                    key = _dialogue_key([u["text"] for u in d])
                    if key not in train_keys and key not in seen:
                        keep.add(i)
                    seen.add(key)
            rows = [
                (i, turn, "AB"[turn % 2], u["text"], DAILYDIALOG_MAP.get(u["emotion"]))
                for i, dialogue in enumerate(dialogues[source_split])
                if keep is None or i in keep
                for turn, u in enumerate(dialogue)
            ]
            frame = pd.DataFrame(
                rows,
                columns=[
                    "source_dialogue_id",
                    "turn",
                    "speaker",
                    "raw_text",
                    "emotion",
                ],
            )
        frames.append(_finish(frame, dataset, split))
    return pd.concat(frames, ignore_index=True)


def load_meld(splits=None, *, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Convenience form of load('meld', splits, root=root)."""
    return load("meld", splits, root=root)


def load_dailydialog(splits=None, *, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Convenience form of load('dailydialog', splits, root=root)."""
    return load("dailydialog", splits, root=root)


def load_emowoz(splits=None, *, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Convenience form of load('emowoz', splits, root=root)."""
    return load("emowoz", splits, root=root)


def duplicate_dialogues(reference, target):
    """Target dialogue IDs with verbatim raw-text matches in reference.

    Matching is case-sensitive and ordered by turn; whitespace is collapsed.
    Both tables must retain raw_text from the shared loader.
    """

    def fingerprints(frame):
        ordered = frame.sort_values(["dialogue_id", "turn"])
        return ordered.groupby("dialogue_id", sort=False)["raw_text"].agg(_dialogue_key)

    keys = set(fingerprints(reference))
    target_keys = fingerprints(target)
    return set(target_keys[target_keys.isin(keys)].index)


def dailydialog_leakage(*, root: Path = DATA_ROOT) -> dict:
    """Verbatim overlap statistics using the same raw-text rule as test_dedup."""
    raw = Path(root) / "dailydialog"
    train = [[u["text"] for u in d] for d in _read_dd(raw, "train")]
    test = [[u["text"] for u in d] for d in _read_dd(raw, "test")]
    train_keys = {_dialogue_key(d) for d in train}
    test_keys = {_dialogue_key(d) for d in test}
    train_two = {_dialogue_key(d[:2]) for d in train}
    return {
        "test_dialogues": len(test),
        "unique_test_dialogues": len(test_keys),
        "unique_test_verbatim_in_train": len(test_keys & train_keys),
        "test_sharing_first_two_turns": sum(_dialogue_key(d[:2]) in train_two for d in test),
    }


def load_meld_fr(split, *, root: Path = DATA_ROOT):
    """MELD-FR emotion-flip instances in long form, one row per context utterance.

    Each instance is a dialogue prefix ending at a flip utterance (`is_target`);
    `is_trigger` marks the utterances annotators judged responsible for the flip.
    """
    _select_splits("meld", split)
    raw = json.loads((Path(root) / "meld" / f"MELD_{_MELD_FR_FILES[split]}_efr.json").read_text())
    rows = []
    for i, inst in enumerate(raw):
        labels = inst.get("triggers", inst.get("labels"))  # the test file names it "labels"
        n = len(inst["utterances"])
        for pos in range(n):
            rows.append(
                {
                    "instance_id": f"meldfr-{split}-{i}",
                    "position": pos,
                    "distance": n - 1 - pos,  # 0 = the flip utterance itself
                    "speaker": inst["speakers"][pos],
                    "raw_text": inst["utterances"][pos],
                    "text": clean_text(inst["utterances"][pos]),
                    "emotion": inst["emotions"][pos],
                    "is_target": pos == n - 1,
                    "is_trigger": str(labels[pos]) == "1.0",
                }
            )
    return pd.DataFrame(rows)
