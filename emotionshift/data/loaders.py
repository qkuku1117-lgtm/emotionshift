"""Load MELD and DailyDialog into one Dialogue/Turn table.

Every row is one utterance with the columns

    dataset, split, dialogue_id, turn, speaker, text, emotion,
    prev_emotion, stratum, shift_type

``split`` is one of ``train`` / ``dev`` / ``test``; for DailyDialog there is
also ``test_dedup`` (proposal: "Data hygiene" -- the headline test set).
Run ``scripts/fetch_data.sh`` first to download the raw files.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .labels import DAILYDIALOG_MAP, MELD_MAP, EMOWOZ_LABELS, EMOWOZ_MAP
from .shifts import add_shift_labels

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# MELD's CSVs contain Windows-1252 mojibake for quotes and dashes.
_MOJIBAKE = {
    "\u0092": "'", "\u0091": "'", "\u0093": '"', "\u0094": '"',
    "\u0085": "...", "\u0096": "-", "\u0097": "-", " ": " ",
    "â\x80\x99": "'", "â\x80\x98": "'", "â\x80\x9c": '"', "â\x80\x9d": '"',
    "â\x80\x94": "-", "â\x80\x93": "-", "â\x80¦": "...",
}


def clean_text(s: str) -> str:
    for bad, good in _MOJIBAKE.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    return re.sub(r"\s+", " ", s).strip()


def load_meld(raw: Path = RAW / "meld") -> pd.DataFrame:
    frames = []
    for split in ["train", "dev", "test"]:
        d = pd.read_csv(raw / f"{split}_sent_emo.csv", encoding="utf-8")
        frames.append(pd.DataFrame({
            "dataset": "meld",
            "split": split,
            "dialogue_id": d["Dialogue_ID"].astype(int),
            "turn": d["Utterance_ID"].astype(int),
            "speaker": d["Speaker"].astype(str),
            "text": d["Utterance"].astype(str).map(clean_text),
            "emotion": d["Emotion"].map(MELD_MAP),
        }))
    df = pd.concat(frames, ignore_index=True)
    assert df["emotion"].notna().all()
    return add_shift_labels(df).reset_index(drop=True)


def _dialogue_key(texts: list[str]) -> str:
    """Whole-dialogue key for *verbatim* duplicates (case-sensitive, raw text,
    whitespace collapsed). This definition gives the proposal's 105 of 996."""
    return " ||| ".join(re.sub(r"\s+", " ", t).strip() for t in texts)


def load_dailydialog(raw: Path = RAW / "dailydialog") -> pd.DataFrame:
    """Official 11,118 / 1,000 / 1,000 split; speakers alternate A, B, A, ...

    Adds ``test_dedup``: the test set minus dialogues that appear verbatim in
    train, and minus repeated dialogues within test (first copy kept).
    """
    rows, keys = [], {}
    for split, fname in [("train", "train"), ("dev", "valid"), ("test", "test")]:
        keys[split] = []
        with open(raw / f"{fname}.json", encoding="utf-8") as f:
            for i, line in enumerate(f):
                dlg = json.loads(line)["dialogue"]
                keys[split].append(_dialogue_key([u["text"] for u in dlg]))
                texts = [clean_text(u["text"]) for u in dlg]
                for t, (u, text) in enumerate(zip(dlg, texts)):
                    rows.append((split, i, t, "AB"[t % 2], text, DAILYDIALOG_MAP[u["emotion"]]))
    df = pd.DataFrame(rows, columns=["split", "dialogue_id", "turn", "speaker", "text", "emotion"])
    df.insert(0, "dataset", "dailydialog")

    train_keys = set(keys["train"])
    seen, keep = set(), []
    for i, k in enumerate(keys["test"]):
        if k not in train_keys and k not in seen:
            keep.append(i)
        seen.add(k)
    dedup = df[(df["split"] == "test") & df["dialogue_id"].isin(keep)].copy()
    dedup["split"] = "test_dedup"
    df = pd.concat([df, dedup], ignore_index=True)
    return add_shift_labels(df).reset_index(drop=True)


def dailydialog_leakage(raw: Path = RAW / "dailydialog") -> dict:
    """Numbers quoted in the proposal's "Data hygiene" section."""
    keys = {}
    for split, fname in [("train", "train"), ("test", "test")]:
        with open(raw / f"{fname}.json", encoding="utf-8") as f:
            keys[split] = [[u["text"] for u in json.loads(l)["dialogue"]] for l in f]
    train_full = {_dialogue_key(d) for d in keys["train"]}
    train_two = {_dialogue_key(d[:2]) for d in keys["train"]}
    test_unique = {_dialogue_key(d) for d in keys["test"]}
    return {
        "test_dialogues": len(keys["test"]),
        "unique_test_dialogues": len(test_unique),
        "unique_test_verbatim_in_train": len(test_unique & train_full),
        "test_sharing_first_two_turns": sum(_dialogue_key(d[:2]) in train_two for d in keys["test"]),
    }


def load(dataset: str) -> pd.DataFrame:
    return {"meld": load_meld, "dailydialog": load_dailydialog,
            "emowoz": load_emowoz}[dataset]()


def load_emowoz(raw: Path = RAW / "emowoz") -> pd.DataFrame:
    """Official Zenodo 6506504 files; labelled user turns only.

    Original turn indices and native labels are retained. See docs/EMOWOZ.md
    for the lossy mapping. System turns are unlabelled, never neutral targets.
    """
    with open(raw / "data-split.json", encoding="utf-8") as f:
        splits = json.load(f)
    sources = {}
    for source in ("multiwoz", "dialmage"):
        with open(raw / f"emowoz-{source}.json", encoding="utf-8") as f:
            sources[source] = json.load(f)
    rows, assigned = [], set()
    for split in ("train", "dev", "test"):
        for source, dialogues in sources.items():
            # The official manifest repeats one training ID. Keep its first copy.
            for dialogue_id in dict.fromkeys(splits[split][source]):
                if dialogue_id in assigned:
                    raise ValueError(f"EmoWOZ dialogue appears in multiple partitions: {dialogue_id}")
                assigned.add(dialogue_id)
                for turn, utterance in enumerate(dialogues[dialogue_id]["log"]):
                    if turn % 2:
                        continue
                    native_id = utterance["emotion"][3]["emotion"]
                    if type(native_id) is not int or native_id not in EMOWOZ_LABELS:
                        raise ValueError(f"Invalid EmoWOZ label {native_id!r}: {dialogue_id}:{turn}")
                    native = EMOWOZ_LABELS[native_id]
                    rows.append({
                        "dataset": "emowoz", "split": split,
                        "dialogue_id": dialogue_id, "turn": turn, "speaker": "user",
                        "text": clean_text(utterance["text"]), "emotion": EMOWOZ_MAP[native],
                        "emotion_original_id": native_id, "emotion_original": native,
                        "source": source,
                    })
    columns = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion",
               "emotion_original_id", "emotion_original", "source"]
    df = add_shift_labels(pd.DataFrame(rows, columns=columns)).reset_index(drop=True)
    shared = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion",
              "prev_emotion", "stratum", "shift_type"]
    return df[shared + ["emotion_original_id", "emotion_original", "source"]]
