"""Load MELD, DailyDialog and MELD-FR into one utterance-level schema.

Every loader returns a DataFrame with one row per utterance and these columns:

    dataset      "meld" or "dailydialog"
    split        "train", "dev" or "test" (DailyDialog's "validation" is renamed)
    dialogue_id  unique across datasets and splits, e.g. "meld-train-0"
    turn         0-based position in the dialogue
    speaker      MELD character name; "A"/"B" for DailyDialog, whose speakers alternate
    text         cleaned utterance text (see `clean_text`)
    emotion      one of EMOTIONS

Run `bash scripts/fetch_data.sh` first; data is read from data/raw/.
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# Ekman's six basic emotions plus neutral, shared by MELD and DailyDialog
EMOTIONS = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")
VALENCE = {"neutral": "neutral", "joy": "positive", "surprise": "ambiguous", "anger": "negative",
           "sadness": "negative", "disgust": "negative", "fear": "negative"}

SPLITS = ("train", "dev", "test")
_DD_FILES = {"train": "train", "dev": "validation", "test": "test"}
_DD_EMOTIONS = ("neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise")
_MELD_FR_FILES = {"train": "train", "dev": "val", "test": "test"}

_PUNCT = str.maketrans({
    "’": "'", "‘": "'", "\x92": "'", "\x91": "'",
    "“": '"', "”": '"', "\x93": '"', "\x94": '"',
    "…": "...", "\x85": "...",
    "—": "-", "–": "-", "\x96": "-", "\x97": "-",
})


def clean_text(text):
    """Normalize typographic punctuation and undo DailyDialog's tokenized spacing."""
    text = str(text).translate(_PUNCT)
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)  # "I ' ll" -> "I'll"
    text = re.sub(r" ([,.!?;:%)])", r"\1", text)  # "word ," -> "word,"
    text = re.sub(r"\( ", "(", text)
    return " ".join(text.split())


def load_meld(split):
    """MELD utterances for one split, ordered by dialogue and turn."""
    df = pd.read_csv(DATA_ROOT / "meld" / f"{split}_sent_emo.csv")
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"])
    return pd.DataFrame({
        "dataset": "meld",
        "split": split,
        "dialogue_id": "meld-" + split + "-" + df["Dialogue_ID"].astype(str),
        "turn": df["Utterance_ID"].to_numpy(),
        "speaker": df["Speaker"].to_numpy(),
        "text": df["Utterance"].map(clean_text).to_numpy(),
        "emotion": df["Emotion"].to_numpy(),
    })


def load_dailydialog(split):
    """DailyDialog utterances for one split; speakers alternate A, B, A, ..."""
    raw = pd.read_parquet(DATA_ROOT / "dailydialog" / f"{_DD_FILES[split]}.parquet")
    rows = [
        (f"dailydialog-{split}-{d}", t, "AB"[t % 2], clean_text(text), _DD_EMOTIONS[e])
        for d, r in enumerate(raw.itertuples())
        for t, (text, e) in enumerate(zip(r.utterances, r.emotions))
    ]
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn", "speaker", "text", "emotion"])
    df.insert(0, "split", split)
    df.insert(0, "dataset", "dailydialog")
    return df


def load(dataset, splits=SPLITS):
    """Concatenate the requested splits of "meld" or "dailydialog"."""
    loader = {"meld": load_meld, "dailydialog": load_dailydialog}[dataset]
    return pd.concat([loader(s) for s in splits], ignore_index=True)


def load_meld_fr(split):
    """MELD-FR emotion-flip instances in long form, one row per context utterance.

    Each instance is a dialogue prefix ending at a flip utterance (`is_target`);
    `is_trigger` marks the utterances annotators judged responsible for the flip.
    """
    raw = json.loads((DATA_ROOT / "meld" / f"MELD_{_MELD_FR_FILES[split]}_efr.json").read_text())
    rows = []
    for i, inst in enumerate(raw):
        labels = inst.get("triggers", inst.get("labels"))  # the test file names it "labels"
        n = len(inst["utterances"])
        for pos in range(n):
            rows.append({
                "instance_id": f"meldfr-{split}-{i}",
                "position": pos,
                "distance": n - 1 - pos,  # 0 = the flip utterance itself
                "speaker": inst["speakers"][pos],
                "text": clean_text(inst["utterances"][pos]),
                "emotion": inst["emotions"][pos],
                "is_target": pos == n - 1,
                "is_trigger": str(labels[pos]) == "1.0",
            })
    return pd.DataFrame(rows)


def duplicate_dialogues(reference, target):
    """IDs of `target` dialogues whose full text also appears as a dialogue in `reference`.

    Used to remove DailyDialog test dialogues that leak verbatim from train.
    """
    def fingerprints(df):
        return df.groupby("dialogue_id", sort=False)["text"].agg(lambda s: "||".join(s.str.lower()))

    ref = set(fingerprints(reference))
    tgt = fingerprints(target)
    return set(tgt[tgt.isin(ref)].index)
