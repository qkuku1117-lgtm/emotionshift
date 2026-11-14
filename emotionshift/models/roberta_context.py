"""B4: RoBERTa with speaker names and the past k turns (proposal: "Models", Table 4 #5).

Same encoder, head and training loop as B3 (``RobertaUtterance``); the only change is
the input. Every utterance is fed together with the turns before it, EmoBERTa-style::

    Monica: Oh my God! </s> Chandler: What? </s> Monica: I can't believe it.
    |------------- past k turns -------------|   |---- current utterance ---|

The current utterance is always last. Inputs longer than ``max_len`` are truncated from
the **left**, so the oldest context is dropped first and the current utterance is kept.
Only past turns are used (causal), so the model can run turn by turn in the demo.

``context_inputs`` also builds the E4 controls, applied to the *input* of a trained model:

* ``"none"``      no context: the current utterance with its speaker only
* ``"shuffled"``  the same number of past turns, drawn from random other dialogues
* ``"speakers"``  real context, speaker names shuffled within each dialogue

Usage::

    from emotionshift.models.roberta_context import RobertaContext
    m = RobertaContext(RobertaConfig(max_len=256), k=3).fit_frames(train_df, dev_df)
    m.predict_frame(test_df)                      # one label per row of test_df
    m.predict_frame(test_df, mode="shuffled")     # E4 control
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .roberta import RobertaConfig, RobertaUtterance

MODES = ("real", "none", "shuffled", "speakers")
ALL = "all"


def parse_k(k) -> int | None:
    """``"all"`` / ``None`` -> None (every past turn); otherwise a non-negative int."""
    if k is None or str(k).lower() == ALL:
        return None
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0 or 'all'")
    return k


def k_name(k) -> str:
    k = parse_k(k)
    return ALL if k is None else str(k)


def context_inputs(
    df: pd.DataFrame, k, sep: str = "</s>", mode: str = "real", seed: int = 0
) -> list[str]:
    """One input string per row of ``df`` (same order): past ``k`` turns, then the current one.

    ``df`` holds whole dialogues of one split with ``dialogue_id``, ``turn``, ``speaker``
    and ``text``. Gold emotions are never read.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    k = parse_k(k)
    rng = np.random.default_rng(seed)
    order = df.sort_values(["dialogue_id", "turn"], kind="stable")
    pos = order.index.to_numpy()
    dialogue = order["dialogue_id"].to_numpy()
    speaker = order["speaker"].astype(str).to_numpy().copy()
    text = order["text"].astype(str).to_numpy()
    n = len(order)
    starts = np.flatnonzero(np.r_[True, dialogue[1:] != dialogue[:-1]])
    ends = np.r_[starts[1:], n]

    if mode == "speakers":
        for a, b in zip(starts, ends, strict=False):
            speaker[a:b] = rng.permutation(speaker[a:b])
    turns = [f"{s}: {t}" for s, t in zip(speaker, text, strict=False)]

    out = [""] * n
    for a, b in zip(starts, ends, strict=False):
        for i in range(a, b):
            n_past = 0 if mode == "none" else (i - a if k is None else min(k, i - a))
            if mode == "shuffled" and n_past:
                # Same number of turns, from other dialogues (at most a few redraws needed).
                pick = rng.integers(0, n, size=n_past)
                while (bad := (pick >= a) & (pick < b)).any():
                    pick[bad] = rng.integers(0, n, size=int(bad.sum()))
                past = [turns[j] for j in pick]
            else:
                past = turns[i - n_past : i]
            out[i] = f" {sep} ".join([*past, turns[i]])
    return pd.Series(out, index=pos).reindex(df.index).tolist()


class RobertaContext(RobertaUtterance):
    """``RobertaUtterance`` on context inputs; ``k`` past turns (``"all"`` = every past turn)."""

    def __init__(
        self, cfg: RobertaConfig | None = None, device: str | None = None, k=3, *, labels=None
    ):
        super().__init__(cfg or RobertaConfig(max_len=256), device=device, labels=labels)
        self.tokenizer.truncation_side = (
            "left"  # drop the oldest context, keep the current utterance
        )
        self.k = k_name(k)

    def inputs(self, df: pd.DataFrame, mode: str = "real", seed: int = 0) -> list[str]:
        return context_inputs(df, self.k, self.tokenizer.sep_token, mode, seed)

    def fit_frames(self, train: pd.DataFrame, dev: pd.DataFrame):
        return self.fit(self.inputs(train), train.emotion, self.inputs(dev), dev.emotion)

    def predict_frame(self, df: pd.DataFrame, mode: str = "real", seed: int = 0) -> list[str]:
        return self.predict(self.inputs(df, mode, seed))

    def predict_dialogue(self, speakers, texts) -> list[str]:
        """Label every turn of one dialogue given so far (the demo's turn-by-turn call)."""
        df = pd.DataFrame(
            {
                "dialogue_id": 0,
                "turn": range(len(texts)),
                "speaker": list(speakers),
                "text": list(texts),
            }
        )
        return self.predict_frame(df)

    def save(self, path):
        super().save(path)
        (Path(path) / "b4_config.json").write_text(json.dumps({"k": self.k}))

    @classmethod
    def load(cls, path, device: str | None = None) -> RobertaContext:
        m = super().load(path, device=device)
        m.k = json.loads((Path(path) / "b4_config.json").read_text())["k"]
        return m
