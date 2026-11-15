"""Input builder for context-aware models (B4, Table 4 #5): the previous ``k`` turns of a dialogue.

Produces exactly the text of the shared ``UtteranceDataset(frame, context_turns=k)``
(``emotionshift.data.torch_data``, docs/TRAINING.md): ``Speaker: text`` lines, oldest first, the
target last, joined by newlines. ``k = 0`` is the bare utterance (B3's input) and ``k = None`` the
whole past. Pair it with a tokenizer whose ``truncation_side`` is ``"left"`` so a long history loses
its oldest turns, never the target. This module needs no PyTorch; ``tests/test_context.py`` checks
it against ``UtteranceDataset`` whenever PyTorch is installed.

Rules shared with the rest of the project:

* history never crosses a dialogue or a split: the key is ``(dataset, split, dialogue_id)``;
* only ``turn``, ``speaker`` and ``text`` are read, never a label ("no gold at inference");
* EmoWOZ system turns (``emotion`` missing) are context and are never targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"]
KEY = ["dataset", "split", "dialogue_id"]


def parse_k(token) -> int | None:
    """``"all"`` / ``None`` -> None (whole past), otherwise a non-negative int."""
    if token is None or str(token).lower() == "all":
        return None
    k = int(token)
    if k < 0:
        raise ValueError("k must be >= 0 or 'all'")
    return k


def k_label(k: int | None) -> str:
    return "all" if k is None else str(k)


def dialogue_inputs(turns: list[tuple[str, str]], k: int | None) -> list[str]:
    """Model input for every turn of one dialogue, given ``(speaker, text)`` pairs in order."""
    out = []
    for j in range(len(turns)):
        window = turns[: j + 1] if k is None else turns[max(0, j - k): j + 1]
        out.append(window[-1][1] if k == 0 else "\n".join(f"{s}: {t}" for s, t in window))
    return out


def build_inputs(frame: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """The labelled rows of ``frame`` (in ``frame`` order) with the model input in column ``input``.

    ``frame`` is a loader table, any number of splits. For EmoWOZ pass the full table: the system
    turns feed the history but are not returned, because they have no label.
    """
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"frame needs columns {missing}")
    if not frame.index.is_unique:
        raise ValueError("frame.index must be unique (call reset_index(drop=True))")
    inputs = {}
    for _, g in frame.groupby(KEY, sort=False):
        g = g.sort_values("turn", kind="stable")
        if g["turn"].duplicated().any():
            raise ValueError(f"duplicate turn numbers in dialogue {g['dialogue_id'].iloc[0]!r}")
        for idx, text in zip(g.index, dialogue_inputs(list(zip(g["speaker"], g["text"])), k), strict=True):
            inputs[idx] = text
    out = frame[frame["emotion"].notna()].copy()
    out["input"] = [inputs[i] for i in out.index]
    return out


# ---- E4 controls: the same trained model, a corrupted test input -------------------------------
CONTROLS = ["none", "shuffled_context", "shuffled_speakers"]


def control_inputs(frame: pd.DataFrame, k: int | None, mode: str, seed: int = 0) -> pd.DataFrame:
    """Model inputs for an E4 control (proposal E4): ``build_inputs`` with the history changed.

    * ``none``: the bare utterance (history removed at test time);
    * ``shuffled_context``: as many history turns as the real input has, but taken from a random
      *other* dialogue of the same split (a random contiguous span, speaker names kept);
    * ``shuffled_speakers``: the real history with the speaker names permuted within each dialogue.

    Only turn / speaker / text are read, never a label. Deterministic given ``seed``.
    """
    if mode == "none" or k == 0:
        return build_inputs(frame, 0)
    if mode == "shuffled_speakers":
        rng = np.random.default_rng(seed)
        f = frame.copy()
        for _, idx in f.groupby(KEY, sort=False).groups.items():
            idx = list(idx)
            f.loc[idx, "speaker"] = rng.permutation(f.loc[idx, "speaker"].to_numpy())
        return build_inputs(f, k)
    if mode != "shuffled_context":
        raise ValueError(f"unknown control {mode!r}; choose from {CONTROLS}")
    if not frame.index.is_unique:
        raise ValueError("frame.index must be unique (call reset_index(drop=True))")
    rng = np.random.default_rng(seed)
    turns = {}                                           # (dataset, split, dialogue) -> [(speaker, text)]
    for key, g in frame.groupby(KEY, sort=False):
        g = g.sort_values("turn", kind="stable")
        turns[key] = list(zip(g["speaker"], g["text"]))
    keys = list(turns)
    inputs = {}
    for key, g in frame.groupby(KEY, sort=False):
        g = g.sort_values("turn", kind="stable")
        others = [o for o in keys if o != key and o[:2] == key[:2]]
        for j, (idx, (s, t)) in enumerate(zip(g.index, turns[key])):
            n = j if k is None else min(k, j)
            hist: list[tuple[str, str]] = []
            if n and others:
                long = [o for o in others if len(turns[o]) >= n] or others
                src = turns[long[rng.integers(len(long))]]
                n = min(n, len(src))
                start = int(rng.integers(0, len(src) - n + 1))
                hist = src[start:start + n]
            inputs[idx] = "\n".join(f"{a}: {b}" for a, b in [*hist, (s, t)])
    out = frame[frame["emotion"].notna()].copy()
    out["input"] = [inputs[i] for i in out.index]
    return out
