"""PyTorch datasets and CPU batch collators over the shared utterance table.

Install with ``uv sync --extra training``. Import this module explicitly so
the sklearn pipeline does not require PyTorch. Tokenizers are injected and
must support the Hugging Face batch-call interface; no model is downloaded.

Samples contain targets separately from tokenized inputs. Gold history and
evaluation strata never enter the tokenizer. Pass exactly one split per dataset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import LABEL2ID
from .shifts import add_shift_labels

IGNORE_INDEX = -100

type Metadata = dict[str, str | int]
type Tokens = dict[str, torch.Tensor]


class Tokenizer(Protocol):
    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def truncation_side(self) -> str: ...

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


class UtteranceSample(TypedDict):
    text: str
    label: int
    shift_label: float
    shift_mask: bool
    speaker_id: int
    turn_gap: int
    requires_left_truncation: bool
    metadata: Metadata


class DialogueSample(TypedDict):
    texts: list[str]
    labels: list[int]
    shift_labels: list[float]
    shift_mask: list[bool]
    speaker_ids: list[int]
    turn_gaps: list[int]
    metadata: list[Metadata]


class _SupervisedBatch(TypedDict):
    tokens: Tokens
    labels: torch.Tensor
    shift_labels: torch.Tensor
    shift_mask: torch.Tensor
    speaker_ids: torch.Tensor
    turn_gaps: torch.Tensor


class UtteranceBatch(_SupervisedBatch):
    metadata: list[Metadata]


class DialogueBatch(_SupervisedBatch):
    lengths: torch.Tensor
    turn_mask: torch.Tensor
    metadata: list[list[Metadata]]


@dataclass(frozen=True)
class _Turn:
    text: str
    speaker: str
    speaker_id: int
    label: int
    shift_label: float
    shift_mask: bool
    turn_gap: int
    metadata: Metadata


def _prepare_dialogues(frame: pd.DataFrame) -> tuple[tuple[_Turn, ...], ...]:
    required = {"dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing utterance columns: {sorted(missing)}")
    if frame.empty or frame["split"].nunique() != 1:
        raise ValueError(
            "Pass a nonempty table with exactly one split, e.g. load(..., splits='train')"
        )
    if frame[list(required)].isna().any().any():
        raise ValueError("Utterance columns must not contain missing values")
    if not pd.api.types.is_integer_dtype(frame["turn"]) or (frame["turn"] < 0).any():
        raise ValueError("turn must contain nonnegative integers")
    for column in ("text", "speaker"):
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain strings")
    if not frame["emotion"].isin(LABEL2ID).all():
        raise ValueError("Emotion labels must use the shared seven-class label space")
    keys = ["dataset", "split", "dialogue_id"]
    if frame.duplicated([*keys, "turn"]).any():
        raise ValueError("Duplicate utterance keys")
    # Recompute supervision from ordered turns, rather than trusting stale metadata.
    frame = add_shift_labels(frame).reset_index(drop=True)
    metadata_columns = [*keys, "turn"] + [
        c for c in ("source_split", "source_dialogue_id") if c in frame.columns
    ]
    dialogues = []
    for _, dialogue in frame.groupby(keys, sort=False):
        speakers = {}
        turns = []
        for row in dialogue.sort_values("turn").to_dict("records"):
            speaker_id = speakers.setdefault(row["speaker"], len(speakers))
            turns.append(
                _Turn(
                    text=row["text"],
                    speaker=row["speaker"],
                    speaker_id=speaker_id,
                    label=LABEL2ID[row["emotion"]],
                    shift_label=float(row["stratum"] == "shift"),
                    shift_mask=row["stratum"] != "first",
                    turn_gap=0 if pd.isna(row["turn_gap"]) else int(row["turn_gap"]),
                    metadata={c: row[c] for c in metadata_columns},
                )
            )
        dialogues.append(tuple(turns))
    return tuple(dialogues)


class UtteranceDataset(Dataset[UtteranceSample]):
    """One target turn per sample, optionally with a causal context window.

    ``context_turns`` counts previous utterances (None means all history).
    ERC includes the target text; forecast excludes it and every future turn.
    Forecasting requires a positive window or None. The next speaker is supplied
    as a dialogue-local integer ID. Context text uses ``Speaker: text`` lines.
    Shift supervision is masked on each speaker's first turn.
    """

    def __init__(self, frame, *, context_turns: int | None = 0, task: str = "erc"):
        if context_turns is not None and (
            not isinstance(context_turns, int)
            or isinstance(context_turns, bool)
            or context_turns < 0
        ):
            raise ValueError("context_turns must be a nonnegative integer or None")
        if task not in ("erc", "forecast"):
            raise ValueError("task must be 'erc' or 'forecast'")
        if task == "forecast" and context_turns == 0:
            raise ValueError("Forecasting requires context_turns > 0 or None")
        self.context_turns = context_turns
        self.task = task
        self._dialogues = _prepare_dialogues(frame)
        self._index = tuple(
            (d, t) for d, dialogue in enumerate(self._dialogues) for t in range(len(dialogue))
        )

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[UtteranceSample]:
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: int) -> UtteranceSample:
        d, t = self._index[index]
        dialogue = self._dialogues[d]
        target = dialogue[t]
        uses_context = self.context_turns != 0
        if uses_context:
            start = 0 if self.context_turns is None else max(0, t - self.context_turns)
            end = t + 1 if self.task == "erc" else t
            text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in dialogue[start:end])
        else:
            text = target.text
        return {
            "text": text,
            "label": target.label,
            "shift_label": target.shift_label,
            "shift_mask": target.shift_mask,
            "speaker_id": target.speaker_id,
            "turn_gap": target.turn_gap,
            "requires_left_truncation": uses_context,
            "metadata": dict(target.metadata),
        }


class DialogueDataset(Dataset[DialogueSample]):
    """One complete ordered dialogue per sample; speaker IDs restart at zero.

    Shuffle dialogues with DataLoader, keeping turns inside each sample ordered.
    Dialogue models must enforce causal recurrence/attention. A forecast head
    must read history state before consuming the current utterance embedding.
    """

    def __init__(self, frame):
        self._dialogues = _prepare_dialogues(frame)

    def __len__(self) -> int:
        return len(self._dialogues)

    def __iter__(self) -> Iterator[DialogueSample]:
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: int) -> DialogueSample:
        turns = self._dialogues[index]
        return {
            "texts": [t.text for t in turns],
            "labels": [t.label for t in turns],
            "shift_labels": [t.shift_label for t in turns],
            "shift_mask": [t.shift_mask for t in turns],
            "speaker_ids": [t.speaker_id for t in turns],
            "turn_gaps": [t.turn_gap for t in turns],
            "metadata": [dict(t.metadata) for t in turns],
        }


def _tokenize(tokenizer: Tokenizer, texts: list[str], max_length: int) -> Tokens:
    tokens = dict(
        tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    )
    if "input_ids" not in tokens or "attention_mask" not in tokens:
        raise ValueError("Tokenizer must return input_ids and attention_mask")
    shape = tokens["input_ids"].shape
    if len(shape) != 2 or shape[0] != len(texts):
        raise ValueError("Tokenizer must return [samples, tokens] tensors")
    if any(
        not isinstance(value, torch.Tensor) or value.shape != shape for value in tokens.values()
    ):
        raise ValueError("Tokenizer outputs must be tensors with matching [samples, tokens] shape")
    if any(value.device.type != "cpu" for value in tokens.values()):
        raise ValueError(
            "Collators must create CPU tensors; move batches to the device in training"
        )
    return tokens


class UtteranceCollator:
    """Tokenize and pad to the longest input in the batch, capped by max_length.

    For context/forecast samples set tokenizer.truncation_side='left' so that
    truncation retains the target or most recent history. Tokenizers and
    collators should be constructed outside Dataset.__getitem__.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_length: int = 256):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[UtteranceSample]) -> UtteranceBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        if (
            any(s["requires_left_truncation"] for s in samples)
            and getattr(self.tokenizer, "truncation_side", None) != "left"
        ):
            raise ValueError("Set tokenizer.truncation_side='left' for causal context windows")
        return {
            "tokens": _tokenize(self.tokenizer, [s["text"] for s in samples], self.max_length),
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "shift_labels": torch.tensor([s["shift_label"] for s in samples], dtype=torch.float),
            "shift_mask": torch.tensor([s["shift_mask"] for s in samples], dtype=torch.bool),
            "speaker_ids": torch.tensor([s["speaker_id"] for s in samples], dtype=torch.long),
            "turn_gaps": torch.tensor([s["turn_gap"] for s in samples], dtype=torch.long),
            "metadata": [s["metadata"] for s in samples],
        }


class DialogueCollator:
    """Pad tokens and turns: tokens [B,T,L], targets/masks [B,T], lengths [B].

    Padding uses the tokenizer's pad_token_id, label -100 and speaker ID -1.
    turn_mask identifies actual turns; shift_mask also excludes first-speaker
    turns. Metadata contains only real turns. All returned tensors stay on CPU.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_length: int = 128):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Dialogue tokenization requires a pad_token_id")
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[DialogueSample]) -> DialogueBatch:
        if not samples or any(not s["texts"] for s in samples):
            raise ValueError("Batch must contain nonempty dialogues")
        lengths = torch.tensor([len(s["texts"]) for s in samples], dtype=torch.long)
        batch_size, turns = len(samples), int(lengths.max())
        flat = _tokenize(self.tokenizer, [t for s in samples for t in s["texts"]], self.max_length)
        tokens = {
            key: torch.full(
                (batch_size, turns, value.shape[1]),
                self.pad_token_id if key == "input_ids" else 0,
                dtype=value.dtype,
            )
            for key, value in flat.items()
        }
        batch: DialogueBatch = {
            "tokens": tokens,
            "lengths": lengths,
            "turn_mask": torch.arange(turns)[None, :] < lengths[:, None],
            "labels": torch.full((batch_size, turns), IGNORE_INDEX, dtype=torch.long),
            "shift_labels": torch.zeros((batch_size, turns), dtype=torch.float),
            "shift_mask": torch.zeros((batch_size, turns), dtype=torch.bool),
            "speaker_ids": torch.full((batch_size, turns), -1, dtype=torch.long),
            "turn_gaps": torch.zeros((batch_size, turns), dtype=torch.long),
            "metadata": [s["metadata"] for s in samples],
        }
        offset = 0
        for i, sample in enumerate(samples):
            n = len(sample["texts"])
            for key, value in tokens.items():
                value[i, :n] = flat[key][offset : offset + n]
            for key in ("labels", "shift_labels", "shift_mask", "speaker_ids", "turn_gaps"):
                batch[key][i, :n] = torch.tensor(sample[key], dtype=batch[key].dtype)
            offset += n
        return batch
