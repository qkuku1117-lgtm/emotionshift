"""Exercise batching, causal inputs and actual CPU optimization without downloads."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Install the training extra for PyTorch tests")

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from emotionshift.data.labels import LABEL2ID
from emotionshift.data.loaders import DATA_ROOT, load
from emotionshift.data.torch_data import (
    IGNORE_INDEX,
    DialogueCollator,
    DialogueDataset,
    UtteranceCollator,
    UtteranceDataset,
)


class TinyTokenizer:
    """Small offline tokenizer with a nonzero pad ID, also picklable for spawn."""

    pad_token_id = 9

    def __init__(self, truncation_side="right"):
        self.truncation_side = truncation_side

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        sequences = [[1, *[10 + len(word) % 10 for word in text.split()], 2] for text in texts]
        sequences = [
            s[-max_length:] if self.truncation_side == "left" else s[:max_length] for s in sequences
        ]
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence)
            mask[i, : len(sequence)] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "dataset": ["meld"] * 5,
            "split": ["train"] * 5,
            "dialogue_id": ["d0"] * 4 + ["d1"],
            "turn": [0, 1, 2, 3, 0],
            "speaker": ["A", "B", "A", "B", "C"],
            "text": ["PAST", "OTHER", "TARGET text", "FUTURE", "short"],
            "emotion": ["neutral", "joy", "anger", "joy", "sadness"],
            # Stale evaluation columns must never become features or override targets.
            "prev_emotion": ["fear"] * 5,
            "stratum": ["shift"] * 5,
        }
    ).sample(frac=1, random_state=0)


def test_context_and_forecast_have_no_future_or_gold_history(frame):
    erc = UtteranceDataset(frame, context_turns=1)
    forecast = UtteranceDataset(frame, context_turns=None, task="forecast")
    target = next(
        s for s in erc if s["metadata"]["dialogue_id"] == "d0" and s["metadata"]["turn"] == 2
    )
    predicted = next(s for s in forecast if s["metadata"] == target["metadata"])
    assert target["text"] == "B: OTHER\nA: TARGET text"
    assert predicted["text"] == "A: PAST\nB: OTHER"
    assert "TARGET" not in predicted["text"] and "FUTURE" not in predicted["text"]
    assert "fear" not in target["text"]
    assert target["label"] == LABEL2ID["anger"]
    assert target["shift_label"] == 1 and target["shift_mask"]
    assert target["turn_gap"] == 2 and target["speaker_id"] == 0


def test_changing_gold_labels_cannot_change_tokenized_inputs(frame):
    changed = frame.assign(emotion="fear")
    a = UtteranceDataset(frame, context_turns=None)
    b = UtteranceDataset(changed, context_turns=None)
    collate = UtteranceCollator(TinyTokenizer("left"))
    first, second = collate(list(a)), collate(list(b))
    for key in first["tokens"]:
        assert torch.equal(first["tokens"][key], second["tokens"][key])
    assert not torch.equal(first["labels"], second["labels"])


def test_dialogue_order_local_speakers_and_first_turn_masks(frame):
    dataset = DialogueDataset(frame)
    long = next(s for s in dataset if len(s["texts"]) == 4)
    short = next(s for s in dataset if len(s["texts"]) == 1)
    assert long["texts"] == ["PAST", "OTHER", "TARGET text", "FUTURE"]
    assert long["speaker_ids"] == [0, 1, 0, 1]
    assert short["speaker_ids"] == [0]
    assert long["shift_mask"] == [False, False, True, True]
    assert long["shift_labels"] == [0, 0, 1, 0]
    assert long["turn_gaps"] == [0, 0, 2, 2]

    batch = DialogueCollator(TinyTokenizer(), max_length=8)([short, long])
    assert batch["tokens"]["input_ids"].shape == (2, 4, 4)
    assert batch["lengths"].tolist() == [1, 4]
    assert batch["turn_mask"].tolist() == [[True, False, False, False], [True] * 4]
    assert batch["labels"][0, 1:].tolist() == [IGNORE_INDEX] * 3
    assert batch["speaker_ids"][0, 1:].tolist() == [-1] * 3
    assert not batch["shift_mask"][0].any()
    assert (batch["tokens"]["input_ids"][0, 1:] == 9).all()
    assert not batch["tokens"]["attention_mask"][0, 1:].any()
    assert len(batch["metadata"][0]) == 1


def test_context_requires_left_truncation_and_batch_length_is_capped(frame):
    dataset = UtteranceDataset(frame, context_turns=None)
    with pytest.raises(ValueError, match="truncation_side"):
        UtteranceCollator(TinyTokenizer())(list(dataset))
    batch = UtteranceCollator(TinyTokenizer("left"), max_length=3)(list(dataset))
    assert batch["tokens"]["input_ids"].shape == (5, 3)
    assert batch["labels"].dtype == torch.long
    assert batch["shift_labels"].dtype == torch.float
    assert batch["shift_mask"].dtype == torch.bool


@pytest.mark.parametrize("kind", ["utterance", "dialogue"])
def test_real_dataloader_and_cpu_training_step(frame, kind):
    torch.manual_seed(0)
    dialogue = kind == "dialogue"
    dataset = DialogueDataset(frame) if dialogue else UtteranceDataset(frame)
    collate = DialogueCollator(TinyTokenizer()) if dialogue else UtteranceCollator(TinyTokenizer())
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)))
    embedding = nn.Embedding(32, 8, padding_idx=9)
    classifier = nn.Linear(8, 7)
    shift_head = nn.Linear(8, 1)
    recurrent = nn.GRU(8, 8, batch_first=True)
    parameters = [*embedding.parameters(), *classifier.parameters(), *shift_head.parameters()]
    if dialogue:
        parameters += list(recurrent.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    tokens = batch["tokens"]
    mask = tokens["attention_mask"].unsqueeze(-1)
    features = (embedding(tokens["input_ids"]) * mask).sum(-2) / mask.sum(-2).clamp_min(1)
    if dialogue:
        packed = pack_padded_sequence(
            features, batch["lengths"], batch_first=True, enforce_sorted=False
        )
        output, _ = recurrent(packed)
        features, _ = pad_packed_sequence(
            output, batch_first=True, total_length=batch["turn_mask"].shape[1]
        )
    logits = classifier(features)
    loss = nn.functional.cross_entropy(logits.reshape(-1, 7), batch["labels"].reshape(-1))
    shift_loss = nn.functional.binary_cross_entropy_with_logits(
        shift_head(features).squeeze(-1),
        batch["shift_labels"],
        reduction="none",
    )
    loss = loss + (shift_loss * batch["shift_mask"]).sum() / batch["shift_mask"].sum().clamp_min(1)
    before = classifier.weight.detach().clone()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step()
    assert not torch.equal(before, classifier.weight)


def test_dataloader_with_spawn_worker(frame):
    loader = DataLoader(
        UtteranceDataset(frame),
        batch_size=2,
        collate_fn=UtteranceCollator(TinyTokenizer()),
        num_workers=1,
        multiprocessing_context="spawn",
    )
    batches = list(loader)
    assert sum(len(batch["labels"]) for batch in batches) == len(frame)


def test_mixed_splits_are_rejected(frame):
    mixed = frame.copy()
    mixed.loc[mixed.index[0], "split"] = "test"
    with pytest.raises(ValueError, match="exactly one split"):
        UtteranceDataset(mixed)


@pytest.mark.parametrize(
    "options",
    [
        {"context_turns": -1},
        {"context_turns": 1.5},
        {"task": "offline"},
        {"task": "forecast"},
    ],
)
def test_invalid_sampling_options(frame, options):
    with pytest.raises(ValueError):
        UtteranceDataset(frame, **options)


def test_duplicate_turns_are_rejected(frame):
    with pytest.raises(ValueError, match="Duplicate"):
        DialogueDataset(pd.concat([frame, frame.iloc[:1]]))


@pytest.mark.parametrize("dataset", ["meld", "dailydialog"])
def test_shared_training_data_reaches_pytorch_batches(dataset):
    if not (DATA_ROOT / dataset).exists():
        pytest.skip("Raw data not downloaded")
    frame = load(dataset, splits="train")
    samples = UtteranceDataset(frame, context_turns=3)
    batch = next(
        iter(
            DataLoader(
                samples,
                batch_size=8,
                collate_fn=UtteranceCollator(TinyTokenizer("left")),
            )
        )
    )
    assert batch["labels"].shape == (8,)
    assert batch["tokens"]["input_ids"].shape[0] == 8
    assert all(m["dataset"] == dataset and m["split"] == "train" for m in batch["metadata"])
