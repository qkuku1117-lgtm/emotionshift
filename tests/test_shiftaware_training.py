"""Exercise optimizer groups, scheduler updates and accumulated training."""

import pytest

torch = pytest.importorskip("torch", reason="Install the training extra")

import pandas as pd
from torch import nn

from emotionshift.models.shiftaware import ShiftAware, ShiftAwareConfig, ShiftAwareLoss
from emotionshift.models.shiftaware.training import (
    TrainingConfig,
    emotion_class_weights,
    make_optimizer,
    make_scheduler,
    selection_score,
    train_epoch,
)


class TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(16, 12)
        self.dynamics = ShiftAware(
            ShiftAwareConfig(input_dim=12, hidden_dim=16, time_dim=4, num_heads=2, dropout=0)
        )

    def forward(self, tokens, speaker_ids, turn_mask, turn_gaps):  # ty: ignore[invalid-mutable-override]
        embeddings = self.encoder(tokens["input_ids"]).mean(-2)
        return self.dynamics(embeddings, speaker_ids, turn_mask, turn_gaps)


def test_accumulated_training_steps_scheduler_on_optimizer_updates():
    torch.manual_seed(5)
    model = TinyTextModel()
    settings = TrainingConfig(
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        warmup_ratio=0,
        min_lr_ratio=0.1,
    )
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, 2, settings)
    encoder_groups = [
        group
        for group in optimizer.param_groups
        if any(parameter is model.encoder.weight for parameter in group["params"])
    ]
    assert len(encoder_groups) == 1
    assert encoder_groups[0]["weight_decay"] == settings.weight_decay
    assert any(group["weight_decay"] == 0 for group in optimizer.param_groups)
    batch = {
        "tokens": {"input_ids": torch.tensor([[[1, 2], [3, 4], [5, 6]]])},
        "speaker_ids": torch.tensor([[0, 1, 0]]),
        "turn_mask": torch.ones(1, 3, dtype=torch.bool),
        "turn_gaps": torch.tensor([[0, 0, 2]]),
        "labels": torch.tensor([[0, 1, 2]]),
        "shift_labels": torch.tensor([[0.0, 0.0, 1.0]]),
        "shift_mask": torch.tensor([[False, False, True]]),
    }
    before = model.dynamics.heads.evidence_emotion.weight.detach().clone()
    logged = []
    result = train_epoch(
        model,
        [batch, batch, batch],
        ShiftAwareLoss(),
        optimizer,
        scheduler,
        torch.device("cpu"),
        settings,
        log=logged.append,
    )
    assert result["updates"] == 2
    assert [row["update"] for row in logged] == [2]
    assert logged[0]["of"] == 2
    assert set(logged[0]["lr"]) == {"encoder", "dynamics"}
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert scheduler.last_epoch == 2
    assert not torch.equal(before, model.dynamics.heads.evidence_emotion.weight)
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
        assert group["lr"] == pytest.approx(base_lr * settings.min_lr_ratio)


def test_log_every_must_be_nonnegative():
    with pytest.raises(ValueError, match="log_every"):
        TrainingConfig(log_every=-1)


def test_class_weights_upweight_rare_emotions():
    from emotionshift.data.labels import EMOTIONS

    emotions = pd.Series(["neutral"] * 40 + [e for e in EMOTIONS for _ in range(10)])
    assert emotion_class_weights(emotions, "none") is None
    inverse = emotion_class_weights(emotions, "inverse")
    soft = emotion_class_weights(emotions, "sqrt_inverse")
    assert inverse is not None and soft is not None
    assert inverse[0] < inverse[1] and inverse[1] == inverse[6]
    assert torch.allclose(soft, inverse.sqrt())
    with pytest.raises(ValueError, match="No training examples"):
        emotion_class_weights(pd.Series(["neutral", "joy"]), "inverse")
    with pytest.raises(ValueError, match="class_weighting"):
        TrainingConfig(class_weighting="balanced")


def test_selection_prefers_epochs_that_also_learn_rare_emotions():
    early = {"micro_f1_no_neutral": 0.601, "macro_f1": 0.367}
    later = {"micro_f1_no_neutral": 0.591, "macro_f1": 0.437}
    headline = "micro_f1_no_neutral"
    assert selection_score(early, headline, "headline") > selection_score(
        later, headline, "headline"
    )
    assert selection_score(later, headline, "headline_macro") == pytest.approx(0.514)
    assert selection_score(later, headline, "headline_macro") > selection_score(
        early, headline, "headline_macro"
    )
    assert TrainingConfig().selection == "headline_macro"
    with pytest.raises(ValueError, match="selection"):
        TrainingConfig(selection="macro")
    with pytest.raises(ValueError, match="selection"):
        selection_score(early, headline, "macro")
