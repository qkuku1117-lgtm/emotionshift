"""Joint supervised loss. Gold labels enter only here, never the model forward."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from emotionshift.data.labels import EMOTIONS

from .model import ShiftAwareOutput


class ShiftAwareLoss(nn.Module):
    class_weights: torch.Tensor | None

    def __init__(
        self,
        *,
        class_weights: torch.Tensor | None = None,
        shift_weight: float = 1.0,
        forecast_weight: float = 0.5,
    ):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (shift_weight, forecast_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if class_weights is not None:
            if class_weights.shape != (len(EMOTIONS),):
                raise ValueError("class_weights must have seven entries")
            if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
                raise ValueError("class_weights must be finite and positive")
            class_weights = class_weights.detach().clone().float()
        self.register_buffer("class_weights", class_weights)
        self.shift_weight = shift_weight
        self.forecast_weight = forecast_weight

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        output: ShiftAwareOutput,
        labels: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid = output.turn_mask
        if labels.shape != valid.shape or shift_labels.shape != valid.shape:
            raise ValueError("Targets must have shape [B,T]")
        eligible = output.shift_mask & valid
        if shift_mask is not None:
            if shift_mask.shape != valid.shape or shift_mask.dtype != torch.bool:
                raise ValueError("shift_mask must be boolean [B,T]")
            eligible = eligible & shift_mask
        emotion = F.nll_loss(
            output.emotion_log_probs[valid], labels[valid], weight=self.class_weights
        )
        forecast_emotion = F.cross_entropy(
            output.forecast_emotion_logits[valid].float(), labels[valid], weight=self.class_weights
        )
        if eligible.any():
            targets = shift_labels[eligible].float()
            if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
                raise ValueError("Eligible shift targets must be finite and in [0,1]")
            shift = F.binary_cross_entropy_with_logits(
                output.shift_logits[eligible].float(), targets
            )
            forecast_shift = F.binary_cross_entropy_with_logits(
                output.forecast_shift_logits[eligible].float(), targets
            )
        else:
            shift = output.shift_logits.sum() * 0
            forecast_shift = output.forecast_shift_logits.sum() * 0
        total = (
            emotion
            + self.shift_weight * shift
            + self.forecast_weight * (forecast_emotion + forecast_shift)
        )
        return {
            "total": total,
            "emotion": emotion,
            "shift": shift,
            "forecast_emotion": forecast_emotion,
            "forecast_shift": forecast_shift,
        }
