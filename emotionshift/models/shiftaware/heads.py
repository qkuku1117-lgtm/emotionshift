"""Emotion priors, history-only forecasts, evidence, shift detection and mixing."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ShiftAwareConfig


def projection(inputs: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout))


@dataclass
class Prediction:
    emotion_log_probs: torch.Tensor
    prior_logits: torch.Tensor
    evidence_logits: torch.Tensor
    shift_logits: torch.Tensor
    mixing_weights: torch.Tensor


class PredictionHeads(nn.Module):
    def __init__(self, config: ShiftAwareConfig):
        super().__init__()
        d, c, td = config.hidden_dim, config.num_emotions, config.time_dim
        self.gate_kind = config.gate
        self.prior = nn.Sequential(projection(d + c + td, d, config.dropout), nn.Linear(d, c))
        self.forecast = projection(3 * d + c + td, d, config.dropout)
        self.forecast_emotion = nn.Linear(d, c)
        self.forecast_shift = nn.Linear(d, 1)
        self.evidence = projection(4 * d + c + td, d, config.dropout)
        self.evidence_emotion = nn.Linear(d, c)
        self.shift = nn.Linear(d, 1)
        self.gate = nn.Linear(d + 1, 1) if config.gate == "learned" else None

    def forecast_from_history(self, speaker, global_summary, interaction, previous_probs, time):
        inputs = torch.cat((speaker, global_summary, interaction, previous_probs, time), dim=-1)
        hidden = self.forecast(inputs)
        return self.forecast_emotion(hidden), self.forecast_shift(hidden).squeeze(-1)

    def forward(self, utterance, speaker, global_summary, interaction, previous_probs, time):  # ty: ignore[invalid-mutable-override]
        prior_logits = self.prior(torch.cat((speaker, previous_probs, time), dim=-1))
        hidden = self.evidence(
            torch.cat((utterance, speaker, global_summary, interaction, previous_probs, time), -1)
        )
        evidence_logits = self.evidence_emotion(hidden)
        shift_logits = self.shift(hidden).squeeze(-1)
        gate_logits = (
            shift_logits
            if self.gate is None
            else self.gate(torch.cat((hidden, shift_logits.sigmoid()[:, None]), -1)).squeeze(-1)
        )
        # Mix probabilities in log space to avoid log(0) when a gate saturates.
        log_probs = torch.logaddexp(
            F.logsigmoid(-gate_logits.float())[:, None] + prior_logits.float().log_softmax(-1),
            F.logsigmoid(gate_logits.float())[:, None] + evidence_logits.float().log_softmax(-1),
        )
        return Prediction(
            log_probs, prior_logits, evidence_logits, shift_logits, gate_logits.sigmoid()
        )
