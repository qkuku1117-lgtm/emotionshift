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
        self.mixing = config.mixing
        self.prior = (
            nn.Sequential(projection(d + c + td, d, config.dropout), nn.Linear(d, c))
            if config.prior == "speaker"
            else None
        )
        self.forecast = projection(3 * d + c + td, d, config.dropout)
        self.forecast_emotion = nn.Linear(d, c)
        self.forecast_shift = nn.Linear(d, 1)
        full = 4 * d + c + td
        local = config.evidence == "local"
        self.evidence = projection(2 * d + td if local else full, d, config.dropout)
        self.evidence_emotion = nn.Linear(d, c)
        # With local evidence the shift detector needs its own view of the history.
        self.shift_hidden = projection(full, d, config.dropout) if local else None
        self.shift = nn.Linear(d, 1)
        self.gate = nn.Linear(d + 1, 1) if config.gate == "learned" else None

    def forecast_from_history(self, speaker, global_summary, interaction, previous_probs, time):
        inputs = torch.cat((speaker, global_summary, interaction, previous_probs, time), dim=-1)
        hidden = self.forecast(inputs)
        return self.forecast_emotion(hidden), self.forecast_shift(hidden).squeeze(-1)

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        utterance,
        speaker,
        global_summary,
        interaction,
        previous_probs,
        time,
        forecast_logits,
    ):
        prior_logits = (
            forecast_logits
            if self.prior is None
            else self.prior(torch.cat((speaker, previous_probs, time), dim=-1))
        )
        full = torch.cat(
            (utterance, speaker, global_summary, interaction, previous_probs, time), -1
        )
        if self.shift_hidden is None:
            hidden = shift_hidden = self.evidence(full)
        else:
            hidden = self.evidence(torch.cat((utterance, interaction, time), -1))
            shift_hidden = self.shift_hidden(full)
        evidence_logits = self.evidence_emotion(hidden)
        shift_logits = self.shift(shift_hidden).squeeze(-1)
        gate_logits = (
            shift_logits
            if self.gate is None
            else self.gate(torch.cat((shift_hidden, shift_logits.sigmoid()[:, None]), -1)).squeeze(
                -1
            )
        )
        prior_log_probs = prior_logits.float().log_softmax(-1)
        evidence_log_probs = evidence_logits.float().log_softmax(-1)
        if self.mixing == "product":
            # Evidence reweighted by the prior raised to (1 - beta): the prior can
            # break ties on stable turns but cannot override confident evidence.
            weight = F.logsigmoid(-gate_logits.float()).exp()[:, None]
            log_probs = (evidence_log_probs + weight * prior_log_probs).log_softmax(-1)
        else:
            # Mix probabilities in log space to avoid log(0) when a gate saturates.
            log_probs = torch.logaddexp(
                F.logsigmoid(-gate_logits.float())[:, None] + prior_log_probs,
                F.logsigmoid(gate_logits.float())[:, None] + evidence_log_probs,
            )
        return Prediction(
            log_probs, prior_logits, evidence_logits, shift_logits, gate_logits.sigmoid()
        )
