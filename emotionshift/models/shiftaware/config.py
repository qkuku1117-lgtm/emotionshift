"""Architecture settings, independent of datasets, tokenizers and training loops."""

from dataclasses import dataclass
from typing import Literal

from emotionshift.data.labels import EMOTIONS


@dataclass(frozen=True)
class ShiftAwareConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    time_dim: int = 32
    num_heads: int = 4
    distance_buckets: int = 16
    dropout: float = 0.1
    global_memory: Literal["gated_delta", "gru"] = "gated_delta"
    gate: Literal["learned", "tied"] = "learned"
    # "forecast" reuses the history-only forecast head as the inertia prior.
    prior: Literal["speaker", "forecast"] = "speaker"
    # "local" limits evidence to the utterance and other speakers' turns, leaving
    # the speaker's own trajectory to the prior; the shift head still sees both.
    evidence: Literal["full", "local"] = "full"
    # "probability": (1 - beta) * P(prior) + beta * P(evidence).
    # "product": P(evidence) * P(prior) ** (1 - beta), renormalized.
    mixing: Literal["probability", "product"] = "probability"
    num_emotions: int = len(EMOTIONS)

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "time_dim", "num_heads", "distance_buckets"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.global_memory not in ("gated_delta", "gru"):
            raise ValueError("global_memory must be 'gated_delta' or 'gru'")
        if self.gate not in ("learned", "tied"):
            raise ValueError("gate must be 'learned' or 'tied'")
        if self.prior not in ("speaker", "forecast"):
            raise ValueError("prior must be 'speaker' or 'forecast'")
        if self.evidence not in ("full", "local"):
            raise ValueError("evidence must be 'full' or 'local'")
        if self.mixing not in ("probability", "product"):
            raise ValueError("mixing must be 'probability' or 'product'")
        if self.num_emotions != len(EMOTIONS):
            raise ValueError("Use the shared seven-emotion label space")
