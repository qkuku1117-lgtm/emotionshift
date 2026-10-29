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
        if self.num_emotions != len(EMOTIONS):
            raise ValueError("Use the shared seven-emotion label space")
