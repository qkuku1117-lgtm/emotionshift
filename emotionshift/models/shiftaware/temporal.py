"""Turn-based time encodings. No timestamps or gold emotion features are needed."""

import torch
from torch import nn


def distance_bucket(distance: torch.Tensor, buckets: int) -> torch.Tensor:
    """Logarithmic buckets: distances 0, 1, 2-3, 4-7, ...; cap large distances."""
    return torch.where(
        distance > 0,
        torch.floor(torch.log2(distance.float().clamp_min(1))).long() + 1,
        0,
    ).clamp_max(buckets - 1)


class TemporalEncoding(nn.Module):
    """Encode same-speaker gap and whether this is the speaker's first turn."""

    def __init__(self, time_dim: int, buckets: int):
        super().__init__()
        self.buckets = buckets
        self.gap_embedding = nn.Embedding(buckets, time_dim)
        self.first_embedding = nn.Embedding(2, time_dim)

    def forward(self, gaps: torch.Tensor, first: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.gap_embedding(distance_bucket(gaps, self.buckets)) + self.first_embedding(
            first.long()
        )
