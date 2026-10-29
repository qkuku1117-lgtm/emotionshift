"""Causal retrieval over turns within speakers, followed by attention over speakers."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .temporal import distance_bucket


@dataclass
class AttentionResult:
    context: torch.Tensor
    turn_weights: torch.Tensor
    null_weight: torch.Tensor


class CrossSpeakerAttention(nn.Module):
    """Retrieve only from other speakers' earlier turns.

    The caller supplies history strictly before the target. Speaker IDs only
    group keys; they are never learned identity embeddings. The null candidate
    is always available, including when all history belongs to the target.
    Returned weights are retrieval diagnostics, not causal explanations.
    """

    def __init__(self, hidden_dim: int, num_heads: int, distance_buckets: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.distance_buckets = distance_buckets
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.speaker_query = nn.Linear(hidden_dim, hidden_dim)
        self.speaker_key = nn.Linear(hidden_dim, hidden_dim)
        self.distance_bias = nn.Embedding(distance_buckets, num_heads)
        self.null_value = nn.Parameter(torch.zeros(hidden_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        history_speakers: torch.Tensor,
        history_mask: torch.Tensor,
        current_speakers: torch.Tensor,
        distances: torch.Tensor,
        num_speakers: int,
    ) -> AttentionResult:
        batch, turns, hidden = history.shape
        if turns == 0:
            return AttentionResult(
                self.output(self.null_value.expand(batch, -1)),
                query.new_zeros(batch, 0),
                query.new_ones(batch),
            )
        heads, dim = self.num_heads, self.head_dim
        q = self.query(query).reshape(batch, heads, dim)
        k = self.key(history).reshape(batch, turns, heads, dim).transpose(1, 2)
        v = self.value(history).reshape(batch, turns, heads, dim).transpose(1, 2)
        scores = torch.einsum("bhd,bhld->bhl", q, k) / math.sqrt(dim)
        bias = self.distance_bias(distance_bucket(distances, self.distance_buckets))
        scores = scores + bias.permute(0, 2, 1)
        groups = torch.arange(num_speakers, device=query.device)
        valid = (
            (history_speakers[:, None, :] == groups[None, :, None])
            & history_mask[:, None, :]
            & (groups[None, :, None] != current_speakers[:, None, None])
        )
        # Finite masking plus explicit renormalization handles empty speaker groups.
        weights = scores[:, :, None, :].expand(-1, -1, num_speakers, -1)
        weights = weights.float().masked_fill(~valid[:, None], torch.finfo(torch.float32).min)
        weights = weights.softmax(-1) * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        summaries = torch.einsum("bhsl,bhld->bhsd", weights.to(v.dtype), v)
        summaries = summaries.transpose(1, 2).reshape(batch, num_speakers, hidden)
        candidates = torch.cat((summaries, self.null_value.expand(batch, 1, -1)), dim=1)
        sq = self.speaker_query(query).reshape(batch, heads, dim)
        sk = self.speaker_key(candidates).reshape(batch, -1, heads, dim).transpose(1, 2)
        speaker_scores = torch.einsum("bhd,bhsd->bhs", sq, sk) / math.sqrt(dim)
        available = torch.cat(
            (valid.any(-1), torch.ones(batch, 1, device=query.device, dtype=torch.bool)), dim=1
        )
        speaker_weights = speaker_scores.float().masked_fill(~available[:, None], -torch.inf)
        speaker_weights = speaker_weights.softmax(-1)
        values = candidates.reshape(batch, -1, heads, dim).transpose(1, 2)
        context = torch.einsum("bhs,bhsd->bhd", speaker_weights.to(values.dtype), values)
        turn_weights = (speaker_weights[:, :, :-1, None] * weights).sum(2).mean(1)
        return AttentionResult(
            self.output(context.flatten(1)), turn_weights, speaker_weights[:, :, -1].mean(1)
        )
