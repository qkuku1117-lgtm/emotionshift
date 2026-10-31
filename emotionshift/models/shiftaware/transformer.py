"""Parallel causal dynamics: a small transformer over utterance embeddings.

Replaces the recurrent loop (global memory, speaker GRU, cross-speaker retrieval,
fed-back predictions and the prior/evidence gate) with one pass over all turns.
Turn t attends to turns 0..t; speakers enter only as a same/other attention bias,
so there are no identity embeddings and renaming speakers changes nothing.
The forecast for turn t attends to turns strictly before t, plus a null slot,
and never reads utterance t. With fusion="shift_gate" that forecast is mixed
back in as a prior, weighted by 1 - P(shift).
"""

import math

import torch
from torch import nn

from .config import ShiftAwareConfig
from .heads import projection
from .temporal import TemporalEncoding, distance_bucket


class BiasedAttention(nn.Module):
    """Multi-head attention with learned same-speaker and distance biases."""

    def __init__(self, hidden_dim: int, num_heads: int, distance_buckets: int, *, null: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.distance_buckets = distance_buckets
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key_value = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.speaker_bias = nn.Embedding(2, num_heads)
        self.distance_bias = nn.Embedding(distance_buckets, num_heads)
        self.null = nn.Parameter(torch.zeros(2, hidden_dim)) if null else None
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        allowed: torch.Tensor,
        same_speaker: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """queries [B,Q,D], keys [B,K,D], masks [B,Q,K]; returns output and weights [B,Q,K(+1)]."""
        batch, length, hidden = queries.shape
        heads, dim = self.num_heads, self.head_dim
        q = self.query(queries).reshape(batch, length, heads, dim).transpose(1, 2)
        k, v = self.key_value(keys).reshape(batch, -1, 2, heads, dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
        bias = self.speaker_bias(same_speaker.long()) + self.distance_bias(
            distance_bucket(distances, self.distance_buckets)
        )
        scores = scores + bias.permute(0, 3, 1, 2)
        if self.null is not None:
            null_k, null_v = self.null.reshape(2, 1, heads, 1, dim).expand(-1, batch, -1, -1, -1)
            scores = torch.cat((scores, q @ null_k.transpose(-1, -2) / math.sqrt(dim)), -1)
            v = torch.cat((v, null_v), 2)
            allowed = nn.functional.pad(allowed, (0, 1), value=True)
        weights = scores.float().masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
        context = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(batch, length, hidden)
        return self.output(context), weights.mean(1)


class ContextLayer(nn.Module):
    """Pre-norm transformer layer with causal, speaker-aware attention."""

    def __init__(self, config: ShiftAwareConfig):
        super().__init__()
        d = config.hidden_dim
        self.attention_norm = nn.LayerNorm(d)
        self.attention = BiasedAttention(d, config.num_heads, config.distance_buckets, null=False)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * d, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h, allowed, same_speaker, distances):  # ty: ignore[invalid-mutable-override]
        normalized = self.attention_norm(h)
        attended, weights = self.attention(normalized, normalized, allowed, same_speaker, distances)
        h = h + self.dropout(attended)
        return h + self.dropout(self.feedforward(h)), weights


class TransformerDynamics(nn.Module):
    """Recognition, shift detection and history-only forecasts in one parallel pass.

    context_layers=0 is the utterance-only baseline: recognition and shift
    detection then read the current utterance alone, and forecasts still read
    the earlier utterances.
    """

    def __init__(self, config: ShiftAwareConfig):
        super().__init__()
        d, td, c = config.hidden_dim, config.time_dim, config.num_emotions
        self.temporal = TemporalEncoding(td, config.distance_buckets)
        self.time_projection = nn.Linear(td, d)
        self.layers = nn.ModuleList(ContextLayer(config) for _ in range(config.context_layers))
        self.norm = nn.LayerNorm(d)
        self.emotion = nn.Sequential(projection(d, d, config.dropout), nn.Linear(d, c))
        self.shift = nn.Sequential(projection(d, d, config.dropout), nn.Linear(d, 1))
        self.forecast_query = nn.Parameter(torch.zeros(d))
        self.forecast_attention = BiasedAttention(
            d, config.num_heads, config.distance_buckets, null=True
        )
        self.forecast_norm = nn.LayerNorm(d)
        self.forecast = projection(d, d, config.dropout)
        self.forecast_emotion = nn.Linear(d, c)
        self.forecast_shift = nn.Linear(d, 1)
        self.fusion = config.fusion

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        turn_mask: torch.Tensor,
        turn_gaps: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, turns, _ = x.shape
        positions = torch.arange(turns, device=x.device)
        earlier = positions[:, None] > positions[None, :]
        same = speakers[:, :, None] == speakers[:, None, :]
        keys = turn_mask[:, None, :]
        distances = (positions[:, None] - positions[None, :]).clamp_min(0).expand(batch, -1, -1)

        previous_own = same & earlier & keys
        last = torch.where(previous_own, positions, -1).amax(-1)
        first = last < 0
        gaps = (positions - last).masked_fill(first, 0)
        if turn_gaps is not None:
            gaps = turn_gaps.masked_fill(first | ~turn_mask, 0)
        time = self.time_projection(self.temporal(gaps, first).to(x.dtype))

        # Each turn reads itself and earlier real turns; padded rows read themselves.
        allowed = ((earlier | positions[:, None].eq(positions[None, :])) & keys) | torch.eye(
            turns, dtype=torch.bool, device=x.device
        )
        h = x + time if self.layers else x
        weights = None
        for layer in self.layers:
            h, weights = layer(h, allowed, same, distances)
        h = self.norm(h)

        query = (self.forecast_query + time).to(h.dtype)
        forecast, forecast_weights = self.forecast_attention(
            query, h, earlier & keys, same, distances
        )
        forecast = self.forecast(self.forecast_norm(query + forecast))
        mask = turn_mask[..., None]
        emotion_logits = self.emotion(h).float()
        shift_logits = self.shift(h).squeeze(-1)
        forecast_logits = self.forecast_emotion(forecast)
        log_probs = emotion_logits.log_softmax(-1)
        evidence_weight = torch.ones_like(shift_logits, dtype=torch.float)
        if self.fusion == "shift_gate":
            evidence_weight = shift_logits.float().sigmoid()
            prior = forecast_logits.float().log_softmax(-1)
            log_probs = (log_probs + (1 - evidence_weight)[..., None] * prior).log_softmax(-1)
        return {
            "emotion_log_probs": log_probs.masked_fill(~mask, 0),
            "emotion_logits": emotion_logits.masked_fill(~mask, 0),
            "shift_logits": shift_logits.masked_fill(~turn_mask, 0),
            "forecast_emotion_logits": forecast_logits.masked_fill(~mask, 0),
            "forecast_shift_logits": self.forecast_shift(forecast)
            .squeeze(-1)
            .masked_fill(~turn_mask, 0),
            "shift_mask": ~first & turn_mask,
            "mixing_weights": evidence_weight * turn_mask,
            "attention_weights": weights,
            "forecast_attention_weights": forecast_weights,
        }
