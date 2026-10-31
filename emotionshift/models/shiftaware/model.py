"""Causal dialogue orchestration over cached, utterance-only embeddings."""

from dataclasses import dataclass, fields

import torch
from torch import nn

from .attention import CrossSpeakerAttention
from .config import ShiftAwareConfig
from .heads import PredictionHeads, projection
from .memory import GatedDeltaMemory, GRUMemory, SpeakerMemory
from .temporal import TemporalEncoding
from .transformer import TransformerDynamics


@dataclass
class ShiftForecast:
    """History-only prediction [B,C] / [B]; shift_mask excludes unseen speakers."""

    emotion_probs: torch.Tensor
    shift_probs: torch.Tensor
    shift_mask: torch.Tensor


@dataclass
class ShiftAwareOutput:
    """All tensors are [B,T,...]; padding is zero and masks identify valid rows.

    emotion_log_probs is a normalized log distribution on real turns, not raw
    logits. Use the accompanying loss rather than applying cross_entropy again.
    Attention diagnostics are [B,T,T] with zero weight on current/future turns.
    """

    emotion_log_probs: torch.Tensor
    prior_logits: torch.Tensor
    evidence_logits: torch.Tensor
    shift_logits: torch.Tensor
    forecast_emotion_logits: torch.Tensor
    forecast_shift_logits: torch.Tensor
    mixing_weights: torch.Tensor
    turn_mask: torch.Tensor
    shift_mask: torch.Tensor
    attention_weights: torch.Tensor | None = None
    forecast_attention_weights: torch.Tensor | None = None
    attention_null_weights: torch.Tensor | None = None

    @property
    def emotion_probs(self) -> torch.Tensor:
        return self.emotion_log_probs.exp() * self.turn_mask[..., None]

    @property
    def shift_probs(self) -> torch.Tensor:
        """First-speaker turns have no defined shift; consult shift_mask."""
        return self.shift_logits.sigmoid() * self.shift_mask

    @property
    def forecast_emotion_probs(self) -> torch.Tensor:
        return self.forecast_emotion_logits.float().softmax(-1) * self.turn_mask[..., None]

    @property
    def forecast_shift_probs(self) -> torch.Tensor:
        return self.forecast_shift_logits.sigmoid() * self.shift_mask


class ShiftAware(nn.Module):
    """Recognize and forecast emotions without reading labels or future turns.

    Inputs: embeddings [B,T,input_dim], dialogue-local speaker IDs [B,T],
    optional right-padding mask [B,T], and optional same-speaker gaps [B,T].
    Derive gaps from turn positions when omitted. IDs are nonnegative for real
    turns; padding convention is -1. States reset on every forward call.

    Encoding each turn independently is essential: cached embeddings containing
    future context would invalidate causality regardless of this module's masks.
    """

    def __init__(self, config: ShiftAwareConfig | None = None):
        super().__init__()
        self.config = config or ShiftAwareConfig()
        cfg = self.config
        d, td, c = cfg.hidden_dim, cfg.time_dim, cfg.num_emotions
        self.utterance_projection = nn.Sequential(
            nn.Linear(cfg.input_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)
        )
        if cfg.dynamics == "transformer":
            self.transformer = TransformerDynamics(cfg)
            return
        self.temporal = TemporalEncoding(td, cfg.distance_buckets)
        self.attention = CrossSpeakerAttention(d, cfg.num_heads, cfg.distance_buckets)
        self.forecast_query = projection(2 * d + c + td, d, cfg.dropout)
        self.evidence_query = projection(3 * d + td, d, cfg.dropout)
        self.global_input = projection(2 * d + td, d, cfg.dropout)
        self.global_memory = (
            GatedDeltaMemory(d, cfg.num_heads)
            if cfg.global_memory == "gated_delta"
            else GRUMemory(d)
        )
        self.speaker_memory = SpeakerMemory(d, td)
        self.heads = PredictionHeads(cfg)

    def _validate(self, embeddings, speaker_ids, turn_mask, turn_gaps):
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.input_dim:
            raise ValueError("embeddings must have shape [B,T,input_dim]")
        if not embeddings.is_floating_point() or min(embeddings.shape[:2]) == 0:
            raise ValueError("embeddings must be floating point with nonempty B and T")
        shape = embeddings.shape[:2]
        if speaker_ids.shape != shape or speaker_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("speaker_ids must be integer [B,T]")
        if speaker_ids.device != embeddings.device:
            raise ValueError("Inputs must be on the same device")
        if turn_mask is None:
            turn_mask = speaker_ids >= 0
        if turn_mask.shape != shape or turn_mask.dtype != torch.bool:
            raise ValueError("turn_mask must be boolean [B,T]")
        if turn_mask.device != embeddings.device:
            raise ValueError("Inputs must be on the same device")
        if not turn_mask.any(1).all() or (turn_mask[:, 1:] & ~turn_mask[:, :-1]).any():
            raise ValueError("Each dialogue must be nonempty and right padded")
        if (speaker_ids[turn_mask] < 0).any():
            raise ValueError("Real turns require nonnegative speaker IDs")
        if turn_gaps is not None:
            if turn_gaps.shape != shape or turn_gaps.dtype not in (torch.int32, torch.int64):
                raise ValueError("turn_gaps must be integer [B,T]")
            if turn_gaps.device != embeddings.device or (turn_gaps[turn_mask] < 0).any():
                raise ValueError("turn_gaps must be nonnegative and on the input device")
        return turn_mask

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
        turn_mask: torch.Tensor | None = None,
        turn_gaps: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> ShiftAwareOutput:
        turn_mask = self._validate(embeddings, speaker_ids, turn_mask, turn_gaps)
        if self.config.dynamics == "transformer":
            return self._forward_transformer(
                embeddings, speaker_ids, turn_mask, turn_gaps, return_attention
            )
        batch, turns = speaker_ids.shape
        cfg = self.config
        num_speakers = int(speaker_ids[turn_mask].max()) + 1
        # Ignore arbitrary padding values, including NaN embeddings and large IDs.
        x = self.utterance_projection(embeddings.masked_fill(~turn_mask[..., None], 0))
        ids = speaker_ids.masked_fill(~turn_mask, 0).long()
        states = x.new_zeros(batch, num_speakers, cfg.hidden_dim)
        previous_probs = x.new_full((batch, num_speakers, cfg.num_emotions), 1 / cfg.num_emotions)
        last_turns = torch.full((batch, num_speakers), -1, device=x.device, dtype=torch.long)
        global_state = self.global_memory.initial_state(x[:, 0])
        rows = torch.arange(batch, device=x.device)
        output_names = (
            "emotion_log_probs",
            "prior_logits",
            "evidence_logits",
            "shift_logits",
            "forecast_emotion_logits",
            "forecast_shift_logits",
            "mixing_weights",
            "shift_mask",
        )
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in output_names}
        attention_rows, forecast_rows, null_rows = [], [], []
        for t in range(turns):
            active, speaker = turn_mask[:, t], ids[:, t]
            own = states[rows, speaker]
            previous = previous_probs[rows, speaker]
            last = last_turns[rows, speaker]
            first = last < 0
            gaps = (t - last).masked_fill(first, 0)
            if turn_gaps is not None:
                gaps = turn_gaps[:, t].masked_fill(first | ~active, 0)
            time = self.temporal(gaps, first).to(x.dtype)
            distances = (t - torch.arange(t, device=x.device)).expand(batch, -1)
            retrieval_inputs = (
                x[:, :t],
                ids[:, :t],
                turn_mask[:, :t],
                speaker,
                distances,
                num_speakers,
            )

            # Forecast reads history before any access to the target embedding.
            forecast_query = self.forecast_query(
                torch.cat((own, global_state.summary, previous, time), -1)
            )
            forecast_context = self.attention(forecast_query, *retrieval_inputs)
            forecast_emotion, forecast_shift = self.heads.forecast_from_history(
                own, global_state.summary, forecast_context.context, previous, time
            )

            query = self.evidence_query(torch.cat((x[:, t], own, global_state.summary, time), -1))
            interaction = self.attention(query, *retrieval_inputs)
            prediction = self.heads(
                x[:, t],
                own,
                global_state.summary,
                interaction.context,
                previous,
                time,
                forecast_emotion,
            )
            values = {
                "emotion_log_probs": prediction.emotion_log_probs,
                "prior_logits": prediction.prior_logits,
                "evidence_logits": prediction.evidence_logits,
                "shift_logits": prediction.shift_logits,
                "forecast_emotion_logits": forecast_emotion,
                "forecast_shift_logits": forecast_shift,
                "mixing_weights": prediction.mixing_weights,
                "shift_mask": ~first,
            }
            for name, value in values.items():
                mask = active if value.ndim == 1 else active[:, None]
                collected[name].append(value.masked_fill(~mask, 0))
            if return_attention:
                attention_rows.append(
                    nn.functional.pad(interaction.turn_weights * active[:, None], (0, turns - t))
                )
                forecast_rows.append(
                    nn.functional.pad(
                        forecast_context.turn_weights * active[:, None], (0, turns - t)
                    )
                )
                null_rows.append(interaction.null_weight * active)

            # Current evidence enters recurrent state only after both predictions.
            global_inputs = self.global_input(torch.cat((x[:, t], own, time), -1))
            global_state = self.global_memory(global_inputs, global_state).keep_inactive(
                global_state, active
            )
            updated = self.speaker_memory(
                x[:, t], interaction.context, global_state.summary.to(x.dtype), time, own
            )
            index = speaker[:, None, None]
            states = states.scatter(
                1,
                index.expand(-1, 1, cfg.hidden_dim),
                torch.where(active[:, None], updated, own)[:, None],
            )
            predicted_probs = prediction.emotion_log_probs.exp().to(x.dtype)
            previous_probs = previous_probs.scatter(
                1,
                index.expand(-1, 1, cfg.num_emotions),
                torch.where(active[:, None], predicted_probs, previous)[:, None],
            )
            last_turns = last_turns.scatter(
                1, speaker[:, None], torch.where(active, t, last)[:, None]
            )

        return ShiftAwareOutput(
            **{name: torch.stack(values, 1) for name, values in collected.items()},
            turn_mask=turn_mask,
            attention_weights=torch.stack(attention_rows, 1) if return_attention else None,
            forecast_attention_weights=torch.stack(forecast_rows, 1) if return_attention else None,
            attention_null_weights=torch.stack(null_rows, 1) if return_attention else None,
        )

    def _forward_transformer(
        self, embeddings, speaker_ids, turn_mask, turn_gaps, return_attention
    ) -> ShiftAwareOutput:
        x = self.utterance_projection(embeddings.masked_fill(~turn_mask[..., None], 0))
        ids = speaker_ids.masked_fill(~turn_mask, 0).long()
        out = self.transformer(x, ids, turn_mask, turn_gaps)
        rows = turn_mask[..., None]
        attention = forecast_attention = null = None
        if return_attention:
            turns = turn_mask.shape[1]
            earlier = torch.ones(turns, turns, device=x.device, dtype=torch.bool).tril(-1)
            weights = out["attention_weights"]
            if weights is None:  # utterance-only: all weight stays on the current turn
                weights = torch.eye(turns, device=x.device).expand(len(x), -1, -1)
            # History weights exclude the current turn; its own weight is reported as null.
            attention = weights.masked_fill(~earlier, 0) * rows
            null = weights.diagonal(dim1=1, dim2=2) * turn_mask
            forecast_attention = out["forecast_attention_weights"][..., :turns] * rows
        return ShiftAwareOutput(
            emotion_log_probs=out["emotion_log_probs"],
            # The history-only forecast plays the prior's role; it is mixed in
            # only with fusion="shift_gate".
            prior_logits=out["forecast_emotion_logits"],
            evidence_logits=out["emotion_logits"],
            shift_logits=out["shift_logits"],
            forecast_emotion_logits=out["forecast_emotion_logits"],
            forecast_shift_logits=out["forecast_shift_logits"],
            mixing_weights=out["mixing_weights"],
            turn_mask=turn_mask,
            shift_mask=out["shift_mask"],
            attention_weights=attention,
            forecast_attention_weights=forecast_attention,
            attention_null_weights=null,
        )

    def forecast_next(
        self,
        history_embeddings: torch.Tensor,
        history_speaker_ids: torch.Tensor,
        next_speaker_ids: torch.Tensor,
        turn_mask: torch.Tensor | None = None,
        *,
        history_turn_gaps: torch.Tensor | None = None,
        next_turn_gaps: torch.Tensor | None = None,
    ) -> ShiftForecast:
        """Forecast without a current utterance, including from empty history.

        This convenience method recomputes the prefix. It is not a persistent
        streaming cache. A placeholder is appended internally, but only its
        history-only forecast is returned; its evidence prediction is discarded.
        Supply both gap tensors when source turn IDs are nonconsecutive, matching
        the gaps passed to forward during training.
        """
        if history_embeddings.ndim != 3 or history_embeddings.shape[-1] != self.config.input_dim:
            raise ValueError("history_embeddings must be [B,T,input_dim]")
        batch, turns = history_embeddings.shape[:2]
        if history_speaker_ids.shape != (batch, turns) or next_speaker_ids.shape != (batch,):
            raise ValueError("History speakers must be [B,T] and next speakers [B]")
        if turn_mask is None:
            turn_mask = history_speaker_ids >= 0
        if turn_mask.shape != (batch, turns) or turn_mask.dtype != torch.bool:
            raise ValueError("turn_mask must be boolean [B,T]")
        if history_speaker_ids.dtype not in (
            torch.int32,
            torch.int64,
        ) or next_speaker_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Speaker IDs must be integers")
        if any(
            value.device != history_embeddings.device
            for value in (history_speaker_ids, next_speaker_ids, turn_mask)
        ):
            raise ValueError("Inputs must be on the same device")
        if (next_speaker_ids < 0).any() or (turn_mask[:, 1:] & ~turn_mask[:, :-1]).any():
            raise ValueError("Next speakers must be nonnegative and history right padded")
        if (history_turn_gaps is None) != (next_turn_gaps is None):
            raise ValueError("Supply both history_turn_gaps and next_turn_gaps, or neither")
        if (
            history_turn_gaps is not None
            and next_turn_gaps is not None
            and (
                history_turn_gaps.shape != (batch, turns)
                or next_turn_gaps.shape != (batch,)
                or history_turn_gaps.dtype not in (torch.int32, torch.int64)
                or next_turn_gaps.dtype not in (torch.int32, torch.int64)
                or history_turn_gaps.device != history_embeddings.device
                or next_turn_gaps.device != history_embeddings.device
                or (history_turn_gaps[turn_mask] < 0).any()
                or (next_turn_gaps < 0).any()
            )
        ):
            raise ValueError("Turn gaps must be nonnegative integer history [B,T] and next [B]")
        lengths = turn_mask.sum(1)
        padded_x = nn.functional.pad(history_embeddings, (0, 0, 0, 1))
        # A shorter row's next position may contain arbitrary old padding.
        query_positions = torch.arange(turns + 1, device=history_embeddings.device)[None, :]
        padded_x = padded_x.masked_fill((query_positions == lengths[:, None])[..., None], 0)
        padded_ids = nn.functional.pad(history_speaker_ids, (0, 1), value=-1).long()
        padded_ids = padded_ids.scatter(1, lengths[:, None], next_speaker_ids.long()[:, None])
        padded_mask = nn.functional.pad(turn_mask, (0, 1), value=False)
        padded_mask = padded_mask.scatter(1, lengths[:, None], True)
        padded_gaps = None
        if history_turn_gaps is not None and next_turn_gaps is not None:
            padded_gaps = nn.functional.pad(history_turn_gaps, (0, 1))
            padded_gaps = padded_gaps.scatter(1, lengths[:, None], next_turn_gaps[:, None])
        output = self(padded_x, padded_ids, padded_mask, padded_gaps)
        rows = torch.arange(batch, device=history_embeddings.device)
        return ShiftForecast(
            output.forecast_emotion_probs[rows, lengths],
            output.forecast_shift_probs[rows, lengths],
            output.shift_mask[rows, lengths],
        )

    def save(self, path):
        """Save architecture and weights, without a tokenizer or text encoder."""
        torch.save(
            {
                "config": {f.name: getattr(self.config, f.name) for f in fields(self.config)},
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, *, map_location: str | torch.device = "cpu") -> "ShiftAware":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(ShiftAwareConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location).eval()
