"""Optional Hugging Face encoder adapter; imports never download weights.

The cached-embedding model only needs torch. Install the training extra
to construct a pretrained encoder. An injected backbone also supports offline
experiments, and must return an object with last_hidden_state [N,L,D].
"""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from importlib import import_module

import numpy as np
import torch
from torch import nn

from emotionshift.data.labels import EMOTIONS
from emotionshift.data.torch_data import Tokenizer

from .model import ShiftAware, ShiftAwareOutput


class TransformerUtteranceEncoder(nn.Module):
    """Encode real turns independently, skipping padded turns entirely.

    Frozen backbones remain in eval mode even when the dialogue model trains.
    Chunking bounds encoder activation memory; it does not truncate dialogues.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        *,
        frozen: bool = True,
        pooling: str = "cls",
        chunk_size: int = 64,
    ):
        super().__init__()
        if embedding_dim <= 0 or chunk_size <= 0:
            raise ValueError("embedding_dim and chunk_size must be positive")
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.frozen = frozen
        self.pooling = pooling
        self.chunk_size = chunk_size
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        name: str = "FacebookAI/roberta-base",
        *,
        frozen: bool = True,
        pooling: str = "cls",
        chunk_size: int = 64,
        local_files_only: bool = False,
    ) -> "TransformerUtteranceEncoder":
        """Explicitly load weights; only this factory may access the model hub."""
        try:
            auto_model = import_module("transformers").AutoModel
        except ImportError as exc:
            raise ImportError("Install the encoder with: uv sync --extra training") from exc
        backbone = auto_model.from_pretrained(name, local_files_only=local_files_only)
        return cls(
            backbone,
            backbone.config.hidden_size,
            frozen=frozen,
            pooling=pooling,
            chunk_size=chunk_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, tokens: Mapping[str, torch.Tensor], turn_mask: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise ValueError("tokens must contain input_ids and attention_mask")
        shape = tokens["input_ids"].shape
        if len(shape) != 3 or turn_mask.shape != shape[:2] or turn_mask.dtype != torch.bool:
            raise ValueError("tokens must be [B,T,L] and turn_mask boolean [B,T]")
        if any(t.shape != shape or t.device != turn_mask.device for t in tokens.values()):
            raise ValueError("Token tensors must share shape and device with the mask")
        if not turn_mask.any():
            raise ValueError("At least one real turn is required")
        selected = {key: value[turn_mask] for key, value in tokens.items()}
        if (selected["attention_mask"].sum(-1) == 0).any():
            raise ValueError("Each real turn must have at least one unmasked token")
        chunks = []
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            for start in range(0, len(selected["input_ids"]), self.chunk_size):
                inputs = {
                    key: value[start : start + self.chunk_size] for key, value in selected.items()
                }
                hidden = self.backbone(**inputs).last_hidden_state
                if hidden.shape != (*inputs["input_ids"].shape, self.embedding_dim):
                    raise ValueError("Backbone last_hidden_state has an unexpected shape")
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = inputs["attention_mask"].to(hidden.dtype)[..., None]
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                chunks.append(pooled)
        encoded = torch.cat(chunks)
        indices = turn_mask.flatten().nonzero().flatten()
        flat = encoded.new_zeros(shape[0] * shape[1], self.embedding_dim)
        return flat.index_copy(0, indices, encoded).reshape(*shape[:2], self.embedding_dim)


class TextShiftAware(nn.Module):
    """Tokenized-dialogue adapter over the exact same cached-embedding model."""

    def __init__(self, encoder: TransformerUtteranceEncoder, dynamics: ShiftAware):
        super().__init__()
        if encoder.embedding_dim != dynamics.config.input_dim:
            raise ValueError("Encoder embedding_dim must match model input_dim")
        self.encoder = encoder
        self.dynamics = dynamics

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        tokens: Mapping[str, torch.Tensor],
        speaker_ids: torch.Tensor,
        turn_mask: torch.Tensor | None = None,
        turn_gaps: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> ShiftAwareOutput:
        if turn_mask is None:
            turn_mask = speaker_ids >= 0
        embeddings = self.encoder(tokens, turn_mask)
        return self.dynamics(
            embeddings, speaker_ids, turn_mask, turn_gaps, return_attention=return_attention
        )


class ShiftAwarePredictor:
    """Demo adapter. Each call is one fresh dialogue, returning shared label names.

    Supply speakers for actual conversations. Without them, all strings belong
    to one speaker; we never guess alternating speakers or parse names from text.
    Use this only with trained encoder and dynamics weights.
    """

    def __init__(self, model: TextShiftAware, tokenizer: Tokenizer, *, max_length: int = 128):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict_proba(
        self, utterances: Sequence[str], *, speakers: Sequence[str] | None = None
    ) -> np.ndarray:
        if any(not isinstance(text, str) for text in utterances):
            raise ValueError("utterances must contain strings")
        if speakers is None:
            speakers = ["speaker"] * len(utterances)
        if len(speakers) != len(utterances) or any(not isinstance(s, str) for s in speakers):
            raise ValueError("speakers must contain one string per utterance")
        if not utterances:
            return np.empty((0, len(EMOTIONS)), dtype=np.float32)
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            list(utterances),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device)[None] for key, value in tokens.items()}
        mapping = {}
        ids = [mapping.setdefault(speaker, len(mapping)) for speaker in speakers]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                output = self.model(tokens, torch.tensor([ids], device=device))
                return output.emotion_probs[0].float().cpu().numpy()
        finally:
            self.model.train(was_training)

    def predict(
        self, utterances: Sequence[str], *, speakers: Sequence[str] | None = None
    ) -> list[str]:
        probabilities = self.predict_proba(utterances, speakers=speakers)
        return [EMOTIONS[int(index)] for index in probabilities.argmax(-1)]
