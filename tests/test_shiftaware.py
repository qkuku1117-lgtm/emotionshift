"""Check causal information flow, speaker isolation, masking and real optimization."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Install the training extra for ShiftAware tests")

from torch import nn

from emotionshift.data.torch_data import DialogueCollator, DialogueDataset
from emotionshift.models.shiftaware import (
    ShiftAware,
    ShiftAwareConfig,
    ShiftAwareLoss,
    ShiftAwarePredictor,
    TextShiftAware,
    TransformerUtteranceEncoder,
)
from emotionshift.models.shiftaware.memory import GatedDeltaMemory


@pytest.fixture
def config():
    return ShiftAwareConfig(
        input_dim=12,
        hidden_dim=16,
        time_dim=4,
        num_heads=2,
        dropout=0,
    )


@pytest.fixture(params=["recurrent", "transformer"])
def any_config(request):
    return ShiftAwareConfig(
        input_dim=12, hidden_dim=16, time_dim=4, num_heads=2, dropout=0, dynamics=request.param
    )


DESIGNS = [("speaker", "full", "probability"), ("forecast", "local", "product")]


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    embeddings = torch.randn(2, 5, 12)
    speakers = torch.tensor([[0, 1, 2, 0, 1], [0, 0, 1, -1, -1]])
    return embeddings, speakers


def test_distributions_shift_eligibility_and_retrieval(config, inputs):
    embeddings, speakers = inputs
    output = ShiftAware(config).eval()(embeddings, speakers, return_attention=True)
    mask = speakers >= 0
    assert output.emotion_probs.shape == (2, 5, 7)
    torch.testing.assert_close(output.emotion_probs.sum(-1), mask.float())
    torch.testing.assert_close(output.forecast_emotion_probs.sum(-1), mask.float())
    expected = (1 - output.mixing_weights[..., None]) * output.prior_logits.softmax(
        -1
    ) + output.mixing_weights[..., None] * output.evidence_logits.softmax(-1)
    torch.testing.assert_close(output.emotion_probs[mask], expected[mask])
    assert output.shift_mask.tolist() == [
        [False, False, False, True, True],
        [False, True, False, False, False],
    ]
    assert not output.emotion_probs[~mask].any()
    assert not output.shift_probs[~output.shift_mask].any()
    assert output.attention_weights is not None
    assert output.forecast_attention_weights is not None
    assert output.attention_null_weights is not None
    allowed = (
        (torch.arange(5)[None, :, None] > torch.arange(5)[None, None, :])
        & (speakers[:, :, None] != speakers[:, None, :])
        & mask[:, :, None]
        & mask[:, None, :]
    )
    for weights in (output.attention_weights, output.forecast_attention_weights):
        assert not weights[~allowed].any()
    torch.testing.assert_close(
        output.attention_weights.sum(-1) + output.attention_null_weights, mask.float()
    )
    assert output.attention_null_weights[0, 0] == 1
    # This speaker's history contains only themself: retrieval must choose null.
    assert output.attention_null_weights[1, 1] == 1


@pytest.mark.parametrize("backend", ["gated_delta", "gru"])
@pytest.mark.parametrize(("prior", "evidence", "mixing"), DESIGNS)
def test_future_and_current_text_cannot_leak_into_forecast(
    inputs, backend, prior, evidence, mixing
):
    model = ShiftAware(
        ShiftAwareConfig(
            input_dim=12,
            hidden_dim=16,
            time_dim=4,
            num_heads=2,
            dropout=0,
            global_memory=backend,
            prior=prior,
            evidence=evidence,
            mixing=mixing,
        )
    ).eval()
    embeddings, speakers = inputs
    original = model(embeddings, speakers)
    changed = embeddings.clone()
    changed[:, 3:] = 100 * torch.randn_like(changed[:, 3:])
    modified = model(changed, speakers)
    for name in ("emotion_log_probs", "shift_logits", "mixing_weights"):
        torch.testing.assert_close(getattr(original, name)[:, :3], getattr(modified, name)[:, :3])
    for name in ("forecast_emotion_logits", "forecast_shift_logits"):
        torch.testing.assert_close(getattr(original, name)[:, :4], getattr(modified, name)[:, :4])
    assert not torch.allclose(original.emotion_probs[0, 3], modified.emotion_probs[0, 3])

    differentiable = embeddings.clone().requires_grad_()
    output = model(differentiable, speakers)
    gradient = torch.autograd.grad(output.forecast_emotion_logits[0, 3, 0], differentiable)[0]
    assert not gradient[:, 3:].any()
    assert gradient[0, :3].abs().sum() > 0
    assert not gradient[1].any()


def test_batching_padding_and_speaker_renaming_do_not_change_predictions(any_config, inputs):
    model = ShiftAware(any_config).eval()
    embeddings, speakers = inputs
    output = model(embeddings, speakers)
    for row, length in ((0, 5), (1, 3)):
        separate = model(embeddings[row : row + 1, :length], speakers[row : row + 1, :length])
        for name in ("emotion_log_probs", "forecast_emotion_logits", "shift_logits"):
            torch.testing.assert_close(
                getattr(output, name)[row, :length],
                getattr(separate, name)[0],
                atol=1e-6,
                rtol=1e-5,
            )
    renamed = torch.where(speakers >= 0, (speakers + 1) % 3, speakers)
    renamed_output = model(embeddings, renamed)
    torch.testing.assert_close(output.emotion_probs, renamed_output.emotion_probs)
    # Padding must not be encoded, stored as another speaker, or influence a loss.
    dirty = embeddings.masked_fill((speakers < 0)[..., None], torch.nan)
    dirty_speakers = speakers.masked_fill(speakers < 0, 1000)
    padded = model(dirty, dirty_speakers, turn_mask=speakers >= 0)
    torch.testing.assert_close(output.emotion_probs, padded.emotion_probs)


def test_explicit_speaker_gaps_and_checkpoint_roundtrip(any_config, inputs, tmp_path):
    model = ShiftAware(any_config).eval()
    embeddings, speakers = inputs
    automatic = model(embeddings, speakers)
    gaps = torch.tensor([[0, 0, 0, 3, 3], [0, 1, 0, 0, 0]])
    explicit = model(embeddings, speakers, turn_gaps=gaps)
    torch.testing.assert_close(automatic.emotion_probs, explicit.emotion_probs)
    gaps[0, 3] = 64
    delayed = model(embeddings, speakers, turn_gaps=gaps)
    assert not torch.allclose(
        explicit.forecast_emotion_probs[0, 3], delayed.forecast_emotion_probs[0, 3]
    )
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = ShiftAware.load(checkpoint)
    torch.testing.assert_close(
        automatic.emotion_probs, restored(embeddings, speakers).emotion_probs
    )
    assert restored.config == any_config


def test_forecast_next_requires_only_history_and_next_speaker(any_config, inputs):
    embeddings, speakers = inputs
    model = ShiftAware(any_config).eval()
    full = model(embeddings, speakers)
    forecast = model.forecast_next(embeddings[:, :2], speakers[:, :2], speakers[:, 2])
    torch.testing.assert_close(forecast.emotion_probs, full.forecast_emotion_probs[:, 2])
    assert not forecast.shift_mask.any()  # Both target speakers are new here.
    known = model.forecast_next(embeddings[:1, :3], speakers[:1, :3], speakers[:1, 3])
    torch.testing.assert_close(known.shift_probs, full.forecast_shift_probs[:1, 3])
    assert known.shift_mask.all()
    empty = model.forecast_next(embeddings[:, :0], speakers[:, :0], torch.tensor([0, 1]))
    torch.testing.assert_close(empty.emotion_probs.sum(-1), torch.ones(2))
    assert not empty.shift_mask.any()
    # Histories of different lengths append their query at each true end.
    mixed = model.forecast_next(embeddings, speakers, torch.tensor([0, 1]))
    single = model.forecast_next(embeddings[1:2, :3], speakers[1:2, :3], torch.tensor([1]))
    torch.testing.assert_close(mixed.emotion_probs[1], single.emotion_probs[0])


def test_forecast_next_preserves_source_turn_gaps(any_config, inputs):
    embeddings, speakers = inputs
    model = ShiftAware(any_config).eval()
    gaps = torch.tensor([[0, 0, 0, 64, 3], [0, 3, 0, 0, 0]])
    full = model(embeddings, speakers, turn_gaps=gaps)
    forecast = model.forecast_next(
        embeddings[:, :4],
        speakers[:, :4],
        torch.tensor([1, 0]),
        history_turn_gaps=gaps[:, :4],
        next_turn_gaps=torch.tensor([3, 0]),
    )
    torch.testing.assert_close(forecast.emotion_probs[0], full.forecast_emotion_probs[0, 4])
    shorter = model.forecast_next(
        embeddings[1:2, :3],
        speakers[1:2, :3],
        torch.tensor([0]),
        history_turn_gaps=gaps[1:2, :3],
        next_turn_gaps=torch.tensor([0]),
    )
    torch.testing.assert_close(forecast.emotion_probs[1], shorter.emotion_probs[0])
    with pytest.raises(ValueError, match="both"):
        model.forecast_next(
            embeddings[:, :2], speakers[:, :2], speakers[:, 2], next_turn_gaps=gaps[:, 2]
        )


@pytest.mark.parametrize("backend", ["gated_delta", "gru"])
@pytest.mark.parametrize("gate", ["learned", "tied"])
@pytest.mark.parametrize(("prior", "evidence", "mixing"), DESIGNS)
def test_joint_training_under_bfloat16_autocast(inputs, backend, gate, prior, evidence, mixing):
    model = ShiftAware(
        ShiftAwareConfig(
            input_dim=12,
            hidden_dim=16,
            time_dim=4,
            num_heads=2,
            dropout=0,
            global_memory=backend,
            gate=gate,
            prior=prior,
            evidence=evidence,
            mixing=mixing,
        )
    )
    embeddings, speakers = inputs
    labels = torch.tensor([[0, 1, 3, 2, 4], [0, 2, 1, -100, -100]])
    targets = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan"), 1, 1],
            [float("nan"), 1, float("nan"), float("nan"), float("nan")],
        ]
    )
    criterion = ShiftAwareLoss(class_weights=torch.arange(1, 8).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    before = model.heads.evidence_emotion.weight.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(embeddings, speakers)
        losses = criterion(output, labels, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for module in (model.attention, model.global_memory, model.speaker_memory, model.heads):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    optimizer.step()
    assert not torch.equal(before, model.heads.evidence_emotion.weight)
    if prior == "forecast":
        torch.testing.assert_close(output.prior_logits, output.forecast_emotion_logits)
    if gate == "tied":
        torch.testing.assert_close(
            output.mixing_weights, output.shift_logits.sigmoid() * output.turn_mask
        )


def test_first_turn_only_dialogues_have_no_shift_loss(any_config):
    model = ShiftAware(any_config)
    output = model(torch.randn(1, 3, 12), torch.tensor([[0, 1, 2]]))
    losses = ShiftAwareLoss()(output, torch.tensor([[0, 1, 2]]), torch.full((1, 3), torch.nan))
    assert losses["shift"] == 0 and losses["forecast_shift"] == 0
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_component_loss_supervises_prior_and_evidence_directly(config):
    torch.manual_seed(0)
    model = ShiftAware(config)
    output = model(torch.randn(1, 4, 12), torch.tensor([[0, 1, 0, 1]]))
    labels, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    plain = ShiftAwareLoss()(output, labels, targets)
    losses = ShiftAwareLoss(component_weight=0.5)(output, labels, targets)
    torch.testing.assert_close(
        losses["total"], plain["total"] + 0.5 * (losses["prior"] + losses["evidence"])
    )
    torch.testing.assert_close(
        losses["prior"], nn.functional.cross_entropy(output.prior_logits[0], labels[0])
    )
    model.zero_grad()
    losses["prior"].backward()
    assert model.heads.prior[-1].weight.grad.abs().sum() > 0


@pytest.fixture
def transformer_config():
    return ShiftAwareConfig(
        input_dim=12, hidden_dim=16, time_dim=4, num_heads=2, dropout=0, dynamics="transformer"
    )


def test_transformer_distributions_causality_and_attention(transformer_config, inputs):
    embeddings, speakers = inputs
    model = ShiftAware(transformer_config).eval()
    assert not hasattr(model, "heads")  # no prior, gate or mixture
    output = model(embeddings, speakers, return_attention=True)
    mask = speakers >= 0
    torch.testing.assert_close(output.emotion_probs.sum(-1), mask.float())
    torch.testing.assert_close(output.forecast_emotion_probs.sum(-1), mask.float())
    torch.testing.assert_close(
        output.emotion_probs, output.evidence_logits.softmax(-1) * mask[..., None]
    )
    assert output.shift_mask.tolist() == [
        [False, False, False, True, True],
        [False, True, False, False, False],
    ]
    assert output.attention_weights is not None and output.attention_null_weights is not None
    assert output.forecast_attention_weights is not None
    earlier = torch.ones(5, 5, dtype=torch.bool).tril(-1)
    allowed = earlier & mask[:, :, None] & mask[:, None, :]
    for weights in (output.attention_weights, output.forecast_attention_weights):
        assert not weights[~allowed].any()
    torch.testing.assert_close(
        output.attention_weights.sum(-1) + output.attention_null_weights, mask.float()
    )

    changed = embeddings.clone()
    changed[:, 3:] = 100 * torch.randn_like(changed[:, 3:])
    modified = model(changed, speakers)
    for name in ("emotion_log_probs", "shift_logits"):
        torch.testing.assert_close(getattr(output, name)[:, :3], getattr(modified, name)[:, :3])
    for name in ("forecast_emotion_logits", "forecast_shift_logits"):
        torch.testing.assert_close(getattr(output, name)[:, :4], getattr(modified, name)[:, :4])
    differentiable = embeddings.clone().requires_grad_()
    forecast = model(differentiable, speakers).forecast_emotion_logits[0, 3, 0]
    gradient = torch.autograd.grad(forecast, differentiable)[0]
    assert not gradient[:, 3:].any() and not gradient[1].any()
    assert gradient[0, :3].abs().sum() > 0


def test_utterance_only_transformer_reads_only_the_current_turn(transformer_config, inputs):
    embeddings, speakers = inputs
    config = ShiftAwareConfig(**{**transformer_config.__dict__, "context_layers": 0})
    model = ShiftAware(config).eval()
    output = model(embeddings, speakers, return_attention=True)
    changed = embeddings.clone()
    changed[:, [0, 1, 3]] = torch.randn_like(changed[:, [0, 1, 3]])
    modified = model(changed, speakers)
    for name in ("emotion_log_probs", "shift_logits"):
        torch.testing.assert_close(getattr(output, name)[:, 2], getattr(modified, name)[:, 2])
    assert output.attention_null_weights is not None
    torch.testing.assert_close(output.attention_null_weights, (speakers >= 0).float())


@pytest.mark.parametrize("layers", [0, 2])
def test_transformer_training_under_bfloat16_autocast(inputs, layers):
    torch.manual_seed(0)
    model = ShiftAware(
        ShiftAwareConfig(
            input_dim=12,
            hidden_dim=16,
            time_dim=4,
            num_heads=2,
            dropout=0.1,
            dynamics="transformer",
            context_layers=layers,
        )
    )
    embeddings, speakers = inputs
    labels = torch.tensor([[0, 1, 3, 2, 4], [0, 2, 1, -100, -100]])
    targets = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = ShiftAwareLoss(class_weights=torch.arange(1, 8).float())(
            model(embeddings, speakers), labels, targets
        )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    for module in (
        model.utterance_projection,
        model.transformer.emotion,
        model.transformer.shift,
        model.transformer.forecast_attention,
        *model.transformer.layers,
    ):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0


def test_gated_delta_rank_one_update_matches_dense_equation():
    torch.manual_seed(3)
    memory = GatedDeltaMemory(12, 3)
    x = torch.randn(2, 12)
    state = memory.initial_state(x)
    state.matrix = torch.randn_like(state.matrix)
    normalized = memory.input_norm(x)
    _, k, v = memory.qkv(normalized).reshape(2, 3, 3, 4).unbind(1)
    k = nn.functional.normalize(k, dim=-1)
    alpha = memory.decay(normalized).sigmoid()[..., None, None]
    beta = memory.write(normalized).sigmoid()[..., None, None]
    # Independent dense matrix implementation of the paper's Eq. 8.
    transition = torch.eye(4) - beta * (k[..., :, None] @ k[..., None, :])
    expected = state.matrix @ (alpha * transition) + beta * (v[..., :, None] @ k[..., None, :])
    actual = memory(x, state)
    torch.testing.assert_close(actual.matrix, expected)


class OfflineBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):  # ty: ignore[invalid-mutable-override]
        # Mix token information as a tiny stand-in for a contextual encoder.
        hidden = self.embedding(input_ids)
        mean = (hidden * attention_mask[..., None]).sum(1, keepdim=True)
        mean = mean / attention_mask.sum(1)[:, None, None]
        return SimpleNamespace(last_hidden_state=self.dropout(hidden + mean))


class OfflineTokenizer:
    pad_token_id = 0
    truncation_side = "right"

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        sequences = [
            [1, *[2 + len(word) % 25 for word in text.split()], 2][:max_length] for text in texts
        ]
        length = max(map(len, sequences))
        ids = torch.zeros(len(texts), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence)
            mask[i, : len(sequence)] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.mark.parametrize("frozen", [True, False])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_encoder_chunking_padding_freeze_and_gradients(frozen, pooling):
    backbone = OfflineBackbone()
    encoder = TransformerUtteranceEncoder(
        backbone, 12, frozen=frozen, pooling=pooling, chunk_size=1
    )
    encoder.train()
    tokens = {
        "input_ids": torch.tensor([[[1, 2, 0], [1000, 1000, 1000]]]),
        "attention_mask": torch.tensor([[[1, 1, 0], [0, 0, 0]]]),
    }
    mask = torch.tensor([[True, False]])
    result = encoder(tokens, mask)
    assert result.shape == (1, 2, 12)
    assert not result[0, 1].any()
    if frozen:
        assert not backbone.training and not result.requires_grad
        torch.testing.assert_close(result, encoder(tokens, mask))
    else:
        assert backbone.training and result.requires_grad
        result.sum().backward()
        assert backbone.embedding.weight.grad is not None
        assert backbone.embedding.weight.grad.abs().sum() > 0


def test_shared_dialogue_batch_text_adapter_and_demo(config):
    frame = pd.DataFrame(
        {
            "dataset": ["meld"] * 5,
            "split": ["train"] * 5,
            "dialogue_id": ["a", "a", "a", "b", "b"],
            "turn": [0, 1, 2, 0, 1],
            "speaker": ["Alice", "Bob", "Alice", "Charlie", "Charlie"],
            "text": [
                "Wonderful trip",
                "It was cancelled",
                "That is upsetting",
                "Hello",
                "Great news",
            ],
            "emotion": ["joy", "neutral", "sadness", "neutral", "joy"],
        }
    )
    tokenizer = OfflineTokenizer()
    dataset = DialogueDataset(frame)
    batch = DialogueCollator(tokenizer)(list(dataset))
    backbone = OfflineBackbone()
    encoder = TransformerUtteranceEncoder(backbone, 12)
    dynamics = ShiftAware(config)
    model = TextShiftAware(encoder, dynamics).eval()
    inputs = {name: batch[name] for name in ("tokens", "speaker_ids", "turn_mask", "turn_gaps")}
    output = model(**inputs)
    cached = encoder(batch["tokens"], batch["turn_mask"])
    direct = dynamics(cached, batch["speaker_ids"], batch["turn_mask"], batch["turn_gaps"])
    torch.testing.assert_close(output.emotion_probs, direct.emotion_probs)
    assert torch.equal(output.shift_mask, batch["shift_mask"])
    losses = ShiftAwareLoss()(output, batch["labels"], batch["shift_labels"], batch["shift_mask"])
    losses["total"].backward()
    assert dynamics.heads.forecast_emotion.weight.grad is not None
    assert dynamics.heads.forecast_emotion.weight.grad.abs().sum() > 0
    assert backbone.embedding.weight.grad is None

    predictor = ShiftAwarePredictor(model, tokenizer)
    probabilities = predictor.predict_proba(
        frame["text"][:3].tolist(), speakers=frame["speaker"][:3].tolist()
    )
    np.testing.assert_allclose(probabilities, output.emotion_probs[0].detach().numpy(), atol=1e-6)
    assert len(predictor.predict(["Hello", "Goodbye"])) == 2
    assert predictor.predict([]) == []
    assert predictor.predict_proba([]).shape == (0, 7)


def test_real_roberta_adapter_without_pretrained_download(config):
    transformers = pytest.importorskip("transformers", reason="Optional text encoder dependency")
    backbone = transformers.RobertaModel(
        transformers.RobertaConfig(
            vocab_size=32,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=32,
            pad_token_id=0,
        )
    )
    encoder = TransformerUtteranceEncoder(backbone, 12, frozen=False)
    model = TextShiftAware(encoder, ShiftAware(config))
    tokens = OfflineTokenizer()(
        ["Hello", "Wonderful news"],
        padding=True,
        truncation=True,
        max_length=8,
        return_tensors="pt",
    )
    output = model({key: value[None] for key, value in tokens.items()}, torch.tensor([[0, 0]]))
    (-output.emotion_log_probs[0, 1, 1]).backward()
    assert backbone.embeddings.word_embeddings.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "settings",
    [
        {"hidden_dim": 15},
        {"num_heads": 0},
        {"dropout": 1},
        {"global_memory": "unknown"},
        {"gate": "unknown"},
        {"prior": "unknown"},
        {"evidence": "unknown"},
        {"mixing": "unknown"},
        {"dynamics": "unknown"},
        {"context_layers": -1},
        {"num_emotions": 3},
    ],
)
def test_invalid_architecture_settings(settings):
    with pytest.raises(ValueError):
        ShiftAwareConfig(**settings)


def test_invalid_dialogue_inputs(config):
    model = ShiftAware(config)
    with pytest.raises(ValueError, match="right padded"):
        model(torch.randn(1, 3, 12), torch.tensor([[0, -1, 1]]))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.randn(1, 2, 12), torch.tensor([[-1, -1]]))
    with pytest.raises(ValueError, match="input_dim"):
        model(torch.randn(1, 2, 13), torch.tensor([[0, 1]]))


def test_unlabelled_turns_add_context_but_no_emotion_loss(any_config):
    # EmoWOZ system turns carry label -100: they are encoded, but not supervised.
    torch.manual_seed(0)
    model = ShiftAware(any_config)
    output = model(torch.randn(1, 4, 12), torch.tensor([[0, 1, 0, 1]]))
    labels = torch.tensor([[0, -100, 6, -100]])
    shift_mask = torch.tensor([[False, False, True, False]])
    targets = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    weights = torch.arange(1, 8).float()
    losses = ShiftAwareLoss(class_weights=weights)(output, labels, targets, shift_mask)
    keep = labels[0] >= 0
    torch.testing.assert_close(
        losses["emotion"],
        nn.functional.nll_loss(output.emotion_log_probs[0][keep], labels[0][keep], weight=weights),
    )
    torch.testing.assert_close(
        losses["shift"],
        nn.functional.binary_cross_entropy_with_logits(
            output.shift_logits[0][2:3].float(), targets[0][2:3]
        ),
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_shift_gate_mixes_the_forecast_in_by_one_minus_shift_probability(
    transformer_config, inputs
):
    embeddings, speakers = inputs
    config = ShiftAwareConfig(**{**transformer_config.__dict__, "fusion": "shift_gate"})
    model = ShiftAware(config).eval()
    output = model(embeddings, speakers)
    mask = speakers >= 0
    # Opening turns have no history to forecast from and read the evidence alone.
    beta = output.shift_logits.sigmoid()
    beta[:, 0] = 1
    torch.testing.assert_close(output.mixing_weights[mask], beta[mask])
    torch.testing.assert_close(output.emotion_probs[:, 0], output.evidence_logits[:, 0].softmax(-1))
    assert (output.mixing_weights[:, 1:][mask[:, 1:]] < 1).all()
    expected = (
        output.evidence_logits.log_softmax(-1)
        + (1 - beta)[..., None] * output.forecast_emotion_logits.log_softmax(-1)
    ).softmax(-1)
    torch.testing.assert_close(output.emotion_probs[mask], expected[mask])
    torch.testing.assert_close(output.emotion_probs.sum(-1), mask.float())

    # Still causal: changing turns 3+ leaves the recognition of turns 0-2 alone.
    changed = embeddings.clone()
    changed[:, 3:] = 100 * torch.randn_like(changed[:, 3:])
    modified = model(changed, speakers)
    torch.testing.assert_close(output.emotion_log_probs[:, :3], modified.emotion_log_probs[:, :3])

    # The recognition loss alone trains the gate and the forecast.
    model.train().zero_grad()
    labels = torch.tensor([[0, 1, 3, 2, 4], [0, 2, 1, -100, -100]])
    targets = torch.zeros(2, 5)
    losses = ShiftAwareLoss(shift_weight=0, forecast_weight=0)(
        model(embeddings, speakers), labels, targets
    )
    losses["total"].backward()
    for module in (model.transformer.shift, model.transformer.forecast_emotion):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_fusion_needs_transformer_dynamics():
    with pytest.raises(ValueError, match="fusion"):
        ShiftAwareConfig(fusion="shift_gate")
    with pytest.raises(ValueError, match="fusion"):
        ShiftAwareConfig(dynamics="transformer", fusion="mean")
