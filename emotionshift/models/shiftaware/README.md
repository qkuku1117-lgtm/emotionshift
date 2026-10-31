# ShiftAware

Trainable, causal emotion dynamics for multi-speaker dialogues. This implements
the revised architecture discussed on 29 October 2026. There are no trained
weights or measured accuracy results yet.

## Install and check

```bash
uv sync --extra training
uv run --extra training python scripts/smoke_shiftaware.py
uv run --extra training pytest tests/test_shiftaware.py
```

The smoke command optimizes a tiny model on synthetic embeddings. It needs no
data downloads, pretrained weights or GPU and does not measure task accuracy.
Use `--device cuda` on an L40S, or `--global-memory gru` to exercise the alternative.

To train the text model on local MELD data, run:

```bash
uv sync --extra training
uv run --extra training python scripts/train_shiftaware.py \
  --dataset meld --output-dir models/shiftaware/meld_seed42
```

The runner reads `configs/shiftaware.yaml` for architecture and
`configs/shiftaware_train.yaml` for training. It loads only train and dev,
selects `best.pt` by the dataset's headline dev F1, and writes `history.json`,
`tokenizer/` and `encoder_config/` in the output directory.
These optimization settings are starting values; tune them on dev after the
first real run. The optimizer uses AdamW with separate encoder and dynamics
learning rates, no weight decay on bias/norm vectors, gradient accumulation,
gradient clipping and warmup followed by cosine decay. CUDA uses BF16 when
available, otherwise FP16 with gradient scaling. Early stopping uses dev F1.
The checkpoint stores the full text model's weights and its encoder name;
the adjacent tokenizer and encoder config make reconstruction possible without
downloading the original encoder again.

```python
from emotionshift.models.shiftaware import ShiftAwarePredictor
from emotionshift.models.shiftaware.training import load_trained_model

text_model, tokenizer = load_trained_model("models/shiftaware/meld_seed42")
predictor = ShiftAwarePredictor(text_model, tokenizer)
labels = predictor.predict(["Hello", "What happened?"], speakers=["A", "B"])
```

For RoBERTa, install `uv sync --extra training`. Model construction and
imports never download weights. Only an explicit `from_pretrained(...)` call
may download a model; use `local_files_only=True` for a local checkpoint.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Validated, serializable architecture settings |
| `encoder.py` | Optional transformer adapter, tokenized-dialogue wrapper and demo predictions |
| `temporal.py` | Log-bucketed speaker gaps and first-turn embeddings |
| `memory.py` | Gated delta global memory, GRU alternative, shared speaker GRU |
| `attention.py` | Turn attention within other speakers, then attention across speakers |
| `heads.py` | Speaker prior, current evidence, shift detector, mixing gate, forecasts |
| `model.py` | Causal read/predict/update order, padding and speaker-state management |
| `loss.py` | Joint masked losses; the only model module accepting gold targets |

Use `ShiftAware` with cached embeddings or `TextShiftAware` with tokenized
dialogues. Both produce the same `ShiftAwareOutput` and use `ShiftAwareLoss`.
The architecture defaults are in `configs/shiftaware.yaml`; construct a config
with `ShiftAwareConfig(**yaml.safe_load(path.read_text()))`.

## Architecture and causality

For each target turn, the model reads the global summary, the target speaker's
previous GRU state, their previous predicted emotion distribution, and their
turn gap. Unknown speakers start with a zero state and uniform distribution.
There are no learned character-name embeddings or fixed two-speaker limit.
GRU parameters are shared, while states are separate for each dialogue/speaker.

1. Construct a history-only query and retrieve context for forecasting. Predict
   next-emotion logits and a shift logit without reading the current embedding.
2. Construct another query using the current embedding. Retrieve relevant
   earlier turns from other speakers. First attend over each speaker's turns,
   then over the resulting speaker summaries. A learned null candidate lets
   the model retrieve no interaction. Relative turn distance supplies a learned
   per-head attention bias.
3. Predict a prior from the speaker's trajectory and current evidence from text
   plus context. Predict a shift probability `r` and a distinct learned gate
   `beta`, which also receives `r`. Mix distributions, not logits:

   `P(emotion) = (1 - beta) * P(prior) + beta * P(evidence)`.

4. Update global memory, then only the current speaker's GRU and stored predicted
   distribution. Make the utterance available for subsequent retrieval.

The prior uses the speaker gap; evidence and forecast queries use both gap and
first-turn encodings. Gaps are differences in turn positions, not elapsed
seconds. The optional `turn_gaps` input accepts the shared loader's gaps when
source turn IDs are nonconsecutive. First-turn status is always derived from
observed speaker history, never supplied as a gold shift label.

Every forward call starts fresh. Dialogues may be shuffled as samples; turns
inside each dialogue must remain ordered. No gold previous emotions, current
gold labels, shift labels or evaluation strata enter the forward interface.
In particular, do not build a cache using a bidirectional dialogue encoder or
future context: this model cannot remove leakage already inside embeddings.

## Cached-embedding training

```python
import torch
from emotionshift.models.shiftaware import (
    ShiftAware,
    ShiftAwareConfig,
    ShiftAwareLoss,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShiftAware(ShiftAwareConfig(input_dim=768)).to(device)
criterion = ShiftAwareLoss(shift_weight=1.0, forecast_weight=0.5).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# embeddings: [B,T,768], obtained independently per utterance.
# speaker_ids: [B,T], dialogue-local integers; padding -1.
# turn_mask: boolean [B,T], real turns followed by right padding.
# labels: long [B,T], shared EMOTIONS order; padding -100.
# shift_labels: float [B,T]. shift_mask excludes padding and first turns.
def train_step(embeddings, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        embeddings.to(device),
        batch["speaker_ids"].to(device),
        batch["turn_mask"].to(device),
        batch["turn_gaps"].to(device),
    )
    losses = criterion(
        output,
        batch["labels"].to(device),
        batch["shift_labels"].to(device),
        batch["shift_mask"].to(device),
    )
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}
```

The total loss is emotion NLL plus weighted shift BCE and weighted forecasting
emotion CE + shift BCE. Emotion weights are optional, in the fixed seven-class
order. Emotion and forecast-emotion losses include first turns; shift losses
exclude them. For batches with no eligible shifts, shift losses are differentiable
zero. Targets in excluded/padded rows do not contribute. All computations retain
gradients through dialogue history, including previous predicted distributions.

`emotion_log_probs` already contains the log of a normalized mixture: use NLL,
not another softmax/cross-entropy. `emotion_probs` and forecast emotion
probabilities sum to one on real turns and zero on padding. Shift probabilities
are zero where `shift_mask` is false; that sentinel does not mean a new speaker
is predicted stable. Consult the mask when evaluating or displaying shifts.

## Connect the shared text batches

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from emotionshift.data import load
from emotionshift.data.torch_data import DialogueDataset, DialogueCollator
from emotionshift.models.shiftaware import TextShiftAware, TransformerUtteranceEncoder

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
encoder = TransformerUtteranceEncoder.from_pretrained(
    "FacebookAI/roberta-base",
    frozen=True,
)
text_model = TextShiftAware(encoder, model).to(device)
loader = DataLoader(
    DialogueDataset(load("meld", splits="train")),
    batch_size=8,
    shuffle=True,
    collate_fn=DialogueCollator(tokenizer),
)
batch = next(iter(loader))
tokens = {key: value.to(device) for key, value in batch["tokens"].items()}
output = text_model(
    tokens,
    batch["speaker_ids"].to(device),
    batch["turn_mask"].to(device),
    batch["turn_gaps"].to(device),
)
```

Prefer a separately fine-tuned encoder checkpoint for actual experiments.
Frozen encoders stay in eval mode even when the dynamics model trains, making
their embeddings suitable for caching. The encoder skips padded turns and
supports chunked execution. Set `frozen=False` for end-to-end training and
create an optimizer over `text_model.parameters()` so it includes the encoder.
Chunking limits each encoder call; trainable encoder activations still remain
in the autograd graph until backward.

For the two-stage workflow, run the frozen encoder in eval mode once and save
its output with utterance/source keys and encoder checkpoint metadata. Cache
each split separately. Never reuse a cache across different encoder checkpoints.
The model interface accepts these tensors; a persistent cache-generation CLI
remains separate follow-up work. The training runner above uses text batches.

## Forecast, inspect, save and predict

```python
model.eval()
with torch.inference_mode():
    forecast = model.forecast_next(
        history_embeddings,
        history_speaker_ids,
        next_speaker_ids,
        turn_mask=history_mask,
        history_turn_gaps=history_gaps,
        next_turn_gaps=next_gap,
    )
    # forecast.emotion_probs [B,7]; shift_probs and shift_mask [B].
    output = model(embeddings, speaker_ids, return_attention=True)
    # output.attention_weights: [B,target_turn,history_turn].
    # output.forecast_attention_weights: history-only retrieval diagnostics.
model.save("checkpoints/shiftaware.pt")  # create the parent directory first
restored = ShiftAware.load("checkpoints/shiftaware.pt", map_location="cpu")
```

`forecast_next` accepts empty history and does not need the unknown utterance.
Supply both gap arguments when training or inference uses source turn IDs with
missing positions; otherwise it derives gaps from contiguous turn positions.
It recomputes the prefix; persistent streaming state is not implemented yet.
Cached-model checkpoints contain config and dynamics weights, not the text
encoder, tokenizer or optimizer. Save those separately for a complete run.

`ShiftAwarePredictor(text_model, tokenizer)` exposes `predict` and
`predict_proba` for the demo. Pass `speakers=[...]` alongside the utterances.
Without speakers, strings are treated as one person's turns; the adapter does
not guess speaker alternation. Every call starts a fresh dialogue.

## Scope and performance

- The global-memory alternatives are `gated_delta` and `gru`. The gate can be
  `learned` or `tied` (the original `beta = P(shift)` formulation).
- The gated delta implementation uses the paper's matrix update with normalized
  queries/keys and FP32 accumulation under BF16 autocast. It is a small
  utterance-level adaptation, not the full published Gated DeltaNet block.
  It omits short convolutions and fused/chunkwise CUDA kernels. See
  [the paper](https://arxiv.org/abs/2412.06464) and
  [the official implementation](https://github.com/NVlabs/GatedDeltaNet).
- The implementation batches dialogues but iterates over turns. Speaker IDs
  should be small dialogue-local integers as produced by `DialogueDataset`.
  Full-history hierarchical attention adds quadratic dialogue-length cost;
  global recurrent memory alone does not make the full model linear-time.
- Attention weights are retrieval diagnostics. They are not validated trigger
  explanations; MELD-FR attribution evaluation remains separate.
- CPU optimization, BF16 autocast, padding invariance, speaker renaming,
  causal values and gradients, checkpoint reload, and a real randomly
  initialized RoBERTa adapter are covered by tests. CUDA performance and task
  accuracy have not been measured. Record this implementation as AI-assisted.
