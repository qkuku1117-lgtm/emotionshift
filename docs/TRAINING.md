# PyTorch training batches

Install the optional training dependencies and run the training checks with:

```bash
uv sync --extra training
uv run --extra training pytest
```

`emotionshift.data.torch_data` provides `UtteranceDataset`, `DialogueDataset`,
`UtteranceCollator`, and `DialogueCollator`. Import this module explicitly;
the basic data loader and sklearn baseline do not require PyTorch.
Tokenization stays model-specific: supply a tokenizer with the Hugging Face
batch-call interface, returning CPU `input_ids` and `attention_mask` tensors.
The training extra installs PyTorch; install your chosen tokenizer library
alongside your model's dependencies.

```python
from torch.utils.data import DataLoader
from emotionshift.data import load
from emotionshift.data.torch_data import UtteranceDataset, UtteranceCollator


def make_train_loader(tokenizer):
    tokenizer.truncation_side = "left"  # retain the target and newest context
    train = load("meld", splits="train")
    samples = UtteranceDataset(train, context_turns=3)
    return DataLoader(
        samples,
        batch_size=16,
        shuffle=True,
        collate_fn=UtteranceCollator(tokenizer, max_length=256),
    )
```

Use `context_turns=0` for utterance-only training, a positive number for a
window of previous turns, or `None` for all prior turns. Context samples include
the target and use `Speaker: text` lines. Set `task="forecast"` with a positive
window or `None` to exclude the current utterance; the next speaker ID is still
available. Context and forecast collators require left truncation, which keeps
the target or latest history when the token budget is exceeded. Plain
utterance-only tokenization can use either truncation side.

Both datasets require a nonempty table containing exactly one split. They
preserve source keys, order turns, derive targets from the shared label space,
and assign speaker IDs by first occurrence within each dialogue.

| Batch field | Utterance batch | Dialogue batch |
|---|---|---|
| `tokens` | Tokenizer tensors `[B, L]` | Tokenizer tensors `[B, T, L]` |
| `labels` | Emotion IDs `[B]` | Emotion IDs `[B, T]`; padding is `-100` |
| `shift_labels`, `shift_mask` | Float targets and boolean mask `[B]` | Targets and mask `[B, T]` |
| `speaker_ids`, `turn_gaps` | Integer features `[B]` | Integer features `[B, T]` |
| `lengths`, `turn_mask` | — | Real turn counts `[B]` and mask `[B, T]` |
| `metadata` | Source keys per target | Source keys per real turn |

For dialogue models, pair `DialogueDataset(train)` with
`DialogueCollator(tokenizer)`. Shuffle dialogue samples, retaining their internal
turn order. Token padding uses the tokenizer's pad ID; padded speakers use `-1`
(replace with a valid index before a speaker embedding and mask padded turns).
Emotion cross-entropy can ignore `-100`; multiply unreduced shift BCE by
`shift_mask` and divide by `shift_mask.sum().clamp_min(1)`. The shift mask excludes
padding and each speaker's first turn. First-turn gaps are zero, with eligibility
recorded in the shift mask.

Only `batch["tokens"]` contains tokenizer inputs. Gold history and evaluation
strata are never model inputs; labels and shift labels are separate supervision.
Move tensor fields to your device in the training loop. Whole-dialogue models
must still enforce causal recurrence or attention; forecast heads must read
history state before consuming the current utterance. The collator's padding
mask alone does not enforce causality. All collator tensors are created on CPU
so DataLoader workers and optional pinned memory remain usable.
