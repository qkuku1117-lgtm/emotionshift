# EmotionShift — EE6405 group project

Recognizing, anticipating and explaining emotional turning points in conversation.

The team's working proposal is the shared doc
[EmotionShift Project Proposal](https://claude.ai/artifact/JbtymfhtEeBZyuho7A4Eka).
`docs/proposal.md` is a local snapshot of it and may lag behind the doc.

## Setup

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
bash scripts/fetch_data.sh
uv run pytest
```

`uv sync` creates `.venv/` with Python 3.12 and installs `emotionshift` in editable mode, so notebooks can `import emotionshift`. Data goes to `data/raw/`, which is not committed.

## Layout

| Path | Contents |
|---|---|
| `emotionshift/data/` | Loaders for MELD, DailyDialog and MELD-FR into one utterance schema; text cleanup; DailyDialog leakage check; shift labels |
| `emotionshift/eval/` | Metrics (weighted-F1, macro-F1, micro-F1 without neutral) per stratum and shift type; dialogue-level paired bootstrap |
| `emotionshift/models/` | Baselines, encoder, dynamics module, heads (empty) |
| `emotionshift/attribution/` | Occlusion, Integrated Gradients, trigger classifier (empty) |
| `notebooks/` | `01_data_exploration`, `02_quick_baselines` |
| `tests/` | Shift labels, text cleanup, metrics against sklearn, checks against the real data |
| `reports/` | Figures and tables written by the notebooks |
| `scripts/` | `fetch_data.sh`; later `train.py`, `evaluate.py`, `cache_embeddings.py` |
| `pilot/` | The pilot statistics quoted in the proposal, kept as run |
| `docs/` | Proposal snapshot and other write-ups |
| `configs/`, `app/` | Experiment configs and the Streamlit demo (empty) |

## Notebooks

Open with `uv run jupyter notebook` and *Run All*. They are saved with outputs, so results can be read without running them. `02_quick_baselines` takes a few minutes because of the DailyDialog fits.

## AI use

The capstone requires a declaration of how AI tools were used, and code "generated purely by AI tools" is penalized. The pilot scripts, the `emotionshift` data and eval modules, the tests and the first two notebooks were written with AI assistance; record that, and any other AI use, in the declaration, and make sure the team understands and owns this code.
