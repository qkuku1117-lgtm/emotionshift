# EmotionShift — EE6405 group project

Recognizing, anticipating and explaining emotional turning points in conversation.

The team's working proposal is the shared doc
[EmotionShift Project Proposal](https://claude.ai/artifact/JbtymfhtEeBZyuho7A4Eka).
`docs/proposal.md` is a local snapshot of it and may lag behind the doc.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Proposal snapshot and other write-ups |
| `pilot/` | Pilot statistics quoted in the proposal (data fetch script, analysis, saved output) |
| `emotionshift/data/` | Loaders to one Dialogue/Turn schema, label maps, de-duplication, shift labels |
| `emotionshift/models/` | Baselines, encoder, dynamics module, heads |
| `emotionshift/attribution/` | Occlusion, Integrated Gradients, trigger classifier |
| `emotionshift/eval/` | Metrics, stratified reports, bootstrap, transition-matrix fidelity |
| `configs/` | One YAML file per experiment |
| `scripts/` | `train.py`, `evaluate.py`, `cache_embeddings.py`, `sweep.sh` |
| `app/` | Streamlit demo |
| `tests/` | Shift labels, leakage checks, metric checks |
| `reports/` | Generated tables and figures |

The folders under `emotionshift/`, `configs/`, `scripts/`, `app/`, `tests/` and `reports/` are empty placeholders for the structure in the proposal.

## Pilot statistics

From this folder:

```bash
bash pilot/fetch_data.sh
uv run python pilot/pilot_stats.py
```

Downloaded data goes to `pilot/data/`, which is not committed. The last run's output is in `pilot/pilot_output.txt`.

## AI use

The capstone requires a declaration of how AI tools were used, and code "generated purely by AI tools" is penalized. The pilot scripts were written with AI assistance; record that, and any other AI use, in the declaration.
