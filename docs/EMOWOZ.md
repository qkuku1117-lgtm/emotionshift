# EmoWOZ integration

Run from the repository root (Python 3.11, CPU; no new dependencies):

```powershell
uv run python scripts/fetch_emowoz.py
uv run pytest
uv run python scripts/emowoz_stats.py
```

Download time depends on the network (about 178 MB); loading and statistics take
seconds on a typical laptop. Existing MELD/DailyDialog downloads are untouched.
Use `from emotionshift.data.loaders import load; df = load("emowoz")`.
The first ten columns match the shared schema. Extra columns preserve
`emotion_original_id`, `emotion_original`, and `source` (multiwoz/dialmage).

## Data contract

Use the original release at https://zenodo.org/records/6506504, including
`emowoz-multiwoz.json`, `emowoz-dialmage.json`, and `data-split.json`.
Keep official train/dev/test partitions; deduplicate the repeated training
manifest entry by dialogue ID, preserving order. Reject overlap between
partitions. Dialogue IDs retain their original filename and turns retain their
zero-based position in the full dialogue (0, 2, 4, ...).

The release alternates user/system turns, starting with the user. Only user
turns are labelled targets. Read the final adjudicated label at
`log[turn]["emotion"][3]["emotion"]`, not individual annotator votes.
Unlabelled system turns are excluded, never assigned neutral. The original
download remains available for future context models; this target table alone
does not include system context. Do not feed it into code assuming A/B alternate.

## Explicit, lossy project mapping

| Native ID | Original name | Shared label |
|---|---|---|
| 0 | neutral | neutral |
| 1 | fearful | fear |
| 2 | dissatisfied | anger |
| 3 | apologetic | sadness |
| 4 | abusive | anger |
| 5 | excited | joy |
| 6 | satisfied | joy |

This is a project interoperability convention, **not an official EmoWOZ
equivalence**. Fearful also covers sad/disappointed reactions; dissatisfied
is not necessarily anger, and apologetic is not necessarily sadness. These
approximations lose elicitor/conduct distinctions. Excited/satisfied and
dissatisfied/abusive collapse, so a native label change may become shared
`stable`. Surprise and disgust have no mapped examples. Keep native-label
statistics alongside shared-label results, and label results as mapped EmoWOZ.
Do not claim directly comparable native seven-class benchmark performance.

The shared emotion order and `shifts.py` are unchanged. Shifts compare the
mapped label to the same user's previous turn in the dialogue, with the first
user turn marked `first`. The existing shift-type priority applies unchanged.
The shift-rate denominator excludes `first`.

## Reproducible outputs and hand-in

`scripts/emowoz_stats.py` writes `reports/emowoz/summary.json` with raw SHA-256
hashes and split counts, plus native/shared label, stratum and shift-type CSVs.
Raw data stays in ignored `data/raw/`. Tests include a tiny offline fixture and
optional full-release checks; the existing MELD/DailyDialog regression tests
remain the acceptance criteria for their unchanged behavior.

For a video code snippet, show the `load_emowoz` label-extraction and row
construction block (at most 15 lines); demonstrate `load("emowoz")` and the
summary tables. This change supplies dataset integration and source tables,
not trained-model scores. Slides, recorded narration, signed forms and the
executed presentation notebook remain separate team hand-in items described
in `docs/TEAM_GUIDE.md`.

## VS Code

Open the repository folder itself. Select `.venv/Scripts/python.exe` with
Python: Select Interpreter if VS Code has remembered another interpreter.
The workspace default points to this environment; run commands from the root.
Notebook kernels must also use this environment (install ipykernel if needed).

## References

Feng, S., Lubis, N., Geishauser, C., Lin, H.-C., Heck, M., van Niekerk, C., &
Gašić, M. (2022). EmoWOZ: A large-scale corpus and labelling scheme for emotion
recognition in task-oriented dialogue systems. Proceedings of LREC, 4096–4113.
https://aclanthology.org/2022.lrec-1.436/

Author-maintained format reference:
https://huggingface.co/datasets/hhu-dsml/emowoz/blob/main/emowoz.py

Dataset licence: CC BY-NC 4.0; retain attribution and use for non-commercial research.
