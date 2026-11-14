# Native EmoWOZ context RoBERTa: workstream 5

Status: five seeds, official test evaluation, E4 controls and artifact audits completed. Mean non-neutral weighted-F1: 84.62% (population SD 0.25 points).

This extension reuses the original B4 implementation and credits its original
author. HE RUIZHI contributes native EmoWOZ integration, experiments and audits.
Original MELD/DailyDialog reports and experiment settings remain unchanged.

## Predeclared protocol

Use all official turns to assemble past-only context, including unlabelled
system responses. Only native-labelled user turns are targets: train 66,474,
dev 8,509, test 8,634. Labels remain neutral, fearful, dissatisfied, apologetic,
abusive, excited, satisfied. Missing system labels are not neutral labels.
Gold emotion columns are never included in the input string. Each split is
assembled independently and each dialogue is isolated; future turns are excluded.

Reuse k=all, selected on MELD dev in reports/b4/selection.json, exactly as the
original DailyDialog extension reused that choice. There is no EmoWOZ window
search and no test-driven selection. The MELD-only B4+ variant is not part of
this plain context baseline extension.

Use the same pinned RoBERTa-base revision as native B3, three epochs, learning
rate 1e-5, effective batch 32, warmup 0.1, weight decay 0.01 and plain CE.
Context inputs include speaker tags and have a 256-token limit, truncated
from the left to retain the most recent content. A 6GB GPU uses micro-batch 2,
accumulation 16, evaluation batch 4 and gradient checkpointing. See
configs/emowoz_b4.yaml. Native non-neutral weighted-F1 on dev chooses the best
epoch per seed and the demo seed. All five seeds finish before test evaluation.

Compared with B3, this follows the existing B4 protocol's changes to history,
speaker prefixes and token budget; B4 minus B3 is not a perfectly isolated
history-only causal effect. E4 controls remove history, shuffle history across
other dialogues, or shuffle speaker roles within dialogues at inference time.
They are not separately retrained models and involve input-distribution changes.

## Reproduce and resume

Use the prepared CUDA environment with the project and transformer dependencies:

```powershell
python -u -X utf8 scripts/train_emowoz_context.py
# Optional single-seed invocation:
python -u -X utf8 scripts/train_emowoz_context.py --seeds 0
```

Models are separate under models/b4_emowoz/seed<k>/; temporary epoch checkpoints
are under ignored checkpoints/b4_emowoz/. Completed seeds are skipped. Partial
runs resume only if source, configuration, raw data and runtime signatures match.
An interruption before an epoch checkpoint retrains that seed. Large model
weights stay outside Git and require separate sharing for the demo.

Reports use reports/b4/emowoz/ and predictions use
predictions/b4_emowoz_test_seed<k>.csv. The original twelve columns stay first;
source metadata and pred_none/pred_shuffled/pred_speakers follow them.
Native shared metrics, class/stratum tables, confusion matrices, derived shift
F1 and 1,000-resample whole-dialogue seed-0 intervals are reported. Seed SD and
bootstrap uncertainty have different meanings. No test score chooses a model.

English project artifacts are separate from personal Chinese notes. Disclose
AI assistance per the team guide; slides, own-voice narration and personal
forms remain separate deliverables.
