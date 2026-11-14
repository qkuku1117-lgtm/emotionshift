# Native EmoWOZ RoBERTa: workstream 4

Five seeds, official splits, native labels, current utterance only.

| Model | Non-neutral weighted-F1 | Population SD | Macro-F1 excl. neutral |
|---|---:|---:|---:|
| b1 | 76.50 | 0.00 | 47.10 |
| b2_bigru | 77.05 | 0.45 | 48.24 |
| b2_bilstm | 77.01 | 0.19 | 48.73 |
| b3_roberta | 78.09 | 0.29 | 50.00 |

Demo seed selected on dev: 1. Test is not used for selection.

The comparison reuses the audited SVM/BiGRU/BiLSTM results without overwriting them. All models use the same native EmoWOZ test targets and headline metric. Single-model bootstrap intervals do not establish pairwise significance. Native scores are not directly interchangeable with the MELD/DailyDialog metrics.

## Interpretation

RoBERTa reaches 78.09% non-neutral weighted-F1, while non-neutral macro-F1 is 50.00%. The gap indicates uneven class performance; per-class results should accompany the headline.
First-target F1 is 57.75% for RoBERTa versus 44.83% for BiGRU; shift-subset F1 is 82.75% versus 82.83%. Thus the aggregate increase does not establish an improvement on shifted targets. These are descriptive within-subset comparisons; no paired significance test is claimed.

See docs/EMOWOZ_B3.md for model settings, pinned pretraining revision, attribution, reproduction and checkpoint sharing. Large weights remain outside Git. Slides, own-voice narration and personal declarations remain separate deliverables.
