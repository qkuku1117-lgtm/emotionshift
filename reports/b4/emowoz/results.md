# Native EmoWOZ RoBERTa: workstream 5

Five seeds, official splits, native labels, past user/system context; labelled user targets only.

| Model | Non-neutral weighted-F1 | Population SD | Macro-F1 excl. neutral |
|---|---:|---:|---:|
| b1 | 76.50 | 0.00 | 47.10 |
| b2_bigru | 77.05 | 0.45 | 48.24 |
| b2_bilstm | 77.01 | 0.19 | 48.73 |
| b3_roberta | 78.09 | 0.29 | 50.00 |
| b4_context | 84.62 | 0.25 | 55.72 |

Demo seed selected on dev: 0. Test is not used for selection.

The comparison reuses the audited SVM/BiGRU/BiLSTM and single-utterance RoBERTa results without overwriting them. All models use the same native EmoWOZ test targets and headline metric. Single-model bootstrap intervals do not establish pairwise significance. Native scores are not directly interchangeable with the MELD/DailyDialog metrics.

## Context comparisons

See paired_comparisons.csv for B4 minus B3 on all/first/stable/shift targets and real-history minus each E4 control. Intervals resample shared whole dialogues and average differences across the fixed five seed pairs; they do not resample training seeds.
B4 versus B3 also changes speaker tags and token budget. E4 controls perturb inputs of the same context-trained model and are not retrained baselines. No context-gain conclusion is assumed in advance.


| Comparison | Group | n | Mean delta (points) | 95% paired dialogue CI (points) |
|---|---|---:|---:|---|
| b4_minus_b3 | all | 8634 | +6.53 | [+5.17, +8.16] |
| b4_minus_b3 | first | 1100 | +3.73 | [-0.09, +7.95] |
| b4_minus_b3 | stable | 4884 | +11.33 | [+8.29, +14.59] |
| b4_minus_b3 | shift | 2650 | +4.44 | [+3.17, +5.65] |
| real_minus_none | all | 8634 | +21.82 | [+18.83, +24.84] |
| real_minus_shuffled | all | 8634 | +14.72 | [+11.77, +17.87] |
| real_minus_speakers | all | 8634 | -0.04 | [-0.12, +0.06] |

Intervals are exploratory and unadjusted for multiple comparisons, conditional on the five trained seed pairs.

## Interpretation

RoBERTa reaches 84.62% non-neutral weighted-F1, while non-neutral macro-F1 is 55.72%. The gap indicates uneven class performance; per-class results should accompany the headline.

See docs/EMOWOZ_B4.md for model settings, pinned pretraining revision, attribution, reproduction and checkpoint sharing. Large weights remain outside Git. Slides, own-voice narration and personal declarations remain separate deliverables.
