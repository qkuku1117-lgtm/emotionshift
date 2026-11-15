# B4 · RoBERTa with context (workstream #5)

Status: **done on MELD, DailyDialog and EmoWOZ.** Pipeline, runs and analysis in this version: HU ANQI.
The protocol (window sweep on MELD dev, E4 input controls, B4+ for R3) and the first implementation on MELD and DailyDialog are
by QIN LIXIAN (PRs #10, #12); the first EmoWOZ extension is by HE RUIZHI (PR #19). Their runs used k = all; this version
re-runs the whole workstream in one pipeline, adds a second window (k = 10) on every dataset and a window sweep on EmoWOZ, and
reproduces their k = all numbers (see "Agreement with the earlier runs").

Results: [`reports/b4/results.md`](../reports/b4/results.md) · notebook: [`notebooks/b4_context.ipynb`](../notebooks/b4_context.ipynb)

## What it is

The same RoBERTa-base as B3; only the input changes. Each utterance is read together with the previous k turns of its dialogue and
their speaker names, in the team's shared format (`Speaker: text` lines, oldest first, target last, past only; docs/TRAINING.md).
k = 0 is the B3 input. Recipe as B3: lr 1e-5, batch 32, warmup 0.1, weight decay 0.01, fp16, plain cross-entropy, best epoch on
dev; 4 epochs on MELD and EmoWOZ, 3 on DailyDialog; max 256 tokens, cut from the left so the target and the newest history stay.
EmoWOZ uses its own seven labels; system turns are history only, never targets.

- Code: `emotionshift/data/context.py` (input builder and E4 control inputs), `emotionshift/models/roberta_ctx.py`,
  `emotionshift/eval/controls.py`, `scripts/train_b4.py`, `scripts/analyse_b4_ctx.py`, `scripts/controls_b4.py`,
  `scripts/merge_b4_all.py`, `scripts/finish_b4.py`; config `configs/b4_roberta_ctx.yaml`.
- Tests: `tests/test_context.py`, `test_roberta_ctx.py`, `test_train_b4.py`, `test_controls.py`, `test_merge_b4_all.py`.

## Protocol

1. **Window sweep (E3).** k in {1, 3, 5, 10, all} on MELD and {1, 3, 5, 10} on EmoWOZ, 3 seeds each, scored on dev only.
   DailyDialog reuses MELD's window.
2. **Final runs.** 5 seeds (0 to 4) at the chosen window, test scored once. Tag `b4`.
3. **k = all.** The same 5 seeds at k = all on every dataset. Tag `b4all`.
4. **E4 controls.** Each trained seed is scored again with no history, a shuffled history (the same number of turns from another
   dialogue) and shuffled speaker names.
5. **B4+.** Class-weighted loss on MELD at k = 10 (`b4plus`) and k = all (`b4allplus`), for R3.

## Results (test, 5 seeds, mean ± SD, %)

| Dataset (headline metric) | B3, no context | B4, k = 10 | B4, k = all | k = all − k = 10 [95% CI] |
|---|---|---|---|---|
| MELD (weighted-F1) | 62.85 | **63.34 ± 0.41** | 63.30 ± 0.42 | −0.04 [−1.33, +0.58] |
| DailyDialog, test_dedup (micro-F1 excl. neutral) | 55.34 | **58.71 ± 0.46** | 59.62 ± 0.82 | +0.92 [−0.42, +1.91] |
| EmoWOZ (weighted-F1 excl. neutral) | 78.09 | **84.38 ± 0.16** | 84.31 ± 0.22 | −0.07 [−0.44, +0.57] |

**The window does not matter once it is large enough:** k = 10 and k = all are within noise on all three datasets (every interval
includes 0; paired dialogue bootstrap on seed 0).

| Target | Result | Verdict |
|---|---|---|
| R2: context RoBERTa ≥ 62 weighted-F1 on MELD | 63.34 | met |
| R3: B4+ − B3, MELD macro-F1, CI excludes 0 | +3.21 [+1.17, +5.17] at k = 10 (43.77 ± 0.76 vs 40.56); +3.03 at k = all | met |
| H1: context gain concentrates on stable turns | see below | supported on all three datasets |
| H2: shuffled context recovers less than half of the gain | see below | supported; narrowly on EmoWOZ |

The R3 and H1 intervals are the team's paired dialogue bootstrap over the five seed pairs (`scripts/summarize_all.py`).

## Where the gain comes from (E2 / H1): B4 (k = 10) − B3

| Dataset | All | Stable turns | Shift turns |
|---|---|---|---|
| MELD | +0.49 [−0.39, +1.38] | **+3.12** [+1.67, +4.68] | −1.05 [−2.52, +0.47] |
| DailyDialog | +3.37 [+1.59, +5.25] | **+7.38** [+3.88, +10.73] | +0.73 [−1.25, +2.65] |
| EmoWOZ | +6.30 [+4.88, +7.91] | **+11.15** [+7.79, +14.65] | +4.33 [+3.16, +5.67] |

Context helps most where the emotion does not change. On MELD and DailyDialog the shift turns gain nothing measurable; on
EmoWOZ they gain, but less than half as much as the stable turns. On MELD the overall gain over B3 is not significant at k = 10.

## Window sweep (E3)

Dev: MELD 59.80 (k=1), 59.58 (3), 59.82 (5), **60.23 (10)**, 60.17 (all): a spread of 0.65 points with seed SDs of 0.4 to 0.7, so
the choice is weak. EmoWOZ 81.88 (1), 83.01 (3), 83.5 (5), 83.5 (10), against 78.98 with no context: the gain grows up to k = 5
and is flat after that. On test, by stratum (`reports/b4/<dataset>/window_sweep_test.csv`, figure
`reports/b4/figures/window_sweep_test.png`): more context raises the stable turns (MELD 69.9 → 73.5, EmoWOZ 70.2 → 80.0) and
leaves MELD's shift turns flat (55.3 → 54.6).

## Input controls (E4 / H2), 5 seeds

| Dataset, window | Real history | No history | Shuffled history | Shuffled speakers |
|---|---|---|---|---|
| MELD, k = 10 | 63.34 ± 0.45 | 62.11 ± 0.32 | 61.38 ± 0.52 | 63.25 ± 0.44 |
| DailyDialog, k = 10 (see caveat 1) | 59.47 ± 0.70 | 51.86 ± 0.67 | 50.06 ± 0.79 | 59.42 ± 0.68 |
| EmoWOZ, k = all | 84.31 ± 0.25 | 60.44 ± 1.08 | 70.44 ± 0.28 | 84.36 ± 0.28 |

On MELD and DailyDialog a shuffled history is worse than none, so the model uses the right dialogue, not just "some context".
On EmoWOZ a shuffled history recovers 42% of the gap between no history and the real one, still under half; taking a model
trained without context as the reference (B3, 78.09), the shuffled input is below it. Speaker names change nothing anywhere. The
k = all controls for MELD and DailyDialog are in `reports/b4all/`. These are the same trained models with a changed input, not
retrained baselines.

## Efficiency (E9)

124.7 M parameters, as B3. CPU latency per live step: median 61 to 86 ms, p95 96 to 145 ms. Training per seed on a Colab A100:
MELD 1.5 min, DailyDialog 9.1 min, EmoWOZ 9.6 min (k = all). The window sweeps ran partly on a T4, so sweep times are not comparable.

## Agreement with the earlier runs

| | Earlier run (k = all) | This pipeline (k = all) |
|---|---|---|
| MELD weighted-F1 | 63.82 ± 0.30 (Lixian) | 63.30 ± 0.42 |
| DailyDialog micro-F1 excl. neutral | 59.61 ± 0.34 (Lixian) | 59.62 ± 0.82 |
| EmoWOZ weighted-F1 excl. neutral | 84.62 ± 0.25 (Ruizhi, 3 epochs) | 84.31 ± 0.22 (4 epochs) |

Separate code, separate trainings, a different GPU. The MELD dev sweep here picks k = 10 where the earlier one picked k = all;
both sweeps have the two windows within 0.7 dev points.

## Caveats

1. **DailyDialog, k = 10: E4 and the headline are two trainings.** The E4 rows come from one 5-seed run, the headline (58.71) from
   a later 5-seed run of the same configuration; the E4 "real" row (59.47) is that table's own reference.
2. **EmoWOZ sweep: two jobs overlapped.** Two training jobs ran on EmoWOZ at the same time and each overwrote one sweep run of the
   other (k = 5 seed 1, k = 10 seed 0). From the run files as they are now, dev is 83.55 (k = 5) and 83.54 (k = 10): a tie. The
   summary records k = 10; k = 5 gives 84.16 ± 0.13 on test, k = 10 gives 84.38. Three of the five k = 10 seeds trained on a slower
   GPU, so that row's training time (21.9 min) is a mix and is not quoted above.
3. **EmoWOZ at k = 10 has no E4 table.** The per-seed controls were run at k = all for EmoWOZ.
4. Intervals marked "seed 0" resample dialogues for one seed; they need not be centred on the 5-seed mean.
5. Shift types with fewer than 30 turns are listed in the tables but not interpreted.
6. Run files under `reports/b4*/<dataset>/runs/` keep the predictions without the utterance text; the full prediction files are in `predictions/`.

## Reproduce

```bash
python scripts/train_b4.py --datasets meld dailydialog emowoz                       # sweep + final seeds (GPU)
python scripts/train_b4.py --datasets meld dailydialog emowoz --k all --tag b4all --controls
python scripts/train_b4.py --datasets meld --k 10  --class-weight balanced --tag b4plus
python scripts/train_b4.py --datasets meld --k all --class-weight balanced --tag b4allplus
python scripts/analyse_b4_ctx.py --compare b4all                                     # tables and figures (CPU, seconds)
```

Every run is resumable; `--fresh` retrains, `--controls` scores E4 on each seed as it trains, `--out` writes to another folder
(Drive on Colab). After merging, `python scripts/summarize_all.py` refreshes the group summary.
