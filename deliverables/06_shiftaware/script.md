# ShiftAware — narration script

Workstream 6 · Wang Zhoufu · Three narrated slides, followed by a silent reference slide.
Target: about 90 seconds at a relaxed 140 words per minute, including short pauses.
Read only the quoted paragraphs. Directions and source notes are not narration.

## Slide 1 — ShiftAware balances history with current evidence (~30 s)

> Hi, I'm Wang Zhoufu. My part is ShiftAware. The idea is simple: can remembering each speaker help us recognise changes in emotion? We use RoBERTa to read each sentence, alongside memories of the speaker and the conversation. We then balance what the speaker's history suggests with what they're saying now. The model makes its prediction before updating its memory, using only the conversation so far.

Point to the two left boxes, then the gate, then the prediction.
"Shift" always means a change since the same speaker's previous turn, not the previous utterance.

## Slide 2 — A learned gate mixes two emotion distributions (~25 s)

> Here's how that balance works in Python. We turn the history and current evidence into two sets of emotion probabilities. A learned gate, called beta, mixes them together. It also receives a prediction of whether the speaker's emotion has changed. We calculate this in log space to keep it stable. The memory stores our own predictions, rather than the correct labels.

The slide shows actual source: `emotionshift/models/shiftaware/heads.py`, lines 85–86 and 94–97.
The product-of-experts alternative (`mixing="product"`) is omitted; it is not the mixture shown here.

## Slide 3 — Higher macro-F1 did not yield better shift recognition (~35 s)

> These results are averaged over five runs. On MELD, ShiftAware gives our highest macro F1, at 46.3. But class weighting also differs, so we can't credit that gain to the shift head alone. On turns where emotion changes, it scores slightly below context RoBERTa, with no clear improvement. Scores also fall on stable turns, and on DailyDialog's shift turns. So we didn't meet our target. Detecting that an emotion changed doesn't necessarily help us identify the new emotion.

Point to the macro-F1 bars, then the callout box from top to bottom. Keep the slide up through the last sentence.
Show the references slide briefly afterwards, without narration.

Read “RoBERTa” as “roh-BER-tuh,” “beta” as “bay-tuh,” and “F1” as “F one.”
Say “46.3” as “forty-six point three.” Pause briefly after the opening question and before “So we didn't meet our target.”

## Sources and interpretation

- Main scores: `reports/summary/e1_main.csv`, MELD `test`, mean over five seeds.
  ShiftAware seeds are 0, 1, 2, 3, 42; the baselines use 0–4.
- "Best macro F1": 46.33 is the highest MELD macro-F1 in `e1_main.csv` (next: B4+ at 44.11).
- Selected variants (dev only): MELD `shiftaware_cw-sqrt`, DailyDialog `shiftaware`; `reports/shiftaware/selection.md`.
- H3 paired comparison, `reports/summary/paired.csv`, `a=b4`, `b=shiftaware`, 95% dialogue bootstrap:
  - MELD shift turns, weighted F1: −0.3585 pp, CI [−2.4442, +1.7150], p=0.7411.
  - MELD stable turns, weighted F1: −2.6793 pp, CI [−4.6156, −0.5944], p=0.0080.
  - DailyDialog shift turns, micro F1 without neutral (`test_dedup`): −4.7219 pp, CI [−6.6988, −2.8094], p=0.0002.
- Shift-head AUROC, MELD test: 74.75 ± 0.54 (5 seeds); `reports/shiftaware/results.md`.
- Five-seed standard deviations and bootstrap confidence intervals describe different uncertainty; do not interchange them.
- B3 and B4 use unweighted loss. ShiftAware's MELD variant uses square-root inverse class weights,
  so macro-F1 improvement alone cannot establish a benefit from the shift head.
- The shift head sees the current utterance: call this detection, not history-only forecasting.
- Exploratory transformer variants and EmoWOZ runs (≤3 seeds) are omitted to fit the time budget.

## Recording hand-in

Use your own voice. Show slides full screen at 1920×1080, 16:9. Export one MP4 as
`deliverables/06_shiftaware/06_Fpengz.mp4` (or one audio file per slide, `06_Fpengz_slide1.m4a` …),
then submit through a pull request. Check the duration from the actual recording; the timings above are estimates.
