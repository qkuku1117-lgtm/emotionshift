# Extend context RoBERTa to native EmoWOZ

Reuse the original B4 implementation with its author attribution. Keep
unlabelled system turns as history and score labelled user turns only.
Five seeds achieve 84.62% mean non-neutral weighted-F1
(population SD 0.25 points).

- Fix k=all from the original MELD dev selection; no EmoWOZ test-driven tuning.
- Keep native labels, causal per-dialogue/per-split context and effective batch 32.
- Supply five shared-format test prediction CSVs, class/stratum metrics,
  three E4 inference controls and paired whole-dialogue bootstrap comparisons.
- Include reproducible English plots, source CSVs, executed notebook, manifests
  and checkpoint checksums. Large weights stay ignored for separate sharing.
- Preserve original MELD/DailyDialog configurations and shared shift definitions.

Validation: 172 passed in 31.41s. Predictions, native checkpoint round trip, E4
controls and the notebook were audited after training. The report distinguishes changes in history,
speaker prefixes and token budget; E4 controls are not separately retrained.
Paired intervals are exploratory, unadjusted and conditional on five seed pairs.
