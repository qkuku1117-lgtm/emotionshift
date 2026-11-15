# Workstream 1 — B0 baseline video materials · He Ruizhi

- [01_inertia.mp4](01_inertia.mp4): recorded B0 baseline presentation.
- [01_inertia.pptx](01_inertia.pptx): editable 16:9 PowerPoint export with six slides: cover, baseline rules and formulas, causal-history code, test results, Markov decision analysis, and references.
- [script.md](script.md): final explanatory narration, organized by slide, with delivery cues and interpretation notes. The reference slide needs no narration.
- [contribution.txt](contribution.txt): full contribution statement covering B0, EmoWOZ data integration and five-seed model extensions for the group reflection.
- [refs.txt](refs.txt): APA-style references, EmoWOZ release/format sources and implementation references for the full contribution scope.

Open the PowerPoint in slide-show mode and use `script.md` as the narration reference.

## Scope

The video presents B0 majority, predicted-history inertia and Markov transition rules. It explains why prediction must precede the history update, distinguishes gold-history oracle references from deployable results, and interprets the Markov rule's neutral-heavy predictions. Headline results cover MELD and DailyDialog; no other model workstreams are presented.

The contribution statement and references also cover He Ruizhi's EmoWOZ integration and model experiment extensions. The recording itself remains focused on B0 baselines. Original model authors retain their attribution.

## Supporting code and results

These materials already exist elsewhere in the repository:

- [Baseline protocol](../../docs/B0_BASELINES.md) and [implementation](../../emotionshift/models/b0.py).
- [EmoWOZ integration and label policy](../../docs/EMOWOZ.md), plus the [SVM/BiRNN](../../docs/EMOWOZ_B1_B2.md), [single-utterance RoBERTa](../../docs/EMOWOZ_B3.md) and [context RoBERTa](../../docs/EMOWOZ_B4.md) extension protocols.
- [Executed notebook](../../notebooks/b0_baselines.ipynb) and [results](../../reports/b0/results.md).
- [Worked inertia failures](../../reports/b0/inertia_examples/README.md), [paired dialogue-bootstrap comparisons](../../reports/b0/paired/README.md), and [training-transition interpretation](../../reports/b0/transitions/README.md).

MELD uses weighted-F1; DailyDialog uses micro-F1 excluding neutral on `test_dedup`. These headline scores use different metrics. Paired intervals are exploratory, unadjusted, fixed-checkpoint comparisons. The decision diagram illustrates one training transition row; the 94.0% neutral-prediction callout describes MELD test predictions for seed 0.
