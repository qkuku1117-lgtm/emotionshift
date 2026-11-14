# Extend single-utterance RoBERTa to native EmoWOZ

Complete workstream 4 on official EmoWOZ splits with native labels, using the
original B3 implementation and preserving its author attribution. Five seeds
achieve 78.09% mean non-neutral weighted-F1 (population SD 0.29 points).

## Changes

- Add optional model label order with legacy checkpoint and B4 loader compatibility.
- Keep the original defaults; add micro-batch, evaluation-batch, gradient
  checkpointing and persistent epoch-resume controls for a 6GB GPU.
- Explicitly handle mean classification loss under gradient accumulation.
- Use a pinned RoBERTa-base revision, three fixed epochs and effective batch 32.
  Choose checkpoints and the demo seed on dev only, then evaluate all five seeds.
- Export five shared-format native-label prediction CSVs, six recognition metrics,
  per-class/stratum/confusion tables, derived shift F1 and dialogue-bootstrap CIs.
- Supply audited comparison figures with CSV sources and an executed notebook.
  Large weights remain ignored; checkpoint checksums support separate sharing.

## Validation

- 167 project tests passed in the CUDA/Transformers environment.
- Five prediction CSVs audited for row identity, metadata, native labels and scores.
- Dev selection, source/data hashes and native checkpoint round trip verified.
- Existing MELD/DailyDialog reports, configs and shared shift definition unchanged.
- English submission artifacts; personal reference notes remain outside Git.

The aggregate score increase does not establish a shift-subset improvement or
pairwise statistical significance. See reports/b3/emowoz/results.md for details.
