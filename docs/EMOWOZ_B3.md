# Native EmoWOZ: single-utterance RoBERTa (workstream 4)

Status: all five seeds and official test evaluation completed. Results, verified
predictions, figures and an executed notebook are available under reports/b3/emowoz/
and notebooks/emowoz_b3.ipynb. The headline non-neutral weighted-F1 is 78.09%
(five-seed mean); population SD is 0.29 percentage points.

Reuse the original B3 RoBERTa-base implementation, retaining its original author
credit. HE RUIZHI adds the native EmoWOZ label handling and experiment runner.
No MELD/DailyDialog experiments, shared loaders or shift definitions are changed.

Official labelled user targets: 66,474 train / 8,509 dev / 8,634 test.
Labels: neutral, fearful, dissatisfied, apologetic, abusive, excited, satisfied.
System turns are unlabelled and are not targets. Inputs use the current utterance
only. No Ekman mapping, dialogue context or cross-domain transfer is used.

Predeclare 3 epochs (as for the larger original DailyDialog experiment), seeds
0-4, learning rate 1e-5, effective batch 32, max length 128, warmup 0.1,
weight decay 0.01 and unweighted cross-entropy. Select the best epoch per seed
on native non-neutral weighted-F1, and the demo seed by best dev score.
Test evaluation waits until all five seeds and the dev choice are saved.
Report all five seeds, all shared native-label metrics, class/stratum tables,
derived shift F1, population SD and 1,000 whole-dialogue bootstrap intervals
for seed 0. These intervals do not describe uncertainty of the five-seed mean.

The official FacebookAI/roberta-base revision is fixed in configs/emowoz_b3.yaml.
A 6GB GPU uses micro-batches of 4 with accumulation 8, eval batches of 8 and
gradient checkpointing. Effective batch remains 32; numerical results need not
match a larger physical batch exactly. Default B3 training settings remain
unchanged when these optional controls are omitted.

Use an environment with the project installed and CUDA PyTorch, transformers
and accelerate (the prepared CUDA environment uses torch 2.6.0+cu126,
transformers 5.21.1 and accelerate 1.15.0):

```powershell
python -X utf8 scripts/train_emowoz_roberta.py
# Train one seed at a time; completed seeds are skipped on the next invocation:
python -X utf8 scripts/train_emowoz_roberta.py --seeds 0
```

Resume uses retained epoch checkpoints in ignored checkpoints/b3_emowoz/.
Source/data/config/runtime signatures guard against mixing experiments.
An interruption before the first epoch checkpoint retrains from that seed.
Large model weights remain ignored under models/b3_emowoz/ and must be shared
separately for the demo. Test predictions and reports use dedicated new paths.
Close competing GPU applications before starting; the runner checks for at
least 3.5GB free GPU memory. Actual memory needs can still depend on inputs.

Gradient accumulation and checkpointing follow the official Trainer guidance:
https://huggingface.co/docs/transformers/grad_accumulation
https://huggingface.co/docs/transformers/grad_checkpointing

AI assistance must be declared per the team guide. English project artifacts
are separate from personal Chinese reference notes.

Fresh seeds 2-4 took approximately 52-57 minutes each on this laptop during the
final run. Earlier runs under different resource pressure were much slower.
Resumed seed metadata records time for that invocation rather than total time.
Large model weights are excluded from Git; checksums are in verification.json.
