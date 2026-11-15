# B0 baselines: video narration

He Ruizhi | Workstream 1 | EE6405 Table 4

Narration for the six-slide baseline presentation. Slides 1–5 contain the spoken script (210 words); slide 6 shows references without narration.

## Slide 1 — Introduction

My workstream establishes simple reference points before introducing more complex models.

The question is: how far can basic rules take us?

## Slide 2 — Three alternative rules

We compare three alternatives.

Majority ignores history and predicts the most common training emotion—the class prior.

Inertia assumes emotion persists. If a supporting recognizer predicted joy for the same speaker’s previous utterance, inertia predicts joy again.

But emotion can change. Markov learns transition probabilities from training data and uses argmax to select the most likely emotion given that previous prediction.

## Slide 3 — Fair information access and oracle diagnosis

To evaluate these rules fairly, we must control what information they receive.

We predict before updating history, excluding current-turn information.

We also report oracle versions using true previous labels. These help distinguish historical prediction errors from the limitations of history itself. They are diagnostic references, not deployable results.

## Slide 4 — Results

With that distinction clear, the results show inertia outperforming majority.

So previous predictions contain useful information—but learning transitions does not automatically improve F1.

## Slide 5 — Interpretation and handover

Why does Markov struggle?

The diagram shows that after joy, Markov prefers neutral because its training probability is higher.

Many history labels lead to neutral, reducing coverage of emotional classes.

More fundamentally, these rules do not read the current utterance.

That leads to the next workstream: can TF-IDF and SVM capture new emotional evidence directly from the words?

## Slide 6 — References

No narration. Hold for 2 seconds, then stop.

## Delivery cues

Pause between the three independent rules. Do not present them as sequential stages. Inertia and Markov use previous same-speaker predictions from the supporting recognizer, not previous gold labels or recursively copied inertia outputs. Oracle variants are diagnostic-only. The transition diagram shows one illustrative training row; the 94.0% callout separately describes test predictions. The final question hands over to workstream 2; no other model results are presented.

The paired interval in the References note is exploratory, unadjusted and conditional on seed 0; do not infer across-training-seed uncertainty. The slide 4 MELD and DailyDialog values use different metrics. References and all prediction data are unchanged. Native Canva presenter notes remain template notes; use this script.
