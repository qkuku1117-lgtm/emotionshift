# KL Y (#4 RoBERTa, utterance only): narration script

One section per slide; the same text is in each slide's speaker notes.
Slides: copy of the team Canva template, "04_kly EmotionShift B3 RoBERTa".
Every number comes from `reports/b3/` (5 seeds, mean). About 200 words, roughly 2 minutes.

## 01 · RoBERTa, utterance only

Hi, I'm KL Y. My part is B3: RoBERTa, reading one sentence at a time, with no context. The question is: how far can it get from the words alone?

## 02 · How B3 works

RoBERTa-base splits the sentence into sub-word tokens and encodes it. The first token's vector goes to a linear layer and a softmax over seven emotions. I fine-tune it with cross-entropy. There are no class weights, so rare emotions like disgust and fear are almost never predicted.

## 03 · The core in code

I use the Hugging Face Trainer. It checks the dev set after each epoch and keeps the best one, so the test set is never used for choosing. I ran five seeds. On a laptop CPU it takes about 100 milliseconds per sentence.

## 04 · Test results

B3 gets 62.9 on MELD and 55.3 on DailyDialog. On MELD that is six points above TF-IDF and eight above BiGRU. So pre-training helps, even without context.

## 05 · Without context, shift turns stay 14 points harder

The key finding: shift turns are still much harder. B3 gets 69.7 on stable turns but only 55.9 on shift turns, where the emotion changes. Integrated Gradients show it relies on emotion words like "sorry" and "stupid". When the words don't show the change, the model needs context. That is what our context models are for.

## 06 · References

No narration.
