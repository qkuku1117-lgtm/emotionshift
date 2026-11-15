# HU ANQI (#5 RoBERTa with context): narration script

One section per slide; the same text is in each slide's speaker notes.
Slides: `slides.pptx` (built in the style of the team deck; `slides.pdf` is the same deck as a PDF).
Every number comes from `reports/b4/` and `reports/b4all/` (5 seeds, mean). About 275 words, roughly 2 minutes.
To shorten it to 1 min 30 s, skip slide 03 and the last sentence of slide 06.

## 01 · RoBERTa with context

Hi, I'm Anqi. My part is B4: RoBERTa with context. B3 read one sentence at a time. My question is: what changes when the model can also read the conversation before it?

## 02 · How B4 works

It is the same RoBERTa as B3, with the same training recipe. Only the input changes. Before the target sentence we add the previous k turns, each with its speaker's name. If the input gets too long, we cut the oldest turns, so the target is always kept. I chose k on the dev set.

## 03 · The core in code

This is the core. For every turn, it takes a window of the last k turns and joins them as speaker, colon, text. With k equal to zero it returns just the sentence, which is exactly B3's input. So the comparison with B3 is fair.

## 04 · Test results

On the test sets, context helps on all three datasets: 63.3 on MELD, 58.7 on DailyDialog and 84.4 on EmoWOZ. That is half a point, three points and six points above B3. And ten turns or the whole history give the same score.

## 05 · Context helps most on stable turns

The key finding is where the gain comes from. Stable turns, where the emotion stays the same, gain three, seven and eleven points. Shift turns gain almost nothing on MELD and DailyDialog. So context helps the model continue an emotion, not notice a change.

## 06 · It uses the right conversation

Does it really use the conversation? If we shuffle in turns from another dialogue, the score drops below having no history at all. Shuffling the speaker names changes nothing. And with class weights, B4-plus beats B3 on macro-F1 by three points.

## 07 · References

No narration.
