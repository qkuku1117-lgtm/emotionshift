# 05_anqi · workstream #5, RoBERTa with context (B4)

| File | What it is |
|---|---|
| `05_anqi.mp4` | my recording: the slides with my own voice (2 min 18 s, 1920 x 1080) |
| `slides.pptx`, `slides.pdf` | 7 slides in the team deck's style; the narration is in each slide's speaker notes |
| `script.md` | narration, one section per slide (about 2 minutes; how to cut it to 1 min 30 s is at the top) |
| `figures/` | `gain_stable_vs_shift`, `controls_dailydialog` as PNG (1800 px wide) with the CSV behind each; `headline.csv`; `make_figures.py` redraws them |
| `refs.txt` | references, APA |
| `contribution.txt` | contribution note for the group reflection |
| `build_slides.js` | rebuilds `slides.pptx` (pptxgenjs) |


**Code shown in the video:** slide 03, `emotionshift/data/context.py` lines 40 to 46 (`dialogue_inputs`), also cell 2 of
`notebooks/b4_context.ipynb`.

**How to run:** `python scripts/train_b4.py --datasets meld dailydialog emowoz` (GPU; about 1.5 / 9 / 10 minutes per seed on a Colab
A100), then `python scripts/analyse_b4_ctx.py --compare b4all` (CPU, seconds). No extra packages beyond `.[transformers]`.

**Numbers:** `reports/b4/results.md`; write-up `docs/B4_CONTEXT.md`. The code and results are in the pull request
"Workstream 5: context RoBERTa re-run in one pipeline on MELD, DailyDialog and EmoWOZ".
