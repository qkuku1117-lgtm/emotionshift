# Workstream 6 — ShiftAware video materials · Wang Zhoufu

- `06_Fpengz.pptx`: editable, 16:9 PowerPoint; three narrated slides and one silent reference slide. Narration is also in speaker notes.
- `script.md`: spoken narration for approximately 90 seconds, slide directions, and source details.
- `figures/meld_f1.png` + `meld_f1.csv`: slide 3 chart as a standalone 2000 px figure (mean ± SD over 5 seeds).
- `figures/shift_vs_context_roberta.csv`: H3 paired tests (ShiftAware vs context RoBERTa) by stratum, MELD and DailyDialog.
- `results.csv`: shared metrics (weighted, macro, micro-no-neutral F1; mean, SD, 95% CI) for B3, B4 and ShiftAware on both datasets.
- `contribution.txt`: 4-line contribution note for the group reflection.
- `refs.txt`: APA references.
- `slides/`, `preview.png`: current 1920×1080 renders.
- `build_slides.py`: reproduces the deck, CSVs and figure using `03_thanh/03_thanh.pptx` as the template
  reference (keeps its canvas, theme and masters; replaces the slide contents).

Open the PowerPoint in slide-show mode and rehearse with `script.md`.
The reference slide needs no narration.

Recorded video: `06_Fpengz.mp4` (Wang Zhoufu). Submit the video and supporting materials through a pull request.

Rebuild from the repository root:

```bash
uv run --no-project --with python-pptx --with matplotlib python deliverables/06_shiftaware/build_slides.py
```

Results use the final group comparison, including B4. The older ShiftAware
analysis notebook's statement that B4 is pending is superseded by
`reports/summary/summary.md`.
