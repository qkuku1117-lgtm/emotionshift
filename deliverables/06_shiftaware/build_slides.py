"""Build the ShiftAware video slides from Thanh's committed team-template deck.

Run from the repo root:
uv run --no-project --with python-pptx --with matplotlib python deliverables/06_shiftaware/build_slides.py
"""

import csv
import re
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
prs = Presentation(ROOT / 'deliverables/03_thanh/03_thanh.pptx')
# Retain the reference deck's canvas, theme and masters; replace its four slides' contents.
for slide in prs.slides:
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)

INK, BLUE, PALE, WHITE, MUTED = '16202E', '256ABF', 'EEF2F7', 'FCFCFB', '586574'
SCALE = prs.slide_width / Inches(20)


def pos(v):
    return int(Inches(v) * SCALE)


def text(s, value, x, y, w, h, size=30, color=INK, bold=False, font='Calibri'):
    box = s.shapes.add_textbox(pos(x), pos(y), pos(w), pos(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name, p.font.size = font, Pt(size * SCALE)
        p.font.bold, p.font.color.rgb = bold, RGBColor.from_string(color)
        p.space_after = Pt((0 if font == 'Courier New' else 10) * SCALE)
    return box


def rect(s, x, y, w, h, fill=PALE):
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, pos(x), pos(y), pos(w), pos(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    return shape


def base(index, title, source):
    s = prs.slides[index]
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string(WHITE)
    text(s, title, 1.2, .7, 17.6, 1.3, 49, bold=True, font='Cambria')
    text(s, source, 1.2, 10.05, 16.9, .65, 19, MUTED)
    text(s, str(index + 1), 18.55, 10.4, .3, .4, 22, MUTED)
    return s


script = (OUT / 'script.md').read_text()
narration = re.findall(r'^> (.+)$', script, re.M)

s = base(0, 'ShiftAware balances history with current evidence',
         'Source: emotionshift/models/shiftaware/{model,heads}.py · RoBERTa: Liu et al. (2019)')
text(s, 'If we track each speaker, can we better recognise when their emotion changes?', 1.2, 2.0, 17.6, .6, 31, BLUE)
for x, y, heading, body in [
    (1.2, 3.25, 'SPEAKER PRIOR', 'Own speaker memory\nPrevious predicted emotion'),
    (1.2, 6.0, 'CURRENT EVIDENCE', 'Current utterance (RoBERTa)\nOther speakers + conversation'),
    (8.3, 4.6, 'LEARNED GATE β', 'Receives shift probability\nBalances prior and evidence'),
    (14.4, 4.6, 'PREDICT', '7 emotion probabilities\nThen update memory'),
]:
    width = 6.0 if x == 1.2 else (5.2 if x == 8.3 else 4.4)
    rect(s, x, y, width, 2.1, INK if x == 14.4 else PALE)
    text(s, heading, x+.35, y+.25, width-.7, .45, 24, '9EC5F4' if x == 14.4 else BLUE, True)
    text(s, body, x+.35, y+.85, width-.7, 1.05, 27, WHITE if x == 14.4 else INK)
text(s, '↘', 7.2, 4.1, .8, .8, 48, BLUE)
text(s, '↗', 7.2, 6.0, .8, .8, 48, BLUE)
text(s, '→', 13.55, 5.1, .8, .8, 48, BLUE)
text(s, 'P(emotion) = (1 − β) P(prior) + β P(evidence)', 1.2, 8.75, 17.6, .65, 36, bold=True)
text(s, 'Wang Zhoufu · Workstream 6', 1.2, 9.5, 12, .4, 21, MUTED)
s.notes_slide.notes_text_frame.text = narration[0]

s = base(1, 'A learned gate mixes two emotion distributions',
         'Source: emotionshift/models/shiftaware/heads.py · lines 85–86 and 94–97 (mixture branch)')
rect(s, 1.2, 2.5, 17.6, 4.1)
source = (ROOT / 'emotionshift/models/shiftaware/heads.py').read_text().splitlines()
snippet = '\n'.join([line[8:] for line in source[84:86]] + [''] + [line[12:] for line in source[93:97]])
text(s, snippet, 1.55, 2.95, 16.9, 3.4, 25, font='Courier New')
text(s, 'Normalise separately → combine in log space', 1.2, 7.15, 17.6, .7, 36, BLUE, True)
text(s, 'β is a learned gate; the shift probability is one of its inputs.\nMemory stores predictions; gold labels enter only the loss.',
     1.2, 8.15, 17.6, 1.2, 30)
s.notes_slide.notes_text_frame.text = narration[1]

with (ROOT / 'reports/summary/e1_main.csv').open() as f:
    rows = {r['model']: r for r in csv.DictReader(f) if r['dataset'] == 'meld'}
models = [('b3', 'Utterance RoBERTa'), ('b4', 'Context RoBERTa'), ('shiftaware', 'ShiftAware')]
chart_data = CategoryChartData()
chart_data.categories = [name for _, name in models]
for metric, label in [('weighted_f1', 'Weighted F1'), ('macro_f1', 'Macro F1')]:
    chart_data.add_series(label, [float(rows[k][metric+'_mean']) for k, _ in models])

s = base(2, 'Higher macro-F1 did not yield better shift recognition',
         'Sources: reports/summary/{e1_main,paired}.csv · MELD test · 5 seeds · variants chosen on dev')
chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                           pos(1.1), pos(2.45), pos(10.6), pos(6.45), chart_data).chart
chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout = False
chart.legend.font.size = Pt(23 * SCALE)
chart.value_axis.minimum_scale, chart.value_axis.maximum_scale = 0, 75
chart.value_axis.major_unit = 25
chart.value_axis.tick_labels.font.size = Pt(23 * SCALE)
chart.category_axis.tick_labels.font.size = Pt(22 * SCALE)
chart.category_axis.tick_labels.font.name = 'Calibri'
plot = chart.plots[0]
plot.has_data_labels = True
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.number_format = '0.00'
plot.data_labels.font.size = Pt(23 * SCALE)
for series, color in zip(chart.series, ['A8B6C6', BLUE]):
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = RGBColor.from_string(color)
    series.format.line.fill.background()
text(s, 'MELD F1 (%)', 1.4, 2.05, 5, .4, 23, MUTED)
rect(s, 12.2, 2.45, 6.6, 6.6)
text(s, 'SHIFT TARGET NOT MET', 12.6, 2.75, 5.8, .55, 25, BLUE, True)
text(s, '−0.36 pp', 12.6, 3.35, 5.8, .85, 52, bold=True)
text(s, 'MELD shift turns vs context\nRoBERTa · 95% CI [−2.44, +1.71]', 12.6, 4.4, 5.8, 1.1, 24)
text(s, 'MELD stable turns: −2.68 pp\n95% CI [−4.62, −0.59]', 12.6, 5.55, 5.8, 1.1, 24)
text(s, 'DailyDialog shift turns: −4.72 pp\n95% CI [−6.70, −2.81]', 12.6, 6.7, 5.8, 1.1, 24)
text(s, 'Shift head AUROC ≈ 75: detects change,\nnot the new emotion', 12.6, 7.85, 5.8, 1.1, 22, MUTED)
text(s, 'Only ShiftAware uses class weighting, so its macro-F1 gain does not isolate the shift head.',
     1.2, 9.15, 17.6, .55, 25, MUTED)
s.notes_slide.notes_text_frame.text = narration[2]

s = base(3, 'References', 'EE6405 · EmotionShift · Workstream 6 · Wang Zhoufu')
text(s, 'Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O.,\nLewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa:\nA robustly optimized BERT pretraining approach. arXiv.\nhttps://arxiv.org/abs/1907.11692',
     1.2, 2.35, 17.6, 2.8, 28)
text(s, 'EmotionShift project team. (2026). EmotionShift [Source code and\nexperimental results]. GitHub.\nhttps://github.com/qkuku1117-lgtm/emotionshift', 1.2, 5.8, 17.6, 2.0, 28)
text(s, 'Result sources: reports/summary/e1_main.csv; reports/summary/paired.csv\nSelection: reports/shiftaware/selection.md', 1.2, 8.2, 17.6, 1.1, 23, MUTED)
s.notes_slide.notes_text_frame.text = 'No narration. Show briefly after the conclusion; merge into group references if preferred.'

prs.save(OUT / '06_Fpengz.pptx')

# Shared result numbers for integration: headline metrics on both datasets, plus the H3 paired tests.
with (ROOT / 'reports/summary/e1_main.csv').open() as f:
    every = [r for r in csv.DictReader(f)
             if r['model'] in dict(models) and r['dataset'] in ('meld', 'dailydialog')]
with (OUT / 'results.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    metrics = ['weighted_f1', 'macro_f1', 'micro_f1_no_neutral']
    writer.writerow(['dataset', 'model', 'seeds'] + [f'{m}_{k}' for m in metrics
                                                     for k in ('mean', 'sd', 'ci_low', 'ci_high')])
    for r in every:
        writer.writerow([r['dataset'], dict(models)[r['model']], r['seeds']]
                        + [r[f'{m}_{k}'] for m in metrics for k in ('mean', 'sd', 'ci_low', 'ci_high')])
with (ROOT / 'reports/summary/paired.csv').open() as f:
    paired = [r for r in csv.DictReader(f)
              if r['a'] == 'b4' and r['b'] == 'shiftaware' and r['dataset'] in ('meld', 'dailydialog')]
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
with (FIG / 'shift_vs_context_roberta.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, ['dataset', 'group', 'metric', 'n', 'diff', 'ci_low', 'ci_high', 'p'],
                            extrasaction='ignore')
    writer.writeheader()
    writer.writerows(paired)

# Slide 3 chart as a standalone figure (team guide: PNG >= 1600 px plus matching CSV).
with (FIG / 'meld_f1.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'weighted_f1_mean', 'weighted_f1_sd', 'macro_f1_mean', 'macro_f1_sd'])
    for k, name in models:
        r = rows[k]
        writer.writerow([name, r['weighted_f1_mean'], r['weighted_f1_sd'], r['macro_f1_mean'], r['macro_f1_sd']])
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed: skipped figures/meld_f1.png')
else:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    xs = range(len(models))
    for offset, (metric, label, color) in zip((-.19, .19), [('weighted_f1', 'Weighted F1', '#A8B6C6'),
                                                         ('macro_f1', 'Macro F1', '#256ABF')]):
        means = [float(rows[k][metric + '_mean']) for k, _ in models]
        sds = [float(rows[k][metric + '_sd']) for k, _ in models]
        bars = ax.bar([x + offset for x in xs], means, .38, yerr=sds, capsize=4, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=11)
    ax.set_xticks(list(xs), [name for _, name in models], fontsize=12)
    ax.set_ylabel('MELD test F1 (%), mean ± SD over 5 seeds', fontsize=11)
    ax.set_ylim(0, 75)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'meld_f1.png')
print('Wrote 06_Fpengz.pptx, results.csv and figures/')
print('Narration words:', sum(len(p.split()) for p in narration))
