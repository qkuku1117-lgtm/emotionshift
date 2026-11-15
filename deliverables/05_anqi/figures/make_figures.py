import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1])
INK, MUTED, BLUE, ORANGE, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white"})

# 1. gain over B3 on stable vs shift turns (points, 5 seed pairs, 95% CI from reports/summary method)
g = pd.DataFrame({"dataset": ["MELD", "DailyDialog", "EmoWOZ"],
                  "metric": ["weighted-F1", "micro-F1 excl. neutral", "weighted-F1 excl. neutral"],
                  "stable_gain": [3.12, 7.38, 11.15], "stable_ci_low": [1.67, 3.88, 7.79], "stable_ci_high": [4.68, 10.73, 14.65],
                  "shift_gain": [-1.05, 0.73, 4.33], "shift_ci_low": [-2.52, -1.25, 3.16], "shift_ci_high": [0.47, 2.65, 5.67]})
g.to_csv(out / "gain_stable_vs_shift.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
x = range(len(g)); w = 0.36
for off, col, name, c in [(-w / 2 - 0.01, "stable", "stable turns", BLUE), (w / 2 + 0.01, "shift", "shift turns", ORANGE)]:
    v = g[f"{col}_gain"]
    ax.bar([i + off for i in x], v, w, color=c, label=name, zorder=3)
    ax.vlines([i + off for i in x], g[f"{col}_ci_low"], g[f"{col}_ci_high"], color=INK, lw=1.4, zorder=4)
    for i, val in zip(x, v):
        ax.text(i + off + 0.05, val + (0.5 if val >= 0 else -1.1), f"{val:+.1f}", ha="left", color=INK, fontsize=12)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(list(x), g.dataset); ax.set_ylabel("B4 (k = 10) − B3, points")
ax.grid(axis="y", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Context helps most where the emotion does not change", loc="left", color=INK, fontsize=15)
fig.tight_layout(); fig.savefig(out / "gain_stable_vs_shift.png"); plt.close(fig)

# 2. input controls on DailyDialog (k = 10, 5 seeds)
c = pd.DataFrame({"input": ["real history", "no history", "shuffled history", "shuffled speakers"],
                  "micro_f1_no_neutral": [59.47, 51.86, 50.06, 59.42], "sd": [0.70, 0.67, 0.79, 0.68]})
c.to_csv(out / "controls_dailydialog.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
ax.bar(c.input, c.micro_f1_no_neutral, 0.55, color=BLUE, zorder=3)
ax.vlines(range(len(c)), c.micro_f1_no_neutral - c.sd, c.micro_f1_no_neutral + c.sd, color=INK, lw=1.4, zorder=4)
for i, v in enumerate(c.micro_f1_no_neutral):
    ax.text(i + 0.06, v + 1.0, f"{v:.1f}", ha="left", color=INK, fontsize=12)
ax.set_ylim(0, 68); ax.set_ylabel("micro-F1 excl. neutral (%)")
ax.grid(axis="y", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("DailyDialog: a shuffled history is worse than none", loc="left", color=INK, fontsize=15)
fig.tight_layout(); fig.savefig(out / "controls_dailydialog.png"); plt.close(fig)

# 3. headline numbers
pd.DataFrame({"dataset": ["MELD", "DailyDialog (test_dedup)", "EmoWOZ"], "metric": g.metric,
              "b3_no_context": [62.85, 55.34, 78.09], "b4_k10": [63.34, 58.71, 84.38], "b4_k10_sd": [0.41, 0.46, 0.16],
              "b4_k_all": [63.30, 59.62, 84.31], "b4_k_all_sd": [0.42, 0.82, 0.22],
              "k_all_minus_k10": [-0.04, 0.92, -0.07], "ci_low": [-1.33, -0.42, -0.44], "ci_high": [0.58, 1.91, 0.57]}).to_csv(out / "headline.csv", index=False)
from PIL import Image
for f in out.glob("*.png"): print(f.name, Image.open(f).size)
