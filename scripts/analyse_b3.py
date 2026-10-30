"""Figures and tables for the B3 slides, from the outputs of ``train_b3.py``.

    python scripts/analyse_b3.py            # needs predictions/b3_* and models/b3_meld (a few minutes on GPU)
    python scripts/analyse_b3.py --quick    # reads outputs_quick/ instead
    python scripts/analyse_b3.py --out DIR  # reads the output root given to train_b3.py --out

Writes to reports/b3/figures/ (every PNG is at least 1600 px wide and has a CSV of its numbers):
    meld_strata.png / .csv            weighted-F1 by stratum and shift type (shared chart style)
    meld_b1_vs_b3.png / .csv          B1 TF-IDF vs B3 RoBERTa on the same strata (if B1 predictions exist)
    meld_confusion.png / .csv         row-normalised confusion matrix
    meld_per_class.png / .csv         F1 per emotion
    meld_length.png / .csv            weighted-F1 by utterance length
    meld_ig_top_words.png / .csv      words with the highest Integrated-Gradients attribution per emotion
    meld_ig_examples.html / .csv      coloured example utterances (screenshot for the slide)
    transfer_dailydialog.csv          MELD-trained vs DailyDialog-trained B3 on DailyDialog
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.labels import EMOTIONS  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.eval import plots as P  # noqa: E402
from emotionshift.models.roberta import RobertaUtterance  # noqa: E402

DPI = 260   # 6.4 in x 260 = 1664 px: the narrowest figure still clears the 1600 px hand-in rule


def load_preds(out: Path, model: str, dataset: str, split: str) -> list[pd.DataFrame]:
    return [pd.read_csv(f) for f in sorted((out / "predictions").glob(f"{model}_{dataset}_{split}_seed*.csv"))]


def mean_stratified(runs: list[pd.DataFrame]) -> pd.DataFrame:
    strat = pd.concat([M.stratified(t) for t in runs])
    return strat.groupby("group", sort=False).mean(numeric_only=True).reset_index()


def save(fig, table: pd.DataFrame, fig_dir: Path, name: str) -> None:
    fig.savefig(fig_dir / f"{name}.png", dpi=DPI)
    table.to_csv(fig_dir / f"{name}.csv", index=False)
    plt.close(fig)


def compare_chart(both: pd.DataFrame, title: str):
    """B1 vs B3 weighted-F1 per stratum / shift type: two series, direct labels, no legend box."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(both))
    for k, (col, name) in enumerate([("b1", "B1 TF-IDF"), ("b3", "B3 RoBERTa")]):
        xs = x + (k - 0.5) * 0.36
        ax.bar(xs, both[col], width=0.34, color=P.SERIES[k], edgecolor=P.SURFACE, linewidth=1.5, label=name)
        for xi, v in zip(xs, both[col]):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=P.INK)
    ax.set_xticks(x, [f"{g}\nn={n}" for g, n in zip(both.label, both.n)])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 100)
    ax.set_ylabel("weighted-F1 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(title, pad=30)
    fig.tight_layout()
    return fig


def length_chart(t: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    P._bars(ax, list(t.length), t.weighted_f1.values, t.n.values)
    ax.set_xlabel("utterance length (words)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def top_words_chart(top: pd.DataFrame, title: str):
    emos = [e for e in EMOTIONS if e in set(top.emotion) and e != "neutral"]
    fig, axes = plt.subplots(1, len(emos), figsize=(max(2.3 * len(emos), 7), 3.8))
    for ax, emo in zip(np.atleast_1d(axes), emos):
        d = top[top.emotion == emo].iloc[::-1]
        ax.barh(d.word, d.mean_attribution, color=P.SERIES[0], height=0.65)
        ax.set_title(emo, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def render(words, scores, label, p, gold) -> str:
    m = max(np.abs(scores).max(), 1e-6)
    spans = []
    for w, s in zip(words, scores):
        rgb = "235,104,52" if s > 0 else "42,120,214"      # orange = towards, blue = away (P.SERIES)
        spans.append(f'<span style="background:rgba({rgb},{abs(s) / m:.2f});padding:2px 4px;'
                     f'border-radius:4px">{html.escape(w)}</span>')
    return (f'<div style="margin:10px 0;font:20px/1.6 sans-serif"><b>pred {label}</b> ({p:.2f}) · gold {gold}: '
            + " ".join(spans) + "</div>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", help="output root used by train_b3.py --out (default: the repo)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "outputs_quick" if a.quick else ROOT
    fig_dir = out / "reports" / "b3" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    b3 = load_preds(out, "b3", "meld", "test")
    t = b3[0]
    if a.quick:
        print("quick mode: skipping charts that need every shift type")
    else:
        strat = mean_stratified(b3)
        fig = P.strata_chart(strat, "MELD test: B3 RoBERTa, utterance only, by turn type")
        save(fig, strat.round(4), fig_dir, "meld_strata")

    b1 = load_preds(ROOT, "b1", "meld", "test")
    if b1 and not a.quick:
        s1 = mean_stratified(b1).set_index("group")
        s3 = strat.set_index("group")
        rows = ["all", "stratum:first", "stratum:stable", "stratum:shift", "shift_type:onset", "shift_type:offset",
                "shift_type:surprise", "shift_type:within_negative", "shift_type:valence_flip"]
        labels = ["all", "first", "stable", "shift", "onset", "offset", "surprise", "within-\nnegative", "valence\nflip"]
        both = pd.DataFrame({"group": rows, "label": labels, "n": s3.loc[rows, "n"].astype(int).values,
                             "b1": 100 * s1.loc[rows, "weighted_f1"].values,
                             "b3": 100 * s3.loc[rows, "weighted_f1"].values})
        gain = both.loc[0, "b3"] - both.loc[0, "b1"]
        where = "on every turn type" if (both.b3 > both.b1).all() else "overall"
        fig = compare_chart(both, f"MELD test: RoBERTa gains {gain:.1f} points over TF-IDF {where}")
        save(fig, both.drop(columns="label").round(2), fig_dir, "meld_b1_vs_b3")

    cm = M.confusion(t.emotion, t.pred)
    fig = P.confusion_chart(cm, "MELD test: recall per gold emotion (%)")
    save(fig, cm.reset_index().rename(columns={"index": "gold"}), fig_dir, "meld_confusion")

    pc = M.per_class_f1(t.emotion, t.pred)
    fig = P.per_class_chart(pc, "MELD test: F1 per emotion (B3, first seed)")
    save(fig, pc.round(4), fig_dir, "meld_per_class")

    t = t.assign(n_words=t.text.str.split().str.len())
    t["length"] = pd.cut(t.n_words, [0, 3, 7, 15, 1000], labels=["1-3", "4-7", "8-15", "16+"])
    ln = pd.DataFrame([{"length": str(k), "n": len(g), "weighted_f1": 100 * M.scores(g.emotion, g.pred)["weighted_f1"]}
                       for k, g in t.groupby("length", observed=True)])
    fig = length_chart(ln, "MELD test: longer utterances are harder")
    save(fig, ln.round(2), fig_dir, "meld_length")

    # Integrated Gradients on the saved model (best seed on dev).
    model = RobertaUtterance.load(out / "models" / "b3_meld")
    model_preds = t.assign(pred_saved=model.predict(t.text))
    correct = model_preds[model_preds.pred_saved == model_preds.emotion]
    n_per = 10 if a.quick else 120
    recs = []
    for emo in EMOTIONS:
        cand = correct[correct.emotion == emo]
        for text in cand.sample(min(n_per, len(cand)), random_state=0).text:
            words, scores, _, _ = model.attributions(text)
            for w, s in zip(words, scores):
                w = re.sub(r"[^\w']", "", w.lower())
                if w:
                    recs.append((emo, w, s))
    att = pd.DataFrame(recs, columns=["emotion", "word", "attribution"])
    top = (att.groupby(["emotion", "word"]).attribution.agg(mean_attribution="mean", count="count").reset_index()
           .query("count >= 3").sort_values("mean_attribution", ascending=False).groupby("emotion").head(6))
    if len(top):
        fig = top_words_chart(top, "Integrated Gradients: the words B3 relies on for each emotion")
        save(fig, top.round(4), fig_dir, "meld_ig_top_words")

    examples = []
    for emo in EMOTIONS:
        cand = correct[(correct.emotion == emo) & correct.n_words.between(4, 14)]
        if len(cand):
            examples.append(cand.sample(1, random_state=1).iloc[0])
    wrong = model_preds[(model_preds.pred_saved != model_preds.emotion) & (model_preds.stratum == "shift")
                        & model_preds.n_words.between(4, 14)]
    examples += [r for _, r in wrong.sample(min(3, len(wrong)), random_state=0).iterrows()]
    blocks, rows = [], []
    for r in examples:
        words, scores, label, p = model.attributions(r.text)
        blocks.append(render(words, scores, label, p, r.emotion))
        rows.append({"text": r.text, "gold": r.emotion, "pred": label, "prob": round(p, 3),
                     "attributions": " ".join(f"{w}:{s:+.2f}" for w, s in zip(words, scores))})
    (fig_dir / "meld_ig_examples.html").write_text(
        '<html><body style="background:#fcfcfb;padding:24px"><p style="font:18px sans-serif">'
        "Integrated Gradients: orange pushes towards the predicted emotion, blue away.</p>"
        + "".join(blocks) + "</body></html>")
    pd.DataFrame(rows).to_csv(fig_dir / "meld_ig_examples.csv", index=False)

    # Transfer: MELD-trained vs DailyDialog-trained, both scored on DailyDialog.
    rows = []
    for model_name, label in [("b3meld", "trained on MELD (transfer)"), ("b3", "trained on DailyDialog")]:
        for split in ["test_dedup", "test"]:
            runs = load_preds(out, model_name, "dailydialog", split)
            if runs:
                s = [M.scores(r.emotion, r.pred) for r in runs]
                d = [M.derived_shift_detection(r)["shift_f1"] for r in runs]
                rows.append({"model": label, "split": split, "n": s[0]["n"],
                             **{k: round(100 * np.mean([x[k] for x in s]), 2)
                                for k in ["micro_f1_no_neutral", "macro_f1", "weighted_f1"]},
                             "derived_shift_f1": round(100 * np.mean(d), 2)})
    pd.DataFrame(rows).to_csv(fig_dir / "transfer_dailydialog.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("figures in", fig_dir)


if __name__ == "__main__":
    main()
