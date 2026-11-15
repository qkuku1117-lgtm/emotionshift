"""Tables and a figure from the context-aware RoBERTa runs (B4), CPU only, seconds.

    python scripts/analyse_b4_ctx.py [--tag b4] [--baseline b3] [--datasets meld dailydialog emowoz]

Reads what ``scripts/train_b4.py`` wrote (``reports/<tag>/<dataset>/summary.json``, ``sweep.csv``,
``stratified_<split>.csv`` and ``predictions/<tag>_<dataset>_<split>_seed<k>.csv``) and writes

    reports/<tag>/results_ctx.md                 every dataset in one file
    reports/<tag>/<dataset>/context_gain.csv     B4 - no-context reference by stratum / shift type (paired)
    reports/<tag>/figures/window_sweep.png       dev score against the window k, per dataset

The no-context reference is B3 (``predictions/b3_*``) when its files exist for the dataset,
otherwise the k = 0 control (``predictions/<tag>k0_*``) that ``train_b4.py`` trains for EmoWOZ.
Nothing is trained or overwritten except the files above; datasets not run yet are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.labels import emotions  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402

PRED = ROOT / "predictions"
GROUPS = ["all", "stratum:first", "stratum:stable", "stratum:shift"]
N_BOOT = 1000
MIN_N = 30          # groups smaller than this are listed but not interpreted or plotted


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def load_runs(prefix: str, ds: str, split: str, seeds: list[int]) -> list[pd.DataFrame] | None:
    files = [PRED / f"{prefix}_{ds}_{split}_seed{s}.csv" for s in seeds]
    return [pd.read_csv(f) for f in files] if all(f.exists() for f in files) else None


def reference(ds: str, split: str, seeds: list[int], tag: str, baseline: str):
    """(name, runs) of the no-context reference, or (None, None)."""
    for name in (baseline, f"{tag}k0", "b4k0"):
        runs = load_runs(name, ds, split, seeds)
        if runs:
            return name, runs
    return None, None


def context_gain(ds: str, split: str, seeds: list[int], tag: str, baseline: str, headline: str):
    """B4 minus the reference per group: mean over seeds, plus a paired dialogue-bootstrap CI (first seed)."""
    labels = emotions(ds)
    b4 = load_runs(tag, ds, split, seeds)
    name, ref = reference(ds, split, seeds, tag, baseline)
    if not b4 or not ref:
        return None, None
    per_b4 = pd.concat([M.stratified(t, labels=labels).set_index("group")[headline].rename(i) for i, t in enumerate(b4)], axis=1)
    per_ref = pd.concat([M.stratified(t, labels=labels).set_index("group")[headline].rename(i) for i, t in enumerate(ref)], axis=1)
    n = M.stratified(b4[0], labels=labels).set_index("group")["n"]
    key = ["dialogue_id", "turn"]
    joined = b4[0].merge(ref[0][key + ["pred"]], on=key, suffixes=("", "_ref"), validate="one_to_one")
    rows = []
    for g in per_b4.index:
        if n[g] == 0:
            continue
        mask = None
        if g.startswith("stratum:"):
            mask = joined.stratum == g.split(":")[1]
        elif g.startswith("shift_type:"):
            mask = joined.shift_type == g.split(":")[1]
        row = {"group": g, "n": int(n[g]), "ref": per_ref.loc[g].mean(), "b4": per_b4.loc[g].mean(),
               "delta": (per_b4.loc[g] - per_ref.loc[g]).mean()}
        if mask is None or mask.sum() > 0:
            ci = M.paired_bootstrap(joined, "pred_ref", "pred", metric=headline, mask=mask, labels=labels, n_boot=N_BOOT)
            row.update(ci_low=ci["ci_low"], ci_high=ci["ci_high"])
        rows.append(row)
    return name, pd.DataFrame(rows)


def b4plus_vs_b3(ds: str, split: str, seeds: list[int], plus_tag: str, baseline: str):
    """R3: B4+ minus B3 macro-F1, paired dialogue bootstrap on seed 0, mean difference over seeds."""
    labels = emotions(ds)
    plus, b3 = load_runs(plus_tag, ds, split, seeds), load_runs(baseline, ds, split, seeds)
    if not plus or not b3:
        return None
    key = ["dialogue_id", "turn"]
    joined = plus[0].merge(b3[0][key + ["pred"]], on=key, suffixes=("", "_ref"), validate="one_to_one")
    ci = M.paired_bootstrap(joined, "pred_ref", "pred", metric="macro_f1", labels=labels, n_boot=N_BOOT)
    diff = np.mean([M.scores(p.emotion, p.pred, labels)["macro_f1"] - M.scores(b.emotion, b.pred, labels)["macro_f1"]
                    for p, b in zip(plus, b3)])
    return {"diff": float(diff), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "seeds": len(plus)}


def window_compare(summaries: dict, reports: Path, tag: str, other: str) -> list[str]:
    """Chosen window (``tag``) against the same model at another window (``other``, e.g. b4all = k = all)."""
    lines = []
    for d, s in summaries.items():
        f = ROOT / "reports" / other / d / "summary.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text())
        split, h = next(iter(s["test"])), M.HEADLINE[d]
        a, b = s["test"][split][h], o["test"][split][h]
        a_runs, b_runs = load_runs(tag, d, split, s["seeds"]), load_runs(other, d, split, o["seeds"])
        ci = "n/a"
        if a_runs and b_runs:
            key = ["dialogue_id", "turn"]
            j = b_runs[0].merge(a_runs[0][key + ["pred"]], on=key, suffixes=("", "_ref"), validate="one_to_one")
            r = M.paired_bootstrap(j, "pred_ref", "pred", metric=h, labels=emotions(d), n_boot=N_BOOT)
            ci = f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}]"
        lines.append(f"| {d} | {split} | {s['chosen_k']} | {pct(a['mean'])} ± {pct(a['std'])} | {o['chosen_k']} | "
                     f"{pct(b['mean'])} ± {pct(b['std'])} | {100 * (b['mean'] - a['mean']):+.2f} | {ci} |")
    if not lines:
        return []
    return ["| Dataset | Split | k (chosen) | Headline | k (other) | Headline | Other - chosen | 95% CI (seed 0) |",
            "|---|---|---|---|---|---|---|---|"] + lines + [""]


def window_test(ds: str, summary: dict, tag: str, baseline: str) -> pd.DataFrame | None:
    """E3 on test: headline score against the window k by stratum, mean +- SD over the sweep seeds (k = 0: B3 or the control)."""
    labels, h = emotions(ds), M.HEADLINE[ds]
    split, seeds = next(iter(summary["test"])), summary.get("sweep_seeds") or summary["seeds"]
    ks = [x["k"] for x in summary.get("sweep", [])]
    rows = []

    def add(k, frames):
        per = pd.concat([M.stratified(t, labels=labels).set_index("group")[h].rename(i) for i, t in enumerate(frames)], axis=1)
        for g in GROUPS:
            if g in per.index:
                rows.append({"k": k, "group": g, "mean": per.loc[g].mean(),
                             "std": per.loc[g].std(ddof=1) if per.shape[1] > 1 else 0.0, "seeds": len(frames)})

    if "0" not in ks:
        b3 = load_runs(baseline, ds, split, seeds)
        if b3:
            add("0", b3)
    for k in ["0"] + [x for x in ks if x != "0"] if "0" in ks else ks:
        frames = [pd.read_csv(f) for sd in seeds
                  if (f := ROOT / "reports" / tag / ds / "runs" / f"k{k}_seed{sd}_{split}.csv").exists()]
        if frames:
            add(k, frames)
    return pd.DataFrame(rows) if rows else None


def window_test_figure(tables: dict, path: Path) -> None:
    if not tables:
        return
    fig, axes = plt.subplots(1, len(tables), figsize=(4.4 * len(tables), 3.6), squeeze=False)
    names = {"all": "all turns", "stratum:stable": "stable turns", "stratum:shift": "shift turns"}
    for ax, (d, t) in zip(axes[0], tables.items()):
        for g, label in names.items():
            x = t[t.group == g]
            ax.errorbar(range(len(x)), 100 * x["mean"], yerr=100 * x["std"], marker="o", capsize=3, label=label)
        ks = t[t.group == "all"].k.tolist()
        ax.set_xticks(range(len(ks)), [f"{k}\n(no ctx)" if k == "0" else k for k in ks])
        ax.set_xlabel("past turns in the input (k)")
        ax.set_ylabel(f"{M.HEADLINE[d]} on test (%)")
        ax.set_title(d)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gain_figure(g: pd.DataFrame, ds: str, split: str, path: Path) -> None:
    d = g.dropna(subset=["ci_low"]).iloc[::-1] if "ci_low" in g else g.iloc[::-1]
    d = d[d.n >= MIN_N]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(d) + 1.2))
    y = list(range(len(d)))
    ax.barh(y, 100 * d.delta, color=["#C0392B" if "shift" in x else "#256ABF" for x in d.group])
    if "ci_low" in d:                       # the CI is a paired bootstrap on one seed, so it need not be centred on the bar
        ax.hlines(y, 100 * d.ci_low, 100 * d.ci_high, color="#16202E", lw=1.5)
    ax.set_yticks(y, d.group.str.replace("stratum:", "").str.replace("shift_type:", "shift: "))
    ax.axvline(0, color="#16202E", lw=1)
    ax.set_xlabel(f"B4 - no context, {M.HEADLINE[ds]} on {ds} {split} (points, bar = seed mean, line = 95% CI)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def window_figure(summaries: dict, path: Path) -> None:
    sweeps = {d: s["sweep"] for d, s in summaries.items() if "sweep" in s}
    if not sweeps:
        return
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.2 * len(sweeps), 3.4), squeeze=False)
    for ax, (d, rows) in zip(axes[0], sweeps.items()):
        ks = [r["k"] for r in rows]
        ax.errorbar(range(len(ks)), [100 * r["dev_mean"] for r in rows], [100 * r["dev_std"] for r in rows],
                    marker="o", capsize=3)
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlabel("window k (previous turns)")
        ax.set_ylabel(f"dev {summaries[d]['selected_on_dev']} (%)")
        ax.set_title(f"{d}: chosen k = {summaries[d]['chosen_k']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build(tag: str, baseline: str, datasets: list[str], plus_tag: str = "b4plus", compare: str | None = None) -> str:
    reports = ROOT / "reports" / tag
    summaries = {}
    for d in datasets:
        f = reports / d / "summary.json"
        if f.exists():
            summaries[d] = json.loads(f.read_text())
    if not summaries:
        sys.exit(f"no summary.json under reports/{tag}/<dataset>/; run scripts/train_b4.py first")
    window_figure(summaries, reports / "figures" / "window_sweep.png")

    head = ["| Dataset | Test split | k | n | Headline | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Weighted-F1 excl. neutral | Derived shift F1 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for d, s in summaries.items():
        for split, r in s["test"].items():
            def c(k, r=r):
                return f"{pct(r[k]['mean'])} ± {pct(r[k]['std'])}"
            head.append(f"| {d} | {split} | {s['chosen_k']} | {r['n']} | {M.HEADLINE[d]} | {c('weighted_f1')} | "
                        f"{c('macro_f1')} | {c('micro_f1_no_neutral')} | {c('weighted_f1_no_neutral')} | "
                        f"{pct(r['derived_shift_detection']['shift_f1_mean'])} ± {pct(r['derived_shift_detection']['shift_f1_std'])} |")

    sweep = []
    for d, s in summaries.items():
        if "sweep" in s:
            sweep += [f"**{d}** (dev {s['selected_on_dev']}, {len(s['sweep_seeds'])} sweep seeds, chosen k = {s['chosen_k']}):", "",
                      "| k | seeds | dev mean | dev std |", "|---|---|---|---|"]
            sweep += [f"| {x['k']} | {x['n_seeds']} | {pct(x['dev_mean'])} | {pct(x['dev_std'])} |" for x in s["sweep"]] + [""]
        else:
            note = " (final seeds retrained with --k)"
            by = s["chosen_by"].replace(note, "") + (note if note in s["chosen_by"] else "")
            sweep += [f"**{d}**: window taken from {by} (k = {s['chosen_k']}).", ""]

    gain = []
    for d, s in summaries.items():
        split = next(iter(s["test"]))
        name, g = context_gain(d, split, s["seeds"], tag, baseline, M.HEADLINE[d])
        if g is None:
            gain += [f"**{d}**: no no-context reference found (needs `predictions/{baseline}_{d}_*` or `{tag}k0_{d}_*`).", ""]
            continue
        g.round(4).to_csv(reports / d / "context_gain.csv", index=False)
        gain_figure(g, d, split, reports / "figures" / f"context_gain_{d}.png")
        gain += [f"**{d}**, {split}, {M.HEADLINE[d]} (%), {tag} k={s['chosen_k']} minus `{name}`; "
                 f"CI = paired dialogue bootstrap on seed {s['seeds'][0]}:", "",
                 "| Group | n | Reference | B4 | Delta | 95% CI |", "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            ci = f"[{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]" if "ci_low" in r and not np.isnan(r.get("ci_low", np.nan)) else "n/a"
            if r.n < MIN_N:                                  # a handful of turns: the numbers mean nothing
                ci = f"n < {MIN_N}, not interpreted"
            gain.append(f"| {r.group} | {r.n} | {pct(r.ref)} | {pct(r.b4)} | {100 * r.delta:+.2f} | {ci} |")
        gain += ["", f"![context gain](figures/context_gain_{d}.png)", ""]

    wt = {}
    for d, s in summaries.items():
        t = window_test(d, s, tag, baseline)
        if t is not None and t.k.nunique() > 1:
            wt[d] = t
            t.round(4).to_csv(reports / d / "window_sweep_test.csv", index=False)
    window_test_figure(wt, reports / "figures" / "window_sweep_test.png")
    wtxt = []
    for d, t in wt.items():
        wtxt += [f"**{d}**, test {M.HEADLINE[d]} (%), mean ± SD over the sweep seeds (k = 0: no context):", "",
                 "| k | Seeds | All | First | Stable | Shift |", "|---|---|---|---|---|---|"]
        for k in t.k.unique():
            v = t[t.k == k].set_index("group")
            wtxt.append(f"| {k} | {int(v.seeds.iloc[0])} | " + " | ".join(
                f"{pct(v.loc[g, 'mean'])} ± {pct(v.loc[g, 'std'])}" if g in v.index else "-" for g in GROUPS) + " |")
        wtxt.append("")
    if wt:
        wtxt += ["![window sweep on test](figures/window_sweep_test.png)", ""]

    strata = []
    for d, s in summaries.items():
        split = next(iter(s["test"]))
        f = reports / d / f"stratified_{split}.csv"
        if f.exists():
            t = pd.read_csv(f).set_index("group")
            keep = [g for g in t.index if g in GROUPS or g.startswith("shift_type:")]
            strata += [f"**{d}**, {split} (mean over seeds, %):", "",
                       "| Group | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral |", "|---|---|---|---|---|"]
            strata += [f"| {g} | {int(t.loc[g, 'n'])} | {pct(t.loc[g, 'weighted_f1'])} | {pct(t.loc[g, 'macro_f1'])} | "
                       f"{pct(t.loc[g, 'micro_f1_no_neutral'])} |" for g in keep] + [""]

    ctl = []
    for d, s in summaries.items():
        f = reports / d / "controls.csv"
        if f.exists():
            t = pd.read_csv(f)
            ctl += [(f"**{d}**, {M.HEADLINE[d]} (%), mean ± SD over {int(t['seeds'].iloc[0])} seeds, k = {s['chosen_k']}:" if "seeds" in t.columns
                     else f"**{d}**, {M.HEADLINE[d]} (%), one model (the saved seed), chosen k = {s['chosen_k']}:"), "",
                    "| Input | All | First | Stable | Shift |", "|---|---|---|---|---|"]
            for name in t.input.unique():
                v = t[t.input == name].set_index("group")
                if "mean" in v.columns:
                    ctl.append(f"| {name} | " + " | ".join(
                        f"{pct(v['mean'].get(g))} ± {pct(v['std'].get(g))}" if g in v.index else "-" for g in GROUPS) + " |")
                else:
                    ctl.append(f"| {name} | " + " | ".join(pct(v.score.get(g)) for g in GROUPS) + " |")
            ctl.append("")
    r3 = []
    plus_dir = ROOT / "reports" / plus_tag
    if "meld" in summaries and (plus_dir / "meld" / "summary.json").exists():
        sp = json.loads((plus_dir / "meld" / "summary.json").read_text())
        split = next(iter(sp["test"]))
        r = b4plus_vs_b3("meld", split, sp["seeds"], plus_tag, baseline)
        if r:
            r3 = [f"B4+ (class-weighted, k = {sp['chosen_k']}) minus B3, MELD {split} macro-F1: **{100 * r['diff']:+.2f}** points, "
                  f"95% CI [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] ({r['seeds']} seeds, paired dialogue bootstrap).",
                  "", "| Model | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral |", "|---|---|---|---|"]
            for label, r_ in [("B4+", sp["test"][split])]:
                r3.append(f"| {label} | {pct(r_['weighted_f1']['mean'])} ± {pct(r_['weighted_f1']['std'])} | "
                          f"{pct(r_['macro_f1']['mean'])} ± {pct(r_['macro_f1']['std'])} | "
                          f"{pct(r_['micro_f1_no_neutral']['mean'])} ± {pct(r_['micro_f1_no_neutral']['std'])} |")
            r3.append("")

    cmp_ = window_compare(summaries, reports, tag, compare) if compare else []

    eff = ["| Dataset | Parameters | Train time per seed | CPU latency per live step (median / p95) | Batch throughput |",
           "|---|---|---|---|---|"]
    for d, s in summaries.items():
        e = s["efficiency"]
        lat, thr = e["latency_one_utterance_cpu"], e["batch_throughput"]
        eff.append(f"| {d} | {e['parameters'] / 1e6:.1f} M | {e['train_seconds']['mean'] / 60:.1f} min | "
                   f"{lat['median_ms']:.1f} / {lat['p95_ms']:.1f} ms | {thr['utterances_per_second']:,.0f} utt/s ({thr['device']}) |")

    c = next(iter(summaries.values()))["config"]
    return (f"# {tag}: context-aware RoBERTa, all datasets\n\n"
            f"Mean ± std over seeds, in %. {c['model_name']}, lr {c['lr']}, batch {c['batch_size']}, max_len {c['max_len']}, "
            "speaker names on, past turns only, history cut on the left.\n\n"
            "## Test results\n\n" + "\n".join(head) + "\n\n"
            "## Window sweep (E3, dev only)\n\n" + "\n".join(sweep) + "\n"
            + ("![window sweep](figures/window_sweep.png)\n\n" if (reports / "figures" / "window_sweep.png").exists() else "")
            + ("## Window sweep on test, by stratum (E3)\n\n" + "\n".join(wtxt) + "\n" if wtxt else "")
            + "## Gain over no context (E2)\n\n" + "\n".join(gain) + "\n"
            "## By stratum and shift type\n\n" + "\n".join(strata) + "\n"
            + ("## E4: input controls\n\n" + "\n".join(ctl) + "\n" if ctl else "")
            + (f"## Chosen window against {compare}\n\nSame model and recipe at the other window "
               f"(headline metric per dataset, %).\n\n" + "\n".join(cmp_) + "\n" if cmp_ else "")
            + ("## R3: B4+ against B3\n\n" + "\n".join(r3) + "\n" if r3 else "")
            + "## Efficiency (E9)\n\n" + "\n".join(eff) + "\n\n"
            "Training times depend on the GPU the run used (T4, L4 or A100); parameters and latency do not.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b4")
    ap.add_argument("--baseline", default="b3", help="prefix of the no-context predictions in predictions/")
    ap.add_argument("--compare-plus-tag", default="b4allplus", help="B4+ tag of the --compare run")
    ap.add_argument("--plus-tag", default="b4plus", help="tag of the class-weighted run (B4+)")
    ap.add_argument("--compare", help="tag of another window run to compare with, e.g. b4all (k = all)")
    ap.add_argument("--datasets", nargs="+", default=["meld", "dailydialog", "emowoz"])
    a = ap.parse_args()
    md = build(a.tag, a.baseline, a.datasets, a.plus_tag, a.compare)
    out = ROOT / "reports" / a.tag / "results_ctx.md"
    out.write_text(md)
    print(f"wrote {out}")
    if a.compare and (ROOT / "reports" / a.compare).exists():
        other = build(a.compare, a.baseline, a.datasets, a.compare_plus_tag).replace("](figures/", f"](../{a.compare}/figures/")
        (ROOT / "reports" / a.compare / "results_ctx.md").write_text(other.replace(f"](../{a.compare}/figures/", "](figures/"))
        overall = ("# B4 overall: chosen window and k = all\n\nPart 1 is B4 at the window chosen on dev (tag "
                   f"`{a.tag}`), part 2 is the same model at k = all (tag `{a.compare}`). Both use the same recipe, seeds and splits.\n\n"
                   "---\n\n## Part 1: chosen window\n\n" + md.split("\n", 1)[1] +
                   "\n---\n\n## Part 2: k = all\n\n" + other.split("\n", 1)[1])
        (ROOT / "reports" / a.tag / "results_overall.md").write_text(overall)
        print(f"wrote {ROOT / 'reports' / a.tag / 'results_overall.md'}")


if __name__ == "__main__":
    main()
