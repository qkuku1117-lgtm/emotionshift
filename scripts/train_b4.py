"""Train and evaluate B4 (context-aware RoBERTa) on MELD, DailyDialog and EmoWOZ.

    python scripts/train_b4.py                          # all datasets (sweep k on MELD, reuse it elsewhere)
    python scripts/train_b4.py --datasets meld          # one dataset (do MELD first: the others reuse its k)
    python scripts/train_b4.py --datasets meld --k 5    # skip the sweep, fixed window
    python scripts/train_b4.py --quick                  # DistilRoBERTa, tiny subset: a CPU smoke test
    python scripts/train_b4.py --out /content/drive/MyDrive/b4   # write everything to Drive (Colab)
    python scripts/train_b4.py --class-weight balanced --tag b4plus   # B4+ (class-weighted loss)

Protocol (configs/b4_roberta_ctx.yaml)
    1. E3 sweep: every k in the grid is trained with ``sweep_seeds`` and scored on DEV only.
    2. The k with the best mean dev score is chosen (ties -> smaller k). Test is never used to choose.
    3. The chosen k is completed to ``seeds`` (the sweep seeds are reused), then scored on test.
    4. ``controls`` (EmoWOZ: k = 0, i.e. B3's input) are trained for all seeds and reported beside it.

Interrupted? Run the same command again: every finished (k, seed) is reused from
reports/<tag>/<dataset>/runs/. ``--fresh`` retrains everything.

Writes (<tag> = b4 by default; names match scripts/summarize_all.py)
    reports/<tag>/<dataset>/summary.json    chosen k, sweep, dev history, test metrics (mean, std, 95% CI), efficiency
    reports/<tag>/<dataset>/sweep.csv       dev score of every (k, seed) run: the E3 figure's data
    reports/<tag>/<dataset>/stratified_<split>.csv, per_class_<split>.csv, confusion_<split>.csv
    reports/<tag>/results.md                one table for the slides
    predictions/<tag>_<dataset>_<split>_seed<k>.csv     shared prediction format (labelled turns only)
    predictions/<tag>k0_<dataset>_<split>_seed<k>.csv   the k = 0 control, where run
    models/<tag>_<dataset>/                 best-on-dev seed at the chosen k (used by the demo)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.context import k_label, parse_k  # noqa: E402
from emotionshift.data.labels import emotions  # noqa: E402
from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import efficiency as E  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.eval.controls import aggregate, score_controls  # noqa: E402

PRED_COLS = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion", "pred",
             "stratum", "shift_type", "model", "seed"]
KEYS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "weighted_f1_no_neutral", "accuracy"]
OUT = ROOT   # --quick writes to outputs_quick/ (git-ignored) so it never overwrites real results


# ---- model access (imported lazily so the plumbing can be tested without PyTorch) ----------------
def make_model(mcfg: dict, dataset: str, k, seed: int):
    from emotionshift.models.roberta_ctx import CtxConfig, RobertaContext
    return RobertaContext(CtxConfig(**mcfg, dataset=dataset, k=k, seed=seed))


def load_model(path, device=None):
    from emotionshift.models.roberta_ctx import RobertaContext
    return RobertaContext.load(path, device=device)


# ---- small helpers ----------------------------------------------------------------------------
def kkey(k) -> float:
    return float("inf") if k is None else k


def pred_path(prefix: str, dataset: str, split: str, seed: int) -> Path:
    return OUT / "predictions" / f"{prefix}_{dataset}_{split}_seed{seed}.csv"


def subset(df: pd.DataFrame, n_dialogues: int) -> pd.DataFrame:
    """First ``n_dialogues`` dialogues of every split (``--quick``)."""
    keep = df.drop_duplicates(["split", "dialogue_id"]).groupby("split").head(n_dialogues)
    return df.merge(keep[["split", "dialogue_id"]], on=["split", "dialogue_id"]).reset_index(drop=True)


def labelled(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[(df.split == split) & df.emotion.notna()]


def score_runs(runs: list[pd.DataFrame], n_boot: int, labels: list[str]) -> dict:
    """Mean/std over seeds, dialogue-bootstrap CI on the first seed, derived shift detection."""
    per_seed = [M.scores(t.emotion, t.pred, labels) for t in runs]
    res = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
               "std": float(np.std([p[k] for p in per_seed])),
               "per_seed": [round(p[k], 4) for p in per_seed]} for k in KEYS}
    for k in KEYS[:4]:
        res[k]["ci95_seed0"] = M.bootstrap_ci(runs[0], k, n_boot=n_boot, labels=labels)
    res["n"] = int(len(runs[0]))
    shift = [M.derived_shift_detection(t) for t in runs]
    res["derived_shift_detection"] = {**shift[0], "shift_f1_mean": float(np.mean([s["shift_f1"] for s in shift])),
                                      "shift_f1_std": float(np.std([s["shift_f1"] for s in shift]))}
    return res


def stratified(t: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Scores overall, per stratum and per shift type; groups empty in a ``--quick`` subset are skipped."""
    groups = [("all", t)] + [(f"stratum:{s}", t[t.stratum == s]) for s in ["first", "stable", "shift"]] + \
             [(f"shift_type:{s}", t[t.shift_type == s]) for s in sorted(t.shift_type.dropna().unique())]
    return pd.DataFrame([{"group": n, **M.scores(g.emotion, g.pred, labels)} for n, g in groups if len(g)])


def report_split(out: Path, split: str, runs: list[pd.DataFrame], labels: list[str]) -> None:
    strat = pd.concat([stratified(t, labels).assign(seed=t.seed.iloc[0]) for t in runs])
    num = [c for c in strat.columns if c not in ("group", "seed")]
    strat.groupby("group", sort=False)[num].mean().round(4).reset_index().to_csv(
        out / f"stratified_{split}.csv", index=False)
    first = runs[0]
    M.per_class_f1(first.emotion, first.pred, labels).round(4).to_csv(out / f"per_class_{split}.csv", index=False)
    M.confusion(first.emotion, first.pred, labels).to_csv(out / f"confusion_{split}.csv")


# ---- one (k, seed) run ------------------------------------------------------------------------
def run_one(env: dict, k, seed: int) -> dict:
    """Train one model, write its test predictions to the runs folder, keep the best model of this k."""
    runs, tag, ds = env["runs_dir"], env["tag"], env["dataset"]
    name = f"k{k_label(k)}_seed{seed}"
    state = runs / f"{name}.json"
    files = [runs / f"{name}_{s}.csv" for s in env["splits"]]
    if env.get("controls"):
        files.append(runs / f"{name}_controls.csv")
    if state.exists() and (not env["fresh"] or (k, seed) in env["trained"]) and all(f.exists() for f in files):
        done = json.loads(state.read_text())
        print(f"[{ds}] k={k_label(k)} seed {seed}: already finished (dev {done['dev_score']:.4f}), reusing it", flush=True)
        return done

    df, mcfg = env["df"], env["mcfg"]
    print(f"[{ds}] k={k_label(k)} seed {seed}: training {mcfg['model_name']} for {mcfg['epochs']} epochs", flush=True)
    t0 = time.perf_counter()
    m = make_model(mcfg, ds, k, seed).fit_df(df[df.split == "train"], df[df.split == "dev"])
    seconds = time.perf_counter() - t0
    dev_score = max(h[mcfg["select_on"]] for h in m.history)
    print(f"[{ds}] k={k_label(k)} seed {seed}: dev {mcfg['select_on']} {dev_score:.4f} ({seconds / 60:.1f} min)", flush=True)

    for split in env["splits"]:
        full = df[df.split == split]                      # whole split: EmoWOZ system turns are context
        t = labelled(df, split).copy()
        t["pred"] = m.predict_df(full).reindex(t.index).to_numpy()
        t["model"], t["seed"] = f"{tag}_roberta_ctx", seed
        t[PRED_COLS].to_csv(runs / f"{name}_{split}.csv", index=False)

    if env.get("controls"):                              # E4 on this seed, while its model is in memory
        split0 = env["splits"][0]
        ctl = score_controls(m, df[df.split == split0], k, emotions(ds), M.HEADLINE[ds], seed)
        ctl.to_csv(runs / f"{name}_controls.csv", index=False)

    best_file = runs / f"k{k_label(k)}_best.json"         # best seed on DEV for this k keeps its model
    if not best_file.exists() or dev_score > json.loads(best_file.read_text())["dev_score"]:
        m.save(env["sweep_models"] / f"{tag}_{ds}_k{k_label(k)}")
        best_file.write_text(json.dumps({"seed": seed, "dev_score": dev_score}))
    # Written last, so a run interrupted half-way is simply trained again on resume.
    env["trained"].add((k, seed))
    done = {"k": k_label(k), "seed": seed, "dev_score": dev_score, "history": m.history, "train_seconds": seconds}
    state.write_text(json.dumps(done, default=float))
    return done


def read_runs(env: dict, k, seeds: list[int], split: str) -> list[pd.DataFrame]:
    return [pd.read_csv(env["runs_dir"] / f"k{k_label(k)}_seed{s}_{split}.csv") for s in seeds]


def export(env: dict, k, seeds: list[int], prefix: str) -> dict[str, list[pd.DataFrame]]:
    """Copy the runs of ``k`` into predictions/ (the group-wide format) and return them per split."""
    out = {}
    for split in env["splits"]:
        runs = read_runs(env, k, seeds, split)
        for s, t in zip(seeds, runs):
            t["model"] = f"{prefix}_roberta_ctx"
            t[PRED_COLS].to_csv(pred_path(prefix, env["dataset"], split, s), index=False)
        out[split] = runs
    return out


# ---- one dataset ------------------------------------------------------------------------------
def choose_k(env: dict, cfg: dict, a, finished: dict) -> tuple:
    """-> (k, how, sweep_rows). Sweep on dev, or reuse another dataset's k, or take ``--k``."""
    ds, sw = env["dataset"], cfg["sweep"][env["dataset"]]
    if a.k is not None:
        return parse_k(a.k), "--k", []
    if "k_from" in sw and not a.grid:
        src = sw["k_from"]
        if src in finished:
            return finished[src]["chosen_k"], f"k_from:{src}", []
        path = OUT / "reports" / env["tag"] / src / "summary.json"
        if not path.exists():
            raise SystemExit(f"{ds} reuses the window chosen on {src}: run '--datasets {src}' first, "
                             f"or pass --k / --grid.")
        return parse_k(json.loads(path.read_text())["chosen_k"]), f"k_from:{src}", []
    grid = [parse_k(x) for x in (a.grid or sw["grid"])]
    if a.quick:
        grid = grid[:2]
    rows = []
    for k in grid:
        for seed in env["sweep_seeds"]:
            r = run_one(env, k, seed)
            rows.append({"k": k_label(k), "seed": seed, "dev_score": r["dev_score"],
                         "train_seconds": r["train_seconds"]})
    table = pd.DataFrame(rows)
    means = {k: table[table.k == k_label(k)].dev_score.mean() for k in grid}
    best = min(grid, key=lambda k: (-round(means[k], 10), kkey(k)))
    print(f"[{ds}] dev {env['mcfg']['select_on']} by k: "
          + ", ".join(f"{k_label(k)}={means[k]:.4f}" for k in grid) + f"  ->  k={k_label(best)}", flush=True)
    return best, "sweep", rows


def baseline(env: dict, cfg: dict, headline: str, split: str, control: dict | None, labels) -> dict | None:
    """Headline score of the reference without context: B3's files if present, else our k = 0 control."""
    ds, prefix = env["dataset"], cfg["baseline_prefix"]
    files = [pred_path(prefix, ds, split, s) for s in env["seeds"]]
    if all(f.exists() for f in files):
        runs, source = [pd.read_csv(f) for f in files], prefix
    elif control and split in control:
        runs, source = control[split], f"{env['tag']}k0"
    else:
        return None
    return {"source": source, "split": split, "metric": headline,
            "mean": float(np.mean([M.scores(t.emotion, t.pred, labels)[headline] for t in runs]))}


def run_dataset(ds: str, cfg: dict, a, data: dict, finished: dict) -> dict:
    out = OUT / "reports" / a.tag / ds
    runs_dir = out / "runs"
    for d in [runs_dir, OUT / "predictions", OUT / "models" / "_sweep"]:
        d.mkdir(parents=True, exist_ok=True)
    labels = emotions(ds)
    seeds = cfg["seeds"][:2] if a.quick else cfg["seeds"]
    sweep_seeds = cfg["sweep_seeds"][:1] if a.quick else cfg["sweep_seeds"]
    n_boot = 100 if a.quick else cfg["bootstrap"]
    mcfg = {**cfg["model"], "epochs": cfg["epochs"][ds], "select_on": cfg["select_on"][ds]}
    if a.class_weight:
        mcfg["class_weight"] = a.class_weight
    if a.model_name:
        mcfg["model_name"] = a.model_name
    if a.quick:
        mcfg.update(model_name="distilroberta-base", epochs=1)
    env = {"dataset": ds, "df": data[ds], "mcfg": mcfg, "runs_dir": runs_dir, "tag": a.tag, "fresh": a.fresh, "trained": set(),
           "splits": cfg["test_splits"][ds], "seeds": seeds, "sweep_seeds": sweep_seeds,
           "sweep_models": OUT / "models" / "_sweep", "controls": False}
    headline = M.HEADLINE[ds]

    chosen, how, sweep_rows = choose_k(env, cfg, a, finished)
    if a.fresh and a.k is not None:                      # final seeds retrained: forget the old best-model marker of this k
        (runs_dir / f"k{k_label(chosen)}_best.json").unlink(missing_ok=True)
    sweep_seeds_used = sweep_seeds

    env["controls"] = bool(a.controls)                     # only the chosen window is scored (the sweep ran above)
    states = {s: run_one(env, chosen, s) for s in seeds}             # completes the sweep seeds to the full set
    exported = export(env, chosen, seeds, a.tag)
    env["controls"] = False                                 # E4 is for the chosen window only, not the k = 0 control

    controls = {}
    for k0 in cfg["sweep"][ds].get("controls", []):
        if k0 == chosen or a.k is not None:
            continue
        cstates = {s: run_one(env, k0, s) for s in seeds}
        controls[k0] = {"states": cstates, "runs": export(env, k0, seeds, f"{a.tag}k{k_label(k0)}")}
        sweep_rows += [{"k": k_label(k0), "seed": s, "dev_score": r["dev_score"], "train_seconds": r["train_seconds"]}
                       for s, r in cstates.items()]
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(out / "sweep.csv", index=False)

    summary = {"dataset": ds, "model": f"{a.tag} RoBERTa with context", "label_space": labels,
               "selected_on_dev": mcfg["select_on"], "chosen_k": k_label(chosen), "chosen_by": how,
               "config": mcfg, "seeds": seeds, "sweep_seeds": sweep_seeds_used, "dev": {}, "test": {}}
    if sweep_rows:
        tbl = pd.DataFrame(sweep_rows)
        summary["sweep"] = [{"k": k, "n_seeds": len(g), "dev_mean": float(g.dev_score.mean()),
                             "dev_std": float(g.dev_score.std(ddof=0)),
                             "per_seed": [round(v, 4) for v in g.dev_score]}
                            for k, g in tbl.groupby("k", sort=False)]
    prev_f = out / "summary.json"
    if a.k is not None and not sweep_rows and prev_f.exists():       # --k retrain: keep what the sweep run recorded
        old = json.loads(prev_f.read_text())
        if old.get("chosen_k") == k_label(chosen):
            for key in ["sweep", "sweep_seeds", "controls"]:
                if key in old:
                    summary[key] = old[key]
            note = " (final seeds retrained with --k)"
            summary["chosen_by"] = old.get("chosen_by", "--k").replace(note, "") + note
            summary["_old_baseline"] = old.get("vs_baseline")
    for s, r in states.items():
        summary["dev"][f"seed{s}"] = {"best": r["dev_score"], "per_epoch": r["history"]}
    best_seed = max(states, key=lambda s: states[s]["dev_score"])
    summary["best_seed_on_dev"] = best_seed

    for split, runs in exported.items():
        summary["test"][split] = score_runs(runs, n_boot, labels)
        report_split(out, split, runs, labels)
    if controls:
        summary["controls"] = {f"k{k_label(k)}": {sp: score_runs(r, n_boot, labels) for sp, r in c["runs"].items()}
                               for k, c in controls.items()}
    if a.controls:
        e4 = aggregate([pd.read_csv(runs_dir / f"k{k_label(chosen)}_seed{s}_controls.csv") for s in seeds])
        e4.round(4).to_csv(out / "controls.csv", index=False)
        summary["e4"] = {"split": env["splits"][0], "metric": headline, "seeds": seeds,
                         "rows": e4.round(4).to_dict("records")}
    split0 = env["splits"][0]
    ctl0 = controls[0]["runs"] if 0 in controls else None
    base = baseline(env, cfg, headline, split0, ctl0, labels)
    old_base = summary.pop("_old_baseline", None)
    if base is None and old_base:                         # --k retrain: the k = 0 reference from the sweep run
        base = {k: old_base[k] for k in ("source", "split", "metric", "mean")}
    if base:
        got = summary["test"][split0][headline]["mean"]
        summary["vs_baseline"] = {**base, "b4_mean": got, "delta": got - base["mean"]}

    # keep only the chosen k's best model, as models/<tag>_<dataset> (what the demo loads)
    final = OUT / "models" / f"{a.tag}_{ds}"
    sweep_dirs = OUT / "models" / "_sweep"
    chosen_dir = sweep_dirs / f"{a.tag}_{ds}_k{k_label(chosen)}"
    if chosen_dir.exists():
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(chosen_dir), str(final))
    elif not final.exists():
        raise SystemExit(f"[{ds}] no saved model for k={k_label(chosen)}; rerun with --fresh")
    for d in sweep_dirs.glob(f"{a.tag}_{ds}_k*"):
        shutil.rmtree(d, ignore_errors=True)
    for f in runs_dir.glob("k*_best.json"):          # model dirs are gone: a later resume must not trust these
        if f.name != f"k{k_label(chosen)}_best.json":
            f.unlink()

    try:
        summary["efficiency"] = measure_efficiency(final, env, chosen, states, a.quick)
    except (OSError, ValueError) as e:                       # saved model missing or incomplete (e.g. a Drive copy that did not finish)
        prev = out / "summary.json"
        old = json.loads(prev.read_text()).get("efficiency") if prev.exists() else None
        print(f"[{ds}] warning: efficiency not measured ({type(e).__name__}: {str(e)[:120]}); "
              f"{'kept the earlier measurement' if old else 'left out'}. Retrain with --fresh to save the model again.")
        if old:
            summary["efficiency"] = old
        else:
            summary["efficiency"] = {"train_seconds": {"mean": float(np.mean([st["train_seconds"] for st in states.values()]))},
                                     "note": "model files not found, efficiency not measured"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def measure_efficiency(path: Path, env: dict, k, states: dict, quick: bool) -> dict:
    """E9 on the saved model. Latency = one live demo step: the newest turn from its last k + 1 turns, on CPU."""
    df, split = env["df"], env["splits"][0]
    test = df[df.split == split]
    rows = labelled(df, split)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(50 if quick else 300, len(rows)), replace=False)
    prefixes = []
    for i in pick:
        r = rows.iloc[i]
        g = test[(test.dialogue_id == r.dialogue_id) & (test.turn <= r.turn)].sort_values("turn")
        prefixes.append((list(g.text), list(g.speaker)))
    cpu, gpu = load_model(path, device="cpu"), load_model(path)
    return {
        "parameters": cpu.num_parameters(),
        "model_dir_mb": round(sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1e6, 1),
        "train_seconds": {"mean": float(np.mean([s["train_seconds"] for s in states.values()])),
                          "per_seed": [round(s["train_seconds"], 1) for s in states.values()]},
        "latency_one_utterance_cpu": E.latency_ms(lambda b: cpu.predict_last(*b[0]), prefixes, n=len(prefixes)),
        "batch_throughput": {**E.throughput(lambda _: gpu.predict_df(test), list(range(len(rows)))),
                             "device": gpu.device},
        "machine": E.machine(),
    }


def results_table(summaries: list[dict], tag: str) -> str:
    def f(r, k):
        lo, hi = r[k]["ci95_seed0"]
        return f"{100 * r[k]['mean']:.2f} ± {100 * r[k]['std']:.2f} [{100 * lo:.1f}, {100 * hi:.1f}]"

    lines = ["| Dataset | Test split | k | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Weighted-F1 excl. neutral | Derived shift F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for split, r in s["test"].items():
            d = r["derived_shift_detection"]
            lines.append(f"| {s['dataset']} | {split} | {s['chosen_k']} | {r['n']} | {f(r, 'weighted_f1')} | "
                         f"{f(r, 'macro_f1')} | {f(r, 'micro_f1_no_neutral')} | {f(r, 'weighted_f1_no_neutral')} | "
                         f"{100 * d['shift_f1_mean']:.2f} ± {100 * d['shift_f1_std']:.2f} |")
        for name, c in s.get("controls", {}).items():
            for split, r in c.items():
                d = r["derived_shift_detection"]
                lines.append(f"| {s['dataset']} (control {name}) | {split} | {name[1:]} | {r['n']} | {f(r, 'weighted_f1')} | "
                             f"{f(r, 'macro_f1')} | {f(r, 'micro_f1_no_neutral')} | {f(r, 'weighted_f1_no_neutral')} | "
                             f"{100 * d['shift_f1_mean']:.2f} ± {100 * d['shift_f1_std']:.2f} |")
    sweeps = []
    for s in summaries:
        if "sweep" in s:
            sweeps.append(f"**{s['dataset']}** (dev {s['selected_on_dev']}, mean over {s['sweep_seeds']} seeds): "
                          + ", ".join(f"k={x['k']}: {100 * x['dev_mean']:.2f}" for x in s["sweep"]))
        else:
            sweeps.append(f"**{s['dataset']}**: window taken from {s['chosen_by']}")
    deltas = [f"{s['dataset']}: {100 * s['vs_baseline']['delta']:+.2f} {s['vs_baseline']['metric']} vs "
              f"{s['vs_baseline']['source']}" for s in summaries if "vs_baseline" in s]
    eff = ["| Dataset | Parameters | Model files | Train time per seed | CPU latency per live step (median / p95) | Batch throughput |",
           "|---|---|---|---|---|---|"]
    for s in summaries:
        e = s["efficiency"]
        if "latency_one_utterance_cpu" not in e:             # model files were missing: nothing was measured
            eff.append(f"| {s['dataset']} | not measured (model files not found) | | "
                       f"{e['train_seconds']['mean'] / 60:.1f} min | | |")
            continue
        lat, thr = e["latency_one_utterance_cpu"], e["batch_throughput"]
        eff.append(f"| {s['dataset']} | {e['parameters'] / 1e6:.1f} M | {e['model_dir_mb']:.0f} MB | "
                   f"{e['train_seconds']['mean'] / 60:.1f} min | {lat['median_ms']:.1f} / {lat['p95_ms']:.1f} ms | "
                   f"{thr['utterances_per_second']:,.0f} utt/s ({thr['device']}) |")
    m = next((x["efficiency"]["machine"] for x in summaries if "machine" in x["efficiency"]), {})
    c = summaries[0]["config"]
    return (f"# {tag}: context-aware RoBERTa results\n\nMean ± std over {len(summaries[0]['seeds'])} seeds, "
            f"[95% dialogue-bootstrap CI, first seed], in %. {c['model_name']}, lr {c['lr']}, batch {c['batch_size']}, "
            f"max_len {c['max_len']}, speaker names on, history cut on the left.\n\n" + "\n".join(lines) + "\n\n"
            "## Window sweep (E3, dev only)\n\n" + "\n\n".join(sweeps) + "\n\n"
            + ("## Gain over no context\n\n" + "\n".join(f"- {d}" for d in deltas) + "\n\n" if deltas else "")
            + "## Efficiency (E9)\n\n" + "\n".join(eff) + "\n\n"
            f"Latency = the newest turn of a live dialogue from its last k + 1 turns, one call at a time, on CPU "
            f"({m.get('cpu_count', '?')} cores, {m.get('processor', '?')}, Python {m.get('python', '?')}); throughput on the training device.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "b4_roberta_ctx.yaml"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--k", help="fixed window (an int or 'all'); skips the sweep")
    ap.add_argument("--grid", nargs="*", help="override the sweep grid, e.g. --grid 1 3 all")
    ap.add_argument("--tag", default="b4", help="name of reports/, predictions/ and models/ entries (B4+: b4plus)")
    ap.add_argument("--class-weight", choices=["balanced"], help="class-weighted loss (B4+)")
    ap.add_argument("--model-name", help="e.g. distilroberta-base (ablation)")
    ap.add_argument("--controls", action="store_true",
                    help="E4: score the no / shuffled-context and shuffled-speaker controls on every seed of the chosen k, "
                         "right after it trains (use with --k; adds about a minute per seed)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", help="output root (default: the repo; Colab: a Google Drive folder)")
    ap.add_argument("--fresh", action="store_true", help="retrain every run instead of resuming finished ones")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    elif a.quick:
        OUT = ROOT / "outputs_quick"
    cfg = yaml.safe_load(open(a.config))
    for d in ["predictions", "models/_sweep", f"reports/{a.tag}"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    datasets = a.datasets or cfg["datasets"]
    data = {d: load(d) for d in datasets}
    if a.quick:
        data = {d: subset(df, 12) for d, df in data.items()}
    finished, summaries = {}, []
    for d in datasets:
        s = run_dataset(d, cfg, a, data, finished)
        finished[d] = {"chosen_k": parse_k(s["chosen_k"])}
        summaries.append(s)
    md = results_table(summaries, a.tag)
    (OUT / "reports" / a.tag / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
