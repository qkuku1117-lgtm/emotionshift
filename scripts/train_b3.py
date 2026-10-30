"""Train and evaluate B3 (RoBERTa, utterance only) on MELD and DailyDialog.

    python scripts/train_b3.py                       # both datasets, 5 seeds (GPU: ~70 min on a T4)
    python scripts/train_b3.py --datasets meld       # one dataset
    python scripts/train_b3.py --quick               # DistilRoBERTa, 1 seed, small subset (CPU smoke test)
    python scripts/train_b3.py --out /content/drive/MyDrive/b3   # write everything to Drive (Colab)

Interrupted? Run the same command again: seeds that finished are reused (reports/b3/<dataset>/runs/),
only the unfinished ones are trained. ``--fresh`` retrains everything.

Writes
    reports/b3/<dataset>/summary.json        config, dev history, test metrics (mean, std, 95% CI), efficiency
    reports/b3/<dataset>/stratified_<split>.csv, per_class_<split>.csv, confusion_<split>.csv
    reports/b3/transfer_meld_to_dailydialog.json   E8: MELD models scored on DailyDialog
    reports/b3/results.md                    one table for the slides
    predictions/b3_<dataset>_<split>_seed<k>.csv           shared prediction format
    predictions/b3meld_dailydialog_<split>_seed<k>.csv     transfer predictions (MELD-trained)
    models/b3_<dataset>/                     best-on-dev seed, used by the demo, B4 and ShiftAware
    models/b3_<dataset>/embeddings_<split>.npy + embeddings_index.csv   (--embeddings) for ShiftAware
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import efficiency as E  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.models.roberta import RobertaConfig, RobertaUtterance  # noqa: E402

PRED_COLS = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion", "pred",
             "stratum", "shift_type", "model", "seed"]
KEYS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]
OUT = ROOT   # --quick writes to outputs_quick/ (git-ignored) so it never overwrites real results


def subset(df: pd.DataFrame, n_dialogues: int) -> pd.DataFrame:
    """First ``n_dialogues`` dialogues of every split (``--quick``)."""
    return df[df.dialogue_id < n_dialogues]


def pred_path(model: str, dataset: str, split: str, seed: int) -> Path:
    return OUT / "predictions" / f"{model.split('_')[0]}_{dataset}_{split}_seed{seed}.csv"


def write_preds(t: pd.DataFrame, pred, model: str, seed: int, dataset: str, split: str) -> pd.DataFrame:
    t = t.copy()
    t["pred"], t["model"], t["seed"] = pred, model, seed
    t[PRED_COLS].to_csv(pred_path(model, dataset, split, seed), index=False)
    return t


def score_runs(runs: list[pd.DataFrame], n_boot: int) -> dict:
    """Mean/std over seeds, dialogue-bootstrap CI on the first seed, derived shift detection."""
    per_seed = [M.scores(t.emotion, t.pred) for t in runs]
    res = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
               "std": float(np.std([p[k] for p in per_seed])),
               "per_seed": [round(p[k], 4) for p in per_seed]} for k in KEYS}
    for k in KEYS[:3]:
        res[k]["ci95_seed0"] = M.bootstrap_ci(runs[0], k, n_boot=n_boot)
    res["n"] = int(len(runs[0]))
    shift = [M.derived_shift_detection(t) for t in runs]
    res["derived_shift_detection"] = {**shift[0], "shift_f1_mean": float(np.mean([s["shift_f1"] for s in shift])),
                                      "shift_f1_std": float(np.std([s["shift_f1"] for s in shift]))}
    return res


def stratified(t: pd.DataFrame) -> pd.DataFrame:
    """``M.stratified``, but tolerant of groups that are empty in a ``--quick`` subset."""
    groups = [("all", t)] + [(f"stratum:{s}", t[t.stratum == s]) for s in ["first", "stable", "shift"]] + \
             [(f"shift_type:{s}", t[t.shift_type == s]) for s in sorted(t.shift_type.dropna().unique())]
    if all(len(g) for _, g in groups) and t.shift_type.nunique() == 5:
        return M.stratified(t)
    return pd.DataFrame([{"group": name, **M.scores(g.emotion, g.pred)} for name, g in groups if len(g)])


def report_split(out: Path, split: str, runs: list[pd.DataFrame]) -> None:
    strat = pd.concat([stratified(t).assign(seed=t.seed.iloc[0]) for t in runs])
    num = [c for c in strat.columns if c not in ("group", "seed")]
    strat.groupby("group", sort=False)[num].mean().round(4).reset_index().to_csv(
        out / f"stratified_{split}.csv", index=False)
    first = runs[0]
    M.per_class_f1(first.emotion, first.pred).round(4).to_csv(out / f"per_class_{split}.csv", index=False)
    M.confusion(first.emotion, first.pred).to_csv(out / f"confusion_{split}.csv")


def run(dataset: str, cfg: dict, quick: bool, embeddings: bool, data: dict, fresh: bool = False) -> dict:
    out = OUT / "reports" / "b3" / dataset
    out.mkdir(parents=True, exist_ok=True)
    df = data[dataset]
    seeds = cfg["seeds"][:2] if quick else cfg["seeds"]
    n_boot = 100 if quick else cfg["bootstrap"]
    mcfg = {**cfg["model"], "epochs": cfg["epochs"][dataset], "select_on": cfg["select_on"][dataset]}
    if quick:
        mcfg.update(model_name="distilroberta-base", epochs=1)
    tr, dev = df[df.split == "train"], df[df.split == "dev"]

    summary = {"dataset": dataset, "model": "B3 RoBERTa (utterance only)", "selected_on_dev": mcfg["select_on"],
               "config": mcfg, "seeds": seeds, "dev": {}, "test": {}}
    preds = {s: [] for s in cfg["test_splits"][dataset]}
    xfer = cfg["transfer"]
    xfer_preds = {s: [] for s in xfer["splits"]} if dataset == xfer["from"] else {}
    best, train_seconds = (None, -1.0), []
    runs_dir = out / "runs"          # one small state file per finished seed: lets an interrupted run resume
    runs_dir.mkdir(exist_ok=True)

    for seed in seeds:
        state = runs_dir / f"seed{seed}.json"
        files = [pred_path("b3_roberta", dataset, s, seed) for s in preds] + \
                [pred_path("b3meld_roberta", xfer["to"], s, seed) for s in xfer_preds]
        if state.exists() and not fresh and all(f.exists() for f in files):
            done = json.loads(state.read_text())
            print(f"[{dataset}] seed {seed}: already finished (dev {done['dev_score']:.4f}), reusing it", flush=True)
            for split in preds:
                preds[split].append(pd.read_csv(pred_path("b3_roberta", dataset, split, seed)))
            for split in xfer_preds:
                xfer_preds[split].append(pd.read_csv(pred_path("b3meld_roberta", xfer["to"], split, seed)))
            summary["dev"][f"seed{seed}"] = {"best": done["dev_score"], "per_epoch": done["history"]}
            train_seconds.append(done["train_seconds"])
            if done["dev_score"] > best[1]:   # its model was saved when it was the best so far
                best = (seed, done["dev_score"])
            continue

        print(f"[{dataset}] seed {seed}: training {mcfg['model_name']} for {mcfg['epochs']} epochs", flush=True)
        t0 = time.perf_counter()
        m = RobertaUtterance(RobertaConfig(**mcfg, seed=seed)).fit(tr.text, tr.emotion, dev.text, dev.emotion)
        train_seconds.append(time.perf_counter() - t0)
        dev_score = max(h[mcfg["select_on"]] for h in m.history)
        summary["dev"][f"seed{seed}"] = {"best": dev_score, "per_epoch": m.history}
        print(f"[{dataset}] seed {seed}: dev {mcfg['select_on']} {dev_score:.4f} "
              f"({train_seconds[-1] / 60:.1f} min)", flush=True)

        for split in preds:
            t = df[df.split == split]
            preds[split].append(write_preds(t, m.predict(t.text), "b3_roberta", seed, dataset, split))
        for split in xfer_preds:
            t = data[xfer["to"]][data[xfer["to"]].split == split]
            xfer_preds[split].append(write_preds(t, m.predict(t.text), "b3meld_roberta", seed, xfer["to"], split))
        h = preds[cfg["test_splits"][dataset][0]][-1]
        print(f"[{dataset}] seed {seed}: test {M.HEADLINE[dataset]} "
              f"{M.scores(h.emotion, h.pred)[M.HEADLINE[dataset]]:.4f}", flush=True)

        if dev_score > best[1]:       # keep the best seed on DEV (never chosen on test)
            best = (seed, dev_score)
            m.save(OUT / "models" / f"b3_{dataset}")
        # Written last, so a seed interrupted half-way is simply trained again on resume.
        state.write_text(json.dumps({"dev_score": dev_score, "history": m.history,
                                     "train_seconds": train_seconds[-1]}, default=float))
        del m

    summary["best_seed_on_dev"] = best[0]
    for split, runs in preds.items():
        summary["test"][split] = score_runs(runs, n_boot)
        report_split(out, split, runs)

    # Efficiency (E9) on the saved model: CPU latency is what the demo sees.
    path = OUT / "models" / f"b3_{dataset}"
    test_texts = df[df.split == cfg["test_splits"][dataset][0]].text
    cpu = RobertaUtterance.load(path, device="cpu")
    gpu = RobertaUtterance.load(path)
    summary["efficiency"] = {
        "parameters": cpu.num_parameters(),
        "model_dir_mb": round(sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1e6, 1),
        "train_seconds": {"mean": float(np.mean(train_seconds)), "per_seed": [round(s, 1) for s in train_seconds]},
        "latency_one_utterance_cpu": E.latency_ms(cpu.predict, test_texts, n=50 if quick else 300),
        "batch_throughput": {**E.throughput(gpu.predict, test_texts), "device": gpu.device},
        "machine": E.machine(),
    }
    if embeddings:
        index = df[["dataset", "split", "dialogue_id", "turn", "speaker", "emotion", "stratum"]]
        for split in ["train", "dev", "test"] + (["test_dedup"] if dataset == "dailydialog" else []):
            np.save(path / f"embeddings_{split}.npy", gpu.embed(df[df.split == split].text).astype(np.float16))
        index.to_csv(path / "embeddings_index.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    if xfer_preds:
        res = {"from": dataset, "to": xfer["to"], "seeds": seeds,
               "test": {split: score_runs(runs, n_boot) for split, runs in xfer_preds.items()}}
        for split, runs in xfer_preds.items():
            report_split(OUT / "reports" / "b3", f"transfer_{xfer['to']}_{split}", runs)
        (OUT / "reports" / "b3" / f"transfer_{dataset}_to_{xfer['to']}.json").write_text(
            json.dumps(res, indent=1, default=float))
    return summary


def results_table(summaries: list[dict]) -> str:
    def f(r, k):
        lo, hi = r[k]["ci95_seed0"]
        return f"{100*r[k]['mean']:.2f} ± {100*r[k]['std']:.2f} [{100*lo:.1f}, {100*hi:.1f}]"

    lines = ["| Dataset | Test split | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |",
             "|---|---|---|---|---|---|---|"]
    for s in summaries:
        for split, r in s["test"].items():
            d = r["derived_shift_detection"]
            lines.append(f"| {s['dataset']} | {split} | {r['n']} | {f(r, 'weighted_f1')} | {f(r, 'macro_f1')} | "
                         f"{f(r, 'micro_f1_no_neutral')} | {100*d['shift_f1_mean']:.2f} ± {100*d['shift_f1_std']:.2f} |")
    xfer_path = OUT / "reports" / "b3" / "transfer_meld_to_dailydialog.json"
    if xfer_path.exists():
        x = json.loads(xfer_path.read_text())
        for split, r in x["test"].items():
            d = r["derived_shift_detection"]
            lines.append(f"| meld → dailydialog (E8) | {split} | {r['n']} | {f(r, 'weighted_f1')} | {f(r, 'macro_f1')} | "
                         f"{f(r, 'micro_f1_no_neutral')} | {100*d['shift_f1_mean']:.2f} ± {100*d['shift_f1_std']:.2f} |")

    eff = ["| Dataset | Parameters | Model files | Train time per seed | CPU latency per utterance (median / p95) | Batch throughput |",
           "|---|---|---|---|---|---|"]
    for s in summaries:
        e = s["efficiency"]
        lat, thr = e["latency_one_utterance_cpu"], e["batch_throughput"]
        eff.append(f"| {s['dataset']} | {e['parameters'] / 1e6:.1f} M | {e['model_dir_mb']:.0f} MB | "
                   f"{e['train_seconds']['mean'] / 60:.1f} min | {lat['median_ms']:.1f} / {lat['p95_ms']:.1f} ms | "
                   f"{thr['utterances_per_second']:,.0f} utt/s ({thr['device']}) |")
    m = summaries[0]["efficiency"]["machine"]
    cfgs = "; ".join(f"{s['dataset']}: {s['config']['epochs']} epochs, best seed on dev = {s['best_seed_on_dev']}"
                     for s in summaries)
    return (f"# B3 RoBERTa (utterance only) results\n\nMean ± std over {len(summaries[0]['seeds'])} seeds, "
            "[95% dialogue-bootstrap CI, first seed], in %. "
            f"{summaries[0]['config']['model_name']}, lr {summaries[0]['config']['lr']}, batch "
            f"{summaries[0]['config']['batch_size']}; {cfgs}.\n\n" + "\n".join(lines) + "\n\n"
            "## Efficiency (E9)\n\n" + "\n".join(eff) + "\n\n"
            f"Latency measured on CPU ({m['cpu_count']} cores, {m['processor']}, Python {m['python']}), "
            "one utterance per call as in the demo; throughput on the training device.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "b3_roberta.yaml"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--embeddings", action="store_true", help="cache <s> vectors for ShiftAware (#6)")
    ap.add_argument("--out", help="output root (default: the repo; Colab: a Google Drive folder)")
    ap.add_argument("--fresh", action="store_true", help="retrain every seed instead of resuming finished ones")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    elif a.quick:
        OUT = ROOT / "outputs_quick"
    cfg = yaml.safe_load(open(a.config))
    for d in ["predictions", "models", "reports/b3"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    datasets = a.datasets or cfg["datasets"]
    need = set(datasets) | ({cfg["transfer"]["to"]} if cfg["transfer"]["from"] in datasets else set())
    data = {d: load(d) for d in need}
    if a.quick:
        data = {d: subset(df, 12) for d, df in data.items()}
    summaries = [run(d, cfg, a.quick, a.embeddings, data, a.fresh) for d in datasets]
    md = results_table(summaries)
    (OUT / "reports" / "b3" / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
