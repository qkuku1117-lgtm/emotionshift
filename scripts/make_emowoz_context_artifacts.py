"""Audit the native EmoWOZ RoBERTa results and execute an English walkthrough."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from nbclient import NotebookClient

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import load
from emotionshift.eval import efficiency as E
from emotionshift.eval import metrics as M
from emotionshift.models.roberta_context import RobertaContext
from scripts.analyse_emowoz_context import analyse
from scripts.train_b0 import PRED_COLS
from scripts.train_emowoz_baselines import digest, write
from scripts.train_emowoz_context import prepare_context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b4/emowoz"
LABELS = emotions("emowoz")


def checkpoint_digests():
    result = {}
    for seed in range(5):
        path = ROOT / f"models/b4_emowoz/seed{seed}/model.safetensors"
        with path.open("rb") as stream:
            result[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    torch.set_num_threads(4)
    manifest = json.loads((OUT / "manifest.json").read_text())
    for name, sha in manifest["source_sha256_lf"].items():
        assert digest(ROOT / name) == sha, name
    for name, sha in manifest["raw_sha256"].items():
        assert hashlib.sha256((ROOT / "data/raw/emowoz" / name).read_bytes()).hexdigest() == sha, (
            name
        )
    stored = pd.read_csv(OUT / "per_seed.csv").set_index("seed")
    target, model_inputs = prepare_context(
        load("emowoz", splits=["test"]), manifest["config"]["window"]
    )
    hashes = {}
    histories = []
    for seed in range(5):
        path = ROOT / f"predictions/b4_emowoz_test_seed{seed}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        assert frame.columns[:12].tolist() == PRED_COLS
        assert len(frame) == 8634 and frame.pred.isin(LABELS).all()
        assert not frame.duplicated(["dataset", "split", "dialogue_id", "turn"]).any()
        for col in [
            "dataset",
            "split",
            "dialogue_id",
            "turn",
            "speaker",
            "text",
            "emotion",
            "stratum",
            "shift_type",
        ]:
            assert frame[col].astype(str).tolist() == target[col].fillna("").astype(str).tolist(), (
                col
            )
        assert frame.model.eq("b4").all() and frame.seed.eq(seed).all()
        for metric, value in M.scores(frame.emotion, frame.pred, LABELS).items():
            if metric != "n":
                assert np.isclose(value, stored.loc[seed, metric])
        assert np.isclose(
            M.derived_shift_detection(frame)["shift_f1"], stored.loc[seed, "derived_shift_f1"]
        )
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        hist = pd.read_csv(OUT / f"history_seed{seed}.csv")
        rec = json.loads((OUT / f"training_seed{seed}.json").read_text())
        assert np.isclose(rec["dev_score"], hist.weighted_f1_no_neutral.max())
        histories.append(hist.assign(seed=seed))
    selected = json.loads((OUT / "selection.json").read_text())
    records = [json.loads((OUT / f"training_seed{s}.json").read_text()) for s in range(5)]
    assert selected["best_seed_on_dev"] == max(records, key=lambda r: r["dev_score"])["seed"]
    assert not selected["test_used_for_selection"]
    selected.update(
        window=manifest["config"]["window"],
        window_selection_source=manifest["config"]["window_selection_file"],
        max_len=manifest["config"]["model"]["max_len"],
        system_history_retained=True,
    )
    write(OUT / "selection.json", selected)
    model = RobertaContext.load(ROOT / "models/b4_emowoz/seed0", device="cpu")
    assert model.labels == LABELS
    assert model.k == manifest["config"]["window"] and model.tokenizer.truncation_side == "left"
    assert any("system:" in text for text in model_inputs)
    probs = model.predict_proba(model_inputs[:16], batch_size=2)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
    expected = pd.read_csv(ROOT / "predictions/b4_emowoz_test_seed0.csv").pred.iloc[:16].tolist()
    assert [LABELS[i] for i in probs.argmax(axis=1)] == expected
    analyse()
    write(
        OUT / "verification.json",
        {
            "prediction_csvs": 5,
            "targets_per_csv": 8634,
            "native_labels_verified": True,
            "dev_selection_verified": True,
            "metrics_recomputed": True,
            "checkpoint_round_trip_verified": True,
            "system_context_retained_user_targets_only": True,
            "e4_controls_verified": True,
            "paired_comparisons_verified": True,
            "prediction_sha256": hashes,
            "checkpoint_sha256": checkpoint_digests(),
        },
    )
    summary = json.loads((OUT / "summary.json").read_text())
    summary["efficiency"] = {
        "parameters": model.num_parameters(),
        "seed0_model_dir_mb": sum(
            p.stat().st_size for p in (ROOT / "models/b4_emowoz/seed0").rglob("*") if p.is_file()
        )
        / 1e6,
        "inference_device": "cpu",
        "machine": E.machine(),
        "latency_one_utterance": E.latency_ms(model.predict, model_inputs, n=50),
        "batch_throughput": E.throughput(model.predict, model_inputs[:256]),
        "timing_note": "Loaded seed-0 checkpoint after all training. Latency uses 50 sampled preassembled contexts; batch throughput uses the first 256 contexts. Context assembly is excluded. Resumed training time is invocation time, not total time.",
    }
    write(OUT / "summary.json", summary)
    result = summary["test"]["metrics"]
    for metric in M.METRICS:
        assert np.isclose(result[metric]["mean"], stored[metric].mean())
    comparison = pd.read_csv(ROOT / "reports/b3/emowoz/comparison_b1_b2_b3.csv")
    h = result["weighted_f1_no_neutral"]
    row = {
        "model": "b4_context",
        "n": 8634,
        "headline": h["mean"],
        "std": h["std"],
        "ci_low": h["ci95_seed0"][0],
        "ci_high": h["ci95_seed0"][1],
        "macro_f1_no_neutral": result["macro_f1_no_neutral"]["mean"],
        "macro_f1": result["macro_f1"]["mean"],
        "derived_shift_f1": summary["test"]["derived_shift_f1_mean"],
    }
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(OUT / "comparison_b1_b2_b3_b4.csv", index=False)
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(comparison.model, comparison.headline * 100, color="#466b8e")
    for i, r in enumerate(comparison.itertuples(index=False)):
        ax.plot([r.ci_low * 100, r.ci_high * 100], [i, i], color="black")
        ax.text(r.headline * 100 + 1, i, f"{r.headline * 100:.2f}", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Native EmoWOZ weighted-F1 excluding neutral (%)")
    ax.set_title("Five-seed means; dialogue-bootstrap intervals for seed 0")
    fig.savefig(figdir / "headline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    comparison.to_csv(figdir / "headline_comparison.csv", index=False)
    hist = pd.concat(histories, ignore_index=True)
    hist.to_csv(figdir / "learning_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for seed, part in hist.groupby("seed"):
        ax.plot(part.epoch, part.weighted_f1_no_neutral * 100, marker="o", label=f"seed {seed}")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev weighted-F1 excluding neutral (%)")
    ax.set_title("Native EmoWOZ context RoBERTa: dev-only checkpoint selection")
    ax.legend()
    fig.savefig(figdir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    sources = {
        "b1": ROOT / "reports/b1/emowoz/stratified_test.csv",
        "b2_bigru": ROOT / "reports/b2/emowoz/b2_bigru/stratified_test.csv",
        "b2_bilstm": ROOT / "reports/b2/emowoz/b2_bilstm/stratified_test.csv",
        "b3_roberta": ROOT / "reports/b3/emowoz/stratified_test.csv",
        "b4_context": OUT / "stratified_test.csv",
    }
    groups = ["stratum:first", "stratum:stable", "stratum:shift"]
    strata = pd.concat(
        [pd.read_csv(path).assign(model=name) for name, path in sources.items()], ignore_index=True
    )
    strata = strata.loc[strata.group.isin(groups), ["model", "group", "weighted_f1_no_neutral"]]
    strata.to_csv(figdir / "strata.csv", index=False)
    pivot = (
        strata.pivot(index="model", columns="group", values="weighted_f1_no_neutral")[groups] * 100
    )
    ax = pivot.plot.bar(figsize=(10, 5), ylim=(0, 100), rot=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_ylabel("Native EmoWOZ weighted-F1 excluding neutral (%)")
    ax.set_title("Mean subset scores across five seeds; label mixtures differ")
    ax.figure.tight_layout()
    ax.figure.savefig(figdir / "strata.png", dpi=180, bbox_inches="tight")
    plt.close(ax.figure)
    report = [
        "# Native EmoWOZ RoBERTa: workstream 5",
        "",
        "Five seeds, official splits, native labels, past user/system context; labelled user targets only.",
        "",
        "| Model | Non-neutral weighted-F1 | Population SD | Macro-F1 excl. neutral |",
        "|---|---:|---:|---:|",
    ]
    for r in comparison.itertuples(index=False):
        report.append(
            f"| {r.model} | {r.headline * 100:.2f} | {r.std * 100:.2f} | {r.macro_f1_no_neutral * 100:.2f} |"
        )
    report.extend(
        [
            "",
            f"Demo seed selected on dev: {selected['best_seed_on_dev']}. Test is not used for selection.",
            "",
            "The comparison reuses the audited SVM/BiGRU/BiLSTM and single-utterance RoBERTa results without overwriting them. All models use the same native EmoWOZ test targets and headline metric. Single-model bootstrap intervals do not establish pairwise significance. Native scores are not directly interchangeable with the MELD/DailyDialog metrics.",
            "",
            "## Context comparisons",
            "",
            "See paired_comparisons.csv for B4 minus B3 on all/first/stable/shift targets and real-history minus each E4 control. Intervals resample shared whole dialogues and average differences across the fixed five seed pairs; they do not resample training seeds.",
            "B4 versus B3 also changes speaker tags and token budget. E4 controls perturb inputs of the same context-trained model and are not retrained baselines. No context-gain conclusion is assumed in advance.",
            "",
            "## Interpretation",
            "",
            f"RoBERTa reaches {h['mean'] * 100:.2f}% non-neutral weighted-F1, while non-neutral macro-F1 is {result['macro_f1_no_neutral']['mean'] * 100:.2f}%. The gap indicates uneven class performance; per-class results should accompany the headline.",
            "",
            "See docs/EMOWOZ_B4.md for model settings, pinned pretraining revision, attribution, reproduction and checkpoint sharing. Large weights remain outside Git. Slides, own-voice narration and personal declarations remain separate deliverables.",
        ]
    )
    paired = pd.read_csv(OUT / "paired_comparisons.csv")
    table = [
        "",
        "| Comparison | Group | n | Mean delta (points) | 95% paired dialogue CI (points) |",
        "|---|---|---:|---:|---|",
    ]
    for row in paired.itertuples(index=False):
        table.append(
            f"| {row.comparison} | {row.group} | {row.n} | {row.mean_delta * 100:+.2f} | [{row.ci95_low * 100:+.2f}, {row.ci95_high * 100:+.2f}] |"
        )
    table += [
        "",
        "Intervals are exploratory and unadjusted for multiple comparisons, conditional on the five trained seed pairs.",
        "",
    ]
    report[report.index("## Interpretation") : report.index("## Interpretation")] = table
    (OUT / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(
        cells=[
            md(
                "# Native EmoWOZ context RoBERTa\nWorkstream 5 extension; original B4 implementation credited. Training uses the official train/dev partitions and native labels."
            ),
            code(
                "from pathlib import Path\nimport os\nROOT=Path.cwd()\nif ROOT.name=='notebooks':\n    ROOT=ROOT.parent\nos.chdir(ROOT)\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image"
            ),
            md(
                "## Dev selection\nFive seeds, three fixed epochs; test does not select checkpoints."
            ),
            code(
                "display(json.loads(Path('reports/b4/emowoz/selection.json').read_text()))\ndisplay(Image(filename='reports/b4/emowoz/figures/learning_curves.png'))"
            ),
            md(
                "## Test comparison\nSame native labels and targets for four utterance-only baselines and the context model."
            ),
            code(
                "display(pd.read_csv('reports/b4/emowoz/comparison_b1_b2_b3_b4.csv'))\ndisplay(Image(filename='reports/b4/emowoz/figures/headline_comparison.png'))"
            ),
            md(
                "## First / stable / shift\nGold strata are evaluation metadata, not inputs. Different class mixtures mean a higher shift-subset F1 does not establish dialogue understanding."
            ),
            code(
                "display(pd.read_csv('reports/b4/emowoz/figures/strata.csv'))\ndisplay(Image(filename='reports/b4/emowoz/figures/strata.png'))"
            ),
            md(
                "## Paired context comparisons and E4 controls\nWhole-dialogue intervals conditional on the five fixed seed pairs. Controls perturb inference inputs, without retraining."
            ),
            code(
                "display(pd.read_csv('reports/b4/emowoz/paired_comparisons.csv'))\ndisplay(pd.read_csv('reports/b4/emowoz/controls_summary.csv'))"
            ),
            md(
                "## Native-label demo\nLarge weights are shared separately. System turns are not neutral targets."
            ),
            code(
                "from emotionshift.models.roberta_context import RobertaContext\nselection=json.loads(Path('reports/b4/emowoz/selection.json').read_text())\nseed=selection['best_seed_on_dev']\nmodel=RobertaContext.load(f'models/b4_emowoz/seed{seed}',device='cpu')\nprint(model.labels)\nprint('Past window:', model.k)\nspeakers=['user', 'system', 'user']\ntexts=['I need a hotel.', 'I found a restaurant.', 'That was not what I asked for.']\nprint(model.predict_dialogue(speakers,texts)[-1])"
            ),
            md(
                "## Verification and interpretation\nReport per-class metrics alongside weighted-F1; label imbalance can mask minority-class errors. These are in-domain experiments, with past-only context, without cross-dataset transfer."
            ),
            code("display(json.loads(Path('reports/b4/emowoz/verification.json').read_text()))"),
        ]
    )
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (EmotionShift)",
        "language": "python",
        "name": "python3",
    }
    path = ROOT / "notebooks/emowoz_b4.ipynb"
    nbformat.write(nb, path)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(path)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
    nb = nbformat.read(path, as_version=4)
    assert all(len(c.source.splitlines()) <= 15 for c in nb.cells if c.cell_type == "code")
    os.environ["PYTHONUTF8"] = "1"
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
    from scripts.finalize_emowoz_context import finalize

    finalize()
    print(
        "Audited five context prediction CSVs, E4 controls and paired comparisons; executed the notebook.",
        flush=True,
    )


if __name__ == "__main__":
    main()
