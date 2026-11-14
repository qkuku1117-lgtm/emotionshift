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
from emotionshift.eval import efficiency as E
from emotionshift.eval import metrics as M
from emotionshift.models.roberta import RobertaUtterance
from scripts.train_b0 import PRED_COLS
from scripts.train_emowoz_baselines import digest, targets, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b3/emowoz"
LABELS = emotions("emowoz")


def checkpoint_digests():
    result = {}
    for seed in range(5):
        path = ROOT / f"models/b3_emowoz/seed{seed}/model.safetensors"
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
    target = targets(["test"]).reset_index(drop=True)
    hashes = {}
    histories = []
    for seed in range(5):
        path = ROOT / f"predictions/b3_emowoz_test_seed{seed}.csv"
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
        assert frame.model.eq("b3").all() and frame.seed.eq(seed).all()
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
    model = RobertaUtterance.load(ROOT / "models/b3_emowoz/seed0", device="cpu")
    assert model.labels == LABELS
    probs = model.predict_proba(target.text.iloc[:16], batch_size=4)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
    expected = pd.read_csv(ROOT / "predictions/b3_emowoz_test_seed0.csv").pred.iloc[:16].tolist()
    assert [LABELS[i] for i in probs.argmax(axis=1)] == expected
    write(
        OUT / "verification.json",
        {
            "prediction_csvs": 5,
            "targets_per_csv": 8634,
            "native_labels_verified": True,
            "dev_selection_verified": True,
            "metrics_recomputed": True,
            "checkpoint_round_trip_verified": True,
            "prediction_sha256": hashes,
            "checkpoint_sha256": checkpoint_digests(),
        },
    )
    summary = json.loads((OUT / "summary.json").read_text())
    summary["efficiency"] = {
        "parameters": model.num_parameters(),
        "seed0_model_dir_mb": sum(
            p.stat().st_size for p in (ROOT / "models/b3_emowoz/seed0").rglob("*") if p.is_file()
        )
        / 1e6,
        "inference_device": "cpu",
        "machine": E.machine(),
        "latency_one_utterance": E.latency_ms(model.predict, target.text, n=100),
        "batch_throughput": E.throughput(model.predict, target.text),
        "timing_note": "Loaded seed-0 checkpoint after all training. Resumed training metadata records invocation time, not total training time.",
    }
    write(OUT / "summary.json", summary)
    result = summary["test"]["metrics"]
    for metric in M.METRICS:
        assert np.isclose(result[metric]["mean"], stored[metric].mean())
    comparison = pd.read_csv(ROOT / "reports/emowoz_baselines/comparison.csv")
    h = result["weighted_f1_no_neutral"]
    row = {
        "model": "b3_roberta",
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
    comparison.to_csv(OUT / "comparison_b1_b2_b3.csv", index=False)
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
    ax.set_title("Native EmoWOZ RoBERTa: dev-only checkpoint selection")
    ax.legend()
    fig.savefig(figdir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    sources = {
        "b1": ROOT / "reports/b1/emowoz/stratified_test.csv",
        "b2_bigru": ROOT / "reports/b2/emowoz/b2_bigru/stratified_test.csv",
        "b2_bilstm": ROOT / "reports/b2/emowoz/b2_bilstm/stratified_test.csv",
        "b3_roberta": OUT / "stratified_test.csv",
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
        "# Native EmoWOZ RoBERTa: workstream 4",
        "",
        "Five seeds, official splits, native labels, current utterance only.",
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
            "The comparison reuses the audited SVM/BiGRU/BiLSTM results without overwriting them. All models use the same native EmoWOZ test targets and headline metric. Single-model bootstrap intervals do not establish pairwise significance. Native scores are not directly interchangeable with the MELD/DailyDialog metrics.",
            "",
            "## Interpretation",
            "",
            f"RoBERTa reaches {h['mean'] * 100:.2f}% non-neutral weighted-F1, while non-neutral macro-F1 is {result['macro_f1_no_neutral']['mean'] * 100:.2f}%. The gap indicates uneven class performance; per-class results should accompany the headline.",
            f"First-target F1 is {pivot.loc['b3_roberta', 'stratum:first']:.2f}% for RoBERTa versus {pivot.loc['b2_bigru', 'stratum:first']:.2f}% for BiGRU; shift-subset F1 is {pivot.loc['b3_roberta', 'stratum:shift']:.2f}% versus {pivot.loc['b2_bigru', 'stratum:shift']:.2f}%. Thus the aggregate increase does not establish an improvement on shifted targets. These are descriptive within-subset comparisons; no paired significance test is claimed.",
            "",
            "See docs/EMOWOZ_B3.md for model settings, pinned pretraining revision, attribution, reproduction and checkpoint sharing. Large weights remain outside Git. Slides, own-voice narration and personal declarations remain separate deliverables.",
        ]
    )
    (OUT / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(
        cells=[
            md(
                "# Native EmoWOZ single-utterance RoBERTa\nWorkstream 4 extension; original B3 implementation credited. Training uses the official train/dev partitions and native labels."
            ),
            code(
                "from pathlib import Path\nimport os\nROOT=Path.cwd()\nif ROOT.name=='notebooks':\n    ROOT=ROOT.parent\nos.chdir(ROOT)\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image"
            ),
            md(
                "## Dev selection\nFive seeds, three fixed epochs; test does not select checkpoints."
            ),
            code(
                "display(json.loads(Path('reports/b3/emowoz/selection.json').read_text()))\ndisplay(Image(filename='reports/b3/emowoz/figures/learning_curves.png'))"
            ),
            md(
                "## Test comparison\nSame native labels and targets for all four utterance-only baselines."
            ),
            code(
                "display(pd.read_csv('reports/b3/emowoz/comparison_b1_b2_b3.csv'))\ndisplay(Image(filename='reports/b3/emowoz/figures/headline_comparison.png'))"
            ),
            md(
                "## First / stable / shift\nGold strata are evaluation metadata, not inputs. Different class mixtures mean a higher shift-subset F1 does not establish dialogue understanding."
            ),
            code(
                "display(pd.read_csv('reports/b3/emowoz/figures/strata.csv'))\ndisplay(Image(filename='reports/b3/emowoz/figures/strata.png'))"
            ),
            md(
                "## Native-label demo\nLarge weights are shared separately. System turns are not neutral targets."
            ),
            code(
                "from emotionshift.models.roberta import RobertaUtterance\nselection=json.loads(Path('reports/b3/emowoz/selection.json').read_text())\nseed=selection['best_seed_on_dev']\nmodel=RobertaUtterance.load(f'models/b3_emowoz/seed{seed}',device='cpu')\nprint(model.labels)\nprint(model.predict(['Thank you, that was helpful.', 'You did not answer my question.']))"
            ),
            md(
                "## Verification and interpretation\nReport per-class metrics alongside weighted-F1; label imbalance can mask minority-class errors. These are in-domain experiments, without dialogue history or cross-dataset transfer."
            ),
            code("display(json.loads(Path('reports/b3/emowoz/verification.json').read_text()))"),
        ]
    )
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (EmotionShift)",
        "language": "python",
        "name": "python3",
    }
    path = ROOT / "notebooks/emowoz_b3.ipynb"
    nbformat.write(nb, path)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(path)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
    nb = nbformat.read(path, as_version=4)
    assert all(len(c.source.splitlines()) <= 15 for c in nb.cells if c.cell_type == "code")
    os.environ["PYTHONUTF8"] = "1"
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
    print("Audited five prediction CSVs and executed the native RoBERTa notebook.", flush=True)


if __name__ == "__main__":
    main()
