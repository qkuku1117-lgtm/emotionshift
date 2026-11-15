"""Audit B0 prediction exports, create source figures and execute the walkthrough."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import load
from emotionshift.eval import metrics as M
from scripts.analyze_b0_paired import main as paired_analysis
from scripts.analyze_b0_transitions import main as transition_analysis
from scripts.explain_b0_inertia import main as explain_inertia
from scripts.train_b0 import PRED_COLS

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/b0"


def audit():
    checked = 0
    for dataset in ("meld", "dailydialog", "emowoz"):
        source = load(dataset)
        source = source[source.emotion.notna()]
        metrics = pd.read_csv(REPORT / dataset / "per_seed.csv")
        for row in metrics.itertuples(index=False):
            path = ROOT / "predictions" / f"b0_{row.model}_{dataset}_{row.split}_seed{row.seed}.csv"
            frame = pd.read_csv(path, keep_default_na=False)
            target = source[source.split == row.split].reset_index(drop=True)
            assert frame.columns[:12].tolist() == PRED_COLS, path
            assert len(frame) == len(target), path
            for col in ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"]:
                assert frame[col].astype(str).tolist() == target[col].astype(str).tolist(), (
                    path,
                    col,
                )
            assert not frame.duplicated(["dataset", "split", "dialogue_id", "turn"]).any(), path
            assert frame.pred.isin(emotions(dataset)).all(), path
            actual = M.scores(frame.emotion, frame.pred, emotions(dataset))
            for metric in M.METRICS:
                assert np.isclose(actual[metric], getattr(row, metric)), (path, metric)
            derived = M.derived_shift_detection(frame)
            for metric in ["shift_f1", "base_rate", "predicted_shift_rate"]:
                assert np.isclose(derived[metric], getattr(row, metric)), (path, metric)
            checked += 1
    (REPORT / "verification.json").write_text(
        json.dumps(
            {
                "prediction_files_checked": checked,
                "checks": [
                    "original 12 column order",
                    "source row identity",
                    "native label space",
                    "unique keys",
                    "stored metrics recomputed",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checked


def figures():
    out = REPORT / "figures"
    out.mkdir(exist_ok=True)
    comparison = pd.read_csv(REPORT / "comparison.csv")
    comparison.to_csv(out / "headline_comparison.csv", index=False)
    names = ["majority", "donor", "inertia_pred", "markov_pred", "inertia_oracle", "markov_oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, dataset, split in zip(
        axes, ["meld", "dailydialog", "emowoz"], ["test", "test_dedup", "test"], strict=True
    ):
        rows = (
            comparison[(comparison.dataset == dataset) & (comparison.split == split)]
            .set_index("model")
            .loc[names]
        )
        means = rows.headline.to_numpy() * 100
        # CIs describe seed 0, so draw them around seed-0 scores, not seed means.
        per_seed = pd.read_csv(REPORT / dataset / "per_seed.csv")
        s0 = (
            per_seed[(per_seed.split == split) & (per_seed.seed == 0)].set_index("model").loc[names]
        )
        centers = s0[rows.iloc[0].headline_metric].to_numpy() * 100
        y = np.arange(len(names))
        bars = ax.barh(y, means, color=["#466b8e"] * 4 + ["#b5b5b5"] * 2)
        for bar in bars[-2:]:
            bar.set_hatch("//")
        ax.errorbar(
            centers,
            y,
            xerr=[centers - rows.ci_low.to_numpy() * 100, rows.ci_high.to_numpy() * 100 - centers],
            fmt="none",
            color="black",
            capsize=3,
        )
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(f"{dataset} / {split}")
        ax.set_xlabel(rows.iloc[0].headline_metric + " (%)")
    fig.suptitle("B0: seed means; 95% dialogue CI for seed 0; hatched = oracle")
    fig.savefig(out / "headline_comparison.png", dpi=160)
    plt.close(fig)
    strata = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, dataset, split in zip(
        axes, ["meld", "dailydialog", "emowoz"], ["test", "test_dedup", "test"], strict=True
    ):
        metric = M.HEADLINE[dataset]
        for name in names:
            table = pd.read_csv(REPORT / dataset / f"stratified_{name}_{split}.csv").set_index(
                "group"
            )
            for group in ["stable", "shift"]:
                strata.append(
                    {
                        "dataset": dataset,
                        "model": name,
                        "group": group,
                        "metric": metric,
                        "score": table.loc[f"stratum:{group}", metric],
                    }
                )
        subset = pd.DataFrame(strata)
        subset = subset[subset.dataset == dataset]
        for offset, group in [(-0.18, "stable"), (0.18, "shift")]:
            values = subset[subset.group == group].set_index("model").loc[names, "score"]
            ax.bar(np.arange(len(names)) + offset, values * 100, width=0.36, label=group)
        ax.set_xticks(np.arange(len(names)), names, rotation=50, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel(metric + " (%)")
        ax.set_title(dataset + " (seed 0)")
        ax.legend()
    fig.suptitle("Stable vs shift: oracle rows are references, not deployable results")
    fig.savefig(out / "shift_vs_stable.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(strata).to_csv(out / "shift_vs_stable.csv", index=False)


def notebook():
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [
        md(
            "# B0 — Majority, inertia and Markov\n\nWorkstream 1. CPU experiments on the three native label spaces. "
            "Training is reproducible via `scripts/train_b0.py`; this executed notebook audits and explains saved results. "
            "Gold-history variants are **oracle references**, not headline models."
        ),
        code(
            "from pathlib import Path\nimport os\nimport sys\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nos.chdir(ROOT)\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image\nfrom emotionshift.data.loaders import load\nfrom emotionshift.data.labels import emotions\nfrom emotionshift.models.b0 import HistoryBaselines"
        ),
        md(
            "## Data contract\nEmoWOZ retains native classes; only labelled user turns are scored. "
            "DailyDialog headline uses test_dedup; test is secondary."
        ),
        code(
            "counts = []\nfor dataset in ['meld', 'dailydialog', 'emowoz']:\n    frame = load(dataset)\n    targets = frame[frame.emotion.notna()]\n    for split, group in targets.groupby('split'):\n        counts.append({'dataset': dataset, 'split': split, 'targets': len(group)})\ndisplay(pd.DataFrame(counts))"
        ),
        md(
            "## Baseline rules\nThe donor is the supporting utterance-only SVM. Inertia copies its **previous** "
            "prediction for the same speaker. Markov reads only that historical label and train transitions. "
            "The first turn uses train majority. Recursive self-copying would never change class."
        ),
        code(
            "df = load('meld')\ntrain = df[df.split == 'train']\ntest = df[df.split == 'test'].copy()\npath = 'predictions/b0_donor_meld_test_seed0.csv'\ndonor = pd.Series(pd.read_csv(path).pred.to_numpy(), index=test.index)\nb0 = HistoryBaselines(emotions('meld')).fit(train)\npredictions = b0.predict(test.drop(columns=['emotion', 'prev_emotion']), donor)\ndisplay(pd.concat([test[['dialogue_id', 'turn', 'speaker', 'text']], predictions], axis=1).head(8))"
        ),
        md(
            "## Inertia: one-step error carry-over\n"
            "Copy the same speaker's previous **SVM prediction**, not the previous inertia output. "
            "First turns use train majority. The toy below has manually specified labels and donor "
            "outputs; it is an illustration, not a measured SVM result. At A's turn 2 a wrong "
            "history is copied; at turn 4 even correct gold history becomes stale; turn 5 recovers "
            "using the updated donor. Speaker B keeps independent history."
        ),
        code(
            "toy = pd.read_csv('reports/b0/inertia_examples/controlled_example.csv')\n"
            "cols = ['turn', 'speaker', 'text', 'emotion', 'donor_pred', 'history_pred',\n"
            "        'inertia_pred', 'inertia_oracle', 'stratum']\n"
            "display(toy[cols])\n"
            "display(toy[['turn', 'explanation']])"
        ),
        md(
            "## Real saved-prediction cases (MELD, seed 0)\n"
            "Post-hoc compact examples selected to illustrate two mechanisms, not representative "
            "sampling or test-based tuning. target_row marks the explained turn. Other speakers' "
            "intervening turns are shown to make the same-speaker history explicit. "
            "See reports/b0/inertia_examples/README.md for the detailed walkthrough."
        ),
        code(
            "cases = pd.read_csv('reports/b0/inertia_examples/meld_seed0_cases.csv')\n"
            "for name, group in cases.groupby('case', sort=False):\n"
            "    print(name)\n"
            "    display(group[['dialogue_id', *cols, 'target_row']])"
        ),
        md(
            "## Main comparison\nEach panel uses its dataset's headline metric. Bars show seed means; "
            "intervals resample dialogues for seed 0. Deterministic rules have one export; oracle references are flagged."
        ),
        code(
            "comparison = pd.read_csv('reports/b0/comparison.csv')\ndisplay(comparison)\ndisplay(Image(filename='reports/b0/figures/headline_comparison.png'))"
        ),
        md(
            "## Paired differences\nEach bootstrap replicate uses the same sampled dialogues for both methods. "
            "Delta is candidate minus reference in percentage points, comparing seed-0 checkpoints, "
            "not five-seed means. An interval crossing zero does not establish a clear direction. "
            "Oracle comparisons are diagnostic. These exploratory post-hoc pointwise intervals "
            "have no multiple-comparison correction and do not establish causality."
        ),
        code(
            "paired = pd.read_csv('reports/b0/paired/headline_differences.csv')\n"
            "display(paired[['dataset', 'metric', 'reference', 'candidate', 'oracle',\n"
            "                'diff_pp', 'ci_low_pp', 'ci_high_pp', 'interval_direction']])\n"
            "display(Image(filename='reports/b0/paired/headline_differences.png'))"
        ),
        md(
            "## Shift vs stable\nGold shift strata are evaluation metadata only. They are not inference inputs."
        ),
        code(
            "display(Image(filename='reports/b0/figures/shift_vs_stable.png'))\nstrata = pd.read_csv('reports/b0/figures/shift_vs_stable.csv')\ndisplay(strata)"
        ),
        md(
            "## Forecasting\nMarkov shift probabilities are emitted before the current text is read; "
            "only non-first user/speaker turns are eligible. Oracle histories are separated. These tables have point estimates, not forecast CIs."
        ),
        code(
            "for dataset in ['meld', 'dailydialog', 'emowoz']:\n    forecast = pd.read_csv(f'reports/b0/{dataset}/forecast.csv')\n    display(forecast[forecast.seed == 0])"
        ),
        md(
            "## What did Markov learn?\nRows are the previous labelled emotion of the same speaker; "
            "columns are the next/current emotion. Diagonal cells measure persistence; the neutral "
            "column measures return to neutral. All probabilities come from train, with row support "
            "shown on the heatmaps. Argmax may map multiple history labels to neutral, limiting "
            "emotional-class recall. These are population patterns, not current-text understanding. "
            "Small-support rows require caution; no cellwise probability intervals are estimated."
        ),
        code(
            "transition = pd.read_csv('reports/b0/transitions/transition_summary.csv')\n"
            "display(transition)\n"
            "for dataset in ['meld', 'dailydialog', 'emowoz']:\n"
            "    display(Image(filename=f'reports/b0/transitions/{dataset}_transition.png'))\n"
            "distribution = pd.read_csv('reports/b0/transitions/prediction_distribution.csv')\n"
            "display(distribution[distribution.model == 'markov_pred'])"
        ),
        md(
            "## Verification and limits\nRun the artifact audit to recompute every stored recognition metric and check row identity. "
            "Native EmoWOZ scores are not directly comparable to Ekman-label scores. "
            "Neither lagged recognition nor Markov models understand the current text; oracle performance cannot be claimed at deployment."
        ),
        code(
            "from scripts.make_b0_artifacts import audit\nprint('Prediction files verified:', audit())"
        ),
        md(
            "## References and hand-in\nSee `docs/B0_BASELINES.md` and `docs/references/b0_refs.txt`. "
            "The data-loader, donor and metric implementation are shared team work. "
            "This work adds B0 rules, experiments, audit and explanation. "
            "Use your own voice for narration and the shared slide template."
        ),
    ]
    assert all(len(c.source.splitlines()) <= 15 for c in nb.cells if c.cell_type == "code")
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (EmotionShift)",
        "language": "python",
        "name": "python3",
    }
    # nbclient subprocesses inherit UTF-8 mode on Windows.
    os.environ["PYTHONUTF8"] = "1"
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "notebooks/b0_baselines.ipynb")


def result_notes():
    table = pd.read_csv(REPORT / "comparison.csv")
    lines = [
        "# B0 results: majority, predicted-history inertia and Markov",
        "",
        "All values below are percentages except AUROC/AUPRC. Native label spaces;",
        "different dataset headline metrics are not directly comparable.",
        "Recognition means use five donor seeds; deterministic rules use seed 0.",
        "95% intervals resample dialogues 1,000 times for seed 0, not across seeds.",
        "",
        "| Dataset / split | Method | Headline F1 | 95% CI (seed 0) | Macro-F1 | Derived shift F1 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        label = row.model + (" (oracle)" if row.oracle else "")
        lines.append(
            f"| {row.dataset} / {row.split} | {label} | {row.headline * 100:.2f} "
            f"± {row.headline_std * 100:.2f} | [{row.ci_low * 100:.2f}, "
            f"{row.ci_high * 100:.2f}] | {row.macro_f1 * 100:.2f} | {row.shift_f1 * 100:.2f} |"
        )
    lines += [
        "",
        "Headlines: MELD weighted-F1; DailyDialog micro-F1 excluding neutral;",
        "EmoWOZ weighted-F1 excluding neutral. All-neutral output therefore scores",
        "zero on the DailyDialog/EmoWOZ headline despite often high accuracy.",
        "",
        "## What the results show",
        "",
        "On MELD, majority reaches 31.27 weighted-F1; lagged predicted-history",
        "inertia reaches 38.30, compared with 44.86 for oracle inertia and 56.78",
        "for the supporting current-text recognizer. Oracle inertia reproduces",
        "the proposal's 0.449 pilot; it must not be presented as deployable.",
        "Pure persistence cannot detect a new emotion from the current utterance.",
        "The native-label EmoWOZ run is in-domain and does not establish transfer",
        "performance. Its score must not be compared directly with Ekman-label F1.",
        "Five donor fits converged to identical test predictions/scores in this run;",
        "zero standard deviation does not mean zero uncertainty (see dialogue CIs).",
        "",
        "## Markov forecasting (seed 0)",
        "",
        "| Dataset / split | History | Eligible turns | Shift rate | AUROC | AUPRC | Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["meld", "dailydialog", "emowoz"]:
        forecast = pd.read_csv(REPORT / dataset / "forecast.csv")
        for row in forecast[forecast.seed == 0].itertuples(index=False):
            lines.append(
                f"| {dataset} / {row.split} | {row.model} | {row.n} | "
                f"{row.base_rate:.3f} | {row.auroc:.3f} | {row.auprc:.3f} | {row.brier:.3f} |"
            )
    lines += [
        "",
        "Forecast probabilities use no current-utterance text; gold-history",
        "oracle rows are references only. AUPRC should be compared with that",
        "split's shift base rate, and Brier is a probability-error metric (lower",
        "is better). These are point estimates, without forecast confidence intervals.",
        "",
        "## Runtime and verification",
        "",
    ]
    for dataset in ["meld", "dailydialog", "emowoz"]:
        summary = json.loads((REPORT / dataset / "summary.json").read_text(encoding="utf-8"))
        lines.append(
            f"- {dataset}: {summary['seconds']:.1f} seconds including dev selection, fitting, recognition CIs and exports."
        )
    lines += [
        "",
        "CPU run, Python 3.12. The artifact audit recomputed all 72 prediction",
        "files' recognition metrics and verified source identity and CSV schema.",
        "Tests: 70 passed, 5 skipped in this environment (optional neural dependencies absent).",
        "The source manifest hashes raw files, config and training/rule code.",
        "See docs/B0_BASELINES.md for definitions and notebooks/b0_baselines.ipynb",
        "for the executed walkthrough. Training/evaluation commands are not run",
        "again inside the presentation notebook.",
        "",
        "## Paired method differences",
        "",
        "See [paired/README.md](paired/README.md) for candidate-minus-reference",
        "percentage-point differences and pointwise 95% paired dialogue-bootstrap",
        "intervals. These fixed seed-0 comparisons directly assess the difference,",
        "rather than overlap between separate model intervals. They are exploratory",
        "and do not include multiple-comparison correction.",
        "",
        "## Training transition interpretation",
        "",
        "See [transitions/README.md](transitions/README.md) for verified train-only",
        "heatmaps, per-history supports, argmax rules, and seed-0 prediction",
        "distributions. It explains neutral collapse and why transition argmax",
        "need not improve non-neutral F1. No current-text evidence enters the prior.",
    ]
    (REPORT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    checked = audit()
    figures()
    result_notes()
    explain_inertia()
    paired_analysis()
    transition_analysis()
    notebook()
    print(f"Verified {checked} prediction files; wrote figures and executed notebook.")


if __name__ == "__main__":
    main()
