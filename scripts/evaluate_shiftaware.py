"""Score a trained ShiftAware run on dev and test, in the same format as B1.

    uv run scripts/evaluate_shiftaware.py --run-dir models/shiftaware/meld_seed42

Writes
    predictions/<variant>_<dataset>_<split>_seed<k>.csv   shared prediction format (+ shift_prob)
    reports/<variant>/<dataset>/seed<k>/summary.json      dev check, test metrics, CIs, vs B1
    reports/<variant>/<dataset>/seed<k>/stratified_<split>.csv, per_class_<split>.csv,
                                        confusion_<split>.csv

<variant> defaults to "shiftaware"; name ablations distinctly, e.g. shiftaware_cw-sqrt.

Dev is re-scored only to confirm the reloaded checkpoint matches training.
Nothing here feeds back into model selection.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from emotionshift.data import DATA_ROOT, EMOTIONS, load
from emotionshift.data.torch_data import DialogueCollator, DialogueDataset
from emotionshift.eval import metrics as M
from emotionshift.models.shiftaware.training import autocast_dtype, load_trained_model, move_batch

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLITS = {"meld": ["test"], "dailydialog": ["test_dedup", "test"]}  # as in b1_tfidf.yaml
PRED_COLS = [
    "dataset",
    "split",
    "dialogue_id",
    "source_split",
    "source_dialogue_id",
    "turn",
    "speaker",
    "raw_text",
    "text",
    "emotion",
    "pred",
    "stratum",
    "shift_type",
    "model",
    "seed",
]
KEYS = ["dataset", "split", "dialogue_id", "turn"]
METRIC_KEYS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]


@torch.inference_mode()
def predict(model, tokenizer, frame, *, max_length, batch_size, device, dtype) -> pd.DataFrame:
    """Return `frame` with pred, shift_prob and shift_eligible columns."""
    loader = DataLoader(
        DialogueDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DialogueCollator(tokenizer, max_length=max_length),
    )
    rows = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        context = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            output = model(
                batch["tokens"], batch["speaker_ids"], batch["turn_mask"], batch["turn_gaps"]
            )
        predicted = output.emotion_probs.argmax(-1).cpu()
        shift_probs = output.shift_probs.float().cpu()
        eligible = (batch["shift_mask"] & output.shift_mask).cpu()
        for i, turns in enumerate(cpu_batch["metadata"]):
            for t, meta in enumerate(turns):
                rows.append(
                    {
                        **{key: meta[key] for key in KEYS},
                        "pred": EMOTIONS[int(predicted[i, t])],
                        "shift_prob": float(shift_probs[i, t]),
                        "shift_eligible": bool(eligible[i, t]),
                    }
                )
    merged = frame.merge(pd.DataFrame(rows), on=KEYS, how="left", validate="one_to_one")
    if merged["pred"].isna().any():
        raise RuntimeError("Some utterances received no prediction")
    return merged


def shift_head(df: pd.DataFrame) -> dict[str, float | int]:
    """Task T2 from the model's own shift head, on turns with a previous same-speaker turn."""
    part = df[df["shift_eligible"]]
    gold = (part["stratum"] == "shift").astype(int)
    pred = (part["shift_prob"] >= 0.5).astype(int)
    return {
        "n": len(part),
        "base_rate": float(gold.mean()),
        "shift_f1": float(f1_score(gold, pred, zero_division=0)),
        "auroc": float(roc_auc_score(gold, part["shift_prob"])) if gold.nunique() == 2 else None,
        "predicted_shift_rate": float(pred.mean()),
    }


def versus_b1(df: pd.DataFrame, dataset: str, split: str, metric: str, n_boot: int) -> dict | None:
    """Paired dialogue bootstrap of ShiftAware minus B1 (seed 0), overall and on shift turns."""
    path = ROOT / "predictions" / f"b1_{dataset}_{split}_seed0.csv"
    if not path.exists():
        return None
    b1 = pd.read_csv(path, usecols=[*KEYS, "pred"]).rename(columns={"pred": "pred_b1"})
    paired = df.merge(b1, on=KEYS, how="inner", validate="one_to_one")
    if len(paired) != len(df):
        raise RuntimeError(f"B1 predictions in {path} do not cover the same utterances")
    result = {}
    for name, mask in (("all", None), ("shift_turns", paired["stratum"] == "shift")):
        diff = M.paired_bootstrap(
            paired, "pred_b1", "pred", metric=metric, mask=mask, n_boot=n_boot
        )
        result[name] = {key: float(value) for key, value in diff.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--variant", default="shiftaware", help="Model name for output paths")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.run_dir / "best.pt", map_location="cpu", weights_only=True, mmap=True
    )
    dataset, seed = checkpoint["dataset"], checkpoint["training"]["seed"]
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    device = torch.device(args.device)
    dtype = autocast_dtype(device, checkpoint["training"]["amp"])
    model, tokenizer = load_trained_model(args.run_dir, device=device)
    headline = M.HEADLINE[dataset]

    out = ROOT / "reports" / args.variant / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "predictions").mkdir(exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "model": args.variant,
        "run_dir": str(args.run_dir),
        "git": run.get("git"),
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "selected_on_dev": headline,
        "dev": {},
        "test": {},
    }

    for split in ["dev", *TEST_SPLITS[dataset]]:
        frame = load(dataset, splits=split, root=args.data_root)
        df = predict(
            model,
            tokenizer,
            frame,
            max_length=checkpoint["training"]["max_length"],
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )
        df["model"], df["seed"] = args.variant, seed
        overall = M.scores(df["emotion"], df["pred"])
        if split == "dev":
            training_dev = checkpoint["dev_metrics"][headline]
            summary["dev"] = {
                **{key: float(overall[key]) for key in METRIC_KEYS},
                "n": len(df),
                f"{headline}_at_training": training_dev,
            }
            print(f"[dev] {headline} {overall[headline]:.4f} (training run: {training_dev:.4f})")
            continue

        df[[*PRED_COLS, "shift_prob"]].to_csv(
            ROOT / "predictions" / f"{args.variant}_{dataset}_{split}_seed{seed}.csv", index=False
        )
        result: dict[str, Any] = {key: float(overall[key]) for key in METRIC_KEYS}
        result["ci95"] = {
            key: M.bootstrap_ci(df, key, n_boot=args.bootstrap) for key in METRIC_KEYS[:3]
        }
        result["n"] = len(df)
        result["shift_head"] = shift_head(df)
        result["derived_shift_detection"] = M.derived_shift_detection(df)
        result["vs_b1_seed0"] = versus_b1(df, dataset, split, headline, args.bootstrap)
        summary["test"][split] = result

        M.stratified(df).round(4).to_csv(out / f"stratified_{split}.csv", index=False)
        M.per_class_f1(df["emotion"], df["pred"]).round(4).to_csv(
            out / f"per_class_{split}.csv", index=False
        )
        M.confusion(df["emotion"], df["pred"]).to_csv(out / f"confusion_{split}.csv")
        low, high = result["ci95"][headline]
        print(f"[{split}] {headline} {result[headline]:.4f} [{low:.4f}, {high:.4f}]")

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"Wrote {out.relative_to(ROOT)}/summary.json")


if __name__ == "__main__":
    main()
