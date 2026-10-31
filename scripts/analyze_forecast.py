"""What does ShiftAware's history-only forecast read? Run on a trained transformer checkpoint.

    uv run scripts/analyze_forecast.py --run-dir models/shiftaware/emowoz_tf-d768-2layer-hm_seed42

Scored on turns with a previous same-speaker labelled turn (test by default):

* attention: the forecast's mean attention weight on the previous turn (on
  EmoWOZ, the system reply), on the speaker's own previous turn, on older
  turns and on the null slot. With context layers each position also mixes
  in earlier turns, so these weights locate the read, not its content.
* blank_<speaker>: the same scores with that speaker's text replaced by "",
  a direct test of what the forecast and recognition use. EmoWOZ already
  holds 849 empty system turns, so "" is a known input.
* not_last: forecast AUROCs without each dialogue's last turn of the
  speaker, where closing thanks make "satisfied" easy to foresee.

Writes reports/<variant>/<dataset>/seed<k>/forecast_analysis_<split>.json.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from emotionshift.data import DATA_ROOT, load
from emotionshift.data.labels import emotions
from emotionshift.data.torch_data import DialogueCollator, DialogueDataset
from emotionshift.eval import metrics as M
from emotionshift.models.shiftaware.training import autocast_dtype, load_trained_model, move_batch

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["dataset", "split", "dialogue_id", "turn"]
PARTS = ("previous_turn", "own_previous_turn", "older", "null")


@torch.inference_mode()
def collect(model, tokenizer, frame, *, labels, max_length, batch_size, device, dtype):
    """Labelled rows of `frame` with recognition, forecast and forecast-attention columns."""
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
                batch["tokens"],
                batch["speaker_ids"],
                batch["turn_mask"],
                batch["turn_gaps"],
                return_attention=True,
            )
        weights = output.forecast_attention_weights.float().cpu()
        predicted = output.emotion_probs.argmax(-1).cpu()
        forecast = output.forecast_emotion_probs.float().cpu()
        forecast_shift = output.forecast_shift_probs.float().cpu()
        speakers = cpu_batch["speaker_ids"]
        for i, turns in enumerate(cpu_batch["metadata"]):
            for t, meta in enumerate(turns):
                own = [s for s in range(t) if speakers[i, s] == speakers[i, t]]
                w = weights[i, t, :t]
                previous = float(w[t - 1]) if t else 0.0
                own_previous = float(w[own[-1]]) if own and own[-1] != t - 1 else 0.0
                rows.append(
                    {
                        **{key: meta[key] for key in KEYS},
                        "pred": labels[int(predicted[i, t])],
                        "forecast_pred": labels[int(forecast[i, t].argmax())],
                        "forecast_shift_prob": float(forecast_shift[i, t]),
                        **{
                            f"forecast_p_{label}": float(forecast[i, t, k])
                            for k, label in enumerate(labels)
                        },
                        "previous_turn": previous,
                        "own_previous_turn": own_previous,
                        "older": float(w.sum()) - previous - own_previous,
                        "null": 1 - float(w.sum()),
                    }
                )
    merged = frame.merge(pd.DataFrame(rows), on=KEYS, how="left", validate="one_to_one")
    return merged[merged["emotion"].notna()].reset_index(drop=True)


def summarize(df, train, labels, headline):
    """Recognition and forecast scores, plus forecast AUROCs without last turns."""
    forecast = M.forecast_scores(df, train, labels)
    last = df.groupby(["split", "dialogue_id", "speaker"])["turn"].transform("max") == df["turn"]
    not_last = M.forecast_scores(df[~last], train, labels)
    return {
        "recognition": float(M.scores(df["emotion"], df["pred"], labels)[headline]),
        "forecast_shift_auroc": forecast["shift_auroc"]["forecast"],
        "forecast_emotion": forecast["emotion"]["forecast"][headline],
        "forecast_class_auroc": {k: v["forecast"] for k, v in forecast["class_auroc"].items()},
        "not_last": {
            "n": not_last["n"],
            "forecast_shift_auroc": not_last["shift_auroc"]["forecast"],
            "forecast_class_auroc": {k: v["forecast"] for k, v in not_last["class_auroc"].items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variant", default="shiftaware", help="Model name for output paths")
    parser.add_argument(
        "--blank", nargs="*", default=["system"], help="Speakers whose text to blank in turn"
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.run_dir / "best.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if checkpoint["architecture"].get("dynamics") != "transformer":
        raise SystemExit("Forecast attention analysis needs a transformer checkpoint")
    dataset, seed = checkpoint["dataset"], checkpoint["training"]["seed"]
    device = torch.device(args.device)
    model, tokenizer = load_trained_model(args.run_dir, device=device)
    labels, headline = emotions(dataset), M.HEADLINE[dataset]
    train = load(dataset, splits="train", root=args.data_root)
    frame = load(dataset, splits=args.split, root=args.data_root)
    settings = {
        "labels": labels,
        "max_length": checkpoint["training"]["max_length"],
        "batch_size": args.batch_size,
        "device": device,
        "dtype": autocast_dtype(device, checkpoint["training"]["amp"]),
    }

    df = collect(model, tokenizer, frame, **settings)
    eligible = df["stratum"].isin(("stable", "shift"))
    groups = {
        "all": eligible,
        "stable": df["stratum"] == "stable",
        "shift": df["stratum"] == "shift",
        **{label: eligible & (df["emotion"] == label) for label in labels},
    }
    result = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "context_layers": checkpoint["architecture"].get("context_layers"),
        "fusion": checkpoint["architecture"].get("fusion", "none"),
        "attention": {
            name: {"n": int(rows.sum()), **{p: float(df.loc[rows, p].mean()) for p in PARTS}}
            for name, rows in groups.items()
            if rows.any()
        },
        "intact": summarize(df, train, labels, headline),
    }
    for speaker in args.blank:
        if not (frame["speaker"] == speaker).any():
            raise SystemExit(f"No turns by speaker {speaker!r} to blank")
        blanked = frame.assign(text=frame["text"].where(frame["speaker"] != speaker, ""))
        result[f"blank_{speaker}"] = summarize(
            collect(model, tokenizer, blanked, **settings), train, labels, headline
        )

    out = ROOT / "reports" / args.variant / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"forecast_analysis_{args.split}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "attention"}, indent=1))
    print(json.dumps(result["attention"]["all"]))
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
