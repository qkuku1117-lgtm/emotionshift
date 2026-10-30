"""Check whether a trained ShiftAware run uses its prior, evidence and gate.

    uv run scripts/diagnose_shiftaware.py --run-dir models/shiftaware/meld_seed42

Scores each head on its own (dev by default), with the gate's mean weight on the
evidence head for shift and stable turns. A gate near 1 everywhere means the
mixture has collapsed onto the evidence head and the prior plays no role.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from emotionshift.data import DATA_ROOT, EMOTIONS, load
from emotionshift.data.torch_data import DialogueCollator, DialogueDataset
from emotionshift.eval.metrics import scores
from emotionshift.models.shiftaware.training import autocast_dtype, load_trained_model, move_batch

HEADS = {
    "mixture": "emotion_log_probs",
    "prior": "prior_logits",
    "evidence": "evidence_logits",
    "forecast": "forecast_emotion_logits",
}


@torch.inference_mode()
def collect(model, loader, device, dtype) -> dict[str, np.ndarray]:
    names = [
        *HEADS,
        "gold",
        "gate",
        "shift",
        "eligible",
        "shift_prob",
        "forecast_shift_prob",
        "derived_shift_prob",
    ]
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        context = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            output = model(
                batch["tokens"], batch["speaker_ids"], batch["turn_mask"], batch["turn_gaps"]
            )
        valid = batch["turn_mask"]
        for name, field in HEADS.items():
            parts[name].append(getattr(output, field)[valid].argmax(-1))
        parts["gold"].append(batch["labels"][valid])
        parts["gate"].append(output.mixing_weights[valid].float())
        parts["shift"].append(batch["shift_labels"][valid].nan_to_num(0) > 0.5)
        parts["eligible"].append((batch["shift_mask"] & output.shift_mask)[valid])
        parts["shift_prob"].append(output.shift_logits[valid].float().sigmoid())
        parts["forecast_shift_prob"].append(output.forecast_shift_logits[valid].float().sigmoid())
        # Parameter-free shift score: chance that the speaker's previous and current
        # predicted emotions differ. Only meaningful where the speaker spoke before.
        probs = output.emotion_probs.float()
        speakers, turns = batch["speaker_ids"], valid.shape[1]
        positions = torch.arange(turns, device=device)
        previous_own = (
            (speakers[:, :, None] == speakers[:, None, :])
            & (positions[:, None] > positions[None, :])
            & valid[:, None, :]
        )
        last = torch.where(previous_own, positions, 0).amax(-1)
        previous = probs.gather(1, last[..., None].expand(-1, -1, probs.shape[-1]))
        parts["derived_shift_prob"].append((1 - (probs * previous).sum(-1))[valid])
    return {name: torch.cat(values).cpu().numpy() for name, values in parts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.run_dir / "best.pt", map_location="cpu", weights_only=True, mmap=True
    )
    device = torch.device(args.device)
    model, tokenizer = load_trained_model(args.run_dir, device=device)
    loader = DataLoader(
        DialogueDataset(load(checkpoint["dataset"], splits=args.split, root=args.data_root)),
        batch_size=args.batch_size,
        collate_fn=DialogueCollator(tokenizer, max_length=checkpoint["training"]["max_length"]),
    )
    a = collect(model, loader, device, autocast_dtype(device, checkpoint["training"]["amp"]))
    gold = [EMOTIONS[i] for i in a["gold"]]
    eligible, shifted = a["eligible"], a["shift"] & a["eligible"]
    stable = eligible & ~a["shift"]
    result = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "architecture": checkpoint["architecture"],
        "heads": {
            name: {
                key: round(float(value), 4)
                for key, value in scores(gold, [EMOTIONS[i] for i in a[name]]).items()
                if key in ("weighted_f1", "macro_f1", "micro_f1_no_neutral")
            }
            for name in HEADS
        },
        "gate_on_evidence": {
            "mean": round(float(a["gate"].mean()), 4),
            "shift_turns": round(float(a["gate"][shifted].mean()), 4),
            "stable_turns": round(float(a["gate"][stable].mean()), 4),
            "first_turns": round(float(a["gate"][~eligible].mean()), 4),
        },
        "shift_auroc": {
            name: round(float(roc_auc_score(a["shift"][eligible], a[name][eligible])), 4)
            for name in ("shift_prob", "forecast_shift_prob", "derived_shift_prob")
        },
        # Final predictions by stratum, as in the stratified test reports.
        "strata": {
            stratum: {
                key: round(float(value), 4)
                for key, value in scores(
                    [gold[i] for i in np.flatnonzero(rows)],
                    [EMOTIONS[i] for i in a["mixture"][rows]],
                ).items()
                if key in ("n", "weighted_f1", "micro_f1_no_neutral")
            }
            for stratum, rows in (("first", ~eligible), ("stable", stable), ("shift", shifted))
        },
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
