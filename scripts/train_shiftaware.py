"""Train ShiftAware on MELD or DailyDialog, selecting checkpoints on dev only.

Requires the optional training extra, a pretrained encoder and local raw data.
Architecture and optimization settings are loaded from separate YAML files.
"""

import argparse
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from emotionshift.data import DATA_ROOT, EMOTIONS, load
from emotionshift.data.torch_data import DialogueCollator, DialogueDataset
from emotionshift.eval.metrics import HEADLINE, scores
from emotionshift.models.shiftaware import (
    ShiftAware,
    ShiftAwareConfig,
    ShiftAwareLoss,
    TextShiftAware,
    TransformerUtteranceEncoder,
)
from emotionshift.models.shiftaware.training import (
    TrainingConfig,
    autocast_dtype,
    make_optimizer,
    make_scheduler,
    move_batch,
    train_epoch,
)


def read_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return settings


def git_state() -> dict[str, Any]:
    """Commit and dirty flag so a checkpoint can be traced to its code."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def log_line(row: dict) -> None:
    print(json.dumps(row), flush=True)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, *, dtype=None) -> dict[str, float]:
    model.eval()
    gold, predicted, shifts, shift_predictions = [], [], [], []
    loss_sum = 0.0
    count = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        context = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            output = model(
                batch["tokens"], batch["speaker_ids"], batch["turn_mask"], batch["turn_gaps"]
            )
            loss = criterion(output, batch["labels"], batch["shift_labels"], batch["shift_mask"])[
                "total"
            ]
        valid = batch["turn_mask"]
        eligible = batch["shift_mask"] & output.shift_mask
        gold.extend(batch["labels"][valid].cpu().tolist())
        predicted.extend(output.emotion_probs[valid].argmax(-1).cpu().tolist())
        shifts.extend(batch["shift_labels"][eligible].int().cpu().tolist())
        shift_predictions.extend((output.shift_probs[eligible] >= 0.5).int().cpu().tolist())
        n = int(valid.sum())
        loss_sum += loss.item() * n
        count += n
    if not count:
        raise ValueError("Dev loader must be nonempty")
    metrics = scores([EMOTIONS[i] for i in gold], [EMOTIONS[i] for i in predicted])
    return {
        "loss": loss_sum / count,
        "weighted_f1": float(metrics["weighted_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1_no_neutral": float(metrics["micro_f1_no_neutral"]),
        "shift_f1": float(f1_score(shifts, shift_predictions, zero_division=0)) if shifts else 0.0,
        "n": count,
        "n_shift_eligible": len(shifts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("meld", "dailydialog"), required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--architecture", type=Path, default=Path("configs/shiftaware.yaml"))
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/shiftaware_train.yaml")
    )
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    settings = TrainingConfig(**read_settings(args.training_config))
    architecture = ShiftAwareConfig(**read_settings(args.architecture))
    device = torch.device(args.device)
    dtype = autocast_dtype(device, settings.amp)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
        torch.set_float32_matmul_precision("high")

    train_frame = load(args.dataset, splits="train", root=args.data_root)
    dev_frame = load(args.dataset, splits="dev", root=args.data_root)
    try:
        auto_tokenizer = import_module("transformers").AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install the text encoder with: uv sync --extra training") from exc
    tokenizer = cast(
        Any, auto_tokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    )
    encoder = TransformerUtteranceEncoder.from_pretrained(
        args.model_name,
        frozen=settings.freeze_encoder,
        local_files_only=args.local_files_only,
    )
    if encoder.embedding_dim != architecture.input_dim:
        raise ValueError("Architecture input_dim must match the pretrained encoder hidden size")
    model = TextShiftAware(encoder, ShiftAware(architecture)).to(device)
    criterion = ShiftAwareLoss(
        shift_weight=settings.shift_weight, forecast_weight=settings.forecast_weight
    ).to(device)

    collator = DialogueCollator(tokenizer, max_length=settings.max_length)
    train_dataset = DialogueDataset(train_frame)
    dev_dataset = DialogueDataset(dev_frame)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed),
        batch_size=settings.batch_size,
        collate_fn=collator,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=settings.batch_size,
        collate_fn=collator,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
    )
    optimizer = make_optimizer(model, settings)
    steps_per_epoch = math.ceil(len(train_loader) / settings.gradient_accumulation_steps)
    scheduler = make_scheduler(optimizer, steps_per_epoch * settings.epochs, settings)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    objective = HEADLINE[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable = {"encoder": 0, "dynamics": 0}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable["encoder" if name.startswith("encoder.") else "dynamics"] += parameter.numel()
    run = {
        "git": git_state(),
        "args": {key: str(value) for key, value in vars(args).items()},
        "training": asdict(settings),
        "architecture": asdict(architecture),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_dtype": str(dtype) if dtype is not None else None,
        "data": {
            "train_dialogues": len(train_dataset),
            "train_turns": len(train_frame),
            "dev_dialogues": len(dev_dataset),
            "dev_turns": len(dev_frame),
        },
        "trainable_parameters": trainable,
        "updates_per_epoch": steps_per_epoch,
        "total_updates": steps_per_epoch * settings.epochs,
        "objective": objective,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2))
    log_line({"run": run})
    if run["git"]["dirty"]:
        print("Warning: working tree has uncommitted changes", flush=True)
    run_start = time.perf_counter()
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    cast(Any, encoder.backbone).config.save_pretrained(args.output_dir / "encoder_config")
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, settings.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            settings,
            dtype=dtype,
            scaler=scaler,
            log=lambda row, epoch=epoch: log_line({"epoch": epoch, **row}),
        )
        dev_result = evaluate(model, dev_loader, criterion, device, dtype=dtype)
        score = dev_result[objective]
        improved = score > best_score + settings.min_delta
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": asdict(architecture),
                "training": asdict(settings),
                "model_name": args.model_name,
                "dataset": args.dataset,
                "epoch": epoch,
                "dev_metrics": dev_result,
            }
            temporary = args.output_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train": train_result,
            "dev": dev_result,
            "learning_rates": {
                group["group_name"]: group["lr"] for group in optimizer.param_groups
            },
            "best_epoch": best_epoch,
            "seconds": round(time.perf_counter() - epoch_start, 1),
            "peak_gpu_gb": (
                round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                if device.type == "cuda"
                else None
            ),
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        log_line(record)
        if settings.patience and stale_epochs >= settings.patience:
            break
    minutes = (time.perf_counter() - run_start) / 60
    print(f"Best dev {objective}: {best_score:.4f} at epoch {best_epoch} ({minutes:.1f} min)")


if __name__ == "__main__":
    main()
