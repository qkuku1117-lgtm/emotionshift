"""Optimizer and update loop for supervised ShiftAware training."""

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .loss import ShiftAwareLoss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    encoder_lr: float = 2e-5
    dynamics_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 3
    min_delta: float = 0.0
    max_length: int = 128
    num_workers: int = 0
    seed: int = 42
    log_every: int = 20
    amp: str = "auto"
    freeze_encoder: bool = False
    shift_weight: float = 1.0
    forecast_weight: float = 0.5

    def __post_init__(self):
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("patience", "num_workers", "seed", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("encoder_lr", "dynamics_lr", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_decay",
            "warmup_ratio",
            "min_lr_ratio",
            "min_delta",
            "shift_weight",
            "forecast_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.warmup_ratio >= 1 or self.min_lr_ratio > 1:
            raise ValueError("warmup_ratio must be < 1 and min_lr_ratio must be <= 1")
        if self.amp not in ("auto", "none", "bf16", "fp16"):
            raise ValueError("amp must be auto, none, bf16, or fp16")
        if not isinstance(self.freeze_encoder, bool):
            raise ValueError("freeze_encoder must be boolean")


def make_optimizer(model: torch.nn.Module, config: TrainingConfig) -> AdamW:
    """Use a lower encoder LR and exclude bias/norm vectors from decay."""
    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            part = "encoder" if name.startswith("encoder.") else "dynamics"
            groups.setdefault((part, parameter.ndim > 1), []).append(parameter)
    if not groups:
        raise ValueError("No trainable parameters")
    parameter_groups = [
        {
            "params": parameters,
            "group_name": f"{part}_{'decay' if decay else 'no_decay'}",
            "lr": config.encoder_lr if part == "encoder" else config.dynamics_lr,
            "weight_decay": config.weight_decay if decay else 0.0,
        }
        for (part, decay), parameters in groups.items()
    ]
    return AdamW(parameter_groups)


def make_scheduler(optimizer: AdamW, total_steps: int, config: TrainingConfig) -> LambdaLR:
    """Step after each optimizer update, including accumulated updates."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(int(total_steps * config.warmup_ratio), total_steps - 1)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return (
            config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )

    return LambdaLR(optimizer, multiplier)


def autocast_dtype(device: torch.device, mode: str) -> torch.dtype | None:
    if mode == "none":
        return None
    if device.type != "cuda":
        if mode != "auto":
            raise ValueError("bf16 and fp16 AMP require CUDA")
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 AMP is unsupported on this CUDA device")
    return torch.bfloat16 if mode == "bf16" else torch.float16


def move_batch(batch: dict, device: torch.device) -> dict:
    """Transfer tensor fields without moving metadata to the accelerator."""
    return {
        key: (
            {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
            if key == "tokens"
            else value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def train_epoch(
    model: torch.nn.Module,
    loader,
    criterion: ShiftAwareLoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    config: TrainingConfig,
    *,
    dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    log: Callable[[dict], None] | None = None,
) -> dict[str, float | int]:
    """Run one epoch; every `config.log_every` updates, pass running stats to `log`."""
    if len(loader) == 0:
        raise ValueError("Training loader must be nonempty")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    turn_count = 0
    updates = 0
    total_updates = math.ceil(len(loader) / config.gradient_accumulation_steps)
    window_loss = 0.0
    window_turns = 0
    start = time.perf_counter()
    for index, cpu_batch in enumerate(loader):
        batch = move_batch(cpu_batch, device)
        group_start = (
            index // config.gradient_accumulation_steps
        ) * config.gradient_accumulation_steps
        group_size = min(config.gradient_accumulation_steps, len(loader) - group_start)
        context = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            output = model(
                batch["tokens"], batch["speaker_ids"], batch["turn_mask"], batch["turn_gaps"]
            )
            loss = criterion(output, batch["labels"], batch["shift_labels"], batch["shift_mask"])[
                "total"
            ]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {index}")
        (scaler.scale(loss / group_size) if scaler is not None else loss / group_size).backward()
        n = int(batch["turn_mask"].sum())
        loss_sum += loss.detach().item() * n
        turn_count += n
        window_loss += loss.detach().item() * n
        window_turns += n
        if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if scaler is None:
                optimizer.step()
                stepped = True
            else:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= old_scale
            if stepped:
                scheduler.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)
            update = index // config.gradient_accumulation_steps + 1
            if (
                log is not None
                and config.log_every
                and (update % config.log_every == 0 or update == total_updates)
            ):
                log(
                    {
                        "update": update,
                        "of": total_updates,
                        "loss": window_loss / max(1, window_turns),
                        "lr": {
                            group["group_name"].split("_")[0]: group["lr"]
                            for group in optimizer.param_groups
                        },
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                )
                window_loss, window_turns = 0.0, 0
    return {
        "loss": loss_sum / turn_count,
        "updates": updates,
        "seconds": round(time.perf_counter() - start, 1),
    }


def load_trained_model(directory: str | Path, *, device: str | torch.device = "cpu"):
    """Reload a best checkpoint and tokenizer from a training output directory."""
    from .config import ShiftAwareConfig
    from .encoder import TextShiftAware, TransformerUtteranceEncoder
    from .model import ShiftAware

    directory = Path(directory)
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise ImportError("Install the text encoder with: uv sync --extra training") from exc
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    architecture = ShiftAwareConfig(**checkpoint["architecture"])
    encoder_config = transformers.AutoConfig.from_pretrained(
        directory / "encoder_config", local_files_only=True
    )
    encoder = TransformerUtteranceEncoder(
        transformers.AutoModel.from_config(encoder_config), architecture.input_dim, frozen=True
    )
    model = TextShiftAware(encoder, ShiftAware(architecture))
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory / "tokenizer", local_files_only=True
    )
    return model.to(device).eval(), tokenizer
