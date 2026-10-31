"""Exercise actual ShiftAware optimization offline; this is not an accuracy benchmark."""

import argparse
import json

import torch

from emotionshift.models.shiftaware import ShiftAware, ShiftAwareConfig, ShiftAwareLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--global-memory", choices=("gated_delta", "gru"), default="gated_delta")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.manual_seed(42)
    device = torch.device(args.device)
    config = ShiftAwareConfig(
        input_dim=24,
        hidden_dim=32,
        time_dim=8,
        num_heads=4,
        dropout=0,
        global_memory=args.global_memory,
    )
    model = ShiftAware(config).to(device)
    criterion = ShiftAwareLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    embeddings = torch.randn(2, 5, config.input_dim, device=device)
    speakers = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, -1, -1]], device=device)
    labels = torch.tensor([[1, 0, 0, 4, 3], [0, 1, 3, -100, -100]], device=device)
    shifts = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 0, 0]], dtype=torch.float, device=device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(embeddings, speakers)
        losses = criterion(output, labels, shifts)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("Non-finite training loss")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(json.dumps({"step": step + 1, **{k: v.item() for k, v in losses.items()}}))
    print(f"Optimized {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")


if __name__ == "__main__":
    main()
