"""Native EmoWOZ B3: five seeds, dev-only epoch selection, resumable checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
import yaml
from huggingface_hub import snapshot_download

from emotionshift.data.labels import emotions
from emotionshift.eval import metrics as M
from emotionshift.models.roberta import RobertaConfig, RobertaUtterance
from scripts.train_b0 import PRED_COLS, dialogue_intervals
from scripts.train_emowoz_baselines import digest, targets, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b3/emowoz"
LABELS = emotions("emowoz")


def provenance(cfg):
    source = [
        "scripts/train_emowoz_roberta.py",
        "emotionshift/models/roberta.py",
        "emotionshift/data/loaders.py",
        "emotionshift/data/labels.py",
        "emotionshift/eval/metrics.py",
        "scripts/train_emowoz_baselines.py",
        "scripts/train_b0.py",
    ]
    raw = ROOT / "data/raw/emowoz"
    return {
        "config": cfg,
        "labels": LABELS,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "source_sha256_lf": {p: digest(ROOT / p) for p in source},
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw.glob("*.json"))
            if p.name != "download_manifest.json"
        },
    }


def evaluate(cfg):
    rows = []
    test = targets(["test"])
    strata = []
    for seed in cfg["seeds"]:
        model = RobertaUtterance.load(ROOT / f"models/b3_emowoz/seed{seed}", device="cuda")
        frame = test.copy()
        assert model.labels == LABELS
        frame["pred"] = [
            LABELS[i] for i in model.predict_proba(frame.text, batch_size=8).argmax(axis=1)
        ]
        frame["model"], frame["seed"] = "b3", seed
        extra = [c for c in frame if c not in PRED_COLS]
        path = ROOT / f"predictions/b3_emowoz_test_seed{seed}.csv"
        frame[PRED_COLS + extra].to_csv(path, index=False)
        row = dict(
            seed=seed,
            **M.scores(frame.emotion, frame.pred, LABELS),
            derived_shift_f1=M.derived_shift_detection(frame)["shift_f1"],
        )
        rows.append(row)
        strata.append(M.stratified(frame, labels=LABELS).assign(seed=seed))
        M.per_class_f1(frame.emotion, frame.pred, LABELS).to_csv(
            OUT / f"per_class_test_seed{seed}.csv", index=False
        )
        M.confusion(frame.emotion, frame.pred, LABELS).to_csv(
            OUT / f"confusion_test_seed{seed}.csv"
        )
        if seed == 0:
            intervals = dialogue_intervals(frame, LABELS, n_boot=cfg["bootstrap"])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "per_seed.csv", index=False)
    strat = pd.concat(strata, ignore_index=True)
    strat.to_csv(OUT / "stratified_test_per_seed.csv", index=False)
    strat.drop(columns="seed").groupby("group", sort=False).mean().reset_index().to_csv(
        OUT / "stratified_test.csv", index=False
    )
    metrics = {
        k: {
            "mean": float(scores[k].mean()),
            "std": float(scores[k].std(ddof=0)),
            "ci95_seed0": intervals[k],
        }
        for k in M.METRICS
    }
    summary = {
        "dataset": "emowoz",
        "model": "b3",
        "labels": LABELS,
        "seeds": cfg["seeds"],
        "headline_metric": M.HEADLINE["emowoz"],
        "test": {
            "n": len(test),
            "metrics": metrics,
            "derived_shift_f1_mean": float(scores.derived_shift_f1.mean()),
        },
    }
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/emowoz_b3.yaml").read_text(encoding="utf-8"))
    assert torch.cuda.is_available(), "This full run requires CUDA"
    torch.set_num_threads(cfg["threads"])
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = provenance(cfg)
    sig = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    old = OUT / "manifest.json"
    if old.exists():
        assert json.loads(old.read_text()) == manifest, "Protocol changed; do not mix runs"
    write(old, manifest)
    if not args.evaluate_only:
        free, _ = torch.cuda.mem_get_info()
        if free < 3_500_000_000:
            raise RuntimeError(
                "Less than 3.5 GB free GPU memory; close competing GPU applications first"
            )
        snapshot = snapshot_download(
            cfg["pretrained_repo"],
            revision=cfg["pretrained_revision"],
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ],
        )
        train, dev = targets(["train"]), targets(["dev"])
        for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
            assert seed in cfg["seeds"]
            state = OUT / f"training_seed{seed}.json"
            checkpoint = ROOT / f"models/b3_emowoz/seed{seed}"
            if state.exists():
                assert json.loads(state.read_text())["protocol_signature"] == sig
                assert (checkpoint / "model.safetensors").exists()
                print(f"Resume completed seed {seed}", flush=True)
                continue
            runtime = ROOT / f"checkpoints/b3_emowoz/seed{seed}"
            signature = runtime / "protocol.json"
            if signature.exists():
                assert json.loads(signature.read_text())["signature"] == sig, (
                    "Partial checkpoint protocol changed"
                )
            write(signature, {"signature": sig})
            start = time.perf_counter()
            model = RobertaUtterance(
                RobertaConfig(
                    **cfg["model"], model_name=str(snapshot), seed=seed, checkpoint_dir=str(runtime)
                ),
                device="cuda",
                labels=LABELS,
            )
            print(f"Training native EmoWOZ RoBERTa seed {seed}", flush=True)
            model.fit(train.text, train.emotion, dev.text, dev.emotion, log_every=25)
            model.save(checkpoint)
            best = max(model.history, key=lambda h: h[cfg["model"]["select_on"]])
            pd.DataFrame(model.history).to_csv(OUT / f"history_seed{seed}.csv", index=False)
            write(
                state,
                {
                    "seed": seed,
                    "dev_score": best[cfg["model"]["select_on"]],
                    "best_epoch": best["epoch"],
                    "train_seconds_this_invocation": time.perf_counter() - start,
                    "protocol_signature": sig,
                },
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    states = [OUT / f"training_seed{s}.json" for s in cfg["seeds"]]
    if all(p.exists() for p in states):
        dev_rows = [json.loads(p.read_text()) for p in states]
        selected = max(dev_rows, key=lambda row: row["dev_score"])
        write(
            OUT / "selection.json",
            {
                "best_seed_on_dev": selected["seed"],
                "metric": cfg["model"]["select_on"],
                "test_used_for_selection": False,
            },
        )
        evaluate(cfg)
    else:
        print("Saved completed seeds; test evaluation waits for all five seeds.", flush=True)


if __name__ == "__main__":
    main()
