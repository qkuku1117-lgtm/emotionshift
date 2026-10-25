#!/usr/bin/env bash
# Download the public datasets into data/raw/ (gitignored).
set -euo pipefail

DATA="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DATA/meld" "$DATA/dailydialog"

MELD=https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD
for split in train dev test; do
  curl -sfL -o "$DATA/meld/${split}_sent_emo.csv" "$MELD/${split}_sent_emo.csv"
done

# MELD-FR (SemEval-2024 Task 10, subtask C): emotion-flip triggers on MELD dialogues
EFR="https://raw.githubusercontent.com/LCS2-IIITD/EDiReF-SemEval2024/main/Data/Task%20C"
for split in train val test; do
  curl -sfL -o "$DATA/meld/MELD_${split}_efr.json" "$EFR/MELD_${split}_efr.json"
done

DD="https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full"
for split in train validation test; do
  curl -sfL -o "$DATA/dailydialog/$split.parquet" "$DD/$split/0000.parquet"
done

echo "Data written to $DATA"
