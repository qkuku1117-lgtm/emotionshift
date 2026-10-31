#!/usr/bin/env bash
# Download the public datasets into data/raw/ (gitignored).
set -euo pipefail

DATA="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DATA/meld" "$DATA/dailydialog" "$DATA/emowoz"

MELD=https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD
for split in train dev test; do
  curl -sfL -o "$DATA/meld/${split}_sent_emo.csv" "$MELD/${split}_sent_emo.csv"
done

# MELD-FR (SemEval-2024 Task 10, subtask C): emotion-flip triggers on MELD dialogues
EFR="https://raw.githubusercontent.com/LCS2-IIITD/EDiReF-SemEval2024/main/Data/Task%20C"
for split in train val test; do
  curl -sfL -o "$DATA/meld/MELD_${split}_efr.json" "$EFR/MELD_${split}_efr.json"
done

# DailyDialog JSON-lines source shared by every model and notebook.
task_tmp=$(mktemp -d)
trap 'rm -rf "$task_tmp"' EXIT
curl -fsSL https://raw.githubusercontent.com/declare-lab/conv-emotion/master/DialogueRNN/dailydialog.zip -o "$task_tmp/dd.zip"
python3 - "$task_tmp/dd.zip" "$DATA/dailydialog" <<'EOF'
from pathlib import Path
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    for split in ["train", "valid", "test"]:
        (Path(sys.argv[2]) / f"{split}.json").write_bytes(
            archive.read(f"dailydialog/{split}.json")
        )
EOF

# EmoWOZ 1.0 (Feng et al., 2022): MultiWOZ and DialMAGE dialogues plus the official split.
EMOWOZ=https://zenodo.org/record/6506504/files
for file in emowoz-multiwoz.json emowoz-dialmage.json data-split.json; do
  curl -sfL -o "$DATA/emowoz/$file" "$EMOWOZ/$file"
done

echo "Data written to $DATA"
