"""Build notebooks/b3_roberta.ipynb (run once; then execute it on a GPU, e.g. Colab T4)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# B3 · RoBERTa fine-tuned on single utterances

**EE6405 group project · EmotionShift · workstream #4**

This notebook walks through the strong no-context transformer baseline:

1. Load MELD and DailyDialog with the shared data rules
2. Fine-tune RoBERTa-base on one utterance at a time (5 seeds per dataset, best epoch chosen on **dev**)
3. Score once on **test**, overall and split by emotion shift, and compare with B1 (TF-IDF)
4. Look inside the model with Integrated Gradients
5. Transfer (MELD → DailyDialog) and efficiency for the demo

Like B1, B3 reads **one utterance at a time** (no dialogue context, no speaker). B4 (context) and
ShiftAware (#6) build on it: B4 can start from its weights, ShiftAware runs on its cached embeddings.
Needs a GPU: on a Colab T4 the full run takes about 70 minutes."""),
code("""# Colab only: every output goes straight to Google Drive, so an interrupted run resumes where it stopped
# (finished seeds are reused) and the models (~500 MB each) are shared with B4 / ShiftAware / the demo.
import sys
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")"""),
code("""# Setup. On Colab: upload emotionshift.zip (or let it clone the team repo), install, download the data.
import os, sys, subprocess
from pathlib import Path
if "google.colab" in sys.modules:
    if not Path("/content/emotionshift").exists():
        if Path("/content/emotionshift.zip").exists():
            subprocess.run(["unzip", "-q", "/content/emotionshift.zip", "-d", "/content/emotionshift"], check=True)
        else:
            subprocess.run(["git", "clone", "-q", "https://github.com/qkuku1117-lgtm/emotionshift", "/content/emotionshift"], check=True)
    os.chdir("/content/emotionshift")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[transformers]", "pyyaml"], check=True)
    if not Path("data/raw/meld").exists():
        subprocess.run(["bash", "scripts/fetch_data.sh"], check=True)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
OUT = Path("/content/drive/MyDrive/EE6405_EmotionShift/b3_out") if "google.colab" in sys.modules else ROOT

import json
import numpy as np, pandas as pd, torch
from IPython.display import Image, HTML, display
from emotionshift.data.labels import EMOTIONS
from emotionshift.data.loaders import load
from emotionshift.eval import metrics as M
from emotionshift.models.roberta import RobertaConfig, RobertaUtterance

FIG = OUT / "reports" / "b3" / "figures"
pd.set_option("display.precision", 3)
print("device:", "cuda " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")"""),

md("""## 1. Data

Same table, splits, labels and shift definition as every other model (`emotionshift/data/`).
A **shift** = the speaker's emotion differs from their own previous utterance; their first turn is neither.
DailyDialog headline test set = `test_dedup`."""),
code("""meld, dd = load("meld"), load("dailydialog")
pd.DataFrame({"MELD": meld.groupby("split").size(), "DailyDialog": dd.groupby("split").size()}
             ).reindex(["train", "dev", "test", "test_dedup"])"""),
code("""t = meld[meld.split == "test"]
print("MELD test strata:", t.stratum.value_counts().to_dict())
print("MELD test shift types:", t.shift_type.value_counts().to_dict())
t[t.dialogue_id == 3][["turn", "speaker", "text", "emotion", "stratum", "shift_type"]]"""),

md(r"""## 2. Method

**RoBERTa-base** (125M parameters, 12 layers) is a BERT-style encoder pre-trained on 160 GB of English text.
We add a linear classification head on the first token `<s>` and fine-tune everything on one utterance at a time:

$$p(e \mid u) = \mathrm{softmax}\big(W\,h_{\langle s\rangle}(u) + b\big), \qquad
\mathcal{L} = -\log p(e^\ast \mid u)$$

* Byte-level BPE tokenizer, so slang, names and "Ohhh!" never become unknown tokens
* AdamW, learning rate 1e-5, 10% linear warm-up, batch 32, fp16; 4 epochs on MELD, 3 on DailyDialog
* After every epoch the model is scored on **dev**; the best epoch is kept (MELD: weighted-F1,
  DailyDialog: micro-F1 excluding neutral). Test is used once per seed.
* 5 seeds per dataset; the best-on-dev seed is saved for the demo, B4 and ShiftAware

How the tokenizer sees an utterance:"""),
code("""tok = RobertaUtterance(RobertaConfig()).tokenizer
for s in ["Oh my God! You got the job?!", "Ohhh, noooo..."]:
    print(s, "->", tok.tokenize(s))"""),
md("""The core of the training code (`emotionshift/models/roberta.py`, `RobertaUtterance.fit`): Hugging Face
`Trainer`, scored on dev after every epoch, best epoch restored at the end."""),
code("""import inspect
from emotionshift.models import roberta
src = inspect.getsource(roberta.RobertaUtterance.fit).splitlines()
print("\\n".join(src[14:25]))"""),

md("""## 3. Train: 5 seeds × 2 datasets

`scripts/train_b3.py` runs the whole protocol and writes the shared prediction CSVs, the reports and the saved
models (with cached `<s>` embeddings for ShiftAware). About 15 minutes for MELD and 55 minutes for DailyDialog on a T4.
If the run is interrupted, run this cell again: finished seeds are reused, only the rest are trained."""),
code("""!python scripts/train_b3.py --embeddings --out "{OUT}" 2>&1 | grep -E "^\\[|^\\|"  """),

md("## 4. Results, scored once on test"),
code("""print(open(OUT / "reports/b3/results.md").read())"""),
code("""rows = []
for model, name in [("b1", "B1 TF-IDF"), ("b3", "B3 RoBERTa")]:
    for ds, split, metric in [("meld", "test", "weighted_f1"), ("dailydialog", "test_dedup", "micro_f1_no_neutral")]:
        base = OUT if model == "b3" else ROOT
        s = json.loads((base / f"reports/{model}/{ds}/summary.json").read_text())["test"][split]
        rows.append({"model": name, "dataset": f"{ds} ({split})", "headline metric": metric,
                     "score": f"{100*s[metric]['mean']:.2f} ± {100*s[metric]['std']:.2f}",
                     "macro-F1": round(100 * s["macro_f1"]["mean"], 2),
                     "derived shift F1": round(100 * s["derived_shift_detection"]["shift_f1"], 2)})
pd.DataFrame(rows)"""),
md("""For context on MELD (weighted-F1): always-neutral 31.3, copying the speaker's previous gold emotion 44.9,
published context models DialogueGCN 58.1, COSMIC 65.2, EmoBERTa (RoBERTa-*large*, with context) 64.6."""),

md("## 5. Where does B3 fail? Shift vs stable turns"),
code("""!python scripts/analyse_b3.py --out "{OUT}" 2>&1 | tail -8
display(Image(str(FIG / "meld_b1_vs_b3.png"), width=1000))"""),
code("""strat = pd.read_csv(FIG / "meld_strata.csv")
strat.assign(weighted_f1=100 * strat.weighted_f1, macro_f1=100 * strat.macro_f1)[["group", "n", "weighted_f1", "macro_f1"]]"""),
code("""display(Image(str(FIG / "meld_confusion.png"), width=520))
display(Image(str(FIG / "meld_length.png"), width=620))"""),
code("""preds = [pd.read_csv(f) for f in sorted((OUT / "predictions").glob("b3_meld_test_seed*.csv"))]
print("Derived shift detection (predicted label vs the speaker's previous predicted label), first seed:")
M.derived_shift_detection(preds[0])"""),

md("""## 6. Inside the model: Integrated Gradients

For one utterance, each word's embedding is faded in from the padding embedding (32 steps); a word's
attribution is how much the predicted emotion's score rises along the way. The attributions add up to
the change in score, so unlike attention weights they are faithful to the model."""),
code("""display(Image(str(FIG / "meld_ig_top_words.png"), width=1000))
display(HTML((FIG / "meld_ig_examples.html").read_text()))"""),

md("## 7. Try it"),
code("""m = RobertaUtterance.load(OUT / "models/b3_meld")
lines = ["Oh my God! You got the job?!", "Yeah.", "I'm so sorry about your dad.",
         "Get out of my apartment!", "Ew, what is that smell?", "I'm scared, what if he finds out?"]
p = m.predict_proba(lines)
pd.DataFrame({"utterance": lines, "B3 says": m.predict(lines), "confidence": p.max(1).round(2)})"""),

md("## 8. Transfer: MELD → DailyDialog (E8)"),
code("""pd.read_csv(FIG / "transfer_dailydialog.csv")"""),

md("""## 9. Efficiency (E9)

The demo needs < 300 ms per utterance on a laptop CPU. Latency is one utterance per call."""),
code("""rows = []
for ds in ["meld", "dailydialog"]:
    e = json.loads((OUT / f"reports/b3/{ds}/summary.json").read_text())["efficiency"]
    rows.append({"trained on": ds, "parameters (M)": round(e["parameters"] / 1e6, 1), "model files (MB)": e["model_dir_mb"],
                 "train min / seed": round(e["train_seconds"]["mean"] / 60, 1),
                 "CPU median ms": round(e["latency_one_utterance_cpu"]["median_ms"], 1),
                 "CPU p95 ms": round(e["latency_one_utterance_cpu"]["p95_ms"], 1),
                 "GPU utt/s": round(e["batch_throughput"]["utterances_per_second"])})
pd.DataFrame(rows)"""),

md("## Takeaways\n\n(filled in after the run)"),
code("""# Colab only: one zip of the results for the repo (the models stay in the Drive folder for the team).
if "google.colab" in sys.modules:
    subprocess.run(f'cd "{OUT}" && zip -qr b3_results.zip predictions reports', shell=True, check=True)
    print("saved", OUT / "b3_results.zip", "| models:", sorted(p.name for p in (OUT / "models").iterdir()))"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
out = ROOT / "notebooks" / "b3_roberta.ipynb"
nbf.write(nb, out)
print("wrote", out)
