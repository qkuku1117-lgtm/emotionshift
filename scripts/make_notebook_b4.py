"""Build notebooks/b4_context.ipynb with its outputs from the files under reports/ (CPU, seconds; no model is loaded).

    python scripts/make_notebook_b4.py

The cells are executed here by a small runner (no Jupyter needed) and their outputs are stored in the notebook, so it
opens already executed in Colab or Jupyter, where "Run all" reproduces the same outputs.
"""

import ast
import base64
import contextlib
import io
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.strip(), "outputs": [], "execution_count": None}


cells = [
    md("""
# B4 · RoBERTa with context (workstream #5)

The same RoBERTa-base as B3, but each utterance is read together with the previous *k* turns of its dialogue and their speaker
names. This notebook shows the input, then reads the finished runs in `reports/` and `predictions/` (it trains nothing).
Write-up: `docs/B4_CONTEXT.md`. Training: `python scripts/train_b4.py` on a GPU (about 1.5 / 9 / 10 minutes per seed on an A100
for MELD / DailyDialog / EmoWOZ); tables and figures: `python scripts/analyse_b4_ctx.py --compare b4all` (CPU, seconds).
"""),
    code("""
import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd() if (Path.cwd() / "emotionshift").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
pd.set_option("display.width", 200)
"""),
    md("## 1. The input: the last k turns with speaker names (code shown in the video)"),
    code("""
from emotionshift.data.context import dialogue_inputs

dialogue = [("Joey", "Hey, you got the job?"), ("Rachel", "I did not."), ("Joey", "Oh. I am sorry."), ("Rachel", "It is fine, really!")]
for k in (0, 2, None):                       # 0 = B3's input, None = the whole past
    print(f"--- k = {'all' if k is None else k}: input for the last turn")
    print(dialogue_inputs(dialogue, k)[-1])
"""),
    md("## 2. Headline: k = 10 against k = all, 5 seeds (test, %)"),
    code("""
from emotionshift.eval import metrics as M

rows = []
for ds in ["meld", "dailydialog", "emowoz"]:
    for tag in ["b4", "b4all"]:
        s = json.loads((REPORTS / tag / ds / "summary.json").read_text())
        split = next(iter(s["test"])); h = M.HEADLINE[ds]; t = s["test"][split]
        rows.append({"dataset": ds, "split": split, "k": s["chosen_k"], "metric": h,
                     "mean": round(100 * t[h]["mean"], 2), "sd": round(100 * t[h]["std"], 2),
                     "macro_f1": round(100 * t["macro_f1"]["mean"], 2), "seeds": len(s["seeds"])})
pd.DataFrame(rows)
"""),
    md("## 3. Window sweep (E3): dev scores, and test by stratum"),
    code("""
for ds in ["meld", "emowoz"]:
    s = json.loads((REPORTS / "b4" / ds / "summary.json").read_text())
    print(ds, "dev", s["selected_on_dev"], "->", {r["k"]: round(100 * r["dev_mean"], 2) for r in s["sweep"]})
display(Image(str(REPORTS / "b4" / "figures" / "window_sweep_test.png"), width=900))
"""),
    md("## 4. Where the gain comes from (E2 / H1): B4 (k = 10) minus B3"),
    code("""
for ds in ["meld", "dailydialog", "emowoz"]:
    g = pd.read_csv(REPORTS / "b4" / ds / "context_gain.csv")
    g = g[g.n >= 30].assign(**{c: (100 * g[c]).round(2) for c in ["ref", "b4", "delta", "ci_low", "ci_high"]})
    print(ds); display(g)
display(Image(str(REPORTS / "b4" / "figures" / "context_gain_emowoz.png"), width=700))
"""),
    md("## 5. Input controls (E4 / H2): the trained models with a changed input, 5 seeds"),
    code("""
for tag, ds in [("b4", "meld"), ("b4", "dailydialog"), ("b4all", "emowoz")]:
    c = pd.read_csv(REPORTS / tag / ds / "controls.csv")
    t = c.pivot(index="input", columns="group", values="mean").mul(100).round(2).loc[["real", "none", "shuffled_context", "shuffled_speakers"]]
    print(tag, ds, c.metric.iloc[0]); display(t)
"""),
    md("## 6. B4+ (class-weighted) for R3, and efficiency (E9)"),
    code("""
b3 = json.loads((REPORTS / "b3" / "meld" / "summary.json").read_text())["test"]["test"]["macro_f1"]["mean"]
for tag in ["b4plus", "b4allplus"]:
    s = json.loads((REPORTS / tag / "meld" / "summary.json").read_text()); t = s["test"]["test"]
    print(f"B4+ k={s['chosen_k']}: macro-F1 {100 * t['macro_f1']['mean']:.2f} (B3 {100 * b3:.2f}), weighted-F1 {100 * t['weighted_f1']['mean']:.2f}")
eff = []
for ds in ["meld", "dailydialog", "emowoz"]:
    e = json.loads((REPORTS / "b4all" / ds / "summary.json").read_text())["efficiency"]
    eff.append({"dataset": ds, "parameters (M)": round(e["parameters"] / 1e6, 1), "train min / seed (A100)": round(e["train_seconds"]["mean"] / 60, 1),
                "CPU median ms": round(e["latency_one_utterance_cpu"]["median_ms"], 1), "CPU p95 ms": round(e["latency_one_utterance_cpu"]["p95_ms"], 1)})
pd.DataFrame(eff)
"""),
    md("""
## Reading the results

- Context helps on all three datasets, mostly on **stable** turns; shift turns gain little (EmoWOZ) or nothing (MELD, DailyDialog).
- **k = 10 and k = all are within noise** on every dataset: once the window is large enough, its size does not matter.
- A shuffled history is worse than none on MELD and DailyDialog, so the model uses the right dialogue (H2).
- B4+ trades weighted-F1 for macro-F1 and beats B3 on MELD macro-F1 (R3).

Caveats are listed in `docs/B4_CONTEXT.md`.
"""),
]



def rich(obj) -> dict:
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        data["text/html"] = obj._repr_html_()
    if hasattr(obj, "png"):
        data = {"image/png": obj.png, "text/plain": "<image>"}
    return data


def run(cells: list[dict]) -> None:
    """Execute the code cells in one namespace and store stdout, display() calls and the last expression."""
    shown: list[dict] = []

    class Image:
        def __init__(self, filename, width=None):
            self.png = base64.b64encode(Path(filename).read_bytes()).decode()
            self.meta = {"image/png": {"width": width}} if width else {}

    def display(*objs):
        for o in objs:
            shown.append({"output_type": "display_data", "data": rich(o), "metadata": getattr(o, "meta", {})})

    shim = types.ModuleType("IPython.display")
    shim.Image, shim.display = Image, display
    sys.modules.setdefault("IPython", types.ModuleType("IPython"))
    sys.modules["IPython.display"] = shim
    ns: dict = {"__name__": "__main__"}
    import os
    os.chdir(ROOT / "notebooks")
    for n, cell in enumerate([c for c in cells if c["cell_type"] == "code"], 1):
        tree = ast.parse(cell["source"])
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        out = io.StringIO()
        shown.clear()
        with contextlib.redirect_stdout(out):
            exec(compile(tree, f"<cell {n}>", "exec"), ns)
            value = eval(compile(ast.Expression(last.value), f"<cell {n}>", "eval"), ns) if last else None
        # stdout and display() calls are stored in order of kind (stream first), which is how these cells use them
        if out.getvalue():
            cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": out.getvalue()})
        cell["outputs"] += list(shown)
        if value is not None:
            cell["outputs"].append({"output_type": "execute_result", "data": rich(value), "metadata": {}, "execution_count": n})
        cell["execution_count"] = n


run(cells)
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
out = ROOT / "notebooks" / "b4_context.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
