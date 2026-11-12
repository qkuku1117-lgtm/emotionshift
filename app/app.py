"""EmotionShift demo: browse test dialogues and compare model predictions.

Read-only: every label on screen comes from the prediction CSVs in predictions/;
no model is loaded. Models whose predictions are missing are simply not shown.

Run from the repo root:  streamlit run app/app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.eval.metrics import scores  # noqa: E402

PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"

DATASETS = {
    "MELD": {"key": "meld", "split": "test", "metric": "weighted_f1", "metric_name": "Weighted-F1"},
    "DailyDialog": {
        "key": "dailydialog",
        "split": "test_dedup",
        "metric": "micro_f1_no_neutral",
        "metric_name": "Micro-F1 (no neutral)",
    },
}
# Same emoji and colours as the video.
EMOJI = {
    "neutral": "😐",
    "joy": "😄",
    "surprise": "😮",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "disgust": "🤢",
}
COLOUR = {
    "neutral": "#8e929c",
    "joy": "#f5b414",
    "surprise": "#f58723",
    "sadness": "#3c8ceb",
    "anger": "#e8464c",
    "fear": "#9669e4",
    "disgust": "#32b47d",
}
SHIFT_TYPE = {
    "onset": "Onset",
    "offset": "Offset",
    "surprise": "Surprise",
    "within_negative": "Within negative",
    "valence_flip": "Valence flip",
}


def majority_version(key: str) -> tuple[str, str, int] | None:
    # Always predicts the most frequent training label; one run.
    return "b0_majority", "#1 Majority", 0


def inertia_version(key: str) -> tuple[str, str, int] | None:
    # Copies the speaker's previous predicted emotion; every seed gives the same labels.
    return "b0_inertia_pred", "#1 Inertia", 0


def b1_version(key: str) -> tuple[str, str, int] | None:
    # LinearSVC gives the same solution for every seed.
    return "b1", "#2 TF-IDF + SVM", 0


def b2_version(key: str) -> tuple[str, str, int] | None:
    selection = REPORTS / "b2" / key / "selection.json"
    if not selection.exists():
        return None
    arch = json.loads(selection.read_text())["selected_model"]
    summary = json.loads((REPORTS / "b2" / key / arch / "summary.json").read_text())
    seed = max(summary["dev"], key=lambda run: run["dev_score"])["seed"]
    return arch, f"#3 {'BiGRU' if arch == 'b2_bigru' else 'BiLSTM'}", seed


def b3_version(key: str) -> tuple[str, str, int] | None:
    summary = REPORTS / "b3" / key / "summary.json"
    if not summary.exists():
        return None
    return "b3", "#4 RoBERTa-base", json.loads(summary.read_text())["best_seed_on_dev"]


def b4_version(key: str) -> tuple[str, str, int] | None:
    # Window k chosen on MELD dev (reports/b4/selection.json); seed with the best dev score.
    summary = REPORTS / "b4" / key / "summary.json"
    if not summary.exists():
        return None
    dev = json.loads(summary.read_text())["dev"]
    seed = max(dev, key=lambda name: dev[name]["best"])
    return "b4", "#5 Context RoBERTa", int(seed.removeprefix("seed"))


def shiftaware_version(key: str) -> tuple[str, str, int] | None:
    # Variant and seed chosen on dev, as in reports/shiftaware/selection.md.
    metric = DATASET_METRIC[key]
    best = {}
    for variant in ("shiftaware", "shiftaware_cw-sqrt"):
        runs = [
            json.loads(p.read_text()) for p in (REPORTS / variant / key).glob("seed*/summary.json")
        ]
        if len(runs) >= 3:
            best[variant] = (
                sum(r["dev"][metric] for r in runs) / len(runs),
                max(runs, key=lambda r: r["dev"][metric])["seed"],
            )
    if not best:
        return None
    variant = max(best, key=lambda v: best[v][0])
    return variant, "#6 ShiftAware", best[variant][1]


# Add one function per model as its predictions are merged.
VERSIONS = [
    majority_version,
    inertia_version,
    b1_version,
    b2_version,
    b3_version,
    b4_version,
    shiftaware_version,
]
# History-only baselines: left out when picking dialogues where the models disagree.
BASELINES = {"#1 Majority", "#1 Inertia"}
SHIFTAWARE = "#6 ShiftAware"
DATASET_METRIC = {info["key"]: info["metric"] for info in DATASETS.values()}


@st.cache_data
def load_dataset(key: str, split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per turn, with a pred column per available model (dev-selected seed)."""
    base, seeds = None, {}
    for version in VERSIONS:
        found = version(key)
        if found is None:
            continue
        prefix, name, seed = found
        path = PREDICTIONS / f"{prefix}_{key}_{split}_seed{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, keep_default_na=False)
        if base is None:
            base = df[
                [
                    "dataset",
                    "split",
                    "dialogue_id",
                    "turn",
                    "speaker",
                    "text",
                    "emotion",
                    "stratum",
                    "shift_type",
                ]
            ].copy()
        else:
            same = ["dialogue_id", "turn", "emotion", "stratum"]
            if not base[same].equals(df[same]):
                raise ValueError(f"{path.name} is not aligned with the other predictions")
        base[name] = df["pred"].values
        if name == SHIFTAWARE and "shift_prob" in df:
            base["shift_prob"] = pd.to_numeric(df["shift_prob"]).values
        seeds[name] = seed
    return base, seeds


@st.cache_data
def scoreboard(key: str, split: str, metric: str) -> pd.DataFrame:
    df, seeds = load_dataset(key, split)
    rows = []
    for name, seed in seeds.items():
        row = {"Model": name, "Seed": seed, "All": scores(df.emotion, df[name])[metric]}
        for stratum in ("stable", "shift"):
            part = df[df.stratum == stratum]
            row[stratum.capitalize()] = scores(part.emotion, part[name])[metric]
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def dialogue_index(key: str, split: str) -> pd.DataFrame:
    df, seeds = load_dataset(key, split)
    names = [n for n in seeds if n not in BASELINES]
    rows = []
    for dialogue_id, part in df.groupby("dialogue_id", sort=True):
        shifts = part[part.stratum == "shift"]
        rows.append(
            {
                "dialogue_id": dialogue_id,
                "turns": len(part),
                "shifts": len(shifts),
                "disagreements": int((part[names].nunique(axis=1) > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def e(text: object) -> str:
    return html.escape(str(text), quote=True)


def chip(emotion: str, tip: str, extra: str = "") -> str:
    return f'<span class="chip {extra}" title="{e(tip)}">{EMOJI[emotion]}</span>'


TICK = (
    '<svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10.5l4 4 8-9" fill="none" '
    'stroke="#fff" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def soft(emotion: str) -> str:
    return f"color-mix(in srgb, {COLOUR[emotion]} 50%, white)"


def disagreement(part: pd.DataFrame, names: list[str]):
    """Turns where the shown learned models disagree (Majority would flag almost every turn)."""
    learned = [n for n in names if n not in BASELINES] or names
    return part[learned].nunique(axis=1).values > 1


def lanes_html(part: pd.DataFrame, names: list[str]) -> str:
    """One ribbon per speaker (it keeps the speaker's last emotion), then one row of predictions per model."""
    rows = [row for _, row in part.iterrows()]
    grid = f"grid-template-columns: 10rem repeat({len(rows)}, 3.3rem) 2rem;"
    cells = ['<div class="head"></div>']
    for row in rows:
        cls = "shift" if row.stratum == "shift" else ""
        cells.append(
            f'<div class="head {cls}" title="Turn {row.turn}"><span>{row.turn}</span></div>'
        )
    cells.append('<div class="head"></div>')

    for speaker in part.speaker.drop_duplicates():
        cells.append(f'<div class="label">{e(speaker)}</div>')
        held = None
        for row in rows:
            content = ""
            if row.speaker == speaker:
                new = soft(row.emotion)
                if held is None:
                    band = f"linear-gradient(90deg, transparent 50%, {new} 50%)"
                elif held == new:
                    band = new
                else:
                    band = f"linear-gradient(90deg, {held} 0%, {new} 50%)"
                held = new
                tip = f"{row.speaker}: {row.text} ({row.emotion})"
                content = chip(row.emotion, tip, "is-shift" if row.stratum == "shift" else "")
                inner = f'<i class="rib" style="background:{band}"></i>'
            else:
                inner = (
                    '<i class="track"></i>'
                    if held is None
                    else f'<i class="rib" style="background:{held}"></i>'
                )
            cells.append(f'<div class="cell lane">{inner}{content}</div>')
        cells.append(
            f'<div class="cell lane"><i class="rib" style="background:linear-gradient(90deg, {held}, transparent)">'
            "</i></div>"
        )

    risk = SHIFTAWARE in names and "shift_prob" in part
    last = len(names) - (0 if risk else 1)
    for j, name in enumerate(names):
        top, bottom = ("top" if j == 0 else ""), ("bottom" if j == last else "")
        cells.append(f'<div class="label model {top} {bottom}">{e(name)}</div>')
        for row in rows:
            pred = row[name]
            tip = f"{name} predicted {pred}; true emotion is {row.emotion}"
            if pred == row.emotion:
                mark = f'<span class="chip right" title="{e(tip)}">{TICK}</span>'
            else:
                mark = chip(pred, tip, "wrong")
            cells.append(
                f'<div class="cell model {top} {"col" if row.stratum == "shift" else ""}">{mark}</div>'
            )
        cells.append(f'<div class="cell model end {top} {bottom}"></div>')

    if risk:
        cells.append('<div class="label model bottom">#6 shift risk</div>')
        for row in rows:
            col = "col" if row.stratum == "shift" else ""
            if row.stratum == "first":  # no earlier turn by this speaker to shift from
                cells.append(
                    f'<div class="cell model {col}"><span class="risk none">&ndash;</span></div>'
                )
                continue
            p = float(row.shift_prob)
            tip = f"ShiftAware: {p:.0%} chance this turn is a shift ({'it is' if row.stratum == 'shift' else 'it is not'})"
            cells.append(
                f'<div class="cell model {col}" title="{e(tip)}"><span class="risk" style="--p:{p:.3f}">'
                f"<i></i><b>{round(100 * p)}</b></span></div>"
            )
        cells.append('<div class="cell model end bottom"></div>')
    return f'<div class="scroll"><div class="lanes" style="{grid}">{"".join(cells)}</div></div>'


def transcript_html(part: pd.DataFrame, names: list[str]) -> str:
    risk = SHIFTAWARE in names and "shift_prob" in part
    head = "".join(f'<th class="m">{e(n)}</th>' for n in names) + (
        '<th class="m">#6 shift risk</th>' if risk else ""
    )
    rows = []
    for _, row in part.iterrows():
        tag = ""
        if row.stratum == "shift":
            tag = f'<span class="tag">{e(SHIFT_TYPE.get(row.shift_type, "Shift"))}</span>'
        preds = "".join(
            f'<td class="m"><span class="mini {"ok" if row[n] == row.emotion else "bad"}" '
            f'title="{e(n)} predicted {e(row[n])}">{EMOJI[row[n]]}</span></td>'
            for n in names
        )
        if risk:
            preds += (
                '<td class="m">&ndash;</td>'
                if row.stratum == "first"
                else f'<td class="m">{row.shift_prob:.0%}</td>'
            )
        rows.append(
            f'<tr><td>{row.turn}</td><td>{e(row.speaker)}</td><td class="utt">{e(row.text)} {tag}</td>'
            f'<td class="true">{EMOJI[row.emotion]} {e(row.emotion)}</td>{preds}</tr>'
        )
    return (
        f'<div class="scroll"><table class="script"><thead><tr><th>#</th><th>Speaker</th><th>Utterance</th>'
        f"<th>True</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


CSS = """
<style>
.scroll { overflow-x: auto; padding-bottom: .4rem; }
.lanes { display: grid; align-items: stretch; width: max-content; }
.lanes .head { display: flex; align-items: center; justify-content: center; height: 2.2rem; font-size: .82rem;
               font-weight: 700; color: #636368; }
.lanes .head.shift span { display: inline-flex; align-items: center; justify-content: center; min-width: 1.7rem;
                          height: 1.7rem; border-radius: .85rem; background: #1d1d1f; color: #fff; }
.lanes .label { display: flex; align-items: center; font-size: 1rem; font-weight: 700; padding: 0 .8rem 0 .2rem;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lanes .cell { position: relative; display: flex; align-items: center; justify-content: center; height: 3.5rem; }
.lanes .cell.model, .lanes .label.model { background: #f5f5f7; height: 3.1rem; }
.lanes .label.model { font-size: .9rem; padding-left: .9rem; }
.lanes .cell.model.col { background: #e9e9ee; }
.lanes .top { margin-top: 1rem; }
.lanes .label.model.top { border-top-left-radius: 1rem; }
.lanes .label.model.bottom { border-bottom-left-radius: 1rem; }
.lanes .cell.end.top { border-top-right-radius: 1rem; }
.lanes .cell.end.bottom { border-bottom-right-radius: 1rem; }
.rib { position: absolute; left: 0; right: 0; top: 50%; height: 2.2rem; transform: translateY(-50%); }
.track { position: absolute; left: 0; right: 0; top: 50%; height: 2px; background: #dadae0; }
.chip { position: relative; z-index: 1; display: inline-flex; align-items: center; justify-content: center;
        box-sizing: border-box; width: 2.5rem; height: 2.5rem; border-radius: 50%; font-size: 1.45rem; line-height: 1;
        background: #fff; border: 1px solid #e3e3e8; box-shadow: 0 3px 9px rgba(20, 24, 40, .14); }
.chip.is-shift { width: 3rem; height: 3rem; font-size: 1.7rem; border: 3.5px solid #1d1d1f; }
.chip.right { width: 2.1rem; height: 2.1rem; background: #17805a; border-color: #17805a; box-shadow: none; }
.chip.wrong { width: 2.1rem; height: 2.1rem; font-size: 1.15rem; border: 2.5px dashed #c8323a; box-shadow: none; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem 1.4rem; font-size: .88rem; color: #636368;
          margin: .2rem 0 .8rem; }
.legend > span { display: inline-flex; align-items: center; gap: .45rem; }
.legend .chip, .legend .chip.is-shift { width: 1.6rem; height: 1.6rem; font-size: .9rem; }
.legend .chip.is-shift { border-width: 3px; }
.legend .chip svg { width: 12px; height: 12px; }
.legend .band { display: inline-block; width: 2.6rem; height: 1rem; border-radius: .5rem; }
table.script { border-collapse: collapse; font-size: .9rem; width: 100%; }
table.script th { text-align: left; color: #636368; font-weight: 600; border-bottom: 1px solid #d8d8de;
                  padding: .4rem .5rem; white-space: nowrap; vertical-align: bottom; }
table.script th.m { white-space: normal; text-align: center; font-size: .76rem; line-height: 1.2; min-width: 3.1rem;
                    max-width: 4.6rem; padding: .4rem .25rem; }
table.script td { border-bottom: 1px solid #ececf0; padding: .45rem .5rem; vertical-align: middle; }
table.script td.utt { min-width: 16rem; }
table.script td.true { white-space: nowrap; font-weight: 600; }
table.script td.m { text-align: center; color: #636368; padding: .45rem .25rem; }
.mini { display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; width: 2rem;
        height: 2rem; border-radius: 50%; font-size: 1.1rem; line-height: 1; }
.mini.ok { background: color-mix(in srgb, #17805a 16%, white); border: 2px solid #17805a; }
.mini.bad { background: #fff; border: 2px dashed #c8323a; }
.tag { display: inline-block; margin-left: .3rem; padding: .05rem .5rem; border-radius: .7rem; background: #1d1d1f;
       color: #fff; font-size: .72rem; font-weight: 600; white-space: nowrap; }
.risk { position: relative; display: inline-flex; align-items: flex-end; justify-content: center; width: 2.1rem;
        height: 2.1rem; border-radius: 6px; overflow: hidden; background: #fff; font-size: .74rem; color: #1d1d1f;
        padding-bottom: .1rem; }
.risk i { position: absolute; left: 0; right: 0; bottom: 0; height: calc(var(--p) * 100%);
          background: rgba(245, 135, 35, .5); }
.risk b { position: relative; font-weight: 700; }
.risk.none { background: transparent; color: #8e929c; align-items: center; }
</style>
"""

LEGEND = (
    '<div class="legend">'
    f'<span><i class="band" style="background:linear-gradient(90deg, {soft("joy")} 30%, {soft("anger")} 80%)"></i>'
    "Ribbon: the speaker's last emotion; a change of colour is a shift</span>"
    f"<span>{chip('anger', 'True emotion', 'is-shift')}Dark ring: the emotion changed here</span>"
    f'<span><span class="chip right">{TICK}</span>Model predicted the true emotion</span>'
    f"<span>{chip('anger', 'Wrong', 'wrong')}Model predicted this instead</span>"
    "</div>"
)


def main() -> None:
    st.set_page_config(page_title="EmotionShift Demo", page_icon="😮", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title("EmotionShift")
        dataset = st.radio("Dataset", list(DATASETS), horizontal=True, key="dataset")
        info = DATASETS[dataset]
        key, split = info["key"], info["split"]
        df, seeds = load_dataset(key, split)
        names = list(seeds)
        # Keyed per dataset: #3 has a different structure on each, so choices must not carry over.
        selected = st.multiselect("Models", names, default=names, key=f"models_{key}")

        index = dialogue_index(key, split)
        if st.checkbox("Only dialogues with shifts", value=True, key="only_shifts"):
            index = index[index.shifts > 0]
        labels = {
            int(
                r.dialogue_id
            ): f"#{r.dialogue_id} · {r.turns} turns · {r.shifts} shifts · {r.disagreements} disagree"
            for r in index.itertuples()
        }
        # Open on a dialogue short enough to read that has plenty of shifts and disagreement.
        short = index[index.turns <= 14]
        opening = (
            short.assign(rank=short.shifts + short.disagreements)
            .sort_values("rank")
            .dialogue_id.iloc[-1]
        )
        dialogue_id = st.selectbox(
            "Test dialogue",
            list(labels),
            index=list(labels).index(opening),
            format_func=labels.get,
            key=f"dialogue_{key}",
        )
        st.caption(
            "All labels come from saved prediction files. "
            "Each model uses its dev-selected structure and best dev seed."
        )

    part = df[df.dialogue_id == dialogue_id].sort_values("turn")
    shown = selected or names

    st.subheader(f"{dataset} test dialogue #{dialogue_id}")
    a, b, c = st.columns(3)
    a.metric("Turns", len(part))
    b.metric("True shifts", int((part.stratum == "shift").sum()))
    c.metric("Turns where trained models disagree", int(disagreement(part, shown).sum()))

    st.markdown("##### Emotion ribbons")
    st.markdown(LEGEND, unsafe_allow_html=True)
    st.markdown(lanes_html(part, shown), unsafe_allow_html=True)

    st.markdown("##### Shifts caught in this dialogue")
    shift_rows = part[part.stratum == "shift"]
    columns = st.columns(len(shown))
    for column, name in zip(columns, shown, strict=True):
        caught = int((shift_rows[name] == shift_rows.emotion).sum())
        column.metric(name, f"{caught} / {len(shift_rows)}")
    st.caption(
        "A shift counts as caught when the model predicts the new true emotion on that turn."
    )

    st.markdown("##### Transcript")
    st.markdown(transcript_html(part, shown), unsafe_allow_html=True)

    st.markdown(f"##### Whole test set: {info['metric_name']}")
    board = scoreboard(key, split, info["metric"])
    board = board[board.Model.isin(shown)]
    board[["All", "Stable", "Shift"]] = (board[["All", "Stable", "Shift"]] * 100).round(1)
    st.dataframe(board, hide_index=True, width="stretch")
    st.caption("Computed live from the single checkpoint shown here, not the 5-seed mean.")


if __name__ == "__main__":
    main()
