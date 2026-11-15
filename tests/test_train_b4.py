"""scripts/train_b4.py end to end with a stand-in model: no GPU, no torch, no data download.

Checks the plumbing (window sweep on dev -> chosen k -> seeds -> shared prediction CSVs -> reports ->
resume) and the EmoWOZ label space, not RoBERTa itself (tests/test_roberta_ctx.py does that).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from emotionshift.data.labels import EMOTIONS, EMOWOZ_EMOTIONS
from emotionshift.data.shifts import add_shift_labels

REPO = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_b4", REPO / "scripts" / "train_b4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy(dataset="meld", n=12, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    emowoz = dataset == "emowoz"
    labels = EMOWOZ_EMOTIONS if emowoz else EMOTIONS
    rows = []
    for split in ["train", "dev", "test"] + (["test_dedup"] if dataset == "dailydialog" else []):
        for d in range(n):
            did = f"MUL{split[:2]}{d:03d}.json" if emowoz else d
            for t in range(int(rng.integers(4, 8))):
                system = emowoz and t % 2 == 1
                speaker = ("system" if system else "user") if emowoz else ["Joey", "Rachel"][t % 2]
                rows.append((dataset, split, did, t, speaker, f"utterance {d} {t}",
                             None if system else labels[int(rng.integers(0, len(labels)))]))
    df = pd.DataFrame(rows, columns=["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"])
    return add_shift_labels(df).reset_index(drop=True)


class FakeModel:
    """The surface of RobertaContext that train_b4.py uses."""
    fits: list = []

    def __init__(self, mcfg, dataset, k, seed, device=None):
        self.mcfg, self.dataset, self.k, self.seed, self.device = mcfg, dataset, k, seed, device or "cpu"
        self.history = []

    def fit_df(self, train, dev):
        FakeModel.fits.append((self.dataset, self.k, self.seed))
        assert train.split.eq("train").all() and dev.split.eq("dev").all()
        score = 0.30 + 0.01 * (20 if self.k is None else self.k)       # a longer window wins on dev
        self.history = [{"epoch": 1.0, "loss": 1.0, **dict.fromkeys(
            ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "weighted_f1_no_neutral",
             "macro_f1_no_neutral", "accuracy"], score)}]
        return self

    def predict_df(self, frame):
        rows = frame[frame.emotion.notna()]
        labels = EMOWOZ_EMOTIONS if self.dataset == "emowoz" else EMOTIONS
        pick = np.random.default_rng(self.seed + 1).integers(0, len(labels), len(rows))
        return pd.Series([labels[i] for i in pick], index=rows.index)

    def proba_inputs(self, inputs):
        labels = EMOWOZ_EMOTIONS if self.dataset == "emowoz" else EMOTIONS
        inputs = list(inputs)
        return np.random.default_rng(self.seed + len(inputs)).random((len(inputs), len(labels)))

    def predict_last(self, utterances, speakers=None):
        return "neutral"

    def num_parameters(self):
        return 124_650_247

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.bin").write_bytes(b"0" * 1000)


@pytest.fixture
def mod(tmp_path, monkeypatch):
    m = load_script()
    FakeModel.fits = []
    monkeypatch.setattr(m, "OUT", tmp_path)
    monkeypatch.setattr(m, "make_model", lambda mcfg, ds, k, seed: FakeModel(mcfg, ds, k, seed))
    monkeypatch.setattr(m, "load_model", lambda path, device=None: FakeModel({}, "meld", 0, 0, device))
    return m


def args(**kw):
    return SimpleNamespace(**{**dict(tag="b4", quick=False, k=None, grid=None, class_weight=None,
                                     model_name=None, fresh=False, controls=False), **kw})


def config(seeds=(0, 1, 2)):
    cfg = yaml.safe_load(open(REPO / "configs" / "b4_roberta_ctx.yaml"))
    cfg.update(seeds=list(seeds), bootstrap=20)
    return cfg


def test_config_follows_the_proposal():
    cfg = config()
    assert cfg["sweep"]["meld"]["grid"] == [1, 3, 5, 10, "all"]          # proposal: k in {1, 3, 5, 10, all}
    assert cfg["model"]["max_len"] >= 256 and cfg["sweep_seeds"] == [0, 1, 2]
    assert cfg["select_on"]["emowoz"] == "weighted_f1_no_neutral"


def test_sweep_picks_k_on_dev_and_writes_the_group_formats(mod, tmp_path):
    cfg = config()
    s = mod.run_dataset("meld", cfg, args(), {"meld": toy()}, {})
    assert s["chosen_k"] == "all" and s["chosen_by"] == "sweep"
    assert sorted(FakeModel.fits, key=str) == sorted((("meld", k, sd) for k in [1, 3, 5, 10, None] for sd in (0, 1, 2)), key=str)
    sweep = pd.read_csv(tmp_path / "reports/b4/meld/sweep.csv")
    assert set(sweep.k.astype(str)) == {"1", "3", "5", "10", "all"} and len(sweep) == 15

    for seed in range(3):
        p = pd.read_csv(tmp_path / f"predictions/b4_meld_test_seed{seed}.csv")
        assert list(p.columns) == mod.PRED_COLS
        assert set(p.pred) <= set(EMOTIONS) and (p.model == "b4_roberta_ctx").all() and (p.seed == seed).all()
    for f in ["summary.json", "stratified_test.csv", "per_class_test.csv", "confusion_test.csv", "sweep.csv"]:
        assert (tmp_path / "reports/b4/meld" / f).exists()
    assert (tmp_path / "models/b4_meld/model.bin").exists()
    assert not list((tmp_path / "models/_sweep").iterdir())                # only the chosen model is kept
    saved = json.loads((tmp_path / "reports/b4/meld/summary.json").read_text())
    assert saved["chosen_k"] == "all" and saved["efficiency"]["parameters"] == 124_650_247
    assert {"latency_one_utterance_cpu", "batch_throughput", "train_seconds", "model_dir_mb"} <= set(saved["efficiency"])
    assert "Window sweep" in mod.results_table([s], "b4")


def test_tie_goes_to_the_smaller_window(mod, monkeypatch):
    monkeypatch.setattr(FakeModel, "fit_df", lambda self, tr, dv: (setattr(self, "history", [
        dict.fromkeys(["weighted_f1", "macro_f1", "micro_f1_no_neutral", "weighted_f1_no_neutral", "accuracy"], 0.5)]), self)[1])
    s = mod.run_dataset("meld", config(), args(), {"meld": toy()}, {})
    assert s["chosen_k"] == "1"


def test_other_datasets_reuse_the_window_chosen_on_meld(mod):
    cfg = config()
    meld = mod.run_dataset("meld", cfg, args(), {"meld": toy()}, {})
    FakeModel.fits.clear()
    s = mod.run_dataset("dailydialog", cfg, args(), {"dailydialog": toy("dailydialog")}, {"meld": {"chosen_k": None}})
    assert meld["chosen_k"] == s["chosen_k"] == "all" and s["chosen_by"] == "k_from:meld"
    assert sorted(FakeModel.fits, key=str) == [("dailydialog", None, sd) for sd in (0, 1, 2)]      # no sweep
    assert set(s["test"]) == {"test_dedup", "test"}
    assert (mod.OUT / "predictions/b4_dailydialog_test_dedup_seed0.csv").exists()


def test_reuse_works_across_invocations_and_fails_clearly_without_meld(mod):
    cfg = config()
    with pytest.raises(SystemExit, match="datasets meld"):
        mod.run_dataset("dailydialog", cfg, args(), {"dailydialog": toy("dailydialog")}, {})
    mod.run_dataset("meld", cfg, args(), {"meld": toy()}, {})
    s = mod.run_dataset("dailydialog", cfg, args(), {"dailydialog": toy("dailydialog")}, {})   # reads summary.json
    assert s["chosen_k"] == "all"


def test_emowoz_uses_native_labels_user_turns_only_and_a_k0_control(mod, tmp_path):
    cfg = config()
    df = toy("emowoz")
    s = mod.run_dataset("emowoz", cfg, args(), {"emowoz": df}, {})
    p = pd.read_csv(tmp_path / "predictions/b4_emowoz_test_seed0.csv")
    assert set(p.pred) <= set(EMOWOZ_EMOTIONS) and p.speaker.eq("user").all()
    assert len(p) == df[(df.split == "test") & df.emotion.notna()].shape[0]      # system turns are not scored
    assert s["label_space"] == EMOWOZ_EMOTIONS and s["selected_on_dev"] == "weighted_f1_no_neutral"
    assert (tmp_path / "predictions/b4k0_emowoz_test_seed0.csv").exists()
    assert "weighted_f1_no_neutral" in s["test"]["test"] and "k0" in s["controls"]
    assert s["chosen_k"] == "10" and s["vs_baseline"]["source"] == "b4k0" and "delta" in s["vs_baseline"]
    assert ("emowoz", 0, 0) in FakeModel.fits


def test_b3_predictions_are_the_baseline_when_present(mod, tmp_path):
    cfg = config()
    df = toy()
    t = df[df.split == "test"].copy()
    t["pred"] = "neutral"
    (tmp_path / "predictions").mkdir(exist_ok=True)
    for sd in range(3):
        t.to_csv(tmp_path / f"predictions/b3_meld_test_seed{sd}.csv", index=False)
    s = mod.run_dataset("meld", cfg, args(), {"meld": df}, {})
    assert s["vs_baseline"]["source"] == "b3" and s["vs_baseline"]["metric"] == "weighted_f1"


def test_resume_reuses_finished_runs_and_fresh_redoes_them(mod):
    cfg = config(seeds=(0, 1))
    mod.run_dataset("meld", cfg, args(), {"meld": toy()}, {})
    FakeModel.fits.clear()
    mod.run_dataset("meld", cfg, args(), {"meld": toy()}, {})
    assert FakeModel.fits == []
    mod.run_dataset("meld", cfg, args(fresh=True), {"meld": toy()}, {})
    assert len(FakeModel.fits) == 5 * 3                                  # the sweep again; seeds are inside it


def test_fixed_k_skips_the_sweep_and_tag_separates_outputs(mod, tmp_path):
    s = mod.run_dataset("meld", config(), args(k="5", tag="b4plus", class_weight="balanced"), {"meld": toy()}, {})
    assert sorted(FakeModel.fits) == [("meld", 5, sd) for sd in (0, 1, 2)] and s["chosen_by"] == "--k"
    assert (tmp_path / "predictions/b4plus_meld_test_seed0.csv").exists()
    assert (tmp_path / "models/b4plus_meld/model.bin").exists()
    assert s["config"]["class_weight"] == "balanced"


def test_quick_subset_keeps_whole_dialogues(mod):
    df = toy("emowoz", n=30)
    small = mod.subset(df, 5)
    assert small.groupby("split").dialogue_id.nunique().eq(5).all()
    assert len(small) == df[df.dialogue_id.isin(small.dialogue_id)].shape[0]


def test_controls_are_scored_on_every_seed_of_the_chosen_k(mod, tmp_path):
    s = mod.run_dataset("meld", config(), args(k="5", tag="b4c", controls=True), {"meld": toy()}, {})
    c = pd.read_csv(tmp_path / "reports/b4c/meld/controls.csv")
    assert set(c.input) == {"real", "none", "shuffled_context", "shuffled_speakers"}
    assert {"mean", "std", "seeds"} <= set(c.columns) and (c.seeds == 3).all()
    assert len(list((tmp_path / "reports/b4c/meld/runs").glob("k5_seed*_controls.csv"))) == 3
    assert s["e4"]["seeds"] == [0, 1, 2] and s["e4"]["metric"] == "weighted_f1"


def test_controls_work_on_emowoz_and_are_not_scored_for_the_k0_control(mod, tmp_path):
    mod.run_dataset("emowoz", config(), args(controls=True), {"emowoz": toy("emowoz")}, {})
    runs = tmp_path / "reports/b4/emowoz/runs"
    assert (runs / "k10_seed0_controls.csv").exists() and not list(runs.glob("k0_seed*_controls.csv"))
    assert (tmp_path / "reports/b4/emowoz/controls.csv").exists()


def test_without_the_flag_no_controls_are_written(mod, tmp_path):
    mod.run_dataset("meld", config(), args(k="5", tag="b4n"), {"meld": toy()}, {})
    assert not (tmp_path / "reports/b4n/meld/controls.csv").exists()


def test_retraining_the_final_seeds_with_k_keeps_the_sweep_and_the_k0_control(mod, tmp_path):
    cfg = config()
    first = mod.run_dataset("emowoz", cfg, args(), {"emowoz": toy("emowoz")}, {})
    FakeModel.fits.clear()
    s = mod.run_dataset("emowoz", cfg, args(k="10", fresh=True, controls=True), {"emowoz": toy("emowoz")}, {})
    assert sorted(FakeModel.fits) == [("emowoz", 10, sd) for sd in (0, 1, 2)]          # only the final seeds are redone
    assert s["sweep"] == first["sweep"] and "retrained" in s["chosen_by"] and "k0" in s["controls"]
    assert s["vs_baseline"]["source"] == "b4k0" and "delta" in s["vs_baseline"]
    assert (tmp_path / "reports/b4/emowoz/controls.csv").exists()
    assert len(pd.read_csv(tmp_path / "reports/b4/emowoz/sweep.csv")) == len(pd.read_csv(tmp_path / "reports/b4/emowoz/sweep.csv").drop_duplicates())


def test_missing_model_does_not_stop_the_run(mod, tmp_path):
    cfg = config()
    mod.run_dataset("meld", cfg, args(k="5", tag="b4m"), {"meld": toy()}, {})
    for f in (tmp_path / "models").rglob("*"):
        if f.is_file():
            f.unlink()                                   # simulate a model folder without its weights
    mod.load_model = lambda *a_, **k_: (_ for _ in ()).throw(OSError("no model.safetensors"))
    s = mod.run_dataset("meld", cfg, args(k="5", tag="b4m"), {"meld": toy()}, {})
    assert "efficiency" in s and s["test"]


def test_results_table_survives_a_summary_without_latency(mod, tmp_path):
    s = mod.run_dataset("meld", config(), args(k="5", tag="b4x"), {"meld": toy()}, {})
    s["efficiency"] = {"train_seconds": {"mean": 60.0}, "note": "model files not found, efficiency not measured"}
    assert "not measured" in mod.results_table([s], "b4x")
