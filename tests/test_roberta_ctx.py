"""B4 model interface (what the demo and train_b4.py rely on), on a tiny random RoBERTa.

Skipped when the optional ``transformers`` extra is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")

from emotionshift.data.labels import EMOTIONS, EMOWOZ_EMOTIONS  # noqa: E402
from emotionshift.data.shifts import add_shift_labels  # noqa: E402
from emotionshift.models.roberta_ctx import CtxConfig, RobertaContext  # noqa: E402

TINY = "hf-internal-testing/tiny-random-RobertaModel"
UTTS = ["Hi!", "Oh my God!", "I'm so sorry about your dad."]
WHO = ["Joey", "Rachel", "Joey"]


def toy(dataset="meld", split="train", n=8):
    rng = np.random.default_rng(0)
    labels = EMOWOZ_EMOTIONS[:3] if dataset == "emowoz" else EMOTIONS
    rows = []
    for d in range(n):
        for t in range(5):
            system = dataset == "emowoz" and t % 2 == 1
            rows.append((dataset, split, d, t, "system" if system else ("user" if dataset == "emowoz" else "AB"[t % 2]),
                         f"words {d} {t}", None if system else labels[int(rng.integers(0, len(labels)))]))
    df = pd.DataFrame(rows, columns=["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion"])
    return add_shift_labels(df).reset_index(drop=True)


def fit(dataset="meld", k=2):
    cfg = CtxConfig(model_name=TINY, epochs=1, batch_size=8, dataset=dataset, k=k,
                    select_on="weighted_f1_no_neutral" if dataset == "emowoz" else "weighted_f1")
    return RobertaContext(cfg, device="cpu").fit_df(toy(dataset), toy(dataset, "dev", 4))


@pytest.fixture(scope="module")
def model():
    return fit()


def test_truncation_keeps_the_target(model):
    assert model.tokenizer.truncation_side == "left"


def test_predict_one_label_per_turn(model):
    out = model.predict(UTTS, WHO)
    assert len(out) == 3 and set(out) <= set(EMOTIONS)
    assert len(model.predict(UTTS)) == 3                       # speakers default to A/B


def test_predict_last_matches_full_dialogue_prediction(model):
    p = model.predict_proba(UTTS, WHO)
    assert np.allclose(model.predict_proba_last(UTTS, WHO), p[-1], atol=1e-5)
    assert model.predict_last(UTTS, WHO) == model.predict(UTTS, WHO)[-1]


def test_frame_prediction_scores_labelled_rows_only(model):
    df = toy("meld", "test", 3)
    assert model.predict_proba_df(df).shape == (len(df), len(EMOTIONS))
    assert list(model.predict_df(df).index) == list(df.index)


def test_dev_history_recorded(model):
    assert len(model.history) == 1 and "weighted_f1" in model.history[0]


def test_save_load_round_trip(model, tmp_path):
    model.save(tmp_path / "b4")
    again = RobertaContext.load(tmp_path / "b4", device="cpu")
    assert again.cfg.k == 2 and again.cfg.dataset == "meld"
    assert np.allclose(again.predict_proba(UTTS, WHO), model.predict_proba(UTTS, WHO), atol=1e-5)


def test_emowoz_trains_on_its_native_labels():
    m = fit("emowoz", k=1)
    assert m.labels == EMOWOZ_EMOTIONS and m.model.config.num_labels == len(EMOWOZ_EMOTIONS)
    df = toy("emowoz", "test", 3)
    pred = m.predict_df(df)
    assert set(pred) <= set(EMOWOZ_EMOTIONS) and len(pred) == df.emotion.notna().sum()
    assert "weighted_f1_no_neutral" in m.history[0]


def test_class_weighted_loss_trains():
    cfg = CtxConfig(model_name=TINY, epochs=1, batch_size=8, k=1, class_weight="balanced")
    m = RobertaContext(cfg, device="cpu").fit_df(toy(), toy("meld", "dev", 4))
    assert len(m.history) == 1
