"""B3 model interface: what the demo, B4 and ShiftAware rely on.

Uses a tiny random RoBERTa so it runs on CPU in seconds; skipped when the
optional ``transformers`` extra is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("transformers")

from emotionshift.data.labels import EMOTIONS  # noqa: E402
from emotionshift.data.loaders import load_meld  # noqa: E402
from emotionshift.models.roberta import RobertaConfig, RobertaUtterance  # noqa: E402

TINY = "hf-internal-testing/tiny-random-RobertaModel"
TEXTS = ["Oh my God!", "Yeah.", "I'm so sorry about your dad."]


@pytest.fixture(scope="module")
def model():
    df = load_meld()
    tr = df[(df.split == "train") & (df.dialogue_id < 15)]
    dev = df[(df.split == "dev") & (df.dialogue_id < 5)]
    cfg = RobertaConfig(model_name=TINY, epochs=1, batch_size=16)
    return RobertaUtterance(cfg, device="cpu").fit(tr.text, tr.emotion, dev.text, dev.emotion)


def test_predict_returns_shared_labels(model):
    out = model.predict(TEXTS)
    assert len(out) == 3 and set(out) <= set(EMOTIONS)


def test_predict_proba_in_label_order(model):
    p = model.predict_proba(TEXTS)
    assert p.shape == (3, len(EMOTIONS))
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert [EMOTIONS[i] for i in p.argmax(axis=1)] == model.predict(TEXTS)


def test_dev_history_recorded(model):
    assert len(model.history) == 1 and "weighted_f1" in model.history[0]


def test_save_load_round_trip(model, tmp_path):
    model.save(tmp_path / "b3")
    again = RobertaUtterance.load(tmp_path / "b3", device="cpu")
    assert np.allclose(again.predict_proba(TEXTS), model.predict_proba(TEXTS), atol=1e-5)


def test_embeddings_one_vector_per_utterance(model):
    e = model.embed(TEXTS)
    assert e.shape == (3, model.model.config.hidden_size)


def test_attributions_cover_every_word(model):
    words, scores, label, p = model.attributions("Wow, that's really amazing!")
    assert "".join(words).replace(" ", "") == "Wow,that'sreallyamazing!"
    assert len(scores) == len(words) and label in EMOTIONS and 0 <= p <= 1
