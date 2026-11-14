"""Offline native-label and legacy-label RoBERTa regression checks."""

import json

import numpy as np
import pytest

pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, RobertaForSequenceClassification
from transformers import RobertaConfig as HFConfig

from emotionshift.data.labels import EMOTIONS, emotions
from emotionshift.models.roberta import RobertaConfig, RobertaUtterance


@pytest.mark.parametrize("native", [False, True])
def test_native_and_legacy_checkpoint(tmp_path, native):
    import torch

    torch.set_num_threads(2)
    base = tmp_path / "tiny"
    base.mkdir()
    tok = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "hello": 2, "thanks": 3}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    fast.save_pretrained(base)
    hf = HFConfig(
        vocab_size=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        max_position_embeddings=32,
        num_labels=7,
        pad_token_id=0,
    )
    RobertaForSequenceClassification(hf).save_pretrained(base)
    labels = emotions("emowoz") if native else list(EMOTIONS)
    cfg = RobertaConfig(
        model_name=str(base),
        epochs=1,
        batch_size=4,
        micro_batch_size=2,
        eval_batch_size=2,
        max_len=8,
        select_on="weighted_f1_no_neutral" if native else "weighted_f1",
    )
    model = RobertaUtterance(cfg, device="cpu", labels=labels)
    model.fit(["hello thanks"] * 14, labels * 2, ["thanks"] * 7, labels)
    assert 0 < model.training_loss < 3  # Mean CE, not multiplied by accumulation.
    assert cfg.select_on in model.history[0]
    assert list(model.model.config.id2label.values()) == labels
    prob = model.predict_proba(["hello"])
    assert prob.shape == (1, 7) and np.allclose(prob.sum(axis=1), 1)
    assert model.predict(["hello"]) == [labels[prob.argmax()]]
    path = tmp_path / "saved"
    model.save(path)
    if not native:
        meta = json.loads((path / "b3_config.json").read_text())
        meta.pop("labels")
        (path / "b3_config.json").write_text(json.dumps(meta))
    restored = RobertaUtterance.load(path, device="cpu")
    assert restored.labels == labels
    np.testing.assert_allclose(restored.predict_proba(["hello"]), prob, atol=1e-6)
