"""Native context targets retain system history without gold/future leakage."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    RobertaForSequenceClassification,
)
from transformers import (
    RobertaConfig as HFConfig,
)

from emotionshift.data.labels import emotions
from emotionshift.models.roberta import RobertaConfig
from emotionshift.models.roberta_context import RobertaContext
from scripts.train_emowoz_context import prepare_context


def frame():
    rows = []
    for i, label in enumerate(emotions("emowoz")):
        for turn, speaker, text, emotion in [
            (0, "user", "hello", label),
            (1, "system", f"system_hint_{i}", None),
            (2, "user", "thanks", label),
        ]:
            rows.append(
                {
                    "split": "train",
                    "dialogue_id": i,
                    "turn": turn,
                    "speaker": speaker,
                    "text": text,
                    "emotion": emotion,
                }
            )
    return pd.DataFrame(rows)


def test_user_targets_retain_unlabelled_system_context():
    df = frame()
    target, inputs = prepare_context(df)
    assert len(target) == 14 and target.speaker.eq("user").all()
    assert inputs[0] == "user: hello"
    assert "system: system_hint_0" in inputs[1]
    changed = df.copy()
    changed.loc[changed.emotion.notna(), "emotion"] = "abusive"
    assert prepare_context(changed)[1] == inputs
    changed.loc[(changed.dialogue_id == 0) & (changed.turn == 2), "text"] = "FUTURE_SENTINEL"
    assert "FUTURE_SENTINEL" not in prepare_context(changed)[1][0]
    assert "system_hint_1" not in inputs[1]
    assert prepare_context(df, mode="none")[1][1] == "user: thanks"


def test_mixed_splits_rejected():
    df = frame()
    df.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="separately"):
        prepare_context(df)


def test_native_context_checkpoint_round_trip(tmp_path):
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
    config = HFConfig(
        vocab_size=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        max_position_embeddings=48,
        num_labels=7,
        pad_token_id=0,
    )
    RobertaForSequenceClassification(config).save_pretrained(base)
    cfg = RobertaConfig(
        model_name=str(base),
        epochs=1,
        batch_size=4,
        micro_batch_size=2,
        eval_batch_size=2,
        max_len=32,
        select_on="weighted_f1_no_neutral",
    )
    labels = emotions("emowoz")
    model = RobertaContext(cfg, device="cpu", k="all", labels=labels)
    target, inputs = prepare_context(frame())
    model.fit(inputs, target.emotion, inputs, target.emotion)
    assert 0 < model.training_loss < 3
    assert model.tokenizer.truncation_side == "left"
    probs = model.predict_proba(inputs)
    assert probs.shape == (14, 7) and np.allclose(probs.sum(axis=1), 1)
    path = tmp_path / "saved"
    model.save(path)
    restored = RobertaContext.load(path, device="cpu")
    assert restored.labels == labels and restored.k == "all"
    np.testing.assert_allclose(restored.predict_proba(inputs), probs, atol=1e-6)
