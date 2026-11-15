"""B4: context-aware RoBERTa (proposal: "Models", B4; Table 4 #5).

RoBERTa-base reads the target utterance together with the previous ``k`` turns of its dialogue,
with speaker names (EmoBERTa-style, past only), and predicts the target's emotion. ``k = 0`` is
the B3 input, so the gap to ``k > 0`` is the effect of context. The input text is the team's
shared format (``emotionshift.data.context``); training is B3's recipe (``roberta.py``).

One difference from B3: the label space follows ``cfg.dataset`` (``labels.emotions``), so EmoWOZ
trains on its own seven labels (neutral, fearful, dissatisfied, ...) as ``load("emowoz")`` gives
them, while MELD and DailyDialog keep the shared seven.

Usage (also what the demo calls)::

    from emotionshift.models.roberta_ctx import RobertaContext
    m = RobertaContext.load("models/b4_meld")
    m.predict(["Hi!", "Oh my God!"], speakers=["Joey", "Rachel"])   # one dialogue in order -> a label per turn
    m.predict_last(dialogue_so_far)                                 # label of the newest turn only (live demo step)

Needs the optional dependencies: ``pip install -e ".[transformers]"``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from ..data.context import build_inputs, dialogue_inputs
from ..data.labels import emotions
from ..eval import metrics as M
from .roberta import RobertaConfig, RobertaUtterance, _WeightedTrainer


@dataclass
class CtxConfig(RobertaConfig):
    max_len: int = 256            # B3 used 128 for one utterance; the history needs room
    dataset: str = "meld"         # picks the label space: labels.emotions(dataset)
    k: int | None = 4             # previous turns shown; 0 = B3 input, None = the whole past


class _Examples(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels, label2id, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = [label2id[e] for e in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class RobertaContext(RobertaUtterance):
    def __init__(self, cfg: CtxConfig | None = None, device: str | None = None):
        super().__init__(cfg or CtxConfig(), device)
        self.tokenizer.truncation_side = "left"      # keep the target and the newest history
        self.labels = emotions(self.cfg.dataset)
        self.label2id = {e: i for i, e in enumerate(self.labels)}

    # ---- training: B3's recipe, label space and input from the dataset ---------------------
    def fit_df(self, train: pd.DataFrame, dev: pd.DataFrame, log_every: int = 100) -> "RobertaContext":
        """Fine-tune on a loader table; keeps the epoch with the best dev ``cfg.select_on``."""
        tr, dv = build_inputs(train, self.cfg.k), build_inputs(dev, self.cfg.k)
        cfg, labels = self.cfg, self.labels
        set_seed(cfg.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=self.label2id)
        train = _Examples(self.tokenizer, tr["input"], tr["emotion"], self.label2id, cfg.max_len)
        devset = _Examples(self.tokenizer, dv["input"], dv["emotion"], self.label2id, cfg.max_len)

        small = self.device == "mps"                 # Apple-silicon GPUs: half batches, same effective size
        steps = int(np.ceil(len(train) / cfg.batch_size)) * cfg.epochs
        out = tempfile.mkdtemp(prefix=f"b4_seed{cfg.seed}_")
        args = TrainingArguments(
            output_dir=out, seed=cfg.seed, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay, warmup_steps=int(cfg.warmup * steps),
            per_device_train_batch_size=cfg.batch_size // 2 if small else cfg.batch_size,
            gradient_accumulation_steps=2 if small else 1,
            per_device_eval_batch_size=32 if small else 128,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model=cfg.select_on,
            fp16=self.device == "cuda", report_to="none", logging_steps=log_every,
            disable_tqdm=True, use_cpu=self.device == "cpu",
        )

        def compute_metrics(p):
            logits, gold = p
            s = M.scores([labels[i] for i in gold], [labels[i] for i in logits.argmax(-1)], labels)
            return {k: v for k, v in s.items() if k != "n"}

        kwargs = dict(model=self.model, args=args, train_dataset=train, eval_dataset=devset,
                      data_collator=DataCollatorWithPadding(self.tokenizer), compute_metrics=compute_metrics)
        if cfg.class_weight == "balanced":
            counts = np.bincount(train.labels, minlength=len(labels))
            w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float)
            trainer = _WeightedTrainer(**kwargs, class_weights=w)
        else:
            trainer = Trainer(**kwargs)
        trainer.train()
        self.history = [{k.removeprefix("eval_"): v for k, v in h.items()}
                        for h in trainer.state.log_history if "eval_loss" in h]
        self.model = trainer.model.to(self.device).eval()
        shutil.rmtree(out, ignore_errors=True)
        return self

    # ---- inference --------------------------------------------------------------------------
    @torch.no_grad()
    def proba_inputs(self, inputs, batch_size: int = 64) -> np.ndarray:
        """Softmax probabilities for ready-made model inputs, columns in ``self.labels`` order."""
        self.model.eval()
        out, inputs = [], list(inputs)
        for i in range(0, len(inputs), batch_size):
            enc = self.tokenizer(inputs[i:i + batch_size], truncation=True, max_length=self.cfg.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict_proba_df(self, frame: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        """One row per *labelled* row of ``frame`` (a loader table; EmoWOZ system turns are context only)."""
        return self.proba_inputs(build_inputs(frame, self.cfg.k)["input"], batch_size)

    def predict_df(self, frame: pd.DataFrame) -> pd.Series:
        """Predicted label names, indexed like the labelled rows of ``frame``."""
        rows = frame[frame["emotion"].notna()]
        labels = [self.labels[i] for i in self.predict_proba_df(frame).argmax(1)]
        return pd.Series(labels, index=rows.index)

    # ---- demo interface: ``utterances`` = one dialogue, in order ---------------------------
    def _turns(self, utterances, speakers=None) -> list[tuple[str, str]]:
        utterances = list(utterances)
        speakers = list(speakers) if speakers is not None else ["AB"[i % 2] for i in range(len(utterances))]
        if len(speakers) != len(utterances):
            raise ValueError("speakers and utterances must have the same length")
        return list(zip(speakers, utterances, strict=True))

    def predict_proba(self, utterances, speakers=None, batch_size: int = 64) -> np.ndarray:
        """(n, n_labels) probabilities, one row per turn; each turn sees its previous ``k`` turns."""
        return self.proba_inputs(dialogue_inputs(self._turns(utterances, speakers), self.cfg.k), batch_size)

    def predict(self, utterances, speakers=None) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(utterances, speakers).argmax(1)]

    def predict_proba_last(self, utterances, speakers=None) -> np.ndarray:
        """Probabilities for the *newest* turn only, from just its last ``k + 1`` turns.
        This is the cost of one live demo step, which is what E9 latency is measured on."""
        turns = self._turns(utterances, speakers)
        if self.cfg.k is not None:
            turns = turns[-(self.cfg.k + 1):]
        return self.proba_inputs(dialogue_inputs(turns, self.cfg.k)[-1:])[0]

    def predict_last(self, utterances, speakers=None) -> str:
        return self.labels[int(self.predict_proba_last(utterances, speakers).argmax())]

    # ---- persistence ------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "b4_config.json").write_text(json.dumps({"cfg": asdict(self.cfg), "history": self.history}, indent=1))

    @classmethod
    def load(cls, path, device: str | None = None) -> "RobertaContext":
        path = Path(path)
        meta = json.loads((path / "b4_config.json").read_text())
        m = cls(CtxConfig(**{**meta["cfg"], "model_name": str(path)}), device=device)
        m.model = AutoModelForSequenceClassification.from_pretrained(path).to(m.device).eval()
        m.history = meta.get("history", [])
        return m
