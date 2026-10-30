"""B3: RoBERTa fine-tuned on single utterances (proposal: "Models", Table 4 #4).

Like B1 the model sees **one utterance at a time** -- no dialogue context, no
speaker -- so it is the strong no-context transformer baseline. B4 (context) and
ShiftAware (#6) build on it: B4 can start from its weights, ShiftAware runs on its
cached utterance embeddings (``embed``).

Usage (also what the Streamlit demo calls)::

    from emotionshift.models.roberta import RobertaUtterance
    m = RobertaUtterance.load("models/b3_meld")
    m.predict(["Oh my God!", "Yeah."])          # -> ["surprise", "neutral"]
    m.predict_proba(["Oh my God!"])             # -> (n, 7) array, EMOTIONS order
    m.attributions("Wow, that's amazing!")      # words + Integrated Gradients scores
    m.embed(texts)                              # (n, 768) <s> vectors for ShiftAware

Needs the optional dependencies: ``pip install -e ".[transformers]"``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from ..data.labels import EMOTIONS, LABEL2ID
from ..eval import metrics as M


@dataclass
class RobertaConfig:
    model_name: str = "roberta-base"      # "distilroberta-base" = the demo's fast variant
    epochs: int = 4
    lr: float = 1e-5
    batch_size: int = 32
    max_len: int = 128                    # longest MELD utterance is 92 tokens; DailyDialog is cut at 128
    warmup: float = 0.1                   # share of training steps with a linear warm-up
    weight_decay: float = 0.01
    class_weight: str | None = None       # None or "balanced" (B4+ in the proposal)
    select_on: str = "weighted_f1"        # dev metric that picks the best epoch
    seed: int = 0


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Utterances(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = [LABEL2ID[e] for e in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class _WeightedTrainer(Trainer):
    """Cross-entropy with class weights (only used when ``class_weight="balanced"``)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


class RobertaUtterance:
    def __init__(self, cfg: RobertaConfig | None = None, device: str | None = None):
        self.cfg = cfg or RobertaConfig()
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = None
        self.history: list[dict] = []     # dev scores per epoch, filled by fit

    # ---- training -------------------------------------------------------------------------
    def fit(self, texts, labels, dev_texts, dev_labels, log_every: int = 100):
        """Fine-tune on train; keep the epoch with the best dev ``cfg.select_on``."""
        cfg = self.cfg
        set_seed(cfg.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=len(EMOTIONS),
            id2label=dict(enumerate(EMOTIONS)), label2id=LABEL2ID)
        train = _Utterances(self.tokenizer, texts, labels, cfg.max_len)
        dev = _Utterances(self.tokenizer, dev_texts, dev_labels, cfg.max_len)

        # Apple-silicon GPUs share memory with the system: half batches, same effective size.
        small = self.device == "mps"
        steps = int(np.ceil(len(train) / cfg.batch_size)) * cfg.epochs
        out = tempfile.mkdtemp(prefix=f"b3_seed{cfg.seed}_")
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
            s = M.scores([EMOTIONS[i] for i in gold], [EMOTIONS[i] for i in logits.argmax(-1)])
            return {k: v for k, v in s.items() if k != "n"}

        kwargs = dict(model=self.model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorWithPadding(self.tokenizer), compute_metrics=compute_metrics)
        if cfg.class_weight == "balanced":
            counts = np.bincount(train.labels, minlength=len(EMOTIONS))
            w = torch.tensor(counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1)), dtype=torch.float)
            trainer = _WeightedTrainer(**kwargs, class_weights=w)
        else:
            trainer = Trainer(**kwargs)
        trainer.train()
        self.history = [{k.removeprefix("eval_"): v for k, v in h.items()}
                        for h in trainer.state.log_history if "eval_loss" in h]
        self.model = trainer.model.to(self.device).eval()
        shutil.rmtree(out, ignore_errors=True)
        return self

    # ---- inference ------------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        """Softmax probabilities, columns in ``EMOTIONS`` order."""
        self.model.eval()
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=self.cfg.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))

    def predict(self, texts) -> list[str]:
        return [EMOTIONS[i] for i in self.predict_proba(texts).argmax(1)]

    @torch.no_grad()
    def embed(self, texts, batch_size: int = 64) -> np.ndarray:
        """Final-layer ``<s>`` vector per utterance -- the cached input of ShiftAware (#6)."""
        self.model.eval()
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=self.cfg.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            out.append(self.model.base_model(**enc).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)

    def attributions(self, text: str, target: str | None = None, steps: int = 32):
        """Integrated Gradients per word for one utterance.

        Each word's embedding is faded in from the padding embedding; the attribution is how
        much the score of ``target`` (default: the predicted emotion) rises along the way.
        Positive = pushes towards ``target``. Returns ``(words, scores, target, probability)``.
        """
        self.model.eval()
        emb = self.model.get_input_embeddings()
        enc = self.tokenizer(text, truncation=True, max_length=self.cfg.max_len, return_tensors="pt").to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        special = torch.tensor(self.tokenizer.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
                               device=self.device).bool()
        with torch.no_grad():
            probs = self.model(**enc).logits.float().softmax(-1)[0]
            x = emb(ids)
            base_ids = ids.clone()
            base_ids[0, ~special] = self.tokenizer.pad_token_id
            b = emb(base_ids)                                   # baseline keeps <s> and </s>
        t = int(probs.argmax()) if target is None else LABEL2ID[target]
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)[1:].view(-1, 1, 1)
        path = (b + alphas * (x - b)).detach().requires_grad_(True)
        score = self.model(inputs_embeds=path, attention_mask=mask.expand(steps, -1)).logits[:, t].sum()
        grads = torch.autograd.grad(score, path)[0].mean(0)
        attr = ((x - b)[0] * grads).sum(-1).detach().float().cpu().numpy()

        # Merge sub-word pieces back into words (RoBERTa BPE marks a new word with "Ġ").
        words, scores = [], []
        for tok, a, sp in zip(self.tokenizer.convert_ids_to_tokens(ids[0]), attr, special.tolist()):
            if sp:
                continue
            piece = self.tokenizer.convert_tokens_to_string([tok])
            if tok.startswith("Ġ") or not words:
                words.append(piece.strip())
                scores.append(float(a))
            else:
                words[-1] += piece
                scores[-1] += float(a)
        return words, np.array(scores), EMOTIONS[t], float(probs[t])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "b3_config.json").write_text(json.dumps({"cfg": asdict(self.cfg), "history": self.history}, indent=1))

    @classmethod
    def load(cls, path, device: str | None = None) -> "RobertaUtterance":
        path = Path(path)
        meta = json.loads((path / "b3_config.json").read_text())
        cfg = RobertaConfig(**{**meta["cfg"], "model_name": str(path)})
        m = cls(cfg, device=device)
        m.model = AutoModelForSequenceClassification.from_pretrained(path).to(m.device).eval()
        m.history = meta.get("history", [])
        return m
