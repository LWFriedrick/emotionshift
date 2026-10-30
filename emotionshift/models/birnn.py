"""B2: word-level BiGRU / BiLSTM, utterance only, learned embeddings.

Bidirectionality is within the current utterance, never across future turns.
Vocabulary and loss weights use train only; checkpoint selection uses dev only.
"""
from __future__ import annotations

import copy
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from ..data.labels import EMOTIONS
from ..eval.metrics import scores


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


@dataclass
class BiRNNConfig:
    architecture: str = "gru"
    embedding_dim: int = 64
    hidden_dim: int = 64
    max_vocab: int = 15000
    min_freq: int = 2
    max_length: int = 64
    dropout: float = 0.3
    batch_size: int = 128
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    epochs: int = 12
    patience: int = 3
    class_weight: str = "sqrt_inverse"
    seed: int = 0

    def __post_init__(self):
        if self.architecture not in {"gru", "lstm"}:
            raise ValueError("architecture must be gru or lstm")
        if self.class_weight not in {"none", "sqrt_inverse"}:
            raise ValueError("class_weight must be none or sqrt_inverse")
        for name in ("embedding_dim", "hidden_dim", "min_freq", "max_length",
                     "batch_size", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_vocab < 2 or not 0 <= self.dropout < 1:
            raise ValueError("max_vocab >= 2 and 0 <= dropout < 1 required")


class _Encoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout)
        cell = nn.GRU if cfg.architecture == "gru" else nn.LSTM
        self.rnn = cell(cfg.embedding_dim, cfg.hidden_dim, batch_first=True,
                        bidirectional=True)
        self.output = nn.Linear(2 * cfg.hidden_dim, len(EMOTIONS))

    def forward(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # LSTM: use h, not cell state c
        utterance = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(utterance))


def _collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths, torch.tensor(labels)


class BiRNNBaseline:
    def __init__(self, cfg: BiRNNConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or BiRNNConfig()
        self.device = torch.device(device)
        self.vocab: dict[str, int] = {}
        self.network: _Encoder | None = None
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.best_dev_score: float | None = None

    def _encode(self, texts):
        return [torch.tensor([self.vocab.get(t, 1) for t in tokenize(text)[:self.cfg.max_length]]
                             or [1], dtype=torch.long) for text in texts]

    def fit(self, texts, labels, dev_texts, dev_labels, metric="weighted_f1"):
        texts, labels = list(texts), list(labels)
        dev_texts, dev_labels = list(dev_texts), list(dev_labels)
        if not texts or len(texts) != len(labels) or not dev_texts or len(dev_texts) != len(dev_labels):
            raise ValueError("Nonempty, aligned train and dev texts/labels are required")
        if not set(labels + dev_labels) <= set(EMOTIONS):
            raise ValueError("Unknown emotion label")
        if metric not in {"weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"}:
            raise ValueError(f"Unknown selection metric: {metric}")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        counts = Counter(t for text in texts for t in tokenize(text))
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.vocab.update({t: i + 2 for i, t in enumerate(
            [t for t in ranked if counts[t] >= self.cfg.min_freq][:self.cfg.max_vocab - 2])})
        self.network = _Encoder(len(self.vocab), self.cfg).to(self.device)
        targets = [EMOTIONS.index(e) for e in labels]
        generator = torch.Generator().manual_seed(self.cfg.seed)
        loader = DataLoader(list(zip(self._encode(texts), targets)), batch_size=self.cfg.batch_size,
                            shuffle=True, generator=generator, collate_fn=_collate)
        weight = None
        if self.cfg.class_weight == "sqrt_inverse":
            freq = np.bincount(targets, minlength=len(EMOTIONS))
            values = np.zeros(len(EMOTIONS), dtype=np.float32)
            present = freq > 0
            values[present] = 1 / np.sqrt(freq[present])
            values[present] /= values[present].mean()
            weight = torch.tensor(values, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.cfg.learning_rate,
                                      weight_decay=self.cfg.weight_decay)
        best, stale, state = -1.0, 0, None
        self.history = []
        for epoch in range(1, self.cfg.epochs + 1):
            self.network.train()
            loss_sum = 0.0
            for tokens, lengths, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(tokens.to(self.device), lengths), y.to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(y)
            result = scores(dev_labels, self.predict(dev_texts))
            self.history.append({"epoch": epoch, "train_loss": loss_sum / len(texts),
                                 **{f"dev_{k}": float(v) for k, v in result.items() if k != "n"}})
            print(f"  {self.cfg.architecture} seed={self.cfg.seed} epoch={epoch} "
                  f"loss={loss_sum / len(texts):.4f} dev_{metric}={result[metric]:.4f}", flush=True)
            if result[metric] > best:
                best, stale = result[metric], 0
                self.best_epoch = epoch
                state = copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
            if stale >= self.cfg.patience:
                break
        self.network.load_state_dict(state)
        self.network.eval()
        self.best_dev_score = float(best)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Fit or load the model before prediction")
        encoded = self._encode(list(texts))
        if not encoded:
            return np.empty((0, len(EMOTIONS)), dtype=np.float32)
        loader = DataLoader(list(zip(encoded, [0] * len(encoded))),
                            batch_size=self.cfg.batch_size, collate_fn=_collate)
        self.network.eval()
        parts = []
        with torch.inference_mode():
            for tokens, lengths, _ in loader:
                logits = self.network(tokens.to(self.device), lengths)
                parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(parts)

    def predict(self, texts) -> list[str]:
        return [EMOTIONS[int(i)] for i in self.predict_proba(texts).argmax(axis=1)]

    def save(self, path):
        if self.network is None:
            raise RuntimeError("No fitted model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.cfg), "vocab": self.vocab,
                    "labels": list(EMOTIONS), "state_dict": {k: v.cpu() for k, v in self.network.state_dict().items()},
                    "history": self.history, "best_epoch": self.best_epoch,
                    "best_dev_score": self.best_dev_score}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if obj["format_version"] != 1 or obj["labels"] != list(EMOTIONS):
            raise ValueError("Unsupported checkpoint format or label order")
        model = cls(BiRNNConfig(**obj["config"]), device=device)
        model.vocab = obj["vocab"]
        model.network = _Encoder(len(model.vocab), model.cfg).to(model.device)
        model.network.load_state_dict(obj["state_dict"])
        model.network.eval()
        model.history, model.best_epoch = obj["history"], obj["best_epoch"]
        model.best_dev_score = obj["best_dev_score"]
        return model
