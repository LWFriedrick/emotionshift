"""B1: TF-IDF + linear classifier, utterance only (proposal: "Models").

The model sees one utterance at a time -- no dialogue context, no speaker --
so it is the traditional-ML reference point for the context-aware models.

Usage (also what the Streamlit demo calls)::

    from emotionshift.models.tfidf import TfidfBaseline
    m = TfidfBaseline.load("models/b1_meld.joblib")
    m.predict(["Oh my God!", "Yeah."])          # -> ["surprise", "neutral"]
    m.predict_proba(["Oh my God!"])             # -> (n, 7) array, EMOTIONS order
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ..data.labels import EMOTIONS


@dataclass
class TfidfConfig:
    classifier: str = "svm"            # "svm" (LinearSVC) or "logreg"
    C: float = 1.0
    class_weight: str | None = None    # None or "balanced"
    features: str = "word+char"        # "word" or "word+char"
    word_ngram: tuple = (1, 2)
    char_ngram: tuple = (2, 5)
    min_df: int = 2
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _features(cfg: TfidfConfig):
    word = TfidfVectorizer(lowercase=True, ngram_range=tuple(cfg.word_ngram), min_df=cfg.min_df,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[!?]")
    if cfg.features == "word":
        return word
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=tuple(cfg.char_ngram),
                           min_df=cfg.min_df, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def _classifier(cfg: TfidfConfig):
    if cfg.classifier == "svm":
        return LinearSVC(C=cfg.C, class_weight=cfg.class_weight, random_state=cfg.seed, max_iter=5000)
    if cfg.classifier == "logreg":
        return LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, random_state=cfg.seed,
                                  max_iter=2000, solver="lbfgs")
    raise ValueError(cfg.classifier)


class TfidfBaseline:
    def __init__(self, cfg: TfidfConfig | None = None):
        self.cfg = cfg or TfidfConfig()
        self.pipe = Pipeline([("tfidf", _features(self.cfg)), ("clf", _classifier(self.cfg))])

    def fit(self, texts, labels):
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict(self, texts) -> list[str]:
        return list(self.pipe.predict(list(texts)))

    def predict_proba(self, texts) -> np.ndarray:
        """Class scores in ``EMOTIONS`` order. Logistic regression gives real
        probabilities; for the SVM these are a softmax of the decision values
        (good for ranking and display, not calibrated)."""
        clf = self.pipe.named_steps["clf"]
        X = self.pipe.named_steps["tfidf"].transform(list(texts))
        raw = clf.predict_proba(X) if hasattr(clf, "predict_proba") else softmax(clf.decision_function(X), axis=1)
        order = [list(clf.classes_).index(e) for e in EMOTIONS]
        return raw[:, order]

    def top_features(self, k: int = 10) -> dict[str, list[str]]:
        """Highest-weighted n-grams per emotion -- for the interpretation slide."""
        names = self.pipe.named_steps["tfidf"].get_feature_names_out()
        clf = self.pipe.named_steps["clf"]
        out = {}
        for i, e in enumerate(clf.classes_):
            top = np.argsort(clf.coef_[i])[::-1][:k]
            out[e] = [names[j] for j in top]
        return out

    def save(self, path):
        joblib.dump({"cfg": self.cfg, "pipe": self.pipe}, path)

    @classmethod
    def load(cls, path) -> "TfidfBaseline":
        obj = joblib.load(path)
        m = cls(obj["cfg"])
        m.pipe = obj["pipe"]
        return m
