"""B1 model interface: what the demo and the comparison scripts rely on."""

import numpy as np
import pytest

from emotionshift.data.labels import EMOTIONS
from emotionshift.data.loaders import load_meld
from emotionshift.eval.efficiency import latency_ms
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig


@pytest.fixture(scope="module")
def model():
    df = load_meld()
    tr = df[df.split == "train"]
    return TfidfBaseline(TfidfConfig(classifier="svm", C=0.1, class_weight="balanced", features="word")).fit(tr.text, tr.emotion)


def test_predict_returns_shared_labels(model):
    out = model.predict(["Oh my God!", "Yeah.", "I'm so sorry."])
    assert len(out) == 3 and set(out) <= set(EMOTIONS)


def test_predict_proba_in_label_order(model):
    p = model.predict_proba(["Oh my God!", "Ew, gross."])
    assert p.shape == (2, len(EMOTIONS))
    assert np.allclose(p.sum(axis=1), 1)
    assert [EMOTIONS[i] for i in p.argmax(axis=1)] == model.predict(["Oh my God!", "Ew, gross."])


def test_save_load_round_trip(model, tmp_path):
    path = tmp_path / "b1.joblib"
    model.save(path)
    again = TfidfBaseline.load(path)
    texts = ["Get out!", "Wow, really?", "Okay."]
    assert again.predict(texts) == model.predict(texts)


def test_deterministic_across_seeds():
    df = load_meld()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    a = TfidfBaseline(TfidfConfig(C=0.1, class_weight="balanced", features="word", seed=0)).fit(tr.text, tr.emotion)
    b = TfidfBaseline(TfidfConfig(C=0.1, class_weight="balanced", features="word", seed=4)).fit(tr.text, tr.emotion)
    assert a.predict(te.text) == b.predict(te.text)


def test_demo_latency_budget(model):
    # Proposal R5: demo under 300 ms per turn on CPU. B1 alone should be far below.
    assert latency_ms(model.predict, ["Oh my God!", "Yeah.", "Get out of here!"] * 20, n=50)["p95_ms"] < 300
