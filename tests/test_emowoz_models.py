"""Native labels, missing-class probability alignment and checkpoint compatibility."""

import joblib
import numpy as np
import pytest

from emotionshift.data.labels import EMOTIONS, EMOWOZ_EMOTIONS
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig


def test_svm_native_columns_and_roundtrip(tmp_path):
    texts = [
        "plain ordinary",
        "very frightened",
        "poor service",
        "my apologies",
        "you idiot",
        "so exciting",
        "thank you satisfied",
    ] * 3
    model = TfidfBaseline(TfidfConfig(features="word", min_df=1), labels=EMOWOZ_EMOTIONS)
    model.fit(texts, EMOWOZ_EMOTIONS * 3)
    p = model.predict_proba(texts)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert model.predict(texts) == [EMOWOZ_EMOTIONS[i] for i in p.argmax(axis=1)]
    path = tmp_path / "native.joblib"
    model.save(path)
    loaded = TfidfBaseline.load(path)
    assert loaded.labels == EMOWOZ_EMOTIONS
    np.testing.assert_allclose(loaded.predict_proba(texts), p)


def test_svm_missing_classes_and_legacy_checkpoint(tmp_path):
    texts = ["plain words", "happy joy"] * 3
    model = TfidfBaseline(TfidfConfig(features="word", min_df=1)).fit(texts, ["neutral", "joy"] * 3)
    p = model.predict_proba(texts)
    assert p.shape == (6, 7)
    assert (p[:, 2:] == 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    path = tmp_path / "legacy.joblib"
    joblib.dump({"cfg": model.cfg, "pipe": model.pipe}, path)
    loaded = TfidfBaseline.load(path)
    assert loaded.labels == EMOTIONS
    np.testing.assert_allclose(loaded.predict_proba(texts), p)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_rnn_native_selection_and_checkpoint(architecture, tmp_path):
    torch = pytest.importorskip("torch")
    from emotionshift.models.birnn import BiRNNBaseline, BiRNNConfig

    torch.set_num_threads(1)
    cfg = BiRNNConfig(
        architecture=architecture,
        embedding_dim=8,
        hidden_dim=8,
        epochs=1,
        min_freq=1,
        dropout=0,
        batch_size=7,
    )
    texts = ["ordinary", "afraid", "bad service", "sorry", "idiot", "exciting", "thanks"] * 2
    model = BiRNNBaseline(cfg, labels=EMOWOZ_EMOTIONS).fit(
        texts,
        EMOWOZ_EMOTIONS * 2,
        ["devonlytoken", "thanks"],
        ["neutral", "satisfied"],
        metric="weighted_f1_no_neutral",
    )
    assert "devonlytoken" not in model.vocab
    assert model.best_dev_score is not None
    p = model.predict_proba(texts)
    assert model.predict(texts) == [EMOWOZ_EMOTIONS[i] for i in p.argmax(axis=1)]
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = BiRNNBaseline.load(path)
    assert loaded.labels == EMOWOZ_EMOTIONS
    np.testing.assert_allclose(loaded.predict_proba(texts), p)
