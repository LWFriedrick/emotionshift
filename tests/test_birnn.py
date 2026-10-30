"""Behavioral contracts for both neural encoders and the demo checkpoint."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install emotionshift[rnn] for neural model tests")

from emotionshift.data.labels import EMOTIONS
from emotionshift.models.birnn import BiRNNBaseline, BiRNNConfig


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_fit_checkpoint_probability_order_and_padding(architecture, tmp_path):
    torch.set_num_threads(1)
    train = ["ordinary words", "happy day", "what surprise", "very angry", "so sad", "disgusting", "afraid"] * 3
    labels = list(EMOTIONS) * 3
    model = BiRNNBaseline(BiRNNConfig(architecture=architecture, embedding_dim=8,
        hidden_dim=8, min_freq=1, max_length=8, batch_size=7, epochs=2, dropout=0.0))
    model.fit(train, labels, ["devonlytoken", "happy day"], ["neutral", "joy"])
    assert "devonlytoken" not in model.vocab
    probes = ["", "happy", "ordinary words with a considerably longer sequence"]
    proba = model.predict_proba(probes)
    assert proba.shape == (3, 7)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert model.predict(probes) == [EMOTIONS[i] for i in proba.argmax(axis=1)]
    # Packing must prevent other examples' padding from affecting a prediction.
    np.testing.assert_allclose(model.predict_proba(["happy"])[0], proba[1], atol=1e-6)
    assert model.predict([]) == []
    assert model.predict_proba([]).shape == (0, 7)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = BiRNNBaseline.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(probes), proba)
    assert loaded.best_epoch in {1, 2}
    assert loaded.vocab == model.vocab


def test_invalid_training_and_unfitted_prediction():
    with pytest.raises(ValueError):
        BiRNNConfig(architecture="transformer")
    with pytest.raises(RuntimeError):
        BiRNNBaseline().predict(["hello"])
    with pytest.raises(ValueError):
        BiRNNBaseline().fit(["hello"], [], ["hi"], ["neutral"])


