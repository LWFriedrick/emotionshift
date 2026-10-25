"""Data checks from the proposal's "Engineering" section (tests/)."""

import pytest
from sklearn.metrics import f1_score

from emotionshift.data.labels import EMOTIONS
from emotionshift.data.loaders import dailydialog_leakage, load_dailydialog, load_meld
from emotionshift.data.shifts import shift_type
from emotionshift.eval.metrics import scores


@pytest.fixture(scope="module")
def meld():
    return load_meld()


@pytest.fixture(scope="module")
def dd():
    return load_dailydialog()


def test_meld_split_sizes(meld):
    assert meld.groupby("split").size().to_dict() == {"train": 9989, "dev": 1109, "test": 2610}


def test_shift_counts_match_meld_fr(meld):
    # Proposal: our counts 4,001 / 427 / 1,003 vs MELD-FR flips 4,000 / 426 / 1,002.
    counts = meld[meld.stratum == "shift"].groupby("split").size().to_dict()
    assert counts == {"train": 4001, "dev": 427, "test": 1003}
    flips = {"train": 4000, "dev": 426, "test": 1002}
    assert all(abs(counts[s] - flips[s]) <= 1 for s in flips)


def test_meld_test_strata_and_types(meld):
    t = meld[meld.split == "test"]
    assert t.stratum.value_counts().to_dict() == {"shift": 1003, "stable": 861, "first": 746}
    assert t.shift_type.value_counts().to_dict() == {
        "onset": 355, "offset": 340, "surprise": 158, "within_negative": 81, "valence_flip": 69}


def test_shift_type_rules():
    assert shift_type("neutral", "surprise") == "onset"
    assert shift_type("surprise", "neutral") == "offset"
    assert shift_type("joy", "surprise") == "surprise"
    assert shift_type("anger", "sadness") == "within_negative"
    assert shift_type("joy", "fear") == "valence_flip"
    assert shift_type("joy", "joy") is None


def test_labels_in_shared_space(meld, dd):
    assert set(meld.emotion) <= set(EMOTIONS)
    assert set(dd.emotion) <= set(EMOTIONS)


def test_dailydialog_splits_and_leakage(dd):
    assert dd[dd.split != "test_dedup"].groupby("split").dialogue_id.nunique().to_dict() == {
        "train": 11118, "dev": 1000, "test": 1000}
    leak = dailydialog_leakage()
    assert leak["unique_test_dialogues"] == 996
    assert leak["unique_test_verbatim_in_train"] == 105


def test_no_overlap_after_dedup(dd):
    def dialogues(split):
        x = dd[dd.split == split].sort_values(["dialogue_id", "turn"])
        return {tuple(g.text) for _, g in x.groupby("dialogue_id")}
    dedup = dialogues("test_dedup")
    assert len(dedup) == 891
    assert not dedup & dialogues("train")


def test_pilot_numbers(meld, dd):
    t = meld[meld.split == "test"]
    assert round(scores(t.emotion, ["neutral"] * len(t))["weighted_f1"], 3) == 0.313
    assert round(scores(t.emotion, t.prev_emotion.fillna("neutral"))["weighted_f1"], 3) == 0.449
    e = t[t.stratum != "first"]
    assert round((e.stratum == "shift").mean(), 3) == 0.538
    d = dd[dd.split == "test"]
    assert round(scores(d.emotion, d.prev_emotion.fillna("neutral"))["micro_f1_no_neutral"], 3) == 0.397
    assert round((d[d.stratum != "first"].stratum == "shift").mean(), 3) == 0.180


def test_metrics_match_sklearn(meld):
    t = meld[meld.split == "test"]
    pred = t.prev_emotion.fillna("neutral")
    s = scores(t.emotion, pred)
    assert s["weighted_f1"] == pytest.approx(f1_score(t.emotion, pred, average="weighted"))
    assert s["macro_f1"] == pytest.approx(f1_score(t.emotion, pred, labels=EMOTIONS, average="macro"))
