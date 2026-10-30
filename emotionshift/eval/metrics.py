"""Metrics shared by every model (proposal: "Evaluation" section).

* MELD headline: weighted-F1 over the 7 classes.
* DailyDialog headline: micro-F1 *excluding neutral* (standard in the literature).
* Always also: macro-F1, per-class F1, confusion matrix.
* Stratified: the same metrics per stratum (first / stable / shift) and per
  shift type, with ``n`` in every cell.
* 95% bootstrap CIs, resampling whole dialogues (utterances in one dialogue
  are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from ..data.labels import EMOTIONS
from ..data.shifts import SHIFT_TYPES, STRATA

NON_NEUTRAL = [e for e in EMOTIONS if e != "neutral"]
HEADLINE = {"meld": "weighted_f1", "dailydialog": "micro_f1_no_neutral"}


def scores(gold, pred) -> dict:
    gold, pred = np.asarray(gold), np.asarray(pred)
    # Empty strata occur in small runs; newer sklearn rejects empty arrays.
    # Preserve zero_division=0 F1 semantics and undefined empty accuracy.
    if len(gold) == 0 and len(pred) == 0:
        return {"n": 0, "weighted_f1": 0.0, "macro_f1": 0.0,
                "micro_f1_no_neutral": 0.0, "accuracy": float("nan")}
    return {
        "n": int(len(gold)),
        "weighted_f1": f1_score(gold, pred, labels=EMOTIONS, average="weighted", zero_division=0),
        "macro_f1": f1_score(gold, pred, labels=EMOTIONS, average="macro", zero_division=0),
        "micro_f1_no_neutral": f1_score(gold, pred, labels=NON_NEUTRAL, average="micro", zero_division=0),
        "accuracy": float((gold == pred).mean()) if len(gold) else float("nan"),
    }


def per_class_f1(gold, pred) -> pd.DataFrame:
    f1 = f1_score(gold, pred, labels=EMOTIONS, average=None, zero_division=0)
    support = pd.Series(gold).value_counts().reindex(EMOTIONS, fill_value=0)
    return pd.DataFrame({"emotion": EMOTIONS, "f1": f1, "support": support.values})


def confusion(gold, pred) -> pd.DataFrame:
    cm = confusion_matrix(gold, pred, labels=EMOTIONS)
    return pd.DataFrame(cm, index=[f"gold_{e}" for e in EMOTIONS], columns=[f"pred_{e}" for e in EMOTIONS])


def stratified(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Scores per stratum and per shift type. ``df`` needs emotion, stratum,
    shift_type and ``pred_col``."""
    rows = [{"group": "all", **scores(df["emotion"], df[pred_col])}]
    for s in STRATA:
        sub = df[df["stratum"] == s]
        rows.append({"group": f"stratum:{s}", **scores(sub["emotion"], sub[pred_col])})
    for t in SHIFT_TYPES:
        sub = df[df["shift_type"] == t]
        rows.append({"group": f"shift_type:{t}", **scores(sub["emotion"], sub[pred_col])})
    return pd.DataFrame(rows)


def derived_shift_detection(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Task T2, *derived* variant: a predicted shift is a predicted label that
    differs from the speaker's previous *predicted* label (causal, no gold
    labels used). Scored on turns that have a previous same-speaker turn."""
    d = df.sort_values(["split", "dialogue_id", "turn"])
    prev_pred = d.groupby(["split", "dialogue_id", "speaker"], sort=False)[pred_col].shift(1)
    elig = d["stratum"] != "first"
    gold = (d.loc[elig, "stratum"] == "shift").astype(int)
    pred = (d.loc[elig, pred_col] != prev_pred[elig]).astype(int)
    return {
        "n": int(elig.sum()),
        "base_rate": float(gold.mean()),
        "shift_f1": f1_score(gold, pred, zero_division=0),
        "predicted_shift_rate": float(pred.mean()),
    }


def bootstrap_ci(df: pd.DataFrame, metric: str, pred_col: str = "pred",
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of ``metric``, resampling dialogues with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.values for k, g in df.groupby("dialogue_id")}
    keys = np.array(list(groups))
    gold, pred = df["emotion"].to_numpy(), df[pred_col].to_numpy()
    pos = {i: p for p, i in enumerate(df.index)}
    idx_by_key = [np.array([pos[i] for i in groups[k]]) for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx_by_key[p] for p in pick])
        vals.append(scores(gold[ix], pred[ix])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
