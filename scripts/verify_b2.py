"""Audit full-run artifacts against shared data and hand-in contracts."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from emotionshift.data.labels import EMOTIONS
from emotionshift.data.loaders import load
from emotionshift.eval import metrics as M


def main():
    count = 0
    required = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion",
                "pred", "stratum", "shift_type", "model", "seed"]
    for dataset, splits in {"meld": ["test"], "dailydialog": ["test_dedup", "test"]}.items():
        df = load(dataset)
        report = ROOT / "reports" / "b2" / dataset
        dev = pd.read_csv(report / "dev_comparison.csv")
        assert len(dev) == 10
        selection = json.loads((report / "selection.json").read_text())
        assert not selection["test_used_for_selection"]
        means = dev.groupby("model", sort=False).dev_score.mean().sort_values(ascending=False, kind="stable")
        assert selection["selected_model"] == means.index[0]
        for arch in ["gru", "lstm"]:
            name = f"b2_bi{arch}"
            summary = json.loads((report / name / "summary.json").read_text())
            assert summary["seeds"] == list(range(5)) and not summary["quick_smoke_only"]
            for split in splits:
                expected = df[df.split == split].reset_index(drop=True)
                result = summary["test"][split]
                assert result["n"] == len(expected) and result["bootstrap_resamples"] == 1000
                for seed in range(5):
                    path = ROOT / "predictions" / f"{name}_{dataset}_{split}_seed{seed}.csv"
                    pred = pd.read_csv(path, keep_default_na=False)
                    assert list(pred.columns) == required and len(pred) == len(expected)
                    for col in ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion", "stratum", "shift_type"]:
                        assert pred[col].astype(str).tolist() == expected[col].fillna("").astype(str).tolist(), (path, col)
                    assert set(pred.pred) <= set(EMOTIONS)
                    assert set(pred.model) == {name} and set(pred.seed) == {seed}
                    scores = M.scores(pred.emotion, pred.pred)
                    stored = next(r for r in result["per_seed"] if r["seed"] == seed)
                    for metric in ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]:
                        assert np.isclose(scores[metric], stored[metric], rtol=0, atol=1e-12)
                    ckpt = ROOT / "models" / f"{name}_{dataset}_seed{seed}.pt"
                    assert 0 < ckpt.stat().st_size < 50_000_000
                    history = pd.read_csv(report / name / f"history_seed{seed}.csv")
                    devrow = dev[(dev.model == name) & (dev.seed == seed)].iloc[0]
                    assert devrow.best_epoch == int(history.loc[history[f"dev_{M.HEADLINE[dataset]}"].idxmax(), "epoch"])
                    count += 1
    for path in (ROOT / "reports/b2/figures").glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1600
        assert path.with_suffix(".csv").exists()
    nb = nbformat.read(ROOT / "notebooks/b2_bigru_bilstm.ipynb", as_version=4)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    output = {"status": "passed", "prediction_csvs": count, "expected_prediction_csvs": 30,
              "executed_notebook_cells": len(cells), "shared_data_rows_and_metrics_verified": True,
              "checkpoint_limit_mb": 50, "seeds": list(range(5))}
    assert count == 30
    (ROOT / "reports/b2/verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
