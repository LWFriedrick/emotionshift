"""Audit the EmoWOZ B1/B2 extension, plot source figures and execute its notebook."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from nbclient import NotebookClient

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import load
from emotionshift.eval import efficiency as E
from emotionshift.eval import metrics as M
from emotionshift.models.birnn import BiRNNBaseline
from emotionshift.models.tfidf import TfidfBaseline
from scripts.train_b0 import PRED_COLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/emowoz_baselines"
LABELS = emotions("emowoz")
MODELS = {
    "b1": ROOT / "reports/b1/emowoz",
    "b2_bigru": ROOT / "reports/b2/emowoz/b2_bigru",
    "b2_bilstm": ROOT / "reports/b2/emowoz/b2_bilstm",
}


def audit():
    for family in ["b1", "b2"]:
        manifest = json.loads((OUT / f"{family}_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["source_sha256_lf"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            assert actual == expected, name
        for name, expected in manifest["raw_sha256"].items():
            path = ROOT / "data/raw/emowoz" / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    targets = load("emowoz", splits=["test"])
    targets = targets[targets.emotion.notna()].reset_index(drop=True)
    assert len(targets) == 8634 and targets.speaker.eq("user").all()
    rows, files, hashes = [], 0, {}
    for name, report in MODELS.items():
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        assert summary["labels"] == LABELS and summary["seeds"] == list(range(5))
        selection = summary["selection"]
        assert not selection["test_used_for_selection"]
        stored = pd.read_csv(report / "per_seed.csv").set_index("seed")
        actual_scores = []
        for seed in range(5):
            path = ROOT / "predictions" / f"{name}_emowoz_test_seed{seed}.csv"
            frame = pd.read_csv(path, keep_default_na=False)
            assert frame.columns[:12].tolist() == PRED_COLS
            assert len(frame) == len(targets)
            assert not frame.duplicated(["dataset", "split", "dialogue_id", "turn"]).any()
            for col in [
                "dataset",
                "split",
                "dialogue_id",
                "turn",
                "speaker",
                "text",
                "emotion",
                "stratum",
                "shift_type",
            ]:
                assert (
                    frame[col].astype(str).tolist() == targets[col].fillna("").astype(str).tolist()
                ), (
                    path,
                    col,
                )
            assert frame.pred.isin(LABELS).all()
            assert frame.model.eq(name).all() and frame.seed.eq(seed).all()
            score = M.scores(frame.emotion, frame.pred, LABELS)
            for metric in M.METRICS:
                assert np.isclose(score[metric], stored.loc[seed, metric])
            shift = M.derived_shift_detection(frame)
            assert np.isclose(shift["shift_f1"], stored.loc[seed, "derived_shift_f1"])
            actual_scores.append(score)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name.startswith("b2"):
                checkpoint = ROOT / "models" / f"{name}_emowoz_seed{seed}.pt"
                assert checkpoint.stat().st_size < 50_000_000
                history = pd.read_csv(report / f"history_seed{seed}.csv")
                record = json.loads((report / f"training_seed{seed}.json").read_text())
                assert record["best_epoch"] == int(
                    history.loc[history.dev_weighted_f1_no_neutral.idxmax(), "epoch"]
                )
            files += 1
        result = summary["test"]["test"]
        for metric in M.METRICS:
            assert np.isclose(np.mean([x[metric] for x in actual_scores]), result[metric]["mean"])
        headline = result[M.HEADLINE["emowoz"]]
        rows.append(
            {
                "model": name,
                "n": len(targets),
                "headline": headline["mean"],
                "std": headline["std"],
                "ci_low": headline["ci95_seed0"][0],
                "ci_high": headline["ci95_seed0"][1],
                "macro_f1_no_neutral": result["macro_f1_no_neutral"]["mean"],
                "macro_f1": result["macro_f1"]["mean"],
                "derived_shift_f1": result["derived_shift_f1"],
            }
        )
    dev = pd.read_csv(ROOT / "reports/b2/emowoz/dev_comparison.csv")
    selection = json.loads((ROOT / "reports/b2/emowoz/selection.json").read_text())
    means = (
        dev.groupby("model", sort=False)
        .dev_score.mean()
        .sort_values(ascending=False, kind="stable")
    )
    assert selection["selected_model"] == means.index[0]
    assert len(dev) == 10
    svm = pd.read_csv(ROOT / "reports/b1/emowoz/dev_grid.csv")
    selection_b1 = json.loads((ROOT / "reports/b1/emowoz/selection.json").read_text())
    assert len(svm) == 16 and np.isclose(
        selection_b1["dev_score"], svm.weighted_f1_no_neutral.max()
    )
    # Verify checkpoint label order and prediction round trip on a compact probe.
    probes = targets.text.iloc[:16]
    for name in MODELS:
        if name == "b1":
            model = TfidfBaseline.load(ROOT / "models/b1_emowoz.joblib")
        else:
            model = BiRNNBaseline.load(ROOT / "models" / f"{name}_emowoz_seed0.pt")
        assert model.labels == LABELS
        proba = model.predict_proba(probes)
        assert proba.shape == (len(probes), 7)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
        expected = (
            pd.read_csv(ROOT / "predictions" / f"{name}_emowoz_test_seed0.csv")
            .pred.iloc[:16]
            .tolist()
        )
        assert model.predict(probes) == expected
        report = MODELS[name]
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        if name == "b1":
            classifier = model.pipe.named_steps["clf"]
            parameters = classifier.coef_.size + classifier.intercept_.size
            checkpoint = ROOT / "models/b1_emowoz.joblib"
        else:
            parameters = sum(p.numel() for p in model.network.parameters())
            checkpoint = ROOT / "models" / f"{name}_emowoz_seed0.pt"
        summary["efficiency"].update(
            parameters=int(parameters), model_file_mb=checkpoint.stat().st_size / 1e6
        )
        # Measure the three trained checkpoints together in one CPU runtime.
        summary["efficiency"].update(
            latency_one_utterance=E.latency_ms(model.predict, targets.text, n=100),
            batch_throughput=E.throughput(model.predict, targets.text),
            machine=E.machine(),
            inference_device="cpu",
            inference_torch=torch.__version__,
            timing_note="Trained checkpoints, same CPU runtime, after all training completed",
        )
        (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    (OUT / "verification.json").write_text(
        json.dumps(
            {
                "prediction_csvs": files,
                "targets_per_csv": len(targets),
                "seeds": list(range(5)),
                "schema_and_native_labels_verified": True,
                "stored_metrics_recomputed": True,
                "dev_selection_verified": True,
                "checkpoint_round_trips_verified": True,
                "prediction_sha256": hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return pd.DataFrame(rows)


def figures(comparison):
    out = OUT / "figures"
    out.mkdir(exist_ok=True)
    comparison.to_csv(out / "headline_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, r in enumerate(comparison.itertuples(index=False)):
        ax.barh(i, r.headline * 100, color="#466b8e")
        # CI belongs to seed 0 rather than the seed mean; plot endpoints directly.
        ax.plot([r.ci_low * 100, r.ci_high * 100], [i, i], color="black", linewidth=2)
        ax.text(r.headline * 100 + 1, i, f"{r.headline * 100:.2f}", va="center")
    ax.set_yticks(range(len(comparison)), comparison.model)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Native EmoWOZ weighted-F1 excluding neutral (%)")
    ax.set_title("In-domain EmoWOZ: five-seed means; dialogue-bootstrap CI for seed 0")
    fig.savefig(out / "headline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    strata = []
    for name, report in MODELS.items():
        table = pd.read_csv(report / "stratified_test.csv")
        for group in ["first", "stable", "shift"]:
            row = table[table.group.eq("stratum:" + group)].iloc[0]
            strata.append(
                {
                    "model": name,
                    "group": group,
                    "weighted_f1_no_neutral": row.weighted_f1_no_neutral,
                }
            )
    strata = pd.DataFrame(strata)
    strata.to_csv(out / "strata.csv", index=False)
    pivot = strata.pivot(index="model", columns="group", values="weighted_f1_no_neutral") * 100
    ax = pivot.plot.bar(figsize=(10, 5), ylim=(0, 100), rot=0)
    ax.set_ylabel("Native EmoWOZ weighted-F1 excluding neutral (%)")
    ax.set_title("Mean first / stable / shift scores across five seeds")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "strata.png", dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    histories = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, name in zip(axes, ["b2_bigru", "b2_bilstm"], strict=True):
        for seed in range(5):
            history = pd.read_csv(MODELS[name] / f"history_seed{seed}.csv")
            histories.append(history.assign(model=name, seed=seed))
            (line,) = ax.plot(
                history.epoch, history.dev_weighted_f1_no_neutral * 100, label=f"seed {seed}"
            )
            best = history.loc[history.dev_weighted_f1_no_neutral.idxmax()]
            ax.scatter(best.epoch, best.dev_weighted_f1_no_neutral * 100, color=line.get_color())
        ax.set_title(name + " / dev only")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dev weighted-F1 excluding neutral (%)")
        ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.concat(histories, ignore_index=True).to_csv(out / "learning_curves.csv", index=False)


def notes(comparison):
    selection = json.loads((ROOT / "reports/b2/emowoz/selection.json").read_text())
    lines = [
        "# EmoWOZ extension: workstreams 2 and 3",
        "",
        "Native seven-class, in-domain experiments; 66,474 train / 8,509 dev / 8,634 test user targets.",
        "Unlabelled system context is not a target. No Ekman mapping or cross-domain transfer claim.",
        "",
        "| Model | Weighted-F1 excl. neutral | Population SD | 95% dialogue CI (seed 0) | Macro-F1 excl. neutral | Derived shift F1 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for r in comparison.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.headline * 100:.2f} | {r.std * 100:.2f} | "
            f"[{r.ci_low * 100:.2f}, {r.ci_high * 100:.2f}] | "
            f"{r.macro_f1_no_neutral * 100:.2f} | {r.derived_shift_f1 * 100:.2f} |"
        )
    lines += [
        "",
        f"Dev-selected recurrent architecture: **{selection['selected_model']}**. Both architectures are reported regardless of test ranking.",
        "",
        "## Protocol",
        "",
        "SVM: original word / word+char feature families; C = 0.1, 0.3, 1.0, 3.0; unweighted / balanced loss. Sixteen dev configurations, no logistic-regression search in this SVM workstream.",
        "RNN: reuse the existing word-level BiGRU/BiLSTM, learned embeddings, max 12 epochs, patience 3, train-only vocabulary and inverse-square-root class weights. No architecture or hyperparameter tuning on test.",
        "Five seeds: 0, 1, 2, 3, 4. Checkpoints use best native non-neutral dev weighted-F1. The primary RNN uses mean best-dev across seeds, frozen before reading test targets.",
        "All recognition metrics use the native label list. CIs use 1,000 whole-dialogue resamples for seed 0. Standard deviation describes training-seed variation; intervals do not.",
        "CPU inference efficiency is measured using loaded checkpoints and included per model. Training device/runtime is recorded in manifests; hardware differences limit timing comparisons.",
        "",
        "## Interpretation",
        "",
        "The two recurrent test means are close (about 0.04 percentage points apart). The reported single-model intervals do not establish a significant difference; no paired significance claim is made. The dev-selected architecture remains BiLSTM even though BiGRU has the slightly higher test mean.",
        "The recurrent models improve the headline score modestly over SVM in this run. Macro-F1 excluding neutral remains around 47-49%, well below non-neutral weighted-F1, indicating uneven performance across emotion classes. Per-class tables should accompany the headline score.",
        "SVM predictions are identical across the five configured seeds; its zero seed SD does not imply zero sampling uncertainty. Its dialogue-bootstrap interval remains nonzero.",
        "First/stable/shift groups have different label compositions. Their F1 values describe these subsets; a higher shift-subset F1 is not proof that an utterance-only model understands dialogue transitions. Derived shift F1 is reported separately.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "uv sync --locked --extra rnn",
        "uv run --extra rnn python -X utf8 scripts/train_emowoz_baselines.py --device cpu",
        "uv run --extra rnn python -X utf8 scripts/make_emowoz_baseline_artifacts.py",
        "```",
        "",
        "CUDA is optional. With a supported CUDA PyTorch build use its Python interpreter and `--device cuda`; see the runtime manifest. Completed RNN seeds can resume only if source/config/data/runtime signatures match. Epoch-interrupted seeds retrain from their fixed seed.",
        "",
        "## Files and attribution",
        "",
        "Existing B1/B2 model authors' implementations are reused. This contribution extends native-label probability/checkpoint handling, adds the EmoWOZ protocol, trains the models and prepares audited outputs.",
        "Original MELD/DailyDialog experiment configurations, predictions and reports are untouched. Old checkpoint label defaults remain compatible.",
        "`reports/b1/emowoz/` and `reports/b2/emowoz/` contain dev choices, per-seed scores, confusion/stratum tables, histories and summaries. Predictions retain the original twelve columns first, followed by provenance.",
        "Checkpoints are generated under `models/`; small B2 `.pt` checkpoints can be supplied for the demo. SVM joblib remains ignored like other B1 artifacts.",
        "`notebooks/emowoz_b1_b2.ipynb` is the executed walkthrough. Slides, own-voice narration and personal forms remain separate. AI assistance must be disclosed according to the team guide.",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(
        cells=[
            md(
                "# EmoWOZ — SVM, BiGRU and BiLSTM\nWorkstreams 2/3 in-domain extension using native labels. Training is run separately with `scripts/train_emowoz_baselines.py`. No test-driven configuration changes."
            ),
            code(
                "from pathlib import Path\nimport os\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nos.chdir(ROOT)\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image"
            ),
            md(
                "## Labelled targets and dev-only choices\nOnly user turns are scored; system turns are not neutral labels. RNN architecture selection is frozen before test evaluation."
            ),
            code(
                "from emotionshift.data.labels import emotions\nprint('Native labels:', emotions('emowoz'))\nfor path in ['reports/b1/emowoz/selection.json', 'reports/b2/emowoz/selection.json']:\n    display(json.loads(Path(path).read_text()))"
            ),
            md(
                "## Development learning curves\nDots mark each seed's best dev epoch, "
                "not the final training epoch. Architecture selection averages these "
                "best-dev scores; test results do not determine the checkpoint."
            ),
            code("display(Image(filename='reports/emowoz_baselines/figures/learning_curves.png'))"),
            md(
                "## Main test results\nFive-seed means and population SD; 1,000-resample dialogue CI for seed 0, not seed-mean uncertainty."
            ),
            code(
                "display(pd.read_csv('reports/emowoz_baselines/comparison.csv'))\ndisplay(Image(filename='reports/emowoz_baselines/figures/headline_comparison.png'))"
            ),
            md(
                "## First / stable / shift\nGold shift labels are evaluation metadata and are not model inputs."
            ),
            code(
                "display(pd.read_csv('reports/emowoz_baselines/figures/strata.csv'))\ndisplay(Image(filename='reports/emowoz_baselines/figures/strata.png'))"
            ),
            md(
                "## Checkpoint demo\nProbabilities follow the native EmoWOZ label order; SVM probabilities are softmax scores, not calibrated confidences."
            ),
            code(
                "from emotionshift.models.birnn import BiRNNBaseline\nselection = json.loads(Path('reports/b2/emowoz/selection.json').read_text())\nname = selection['selected_model']\nmodel = BiRNNBaseline.load(f'models/{name}_emowoz_seed0.pt')\ntexts = ['Thank you, that is exactly what I needed.', 'You are not listening to my request.']\nprint(model.labels)\nprint(model.predict(texts))\ndisplay(model.predict_proba(texts))"
            ),
            md(
                "## Verification and limits\nNative EmoWOZ scores are not numerically comparable to Ekman-label benchmarks. These are utterance-only models, without system or dialogue history. See results.md for source attribution and runtime details."
            ),
            code(
                "verification = json.loads(Path('reports/emowoz_baselines/verification.json').read_text())\nprint('Prediction CSVs audited:', verification['prediction_csvs'])\nprint('Dev selection verified:', verification['dev_selection_verified'])"
            ),
        ]
    )
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (EmotionShift)",
        "language": "python",
        "name": "python3",
    }
    assert all(len(c.source.splitlines()) <= 15 for c in nb.cells if c.cell_type == "code")
    notebook_path = ROOT / "notebooks/emowoz_b1_b2.ipynb"
    nbformat.write(nb, notebook_path)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(notebook_path)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(notebook_path)], check=True)
    nb = nbformat.read(notebook_path, as_version=4)
    assert all(len(c.source.splitlines()) <= 15 for c in nb.cells if c.cell_type == "code")
    os.environ["PYTHONUTF8"] = "1"
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "notebooks/emowoz_b1_b2.ipynb")


def main():
    torch.set_num_threads(4)
    comparison = audit()
    figures(comparison)
    notes(comparison)
    notebook()
    print("Audited 15 native-label prediction CSVs and executed the notebook.", flush=True)


if __name__ == "__main__":
    main()
