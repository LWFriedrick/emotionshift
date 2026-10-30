"""Create figures, hand-in text, and an executed walkthrough from full B2 runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs_quick" / "b2" / "mplconfig"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "outputs_quick" / "b2" / "jupyter_runtime"))
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient

REPORT = ROOT / "reports" / "b2"
HANDIN = ROOT / "deliverables" / "03_thanh"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    os.chdir(ROOT)
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    HANDIN.mkdir(exist_ok=True)
    (HANDIN / "figures").mkdir(exist_ok=True)
    rows, summaries, selected = [], [], {}
    for dataset, split, metric in [("meld", "test", "weighted_f1"),
                                    ("dailydialog", "test_dedup", "micro_f1_no_neutral")]:
        selected[dataset] = read_json(REPORT / dataset / "selection.json")["selected_model"]
        for arch in ("gru", "lstm"):
            name = f"b2_bi{arch}"
            s = read_json(REPORT / dataset / name / "summary.json")
            summaries.append(s)
            r = s["test"][split]
            lo, hi = r[metric]["ci95_first_seed"]
            rows.append(dict(dataset=dataset, model=name, split=split, metric=metric,
                             n=r["n"], mean=r[metric]["mean"], std=r[metric]["std"],
                             ci95_seed0_low=lo, ci95_seed0_high=hi,
                             selected_on_dev=s["selected_architecture"]))
    headline = pd.DataFrame(rows)
    headline.to_csv(figures / "headline.csv", index=False)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, dataset, title in zip(axes, ("meld", "dailydialog"),
                                  ("MELD test: weighted F1", "DailyDialog test_dedup: micro F1 excl. neutral")):
        part = headline[headline.dataset == dataset]
        ax.bar(["BiGRU", "BiLSTM"], part["mean"] * 100, yerr=part["std"] * 100,
               capsize=5, color=["#2563eb", "#0f766e"])
        baseline = read_json(ROOT / "reports" / "b1" / dataset / "summary.json")
        split = "test" if dataset == "meld" else "test_dedup"
        metric = "weighted_f1" if dataset == "meld" else "micro_f1_no_neutral"
        b1 = baseline["test"][split][metric]["mean"] * 100
        ax.axhline(b1, color="#b45309", linestyle="--", label=f"Repository B1: {b1:.2f}")
        for x, value in enumerate(part["mean"] * 100):
            ax.text(x, value + 2, f"{value:.2f}", ha="center")
        ax.set(title=title, ylabel="F1 (%)", ylim=(0, 100))
        ax.legend(loc="upper right", fontsize=10)
    fig.suptitle("Word-level recurrent baselines | mean and standard deviation, 5 seeds", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "headline.png", dpi=180)
    plt.close(fig)
    strata_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, dataset, split, metric in zip(axes, ("meld", "dailydialog"), ("test", "test_dedup"),
                                         ("weighted_f1", "micro_f1_no_neutral")):
        for offset, (arch, color) in enumerate((("gru", "#2563eb"), ("lstm", "#0f766e"))):
            name = f"b2_bi{arch}"
            part = pd.read_csv(REPORT / dataset / name / f"stratified_{split}.csv")
            part = part.set_index("group").loc[["stratum:first", "stratum:stable", "stratum:shift"]]
            for group, row in part.iterrows():
                strata_rows.append({"dataset": dataset, "model": name, "split": split,
                                    "group": group, "n": int(row["n"]), "metric": metric, "f1": row[metric]})
            xs = [i + (offset - 0.5) * 0.36 for i in range(3)]
            ax.bar(xs, part[metric] * 100, width=0.36, label=f"Bi{arch.upper()}", color=color)
            for x, value in zip(xs, part[metric] * 100):
                ax.text(x, value + 1, f"{value:.1f}", ha="center", fontsize=9)
        ticks = [f"{label}\nn={int(n):,}" for label, n in zip(["First", "Stable", "Shift"], part["n"])]
        ax.set(xticks=range(3), xticklabels=ticks, ylabel="F1 (%)", ylim=(0, 100),
               title=f"{dataset}: {metric}")
        ax.legend()
    fig.suptitle("Emotion recognition by same-speaker gold stratum (mean across 5 seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "strata.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(strata_rows).to_csv(figures / "strata.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    histories = []
    for ax, dataset in zip(axes, ("meld", "dailydialog")):
        metric = "weighted_f1" if dataset == "meld" else "micro_f1_no_neutral"
        for arch in ("gru", "lstm"):
            name = f"b2_bi{arch}"
            h = pd.read_csv(REPORT / dataset / name / "history_seed0.csv")
            histories.append(h.assign(dataset=dataset, model=name, seed=0))
            ax.plot(h.epoch, h[f"dev_{metric}"] * 100, marker="o", label=f"Bi{arch.upper()}")
        ax.set(title=f"{dataset}: dev {metric}", xlabel="Epoch", ylabel="F1 (%)")
        ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Development scores by epoch (seed 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "dev_learning.png", dpi=180)
    plt.close(fig)
    pd.concat(histories).to_csv(figures / "dev_learning.csv", index=False)
    for path in figures.iterdir():
        (HANDIN / "figures" / path.name).write_bytes(path.read_bytes())
    (HANDIN / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    notes = ["# B2 measured results", "", "Five seeds (0-4); architecture selected on mean dev score.",
             "Reported CI is the seed-0 dialogue bootstrap, not an interval over seeds.", ""]
    strata_table = pd.DataFrame(strata_rows)
    for dataset in ("meld", "dailydialog"):
        row = headline[(headline.dataset == dataset) & headline.selected_on_dev].iloc[0]
        baseline = read_json(ROOT / "reports" / "b1" / dataset / "summary.json")
        reference = baseline["test"][row.split][row.metric]["mean"]
        part = strata_table[(strata_table.dataset == dataset) & (strata_table.model == row.model)].set_index("group")
        stable, shift = part.loc["stratum:stable", "f1"], part.loc["stratum:shift", "f1"]
        notes += [f"## {dataset}", "",
                  f"Dev-selected model: {row.model}. Headline {row.metric} on {row.split}: "
                  f"{row['mean']*100:.2f}% +/- {row['std']*100:.2f} percentage points across seeds. "
                  f"Seed-0 95% dialogue-bootstrap CI: [{row.ci95_seed0_low*100:.2f}, {row.ci95_seed0_high*100:.2f}]%.", "",
                  f"Existing repository B1 reference: {reference*100:.2f}%; B2 minus B1 = "
                  f"{(row['mean']-reference)*100:+.2f} percentage points. B1 was not rerun.", "",
                  f"Stable turns: {stable*100:.2f}%; shift turns: {shift*100:.2f}%; "
                  f"stable minus shift: {(stable-shift)*100:.2f} percentage points. These are "
                  "means across seeds of the dataset's headline metric, using gold strata for evaluation only.", ""]
    notes += ["## CPU efficiency", "", "| Dataset | Model | Parameters | Checkpoint MB | Mean fit seconds | Latency median / p95 ms |",
              "|---|---|---|---|---|---|"]
    total_fit = 0.0
    for summary in summaries:
        e = summary["efficiency"]
        lat = e["latency_one_utterance"]
        total_fit += sum(e["train_seconds"]["per_seed"])
        notes.append(f"| {summary['dataset']} | {summary['model']} | {e['parameters']:,} | "
                     f"{e['model_file_mb']:.2f} | {e['train_seconds']['mean']:.1f} | "
                     f"{lat['median_ms']:.2f} / {lat['p95_ms']:.2f} |")
    notes += ["", f"All 20 fits together took {total_fit/60:.1f} minutes on this CPU; "
              "test inference, bootstrap reporting and artifact generation take additional time.", "",
              "## Interpretation limits", "", "Both models learn embeddings from scratch and observe only "
              "the current utterance (up to 64 tokens). They use no conversation history, speakers, audio or images. "
              "A stable/shift gap is descriptive and does not establish that context will improve a model. "
              "DailyDialog micro F1 excludes neutral, so differences in emotion prevalence between strata "
              "also affect the comparison; a higher shift score is not evidence of shift forecasting. "
              "Softmax probabilities have not been calibrated.", "",
              "Figures and matching CSVs are in figures/. Code snippets are named in the executed notebook. "
              "Thanh will prepare the slides and video separately."]
    notes_text = "\n".join(notes) + "\n"
    (REPORT / "analysis.md").write_text(notes_text, encoding="utf-8")
    (HANDIN / "results_notes.md").write_text(notes_text, encoding="utf-8")
    (HANDIN / "contribution.txt").write_text(
        "Draft for Thanh to review and adopt only after reviewing the work:\n"
        "Implemented utterance-level BiGRU and BiLSTM models and the shared demo interface.\n"
        "Ran train/dev/test experiments on MELD and DailyDialog with five seeds.\n"
        "Produced prediction CSVs, dialogue bootstrap CIs, shift analyses and efficiency measurements.\n"
        "Prepared an executed notebook, figures and result notes with AI assistance.\n", encoding="utf-8")
    (HANDIN / "README.md").write_text(
        "# Thanh workstream hand-in\n\nCode and results are in the normal repository folders. "
        "Use docs/B2_BIGRU_BILSTM.md as the entry point.\n\n"
        "results_notes.md explains the measured results and limitations. Figures have matching CSVs. "
        "Thanh will prepare the slides and video separately.\n\n"
        "Human items remaining: review the contribution draft; record own-voice narration; "
        "fill in and sign the AI disclosure, group reflection and peer evaluation. "
        "The team guide requires disclosing this complete unedited AI chat and confirming tool eligibility.\n",
        encoding="utf-8")
    make_notebook()


def make_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md("# B2: BiGRU / BiLSTM executed walkthrough\n\nFull training: `python scripts/train_b2.py`. "
                "This walkthrough inspects those completed runs and executes the saved demo model. "
                "It does not launch training again or choose settings from test results."),
        code('from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nif not (ROOT / "emotionshift").exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nimport torch\nfrom IPython.display import display, Image\nfrom emotionshift.data.loaders import load\nfrom emotionshift.data.labels import EMOTIONS\nfrom emotionshift.models.birnn import BiRNNBaseline\ntorch.set_num_threads(4)\nprint("PyTorch", torch.__version__, "labels:", EMOTIONS)'),
        md("## Shared data and splits\nThe official loader supplies labels, same-speaker strata and DailyDialog test_dedup."),
        code('counts = []\nfor dataset in ["meld", "dailydialog"]:\n    df = load(dataset)\n    table = df.groupby("split").agg(utterances=("text", "size"), dialogues=("dialogue_id", "nunique"))\n    counts.append(table.assign(dataset=dataset).reset_index())\ndisplay(pd.concat(counts, ignore_index=True))'),
        md("## Architecture and dev checkpoint selection\nTrain-only vocabulary; learned embeddings; packed sequences; "
           "bidirectional final states; dropout; linear head. GRU uses reset/update gates, "
           "while LSTM also maintains a cell state. All hyperparameters were fixed before test scoring."),
        code('for dataset in ["meld", "dailydialog"]:\n    report = ROOT / "reports/b2" / dataset\n    display(pd.read_csv(report / "dev_comparison.csv"))\n    print(dataset, json.loads((report / "selection.json").read_text()))'),
        code('display(Image(filename=str(ROOT / "reports/b2/figures/dev_learning.png")))'),
        md("## Results and uncertainty\nMeans and population standard deviations use five seeds. "
           "The 95% intervals resample whole dialogues 1,000 times for seed 0 and are not intervals over seeds. "
           "DailyDialog test_dedup is the headline; official test is a secondary comparison."),
        code('display(pd.read_csv(ROOT / "reports/b2/figures/headline.csv"))\ndisplay(Image(filename=str(ROOT / "reports/b2/figures/headline.png")))'),
        md("## Shared prediction export\nVideo snippet: the following cell, at most 15 lines."),
        code('selection = json.loads((ROOT / "reports/b2/meld/selection.json").read_text())\nname = selection["selected_model"]\npath = ROOT / "predictions" / f"{name}_meld_test_seed0.csv"\npredictions = pd.read_csv(path)\nexpected = ["dataset", "split", "dialogue_id", "turn", "speaker", "text",\n            "emotion", "pred", "stratum", "shift_type", "model", "seed"]\nassert predictions.columns.tolist() == expected\nassert len(predictions) == 2610\nassert set(predictions.pred) <= set(EMOTIONS)\ndisplay(predictions.head(3))'),
        md("## Demo interface\nVideo snippet: the following cell, at most 15 lines. "
           "These invented utterances are interface examples, not an accuracy estimate."),
        code('model = BiRNNBaseline.load(ROOT / "models" / f"{name}_meld_seed0.pt")\ntexts = ["Oh my God!", "Yeah.", "I am so sorry."]\nprint(model.network)\nproba = model.predict_proba(texts)\ndisplay(pd.DataFrame(proba, columns=EMOTIONS).assign(text=texts, pred=model.predict(texts)))\nprint("Checkpoint epoch:", model.best_epoch, "dev score:", model.best_dev_score)'),
        md("## Same-speaker shift analysis\nGold emotion/shift labels are used only for evaluation. "
           "Inference accepts text alone; derived shift detection compares current and previous predicted emotions."),
        code('display(pd.read_csv(ROOT / "reports/b2/figures/strata.csv"))\ndisplay(Image(filename=str(ROOT / "reports/b2/figures/strata.png")))'),
        md("## Efficiency and limits\nModels learn embeddings from scratch and truncate at 64 tokens. "
           "They omit conversation history, speakers, audio and images. Probability outputs are softmax values, "
           "not calibrated confidence estimates. Exact timings depend on CPU and thread settings."),
        code('for dataset in ["meld", "dailydialog"]:\n    for arch in ["gru", "lstm"]:\n        s = json.loads((ROOT / f"reports/b2/{dataset}/b2_bi{arch}/summary.json").read_text())\n        e = s["efficiency"]\n        print(dataset, arch, {k: e[k] for k in ["parameters", "model_file_mb", "train_seconds", "latency_one_utterance"]})'),
        md("## References\nSee `deliverables/03_thanh/refs.txt` for APA references to GRU, LSTM, bidirectional RNN, MELD and DailyDialog.\n\n"
           "AI assistance was used for implementation and preparation. Review, explain and disclose it per the team guide.")]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    notebook_path = ROOT / "notebooks" / "b2_bigru_bilstm.ipynb"
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, notebook_path)
    print("Executed", notebook_path)


if __name__ == "__main__":
    main()
