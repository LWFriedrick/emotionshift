"""Train B2 on shared splits; lock dev choices before any test scoring.

python scripts/train_b2.py
python scripts/train_b2.py --quick  # isolated smoke artifacts, NOT headline results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from emotionshift.data.loaders import load
from emotionshift.eval import efficiency as E, metrics as M
from emotionshift.models.birnn import BiRNNBaseline, BiRNNConfig

PRED_COLS = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion",
             "pred", "stratum", "shift_type", "model", "seed"]
METRICS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def run(dataset, cfg, base, quick, device):
    df = load(dataset)
    train, dev = (df[df.split == s] for s in ("train", "dev"))
    if quick:
        train = train.sample(min(256, len(train)), random_state=0)
        dev = dev.sample(min(128, len(dev)), random_state=0)
    metric = M.HEADLINE[dataset]
    dev_rows, checkpoints = [], {}
    # Complete training and freeze architecture/epoch choices before reading test labels.
    for arch in cfg["architectures"]:
        model_name = f"b2_bi{arch}"
        out = base / "reports" / "b2" / dataset / model_name
        out.mkdir(parents=True, exist_ok=True)
        for seed in cfg["seeds"]:
            model_cfg = BiRNNConfig(**cfg["model"], architecture=arch, seed=seed)
            t0 = time.perf_counter()
            model = BiRNNBaseline(model_cfg, device=device).fit(
                train.text, train.emotion, dev.text, dev.emotion, metric=metric)
            elapsed = time.perf_counter() - t0
            path = base / "models" / f"{model_name}_{dataset}_seed{seed}.pt"
            model.save(path)
            pd.DataFrame(model.history).to_csv(out / f"history_seed{seed}.csv", index=False)
            checkpoints[arch, seed] = (path, elapsed)
            dev_rows.append({"model": model_name, "architecture": arch, "seed": seed,
                             "best_epoch": model.best_epoch, "dev_score": model.best_dev_score,
                             "selection_metric": metric, "train_seconds": elapsed})
    dev_table = pd.DataFrame(dev_rows)
    dev_table.to_csv(base / "reports" / "b2" / dataset / "dev_comparison.csv", index=False)
    rank = dev_table.groupby("model", sort=False).dev_score.mean().sort_values(ascending=False, kind="stable")
    selected = str(rank.index[0])
    write_json(base / "reports" / "b2" / dataset / "selection.json",
               {"selected_model": selected, "metric": metric, "rule": "mean best-dev score across seeds; ties follow config order",
                "dev_means": rank.to_dict(), "test_used_for_selection": False})
    summaries = []
    for arch in cfg["architectures"]:
        name = f"b2_bi{arch}"
        out = base / "reports" / "b2" / dataset / name
        summary = {"dataset": dataset, "model": name, "selected_on_dev": metric,
                   "selected_architecture": name == selected, "config": {**cfg["model"], "architecture": arch},
                   "seeds": cfg["seeds"], "quick_smoke_only": quick, "test": {},
                   "dev": dev_table[dev_table.model == name].to_dict(orient="records")}
        predictions = {split: [] for split in cfg["test_splits"][dataset]}
        for seed in cfg["seeds"]:
            path, seconds = checkpoints[arch, seed]
            # Evaluate reloaded checkpoint on CPU, matching demo deployment.
            model = BiRNNBaseline.load(path)
            for split in predictions:
                frame = df[df.split == split].copy()
                if quick:
                    # Whole dialogues preserve derived same-speaker shift semantics.
                    ids = frame.dialogue_id.unique()[:5]
                    frame = frame[frame.dialogue_id.isin(ids)].copy()
                frame["pred"] = model.predict(frame.text)
                frame["model"], frame["seed"] = name, seed
                (base / "predictions").mkdir(parents=True, exist_ok=True)
                frame[PRED_COLS].to_csv(base / "predictions" / f"{name}_{dataset}_{split}_seed{seed}.csv", index=False)
                predictions[split].append(frame)
            if seed == cfg["seeds"][0]:
                texts = next(iter(predictions.values()))[0].text
                summary["efficiency"] = {
                    "parameters": sum(p.numel() for p in model.network.parameters()),
                    "vocabulary_size": len(model.vocab), "model_file_mb": E.file_size_mb(path),
                    "model_path": str(path.relative_to(base)).replace("\\", "/"),
                    "latency_one_utterance": E.latency_ms(model.predict, texts, n=50 if quick else 500),
                    "batch_throughput": E.throughput(model.predict, texts),
                    "machine": E.machine(), "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "inference_device": "cpu", "training_device": device,
                }
        summary["efficiency"]["train_seconds"] = {
            "per_seed": [checkpoints[arch, s][1] for s in cfg["seeds"]],
            "mean": float(np.mean([checkpoints[arch, s][1] for s in cfg["seeds"]]))}
        for split, runs in predictions.items():
            per_seed = [{"seed": seed, **M.scores(frame.emotion, frame.pred),
                         "derived_shift_detection": M.derived_shift_detection(frame)}
                        for seed, frame in zip(cfg["seeds"], runs)]
            result = {k: {"mean": float(np.mean([r[k] for r in per_seed])),
                          "std": float(np.std([r[k] for r in per_seed]))} for k in METRICS}
            first = runs[0]
            for k in METRICS[:3]:
                result[k]["ci95_first_seed"] = M.bootstrap_ci(first, k, n_boot=20 if quick else cfg["bootstrap"])
            result.update(n=len(first), per_seed=per_seed,
                          derived_shift_detection=per_seed[0]["derived_shift_detection"],
                          derived_shift_detection_seed=cfg["seeds"][0],
                          ci_seed=cfg["seeds"][0], bootstrap_unit="dialogue", bootstrap_resamples=20 if quick else cfg["bootstrap"])
            summary["test"][split] = result
            strata = pd.concat([M.stratified(f).assign(seed=s) for f, s in zip(runs, cfg["seeds"])])
            strata.to_csv(out / f"stratified_{split}_per_seed.csv", index=False)
            strata.groupby("group", sort=False)[["n", *METRICS]].mean().to_csv(out / f"stratified_{split}.csv")
            for seed, frame in zip(cfg["seeds"], runs):
                M.per_class_f1(frame.emotion, frame.pred).to_csv(out / f"per_class_{split}_seed{seed}.csv", index=False)
                M.confusion(frame.emotion, frame.pred).to_csv(out / f"confusion_{split}_seed{seed}.csv")
        write_json(out / "summary.json", summary)
        summaries.append(summary)
        print(f"Completed {dataset} {name}", flush=True)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "b2_birnn.yaml")
    ap.add_argument("--datasets", nargs="+", choices=["meld", "dailydialog"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["model"].update(epochs=2, patience=2, max_vocab=1000, embedding_dim=16, hidden_dim=16)
    torch.set_num_threads(cfg["threads"])
    base = ROOT / "outputs_quick" / "b2" if args.quick else ROOT
    summaries = []
    for dataset in args.datasets or cfg["datasets"]:
        summaries.extend(run(dataset, cfg, base, args.quick, args.device))
    lines = ["# B2 BiGRU / BiLSTM results", "", "Mean +/- population std across seeds; CI is 95% dialogue bootstrap of the first seed.",
             "DailyDialog headline: test_dedup. Architecture selection uses dev only.", "",
             "| Dataset | Model | Dev selected | Split | n | Weighted F1 | Macro F1 | Micro F1 excl. neutral |", "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for split, r in s["test"].items():
            values = [f"{r[k]['mean']:.4f} +/- {r[k]['std']:.4f} "
                      f"[{r[k]['ci95_first_seed'][0]:.4f}, {r[k]['ci95_first_seed'][1]:.4f}]"
                      for k in METRICS[:3]]
            lines.append(f"| {s['dataset']} | {s['model']} | {s['selected_architecture']} | {split} | {r['n']} | " + " | ".join(values) + " |")
    (base / "reports" / "b2" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_files = ["configs/b2_birnn.yaml", "emotionshift/models/birnn.py", "scripts/train_b2.py",
                    "emotionshift/data/loaders.py", "emotionshift/data/labels.py", "emotionshift/eval/metrics.py"]
    write_json(base / "reports" / "b2" / "run_manifest.json",
               {"config": cfg, "quick": args.quick, "python": platform.python_version(),
                "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in source_files},
                "raw_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((ROOT / "data" / "raw").rglob("*")) if p.is_file()}})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
