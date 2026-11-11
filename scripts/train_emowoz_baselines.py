"""In-domain native EmoWOZ extension of B1 SVM and B2 BiGRU/BiLSTM.

Existing MELD/DailyDialog runs and reports are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
import yaml

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import DATA_ROOT, load
from emotionshift.eval import efficiency as E
from emotionshift.eval import metrics as M
from emotionshift.models.birnn import BiRNNBaseline, BiRNNConfig
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig, _classifier, _features
from scripts.train_b0 import PRED_COLS, dialogue_intervals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/emowoz_baselines"
LABELS = emotions("emowoz")
METRIC = M.HEADLINE["emowoz"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def targets(splits):
    frame = load("emowoz", splits=splits)
    frame = frame[frame.emotion.notna()].copy()
    if not frame.speaker.eq("user").all() or not frame.emotion.isin(LABELS).all():
        raise ValueError("Expected native-labelled user targets")
    return frame


def protocol(cfg, family):
    sources = [
        "scripts/train_emowoz_baselines.py",
        "emotionshift/data/loaders.py",
        "emotionshift/data/labels.py",
        "emotionshift/eval/metrics.py",
        "emotionshift/models/tfidf.py" if family == "b1" else "emotionshift/models/birnn.py",
    ]
    return {
        "family": family,
        "config": cfg,
        "labels": LABELS,
        "selection_metric": METRIC,
        "python": platform.python_version(),
        "torch": torch.__version__ if family == "b2" else None,
        "source_sha256_lf": {p: digest(ROOT / p) for p in sources},
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((DATA_ROOT / "emowoz").glob("*.json"))
            if p.name != "download_manifest.json"
        },
    }


def evaluate(name, fitted, train_seconds, cfg, report, selection):
    # Test data is read only after dev choices have been persisted by the caller.
    test = targets(["test"])
    report.mkdir(parents=True, exist_ok=True)
    all_rows, strata, first, timing = [], [], None, None
    for seed in cfg["seeds"]:
        model = fitted[seed]
        if isinstance(model, Path):
            model = BiRNNBaseline.load(model, device="cpu")
        frame = test.copy()
        frame["pred"] = model.predict(frame.text)
        frame["model"], frame["seed"] = name, seed
        extra = [c for c in frame if c not in PRED_COLS]
        (ROOT / "predictions").mkdir(exist_ok=True)
        frame[PRED_COLS + extra].to_csv(
            ROOT / "predictions" / f"{name}_emowoz_test_seed{seed}.csv", index=False
        )
        all_rows.append(
            {
                "seed": seed,
                **M.scores(frame.emotion, frame.pred, LABELS),
                **{f"derived_{k}": v for k, v in M.derived_shift_detection(frame).items()},
            }
        )
        strata.append(M.stratified(frame, labels=LABELS).assign(seed=seed))
        M.per_class_f1(frame.emotion, frame.pred, LABELS).to_csv(
            report / f"per_class_test_seed{seed}.csv", index=False
        )
        M.confusion(frame.emotion, frame.pred, LABELS).to_csv(
            report / f"confusion_test_seed{seed}.csv"
        )
        if seed == cfg["seeds"][0]:
            first = frame
            timing = {
                "latency_one_utterance": E.latency_ms(model.predict, frame.text, n=100),
                "batch_throughput": E.throughput(model.predict, frame.text),
                "machine": E.machine(),
                "inference_device": "cpu",
            }
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(report / "per_seed.csv", index=False)
    stratified = pd.concat(strata, ignore_index=True)
    stratified.to_csv(report / "stratified_test_per_seed.csv", index=False)
    stratified.groupby("group", sort=False)[["n", *M.METRICS]].mean().to_csv(
        report / "stratified_test.csv"
    )
    intervals = dialogue_intervals(first, LABELS, cfg["bootstrap"])
    results = {
        metric: {
            "mean": float(per_seed[metric].mean()),
            "std": float(per_seed[metric].std(ddof=0)),
            "ci95_seed0": intervals[metric],
        }
        for metric in M.METRICS
    }
    summary = {
        "dataset": "emowoz",
        "model": name,
        "labels": LABELS,
        "headline_metric": METRIC,
        "selection": selection,
        "seeds": cfg["seeds"],
        "test": {
            "test": {
                "n": len(test),
                **results,
                "derived_shift_f1": float(per_seed.derived_shift_f1.mean()),
            }
        },
        "efficiency": {**timing, "train_seconds": train_seconds},
        "bootstrap": {
            "unit": "dialogue",
            "n": cfg["bootstrap"],
            "seed": 0,
            "checkpoint_seed": cfg["seeds"][0],
        },
    }
    write(report / "summary.json", summary)
    print(f"Evaluated {name}: {METRIC}={results[METRIC]['mean']:.4f}", flush=True)
    return summary


def b1(cfg, train, dev):
    report = ROOT / "reports/b1/emowoz"
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for features in cfg["b1"]["features"]:
        vectorizer = _features(TfidfConfig(features=features))
        x_train = vectorizer.fit_transform(train.text)
        x_dev = vectorizer.transform(dev.text)
        for c, weight in itertools.product(cfg["b1"]["C"], cfg["b1"]["class_weight"]):
            start = time.perf_counter()
            classifier = _classifier(TfidfConfig(C=c, class_weight=weight, seed=0)).fit(
                x_train, train.emotion
            )
            row = {
                "features": features,
                "C": c,
                "class_weight": weight or "none",
                "fit_seconds": time.perf_counter() - start,
                **M.scores(dev.emotion, classifier.predict(x_dev), LABELS),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(report / "dev_grid.csv", index=False)
            print(
                f"SVM dev: {features} C={c} weight={weight} {METRIC}={row[METRIC]:.4f}", flush=True
            )
    grid = pd.DataFrame(rows).sort_values(METRIC, ascending=False, kind="stable")
    best = grid.iloc[0]
    selected = {
        "C": float(best.C),
        "class_weight": None if best.class_weight == "none" else best.class_weight,
        "features": best.features,
        "classifier": "svm",
    }
    selection = {
        "config": selected,
        "selected_on": "dev only",
        "metric": METRIC,
        "dev_score": float(best[METRIC]),
        "test_used_for_selection": False,
    }
    write(report / "selection.json", selection)
    fitted, seconds = {}, {}
    for seed in cfg["seeds"]:
        start = time.perf_counter()
        model = TfidfBaseline(TfidfConfig(**selected, seed=seed), labels=LABELS).fit(
            train.text, train.emotion
        )
        seconds[str(seed)] = time.perf_counter() - start
        fitted[seed] = model
        if seed == cfg["seeds"][0]:
            model.save(ROOT / "models/b1_emowoz.joblib")
            write(report / "top_features.json", model.top_features(10))
    return evaluate("b1", fitted, seconds, cfg, report, selection)


def b2(cfg, train, dev, device, provenance):
    report = ROOT / "reports/b2/emowoz"
    report.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    dev_rows, checkpoints, times = [], {}, {}
    model_config = {k: v for k, v in cfg["b2"].items() if k != "architectures"}
    for architecture in cfg["b2"]["architectures"]:
        name = "b2_bi" + architecture
        folder = report / name
        folder.mkdir(parents=True, exist_ok=True)
        checkpoints[name], times[name] = {}, {}
        for seed in cfg["seeds"]:
            checkpoint = ROOT / "models" / f"{name}_emowoz_seed{seed}.pt"
            metadata = folder / f"training_seed{seed}.json"
            if checkpoint.exists() and metadata.exists():
                record = json.loads(metadata.read_text(encoding="utf-8"))
                if record["protocol_signature"] != signature or record["device"] != device:
                    raise ValueError(
                        "Existing checkpoint uses a different training protocol/runtime"
                    )
                print(f"Resuming completed {name} seed {seed}", flush=True)
            else:
                start = time.perf_counter()
                model = BiRNNBaseline(
                    BiRNNConfig(**model_config, architecture=architecture, seed=seed),
                    device=device,
                    labels=LABELS,
                ).fit(train.text, train.emotion, dev.text, dev.emotion, metric=METRIC)
                elapsed = time.perf_counter() - start
                model.save(checkpoint)
                pd.DataFrame(model.history).to_csv(folder / f"history_seed{seed}.csv", index=False)
                record = {
                    "model": name,
                    "seed": seed,
                    "best_epoch": model.best_epoch,
                    "dev_score": model.best_dev_score,
                    "train_seconds": elapsed,
                    "protocol_signature": signature,
                    "device": device,
                }
                write(metadata, record)
                print(f"Trained {name} seed={seed} in {elapsed:.1f}s", flush=True)
            dev_rows.append(record)
            checkpoints[name][seed] = checkpoint
            times[name][str(seed)] = record["train_seconds"]
    dev_table = pd.DataFrame(dev_rows)
    dev_table.to_csv(report / "dev_comparison.csv", index=False)
    rank = (
        dev_table.groupby("model", sort=False)
        .dev_score.mean()
        .sort_values(ascending=False, kind="stable")
    )
    selection = {
        "selected_model": rank.index[0],
        "dev_means": rank.to_dict(),
        "metric": METRIC,
        "rule": "mean best-dev across five seeds; config-order ties",
        "test_used_for_selection": False,
    }
    write(report / "selection.json", selection)
    # No test scoring until all architectures/checkpoints and selection are frozen.
    return [
        evaluate(name, paths, times[name], cfg, report / name, selection)
        for name, paths in checkpoints.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["b1", "b2"], default=["b1", "b2"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/emowoz_b1_b2.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(cfg["threads"])
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    (ROOT / "models").mkdir(exist_ok=True)
    train, dev = targets(["train"]), targets(["dev"])
    print(f"Native EmoWOZ: train={len(train)} dev={len(dev)}; B2 device={device}", flush=True)
    for family in args.models:
        provenance = protocol(cfg, family)
        if family == "b2":
            provenance["device"] = device
            provenance["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None
        write(OUT / f"{family}_manifest.json", provenance)
        if family == "b1":
            b1(cfg, train, dev)
        else:
            b2(cfg, train, dev, device, provenance)


if __name__ == "__main__":
    main()
