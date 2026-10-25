"""Train and evaluate B1 (TF-IDF + linear classifier) on MELD and DailyDialog.

    python scripts/train_b1.py                       # both datasets, config defaults
    python scripts/train_b1.py --datasets meld       # one dataset
    python scripts/train_b1.py --quick               # tiny grid, 1 seed (smoke test)

Writes
    reports/b1/<dataset>/dev_grid.csv        every grid point, scored on dev
    reports/b1/<dataset>/summary.json        chosen config + test metrics (mean, std, 95% CI)
    reports/b1/<dataset>/stratified_<split>.csv, per_class_<split>.csv, confusion_<split>.csv
    reports/b1/<dataset>/top_features.json   highest-weighted n-grams per emotion
    reports/b1/results.md                    one table for the slides
    predictions/b1_<dataset>_<split>_seed<k>.csv   shared prediction format
    models/b1_<dataset>.joblib               seed-0 model, used by the demo
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.labels import EMOTIONS  # noqa: E402
from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import efficiency as E  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig, _classifier, _features  # noqa: E402

PRED_COLS = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion", "pred",
             "stratum", "shift_type", "model", "seed"]


def dev_grid(df: pd.DataFrame, grid: dict, metric: str) -> pd.DataFrame:
    tr, dev = df[df.split == "train"], df[df.split == "dev"]
    rows = []
    for feats in grid["features"]:
        vec = _features(TfidfConfig(features=feats))
        Xtr, Xdev = vec.fit_transform(tr.text), vec.transform(dev.text)
        for clf_name, C, cw in itertools.product(grid["classifier"], grid["C"], grid["class_weight"]):
            cfg = TfidfConfig(classifier=clf_name, C=C, class_weight=cw, features=feats)
            t0 = time.time()
            clf = _classifier(cfg).fit(Xtr, tr.emotion)
            s = M.scores(dev.emotion, clf.predict(Xdev))
            rows.append({"classifier": clf_name, "C": C, "class_weight": cw or "none", "features": feats,
                         **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()},
                         "fit_seconds": round(time.time() - t0, 1)})
            print(f"  dev {feats:9s} {clf_name:6s} C={C:<4} cw={cw or 'none':8s} "
                  f"{metric}={s[metric]:.4f} macro={s['macro_f1']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values(metric, ascending=False, kind="stable")


def run(dataset: str, cfg: dict, quick: bool) -> dict:
    out = ROOT / "reports" / "b1" / dataset
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "predictions").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = load(dataset)
    metric = cfg["select_on"][dataset]
    grid = {**cfg["grid"], **cfg.get("grid_overrides", {}).get(dataset, {})}
    seeds = cfg["seeds"]
    if quick:
        grid = {"classifier": ["svm"], "C": [1.0], "class_weight": [None], "features": ["word"]}
        seeds = seeds[:1]

    print(f"[{dataset}] dev grid ({metric})")
    g = dev_grid(df, grid, metric)
    g.to_csv(out / "dev_grid.csv", index=False)
    best = g.iloc[0]
    best_cfg = dict(classifier=best.classifier, C=float(best.C),
                    class_weight=None if best.class_weight == "none" else best.class_weight,
                    features=best.features)
    print(f"[{dataset}] chosen on dev: {best_cfg}")

    train = df[df.split == "train"]
    summary = {"dataset": dataset, "model": "B1 TF-IDF", "selected_on_dev": metric,
               "config": best_cfg, "dev": {k: best[k] for k in ["weighted_f1", "macro_f1", "micro_f1_no_neutral"]},
               "seeds": seeds, "test": {}}
    preds = {s: [] for s in cfg["test_splits"][dataset]}

    fit_seconds = []
    for seed in seeds:
        t0 = time.perf_counter()
        model = TfidfBaseline(TfidfConfig(**best_cfg, seed=seed)).fit(train.text, train.emotion)
        fit_seconds.append(time.perf_counter() - t0)
        if seed == seeds[0]:
            path = ROOT / "models" / f"b1_{dataset}.joblib"
            model.save(path)
            (out / "top_features.json").write_text(json.dumps(model.top_features(15), indent=1))
            clf = model.pipe.named_steps["clf"]
            test_texts = df[df.split == cfg["test_splits"][dataset][0]].text
            summary["efficiency"] = {
                "features": int(clf.coef_.shape[1]),
                "parameters": int(clf.coef_.size + clf.intercept_.size),
                "model_file_mb": round(E.file_size_mb(path), 2),
                "latency_one_utterance": E.latency_ms(model.predict, test_texts),
                "batch_throughput": E.throughput(model.predict, test_texts),
                "machine": E.machine(),
            }
        for split in cfg["test_splits"][dataset]:
            t = df[df.split == split].copy()
            t["pred"] = model.predict(t.text)
            t["model"], t["seed"] = "b1_tfidf", seed
            t[PRED_COLS].to_csv(ROOT / "predictions" / f"b1_{dataset}_{split}_seed{seed}.csv", index=False)
            preds[split].append(t)

    summary["efficiency"]["train_seconds"] = {"mean": float(np.mean(fit_seconds)),
                                              "per_seed": [round(s, 2) for s in fit_seconds]}

    for split, runs in preds.items():
        per_seed = [M.scores(t.emotion, t.pred) for t in runs]
        keys = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]
        res = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                   "std": float(np.std([p[k] for p in per_seed]))} for k in keys}
        first = runs[0]
        for k in keys[:3]:
            res[k]["ci95_seed0"] = M.bootstrap_ci(first, k, n_boot=cfg["bootstrap"] if not quick else 100)
        res["n"] = int(len(first))
        res["derived_shift_detection"] = M.derived_shift_detection(first)
        summary["test"][split] = res

        strat = pd.concat([M.stratified(t).assign(seed=t.seed.iloc[0]) for t in runs])
        num = [c for c in strat.columns if c not in ("group", "seed")]
        strat.groupby("group", sort=False)[num].mean().round(4).reset_index().to_csv(
            out / f"stratified_{split}.csv", index=False)
        M.per_class_f1(first.emotion, first.pred).round(4).to_csv(out / f"per_class_{split}.csv", index=False)
        M.confusion(first.emotion, first.pred).to_csv(out / f"confusion_{split}.csv")

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def results_table(summaries: list[dict]) -> str:
    lines = ["| Dataset | Test split | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |",
             "|---|---|---|---|---|---|---|"]
    for s in summaries:
        for split, r in s["test"].items():
            def f(k):
                lo, hi = r[k]["ci95_seed0"]
                return f"{100*r[k]['mean']:.2f} ± {100*r[k]['std']:.2f} [{100*lo:.1f}, {100*hi:.1f}]"
            lines.append(f"| {s['dataset']} | {split} | {r['n']} | {f('weighted_f1')} | {f('macro_f1')} | "
                         f"{f('micro_f1_no_neutral')} | {100*r['derived_shift_detection']['shift_f1']:.2f} |")
    cfgs = "; ".join(f"{s['dataset']}: {s['config']}" for s in summaries)
    eff = ["| Dataset | Features | Parameters | Model file | Train time | Latency per utterance (median / p95) | Batch throughput |",
           "|---|---|---|---|---|---|---|"]
    for s in summaries:
        e = s["efficiency"]
        lat, thr = e["latency_one_utterance"], e["batch_throughput"]
        eff.append(f"| {s['dataset']} | {e['features']:,} | {e['parameters']:,} | {e['model_file_mb']:.2f} MB | "
                   f"{e['train_seconds']['mean']:.1f} s | {lat['median_ms']:.2f} / {lat['p95_ms']:.2f} ms | "
                   f"{thr['utterances_per_second']:,.0f} utt/s |")
    m = summaries[0]["efficiency"]["machine"]
    return (f"# B1 TF-IDF results\n\nMean ± std over {len(summaries[0]['seeds'])} seeds, "
            "[95% dialogue-bootstrap CI, seed 0], in %.\n"
            f"Configs chosen on dev: {cfgs}\n\n" + "\n".join(lines) + "\n\n"
            "## Efficiency (E9)\n\n" + "\n".join(eff) + "\n\n"
            f"Measured on CPU only ({m['cpu_count']} cores, {m['processor']}, Python {m['python']}); "
            "latency = one utterance per call, as in the demo.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "b1_tfidf.yaml"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    summaries = [run(d, cfg, a.quick) for d in (a.datasets or cfg["datasets"])]
    md = results_table(summaries)
    (ROOT / "reports" / "b1" / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
