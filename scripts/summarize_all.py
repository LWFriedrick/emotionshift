"""Group-wide results: E1 main table, E2 strata, E9 efficiency and paired tests.

    uv run scripts/summarize_all.py            # full run (about a minute)
    uv run scripts/summarize_all.py --quick    # fewer resamples, smoke test only

Every score is recomputed from predictions/<model>_<dataset>_<split>_seed<k>.csv,
so any cell can be checked against the CSVs. A model with no prediction files
shows [TBD] and its comparisons are skipped, then fill in on the next run.

Statistics (proposal, "Evaluation"):
* Scores are the mean over seeds, with the sample SD (ddof=1).
* 95% CIs resample whole test dialogues and recompute the seed mean.
* Paired comparisons pair seeds by rank. The p-value is approximate randomization:
  swap the two systems' predictions dialogue by dialogue (the same swap for every
  seed pair) and count how often the seed-mean difference is at least as large.
  p-values are not corrected for multiple comparisons; only the rows marked
  pre-registered test a hypothesis, the others are descriptive.

Writes reports/summary/: e1_main.csv, e2_strata.csv, e9_efficiency.csv,
paired.csv, summary.md and figures/strata_<dataset>.png.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from emotionshift.data.labels import emotions
from emotionshift.data.shifts import SHIFT_TYPES, STRATA
from emotionshift.eval.metrics import HEADLINE

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
OUT = REPORTS / "summary"

# Headline split per dataset (DailyDialog uses the de-duplicated test set, R4).
# EmoWOZ is scored on its own seven labels (labelled user turns only), so its numbers
# are not comparable with the other two datasets.
DATASETS = {"meld": "test", "dailydialog": "test_dedup", "emowoz": "test"}
LABELS = {d: emotions(d) for d in DATASETS}
assert all(labels[0] == "neutral" for labels in LABELS.values())  # batch_scores relies on it
METRICS = ("weighted_f1", "macro_f1", "micro_f1_no_neutral")
EMOWOZ_METRICS = ("weighted_f1_no_neutral", "macro_f1_no_neutral", *METRICS)
# Second column of the tables: macro-F1, without neutral where the headline drops it too.
SECOND = {"meld": "macro_f1", "dailydialog": "macro_f1", "emowoz": "macro_f1_no_neutral"}
SHORT = {
    "weighted_f1": "weighted-F1",
    "macro_f1": "macro-F1",
    "micro_f1_no_neutral": "micro-F1 without neutral",
    "weighted_f1_no_neutral": "weighted-F1 without neutral",
    "macro_f1_no_neutral": "macro-F1 without neutral",
}
KEYS = ["dataset", "split", "dialogue_id", "turn"]
TBD = "[TBD]"


@dataclass(frozen=True)
class Entry:
    key: str
    name: str
    workstream: int
    prefix: dict[str, str] = field(hash=False)
    reference: bool = False  # uses gold history; shown apart from the comparison


def _same(prefix: str) -> dict[str, str]:
    return dict.fromkeys(DATASETS, prefix)


ENTRIES = [
    Entry("majority", "Majority", 1, _same("b0_majority")),
    Entry("inertia_pred", "Emotional inertia (predicted history)", 1, _same("b0_inertia_pred")),
    Entry("markov_pred", "Markov (predicted history)", 1, _same("b0_markov_pred")),
    Entry("b1", "TF-IDF + SVM", 2, _same("b1")),
    Entry("b2_bigru", "BiGRU", 3, _same("b2_bigru")),
    Entry("b2_bilstm", "BiLSTM", 3, _same("b2_bilstm")),
    Entry("b3", "RoBERTa-base, utterance only", 4, _same("b3")),
    # Window k chosen on MELD dev, see reports/b4/selection.json.
    Entry("b4", "RoBERTa-base with context", 5, _same("b4")),
    # B4+ (fallback improvement R3) was trained on MELD only.
    Entry("b4plus", "RoBERTa-base with context, class-weighted (B4+)", 5, {"meld": "b4plus"}),
    # Variants chosen on dev only, see reports/shiftaware/selection.md. The EmoWOZ runs
    # are exploratory (three seeds: 0, 1, 42); the variant is the dev-best of the three
    # that have three seeds (mean dev weighted-F1 without neutral 81.62, against 81.01
    # for 0layer-hm-gate2 and 77.45 for 0layer-hm).
    Entry(
        "shiftaware",
        "ShiftAware",
        6,
        {
            "meld": "shiftaware_cw-sqrt",
            "dailydialog": "shiftaware",
            "emowoz": "shiftaware_tf-d768-2layer-hm",
        },
    ),
    Entry(
        "inertia_oracle", "Inertia (gold history, reference)", 1, _same("b0_inertia_oracle"), True
    ),
    Entry("markov_oracle", "Markov (gold history, reference)", 1, _same("b0_markov_oracle"), True),
]
BY_KEY = {e.key: e for e in ENTRIES}

# (a, b, pre-registered target). Differences are b - a.
PAIRS = [
    ("b3", "b4", "H1: gain on stable > shift"),
    ("b3", "b4plus", "R3: macro-F1 on MELD, CI excludes 0"),
    ("b4", "shiftaware", "H3: +2 on MELD shift turns, CI excludes 0, no stable loss"),
    ("b3", "shiftaware", ""),
    ("b1", "b3", ""),
    ("b1", "b2_bigru", ""),
    ("b1", "b2_bilstm", ""),
    ("inertia_pred", "b1", ""),
]


# ---------------------------------------------------------------------- scoring


def batch_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """`metrics.scores_from_confusion` over the leading axes of ``cm`` (..., k, k).

    Label 0 is neutral in every label set (asserted at import)."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * tp / (support + predicted))
    emo = np.arange(cm.shape[-1]) != 0
    tp_e, pred_e, sup_e = (
        tp[..., emo].sum(-1),
        predicted[..., emo].sum(-1),
        support[..., emo].sum(-1),
    )
    total = support.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        micro = np.where(pred_e + sup_e > 0, 2 * tp_e / (pred_e + sup_e), 0.0)
        weighted = np.where(total > 0, (f1 * support).sum(-1) / total, 0.0)
        weighted_e = np.where(sup_e > 0, (f1 * support)[..., emo].sum(-1) / sup_e, 0.0)
    return {
        "weighted_f1": weighted,
        "macro_f1": f1.mean(-1),
        "micro_f1_no_neutral": micro,
        "weighted_f1_no_neutral": weighted_e,
        "macro_f1_no_neutral": f1[..., emo].mean(-1),
    }


def dialogue_counts(gold, preds, dialogue, keep, n_dialogues, k=len(LABELS["meld"])) -> np.ndarray:
    """Confusion counts per dialogue and seed: (dialogues, seeds, k * k), rows in ``keep`` only."""
    rows = np.flatnonzero(keep)
    out = np.zeros((n_dialogues, len(preds), k * k))
    for s, pred in enumerate(preds):
        cell = dialogue[rows] * k * k + gold[rows] * k + pred[rows]
        out[:, s] = np.bincount(cell, minlength=n_dialogues * k * k).reshape(n_dialogues, k * k)
    return out


def _seed_mean(flat_cm: np.ndarray, metric: str) -> np.ndarray:
    """flat_cm (..., seeds, k * k) -> metric averaged over seeds (...)."""
    k = round(flat_cm.shape[-1] ** 0.5)
    cm = flat_cm.reshape(*flat_cm.shape[:-1], k, k)
    return batch_scores(cm)[metric].mean(-1)


def bootstrap_seed_mean(counts, metric, n_boot, rng) -> tuple[float, float]:
    """95% percentile CI of the seed-mean metric, resampling whole dialogues."""
    d = counts.shape[0]
    weights = np.stack([np.bincount(rng.integers(0, d, d), minlength=d) for _ in range(n_boot)])
    resampled = np.einsum("rd,dsc->rsc", weights, counts)
    low, high = np.percentile(_seed_mean(resampled, metric), [2.5, 97.5])
    return float(low), float(high)


def paired_test(counts_a, counts_b, metric, n_boot, n_perm, rng) -> dict:
    """Seed-mean difference b - a with a dialogue-bootstrap CI and an
    approximate-randomization p-value (dialogue-level swaps)."""
    d = counts_a.shape[0]
    total_a, total_b = counts_a.sum(0), counts_b.sum(0)
    observed = float(_seed_mean(total_b, metric) - _seed_mean(total_a, metric))

    weights = np.stack([np.bincount(rng.integers(0, d, d), minlength=d) for _ in range(n_boot)])
    boot = _seed_mean(np.einsum("rd,dsc->rsc", weights, counts_b), metric) - _seed_mean(
        np.einsum("rd,dsc->rsc", weights, counts_a), metric
    )
    low, high = np.percentile(boot, [2.5, 97.5])

    swap = rng.integers(0, 2, (n_perm, d)).astype(float)
    moved = np.einsum("rd,dsc->rsc", swap, counts_b - counts_a)
    perm = _seed_mean(total_b - moved, metric) - _seed_mean(total_a + moved, metric)
    p = (np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)
    return {"diff": observed, "ci_low": float(low), "ci_high": float(high), "p": float(p)}


# ---------------------------------------------------------------------- loading


@dataclass
class Loaded:
    base: pd.DataFrame  # one row per utterance: KEYS, emotion, stratum, shift_type
    gold: np.ndarray
    dialogue: np.ndarray
    preds: dict[str, tuple[list[int], np.ndarray]]  # key -> (seeds, codes (seeds, rows))


def _files(prefix: str, dataset: str, split: str) -> list[Path]:
    files = PREDICTIONS.glob(f"{prefix}_{dataset}_{split}_seed*.csv")
    return sorted(files, key=lambda p: int(p.stem.rsplit("seed", 1)[1]))


def load(dataset: str, split: str) -> Loaded:
    index = {lab: i for i, lab in enumerate(LABELS[dataset])}
    anchor = _files("b1", dataset, split)
    if not anchor:
        raise FileNotFoundError(f"no B1 predictions for {dataset} {split}; B1 is the row anchor")
    cols = [*KEYS, "emotion", "stratum", "shift_type"]
    base = pd.read_csv(anchor[0], usecols=cols, dtype={"dialogue_id": str})
    gold = base["emotion"].map(index).to_numpy()
    dialogue = pd.factorize(base["dialogue_id"])[0]

    preds = {}
    for entry in ENTRIES:
        if dataset not in entry.prefix:
            continue
        seeds, codes = [], []
        for path in _files(entry.prefix[dataset], dataset, split):
            df = pd.read_csv(path, usecols=[*KEYS, "emotion", "pred"], dtype={"dialogue_id": str})
            merged = base[[*KEYS, "emotion"]].merge(
                df, on=KEYS, how="left", suffixes=("", "_file"), validate="one_to_one"
            )
            if len(df) != len(base) or merged["pred"].isna().any():
                raise ValueError(f"{path.name}: rows do not match B1")
            if (merged["emotion"] != merged["emotion_file"]).any():
                raise ValueError(f"{path.name}: gold labels differ from B1")
            code = merged["pred"].map(index)
            if code.isna().any():
                raise ValueError(f"{path.name}: unknown labels {set(merged['pred'][code.isna()])}")
            seeds.append(int(path.stem.rsplit("seed", 1)[1]))
            codes.append(code.to_numpy())
        if codes:
            preds[entry.key] = (seeds, np.stack(codes))
    return Loaded(base, gold, dialogue, preds)


# ---------------------------------------------------------------------- tables


def _groups(base: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    groups = [("all", np.ones(len(base), bool))]
    groups += [(s, (base["stratum"] == s).to_numpy()) for s in STRATA]
    groups += [(f"shift:{t}", (base["shift_type"] == t).to_numpy()) for t in SHIFT_TYPES]
    # Empty groups are dropped: within_positive on MELD (one positive emotion), surprise
    # on EmoWOZ (no surprise label).
    return [(name, mask) for name, mask in groups if mask.any()]


def tables(data: dict[str, Loaded], n_boot: int, n_perm: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    e1, e2, paired = [], [], []
    for dataset, loaded in data.items():
        headline, second, k = HEADLINE[dataset], SECOND[dataset], len(LABELS[dataset])
        n_dialogues = int(loaded.dialogue.max()) + 1
        groups = dict(_groups(loaded.base))
        counts = {}  # (key, group) -> per-dialogue counts
        for entry in ENTRIES:
            if dataset not in entry.prefix:
                continue
            if entry.key not in loaded.preds:
                e1.append({"dataset": dataset, "model": entry.key, "seeds": 0})
                continue
            seeds, codes = loaded.preds[entry.key]
            for group, mask in groups.items():
                c = dialogue_counts(loaded.gold, codes, loaded.dialogue, mask, n_dialogues, k)
                counts[entry.key, group] = c
                per_seed = batch_scores(c.sum(0).reshape(len(seeds), k, k))
                row = {"dataset": dataset, "model": entry.key, "group": group, "n": int(mask.sum())}
                for metric in (headline, second):
                    row[f"{metric}_mean"] = 100 * per_seed[metric].mean()
                    row[f"{metric}_sd"] = (
                        100 * per_seed[metric].std(ddof=1) if len(seeds) > 1 else np.nan
                    )
                e2.append(row)
            c = counts[entry.key, "all"]
            row = {
                "dataset": dataset,
                "model": entry.key,
                "seeds": len(seeds),
                "seed_list": " ".join(map(str, seeds)),
            }
            per_seed = batch_scores(c.sum(0).reshape(len(seeds), k, k))
            for metric in EMOWOZ_METRICS if dataset == "emowoz" else METRICS:
                row[f"{metric}_mean"] = 100 * per_seed[metric].mean()
                row[f"{metric}_sd"] = (
                    100 * per_seed[metric].std(ddof=1) if len(seeds) > 1 else np.nan
                )
                low, high = bootstrap_seed_mean(c, metric, n_boot, rng)
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = 100 * low, 100 * high
            e1.append(row)

        for a, b, target in PAIRS:
            if dataset not in BY_KEY[a].prefix or dataset not in BY_KEY[b].prefix:
                continue  # not trained on this dataset (B4+ is MELD only)
            for group, metric in [
                ("all", headline),
                ("stable", headline),
                ("shift", headline),
                ("all", second),
            ]:
                # Both pre-registered targets are on MELD.
                row = {
                    "dataset": dataset,
                    "a": a,
                    "b": b,
                    "group": group,
                    "metric": metric,
                    "target": target if dataset == "meld" else "",
                }
                if (a, group) in counts and (b, group) in counts:
                    ca, cb = counts[a, group], counts[b, group]
                    s = min(ca.shape[1], cb.shape[1])
                    result = paired_test(ca[:, :s], cb[:, :s], metric, n_boot, n_perm, rng)
                    row.update(seed_pairs=s, n=int(groups[group].sum()))
                    row.update({k: (100 * v if k != "p" else v) for k, v in result.items()})
                paired.append(row)
    return pd.DataFrame(e1), pd.DataFrame(e2), pd.DataFrame(paired)


# ---------------------------------------------------------------------- efficiency


def _get(d: dict, *paths):
    """First value found among dotted ``paths`` in nested dict ``d``."""
    for path in paths:
        node = d
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is not None:
            return node
    return None


def efficiency() -> pd.DataFrame:
    sources = {
        "b1": "b1/{d}/summary.json",
        "b2_bigru": "b2/{d}/b2_bigru/summary.json",
        "b2_bilstm": "b2/{d}/b2_bilstm/summary.json",
        "b3": "b3/{d}/summary.json",
        "b4": "b4/{d}/summary.json",
        "shiftaware": "shiftaware/efficiency.json",
    }
    rows = []
    for dataset in DATASETS:
        b0 = REPORTS / "b0" / dataset / "summary.json"
        if b0.exists():
            seconds = json.loads(b0.read_text())["seconds"]
            rows.append(
                {
                    "dataset": dataset,
                    "model": "b0 (all variants)",
                    "train_seconds": seconds,
                    "note": "one run fits and scores every B0 variant; no text model",
                }
            )
        for key, pattern in sources.items():
            path = REPORTS / pattern.format(d=dataset)
            if not path.exists():
                rows.append({"dataset": dataset, "model": key, "note": TBD})
                continue
            s = json.loads(path.read_text())
            e = s.get("efficiency", s)
            train = _get(e, "train_seconds.mean", "train_seconds")
            rows.append(
                {
                    "dataset": dataset,
                    "model": key,
                    "parameters": _get(e, "parameters"),
                    "size_mb": _get(e, "model_file_mb", "model_dir_mb", "size_mb"),
                    "train_seconds": train if isinstance(train, (int, float)) else None,
                    "cpu_median_ms": _get(
                        e,
                        "latency_one_utterance.median_ms",
                        "latency_one_utterance_cpu.median_ms",
                        "latency.median_ms",
                    ),
                    "cpu_p95_ms": _get(
                        e,
                        "latency_one_utterance.p95_ms",
                        "latency_one_utterance_cpu.p95_ms",
                        "latency.p95_ms",
                    ),
                    "machine": _get(e, "machine.platform"),
                    "note": _get(e, "note") or "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- report


def _pm(mean, sd) -> str:
    if pd.isna(mean):
        return TBD
    return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f} ± {sd:.2f}"


def _diff(row) -> str:
    if pd.isna(row.get("diff")):
        return TBD
    p = "p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.3f}"
    return f"{row['diff']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}], {p}"


def _status(e1, e2, paired) -> list[str]:
    def pair(a, b, group, metric, dataset="meld"):
        hit = paired[
            (paired.a == a)
            & (paired.b == b)
            & (paired.group == group)
            & (paired.metric == metric)
            & (paired.dataset == dataset)
        ]
        return hit.iloc[0] if len(hit) else None

    def met(row, threshold=0.0):
        if row is None or pd.isna(row.get("diff")):
            return TBD
        ok = row["diff"] >= threshold and row["ci_low"] > 0
        return "met" if ok else "not met"

    b4 = e1[(e1.model == "b4") & (e1.dataset == "meld")]
    b4_wf1 = b4["weighted_f1_mean"].iloc[0] if len(b4) and b4["seeds"].iloc[0] else np.nan
    h3, h3_stable = (
        pair("b4", "shiftaware", "shift", "weighted_f1"),
        pair("b4", "shiftaware", "stable", "weighted_f1"),
    )
    proxy = pair("b3", "shiftaware", "shift", "weighted_f1")
    r3 = pair("b3", "b4plus", "all", "macro_f1")
    h1s, h1t = pair("b3", "b4", "stable", "weighted_f1"), pair("b3", "b4", "shift", "weighted_f1")
    # MELD and DailyDialog carry the hypotheses; EmoWOZ is an extension (see its note).
    main = e1[
        e1.model.isin([e.key for e in ENTRIES if not e.reference])
        & (e1.seeds > 0)
        & (e1.dataset != "emowoz")
    ]
    lines = [
        "| target | result | status |",
        "|---|---|---|",
        f"| H3: ShiftAware - context RoBERTa, MELD shift turns, ≥ +2 with CI > 0 | {_diff(h3) if h3 is not None else TBD} | {met(h3, 2.0)} |",
        f"| H3 guard: same, stable turns, no significant loss | {_diff(h3_stable) if h3_stable is not None else TBD} | "
        + (
            TBD
            if h3_stable is None or pd.isna(h3_stable.get("diff"))
            else ("met" if h3_stable["ci_high"] > 0 else "not met")
        )
        + " |",
        f"| H3 reference: ShiftAware - RoBERTa utterance-only, MELD shift | {_diff(proxy) if proxy is not None else TBD} | descriptive |",
        f"| R2: context RoBERTa ≥ 62 weighted-F1 on MELD | {TBD if pd.isna(b4_wf1) else f'{b4_wf1:.2f}'} | "
        + (TBD if pd.isna(b4_wf1) else ("met" if b4_wf1 >= 62 else "not met"))
        + " |",
        f"| R3: tuned context RoBERTa (B4+) - utterance-only, MELD macro-F1, CI > 0 | {_diff(r3) if r3 is not None else TBD} | {met(r3)} |",
        f"| H1: context gain on stable vs shift turns (MELD) | stable {_diff(h1s) if h1s is not None else TBD}; shift {_diff(h1t) if h1t is not None else TBD} | "
        + (
            TBD
            if h1s is None or pd.isna(h1s.get("diff"))
            else ("met" if h1s["diff"] > h1t["diff"] else "not met")
        )
        + " |",
        f"| R4: ≥ 3 seeds per compared model (MELD, DailyDialog) | min {int(main.seeds[main.seeds > 1].min())} seeds among trained models | "
        + ("met" if main.seeds[main.seeds > 1].min() >= 3 else "not met")
        + " |",
    ]
    return lines


EMOWOZ_NOTE = [
    "Native seven labels, 8,634 labelled user turns. Workstreams 4 and 5 have not been run on"
    " EmoWOZ. ShiftAware is exploratory here: three seeds (0, 1, 42), variant"
    " `shiftaware_tf-d768-2layer-hm`, the dev-best of the three-seed variants."
    " In the strata table, offset turns have gold label neutral, so a score without neutral is 0"
    " there by construction.",
    "",
]


def write_report(e1, e2, paired, eff, n_boot, n_perm):
    names = {e.key: e.name for e in ENTRIES}
    lines = [
        "# Group results summary",
        "",
        "Generated by `scripts/summarize_all.py` from `predictions/`; do not edit by hand.",
        f"Test sets: MELD `test`, DailyDialog `test_dedup`, EmoWOZ `test` (labelled user turns)."
        f" Scores in %, mean ± sample SD over seeds;"
        f" CIs resample test dialogues ({n_boot} resamples); p-values are approximate randomization"
        f" with dialogue-level swaps ({n_perm} permutations), uncorrected.",
        "Headline: MELD weighted-F1, DailyDialog micro-F1 without neutral, EmoWOZ weighted-F1"
        " without neutral.",
        "EmoWOZ uses its own seven labels (`emotions('emowoz')`), so its scores are in-domain and"
        " not comparable with MELD or DailyDialog. The hypotheses below are tested on MELD only.",
        "",
        "## Hypotheses and requirements",
        "",
        *_status(e1, e2, paired),
        "",
        "## E1 main comparison",
        "",
    ]
    for dataset in DATASETS:
        headline, second = HEADLINE[dataset], SECOND[dataset]
        lines += [
            f"### {dataset}",
            "",
            *(EMOWOZ_NOTE if dataset == "emowoz" else []),
            f"| # | model | seeds | {SHORT[headline]} [95% CI] | {SHORT[second]} |",
            "|---|---|---|---|---|",
        ]
        for entry in ENTRIES:
            if dataset not in entry.prefix:
                continue
            row = e1[(e1.dataset == dataset) & (e1.model == entry.key)].iloc[0]
            if not row["seeds"]:
                lines.append(f"| {entry.workstream} | {entry.name} | 0 | {TBD} | {TBD} |")
                continue
            ci = f" [{row[f'{headline}_ci_low']:.2f}, {row[f'{headline}_ci_high']:.2f}]"
            lines.append(
                f"| {entry.workstream} | {entry.name} | {int(row['seeds'])} | "
                f"{_pm(row[f'{headline}_mean'], row[f'{headline}_sd'])}{ci} | "
                f"{_pm(row[f'{second}_mean'], row[f'{second}_sd'])} |"
            )
        lines.append("")
    lines += ["## E2 strata (headline metric)", ""]
    for dataset in DATASETS:
        headline = HEADLINE[dataset]
        sub = e2[e2.dataset == dataset]
        groups = list(dict.fromkeys(sub.group))
        ns = sub.drop_duplicates("group").set_index("group")["n"]
        lines += [
            f"### {dataset}",
            "",
            "| model | " + " | ".join(f"{g} (n={ns[g]})" for g in groups) + " |",
            "|---" * (len(groups) + 1) + "|",
        ]
        for entry in ENTRIES:
            if dataset not in entry.prefix:
                continue
            rows = sub[sub.model == entry.key].set_index("group")
            if rows.empty:
                lines.append(f"| {entry.name} | " + " | ".join(TBD for _ in groups) + " |")
                continue
            lines.append(
                f"| {entry.name} | "
                + " | ".join(
                    _pm(rows.loc[g, f"{headline}_mean"], rows.loc[g, f"{headline}_sd"])
                    for g in groups
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Paired comparisons (b - a)",
        "",
        "| dataset | a | b | group | metric | difference [95% CI], p | pre-registered target |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['dataset']} | {names[row['a']]} | {names[row['b']]} | {row['group']} | "
            f"{row['metric']} | {_diff(row)} | {row['target']} |"
        )
    lines += [
        "",
        "## E9 efficiency",
        "",
        "Each model was measured on its owner's machine (see `machine`), so compare within a row,"
        " not across machines.",
        "",
        "| dataset | model | parameters | size MB | train s | CPU ms/utterance (median / p95) | machine | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in eff.iterrows():

        def f(v, fmt):
            return "" if v is None or pd.isna(v) else format(v, fmt)

        def text(v):
            return "" if v is None or pd.isna(v) else str(v)

        lat = (
            ""
            if pd.isna(r.get("cpu_median_ms"))
            else f"{r['cpu_median_ms']:.2f} / {r['cpu_p95_ms']:.2f}"
        )
        lines.append(
            f"| {r['dataset']} | {r['model']} | {f(r.get('parameters'), ',.0f')} | {f(r.get('size_mb'), '.2f')} | "
            f"{f(r.get('train_seconds'), '.1f')} | {lat} | {text(r.get('machine'))} | {text(r.get('note'))} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def figures(e2):
    import matplotlib.pyplot as plt

    from emotionshift.eval.plots import INK_2, SERIES

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        headline = HEADLINE[dataset]
        main = [
            e for e in ENTRIES if not e.reference and e.key in set(e2[e2.dataset == dataset].model)
        ]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        x = np.arange(len(main))
        for i, group in enumerate(["all", "stable", "shift"]):
            sub = e2[(e2.dataset == dataset) & (e2.group == group)].set_index("model")
            means = [sub.loc[e.key, f"{headline}_mean"] for e in main]
            sds = [np.nan_to_num(sub.loc[e.key, f"{headline}_sd"]) for e in main]
            n = int(sub["n"].iloc[0])
            ax.bar(
                x + (i - 1) * 0.27,
                means,
                0.27,
                yerr=sds,
                color=SERIES[i],
                label=f"{group} (n={n})",
                error_kw={"elinewidth": 0.8, "ecolor": INK_2},
            )
        ax.set_xticks(
            x,
            [
                e.name.replace(" (predicted history)", "\n(pred. history)").replace(
                    ", utterance only", "\nutterance only"
                )
                for e in main
            ],
            fontsize=8,
        )
        ax.set_ylabel(f"{headline} (%)")
        ax.set_ylim(0, 85 if dataset == "meld" else 100)  # room for the legend
        ax.grid(axis="x", visible=False)
        ax.set_title(
            f"{dataset}: headline score on all, stable and shift turns (mean ± SD over seeds)"
        )
        ax.legend(frameon=False, ncols=3, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"strata_{dataset}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quick", action="store_true", help="200 resamples instead of 2000 (smoke test)"
    )
    args = parser.parse_args()
    n_boot, n_perm = (200, 200) if args.quick else (2000, 5000)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {d: load(d, s) for d, s in DATASETS.items()}
    e1, e2, paired = tables(data, n_boot, n_perm)
    eff = efficiency()
    e1.to_csv(OUT / "e1_main.csv", index=False, float_format="%.4f")
    e2.to_csv(OUT / "e2_strata.csv", index=False, float_format="%.4f")
    paired.to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    eff.to_csv(OUT / "e9_efficiency.csv", index=False)
    write_report(e1, e2, paired, eff, n_boot, n_perm)
    figures(e2)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
