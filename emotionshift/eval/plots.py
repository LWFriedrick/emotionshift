"""Figures for the notebook and slides (matplotlib, light surface).

Palette: validated default from the dataviz reference -- categorical slots in
fixed order, one-hue blue ramp for magnitude, text in neutral ink (never the
series colour), recessive grid, thin marks, selective direct labels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ..data.labels import EMOTIONS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _bars(ax, labels, values, ns=None, color=SERIES[0]):
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(values):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=10)
    ticks = [f"{l}\nn={n}" for l, n in zip(labels, ns)] if ns is not None else labels
    ax.set_xticks(x, ticks)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 100)
    ax.set_ylabel("weighted-F1 (%)")


def strata_chart(strat: pd.DataFrame, title: str):
    """Weighted-F1 by stratum and by shift type (one series, so no legend)."""
    s = strat.set_index("group")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 5]})
    rows = ["stratum:first", "stratum:stable", "stratum:shift"]
    _bars(a1, ["first turn", "stable", "shift"], 100 * s.loc[rows, "weighted_f1"].values,
          s.loc[rows, "n"].astype(int).values)
    a1.set_title("By stratum")
    types = ["onset", "offset", "surprise", "within_negative", "valence_flip"]
    rows = [f"shift_type:{t}" for t in types]
    _bars(a2, ["onset", "offset", "surprise", "within-\nnegative", "valence\nflip"],
          100 * s.loc[rows, "weighted_f1"].values, s.loc[rows, "n"].astype(int).values)
    a2.set_title("Shift turns, by shift type")
    a2.set_ylabel("")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def confusion_chart(cm: pd.DataFrame, title: str):
    """Row-normalised confusion matrix (recall per gold class), blue ramp."""
    m = cm.to_numpy().astype(float)
    norm = m / m.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            v = norm[i, j]
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if v > 0.5 else INK)
    ax.set_xticks(range(7), EMOTIONS, rotation=40, ha="right")
    ax.set_yticks(range(7), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def tuning_chart(grid: pd.DataFrame, metric: str, features: str, title: str):
    """Dev score vs C for each classifier x class-weight combination."""
    g = grid[grid.features == features]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    combos = [("svm", "balanced"), ("svm", "none"), ("logreg", "balanced"), ("logreg", "none")]
    names = {"svm": "Linear SVM", "logreg": "Logistic reg."}
    ends = []
    for k, (clf, cw) in enumerate(combos):
        d = g[(g.classifier == clf) & (g.class_weight == cw)].sort_values("C")
        if d.empty:
            continue
        label = f"{names[clf]}, {'balanced' if cw == 'balanced' else 'unweighted'}"
        ax.plot(d.C, 100 * d[metric], color=SERIES[k], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ends.append([100 * d[metric].iloc[-1], label, d.C.iloc[-1]])
    # Direct end labels, nudged apart so they never collide.
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    gap = 0.06 * span
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label, x in ends:
        ax.annotate(label, (x, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2, annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xticks(sorted(g.C.unique()), [str(c) for c in sorted(g.C.unique())])
    ax.minorticks_off()
    ax.set_xlabel("C (inverse regularisation strength, log scale)")
    ax.set_ylabel(f"dev {metric.replace('_', ' ')} (%)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.set_title(title, pad=40)
    fig.tight_layout()
    return fig


def per_class_chart(pc: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    _bars(ax, list(pc.emotion), 100 * pc.f1.values, pc.support.values)
    ax.set_ylabel("F1 (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
