"""Extra paper figures: validation-scheme comparison and the ablation summary.

Usage (from the repo root): python make_figures_extra.py --results results/main --orient results/orient \
    --noblackhat results/noblackhat --manifest data/manifest.csv --out figures
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import (SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, AQUA, YELLOW, MAGENTA,
                          CLF_ORDER, CLF_SHORT, W1, W2, save)
import config as C

SCHEMES = [("image", "Image-level", MUTED, "o"), ("specimen", "Specimen-level", BLUE, "s"),
           ("article", "Article-level", ORANGE, "D"), ("loso", "Leave-one-article-out", AQUA, "^")]


def fold_metrics(res):
    r = pd.read_csv(res / "metrics.csv")
    r = r[pd.to_numeric(r.fold, errors="coerce").notna()]
    return r


def loso_pooled(res):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p = pd.read_csv(res / "predictions.csv")
    p = p[p.scheme == "loso"]
    out = {}
    for clf in CLF_ORDER:
        x = p[p.classifier == clf]
        if x.empty:
            continue
        f1 = precision_recall_fscore_support(x["true"], x["pred"], labels=C.CLASSES, average="macro", zero_division=0)[2]
        out[clf] = dict(accuracy=accuracy_score(x["true"], x["pred"]), f1_macro=f1)
    return out


def majority(manifest):
    m = pd.read_csv(manifest)
    m = m[m.label.isin(C.CLASSES)]
    return m.label.value_counts().max() / len(m)


def fig_schemes(out, res, manifest):
    r = fold_metrics(res)
    lo = loso_pooled(res)
    maj = majority(manifest)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
    x = np.arange(len(CLF_ORDER))
    for ax, (k, title) in zip(axes, [("accuracy", "Accuracy"), ("f1_macro", "Macro F1")]):
        ax.axhline(maj if k == "accuracy" else 1 / 3 * 0 + np.nan, color=MUTED, lw=0.8, ls="--")
        if k == "accuracy":
            ax.text(len(CLF_ORDER) - 0.6, maj + 0.015, "majority class", color=MUTED, fontsize=7, ha="right")
        for j, (scheme, lab, col, mk) in enumerate(SCHEMES):
            off = (j - 1.5) * 0.18
            if scheme == "loso":
                v = [lo[c][k] for c in CLF_ORDER]
                ax.scatter(x + off, v, color=col, marker=mk, s=16, zorder=3)
            else:
                g = r[r.scheme == scheme].groupby("classifier")[k]
                mu = [g.mean()[c] for c in CLF_ORDER]
                sd = [g.std()[c] for c in CLF_ORDER]
                ax.errorbar(x + off, mu, yerr=sd, fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5, zorder=3)
        ax.set_title(title, color=INK2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([CLF_SHORT[c] for c in CLF_ORDER])
        ax.set_ylim(0, 1.0); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2)
    axes[0].set_ylabel("Score")
    handles = [Line2D([], [], color=col, marker=mk, ls="", ms=4) for _, _, col, mk in SCHEMES]
    fig.legend(handles, [lab for _, lab, _, _ in SCHEMES], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.2, "Folds: mean +/- SD over 5 outer folds. Leave-one-article-out: pooled over all held-out articles.",
             ha="center", color=MUTED, fontsize=7)
    fig.subplots_adjust(wspace=0.1)
    save(fig, out, "fig_schemes")


def fig_ablation(out, runs):
    """runs: list of (label, results_dir). Specimen-level and article-level macro-F1 for each variant."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    x = np.arange(len(CLF_ORDER))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for ax, (scheme, title) in zip(axes, [("specimen", "Specimen-level macro F1"), ("article", "Article-level macro F1")]):
        for j, (lab, res) in enumerate(runs):
            r = fold_metrics(Path(res))
            g = r[r.scheme == scheme].groupby("classifier")["f1_macro"]
            mu = [g.mean()[c] for c in CLF_ORDER]; sd = [g.std()[c] for c in CLF_ORDER]
            off = (j - (len(runs) - 1) / 2) * 0.22
            ax.bar(x + off, mu, 0.2, color=cols[j], edgecolor=SURFACE, lw=0.5, yerr=sd,
                   error_kw=dict(ecolor=INK2, lw=0.6, capsize=1.2))
        ax.set_title(title, color=INK2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([CLF_SHORT[c] for c in CLF_ORDER])
        ax.set_ylim(0, 1.0); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2)
    axes[0].set_ylabel("Macro F1 (mean +/- SD)")
    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, fc=cols[j], ec=SURFACE) for j in range(len(runs))]
    fig.legend(handles, [lab for lab, _ in runs], loc="lower center", ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.1)
    save(fig, out, "fig_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--orient", default="results/orient")
    ap.add_argument("--noblackhat", default="results/noblackhat")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_schemes(out, Path(a.results), a.manifest)
    runs = [("HOG (3780-d)", a.results)]
    if Path(a.noblackhat, "metrics.csv").exists():
        runs.append(("HOG, no black-hat", a.noblackhat))
    if Path(a.orient, "metrics.csv").exists():
        runs.append(("Orientation histogram (18-d)", a.orient))
    fig_ablation(out, runs)


if __name__ == "__main__":
    main()
