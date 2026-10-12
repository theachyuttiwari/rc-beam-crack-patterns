"""fig_schemes: accuracy and macro F1 by fold scheme (the paper's main figure)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from _common import (BASE, CLFS, INK, MAJORITY, MUTED, OBS, ROOT, dotplot,
                     finish, fold_stats, frame, labelled_manifest, source_line,
                     start)

SCRIPT = __file__


N_LAB = len(labelled_manifest())
BASES = pd.read_csv(ROOT / "results/analysis/baselines.csv").set_index(["scheme", "model"])
PRIOR = float(BASES.loc[("specimen", "article_prior"), "accuracy"])   # source prior, specimen folds
BOOT = pd.read_csv(ROOT / "results/analysis/bootstrap.csv")


def load():
    p = pd.read_csv(ROOT / "results/main/predictions.csv")
    p = p[p.scheme == "loso"]
    pooled = {"acc": {}, "f1": {}}
    for c in CLFS:
        g = p[p.classifier == c]
        assert len(g) == N_LAB, (c, len(g))
        pooled["acc"][c] = (accuracy_score(g.true, g.pred), 0.0)
        pooled["f1"][c] = (f1_score(g.true, g.pred, average="macro"), 0.0)
    return pooled


def verify(pooled):
    assert abs(MAJORITY - BASES.loc[("specimen", "majority"), "accuracy"]) < 0.02
    for c in CLFS:
        a = fold_stats("results/main", "specimen", c, "accuracy")[0]
        assert a < PRIOR, (c, a)                         # claim: every classifier trails the prior
        b = BOOT[(BOOT.scheme == "article") & (BOOT.classifier == c) & (BOOT.against == "majority")].iloc[0]
        assert b.accuracy_lo < 0 < b.accuracy_hi, c      # claim: none clearly above majority


def draw(pooled):
    fig = start(grid_axis="x", figsize=(12.0, 9.0))
    fr = frame(fig, [0.04, 0.12, 0.92, 0.65], (0, 1), (0, 1))
    series_acc = [("by image", "results/main", "image", MUTED, "s"),
                  ("by specimen", "results/main", "specimen", BASE, "o"),
                  ("by article", "results/main", "article", OBS, "o"),
                  ("pooled", None, None, INK, "D", pooled["acc"])]
    series_f1 = list(series_acc[:3])
    series_f1.append(("", None, None, INK, "D", pooled["f1"]))

    ax1 = fig.add_axes([0.16, 0.19, 0.38, 0.58])
    dotplot(ax1, series_acc, "accuracy", (0.3, 1.0), 0.24, names=True,
            label_row=0)
    ax1.axvline(MAJORITY, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax1.text(MAJORITY + 0.01, 4.8, f"majority {MAJORITY * 100:.1f} %",
             ha="left", va="center", color=INK, fontsize=14)
    ax1.axvline(PRIOR, color=BASE, lw=1.5, ls=(0, (1, 2)), zorder=1)
    ax1.text(PRIOR + 0.01, 4.55, f"source prior\n{PRIOR * 100:.1f} %",
             ha="left", va="center", color=INK, fontsize=14, linespacing=1.1)
    ax1.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_xticklabels(["30", "", "50", "", "70", "", "90", ""])
    ax1.set_xlabel("Accuracy, %")
    ax1.set_ylim(5.0, -0.55)

    ax2 = fig.add_axes([0.58, 0.19, 0.38, 0.58])
    dotplot(ax2, series_f1, "f1_macro", (0.2, 0.9), 0.24, names=False)
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xticklabels(["20", "40", "60", "80"])
    ax2.set_xlabel("Macro F1, %")
    ax2.set_ylim(5.0, -0.55)
    return fig, fr


if __name__ == "__main__":
    pooled = load()
    verify(pooled)
    fig, fr = draw(pooled)
    finish(fig, fr, "fig_schemes",
           claim="By specimen every classifier trails a source prior that sees no crack; by article none clearly beats the majority.",
           quantity="mean ± SD over five folds",
           source=source_line(),
           caption=("Accuracy and macro F1 of five classifiers under image-, "
                    "specimen- and article-level cross-validation (mean and "
                    "SD over five folds) and pooled leave-one-article-out. "
                    "The dashed line is the majority-class accuracy. The dotted line "
                    "is the source prior under specimen-level folds: each test image "
                    "gets the majority label of its source in the training part."),
           inputs=["data/manifest.csv", "results/main/metrics.csv",
                   "results/main/predictions.csv", "results/analysis/baselines.csv",
                   "results/analysis/bootstrap.csv"],
           script=SCRIPT)
