"""fig_ablation: macro F1 for three descriptors, specimen and article folds."""
from _common import (BASE, CLFS, INK, MUTED, OBS, dotplot, finish, fold_stats,
                     frame, source_line, start)

SCRIPT = __file__
DESC = [("HOG", "results/main", INK, "o"),
        ("no black-hat", "results/noblackhat", BASE, "s"),
        ("orientation histogram", "results/orient", OBS, "D")]


def verify():
    # claim: histogram below HOG by specimen and above HOG by article, four of five each
    below = above = 0
    for c in CLFS:
        sp = [fold_stats(d, "specimen", c, "f1_macro")[0] for _, d, _, _ in DESC]
        ar = [fold_stats(d, "article", c, "f1_macro")[0] for _, d, _, _ in DESC]
        below += sp[2] < sp[0]
        above += ar[2] > ar[0]
    assert (below, above) == (4, 4), (below, above)


def draw():
    fig = start(grid_axis="x", figsize=(12.0, 9.0))
    fr = frame(fig, [0.04, 0.12, 0.92, 0.65], (0, 1), (0, 1))
    for rect, scheme, names, lab, title in (
            ([0.16, 0.19, 0.38, 0.58], "specimen", True, 0, "By specimen"),
            ([0.58, 0.19, 0.38, 0.58], "article", False, None, "By article")):
        ax = fig.add_axes(rect)
        ser = [(n, d, scheme, col, mk) for n, d, col, mk in DESC]
        dotplot(ax, ser, "f1_macro", (0.1, 0.9), 0.24, names=names,
                label_row=lab)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["20", "40", "60", "80"])
        ax.set_xlabel(f"Macro F1, %, folds {title.lower()}")
    return fig, fr


if __name__ == "__main__":
    verify()
    fig, fr = draw()
    finish(fig, fr, "fig_ablation",
           claim="Dropping position lowers macro F1 by specimen and raises it by article, for four of five classifiers.",
           quantity="mean ± SD over five folds",
           source=source_line(),
           caption=("Macro F1 of five classifiers on three descriptors: HOG, "
                    "HOG without the black-hat step, and an 18-bin gradient "
                    "orientation histogram, under specimen-level (left) and "
                    "article-level (right) folds; mean and SD over five folds."),
           inputs=["data/manifest.csv", "results/main/metrics.csv",
                   "results/noblackhat/metrics.csv", "results/orient/metrics.csv"],
           script=SCRIPT)
