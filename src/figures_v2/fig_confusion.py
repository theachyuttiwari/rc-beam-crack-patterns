"""fig_confusion: row-normalised confusion matrices, specimen-level folds."""
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from _common import (BASE, CLASS_NAMES, CLASSES, CLFS, CLF_NAMES, INK, ROOT,
                     finish, frame, source_line, start)

SCRIPT = __file__


def load():
    out = {}
    for c in CLFS:
        df = pd.read_csv(ROOT / f"results/main/confusion_{c}.csv", index_col=0)
        out[c] = df.loc[[f"true_{k}" for k in CLASSES],
                        [f"pred_{k}" for k in CLASSES]].to_numpy(int)
    return out


def verify(cm):
    for c, m in cm.items():
        assert m.sum() == 295, c
        assert list(m.sum(1)) == [149, 115, 31], c
        wrong = 1 - m[2, 2] / m[2].sum()
        assert 0.55 <= wrong <= 0.8, (c, wrong)    # "more than half"


def draw(cm):
    fig = start(grid_axis="x", figsize=(12.0, 9.0))
    fr = frame(fig, [0.04, 0.12, 0.92, 0.65], (0, 1), (0, 1))
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "ramp", ["#FFFFFF", BASE])
    xs = [0.19, 0.45, 0.71]
    ys = [0.55, 0.28]
    w, h = 0.21, 0.19
    for k, c in enumerate(CLFS):
        r, q = divmod(k, 3)
        ax = fig.add_axes([xs[q], ys[r], w, h])
        m = cm[c]
        share = m / m.sum(1, keepdims=True)
        ax.imshow(share, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        fontsize=14, color="white" if share[i, j] > 0.55 else INK)
        ax.set_title(CLF_NAMES[c], fontsize=14, color=INK, pad=6,
                     fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.tick_params(length=0)
        ax.spines[["left", "bottom"]].set_visible(False)
        if r == 1:
            ax.set_xticklabels([CLASS_NAMES[k] for k in CLASSES],
                               rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([])
        if q == 0:
            ax.set_yticklabels([CLASS_NAMES[k] for k in CLASSES])
        else:
            ax.set_yticklabels([])
        if r == 1 and q == 0:
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    # colour key in the sixth slot
    kax = fig.add_axes([xs[2] + 0.02, ys[1] + 0.09, w - 0.04, 0.03])
    kax.imshow(np.linspace(0, 1, 256)[None, :], cmap=ramp, aspect="auto",
               extent=[0, 100, 0, 1])
    kax.set_yticks([])
    kax.set_xticks([0, 50, 100])
    kax.set_xlabel("Share of true class, %")
    kax.grid(False)
    for s in kax.spines.values():
        s.set_visible(False)
    return fig, fr


if __name__ == "__main__":
    cm = load()
    verify(cm)
    fig, fr = draw(cm)
    finish(fig, fr, "fig_confusion",
           claim="Every classifier misses more than half of the shear-flexural images.",
           quantity="counts, shaded by share of each true class; specimen folds",
           source=source_line(),
           caption=("Confusion matrices of the five classifiers under "
                    "specimen-level cross-validation, pooled over the five "
                    "test folds. Cell numbers are image counts; shading is "
                    "the share of each true class."),
           inputs=["data/manifest.csv"] +
                  [f"results/main/confusion_{c}.csv" for c in CLFS],
           script=SCRIPT)
