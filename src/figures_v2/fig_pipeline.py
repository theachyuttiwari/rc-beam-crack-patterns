"""fig_pipeline: the processing chain, image to scores."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import (HAIR, INK, MUTED, PANEL, finish, frame, source_line,
                     start)
import config as C  # noqa: E402
import features as F  # noqa: E402

SCRIPT = __file__


def verify():
    # the box text states these; refuse if the pipeline config moved
    assert F.hog_dim() == 3780
    assert C.HOG_ORIENTATIONS == 9 and C.HOG_PIXELS_PER_CELL == (16, 16)
    assert C.HOG_CELLS_PER_BLOCK == (2, 2) and (C.IMG_H, C.IMG_W) == (128, 256)
    assert list(C.GRIDS) == ["SVM", "RandomForest", "kNN", "DecisionTree",
                             "XGBoost"]


def draw():
    fig = start(grid_axis="y", figsize=(12.0, 6.75))
    # data units are inches: frame is 11.52 x 3.51 in
    fr = frame(fig, [0.02, 0.17, 0.96, 0.52], (0, 11.52), (0, 3.51))
    boxes = [("Image", ""),
             ("Preprocess", ""),
             ("HOG", f"{F.hog_dim():,} values"),
             ("Classifier", "SVM, RF, kNN,\nDT, XGBoost"),
             ("Folds", "by specimen\nor article"),
             ("Scores", "")]
    n, gap = len(boxes), 0.3
    w = (11.52 - gap * (n - 1) - 0.04) / n
    h = 1.9
    y = (3.51 - h) / 2
    for i, (head, detail) in enumerate(boxes):
        x = 0.02 + i * (w + gap)
        fr.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=PANEL, ec=HAIR, lw=2))
        if detail:
            fr.text(x + w / 2, y + h * 0.68, head, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=INK)
            fr.text(x + w / 2, y + h * 0.33, detail, ha="center", va="center",
                    fontsize=14, color=INK, linespacing=1.2)
        else:
            fr.text(x + w / 2, y + h / 2, head, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=INK)
        if i < n - 1:
            fr.add_patch(FancyArrowPatch((x + w + 0.04, y + h / 2),
                                         (x + w + gap - 0.04, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=18,
                                         lw=2, color=MUTED))
    return fig, fr


if __name__ == "__main__":
    verify()
    fig, fr = draw()
    finish(fig, fr, "fig_pipeline",
           claim="Each crack image is cleaned, reduced to edge directions, and scored by five classifiers.",
           quantity="",
           source=source_line(),
           caption=("Processing chain: grey-scale, CLAHE contrast, 128 by 256 "
                    "letterbox, blur and black-hat; HOG with 9 bins, 16 by 16 "
                    "pixel cells and 2 by 2 blocks (3,780 values); five "
                    "classifiers; folds grouped by specimen or by article; "
                    "accuracy, macro precision, recall and F1, and confusion "
                    "matrices."),
           inputs=["data/manifest.csv"], script=SCRIPT)
