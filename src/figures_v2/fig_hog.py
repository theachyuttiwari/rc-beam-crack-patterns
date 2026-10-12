"""fig_hog: original | preprocessed | HOG picture, one image per class."""
from _common import (CLASSES, CLASS_NAMES, HAIR, INK, badge, finish, frame,
                     image_credits, labelled_manifest, source_line, start)
from fig_samples import fit_cell, load_rgb, pick_all
import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as P  # noqa: E402

SCRIPT = __file__


def verify():
    assert C.HOG_PIXELS_PER_CELL == (16, 16)       # "16-pixel cell" in claim
    assert (C.IMG_H, C.IMG_W) == (128, 256)


def draw(m):
    fig = start(grid_axis="y", figsize=(12.0, 7.5))
    W, H = 11.04, 0.62 * 7.5
    fr = frame(fig, [0.04, 0.14, 0.92, 0.62], (0, W), (0, H))
    gap, head_h, rgap = 0.15, 0.45, 0.12
    ch = (H - head_h - 2 * rgap) / 3
    cw = 2 * ch
    label_w = W - 3 * cw - 2 * gap
    picks = pick_all(m)
    shown = []
    for c, head in enumerate(("Image", "Preprocessed", "HOG")):
        x = label_w + c * (cw + gap)
        fr.text(x + cw / 2, H, head, ha="center", va="top", fontsize=15,
                fontweight="bold", color=INK)
    for r, label in enumerate(CLASSES):
        row = picks[label][0]
        shown.append(row)
        rgb = load_rgb(row.image_path)
        pre = P.preprocess(row.image_path)               # 128 x 256, uint8
        hog = F.hog_visualisation(pre)
        y = H - head_h - (r + 1) * ch - r * rgap
        fr.text(0.0, y + ch / 2, CLASS_NAMES[label], ha="left", va="center",
                fontsize=15, fontweight="bold", color=INK)
        for c, (img, cmap) in enumerate(((fit_cell(rgb), None),
                                         (pre, "gray"), (hog, "gray"))):
            x = label_w + c * (cw + gap)
            ax = fig.add_axes([0.04 + 0.92 * x / W, 0.14 + 0.62 * y / H,
                               0.92 * cw / W, 0.62 * ch / H])
            ax.imshow(img, cmap=cmap, aspect="auto",
                      **({"vmin": 0, "vmax": 255} if c == 1 else
                         {"vmin": 0, "vmax": 1} if c == 2 else {}))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(HAIR); sp.set_linewidth(1.5)
            if c == 0:
                badge(ax, len(shown))
    return fig, fr, shown


if __name__ == "__main__":
    verify()
    m = labelled_manifest()
    fig, fr, shown = draw(m)
    finish(fig, fr, "fig_hog",
           claim="Preprocessing makes cracks bright lines, and HOG records their direction in every 16-pixel cell.",
           quantity="",
           source=source_line(m),
           caption=("One image per class: the original, the preprocessed "
                    "128 by 256 image (CLAHE, resize, blur, black-hat) and "
                    "the HOG picture computed from it, drawn with the "
                    "project's own preprocessing and feature code."),
           inputs=["data/manifest.csv"], script=SCRIPT, credits=image_credits(shown))
