"""fig_samples: real example images, one row per class."""
import json

from _common import (CLASSES, CLASS_NAMES, HAIR, INK, ROOT, badge, finish, frame,
                     image_credits, labelled_manifest, source_line, start)

from picks import fit_cell, load_rgb, pick_all  # noqa: E402,F401  (fig_hog imports them from here)

SCRIPT = __file__


def verify(m):
    counts = m.label.value_counts().to_dict()
    summ = json.loads((ROOT / "data/summary.json").read_text())["labels"]
    assert counts == {k: summ[k] for k in CLASSES}, counts
    assert min(counts, key=counts.get) == "shear_flexural"     # claim: rarest


def draw(m):
    fig = start(grid_axis="y", figsize=(12.0, 7.5))
    W, H = 11.04, 0.62 * 7.5
    fr = frame(fig, [0.04, 0.14, 0.92, 0.62], (0, W), (0, H))
    label_w, gap = 2.0, 0.12
    cw = (W - label_w - 3 * gap) / 4
    ch = cw / 2
    rgap = (H - 3 * ch) / 2
    n_total = m.label.value_counts()
    picks = pick_all(m)
    shown = []
    for r, label in enumerate(CLASSES):
        y = H - (r + 1) * ch - r * rgap
        fr.text(0.0, y + ch / 2, f"{CLASS_NAMES[label]},\n{n_total[label]} images",
                ha="left", va="center", fontsize=15, fontweight="bold",
                color=INK, linespacing=1.3)
        for c, row in enumerate(picks[label]):
            x = label_w + c * (cw + gap)
            ax = fig.add_axes([0.04 + 0.92 * x / W, 0.14 + 0.62 * y / H,
                               0.92 * cw / W, 0.62 * ch / H])
            ax.imshow(fit_cell(load_rgb(row.image_path)), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(HAIR); sp.set_linewidth(1.5)
            shown.append(row)
            badge(ax, len(shown))
    return fig, fr, shown


if __name__ == "__main__":
    m = labelled_manifest()
    verify(m)
    fig, fr, shown = draw(m)
    finish(fig, fr, "fig_samples",
           claim="Each class mixes photographs and traced crack maps; shear-flexural is the rarest.",
           quantity="",
           source=source_line(m),
           caption=("Example images for each class, each from a different "
                    "article, scaled to equal cells: two photographs and two "
                    "traced crack maps per row where the data allow. No image "
                    "shown carries printed failure-mode words. Row labels give "
                    "the number of images in the class."),
           inputs=["data/manifest.csv"], script=SCRIPT, credits=image_credits(shown))
