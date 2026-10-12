"""fig_samples: real example images, one row per class."""
import cv2
import numpy as np

from _common import (CLASSES, CLASS_NAMES, HAIR, INK, badge, finish, frame,
                     image_credits, labelled_manifest, source_line, start)

SCRIPT = __file__
CELL_W, CELL_H = 400, 200        # 2:1 cell, the pipeline's own aspect
N_PHOTO, N_MAP = 2, 2


def _aspect_ok(path, lo=1.8, hi=7.0):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img is not None and lo <= img.shape[1] / img.shape[0] <= hi


def pick_all(m, seed=7):
    """Deterministic: per class, two photographs and two traced maps, wide
    beam-shaped images (aspect 1.8 to 7), every image of a class from a different article. Fixed shuffle seed."""
    rng = np.random.default_rng(seed)
    out = {}
    for label in CLASSES:
        rows, used = [], set()
        for mod, n in (("photo", N_PHOTO), ("map", N_MAP)):
            sub = m[(m.label == label) & (m.modality == mod)]
            sub = sub.iloc[rng.permutation(len(sub))]
            got = 0
            for r in sub.itertuples():
                if r.source_slug in used or not _aspect_ok(r.image_path):
                    continue
                rows.append(r)
                used.add(r.source_slug)
                got += 1
                if got == n:
                    break
            assert got == n, (label, mod, got)
        out[label] = rows
    return out


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_cell(rgb, w=CELL_W, h=CELL_H):
    """Scale to fit inside w x h, pad with white; every cell is equal size."""
    ih, iw = rgb.shape[:2]
    s = min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1
                       else cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 255, np.uint8)
    t, l = (h - nh) // 2, (w - nw) // 2
    canvas[t:t + nh, l:l + nw] = small
    return canvas


def verify(m):
    counts = m.label.value_counts().to_dict()
    assert counts == {"flexural": 149, "shear": 115, "shear_flexural": 31}, counts


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
           caption=("Example images for each class: two photographs and two "
                    "traced crack maps per row, from different articles, "
                    "scaled to equal cells. Row labels give the number of "
                    "images in the class."),
           inputs=["data/manifest.csv"], script=SCRIPT, credits=image_credits(shown))
