"""fig_split: image-, specimen- and article-level splitting (illustration)."""
from matplotlib.patches import Rectangle

from _common import (BASE, BASE_SOFT, INK, MUTED, OBS, finish, frame,
                     source_line, start)

SCRIPT = __file__
N_ART, SPEC_PER, IMG_PER = 4, 2, 4
SQ, PITCH, ROW, GAP = 0.34, 0.40, 0.42, 0.18      # inches


def test_mask(scheme):
    """test[row][col]: 8 rows (2 specimens x 4 articles), 4 images each.
    Each scheme holds out the same share (a quarter) of the images."""
    rows = N_ART * SPEC_PER
    t = [[False] * IMG_PER for _ in range(rows)]
    if scheme == "image":
        for r in range(rows):
            t[r][(r * 3 + 1) % IMG_PER] = True
    elif scheme == "specimen":
        for r in (1, 4):
            t[r] = [True] * IMG_PER
    else:
        for r in (2, 3):
            t[r] = [True] * IMG_PER
    return t


def verify():
    for s in ("image", "specimen", "article"):
        m = test_mask(s)
        assert sum(map(sum, m)) == 8 * IMG_PER // 4, s


def draw():
    fig = start(grid_axis="y", figsize=(12.0, 7.5))
    W, H = 11.04, 0.65 * 7.5            # inches
    fr = frame(fig, [0.04, 0.14, 0.92, 0.65], (0, W), (0, H))
    gw = IMG_PER * PITCH - (PITCH - SQ)
    xs = [1.6, 1.6 + 3.55, 1.6 + 7.1]
    top = H - 0.75
    for k, (scheme, head) in enumerate((("image", "by image"),
                                        ("specimen", "by specimen"),
                                        ("article", "by article"))):
        fr.text(xs[k] + gw / 2, H - 0.1, head, ha="center", va="top",
                fontsize=15, fontweight="bold", color=INK)
        mask = test_mask(scheme)
        y = top
        for r in range(N_ART * SPEC_PER):
            if r and r % SPEC_PER == 0:
                y -= GAP
            y -= ROW
            for c in range(IMG_PER):
                x = xs[k] + c * PITCH
                if mask[r][c]:
                    fr.add_patch(Rectangle((x, y), SQ, SQ, fc="white", ec=OBS,
                                           lw=2, hatch="////"))
                else:
                    fr.add_patch(Rectangle((x, y), SQ, SQ, fc=BASE_SOFT,
                                           ec=BASE, lw=1.5))
            if k == 0 and r == 0:
                y_first = y
            if k == 0 and r == 1:
                y_second = y
            if k == 2 and r == 0:
                fr.text(xs[k] + gw + 0.15, y + SQ / 2, "train", ha="left",
                        va="center", fontsize=14, color=INK)
            if k == 2 and r == 2:
                y_t = y
            if k == 2 and r == 3:
                fr.text(xs[k] + gw + 0.15, (y_t + y) / 2 + SQ / 2, "test",
                        ha="left", va="center", fontsize=14, color=INK)
        if k == 0:                           # one article = two specimen rows
            bx = xs[0] - 0.2
            fr.plot([bx, bx - 0.12, bx - 0.12, bx],
                    [y_first + SQ, y_first + SQ, y_second, y_second],
                    color=MUTED, lw=2)
            fr.text(bx - 0.25, (y_first + SQ + y_second) / 2, "one article",
                    ha="right", va="center", fontsize=14, color=INK)
    return fig, fr


if __name__ == "__main__":
    verify()
    fig, fr = draw()
    finish(fig, fr, "fig_split",
           claim="Only splitting by article keeps every image of a paper out of training.",
           quantity="",
           source=source_line(),
           caption=("Illustration of three ways to split images into training "
                    "and test folds. Each row is a specimen, grouped under its "
                    "article; each square is an image; hatched squares are the "
                    "test fold. A drawing of the scheme, not data."),
           inputs=["data/manifest.csv"], script=SCRIPT)
