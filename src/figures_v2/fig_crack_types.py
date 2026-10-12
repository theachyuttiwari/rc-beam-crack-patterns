"""fig_crack_types: schematic of the three crack classes on a four-point beam."""
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from _common import (INK, MUTED, OBS, PANEL, finish, frame, source_line,
                     start)

SCRIPT = __file__
# data units are half-inches: frame is 11.04 x 5.12 in
X0, X1 = 4.2, 20.6            # beam ends
D = 1.7                       # beam depth
SUP = (X0 + 0.9, X1 - 0.9)    # supports
SPAN = SUP[1] - SUP[0]
LOADS = (SUP[0] + SPAN / 3, SUP[0] + 2 * SPAN / 3)


def bend(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)
    return np.c_[(1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
                 (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]]


def cracks(kind, y0):
    out = []
    if kind == "flexural":
        for x in np.linspace(LOADS[0] + 0.7, LOADS[1] - 0.7, 5):
            out.append(np.array([(x, y0), (x + 0.05, y0 + D * 0.72)]))
    elif kind == "shear":
        for sgn, a, b in ((1, SUP[0], LOADS[0]), (-1, SUP[1], LOADS[1])):
            for off in (0.3, 1.1):
                out.append(np.array([(a + sgn * off, y0 + 0.02),
                                     (a + sgn * (off + 3.5), y0 + D)]))
    else:
        for sgn, a, b in ((1, SUP[0], LOADS[0]), (-1, SUP[1], LOADS[1])):
            for off, top in ((1.3, 1.6), (2.7, 0.3)):
                xs = a + sgn * off
                out.append(bend((xs, y0), (xs, y0 + D * 0.5),
                                (b - sgn * top, y0 + D)))
        out.append(np.array([(SPAN / 2 + SUP[0], y0), (SPAN / 2 + SUP[0], y0 + D * 0.6)]))
    return out


def draw():
    fig = start(grid_axis="y", figsize=(12.0, 9.0))
    fr = frame(fig, [0.04, 0.12, 0.92, 0.65], (0, 22.08), (0, 11.7))
    panels = [("Flexural", "flexural"), ("Shear", "shear"),
              ("Flexure-shear", "flexshear")]
    for i, (name, kind) in enumerate(panels):
        y0 = 11.7 - (i + 1) * 3.9 + 1.45        # beam bottom
        fr.text(0.1, y0 + D / 2, name, ha="left", va="center", fontsize=15,
                fontweight="bold", color=INK)
        fr.add_patch(Rectangle((X0, y0), X1 - X0, D, fc=PANEL, ec=MUTED, lw=2))
        for xy in cracks(kind, y0):
            fr.plot(xy[:, 0], xy[:, 1], color=OBS, lw=3, solid_capstyle="round")
        for sx in SUP:                           # supports
            fr.add_patch(Polygon([(sx, y0), (sx - 0.45, y0 - 0.7),
                                  (sx + 0.45, y0 - 0.7)], fc=INK, ec=INK))
        for lx in LOADS:                         # load arrows
            fr.plot([lx, lx], [y0 + D + 0.9, y0 + D + 0.3], color=INK, lw=2.5)
            fr.add_patch(Polygon([(lx, y0 + D), (lx - 0.22, y0 + D + 0.4),
                                  (lx + 0.22, y0 + D + 0.4)], fc=INK, ec=INK))
        if i == 0:
            for lx in LOADS:
                fr.text(lx, y0 + D + 0.95, "P", ha="center", va="bottom",
                        fontsize=14, color=INK)
            fr.text(SUP[0] - 0.8, y0 - 0.35, "support", ha="right", va="center",
                    fontsize=14, color=INK)
            fr.plot([X1 + 0.5, X1 + 0.5], [y0, y0 + D], color=MUTED, lw=2)
            fr.text(X1 + 0.8, y0 + D / 2, "d", ha="left", va="center",
                    fontsize=14, color=INK)
        if i == 1:
            ya = y0 - 1.0
            fr.plot([SUP[0], LOADS[0]], [ya, ya], color=MUTED, lw=2)
            for xx in (SUP[0], LOADS[0]):
                fr.plot([xx, xx], [ya - 0.2, ya + 0.2], color=MUTED, lw=2)
            fr.text((SUP[0] + LOADS[0]) / 2, ya + 0.2, "a", ha="center",
                    va="bottom", fontsize=14, color=INK)
    return fig, fr


if __name__ == "__main__":
    fig, fr = draw()
    finish(fig, fr, "fig_crack_types",
           claim="Flexure cracks stay vertical, shear cracks slant toward the load, and flexure-shear cracks start vertical then bend.",
           quantity="",
           source="schematic of the three classes used for " + source_line().split(",")[0],
           caption=("Idealised crack patterns of a simply supported "
                    "reinforced-concrete beam under four-point loading: "
                    "flexural, shear and flexure-shear. P is a load, a the "
                    "shear span, d the effective depth. A drawing, not data."),
           inputs=["data/manifest.csv"], script=SCRIPT)
