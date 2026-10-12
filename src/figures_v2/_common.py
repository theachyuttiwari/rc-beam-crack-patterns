"""Shared helpers for the figures_v2 scripts (GeoLiquefy figure protocol).

Reads persisted results only. Every figure script defines load / verify / draw
and ends with finish(), which saves PNG (gated, with sidecar) then PDF.
"""
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from geoliquefy_figstyle import (        # noqa: E402
    DECK_LIGHT, PORTAL, apply_web, data_palette, savefig_web, web_figure,
    web_titles)
from geoliquefy_figstyle import draw as _draw   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)           # so sidecar inputs are recorded relative to the paper

# White page for print; the portal's face (DM Sans, vendored) on the
# deck-light ground. Same data palette as every other GeoLiquefy figure.
THEME = DECK_LIGHT.replace(font_stack=tuple(PORTAL.font_stack),
                           font_dirs=tuple(PORTAL.font_dirs))
PAL = data_palette(THEME)
INK, MUTED = THEME.ink, THEME.secondary
HAIR, PANEL = THEME.chrome            # hairline grey, faint panel fill
OBS, BASE = PAL["observed"], PAL["baseline"]
OBS_SOFT, BASE_SOFT = PAL["observed_soft"], PAL["baseline_soft"]

CLFS = ["SVM", "RandomForest", "kNN", "DecisionTree", "XGBoost"]
CLF_NAMES = {"SVM": "SVM", "RandomForest": "Random forest", "kNN": "kNN",
             "DecisionTree": "Decision tree", "XGBoost": "XGBoost"}
CLASSES = ["flexural", "shear", "shear_flexural"]
CLASS_NAMES = {"flexural": "Flexural", "shear": "Shear",
               "shear_flexural": "Shear-flexural"}
# Share of the largest class among labelled images (data_check.py --write-summary).
MAJORITY = json.loads((ROOT / "data/summary.json").read_text())["labelled"]["majority_rate"]


def start(grid_axis="y", figsize=(12.0, 6.75)):
    apply_web(theme=THEME, grid_axis=grid_axis)
    return web_figure(figsize=figsize)


def frame(fig, rect, xlim, ylim):
    """Invisible axes that carries the titles/source and a data coordinate
    system for schematics. Sub-axes sit inside it."""
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    ax.grid(False)
    return ax


def labelled_manifest():
    m = pd.read_csv(ROOT / "data/manifest.csv")
    return m[m.label.isin(CLASSES)].reset_index(drop=True)


def source_line(m=None):
    m = labelled_manifest() if m is None else m
    n_spec = m.groupby(["source_slug", "specimen_id"]).ngroups
    return (f"{len(m)} images, {n_spec} specimens, "
            f"{m.source_slug.nunique()} open-access articles")


def fold_stats(results_dir, scheme, clf, col):
    """(mean, sd) from the persisted 'mean'/'sd' rows; refuse if the per-fold
    rows do not reproduce them (sample SD, ddof=1, over the five folds)."""
    m = pd.read_csv(ROOT / results_dir / "metrics.csv")
    s = m[(m.scheme == scheme) & (m.classifier == clf)]
    folds = s[s.fold.isin(list("01234"))][col].astype(float)
    mean = float(s[s.fold == "mean"][col].iloc[0])
    sd = float(s[s.fold == "sd"][col].iloc[0])
    if len(folds) != 5 or abs(folds.mean() - mean) > 1e-9 \
            or abs(folds.std(ddof=1) - sd) > 1e-9:
        raise AssertionError(f"{results_dir} {scheme} {clf} {col}: fold rows "
                             f"do not reproduce the mean/sd rows")
    return mean, sd


def word_count(fig):
    """Authored words: every visible text except tick labels and axis labels."""
    skip = {id(a) for a in _draw._tick_artists(fig)}
    for ax in fig.axes:
        skip |= {id(ax.xaxis.label), id(ax.yaxis.label)}
    n = 0
    for art in _draw._text_artists(fig):
        if art.get_visible() and id(art) not in skip:
            n += len(art.get_text().split())
    return n


def short_credit(citation):
    """'Liu X, Thermou GE (2024) Title. Journal. doi:10.x/y' -> 'Liu and Thermou (2024), doi:10.x/y'."""
    authors, year = re.match(r"^(.+?) \((\d{4})\) ", citation).groups()
    etal = authors.endswith(" et al.")
    names = [re.sub(r"\s+[A-Z]{1,3}$", "", a.strip())
             for a in authors.removesuffix(" et al.").split(",")]
    if len(names) == 1 and not etal:
        who = names[0]
    elif len(names) == 2 and not etal:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]} et al."
    doi = re.search(r"doi:(\S+)", citation).group(1).rstrip(".")
    return f"{who} ({year}), doi:{doi}"


def image_credits(rows):
    """Numbered credit lines (CC BY attribution) for the dataset images in a figure."""
    return [f"({i}) {short_credit(r.citation)}, specimen {r.specimen_id}, {r.licence}"
            for i, r in enumerate(rows, 1)]


def badge(ax, n):
    """Number an image cell; the number keys it to its credit line."""
    ax.text(0.03, 0.95, str(n), transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=HAIR, lw=1))


def finish(fig, fr, name, claim, quantity, source, caption, inputs, script,
           credits=None):
    """credits: image_credits() lines. They go into the caption and figures/<name>_credits.md."""
    web_titles(fr, claim=claim, quantity=quantity, source=source, theme=THEME)
    OUT.mkdir(exist_ok=True)
    if credits:
        caption += (" Images (numbered in the figure) are cropped and rescaled from: "
                    + "; ".join(credits) + ".")
        (OUT / f"{name}_credits.md").write_text(
            f"# Image credits for {name}\n\n"
            "Each number is the badge on an image cell. Every image is cropped and rescaled "
            "from the open-access article cited; full citations are in data/SOURCES.md.\n\n"
            + "".join(f"- {c}\n" for c in credits))
    png = OUT / f"{name}.png"
    savefig_web(fig, png, caption=caption, inputs=inputs, script=script,
                theme=THEME)
    # savefig_web records the absolute path; keep the committed sidecar machine-independent.
    side = png.with_name(png.name + ".meta.json")
    if side.is_file():
        meta = json.loads(side.read_text())
        meta["path"] = png.relative_to(ROOT).as_posix()
        side.write_text(json.dumps(meta, indent=2) + "\n")
    fig.savefig(OUT / f"{name}.pdf", facecolor=THEME.ground,
                edgecolor="none")
    print(f"{name}: authored words (no ticks/axis labels) = {word_count(fig)}")
    plt.close(fig)


def dotplot(ax, series, col, xlim, step, names=True, label_row=None,
            label_x_pad=0.012):
    """Mean +/- SD per classifier (rows), one dodged mark per series.
    series: list of (label, results_dir, scheme, colour, marker[, stats]),
    where stats is an optional {clf: (mean, sd)} instead of a fold read.
    label_row: label each series right of its bar on that row."""
    n = len(series)
    offs = (np.arange(n) - (n - 1) / 2) * step
    for si, s in enumerate(series):
        label, rdir, scheme, colour, marker = s[:5]
        for i, c in enumerate(CLFS):
            mean, sd = s[5][c] if len(s) > 5 else fold_stats(rdir, scheme, c, col)
            y = i + offs[si]
            ax.errorbar(mean, y, xerr=sd, fmt=marker, color=colour,
                        ms=8, lw=2, capsize=0, zorder=3)
            if label_row is not None and i == label_row:
                ax.text(mean + sd + label_x_pad, y, label, ha="left",
                        va="center", color=INK, fontsize=14,
                        bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlim(*xlim)
    ax.set_ylim(len(CLFS) - 0.45, -0.55)
    ax.set_yticks(range(len(CLFS)))
    ax.set_yticklabels([CLF_NAMES[c] for c in CLFS] if names else [])
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
