"""Print figures for the paper (vector PDF with TrueType fonts + 300-dpi PNG), plain matplotlib, at IEEE
column (3.5 in) and page (7.2 in) widths with 8-pt text. src/figures_v2/ draws the web versions.

Usage: python make_figures.py --manifest data/manifest.csv --results results/main --out figures/print
Figures that need results or images are skipped (or drawn as placeholders) when missing.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

import config as C
import features as F
import preprocess as P

# ---- palette (light mode) -------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CLASS_COL = {"flexural": BLUE, "shear": ORANGE, "shear_flexural": AQUA}
CLASS_LABEL = {"flexural": "Flexural", "shear": "Shear", "shear_flexural": "Shear-flexural"}
CLF_ORDER = ["SVM", "RandomForest", "kNN", "DecisionTree", "XGBoost"]
CLF_SHORT = {"SVM": "SVM", "RandomForest": "RF", "kNN": "kNN", "DecisionTree": "DT", "XGBoost": "XGB"}
CLF_COL = dict(zip(CLF_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
HATCHES = dict(zip(CLF_ORDER, ["", "///", "\\\\\\", "...", "xx"]))
W1, W2 = 3.5, 7.2

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def is_synthetic(m):
    return m is not None and "synthetic" in m.columns and bool(m["synthetic"].astype(str).str.lower().eq("true").all())


def synth_banner(fig):
    fig.text(0.5, 1.0, "SYNTHETIC SMOKE-TEST DATA - NOT REAL BEAMS", ha="center", va="bottom",
             color="#d03b3b", fontsize=8, fontweight="bold")


def load_manifest(path):
    if path and Path(path).exists():
        m = pd.read_csv(path)
        m = m[m["label"].isin(C.CLASSES)].reset_index(drop=True)
        m["abs_path"] = [p if Path(p).is_absolute() else str(C.REPO_ROOT / p) for p in m["image_path"]]
        return m
    return None


# ---------------------------------------------------------------------------
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(W2, 1.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
    boxes = [
        ("Crack image", "photo of\nRC beam face"),
        ("Preprocess", "grey, CLAHE,\nletterbox 256x128,\nblur, black-hat"),
        ("HOG features", "9 orientations,\n16x16 cells,\n2x2 blocks, L2-Hys"),
        ("Classifiers", "SVM, RF, kNN,\nDT, XGBoost"),
        ("Specimen CV", "grouped folds,\ngrid search inside\ntraining folds"),
        ("Metrics", "accuracy, macro\nP / R / F1,\nconfusion matrices"),
    ]
    bw, gap = 13.8, 3.44
    for i, (t, s) in enumerate(boxes):
        x = i * (bw + gap)
        accent = BLUE if i == 4 else AXIS
        ax.add_patch(FancyBboxPatch((x, 3), bw, 14, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#f0efec", ec=accent, lw=1.2 if i == 4 else 0.7))
        ax.text(x + bw / 2, 14.2, t, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + bw / 2, 7.2, s, ha="center", va="center", fontsize=6.3, color=INK2)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.3, 10), (x + bw + gap - 0.3, 10),
                                         arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.8))
    save(fig, out, "fig_pipeline")


def _picks(m):
    """The same images as the paper figures in src/figures_v2 (one article per image, no printed
    failure-mode words), with their CC BY credit lines."""
    from picks import fit_cell, image_credits, load_rgb, pick_all
    return pick_all(m), fit_cell, load_rgb, image_credits


def _badge(ax, n):
    ax.text(0.02, 0.96, str(n), transform=ax.transAxes, ha="left", va="top", fontsize=7,
            fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=AXIS, lw=0.5))


def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.5)


def _write_credits(out, name, credits):
    (out / f"{name}_credits.md").write_text(
        f"# Image credits for {name}\n\nEach number is the badge on an image cell. Every image is cropped "
        "and rescaled from the open-access article cited; full citations are in data/SOURCES.md.\n\n"
        + "".join(f"- {c}\n" for c in credits))


def fig_samples(out, m, n=4):
    picks, fit_cell, load_rgb, image_credits = _picks(m)
    fig, axes = plt.subplots(3, n, figsize=(W2, 2.9))
    shown = []
    for r, c in enumerate(C.CLASSES):
        for j, row in enumerate(picks[c][:n]):
            ax = axes[r, j]
            ax.imshow(fit_cell(load_rgb(row.image_path)), aspect="auto")
            _frame(ax)
            shown.append(row)
            _badge(ax, len(shown))
            if j == 0:
                ax.set_ylabel("%s\n(%d images)" % (CLASS_LABEL[c], int((m.label == c).sum())),
                              color=INK, fontweight="bold", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    _write_credits(out, "fig_samples", image_credits(shown))
    # the same credits, short form, for the paper caption
    short = ["(%d)~%s" % (i, re.sub(r", doi:.*", "", c.split(") ", 1)[1]).replace(" (", " ").rstrip(")")
                              .replace("et al. ", "et al.\\ "))
             for i, c in enumerate(image_credits(shown), 1)]
    (out / "fig_samples_credits.tex").write_text("\\newcommand{\\sampleCredits}{%s}\n" % "; ".join(short))
    save(fig, out, "fig_samples")


def fig_hog(out, m):
    picks, fit_cell, load_rgb, image_credits = _picks(m)
    fig, axes = plt.subplots(3, 3, figsize=(W1, 2.6))
    for j, t in enumerate(["Image", "Preprocessed", "HOG"]):
        axes[0, j].set_title(t, color=INK2, fontsize=8)
    shown = []
    for r, c in enumerate(C.CLASSES):
        row = picks[c][0]
        shown.append(row)
        pre = P.preprocess(row.image_path)
        for j, (img, kw) in enumerate(((fit_cell(load_rgb(row.image_path)), {}),
                                       (pre, dict(cmap="gray", vmin=0, vmax=255)),
                                       (F.hog_visualisation(pre), dict(cmap="gray", vmin=0, vmax=1)))):
            axes[r, j].imshow(img, aspect="auto", **kw)
            _frame(axes[r, j])
        _badge(axes[r, 0], len(shown))
        axes[r, 0].set_ylabel(CLASS_LABEL[c], color=INK, fontweight="bold", fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    _write_credits(out, "fig_hog", image_credits(shown))
    save(fig, out, "fig_hog")


def _metric_table(res):
    mt = pd.read_csv(res / "metrics.csv")
    mt = mt[mt["fold"].isin(["mean", "sd"])]
    mean = mt[mt.fold == "mean"].set_index(["scheme", "classifier"])
    sd = mt[mt.fold == "sd"].set_index(["scheme", "classifier"])
    return mean, sd


def fig_metrics(out, res, m):
    mean, sd = _metric_table(res)
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Macro precision"),
               ("recall_macro", "Macro recall"), ("f1_macro", "Macro F1")]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.3), sharey=True)
    x = np.arange(len(CLF_ORDER)); bw = 0.38
    for ax, (k, title) in zip(axes, metrics):
        for off, scheme, col, hatch in [(-bw / 2, "specimen", BLUE, ""), (bw / 2, "image", ORANGE, "///")]:
            v = [mean.loc[(scheme, c), k] for c in CLF_ORDER]
            e = [sd.loc[(scheme, c), k] for c in CLF_ORDER]
            ax.bar(x + off, v, bw - 0.03, color=col, hatch=hatch, edgecolor=SURFACE, lw=0.6,
                   yerr=e, error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5, capthick=0.7))
        ax.set_title(title, color=INK2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([CLF_SHORT[c] for c in CLF_ORDER])
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(length=2)
    axes[0].set_ylabel("Score (mean +/- SD over folds)")
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec=SURFACE),
               Rectangle((0, 0), 1, 1, fc=ORANGE, ec=SURFACE, hatch="///")]
    fig.legend(handles, ["Specimen-level (grouped) CV", "Image-level (naive) CV, leaky"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(wspace=0.12)
    if is_synthetic(m):
        synth_banner(fig)
    save(fig, out, "fig_metrics")


def fig_confusion(out, res, m):
    cmap = LinearSegmentedColormap.from_list("seq", ["#f5f9fe", "#9ec5f4", "#2a78d6", "#0d366b"])
    short = ["Flex.", "Shear", "S-F"]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6))
    for ax, name in zip(axes.ravel(), CLF_ORDER):
        cm = pd.read_csv(res / f"confusion_{name}.csv", index_col=0).to_numpy().astype(float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(short); ax.set_yticks(range(3)); ax.set_yticklabels(short)
        ax.set_title(name, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("True")
    ax = axes[1, 2]
    ax.axis("off")
    cax = ax.inset_axes([0.15, 0.55, 0.7, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Row-normalised fraction"); cb.outline.set_visible(False)
    ax.text(0.5, 0.2, "Out-of-fold, specimen-level CV.\nFlex. = flexural, S-F = shear-flexural.",
            ha="center", va="center", color=INK2, fontsize=7, transform=ax.transAxes)
    fig.subplots_adjust(wspace=0.3, hspace=0.35)
    if is_synthetic(m):
        synth_banner(fig)
    save(fig, out, "fig_confusion")


def fig_split(out):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))
    rng = np.random.default_rng(3)
    n_spec, n_img = 6, 5
    test_spec = {1, 4}
    for ax, mode in zip(axes, ["image", "specimen"]):
        for s in range(n_spec):
            for i in range(n_img):
                if mode == "image":
                    test = rng.random() < 0.33
                else:
                    test = s in test_spec
                ax.add_patch(Rectangle((i * 1.15, -s * 1.15), 1, 1, fc=ORANGE if test else BLUE,
                                       ec=SURFACE, hatch="///" if test else "", lw=0.6))
            ax.text(-0.3, -s * 1.15 + 0.5, f"Specimen {s + 1}", ha="right", va="center", color=INK2)
        ax.set_xlim(-2.6, n_img * 1.15); ax.set_ylim(-n_spec * 1.15 + 0.1, 1.9)
        ax.set_aspect("equal"); ax.axis("off")
    axes[0].set_title("Image-level split: same specimen\nin train and test (leak)", color=INK, fontsize=8.5)
    axes[1].set_title("Specimen-level split: each specimen\nentirely in train or in test", color=INK, fontsize=8.5)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec=SURFACE), Rectangle((0, 0), 1, 1, fc=ORANGE, ec=SURFACE, hatch="///")]
    fig.legend(handles, ["Train image", "Test image"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.1, "Each square is one image; a row is the images photographed from one specimen.",
             ha="center", color=MUTED, fontsize=7)
    save(fig, out, "fig_split")


def fig_crack_types(out):
    """Simply supported beam, four-point bending: (a) flexural, (b) shear, (c) flexure-shear."""
    H, xs1, xs2, L = 0.8, 3.5, 6.5, 10.0   # beam depth, load points, span (support at 0.5 and 9.5)
    s1, s2 = 0.5, 9.5
    steel = 0.15

    def mirror(x):
        return s1 + s2 - x

    def beam(ax):
        ax.add_patch(Rectangle((s1 - 0.5, 0), L, H, fc="#f0efec", ec=INK2, lw=0.9))
        ax.plot([s1 - 0.5, s2 + 0.5], [steel, steel], color=MUTED, lw=1.2, ls=(0, (4, 2)))
        for sx in (s1, s2):
            ax.add_patch(Polygon([[sx, 0], [sx - 0.25, -0.35], [sx + 0.25, -0.35]], fc=INK2, ec=INK2, lw=0.6))
            ax.plot([sx - 0.4, sx + 0.4], [-0.35, -0.35], color=INK2, lw=0.9)
        for lx in (xs1, xs2):
            ax.add_patch(FancyArrowPatch((lx, 1.45), (lx, H + 0.02), arrowstyle="-|>", mutation_scale=7,
                                         color=INK, lw=1.0))
            ax.text(lx, 1.52, "P/2", ha="center", va="bottom", fontsize=7.5, color=INK)
        # shear span a (left) and d
        ax.annotate("", (s1, -0.62), (xs1, -0.62), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
        ax.text((s1 + xs1) / 2, -0.7, "a", ha="center", va="top", color=INK2, fontstyle="italic")
        ax.annotate("", (s2 + 0.35, H), (s2 + 0.35, steel), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
        ax.text(s2 + 0.48, (H + steel) / 2, "d", ha="left", va="center", color=INK2, fontstyle="italic")
        ax.text((xs1 + xs2) / 2, -0.7, "constant-moment region", ha="center", va="top", color=MUTED, fontsize=7)
        for lx in (xs1, xs2):
            ax.plot([lx, lx], [-0.5, -0.74], color=AXIS, lw=0.6)
        ax.set_xlim(-0.3, 11.2); ax.set_ylim(-0.95, 2.35)
        ax.set_aspect("equal"); ax.axis("off")

    def bez(ax, p0, p1, p2, col):
        path = MPath([p0, p1, p2], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(PathPatch(path, fc="none", ec=col, lw=1.5, capstyle="round"))

    fig, axes = plt.subplots(3, 1, figsize=(W2, 5.6))
    titles = ["(a) Flexural: vertical cracks in the constant-moment region",
              "(b) Shear: diagonal cracks in the shear span",
              "(c) Flexure-shear: start vertical, then incline toward the load point"]
    for ax, t in zip(axes, titles):
        beam(ax)
        ax.text(-0.3, 2.3, t, ha="left", va="top", fontweight="bold", color=INK, fontsize=8.5)
    # (a)
    for x0, h in zip([3.95, 4.6, 5.25, 5.9, 6.3], [0.45, 0.6, 0.55, 0.6, 0.42]):
        axes[0].plot([x0, x0 + 0.03], [0, h], color=BLUE, lw=1.5, solid_capstyle="round")
    # (b)
    for x0, dx in [(1.0, 1.0), (1.6, 1.0), (2.2, 1.0)]:
        for f in (lambda x: x, mirror):
            axes[1].plot([f(x0), f(x0 + dx)], [0.1, H - 0.03], color=ORANGE, lw=1.5, solid_capstyle="round")
    # (c)
    for x0, xe in [(1.4, 3.0), (2.1, 3.3)]:
        for f in (lambda x: x, mirror):
            bez(axes[2], (f(x0), 0), (f(x0 + 0.05), 0.65), (f(xe), H - 0.02), AQUA)
    for x0 in (4.5, 5.5):  # short flexural cracks between the loads
        axes[2].plot([x0, x0 + 0.02], [0, 0.4], color=AQUA, lw=1.5, solid_capstyle="round")
    fig.subplots_adjust(hspace=0.12)
    save(fig, out, "fig_crack_types")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out, res = Path(a.out), Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(a.manifest)
    fig_pipeline(out)
    fig_crack_types(out)
    fig_split(out)
    fig_samples(out, m)
    fig_hog(out, m)
    if (res / "metrics.csv").exists():
        fig_metrics(out, res, m)
        fig_confusion(out, res, m)
    else:
        print("no results/metrics.csv: skipped fig_metrics, fig_confusion")


if __name__ == "__main__":
    main()
