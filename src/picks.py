"""Example images for the figures: which images to show and their CC BY credit lines.

Shared by the paper figures (src/figures_v2) and the print figures (make_figures.py). No plotting
dependency, so both figure sets show the same images with the same credits.
"""
import re

import cv2
import numpy as np

N_PHOTO, N_MAP = 2, 2
CELL_W, CELL_H = 400, 200        # 2:1 cell, the pipeline's own aspect
CLASSES = ["flexural", "shear", "shear_flexural"]


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
            # no printed failure-mode words: the figure must not show the label as text
            sub = m[(m.label == label) & (m.modality == mod) & ~m.overlay.str.contains("mode_text")]
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
        # Too few images of one kind without printed failure-mode words: fill the row from the other kind.
        for r in m[(m.label == label) & ~m.overlay.str.contains("mode_text")].iloc[rng.permutation(
                int(((m.label == label) & ~m.overlay.str.contains("mode_text")).sum()))].itertuples():
            if len(rows) == N_PHOTO + N_MAP:
                break
            if r.source_slug not in used and _aspect_ok(r.image_path):
                rows.append(r)
                used.add(r.source_slug)
        assert len(rows) == N_PHOTO + N_MAP, (label, len(rows))
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
