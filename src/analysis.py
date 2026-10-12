"""Analyses that test what the HOG classifiers learn. Reads the manifest and results/<run>/predictions.csv.

1. baselines   : pixel-free models on the same outer folds as train_eval.py: the majority class, the
                 article prior (the majority label of the test image's programme in the training part),
                 and random forests on image metadata (size, intensity, colour, modality).
2. bootstrap   : 95% intervals of accuracy and macro-F1 minus the majority baseline, resampling
                 programmes (the unit that the article-level split keeps together).
3. overlay     : the out-of-fold predictions scored on images with no overlay and on images without
                 printed failure-mode words.
4. within      : specimen-level folds inside the programmes that hold more than one class. There the
                 programme does not give the label, so a model must use the image.

Usage (from the repo root):
  python src/analysis.py --manifest data/manifest.csv --results results/main --out results/analysis
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import train_eval as T  # noqa: E402

warnings.filterwarnings("ignore")
SEED = C.SEED
CLASSES = C.CLASSES
N_BOOT = 2000
WITHIN_SEEDS = range(10)


def metrics(y, p, full=False):
    lab = range(len(CLASSES))
    out = dict(accuracy=accuracy_score(y, p), f1_macro=f1_score(y, p, labels=lab, average="macro", zero_division=0))
    if full:
        out.update(precision_macro=precision_score(y, p, labels=lab, average="macro", zero_division=0),
                   recall_macro=recall_score(y, p, labels=lab, average="macro", zero_division=0))
    return out


def outer_splits(m, y):
    """The specimen- and article-level outer folds of train_eval.py (same k, seed and grouping)."""
    k = min(C.N_FOLDS, int(m.groupby("label")["group"].nunique().min()))
    k_art = min(C.N_FOLDS, int(m.groupby("label")["programme"].nunique().min()))
    sp = StratifiedGroupKFold(k, shuffle=True, random_state=SEED)
    sa = StratifiedGroupKFold(k_art, shuffle=True, random_state=SEED)
    return {"specimen": list(sp.split(np.zeros(len(y)), y, m["group"])),
            "article": list(sa.split(np.zeros(len(y)), y, m["programme"]))}


def metadata(m):
    """Pixel statistics that say nothing about cracks: size, aspect, grey level, colour, modality."""
    rows = []
    for p, mod in zip(m["abs_path"], m["modality"]):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b, gr, r = (img[:, :, i].astype(np.float32) for i in range(3))
        h, w = g.shape
        rows.append(dict(width=w, height=h, aspect=w / h, grey_mean=g.mean(), grey_std=g.std(),
                         colour=(np.abs(r - gr) + np.abs(gr - b)).mean(), white=(g > 240).mean(),
                         is_map=float(mod == "map"), is_photo_map=float(mod == "photo+map")))
    return pd.DataFrame(rows)


def prior_predict(y, prog, tr, te, fallback):
    """Article prior: each test image gets the majority label of its programme in the training part,
    or the training majority if its programme is not in the training part."""
    lab = pd.Series(y[tr]).groupby(prog[tr]).agg(lambda s: s.value_counts().idxmax())
    return np.array([lab.get(g, fallback) for g in prog[te]])


def baselines(m, y, splits, meta):
    prog = m["programme"].to_numpy()
    size_cols = ["width", "height", "aspect"]
    rows, preds = [], []
    for scheme, folds in splits.items():
        for name in ("majority", "article_prior", "metadata_rf", "size_rf"):
            p_all = np.empty_like(y)
            for tr, te in folds:
                maj = np.bincount(y[tr], minlength=len(CLASSES)).argmax()
                if name == "majority":
                    p = np.full(len(te), maj)
                elif name == "article_prior":
                    p = prior_predict(y, prog, tr, te, maj)
                else:
                    cols = size_cols if name == "size_rf" else list(meta.columns)
                    rf = RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=4)
                    p = rf.fit(meta.iloc[tr][cols], y[tr]).predict(meta.iloc[te][cols])
                p_all[te] = p
                rows.append(dict(scheme=scheme, model=name, **metrics(y[te], p, full=True)))
            preds.append(pd.DataFrame(dict(scheme=scheme, model=name, image_path=m["image_path"],
                                           programme=prog, true=y, pred=p_all)))
    folds = (pd.DataFrame(rows).groupby(["scheme", "model"], sort=False)
             .agg(**{f"{k}{s}": (k, f) for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
                     for s, f in (("", "mean"), ("_sd", "std"))}).reset_index())
    return folds, pd.concat(preds, ignore_index=True)


def bootstrap(df, base, rng):
    """Pooled out-of-fold metric of `df` minus that of `base` (same images), resampling programmes."""
    progs = df["programme"].unique()
    by = {g: i for g, i in df.groupby("programme").indices.items()}
    yt, yp, yb = df["true"].to_numpy(), df["pred"].to_numpy(), base.loc[df.index, "pred"].to_numpy()
    out = {"accuracy": [], "f1_macro": []}
    for _ in range(N_BOOT):
        idx = np.concatenate([by[g] for g in rng.choice(progs, len(progs))])
        a, b = metrics(yt[idx], yp[idx]), metrics(yt[idx], yb[idx])
        for k in out:
            out[k].append(a[k] - b[k])
    point_a, point_b = metrics(yt, yp), metrics(yt, yb)
    return {k: dict(diff=point_a[k] - point_b[k], lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5))) for k, v in out.items()}


def within(m, y, X):
    """Specimen-level folds inside programmes with more than one class. Several seeds, as the subset is small."""
    multi = m.groupby("programme")["label"].nunique()
    keep = m["programme"].isin(multi[multi > 1].index).to_numpy()
    ms, ys, Xs = m[keep].reset_index(drop=True), y[keep], X[keep]
    prog, groups = ms["programme"].to_numpy(), ms["group"].to_numpy()
    k = min(C.N_FOLDS, int(ms.groupby("label")["group"].nunique().min()))
    rows = []
    for seed in WITHIN_SEEDS:
        sp = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
        preds = {n: np.empty_like(ys) for n in ("SVM", "majority", "article_prior")}
        for tr, te in sp.split(Xs, ys, groups):
            maj = np.bincount(ys[tr], minlength=len(CLASSES)).argmax()
            preds["SVM"][te] = T.fit_predict("SVM", Xs, None, ys, groups, tr, te, C.USE_PCA, "specimen")[0]
            preds["majority"][te] = maj
            preds["article_prior"][te] = prior_predict(ys, prog, tr, te, maj)
        for n, p in preds.items():
            rows.append(dict(seed=seed, model=n, **metrics(ys, p)))
    r = pd.DataFrame(rows)
    summ = r.groupby("model").agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                                  f1_macro=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std")).reset_index()
    info = dict(images=int(keep.sum()), specimens=int(ms["group"].nunique()),
                programmes=int(ms["programme"].nunique()), folds=k, seeds=len(WITHIN_SEEDS),
                classes={c: int((ms["label"] == c).sum()) for c in CLASSES},
                svm_beats_prior_seeds=int((r[r.model == "SVM"].f1_macro.values
                                           > r[r.model == "article_prior"].f1_macro.values).sum()))
    return summ, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    m = T.load_manifest(a.manifest)
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    y = m["label"].map(cls_idx).to_numpy()
    splits = outer_splits(m, y)

    # 1. baselines on the same folds
    meta = metadata(m)
    base, base_pred = baselines(m, y, splits, meta)
    base.to_csv(out / "baselines.csv", index=False)
    print(base.round(3).to_string(index=False))

    # 2 and 3. HOG classifiers: bootstrap against the majority baseline, and the overlay subsets
    P = pd.read_csv(Path(a.results) / "predictions.csv")
    P = P.merge(m[["image_path", "overlay"]], on="image_path", how="left", validate="many_to_one")
    assert P["overlay"].notna().all(), "predictions.csv does not match the manifest: re-run train_eval.py"
    P["true"], P["pred"] = P["true"].map(cls_idx), P["pred"].map(cls_idx)
    boot, sub = [], []
    for scheme in ("specimen", "article"):
        bm = base_pred[(base_pred.scheme == scheme) & (base_pred.model == "majority")].set_index("image_path")
        bp = base_pred[(base_pred.scheme == scheme) & (base_pred.model == "article_prior")].set_index("image_path")
        for clf, d in P[P.scheme == scheme].groupby("classifier"):
            d = d.set_index("image_path")
            for ref, b in (("majority", bm), ("article_prior", bp)):
                r = bootstrap(d, b, rng)
                boot.append(dict(scheme=scheme, classifier=clf, against=ref,
                                 **{f"{k}_{s}": v[s] for k, v in r.items() for s in ("diff", "lo", "hi")}))
            for name, sel in (("all", d.index == d.index), ("no_mode_text", ~d.overlay.str.contains("mode_text")),
                              ("overlay_none", d.overlay == "none")):
                dd = d[sel]
                maj = np.bincount(dd["true"], minlength=len(CLASSES)).max() / len(dd)
                sub.append(dict(scheme=scheme, classifier=clf, subset=name, n=len(dd),
                                majority_accuracy=maj, **metrics(dd["true"], dd["pred"])))
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(sub).to_csv(out / "overlay_subsets.csv", index=False)
    print(pd.DataFrame(boot).round(3).to_string(index=False))
    print(pd.DataFrame(sub).round(3).to_string(index=False))

    # 4. inside programmes with more than one class
    X, _, _ = T.build_features(m, "hog", C.USE_BLACKHAT, C.USE_CLAHE)
    wsum, winfo = within(m, y, X)
    wsum.to_csv(out / "within_programme.csv", index=False)
    (out / "within_programme.json").write_text(json.dumps(winfo, indent=1) + "\n")
    print(winfo)
    print(wsum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
