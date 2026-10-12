"""Train and evaluate five classical classifiers on HOG features.

Schemes (outer split; inner grid-search split in brackets)
  specimen : StratifiedGroupKFold, groups = source::specimen_id   [grouped by specimen]
  article  : StratifiedGroupKFold, groups = source article         [grouped by article]
  image    : naive StratifiedKFold over images (LEAKY; reported only to quantify inflation)
                                                                   [grouped by base image]
  loso     : leave-one-source-out, one article held out at a time  [grouped by article]
             (run if there are >= MIN_SOURCES_FOR_LOSO articles)
Hyper-parameters are chosen by inner CV (GridSearchCV) on the training part only. The inner
split uses the same grouping as the claim that the outer split tests, so model selection
cannot reward article cues in the article and LOSO schemes. Grouping by base image keeps
flipped copies of one image on one side of an inner split.

Per-article files (loso.csv, per_source.csv) hold counts and accuracy only. Most articles
contain one class, so a macro average over the three classes is not meaningful per article;
pooled metrics are computed from predictions.csv (see make_tables.py).
"""
import argparse
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import skimage
import xgboost
import cv2
import scipy
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

import config as C
import features as F
import preprocess as P

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("default", category=UndefinedMetricWarning)  # keep: signals a degenerate metric
SEED = C.SEED
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def base_estimator(name):
    if name == "SVM":
        return SVC(class_weight="balanced", random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "kNN":
        return KNeighborsClassifier()           # no class_weight support
    if name == "DecisionTree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    if name == "XGBoost":                        # balanced via sample_weight in fit
        return XGBClassifier(subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                             n_jobs=1, random_state=SEED, verbosity=0)
    raise ValueError(name)


def cpu_name():
    """CPU model name on macOS and Linux; platform.processor() elsewhere."""
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
        except Exception:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def hardware_string():
    cpu = cpu_name()
    return f"{cpu}; {platform.system()} {platform.release()} {platform.machine()}; Python {platform.python_version()}"


def load_manifest(path):
    m = pd.read_csv(path)
    m = m[m["label"].isin(C.CLASSES)].reset_index(drop=True)
    m["abs_path"] = [p if Path(p).is_absolute() else str(C.REPO_ROOT / p) for p in m["image_path"]]
    m["group"] = m["source_slug"].astype(str) + "::" + m["specimen_id"].astype(str)
    return m


def build_features(m, kind, blackhat, clahe):
    """Return X, X_flipped (or None), seconds per image (preprocess + features)."""
    X, Xf = [], []
    t0 = time.perf_counter()
    for p in tqdm(m["abs_path"], desc="features"):
        g = P.load_gray(p)
        X.append(F.extract(P.preprocess_array(g, clahe=clahe, blackhat=blackhat), kind))
        if C.AUGMENT_HFLIP:
            Xf.append(F.extract(P.preprocess_array(g, clahe=clahe, blackhat=blackhat, flip=True), kind))
    per_img = (time.perf_counter() - t0) / len(m) / (2 if C.AUGMENT_HFLIP else 1)
    return np.array(X, np.float32), (np.array(Xf, np.float32) if Xf else None), per_img


def make_search(name, use_pca, n_inner):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", F.make_pca()))
    steps.append(("clf", base_estimator(name)))
    cv = StratifiedGroupKFold(n_inner, shuffle=True, random_state=SEED)
    return GridSearchCV(Pipeline(steps), C.GRIDS[name], cv=cv, scoring="f1_macro",
                        n_jobs=C.GRID_N_JOBS, refit=True)


def fit_predict(name, X, Xf, y, groups, tr, te, use_pca, scheme):
    """Fit on train indices (plus flipped copies if enabled).
    Returns predictions, fit seconds, predict ms/image, chosen params."""
    Xtr, ytr = X[tr], y[tr]
    gtr = groups[tr] if scheme != "image" else tr   # inner groups
    if Xf is not None:
        Xtr = np.vstack([Xtr, Xf[tr]])
        ytr = np.concatenate([ytr, ytr])
        gtr = np.concatenate([gtr, gtr])
    n_inner = max(2, min(C.INNER_FOLDS, int(np.bincount(ytr).min())))
    gs = make_search(name, use_pca, n_inner)
    kw = {"groups": gtr}
    if name == "XGBoost":
        kw["clf__sample_weight"] = compute_sample_weight("balanced", ytr)
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr, **kw)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = gs.predict(X[te])
    pred_ms = (time.perf_counter() - t0) / len(te) * 1000
    return pred, fit_s, pred_ms, str(gs.best_params_)


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, p, labels=range(len(C.CLASSES)), average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), precision_macro=pr, recall_macro=rc, f1_macro=f1)


def counts(y, p):
    """Per-article summary: number of images, correct predictions, accuracy."""
    return dict(n_correct=int((np.asarray(y) == np.asarray(p)).sum()), accuracy=accuracy_score(y, p))


def pred_frame(m, te, p, scheme, name, fold, source=None):
    return pd.DataFrame(dict(
        scheme=scheme, classifier=name, fold=fold,
        image_path=m["image_path"].iloc[te].values,
        source_slug=m["source_slug"].iloc[te].values,
        specimen_id=m["specimen_id"].iloc[te].values,
        true=m["label"].iloc[te].values,
        pred=[C.CLASSES[i] for i in p]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="hog", choices=["hog", "orient"])
    ap.add_argument("--pca", action=argparse.BooleanOptionalAction, default=C.USE_PCA)
    ap.add_argument("--no-blackhat", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--classifiers", default="SVM,RandomForest,kNN,DecisionTree,XGBoost")
    ap.add_argument("--loso-jobs", type=int, default=1, help="parallel held-out sources in LOSO")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.classifiers.split(",")
    blackhat = C.USE_BLACKHAT and not a.no_blackhat
    clahe = C.USE_CLAHE and not a.no_clahe

    m = load_manifest(a.manifest)
    cls_idx = {c: i for i, c in enumerate(C.CLASSES)}
    y = m["label"].map(cls_idx).to_numpy()
    groups = m["group"].to_numpy()
    print(f"{len(m)} images, {m['group'].nunique()} specimens, {m['source_slug'].nunique()} sources")
    print(m["label"].value_counts().to_string())

    X, Xf, feat_s = build_features(m, a.features, blackhat, clahe)
    print("feature matrix", X.shape, f"{feat_s * 1000:.1f} ms/image")

    with open(out / "environment.txt", "w") as f:
        f.write(f"hardware: {hardware_string()}\n")
        f.write(f"python {sys.version.split()[0]}\nscikit-learn {sklearn.__version__}\n")
        f.write(f"scikit-image {skimage.__version__}\nxgboost {xgboost.__version__}\n")
        f.write(f"numpy {np.__version__}\npandas {pd.__version__}\nscipy {scipy.__version__}\n")
        f.write(f"opencv {cv2.__version__}\njoblib {joblib.__version__}\n")
        f.write(f"seed {SEED}\nfeatures {a.features} dim {X.shape[1]} pca {a.pca}\n")
        f.write(f"image size {C.IMG_H}x{C.IMG_W} blackhat {blackhat} clahe {clahe} "
                f"hflip_augment {C.AUGMENT_HFLIP}\n")
        f.write(f"n_images {len(m)} n_specimens {m['group'].nunique()} "
                f"n_sources {m['source_slug'].nunique()}\n")

    k = min(C.N_FOLDS, int(m.groupby("label")["group"].nunique().min()))
    sources_arr = m["source_slug"].to_numpy()
    k_art = min(C.N_FOLDS, int(m.groupby("label")["source_slug"].nunique().min()))
    # scheme -> (splitter, grouping array used for the outer split and the inner grid search)
    splitters = {
        "specimen": (StratifiedGroupKFold(k, shuffle=True, random_state=SEED), groups),
        "article": (StratifiedGroupKFold(k_art, shuffle=True, random_state=SEED), sources_arr),
        "image": (StratifiedKFold(k, shuffle=True, random_state=SEED), groups),
    }

    rows, preds, timing = [], [], []
    for scheme, (sp, garr) in splitters.items():
        folds = list(sp.split(X, y, garr))
        for name in names:
            for fi, (tr, te) in enumerate(tqdm(folds, desc=f"{scheme}/{name}")):
                p, fit_s, pred_ms, best = fit_predict(name, X, Xf, y, garr, tr, te, a.pca, scheme)
                rows.append(dict(scheme=scheme, classifier=name, fold=fi, n_test=len(te),
                                 best_params=best, **scores(y[te], p)))
                timing.append(dict(scheme=scheme, classifier=name, fold=fi,
                                   train_s=fit_s, predict_ms_per_image=pred_ms))
                preds.append(pred_frame(m, te, p, scheme, name, fi))

    # leave-one-source-out
    sources = sorted(m["source_slug"].unique())
    loso_rows = []
    if len(sources) >= C.MIN_SOURCES_FOR_LOSO:
        from joblib import Parallel, delayed
        src_arr = m["source_slug"].to_numpy()

        def _one(name, s):
            te = np.where(src_arr == s)[0]
            tr = np.where(src_arr != s)[0]
            p, fit_s, pred_ms, best = fit_predict(name, X, Xf, y, src_arr, tr, te, a.pca, "loso")
            return name, s, te, p, best, fit_s, pred_ms

        for name in names:
            outs = Parallel(n_jobs=a.loso_jobs)(delayed(_one)(name, s) for s in tqdm(sources, desc=f"loso/{name}"))
            for name_, s, te, p, best, fit_s, pred_ms in outs:
                loso_rows.append(dict(held_out_source=s, classifier=name_, n_test=len(te),
                                      classes_in_test=len(np.unique(y[te])),
                                      best_params=best, **counts(y[te], p)))
                timing.append(dict(scheme="loso", classifier=name_, fold=s,
                                   train_s=fit_s, predict_ms_per_image=pred_ms))
                preds.append(pred_frame(m, te, p, "loso", name_, s))
        pd.DataFrame(loso_rows).to_csv(out / "loso.csv", index=False)
    else:
        print(f"LOSO skipped: {len(sources)} source(s) < {C.MIN_SOURCES_FOR_LOSO}")

    # ---- write outputs
    per_fold = pd.DataFrame(rows)
    summ = []
    for (sch, clf), g in per_fold.groupby(["scheme", "classifier"], sort=False):
        summ.append(dict(scheme=sch, classifier=clf, fold="mean", **g[METRICS].mean().to_dict()))
        summ.append(dict(scheme=sch, classifier=clf, fold="sd", **g[METRICS].std(ddof=1).to_dict()))
    metrics = pd.concat([per_fold.astype({"fold": str}), pd.DataFrame(summ)], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)

    P_all = pd.concat(preds, ignore_index=True)
    P_all.to_csv(out / "predictions.csv", index=False)

    for sch, prefix in [("specimen", "confusion_"), ("article", "confusion_article_"), ("image", "confusion_image_")]:
        for name in names:
            d = P_all[(P_all.scheme == sch) & (P_all.classifier == name)]
            cm = confusion_matrix(d["true"], d["pred"], labels=C.CLASSES)
            df = pd.DataFrame(cm, index=[f"true_{c}" for c in C.CLASSES],
                              columns=[f"pred_{c}" for c in C.CLASSES])
            df.to_csv(out / f"{prefix}{name}.csv")

    ps = []
    for (name, src), d in P_all[P_all.scheme == "specimen"].groupby(["classifier", "source_slug"]):
        yt = d["true"].map(cls_idx).to_numpy()
        yp = d["pred"].map(cls_idx).to_numpy()
        ps.append(dict(classifier=name, source_slug=src, n=len(d),
                       classes_in_test=len(np.unique(yt)), **counts(yt, yp)))
    pd.DataFrame(ps).to_csv(out / "per_source.csv", index=False)

    t = (pd.DataFrame(timing).drop(columns="fold")
         .groupby(["scheme", "classifier"], sort=False).mean(numeric_only=True).reset_index())
    t.insert(2, "feature_extraction_s_per_image", feat_s)
    t["hardware"] = hardware_string()
    t.to_csv(out / "timing.csv", index=False)

    print("\nMean over folds (macro metrics)")
    mean = pd.DataFrame(summ).query("fold == 'mean'").drop(columns="fold")
    print(mean.round(3).to_string(index=False))
    if loso_rows:
        print("\nLOSO, pooled over held-out articles (macro metrics over all images):")
        lp = P_all[P_all.scheme == "loso"]
        for name in names:
            d = lp[lp.classifier == name]
            s = scores(d["true"].map(cls_idx).to_numpy(), d["pred"].map(cls_idx).to_numpy())
            print(f"  {name:13s} accuracy {s['accuracy']:.3f}  f1_macro {s['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
