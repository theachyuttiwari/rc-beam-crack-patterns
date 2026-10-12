"""Write the LaTeX tables and number macros for the paper from results/ and the manifest.

Usage (from the repo root):  python make_tables.py --manifest data/manifest.csv --results results/main --out paper/tables
Every number in the paper's tables comes from here; nothing is typed by hand.
"""
import argparse
import os
import pandas as pd
import numpy as np

import config as C

NAMES = {"SVM": "SVM", "RandomForest": "RF", "kNN": "kNN", "DecisionTree": "DT", "XGBoost": "XGBoost"}
ORDER = ["SVM", "RandomForest", "kNN", "DecisionTree", "XGBoost"]
CLS = {"flexural": "Flexural", "shear": "Shear", "shear_flexural": "Shear-flexural"}


def macro(name, value):
    return "\\newcommand{\\%s}{%s}\n" % (name, value)


def pct(x, d=1):
    return ("%." + str(d) + "f") % (100 * x)


def dataset_table(m):
    m = m[m.label.isin(C.CLASSES)].copy()
    m["group"] = m.source_slug.astype(str) + "::" + m.specimen_id.astype(str)
    rows = []
    for c in C.CLASSES:
        s = m[m.label == c]
        photo = int((s.modality == "photo").sum()) if "modality" in s else None
        rows.append((CLS[c], len(s), s.group.nunique(), s.source_slug.nunique(), photo))
    tot = ("Total", len(m), m.group.nunique(), m.source_slug.nunique(),
           int((m.modality == "photo").sum()) if "modality" in m else None)
    has_mod = "modality" in m
    head = "Class & Images & Specimens & Articles" + (" & Photographs" if has_mod else "") + " \\\\"
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Composition of the compiled dataset. Specimens are the unit of independence; the remaining images are traced crack maps.}" if has_mod else
             "\\caption{Composition of the compiled dataset. Specimens are the unit of independence.}",
             "\\label{tab:dataset}", "\\begin{tabular}{l r r r" + (" r" if has_mod else "") + "}", "\\toprule", head, "\\midrule"]
    for r in rows + [tot]:
        cells = [r[0], str(r[1]), str(r[2]), str(r[3])] + ([str(r[4])] if has_mod else [])
        if r[0] == "Total":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n", rows, tot


def metrics_table(res):
    g = res.groupby(["scheme", "classifier"])
    mean = g[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].mean()
    sd = g[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].std().fillna(0)
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Classification performance under three validation schemes (mean $\\pm$ SD over five outer folds, in \\%). Image-level folds may split a specimen; specimen-level folds never do; article-level folds keep every image of a source article in one fold.}",
             "\\label{tab:metrics}", "\\begin{tabular}{l l r r r r}", "\\toprule",
             "Split & Classifier & Accuracy & Precision (macro) & Recall (macro) & F1 (macro) \\\\", "\\midrule"]
    for scheme, lab in [("image", "Image-level"), ("specimen", "Specimen-level"), ("article", "Article-level")]:
        first = True
        for clf in ORDER:
            if (scheme, clf) not in mean.index:
                continue
            mu, s = mean.loc[(scheme, clf)], sd.loc[(scheme, clf)]
            cells = ["%s $\\pm$ %s" % (pct(mu[k]), pct(s[k])) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
            lines.append("%s & %s & %s \\\\" % (lab if first else "", NAMES[clf], " & ".join(cells)))
            first = False
        if scheme != "article":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n", mean, sd


def loso_table(loso, preds):
    """Pool the out-of-source predictions: most held-out articles contain one class, so
    per-article macro-F1 is meaningless; pooled accuracy and macro-F1 are reported instead."""
    if loso is None or loso.empty or preds is None:
        return "", None
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    d = preds[preds.scheme == "loso"]
    rows = {}
    for clf in ORDER:
        x = d[d.classifier == clf]
        if x.empty:
            continue
        pr, rc, f1, _ = precision_recall_fscore_support(x["true"], x["pred"], labels=C.CLASSES, average="macro", zero_division=0)
        rows[clf] = dict(acc=accuracy_score(x["true"], x["pred"]), prec=pr, rec=rc, f1=f1,
                         n_sources=x.source_slug.nunique())
    agg = pd.DataFrame(rows).T
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Leave-one-source-out: each article is held out in turn and the model is trained on all the others. Out-of-source predictions are pooled over all articles (in \\%).}",
             "\\label{tab:loso}", "\\begin{tabular}{l r r r r}", "\\toprule",
             "Classifier & Accuracy & Precision & Recall & F1 (macro) \\\\", "\\midrule"]
    for clf in ORDER:
        if clf in agg.index:
            r = agg.loc[clf]
            lines.append("%s & %s & %s & %s & %s \\\\" % (NAMES[clf], pct(r.acc), pct(r.prec), pct(r.rec), pct(r.f1)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n", agg


def timing_table(t):
    if t is None:
        return ""
    t = t[t.scheme == "specimen"].groupby("classifier").mean(numeric_only=True)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Cost on one laptop CPU. Feature extraction is shared by all classifiers.}",
             "\\label{tab:timing}", "\\begin{tabular}{l r r}", "\\toprule",
             "Classifier & Train (s) & Predict (ms/image) \\\\", "\\midrule"]
    for clf in ORDER:
        if clf in t.index:
            lines.append("%s & %.2f & %.2f \\\\" % (NAMES[clf], t.loc[clf, "train_s"], t.loc[clf, "predict_ms_per_image"]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n", t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(a.manifest)
    res = pd.read_csv(os.path.join(a.results, "metrics.csv"))
    res = res[pd.to_numeric(res.fold, errors="coerce").notna()].copy()  # drop mean/sd summary rows
    loso_p = os.path.join(a.results, "loso.csv")
    loso = pd.read_csv(loso_p) if os.path.exists(loso_p) else None
    tim_p = os.path.join(a.results, "timing.csv")
    tim = pd.read_csv(tim_p) if os.path.exists(tim_p) else None

    dt, rows, tot = dataset_table(m)
    open(os.path.join(a.out, "dataset_table.tex"), "w").write(dt)
    mt, mean, sd = metrics_table(res)
    open(os.path.join(a.out, "metrics_table.tex"), "w").write(mt)
    preds_p = os.path.join(a.results, "predictions.csv")
    preds = pd.read_csv(preds_p) if os.path.exists(preds_p) else None
    lt, lo = loso_table(loso, preds)
    open(os.path.join(a.out, "loso_table.tex"), "w").write(lt)
    tt, tmean = timing_table(tim)
    open(os.path.join(a.out, "timing_table.tex"), "w").write(tt)

    # ---- macros -----------------------------------------------------------
    mac = ""
    mac += macro("nImages", tot[1]) + macro("nSpecimens", tot[2]) + macro("nArticles", tot[3])
    for (lab, n, ns, na, ph) in rows:
        key = lab.replace("-", "").replace(" ", "")
        mac += macro("n" + key, n) + macro("nSpec" + key, ns)
    if tot[4] is not None:
        mac += macro("nPhotos", tot[4]) + macro("nMaps", tot[1] - tot[4])
    spec = mean.loc["specimen"]
    img = mean.loc["image"] if "image" in mean.index.get_level_values(0) else None
    for clf in ORDER:
        if clf in spec.index:
            k = NAMES[clf].replace("-", "")
            mac += macro("acc" + k, pct(spec.loc[clf, "accuracy"]))
            mac += macro("accSd" + k, pct(sd.loc[("specimen", clf), "accuracy"]))
            mac += macro("fone" + k, pct(spec.loc[clf, "f1_macro"]))
            mac += macro("foneSd" + k, pct(sd.loc[("specimen", clf), "f1_macro"]))
            if img is not None and clf in img.index:
                mac += macro("accImg" + k, pct(img.loc[clf, "accuracy"]))
                mac += macro("foneImg" + k, pct(img.loc[clf, "f1_macro"]))
                mac += macro("inflAcc" + k, pct(img.loc[clf, "accuracy"] - spec.loc[clf, "accuracy"]))
    art = mean.loc["article"] if "article" in mean.index.get_level_values(0) else None
    if art is not None:
        for clf in ORDER:
            if clf in art.index:
                k = NAMES[clf].replace("-", "")
                mac += macro("accArt" + k, pct(art.loc[clf, "accuracy"])) + macro("accArtSd" + k, pct(sd.loc[("article", clf), "accuracy"]))
                mac += macro("foneArt" + k, pct(art.loc[clf, "f1_macro"])) + macro("foneArtSd" + k, pct(sd.loc[("article", clf), "f1_macro"]))
        ba = art.f1_macro.idxmax()
        mac += macro("artBestClf", NAMES[ba]) + macro("artBestAcc", pct(art.loc[ba, "accuracy"])) + macro("artBestFone", pct(art.loc[ba, "f1_macro"]))
        drop = spec.accuracy - art.accuracy
        mac += macro("artDropMin", pct(drop.min())) + macro("artDropMax", pct(drop.max())) + macro("artDropMean", pct(drop.mean()))
    best = spec.f1_macro.idxmax()
    mac += macro("bestClf", NAMES[best]) + macro("bestFone", pct(spec.loc[best, "f1_macro"])) + macro("bestAcc", pct(spec.loc[best, "accuracy"]))
    worst = spec.f1_macro.idxmin()
    mac += macro("worstClf", NAMES[worst]) + macro("worstFone", pct(spec.loc[worst, "f1_macro"]))
    if img is not None:
        infl = (img.accuracy - spec.accuracy)
        mac += macro("inflMin", pct(infl.min())) + macro("inflMax", pct(infl.max())) + macro("inflMean", pct(infl.mean()))
    if lo is not None:
        agg = lo
        b = agg.f1.idxmax()
        mac += macro("losoBestClf", NAMES[b]) + macro("losoBestAcc", pct(agg.loc[b, "acc"])) + macro("losoBestFone", pct(agg.loc[b, "f1"]))
        mac += macro("losoNSources", int(agg.n_sources.max()))
        for clf in ORDER:
            if clf in agg.index:
                k = NAMES[clf].replace("-", "")
                mac += macro("losoAcc" + k, pct(agg.loc[clf, "acc"])) + macro("losoFone" + k, pct(agg.loc[clf, "f1"]))
    # per-class recall of the best classifier from its confusion matrix
    cm_p = os.path.join(a.results, "confusion_%s.csv" % best)
    if os.path.exists(cm_p):
        cm = pd.read_csv(cm_p, index_col=0)
        for c in C.CLASSES:
            r = cm.loc["true_" + c]
            rec = r["pred_" + c] / r.sum() if r.sum() else 0
            key = CLS[c].replace("-", "").replace(" ", "")
            mac += macro("recall" + key, pct(rec))
            # most common confusion
            o = r.drop("pred_" + c)
            mac += macro("conf" + key, CLS[o.idxmax().replace("pred_", "")]) + macro("confN" + key, int(o.max()))
    if tmean is not None:
        mac += macro("featMs", "%.1f" % (1000 * tim[tim.scheme == "specimen"].feature_extraction_s_per_image.mean()))
        mac += macro("trainMax", "%.1f" % tmean.train_s.max()) + macro("predMax", "%.2f" % tmean.predict_ms_per_image.max())
    # majority-class baseline
    mm = m[m.label.isin(C.CLASSES)]
    mac += macro("majorityAcc", pct(mm.label.value_counts().max() / len(mm)))
    open(os.path.join(a.out, "results_macros.tex"), "w").write(mac)
    print(mac)


if __name__ == "__main__":
    main()
