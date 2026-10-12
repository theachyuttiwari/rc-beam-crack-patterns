"""Ablation table and range macros: HOG vs HOG without black-hat vs 18-bin orientation histogram."""
import argparse
import os

import pandas as pd

NAMES = {"SVM": "SVM", "RandomForest": "RF", "kNN": "kNN", "DecisionTree": "DT", "XGBoost": "XGBoost"}
ORDER = ["SVM", "RandomForest", "kNN", "DecisionTree", "XGBoost"]
ap = argparse.ArgumentParser()
ap.add_argument("--hog", default="results/main", help="results dir of the main run")
ap.add_argument("--noblackhat", default="results/noblackhat")
ap.add_argument("--orient", default="results/orient")
ap.add_argument("--out", default="results/tables", help="output dir for the .tex files")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
RUNS = [("Hog", a.hog), ("NoBH", a.noblackhat), ("Orient", a.orient)]

M = {}
for key, r in RUNS:
    m = pd.read_csv(r + "/metrics.csv")
    M[key] = m[m.fold == "mean"].set_index(["scheme", "classifier"])

L = [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{Ablation of the descriptor: macro F1 (mean over five folds, in \%) under specimen-level and article-level folds for the full HOG, HOG without black-hat enhancement, and a global 18-bin orientation histogram that discards position.}",
     r"\label{tab:ablation}", r"\begin{tabular}{l rrr rrr}", r"\toprule",
     r" & \multicolumn{3}{c}{Specimen-level} & \multicolumn{3}{c}{Article-level} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Classifier & HOG & no BH & Orient. & HOG & no BH & Orient. \\", r"\midrule"]
for c in ORDER:
    cells = ["%.1f" % (100 * M[key].loc[(sch, c), "f1_macro"]) for sch in ["specimen", "article"] for key, _ in RUNS]
    L.append("%s & %s \\\\" % (NAMES[c], " & ".join(cells)))
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(a.out, "ablation_table.tex"), "w").write("\n".join(L) + "\n")

mac = ""
for key, _ in RUNS:
    for sch in ["specimen", "article"]:
        v = [100 * M[key].loc[(sch, c), "f1_macro"] for c in ORDER]
        mac += "\\newcommand{\\f%s%s}{%.1f--%.1f}\n" % (key, sch.capitalize(), min(v), max(v))
    acc = [100 * M[key].loc[("article", c), "accuracy"] for c in ORDER]
    mac += "\\newcommand{\\acc%sArt}{%.1f--%.1f}\n" % (key, min(acc), max(acc))
    acc = [100 * M[key].loc[("specimen", c), "accuracy"] for c in ORDER]
    mac += "\\newcommand{\\acc%sSpec}{%.1f--%.1f}\n" % (key, min(acc), max(acc))
open(os.path.join(a.out, "ablation_macros.tex"), "w").write(mac)
print(mac)
print("\n".join(L[10:15]))
