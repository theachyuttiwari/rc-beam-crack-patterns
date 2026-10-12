# Figures

Two sets of the same figures. `figures/*.png|pdf` are the web versions (house style). `figures/print/` holds the print versions that the paper uses: IEEE column (3.5 in) and page (7.2 in) widths, 8-pt text, TrueType fonts (IEEE PDF eXpress rejects the Type 3 fonts of the web set).

Each web PNG has a `.png.meta.json` sidecar: the script that drew it, its inputs (result files under `results/` and the manifest), the caption, and the checks it passed (word count, font floor, palette, nothing off the canvas).

| Figure | Shows |
|---|---|
| `fig_crack_types` | Schematic of the three crack patterns. |
| `fig_samples` | Example images per class from the dataset. |
| `fig_pipeline` | Preprocessing and feature steps. |
| `fig_hog` | One image per class through the pipeline, with its HOG cells. |
| `fig_split` | How image-, specimen- and article-level folds differ. |
| `fig_schemes` | Accuracy and macro F1 under the four validation schemes, with the majority class and the source prior. |
| `fig_confusion` | Confusion matrices under specimen-level folds. |
| `fig_ablation` | Full HOG, HOG without black-hat, and the orientation histogram. |
| `print/fig_metrics` | Print only: four metrics per scheme. |

`fig_samples` and `fig_hog` show images from the dataset, none with printed failure-mode words. Both sets show the same images (`src/picks.py`). Each image cell carries a number. `fig_samples_credits.md` and `fig_hog_credits.md` give the article, DOI, specimen and licence for each number, and the sidecar caption repeats them. Keep these credits with the figure when you reuse it (CC BY 4.0).

Rebuild with `make figures-print` (print set; `print/fig_samples_credits.tex` holds the short credits for the paper caption) and `make figures` (web set; needs `geoliquefy_figstyle`). Each script re-checks the numbers it draws against `results/` and stops if they differ.
