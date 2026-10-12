# Figures

The eight figures of the paper, as PNG (what the DOCX embeds) and PDF (what the LaTeX build embeds).

Each PNG has a `.png.meta.json` sidecar: the script that drew it, its inputs (result files under `results/` and the manifest), the caption, and the checks it passed (word count, font floor, palette, nothing off the canvas).

| Figure | Shows |
|---|---|
| `fig_crack_types` | Schematic of the three crack patterns. |
| `fig_samples` | Example images per class from the dataset. |
| `fig_pipeline` | Preprocessing and feature steps. |
| `fig_hog` | One image per class through the pipeline, with its HOG cells. |
| `fig_split` | How image-, specimen- and article-level folds differ. |
| `fig_schemes` | Accuracy and macro F1 under the four validation schemes. |
| `fig_confusion` | Confusion matrices under specimen-level folds. |
| `fig_ablation` | Full HOG, HOG without black-hat, and the orientation histogram. |

Rebuild with `make figures` (needs `geoliquefy_figstyle`). Each script re-checks the numbers it draws against `results/` and stops if they differ.
