# rc-beam-crack-patterns

A dataset of crack-pattern images of reinforced-concrete (RC) beams, and the code for a baseline that classifies them as flexural, shear or shear-flexural. This is the release repository for the paper "Classification of Flexural, Shear and Shear-Flexural Crack Patterns in Reinforced Concrete Beams with HOG Features and Classical Machine Learning under Specimen-Level and Source-Level Validation" (Tiwari, 2026).

## What is in it

- 375 images cropped from 67 open-access journal articles. 295 carry a failure-mode label that the authors of the article state in their text. 80 are unlabelled. Every label has a verbatim evidence quote.
- A pipeline: grey-scale, CLAHE, letterbox resize, black-hat crack enhancement, HOG features (3780 values), and five classifiers (SVM, random forest, kNN, decision tree, XGBoost).
- Three validation schemes: by specimen, by article, and by image (the leaky one, kept to show the leak). Leave-one-article-out is run too.
- The result files behind every table in the paper, the figure scripts with their provenance sidecars, and the paper source (LaTeX, PDF and DOCX).

## Headline result

The classes are flexural 149, shear 115 and shear-flexural 31 images, from 275 specimens in 50 articles. A model that always answers "flexural" scores 50.5%.

- Specimen-level folds: SVM, kNN and XGBoost reach about 70% accuracy (SVM 73.6%, kNN 69.4%, XGBoost 68.1%).
- Article-level folds, where no article is in both train and test: accuracy falls to 44 to 49%. All five classifiers fall below the majority rate.
- Leave-one-article-out agrees with the article-level folds.
- A descriptor that keeps only gradient direction (18-bin orientation histogram) scores lower at specimen level (44 to 61%) but loses less at article level (42 to 55%). Direction transfers better between articles than layout does.

The 70% result comes from beams of the same test programme appearing on both sides of the split. It does not show that the model reads crack patterns. Split by article when you use this dataset.

## Where is what

| Path | Content |
|---|---|
| `data/manifest.csv` | The dataset: one row per image, with label, evidence quote and licence. |
| `data/manifest_excluded.csv` | Panels removed after review, with reasons. |
| `data/panels/<PMCID>/` | The cropped images. |
| `data/README.md` | Column schema, labelling protocol, counts, caveats. |
| `data/SOURCES.md` | One row per source article: citation, DOI, licence, image counts. |
| `src/` | Code: `config.py`, `preprocess.py`, `features.py`, `train_eval.py`, `make_tables.py`, `make_ablation_table.py`, `data_check.py`, `drop_composites.py`, `ste_lint.py`; draft figure scripts `make_figures.py`, `make_figures_extra.py`. |
| `src/figures_v2/` | The eight paper figures, one script each. Each script reads result files, re-checks the numbers it draws, and writes a `.meta.json` sidecar. |
| `tests/` | pytest suite. |
| `.github/workflows/ci.yml` | Runs the data check and the tests on every push. |
| `results/main/` | Main run (HOG). Metrics, confusion matrices, predictions, leave-one-article-out, timing, environment. |
| `results/noblackhat/` | Ablation: HOG without black-hat enhancement. |
| `results/orient/` | Ablation: 18-bin orientation histogram. |
| `paper/` | `main.tex`, `tables/` (generated), `build.sh`, `main.pdf`, `main.docx`. |
| `figures/` | The eight paper figures as PNG and PDF, each with a `.png.meta.json` provenance sidecar. |
| `refs/` | `references.bib` and `literature.md`. |
| `Makefile` | `setup`, `data-check`, `test`, `train`, `tables`, `figures`, `figures-draft`, `paper`, `lint`, `reproduce`, `clean`. |

## Quickstart

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make data-check
python -m pytest -q
make reproduce
```

`make reproduce` runs the data check, the three training runs and the tables. Each run takes about 15 minutes on a laptop and overwrites `results/`. Results were produced with Python 3.12, seed 42. See `results/main/environment.txt`.

`make figures` rebuilds the paper figures. It needs the `geoliquefy_figstyle` package (the house figure style, not on PyPI); without it, `make figures-draft` draws plain matplotlib versions of the same data into `figures/draft/`. `make paper` builds `paper/main.pdf` with tectonic and `paper/main.docx` with pandoc.

## Licences

- Images: CC BY 4.0, taken from the articles listed in `data/SOURCES.md`. You must credit the original authors. See `DATA-LICENSE.md`.
- Code: MIT, see `LICENSE`.

The Davoudi et al. crack-map set (Mendeley, DOI 10.17632/vkgw6gg8v3.1) is not included. `data/README.md` says how to get it.

## Cite

Use `CITATION.cff` (GitHub shows a "Cite this repository" button). Cite the paper above for the method and the results.

## Add images

Add one row per image to `data/manifest.csv` with these columns: `image_path`, `source_slug`, `specimen_id`, `label`, `label_basis`, `licence`, `citation`, `notes`, `beam_type`, `evidence_quote`, `evidence_location`, `evidence_scope`, `image_ok`, `modality`. The label must be the failure mode the authors state, with a verbatim quote. The full schema and the rules are in `data/README.md` and `CONTRIBUTING.md`. Run `make data-check` after you edit.

## Contact

Achyut Tiwari, achyut@geoliquefy.com, GitHub: theachyuttiwari.
