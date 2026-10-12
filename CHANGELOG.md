# Changelog

## Unreleased

- Label pass against the article texts: 4 rows moved between classes, 6 rows newly labelled, 24 moved to `unknown`. New `label_basis` value `other_mode` for modes outside the three classes. 277 labelled images (257 specimens, 51 articles), 98 unknown.
- `evidence_scope` is `per_specimen` only when the quote names the specimen (116 rows changed); `make data-check` enforces it.
- `make verify-quotes` checks every evidence quote against the article full text.
- New columns `programme` (one test programme reported in two articles is one group) and `overlay` (text and panels on each image, checked by eye). 62 caption bands cropped, 5 rotated beams fixed, 12 images set to `photo+map`, 4 duplicates excluded. 273 labelled images (253 specimens, 50 programmes); 70 have no overlay.
- Removed 314 panel images that no manifest listed.
- Numbered image credits in `fig_samples` and `fig_hog`.
- Leave-one-article-out inner CV is grouped by article.
- All runs re-done on the corrected data; groups are test programmes.
- `src/analysis.py` and `make analysis`: baselines that see no crack (majority class, source prior, metadata random forest), 95% bootstrap intervals that resample programmes, scores without overlay text, and a test inside the programmes with more than one class. Results in `results/analysis/`.
- `make tables` adds the baselines to the metrics table, an overlay-subset table and the number macros of these analyses.
- Print figures for the paper in `figures/print/` (`make figures-print`): IEEE widths, 8-pt text, TrueType fonts. The main figure shows the source prior. Example images carry no printed failure-mode words; `src/picks.py` shares the choice and the credits between both figure sets.
- Figure checks follow the corrected data; claims that no longer hold were rewritten.
- New paper title; `CITATION.cff` no longer names a conference.

## 0.1.0 - 2026-10-12

First public release.

- 375 panels from 67 open-access articles in `data/manifest.csv` (295 with an author-stated failure mode from 275 specimens in 50 articles, 80 unlabelled), plus `data/manifest_excluded.csv` (77 removed panels, with reasons; 37 are composite figures whose per-specimen crops are kept).
- Pipeline: preprocessing, HOG (3780 values) and an 18-bin orientation histogram, five classifiers, specimen-level, article-level and image-level 5-fold validation, leave-one-article-out.
- Results of the three runs (main, no black-hat, orientation histogram) in `results/`.
- Eight paper figures in `figures/` with provenance sidecars, built by `src/figures_v2/`.
- Generated LaTeX tables and number macros in `results/tables/`.
- `make data-check`, `make train`, `make tables`, `make figures`, `make reproduce`, and a pytest suite that GitHub Actions runs on every push.
