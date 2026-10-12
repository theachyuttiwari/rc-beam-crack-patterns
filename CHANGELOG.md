# Changelog

## 0.1.0 - 2026-10-12

First public release.

- 375 panels from 67 open-access articles in `data/manifest.csv` (295 with an author-stated failure mode from 275 specimens in 50 articles, 80 unlabelled), plus `data/manifest_excluded.csv` (77 removed panels, with reasons; 37 are composite figures whose per-specimen crops are kept).
- Pipeline: preprocessing, HOG (3780 values) and an 18-bin orientation histogram, five classifiers, specimen-level, article-level and image-level 5-fold validation, leave-one-article-out.
- Results of the three runs (main, no black-hat, orientation histogram) in `results/`.
- Eight paper figures in `figures/` with provenance sidecars, built by `src/figures_v2/`.
- Generated LaTeX tables and number macros in `results/tables/`.
- `make data-check`, `make train`, `make tables`, `make figures`, `make reproduce`, and a pytest suite that GitHub Actions runs on every push.
