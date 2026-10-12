PY ?= python
MANIFEST = data/manifest.csv

.PHONY: setup data-check train tables figures figures-draft paper lint reproduce clean test

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

data-check:
	$(PY) src/data_check.py --manifest $(MANIFEST)

# Three runs: main (HOG), HOG without black-hat, 18-bin orientation histogram.
# Each run takes several minutes. Results overwrite results/<run>/.
train:
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/main
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/noblackhat --no-blackhat
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/orient --features orient

tables:
	$(PY) src/make_tables.py --manifest $(MANIFEST) --results results/main --out paper/tables
	$(PY) src/make_ablation_table.py --hog results/main --noblackhat results/noblackhat --orient results/orient --out paper/tables

# Paper figures. Needs the geoliquefy_figstyle package (house style, not on PyPI).
figures:
	for f in src/figures_v2/fig_*.py; do $(PY) $$f || exit 1; done
	$(PY) -m geoliquefy_figstyle check figures --require-sidecar

# Plain matplotlib drafts of the same data, no extra dependency.
figures-draft:
	$(PY) src/make_figures.py --manifest $(MANIFEST) --results results/main --out figures/draft
	$(PY) src/make_figures_extra.py --manifest $(MANIFEST) --results results/main --orient results/orient --noblackhat results/noblackhat --out figures/draft

# PDF with tectonic, DOCX with pandoc, from the figures in figures/.
paper:
	bash paper/build.sh ../figures

# Style check of the paper prose. Skipped if paper/main.tex is not present yet.
lint:
	$(PY) src/ste_lint.py paper/main.tex paper/tables/results_body.tex paper/tables/discussion_body.tex paper/tables/conclusion_body.tex

test:
	$(PY) -m pytest -q

reproduce: data-check train tables

clean:
	rm -rf .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
