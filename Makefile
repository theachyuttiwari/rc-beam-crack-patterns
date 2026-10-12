PY ?= python
MANIFEST = data/manifest.csv

.PHONY: setup data-check train tables figures figures-draft reproduce clean test

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
	$(PY) src/make_tables.py --manifest $(MANIFEST) --results results/main --out results/tables
	$(PY) src/make_ablation_table.py --hog results/main --noblackhat results/noblackhat --orient results/orient --out results/tables

# Paper figures. Needs the geoliquefy_figstyle package (house style, not on PyPI).
figures:
	for f in src/figures_v2/fig_*.py; do $(PY) $$f || exit 1; done
	$(PY) -m geoliquefy_figstyle check figures --require-sidecar

# Plain matplotlib drafts of the same data, no extra dependency.
figures-draft:
	$(PY) src/make_figures.py --manifest $(MANIFEST) --results results/main --out figures/draft
	$(PY) src/make_figures_extra.py --manifest $(MANIFEST) --results results/main --orient results/orient --noblackhat results/noblackhat --out figures/draft

test:
	$(PY) -m pytest -q

reproduce: data-check train tables

clean:
	rm -rf .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
