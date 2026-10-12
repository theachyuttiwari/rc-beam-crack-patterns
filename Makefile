# Use the project venv when it exists, else the python on PATH.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MANIFEST = data/manifest.csv

.PHONY: setup data-check verify-quotes train analysis tables figures figures-print paper lint reproduce clean test

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data-check:
	$(PY) src/data_check.py --manifest $(MANIFEST)

# Check every evidence quote against the article full text (needs network the first time).
verify-quotes:
	$(PY) src/verify_quotes.py --manifest $(MANIFEST)

# Three runs: main (HOG), HOG without black-hat, 18-bin orientation histogram.
# The HOG runs take about 15 to 30 minutes each on a laptop, the orientation run about 1 minute.
# Results overwrite results/<run>/.
train:
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/main
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/noblackhat --no-blackhat
	$(PY) src/train_eval.py --manifest $(MANIFEST) --out results/orient --features orient

# Baselines, bootstrap over sources, overlay subsets and the within-source test (needs results/main).
analysis:
	$(PY) src/analysis.py --manifest $(MANIFEST) --results results/main --out results/analysis

tables:
	$(PY) src/make_tables.py --manifest $(MANIFEST) --results results/main --out results/tables
	$(PY) src/make_ablation_table.py --hog results/main --noblackhat results/noblackhat --orient results/orient --out results/tables

# Paper figures. Needs the geoliquefy_figstyle package (house style, not on PyPI).
figures:
	for f in src/figures_v2/fig_*.py; do $(PY) $$f || exit 1; done
	$(PY) -m geoliquefy_figstyle check figures/*.png --require-sidecar

# Print figures for the paper (IEEE widths, 8-pt text, TrueType fonts), plain matplotlib, no extra dependency.
figures-print:
	$(PY) src/make_figures.py --manifest $(MANIFEST) --results results/main --out figures/print
	$(PY) src/make_figures_extra.py --manifest $(MANIFEST) --results results/main --orient results/orient --noblackhat results/noblackhat --out figures/print

# The paper sources live in paper/ (not in the public repository). Needs tectonic and pandoc.
paper: tables figures-print
	cp results/tables/*.tex paper/tables/
	cd paper && bash build.sh ../figures/print

lint:
	python3 paper/ste_lint.py paper/main.tex

test:
	$(PY) -m pytest -q

reproduce: data-check test train analysis tables figures-print

clean:
	rm -rf .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
