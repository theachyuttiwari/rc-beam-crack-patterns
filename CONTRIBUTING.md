# Contributing

## Add labelled images

1. Take images only from articles whose licence allows reuse with credit (CC BY 4.0 or CC BY 3.0). Do not add images under NC or ND licences.
2. Put each cropped panel in `data/panels/<PMCID>/`. Use PNG or JPG. One beam per file.
3. Add one row per image to `data/manifest.csv`. The columns are described in `data/README.md`. Use paths relative to the repository root, for example `data/panels/PMC1234567/Figure3_a.png`.
4. Set `label` to what the authors state for that specimen: `flexural`, `shear` or `shear_flexural`. Never label from how the picture looks. If the authors state nothing, use `unknown` and `label_basis=needs_manual`. If they state a mode outside the three classes (for example debonding or rupture of the strengthening, cover separation, bond failure), use `unknown` and `label_basis=other_mode`, with the quote.
5. Fill `evidence_quote` with a verbatim sentence from the article that names the failure mode. Fill `evidence_location` (section, paragraph or table) and `evidence_scope` (`per_specimen` only if the quote or its location names this beam, `group_level` if it states a mode for a group of beams).
6. Add the article to `data/SOURCES.md` with the citation, DOI link and licence.
7. Run `make data-check`. It must print `data-check OK`.

## Run the checks

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make data-check     # manifest schema, label words, every file exists
python -m pytest -q # unit tests
```

## Code changes

Keep all hyper-parameters in `src/config.py`. Do not change a result file by hand. Re-run `make train` and commit the new files in `results/` together with the code change that caused them.

## Questions

Open an issue, or write to achyut@geoliquefy.com.
