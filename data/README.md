# Dataset

Images of reinforced-concrete (RC) beams after loading, cropped from figures of open-access articles. Each image has a failure mode (flexural, shear or shear-flexural) only where the article states one in its text. No label comes from how an image looks.

## Files

| File | What it holds |
|---|---|
| `manifest.csv` | The dataset. One row per image. |
| `manifest_excluded.csv` | Panels removed after review, with the reason. The image files are kept for audit. |
| `SOURCES.md` | One row per source article: citation, DOI, licence, image counts. |
| `panels/<PMCID>/` | The cropped images. |

Image paths in both manifests are relative to the repository root.

## Columns of `manifest.csv`

| Column | Meaning |
|---|---|
| `image_path` | Path to the cropped panel, relative to the repository root. |
| `source_slug` | Source article, `paper-<PMCID>`. |
| `specimen_id` | Beam name used by the authors (for example `M1B`). The group id for splitting is `source_slug::specimen_id`. |
| `label` | `flexural`, `shear`, `shear_flexural` or `unknown`. |
| `label_basis` | `paper_text` (the authors state the mode) or `needs_manual` (no label yet). |
| `licence` | Licence of the source article. All rows: CC BY 4.0. |
| `citation` | Full citation of the source article. |
| `notes` | Free text: figure number, source file, original size, crop box, reviewer comments. |
| `beam_type` | `plain_RC`, `FRC`, `FRP_reinforced`, `strengthened_FRP`, `corroded`, `fire_damaged`, `deep_beam`, `with_openings`, `other`. |
| `evidence_quote` | Verbatim sentence from the article that states the failure mode. |
| `evidence_location` | Where in the article the quote is. |
| `evidence_scope` | `per_specimen` (the text names this beam) or `group_level` (the text states a mode for a group of beams that includes it). Group-level labels are cruder. |
| `image_ok` | `yes` after review of the crop. |
| `modality` | `photo` (photograph of the beam face) or `map` (traced or drawn crack map). |

`manifest_excluded.csv` has the columns `image_path`, `source_slug`, `specimen_id`, `reason`, `modality`.

## Counts

375 images from 67 articles. 295 images carry a label. 80 are `unknown`. 77 panels are in the excluded manifest, 37 of them composite figures whose per-specimen crops are in `manifest.csv`. 64 labelled images have `group_level` evidence.

| label | images | specimens | photo | map |
|---|---|---|---|---|
| flexural | 149 | 138 | 119 | 30 |
| shear | 115 | 106 | 87 | 28 |
| shear_flexural | 31 | 31 | 24 | 7 |
| unknown | 80 | 72 | 61 | 19 |

The 295 labelled images come from 275 specimens in 50 articles. The majority class (flexural) is 50.5% of them. 33 of the 50 articles contribute one class only.

Composite figures that show several specimens in one image were moved from `manifest.csv` to `manifest_excluded.csv` on 2026-10-12, because their per-specimen crops are already rows and a composite would put a test specimen inside a training image. Each excluded row gives its reason. `summary.json` holds the counts; `python src/data_check.py --write-summary` regenerates it from the manifest.

## Labelling protocol

1. Specimen panels are cropped from composite figures. Gutters between panels are found automatically and corrected by hand where a reviewer flagged an error. The crop box is in `notes`.
2. A reviewer reads the article text and tables and records the failure mode the authors state for that specimen. The mapping is: flexural (flexural or bending failure, steel yielding then concrete crushing), shear (shear failure, diagonal tension, shear compression), shear_flexural (the authors say flexure-shear or combined).
3. Each label row has a verbatim quote (`evidence_quote`) and its location. The quote was re-checked against the article XML (ignoring case and white space). A row that failed the check was set to `unknown` and `needs_manual`.
4. `evidence_scope` records whether the quote names this specimen or a group.

## Caveats

- Crops come from JPG or WebP figures, typically 600 to 1600 px wide. Resolution varies. Some panels contain a photograph plus a crack drawing, load markers or printed labels.
- Many beams are strengthened, fibre-reinforced, FRP-reinforced or have openings (see `beam_type`). The class is the stated failure mode, not a clean population of plain RC beams.
- Prestressed girders, slabs, columns, finite-element contours and schematics are excluded.
- Specimens come from many different test programmes. Always split by article (`source_slug`) as well as by specimen. Splitting by image leaks near-duplicates.
- Classes are imbalanced and n is small. `shear_flexural` is the smallest class. Many articles describe the mode only for a group.
- No dedicated public image dataset of RC beam crack patterns with flexural or shear labels was found when this one was built.

## The Davoudi set (not included)

A separate crack-map dataset of 331 load-step maps from shear-critical tests (Davoudi et al., Mendeley Data, DOI 10.17632/vkgw6gg8v3.1, CC BY 4.0) has no per-specimen failure label. It is not part of this dataset and is not redistributed here. To use it, download it from https://doi.org/10.17632/vkgw6gg8v3.1 and keep it outside `data/panels/`.

## Add images

See `CONTRIBUTING.md` and run `make data-check` after editing the manifest.
