"""Validate the dataset: both manifests, the image files, SOURCES.md and summary.json.

Usage (from the repo root):
  python src/data_check.py [--manifest data/manifest.csv] [--write-summary]
Exit code 0 if everything is valid, 1 otherwise. --write-summary rewrites data/summary.json
from the manifest (the only way summary.json should change).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED = ["image_path", "source_slug", "specimen_id", "label", "label_basis", "licence",
            "citation", "notes", "beam_type", "evidence_quote", "evidence_location",
            "evidence_scope", "image_ok", "modality"]
EXCLUDED_REQUIRED = ["image_path", "source_slug", "specimen_id", "reason"]
CLASSES = ["flexural", "shear", "shear_flexural"]
LABELS = set(CLASSES) | {"unknown"}
# paper_text: the authors state one of the three classes. other_mode: the authors state a mode
# outside the three classes (debonding, rupture, bond, bearing, ...). needs_manual: no usable statement.
BASES = {"paper_text", "other_mode", "needs_manual"}
SCOPES = {"per_specimen", "group_level"}
MODALITIES = {"photo", "map", "photo+map"}
BEAM_TYPES = {"plain_RC", "FRC", "FRP_reinforced", "strengthened_FRP", "corroded", "fire_damaged",
              "deep_beam", "with_openings", "other"}
LICENCES = {"CC BY 4.0", "CC BY 3.0"}
PANELS = "data/panels/"


def group_ids(m):
    return m["source_slug"].astype(str) + "::" + m["specimen_id"].astype(str)


def check_rows(m, root):
    """Row-level checks of manifest.csv. Returns a list of error strings."""
    errors = []
    for col, vocab in (("label", LABELS), ("label_basis", BASES), ("modality", MODALITIES),
                       ("beam_type", BEAM_TYPES), ("licence", LICENCES), ("image_ok", {"yes"})):
        bad = sorted(set(m[col]) - vocab)
        if bad:
            errors.append(f"{col}: unexpected values {bad}")
    bad = sorted(set(m["evidence_scope"]) - SCOPES - {""})
    if bad:
        errors.append(f"evidence_scope: unexpected values {bad}")
    for i, r in m.iterrows():
        where = f"row {i} ({r['image_path']})"
        for col in ("source_slug", "specimen_id", "citation"):
            if not r[col].strip():
                errors.append(f"{where}: empty {col}")
        labelled = r["label"] in CLASSES
        if labelled != (r["label_basis"] == "paper_text"):
            errors.append(f"{where}: label {r['label']!r} does not fit label_basis {r['label_basis']!r}")
        if r["label_basis"] in ("paper_text", "other_mode"):
            for col in ("evidence_quote", "evidence_location", "evidence_scope"):
                if not r[col].strip():
                    errors.append(f"{where}: {r['label_basis']} row needs {col}")
        p = r["image_path"]
        parts = p.split("/")
        if not p.startswith(PANELS) or ".." in parts or len(parts) != 4:
            errors.append(f"{where}: image_path must be {PANELS}<PMCID>/<file>")
            continue
        if r["source_slug"] != "paper-" + parts[2]:
            errors.append(f"{where}: source_slug does not match folder {parts[2]}")
        f = root / p
        if not f.is_file():
            errors.append(f"missing file: {p}")
            continue
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001 - any decode failure is a data error
            errors.append(f"{where}: not a readable image ({e.__class__.__name__})")
    return errors


def check(manifest_path, root=REPO_ROOT, verbose=True):
    """Return a list of error strings (empty if valid)."""
    manifest_path = Path(manifest_path)
    m = pd.read_csv(manifest_path, keep_default_na=False, dtype=str)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if m.empty:
        return ["manifest has no rows"]
    errors = check_rows(m, root)

    dup = m["image_path"][m["image_path"].duplicated()].tolist()
    if dup:
        errors.append(f"duplicate image_path: {dup[:5]}")
    lab = m[m["label"].isin(CLASSES)]
    n_lab = lab.groupby(group_ids(lab))["label"].nunique()
    for g in n_lab[n_lab > 1].index:
        errors.append(f"specimen {g} carries more than one label")

    data_dir = manifest_path.parent
    exc_path = data_dir / "manifest_excluded.csv"
    if exc_path.is_file():
        x = pd.read_csv(exc_path, keep_default_na=False, dtype=str)
        miss = [c for c in EXCLUDED_REQUIRED if c not in x.columns]
        if miss:
            errors.append(f"manifest_excluded.csv: missing columns {miss}")
        else:
            for p in sorted(set(x["image_path"]) & set(m["image_path"])):
                errors.append(f"in both manifests: {p}")
            for i, r in x.iterrows():
                if not (root / r["image_path"]).is_file():
                    errors.append(f"manifest_excluded.csv row {i}: missing file {r['image_path']}")
                if not r["reason"].strip():
                    errors.append(f"manifest_excluded.csv row {i}: empty reason")
        listed = set(m["image_path"]) | set(x.get("image_path", []))
        panels = root / PANELS
        if panels.is_dir():
            for f in sorted(panels.rglob("*")):
                rel = f.relative_to(root).as_posix()
                if f.is_file() and not f.name.startswith(".") and rel not in listed:
                    errors.append(f"image in no manifest: {rel}")

    sources = data_dir / "SOURCES.md"
    if sources.is_file():
        credited = set(re.findall(r"PMC\d+", sources.read_text()))
        used = {s.replace("paper-", "") for s in m["source_slug"]}
        for s in sorted(used - credited):
            errors.append(f"article {s} is not in SOURCES.md")

    summ = data_dir / "summary.json"
    if summ.is_file() and json.loads(summ.read_text()) != summary(m):
        errors.append("summary.json is out of date: run python src/data_check.py --write-summary")

    if verbose:
        print(f"{manifest_path}: {len(m)} images, {m['source_slug'].nunique()} articles, "
              f"{group_ids(m).nunique()} specimens")
        print("\nimages per class:")
        print(m["label"].value_counts().to_string())
        print("\nimages per label_basis:")
        print(m["label_basis"].value_counts().to_string())
    return errors


def summary(m):
    """Counts that data/README.md and the paper quote. Derived from the manifest only."""
    lab = m[m["label"].isin(CLASSES)]
    g = group_ids(lab)
    return {
        "images": int(len(m)),
        "articles": int(m["source_slug"].nunique()),
        "labels": {c: int((m["label"] == c).sum()) for c in CLASSES + ["unknown"]},
        "label_basis": {b: int((m["label_basis"] == b).sum()) for b in sorted(BASES)},
        "labelled": {
            "images": int(len(lab)),
            "specimens": int(g.nunique()),
            "articles": int(lab["source_slug"].nunique()),
            "specimens_per_class": {c: int(g[lab["label"] == c].nunique()) for c in CLASSES},
            "articles_per_class": {c: int(lab.loc[lab["label"] == c, "source_slug"].nunique())
                                   for c in CLASSES},
            "single_class_articles": int((lab.groupby("source_slug")["label"].nunique() == 1).sum()),
            "majority_rate": round(float(lab["label"].value_counts().max() / len(lab)), 4),
            "evidence_scope": {s: int((lab["evidence_scope"] == s).sum()) for s in sorted(SCOPES)},
            "modality_by_class": {c: {k: int(((lab["label"] == c) & (lab["modality"] == k)).sum())
                                      for k in sorted(MODALITIES)} for c in CLASSES},
        },
        "licences": {k: int(v) for k, v in m["licence"].value_counts().sort_index().items()},
    }


def write_summary(manifest_path):
    m = pd.read_csv(manifest_path, keep_default_na=False, dtype=str)
    out = Path(manifest_path).parent / "summary.json"
    out.write_text(json.dumps(summary(m), indent=1) + "\n")
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--write-summary", action="store_true")
    a = ap.parse_args(argv)
    if a.write_summary:
        write_summary(a.manifest)
    errors = check(a.manifest, verbose=not a.quiet)
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("\ndata-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
