"""Validate a manifest: columns, label vocabulary, file existence. Prints counts.

Usage (from the repo root): python src/data_check.py [--manifest data/manifest.csv]
Exit code 0 if the manifest is valid, 1 otherwise.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED = ["image_path", "source_slug", "specimen_id", "label", "label_basis", "licence",
            "citation", "notes", "beam_type", "evidence_quote", "evidence_location",
            "evidence_scope", "image_ok", "modality"]
LABELS = {"flexural", "shear", "shear_flexural", "unknown"}
BASES = {"paper_text", "needs_manual"}
SCOPES = {"per_specimen", "group_level"}
MODALITIES = {"photo", "map"}


def check(manifest_path, root=REPO_ROOT, verbose=True):
    """Return a list of error strings (empty if valid)."""
    errors = []
    m = pd.read_csv(manifest_path, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for col, vocab in (("label", LABELS), ("label_basis", BASES),
                       ("evidence_scope", SCOPES), ("modality", MODALITIES)):
        bad = sorted(set(m[col]) - vocab - {""})
        if bad:
            errors.append(f"{col}: unexpected values {bad}")
    for p in m["image_path"]:
        if not (root / p).is_file():
            errors.append(f"missing file: {p}")
    dup = m["image_path"][m["image_path"].duplicated()].tolist()
    if dup:
        errors.append(f"duplicate image_path: {dup[:5]}")
    lab = m[m["label"].isin(["flexural", "shear", "shear_flexural"])]
    for i, r in lab.iterrows():
        if r["label_basis"] != "paper_text" or not str(r["evidence_quote"]).strip():
            errors.append(f"row {i} ({r['image_path']}): labelled but no paper_text evidence quote")
    for i, r in m.iterrows():
        if not str(r["licence"]).strip() or not str(r["citation"]).strip():
            errors.append(f"row {i} ({r['image_path']}): licence or citation empty")
    if verbose:
        print(f"{manifest_path}: {len(m)} images, {m['source_slug'].nunique()} articles, "
              f"{m['source_slug'].astype(str).add('::').add(m['specimen_id'].astype(str)).nunique()} specimens")
        print("\nimages per class:")
        print(m["label"].value_counts().to_string())
        print("\nimages per article (class counts):")
        print(pd.crosstab(m["source_slug"], m["label"]).to_string())
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
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
