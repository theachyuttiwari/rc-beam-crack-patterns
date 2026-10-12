"""Move composite panels (several specimens in one image) out of the labelled manifest.

Their per-specimen crops already exist in the manifest, so keeping the composite double-counts the
specimens and lets a test specimen appear inside a training image. Single-specimen images that were
flagged 'composite_unsplit' only because they show several load stages are kept.
Run from data/: python3 -I scripts/drop_composites.py
"""
import json
import os
import pandas as pd

KEEP_SINGLE = {"PMC12639117", "PMC11541560", "PMC7344945"}  # one specimen per image, flag was about load stages

m = pd.read_csv("manifest.csv")
notes = m.notes.fillna("")
is_comp = notes.str.contains("crop=composite_unsplit") & (m.label != "unknown")
pmc = m.image_path.str.split("/").str[1]
drop = is_comp & ~pmc.isin(KEEP_SINGLE)
print("composite labelled rows:", int(is_comp.sum()), "dropped:", int(drop.sum()), "kept single-specimen:", int((is_comp & ~drop).sum()))

ex = pd.read_csv("manifest_excluded.csv")
d = m[drop].copy()
d["reason"] = "composite figure of several specimens; per-specimen crops are in manifest.csv"
ex = pd.concat([ex, d], ignore_index=True)
ex.to_csv("manifest_excluded.csv", index=False)
m = m[~drop].copy()
m.to_csv("manifest.csv", index=False)

# absolute-path manifest for training
root = os.path.abspath(".")
a = m[m.image_ok != "no"].copy()
a["image_path"] = [os.path.join(root, p) for p in a.image_path]
a = a[[os.path.exists(p) for p in a.image_path]]
a.to_csv("manifest_abs.csv", index=False)

lab = m[m.label != "unknown"]
s = json.load(open("summary.json"))
s.update(images=len(m), labels=m.label.value_counts().to_dict(),
         specimens=lab.groupby("label").specimen_id.nunique().to_dict(),
         excluded=len(ex), note="2026-10-12: 37 composite panels moved to manifest_excluded.csv")
json.dump(s, open("summary.json", "w"), indent=1)
print(lab.label.value_counts().to_dict(), "specimens", lab.groupby(["source_slug", "specimen_id"]).ngroups, "articles", lab.source_slug.nunique())
