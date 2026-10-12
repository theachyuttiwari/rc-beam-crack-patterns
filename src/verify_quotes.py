"""Check that every evidence quote in a manifest occurs in the source article.

Downloads the full-text XML of each article from Europe PMC (once; cached) and looks for
each quote in the article text, ignoring case and white space. Table cells are joined, so a
table row quoted as its cells in order, separated by spaces, also matches.

Usage (from the repo root): python src/verify_quotes.py [--manifest data/manifest.csv]
                                                        [--cache .cache/xml]
Needs network access the first time. Exit code 0 if every quote is found, 1 otherwise.
"""
import argparse
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def norm(s):
    """Unicode NFKC, case-folded, with all white space removed."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(s)).casefold())


def article_xml(pmcid, cache):
    path = cache / f"{pmcid}.xml"
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(pmcid=pmcid), timeout=60) as r:
            path.write_bytes(r.read())
        time.sleep(0.3)
    return path.read_bytes()


def article_text(xml_bytes):
    """Normalised text of the whole article: body, tables, captions, back matter."""
    return norm("".join(ET.fromstring(xml_bytes).itertext()))


def check(manifest_path, cache, verbose=True):
    """Return a list of (image_path, problem) for quotes that are not found."""
    m = pd.read_csv(manifest_path, keep_default_na=False)
    m = m[m["evidence_quote"].str.strip() != ""]
    texts, bad = {}, []
    for _, r in m.iterrows():
        pmcid = r["source_slug"].replace("paper-", "")
        if pmcid not in texts:
            texts[pmcid] = article_text(article_xml(pmcid, cache))
        if norm(r["evidence_quote"]) not in texts[pmcid]:
            bad.append((r["image_path"], "quote not found in " + pmcid))
    if verbose:
        print(f"{len(m)} quotes in {len(texts)} articles checked, {len(bad)} not found")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--cache", default=str(REPO_ROOT / ".cache" / "xml"))
    a = ap.parse_args(argv)
    bad = check(a.manifest, Path(a.cache))
    for p, why in bad:
        print(f"  {p}: {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
