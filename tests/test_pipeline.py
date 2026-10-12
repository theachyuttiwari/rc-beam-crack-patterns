import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import data_check  # noqa: E402
import features as F  # noqa: E402
import preprocess as P  # noqa: E402


def first_panel():
    m = pd.read_csv(ROOT / "data" / "manifest.csv")
    return ROOT / m["image_path"].iloc[0]


def test_preprocess_shape():
    img = P.preprocess(first_panel())
    assert img.shape == (C.IMG_H, C.IMG_W)
    assert img.dtype.name == "uint8"


def test_hog_length():
    img = P.preprocess(first_panel())
    f = F.hog_features(img)
    assert f.shape == (3780,)
    assert F.hog_dim() == 3780


def test_orientation_histogram_sums_to_one():
    h = F.extract(P.preprocess(first_panel()), "orient")
    assert h.shape == (C.ORIENT_BINS,)
    assert abs(float(h.sum()) - 1.0) < 1e-4


def test_manifest_valid():
    assert data_check.check(ROOT / "data" / "manifest.csv", verbose=False) == []


def test_excluded_manifest_files_exist():
    e = pd.read_csv(ROOT / "data" / "manifest_excluded.csv")
    assert all((ROOT / p).is_file() for p in e["image_path"])
