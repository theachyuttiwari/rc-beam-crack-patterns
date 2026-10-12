"""Feature extraction: HOG (main), global orientation histogram (ablation), PCA."""
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

import config as C


def hog_features(img, visualize=False):
    return hog(img, orientations=C.HOG_ORIENTATIONS,
               pixels_per_cell=C.HOG_PIXELS_PER_CELL,
               cells_per_block=C.HOG_CELLS_PER_BLOCK,
               block_norm=C.HOG_BLOCK_NORM, visualize=visualize,
               feature_vector=True)


def hog_dim(h=C.IMG_H, w=C.IMG_W):
    cy, cx = h // C.HOG_PIXELS_PER_CELL[0], w // C.HOG_PIXELS_PER_CELL[1]
    by, bx = cy - C.HOG_CELLS_PER_BLOCK[0] + 1, cx - C.HOG_CELLS_PER_BLOCK[1] + 1
    return by * bx * C.HOG_CELLS_PER_BLOCK[0] * C.HOG_CELLS_PER_BLOCK[1] * C.HOG_ORIENTATIONS


def hog_visualisation(img):
    """Return the HOG picture (float array, rescaled to 0-1) for figures."""
    from skimage import exposure
    _, vis = hog_features(img, visualize=True)
    return exposure.rescale_intensity(vis, out_range=(0, 1))


def orientation_histogram(img, bins=C.ORIENT_BINS):
    """Global gradient-orientation histogram (unsigned, 0-180 deg), magnitude weighted,
    L1-normalised. Ablation: keeps direction statistics but drops all spatial layout."""
    f = img.astype(np.float32)
    gy, gx = np.gradient(f)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    h, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    s = h.sum()
    return (h / s if s > 0 else h).astype(np.float32)


def extract(img, kind="hog"):
    if kind == "hog":
        return hog_features(img)
    if kind == "orient":
        return orientation_histogram(img)
    raise ValueError(kind)


def make_pca():
    """Fitted inside each training fold by train_eval (never on test data)."""
    return PCA(n_components=C.PCA_VARIANCE, svd_solver="full", random_state=C.SEED)
