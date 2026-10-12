"""All hyper-parameters for the RC beam crack classification pipeline.

Edit values here only; every other module imports from this file.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # manifest image paths are relative to this

# ---- classes ---------------------------------------------------------------
CLASSES = ["flexural", "shear", "shear_flexural"]  # fixed order used everywhere

# ---- preprocessing ---------------------------------------------------------
IMG_H, IMG_W = 128, 256          # target size (rows, cols); letterboxed, aspect kept
AUTO_INVERT_DARK = True          # mostly-dark images (light cracks on black) are inverted
INVERT_MEDIAN_MAX = 64           # ... when the median grey level is below this
USE_CLAHE = True
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
BLUR_KSIZE, BLUR_SIGMA = 3, 0.8  # Gaussian blur after resize
USE_BLACKHAT = True              # crack enhancement (dark thin structures -> bright)
BLACKHAT_KSIZE = 9               # elliptical structuring element, pixels at target size

# ---- features --------------------------------------------------------------
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
ORIENT_BINS = 18                 # ablation descriptor: global gradient-orientation histogram
USE_PCA = False
PCA_VARIANCE = 0.95              # fraction of variance kept when PCA is on

# ---- evaluation ------------------------------------------------------------
SEED = 42
N_FOLDS = 5                      # outer folds
INNER_FOLDS = 3                  # inner CV for grid search
AUGMENT_HFLIP = False            # add horizontally flipped copies of TRAIN images only
GRID_N_JOBS = 4
MIN_SOURCES_FOR_LOSO = 3

# Small fixed grids, searched by inner CV on the training part of each outer fold.
GRIDS = {
    "SVM": {"clf__C": [1, 10, 100], "clf__kernel": ["linear", "rbf"]},
    "RandomForest": {"clf__n_estimators": [300], "clf__max_depth": [None, 20]},
    "kNN": {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["uniform", "distance"]},
    "DecisionTree": {"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [1, 3]},
    "XGBoost": {"clf__n_estimators": [200], "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.1]},
}
