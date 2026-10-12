"""Image preprocessing: grayscale -> CLAHE -> letterbox resize -> blur -> black-hat.

Augmentation policy: only a HORIZONTAL flip is offered. Mirroring a beam left-right
maps flexural cracks to flexural, and shear / flexure-shear cracks to the mirrored
shear span, so the class is unchanged. A VERTICAL flip swaps tension (bottom) and
compression (top) faces and would turn a real crack pattern into a physically
impossible one, so it is deliberately excluded.
"""
import cv2
import numpy as np

import config as C


def load_gray(path):
    """Load any photo or crack map as uint8 grayscale, dark cracks on light background.

    Handles JPG photographs, RGB/RGBA/palette PNGs, 16-bit and binary (0/255) crack maps.
    Transparent pixels (alpha) are composited onto white. If the image is mostly dark
    (median < INVERT_MEDIAN_MAX, e.g. a white-crack-on-black binary map) it is inverted.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:            # BGRA: composite on white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if C.AUTO_INVERT_DARK and np.median(img) < C.INVERT_MEDIAN_MAX:
        img = 255 - img
    return img


def letterbox(gray, h=C.IMG_H, w=C.IMG_W):
    """Resize keeping aspect ratio, pad the rest with the image median grey."""
    ih, iw = gray.shape
    s = min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    small = cv2.resize(gray, (nw, nh), interpolation=interp)
    canvas = np.full((h, w), int(np.median(small)), dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    return canvas


def preprocess_array(gray, clahe=C.USE_CLAHE, blackhat=C.USE_BLACKHAT, flip=False):
    """gray uint8 2-D array -> preprocessed uint8 array of shape (IMG_H, IMG_W)."""
    if clahe:
        gray = cv2.createCLAHE(C.CLAHE_CLIP, C.CLAHE_TILE).apply(gray)
    out = letterbox(gray)
    if C.BLUR_KSIZE > 1:
        out = cv2.GaussianBlur(out, (C.BLUR_KSIZE, C.BLUR_KSIZE), C.BLUR_SIGMA)
    if blackhat:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.BLACKHAT_KSIZE,) * 2)
        bh = cv2.morphologyEx(out, cv2.MORPH_BLACKHAT, k)  # dark cracks -> bright
        out = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    if flip:  # horizontal flip only (see module docstring)
        out = out[:, ::-1].copy()
    return out


def preprocess(path, **kw):
    return preprocess_array(load_gray(path), **kw)
