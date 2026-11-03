"""SPAD93 pixel geometry: 93 active pixels on a hexagonal lattice.

The sensor drawing shows 17 rows of 9 sites (23 um apart along a row, rows
19.92 um apart, odd rows shifted by half a pitch).  Everything here uses that
drawing rotated 90 deg counter-clockwise so the long axis is horizontal:

* column ``c`` (0..16, left to right) is row ``c`` of the drawing, at
  x = 19.92 * c um
* within a column, sites are 23 um apart along y (y points up); odd columns
  are shifted up by 11.5 um
* pixels 0-2 form the leftmost column, pixel 46 is the center pixel

Pixel numbers follow the drawing: row by row from its top-left corner.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np


NUM_PIXELS = 93
N_COLS = 17
SITES_PER_COL = 9
PITCH_X_UM = 19.92          # column spacing along x (row pitch of the drawing)
PITCH_Y_UM = 23.0           # site spacing along y (horizontal pitch of the drawing)
PIXEL_DIAMETER_UM = 20.0    # drawing size of one pixel (nearest neighbours are 23 um apart)
CENTER_PIXEL = 46
CENTER_Y_UM = 4.0 * PITCH_Y_UM   # 92 um: the center (spectral) row

# Mirror the displayed coordinates if the optics flip the image on the detector.
# The active area is symmetric under both flips, so only coordinates change.
FLIP_X = False
FLIP_Y = False

# Inclusive range of active sites in each column (= each row of the drawing).
ACTIVE_SITES: Tuple[Tuple[int, int], ...] = (
    (3, 5), (2, 5), (2, 6), (2, 5), (2, 6), (1, 6), (1, 7), (0, 7), (0, 8),
    (0, 7), (1, 7), (1, 6), (2, 6), (2, 5), (2, 6), (2, 5), (3, 5),
)

X_SPAN_UM = (N_COLS - 1) * PITCH_X_UM


def _site_xy(col: np.ndarray, site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = col * PITCH_X_UM
    y = (site + 0.5 * (col % 2)) * PITCH_Y_UM
    if FLIP_X:
        x = X_SPAN_UM - x
    if FLIP_Y:
        y = 2.0 * CENTER_Y_UM - y
    return x.astype(np.float64), y.astype(np.float64)


def _build_layout():
    cols: List[int] = []
    sites: List[int] = []
    for c, (first, last) in enumerate(ACTIVE_SITES):
        for s in range(first, last + 1):
            cols.append(c)
            sites.append(s)
    col_arr = np.asarray(cols, dtype=np.int64)
    site_arr = np.asarray(sites, dtype=np.int64)
    x, y = _site_xy(col_arr, site_arr)
    disp_col = (N_COLS - 1) - col_arr if FLIP_X else col_arr

    inactive_c: List[int] = []
    inactive_s: List[int] = []
    for c, (first, last) in enumerate(ACTIVE_SITES):
        for s in range(SITES_PER_COL):
            if not first <= s <= last:
                inactive_c.append(c)
                inactive_s.append(s)
    ix, iy = _site_xy(np.asarray(inactive_c), np.asarray(inactive_s))
    return disp_col, site_arr, x, y, np.column_stack((ix, iy))


PIXEL_COL, PIXEL_SITE, PIXEL_X_UM, PIXEL_Y_UM, INACTIVE_XY = _build_layout()
assert PIXEL_COL.size == NUM_PIXELS

COLUMN_X_UM = np.arange(N_COLS, dtype=np.float64) * PITCH_X_UM
# Pixel ids of each column, ordered by y (bottom to top).
COLUMN_PIXELS: List[np.ndarray] = []
for _k in range(N_COLS):
    _ids = np.flatnonzero(PIXEL_COL == _k)
    COLUMN_PIXELS.append(_ids[np.argsort(PIXEL_Y_UM[_ids])])
COLUMN_SIZES = np.array([ids.size for ids in COLUMN_PIXELS], dtype=np.int64)
# Columns whose center-row point is the mean of two pixels (odd columns).
COLUMN_IS_ODD = (np.arange(N_COLS) % 2).astype(bool)

X_RANGE_UM = (float(PIXEL_X_UM.min()), float(PIXEL_X_UM.max()))
Y_RANGE_UM = (float(PIXEL_Y_UM.min()), float(PIXEL_Y_UM.max()))


def pixel_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of the pixel center coordinates (um)."""
    return PIXEL_X_UM.copy(), PIXEL_Y_UM.copy()


def column_of(pixel: int) -> int:
    return int(PIXEL_COL[int(pixel)])


def column_pixels(col: int, excluded: Iterable[int] = ()) -> np.ndarray:
    """Pixel ids of column ``col`` ordered by y, without the excluded ones."""
    ids = COLUMN_PIXELS[int(col)]
    ex = set(int(p) for p in excluded)
    if not ex:
        return ids.copy()
    return np.asarray([p for p in ids if int(p) not in ex], dtype=np.int64)


def column_profile(values: np.ndarray, col: int, excluded: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pixel_ids, y_um, values)`` along column ``col`` (the y linecut)."""
    ids = column_pixels(col, excluded)
    vals = np.asarray(values, dtype=np.float64)
    return ids, PIXEL_Y_UM[ids], vals[ids]


_ROW_WEIGHT_CACHE: Dict[Tuple[float, FrozenSet[int]], np.ndarray] = {}


def row_weights(y_um: float = CENTER_Y_UM, excluded: Iterable[int] = ()) -> np.ndarray:
    """Weights (N_COLS x NUM_PIXELS) that sample every column at height ``y_um``.

    A pixel exactly at ``y_um`` is used alone.  Otherwise the two pixels that
    straddle ``y_um`` are interpolated linearly; on the center row of an odd
    column that is the plain mean of the two nearest pixels.  If one of the
    two is excluded the other is used alone.  A column that cannot be sampled
    gets a row of NaN, so ``weights @ values`` is NaN there.
    """
    ex = frozenset(int(p) for p in excluded)
    key = (round(float(y_um), 6), ex)
    cached = _ROW_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w = np.zeros((N_COLS, NUM_PIXELS), dtype=np.float64)
    tol = 1e-6
    for k in range(N_COLS):
        ids = COLUMN_PIXELS[k]
        ys = PIXEL_Y_UM[ids]
        exact = ids[np.abs(ys - y_um) < tol]
        if exact.size:
            p = int(exact[0])
            if p in ex:
                w[k, :] = np.nan
            else:
                w[k, p] = 1.0
            continue
        below = ids[ys < y_um]
        above = ids[ys > y_um]
        if below.size == 0 or above.size == 0:
            w[k, :] = np.nan
            continue
        lo = int(below[np.argmax(PIXEL_Y_UM[below])])
        hi = int(above[np.argmin(PIXEL_Y_UM[above])])
        y_lo, y_hi = PIXEL_Y_UM[lo], PIXEL_Y_UM[hi]
        if y_hi - y_lo > PITCH_Y_UM + tol:
            w[k, :] = np.nan
            continue
        frac = (y_um - y_lo) / (y_hi - y_lo)
        lo_ok, hi_ok = lo not in ex, hi not in ex
        if lo_ok and hi_ok:
            w[k, lo] = 1.0 - frac
            w[k, hi] = frac
        elif lo_ok:
            w[k, lo] = 1.0
        elif hi_ok:
            w[k, hi] = 1.0
        else:
            w[k, :] = np.nan

    if len(_ROW_WEIGHT_CACHE) > 256:
        _ROW_WEIGHT_CACHE.clear()
    w.setflags(write=False)
    _ROW_WEIGHT_CACHE[key] = w
    return w


def row_profile(values: np.ndarray, y_um: float = CENTER_Y_UM, excluded: Iterable[int] = ()) -> np.ndarray:
    """Intensity of every column at height ``y_um`` (17 values, NaN where unavailable)."""
    return row_weights(y_um, excluded) @ np.asarray(values, dtype=np.float64)


def center_row_matrix(counts: np.ndarray, excluded: Iterable[int] = ()) -> np.ndarray:
    """Center-row profile of every time bin: ``[bins, 93] -> [bins, 17]``."""
    return np.asarray(counts, dtype=np.float64) @ row_weights(CENTER_Y_UM, excluded).T


def nearest_pixel(x_um: float, y_um: float, max_dist_um: Optional[float] = 0.5 * PITCH_Y_UM) -> Optional[int]:
    d2 = (PIXEL_X_UM - x_um) ** 2 + (PIXEL_Y_UM - y_um) ** 2
    idx = int(np.argmin(d2))
    if max_dist_um is not None and float(np.sqrt(d2[idx])) > max_dist_um:
        return None
    return idx


def nearest_column(x_um: float) -> int:
    return int(np.clip(np.rint(x_um / PITCH_X_UM), 0, N_COLS - 1))
