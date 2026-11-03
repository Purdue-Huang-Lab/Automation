"""SPAD23 pixel geometry: 23 pixels on a hexagonal lattice (as in spad23_counts_gui).

Five horizontal rows, top to bottom::

    row 0:   0   1   2   3   4
    row 1:     5   6   7   8
    row 2:   9  10  11  12  13
    row 3:    14  15  16  17
    row 4:  18  19  20  21  22

* sites in a row are 23 um apart along x; the 4-pixel rows are shifted by 11.5 um
* rows are 19.92 um apart along y; y = 19.92 * row, so row 0 is at y = 0 and the
  plots invert y to draw it at the top (the spad23_counts_gui convention)
* pixel 11 is the center pixel (x = 46 um, y = 39.84 um)

Spectral mode disperses along y: each row is one wavelength (the SPAD93
"column"), and the spectrum is read along the center column x = 46 um, where
the 4-pixel rows are sampled by the mean of their two middle pixels.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np


NUM_PIXELS = 23
N_ROWS = 5
SITE_PITCH_UM = 23.0        # pixel spacing within a row (along x)
LINE_PITCH_UM = 19.92       # row spacing (along y) = spacing of the wavelength lines
PITCH_X_UM = SITE_PITCH_UM  # 2D fits: lattice pitch along x
PITCH_Y_UM = LINE_PITCH_UM  # 2D fits: lattice pitch along y
PIXEL_DIAMETER_UM = 20.0    # drawing size of one pixel (nearest neighbours are 23 um apart)
CENTER_PIXEL = 11

ROWS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12, 13),
    (14, 15, 16, 17),
    (18, 19, 20, 21, 22),
)
_MAX_ROW = max(len(r) for r in ROWS)


def _build_layout():
    x = np.zeros(NUM_PIXELS, dtype=np.float64)
    y = np.zeros(NUM_PIXELS, dtype=np.float64)
    row = np.zeros(NUM_PIXELS, dtype=np.int64)
    for r, ids in enumerate(ROWS):
        x0 = 0.5 * SITE_PITCH_UM if len(ids) < _MAX_ROW else 0.0
        for k, p in enumerate(ids):
            x[p] = x0 + k * SITE_PITCH_UM
            y[p] = r * LINE_PITCH_UM
            row[p] = r
    return row, x, y


PIXEL_ROW, PIXEL_X_UM, PIXEL_Y_UM = _build_layout()
assert sorted(p for r in ROWS for p in r) == list(range(NUM_PIXELS))

# Sites of the lattice that are not pixels (none are known for SPAD23).
INACTIVE_XY = np.zeros((0, 2), dtype=np.float64)

ROW_Y_UM = np.arange(N_ROWS, dtype=np.float64) * LINE_PITCH_UM
LINE_SPAN_UM = (N_ROWS - 1) * LINE_PITCH_UM
# Pixel ids of each row, ordered by x (left to right).
ROW_PIXELS: List[np.ndarray] = [np.asarray(ids, dtype=np.int64) for ids in ROWS]
ROW_SIZES = np.array([ids.size for ids in ROW_PIXELS], dtype=np.int64)
# Rows whose center-column point is the mean of two pixels (the 4-pixel rows).
ROW_IS_SHORT = ROW_SIZES < _MAX_ROW
CENTER_X_UM = float(PIXEL_X_UM[CENTER_PIXEL])  # 46 um: the center (spectral) column

X_RANGE_UM = (float(PIXEL_X_UM.min()), float(PIXEL_X_UM.max()))
Y_RANGE_UM = (float(PIXEL_Y_UM.min()), float(PIXEL_Y_UM.max()))


def pixel_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of the pixel center coordinates (um)."""
    return PIXEL_X_UM.copy(), PIXEL_Y_UM.copy()


def row_of(pixel: int) -> int:
    return int(PIXEL_ROW[int(pixel)])


def row_pixels(row: int, excluded: Iterable[int] = ()) -> np.ndarray:
    """Pixel ids of row ``row`` ordered by x, without the excluded ones."""
    ids = ROW_PIXELS[int(row)]
    ex = set(int(p) for p in excluded)
    if not ex:
        return ids.copy()
    return np.asarray([p for p in ids if int(p) not in ex], dtype=np.int64)


def row_profile(values: np.ndarray, row: int, excluded: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pixel_ids, x_um, values)`` along row ``row`` (the x linecut)."""
    ids = row_pixels(row, excluded)
    vals = np.asarray(values, dtype=np.float64)
    return ids, PIXEL_X_UM[ids], vals[ids]


_COL_WEIGHT_CACHE: Dict[Tuple[float, FrozenSet[int]], np.ndarray] = {}


def column_weights(x_um: float = CENTER_X_UM, excluded: Iterable[int] = ()) -> np.ndarray:
    """Weights (N_ROWS x NUM_PIXELS) that sample every row at ``x_um``.

    A pixel exactly at ``x_um`` is used alone.  Otherwise the two pixels that
    straddle ``x_um`` are interpolated linearly; on the center column of a
    4-pixel row that is the plain mean of its two middle pixels.  If one of the
    two is excluded the other is used alone.  A row that cannot be sampled
    gets a row of NaN, so ``weights @ values`` is NaN there.
    """
    ex = frozenset(int(p) for p in excluded)
    key = (round(float(x_um), 6), ex)
    cached = _COL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w = np.zeros((N_ROWS, NUM_PIXELS), dtype=np.float64)
    tol = 1e-6
    for r in range(N_ROWS):
        ids = ROW_PIXELS[r]
        xs = PIXEL_X_UM[ids]
        exact = ids[np.abs(xs - x_um) < tol]
        if exact.size:
            p = int(exact[0])
            if p in ex:
                w[r, :] = np.nan
            else:
                w[r, p] = 1.0
            continue
        left = ids[xs < x_um]
        right = ids[xs > x_um]
        if left.size == 0 or right.size == 0:
            w[r, :] = np.nan
            continue
        lo = int(left[np.argmax(PIXEL_X_UM[left])])
        hi = int(right[np.argmin(PIXEL_X_UM[right])])
        x_lo, x_hi = PIXEL_X_UM[lo], PIXEL_X_UM[hi]
        if x_hi - x_lo > SITE_PITCH_UM + tol:
            w[r, :] = np.nan
            continue
        frac = (x_um - x_lo) / (x_hi - x_lo)
        lo_ok, hi_ok = lo not in ex, hi not in ex
        if lo_ok and hi_ok:
            w[r, lo] = 1.0 - frac
            w[r, hi] = frac
        elif lo_ok:
            w[r, lo] = 1.0
        elif hi_ok:
            w[r, hi] = 1.0
        else:
            w[r, :] = np.nan

    if len(_COL_WEIGHT_CACHE) > 256:
        _COL_WEIGHT_CACHE.clear()
    w.setflags(write=False)
    _COL_WEIGHT_CACHE[key] = w
    return w


def column_profile(values: np.ndarray, x_um: float = CENTER_X_UM, excluded: Iterable[int] = ()) -> np.ndarray:
    """Intensity of every row at ``x_um`` (5 values, NaN where unavailable)."""
    return column_weights(x_um, excluded) @ np.asarray(values, dtype=np.float64)


def center_column_matrix(counts: np.ndarray, excluded: Iterable[int] = ()) -> np.ndarray:
    """Center-column profile of every time bin: ``[bins, 23] -> [bins, 5]``."""
    return np.asarray(counts, dtype=np.float64) @ column_weights(CENTER_X_UM, excluded).T


def nearest_pixel(x_um: float, y_um: float, max_dist_um: Optional[float] = 0.5 * SITE_PITCH_UM) -> Optional[int]:
    d2 = (PIXEL_X_UM - x_um) ** 2 + (PIXEL_Y_UM - y_um) ** 2
    idx = int(np.argmin(d2))
    if max_dist_um is not None and float(np.sqrt(d2[idx])) > max_dist_um:
        return None
    return idx


def nearest_row(y_um: float) -> int:
    return int(np.clip(np.rint(y_um / LINE_PITCH_UM), 0, N_ROWS - 1))
