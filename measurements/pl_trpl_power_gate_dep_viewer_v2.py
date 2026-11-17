"""
pl_trpl_power_gate_dep_viewer_v2.py

PL + TRPL power / gate dependence offline viewer  —  version 2, one file.

For folders of the PL + TRPL v2 measurement GUI (PL .asc files, SPAD23 or
SPAD93 TRPL_*.csv files, session.md).  Needs numpy, scipy, pandas, PyQt5, pyqtgraph.

    python pl_trpl_power_gate_dep_viewer_v2.py [--folder DATA_FOLDER]

• PL: the PL power/gate viewer v2 (pl_power_gate_dep_viewer_v2.py), copied
  unchanged below (baseline / substrate corrections, polarization, crop,
  maps, derivatives, peak and dip fits, 3-D cube).
• Pump 2 (flipper): pick the pump-2 state to show — files tagged
  _P2off / _P2on (dI/dP mode), or "pump2: off/on" in the .asc header.
• TRPL tab (SPAD23 or SPAD93: 23 or 93 pixel columns, picked per file),
  processing as in the SPAD23 / SPAD93 v2 GUIs:
    – time-zero alignment on each pixel's first peak; each point is then
      moved as a whole to the sweep's time zero;
    – "Point": pixel map, decay traces, sequential 2D Gaussian fit of every
      frame from the reference → sigma²(t), and D from
      sigma² = sigma0² + 2 D (t − t_ref);
    – "Sweep": decay map over the current sweep line and D / sigma0² versus
      power or gate ("Fit sweep", CSV export).
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

try:
    from scipy.signal import savgol_filter as _savgol
except ImportError:
    _savgol = None

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None

DEFAULT_CROP  = (50, 50, 200, 200)
EV_NM         = 1239.841984
GRID_DECIMALS = 9
_DIP_COLORS   = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════

class PLFrame:
    __slots__ = ("path", "power_w", "v_front", "v_back", "wavelength_nm", "image", "meta",
                 "mtime", "angle_deg", "vf_named", "vb_named")

    def __init__(self, path, power_w, v_front, v_back, wavelength_nm, image, meta, mtime,
                 angle_deg=None, vf_named=False, vb_named=False):
        self.path        = path
        self.power_w     = power_w
        self.v_front     = v_front
        self.v_back      = v_back
        self.wavelength_nm = wavelength_nm
        self.image       = image
        self.meta        = meta
        self.mtime       = mtime
        self.angle_deg   = angle_deg
        # Whether the front/back gate value came from the file NAME (the swept
        # target) rather than metadata (a noisy read-back of a fixed gate).
        self.vf_named    = vf_named
        self.vb_named    = vb_named


class _Channel:
    """All state for one measurement channel (data frames + optional substrate)."""

    def __init__(self):
        self.data_dir: Optional[Path]                            = None
        self.frames:   List[PLFrame]                             = []
        self.grid:     Dict[Tuple[float, float], PLFrame]        = {}
        self.powers:   np.ndarray                                = np.array([], dtype=float)
        self.gates:    np.ndarray                                = np.array([], dtype=float)   # unique front-gate values
        self.backs:    np.ndarray                                = np.array([], dtype=float)   # unique back-gate values
        self.backs_for_gate: np.ndarray                          = np.array([], dtype=float)
        # Full 3-D grid keyed by (power, v_front, v_back) — needed for grid-mode data
        self.grid3:    Dict[Tuple[float, float, float], PLFrame] = {}
        self.is_grid:  bool                                      = False
        self.default_power_key: float                            = 0.0
        self.default_gate_key:  float                            = 0.0
        self.default_back_key:  float                            = 0.0
        self.can_power_mode: bool                                = False
        self.can_gate_mode:  bool                                = False
        self.blank_image: Optional[np.ndarray]                   = None
        self.blank_wl:    Optional[np.ndarray]                   = None
        self.sub_single: Optional[PLFrame]                       = None
        self.sub_grid:   Dict[Tuple[float, float], PLFrame]      = {}
        self._img_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
        self.angle_mode: bool                                    = False
        self._angle_calib_angles: Optional[np.ndarray]          = None
        self._angle_calib_powers_w: Optional[np.ndarray]        = None

    @property
    def has_data(self) -> bool:
        return bool(self.frames)

    @property
    def has_substrate(self) -> bool:
        return self.sub_single is not None or bool(self.sub_grid)

    def clear_cache(self) -> None:
        self._img_cache.clear()


# ══════════════════════════════════════════════════════════════════════
# File parsing utilities
# ══════════════════════════════════════════════════════════════════════

def _parse_header(path: Path) -> dict:
    meta = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if not line:
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
            else:
                meta.setdefault("_header", []).append(line)
    return meta


def _load_andor_ascii(path: Path) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    meta = _parse_header(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    wl   = None
    cols = str(meta.get("columns", "")).lower()
    if data.shape[1] > 1 and "wavelength" in cols:
        wl    = data[:, 0]
        image = data[:, 1:].T
    else:
        image = data
    return meta, wl, image


def _parse_float(meta: dict, key: str) -> Optional[float]:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _parse_int(meta: dict, key: str, default: int) -> int:
    val = meta.get(key)
    if val is None:
        return int(default)
    try:
        return int(float(val))
    except Exception:
        return int(default)


def _parse_int_opt(meta: dict, key: str) -> Optional[int]:
    val = meta.get(key)
    if val is None:
        return None
    try:
        return int(float(val))
    except Exception:
        return None


_POWER_RE = re.compile(
    r"(?:^|_)P(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)"
    r"\s*(?P<unit>nW|uW|mW|W)(?=_|\.|$)"
)
_VF_RE = re.compile(r"(?:^|_)Vf(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")
_VB_RE    = re.compile(r"(?:^|_)Vb(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")
_ANGLE_RE = re.compile(r"(?:^|_)A(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)deg(?=_|\.|$)")


def _tok(token: str) -> Optional[float]:
    if not token:
        return None
    sign = 1.0
    if token.startswith("-"):
        sign, token = -1.0, token[1:]
    elif token.startswith("+"):
        token = token[1:]
    if token.startswith("m"):
        sign, token = sign * -1.0, token[1:]
    token = re.sub(r"em(\d+)", r"e-\1", token)
    try:
        val = sign * float(token.replace("p", "."))
    except Exception:
        return None
    return 0.0 if abs(val) < 1e-9 else val


def _parse_from_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    power_w = v_front = v_back = None
    pm = _POWER_RE.search(name)
    if pm:
        val   = _tok(pm.group("val"))
        scale = {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9}.get(pm.group("unit").lower())
        if val is not None and scale is not None:
            power_w = float(val) * float(scale)
    m = _VF_RE.search(name)
    if m:
        v_front = _tok(m.group("val"))
    m = _VB_RE.search(name)
    if m:
        v_back = _tok(m.group("val"))
    return power_w, v_front, v_back


def _parse_angle_from_name(name: str) -> Optional[float]:
    m = _ANGLE_RE.search(name)
    return _tok(m.group("val")) if m else None


# ── Angle → power calibration files ───────────────────────────────────

# Unit token in a power column header: "power (nW)", "power/uw", "power_mW", …
_CALIB_UNIT_RE = re.compile(r"\b(?P<unit>[munµ]?)w\b")
_CALIB_SCALE   = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9}


def _calib_power_scale(header: str) -> float:
    """Watts per unit implied by a power column header (default µW)."""
    h = re.sub(r"[^a-zµ]+", " ", str(header).lower())
    m = _CALIB_UNIT_RE.search(h)
    return _CALIB_SCALE.get(m.group("unit"), 1e-6) if m else 1e-6


def _read_calib_file(path: Path):
    """Read a calibration sheet as a raw, header-less grid of cells."""
    import pandas as pd
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, header=None, dtype=object)
        except ImportError as exc:
            # pandas needs openpyxl (.xlsx) / xlrd (.xls) as a separate install;
            # name the interpreter, since the usual cause is running the viewer
            # with the wrong venv rather than a broken environment.
            pkg = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
            raise ImportError(
                f"Reading .xlsx/.xls needs the '{pkg}' package, which is missing "
                f"from the Python running this viewer:\n\n    {sys.executable}\n\n"
                f"Install it with:\n\n    \"{sys.executable}\" -m pip install {pkg}\n\n"
                "Or launch the viewer through run_pl_power_gate_dep_viewer_v2.ps1, "
                "which uses the Automation .venv where it is already installed."
            ) from exc
    return pd.read_csv(path, header=None, dtype=object, engine="python")


def _parse_angle_power_calib(path: Path):
    """Parse an angle→power calibration file.

    The whole sheet is scanned instead of assuming the table starts at cell A1,
    so a title row above the header ("nd wheel | 690") is handled, and so are
    point pairs pasted into unrelated columns.

    Returns (angles, powers_w, extra_angles, extra_powers_w); the extra_* arrays
    hold numeric pairs found OUTSIDE the labelled angle/power columns, which the
    caller decides whether to trust.
    """
    import pandas as pd

    raw = _read_calib_file(path)
    if raw.empty:
        raise ValueError("The file is empty.")
    nrow, ncol = raw.shape

    # 1. Locate every "angle"/"power" header pair (there may be several blocks).
    blocks = []                      # (angle_col, power_col, header_row, scale)
    for r in range(nrow):
        a_c = p_c = None
        p_hdr = ""
        for c in range(ncol):
            v = raw.iat[r, c]
            if not isinstance(v, str):
                continue
            s = v.strip().lower()
            if a_c is None and "angle" in s:
                a_c = c
            elif p_c is None and "power" in s:
                p_c, p_hdr = c, s
        if a_c is not None and p_c is not None:
            blocks.append((a_c, p_c, r, _calib_power_scale(p_hdr)))
    if not blocks:
        raise ValueError(
            "Could not find 'angle' and 'power' column headers anywhere in "
            f"{path.name}.\n\nExpected a header row such as:  angle | power (nW)")

    # 2. Numeric rows under each labelled block.
    angles, powers_w, used = [], [], set()
    for a_c, p_c, hdr_row, scale in blocks:
        used |= {a_c, p_c}
        a = pd.to_numeric(raw[a_c], errors="coerce")
        p = pd.to_numeric(raw[p_c], errors="coerce")
        keep = a.notna() & p.notna()
        keep.iloc[:hdr_row + 1] = False
        angles.extend(a[keep].astype(float).tolist())
        powers_w.extend((p[keep].astype(float) * scale).tolist())
    if not angles:
        raise ValueError(
            f"Found 'angle'/'power' headers in {path.name} but no numeric rows "
            "underneath them.")

    # 3. Orphan pairs: numerics in adjacent columns that nobody labelled. This is
    #    how points pasted into e.g. columns G/H get noticed instead of dropped.
    scale0 = blocks[0][3]
    extra_a, extra_p = [], []
    free = [c for c in range(ncol) if c not in used]
    i = 0
    while i < len(free) - 1:
        c1, c2 = free[i], free[i + 1]
        if c2 != c1 + 1:
            i += 1
            continue
        x = pd.to_numeric(raw[c1], errors="coerce")
        y = pd.to_numeric(raw[c2], errors="coerce")
        keep = x.notna() & y.notna()
        if keep.any():
            extra_a.extend(x[keep].astype(float).tolist())
            extra_p.extend((y[keep].astype(float) * scale0).tolist())
            i += 2                   # both columns consumed
        else:
            i += 1

    return (np.asarray(angles, dtype=float), np.asarray(powers_w, dtype=float),
            np.asarray(extra_a, dtype=float), np.asarray(extra_p, dtype=float))


def _finalise_calib(angles: np.ndarray, powers_w: np.ndarray):
    """Sort by angle and collapse duplicates — np.interp needs increasing x."""
    order = np.argsort(angles, kind="stable")
    a, p  = angles[order], powers_w[order]
    uniq, idx = np.unique(a, return_index=True)
    if uniq.size != a.size:
        a, p = uniq, p[idx]
    return a, p


def _norm_key(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return None if not np.isfinite(x) else round(x, GRID_DECIMALS)


# ══════════════════════════════════════════════════════════════════════
# Image / linecut helpers
# ══════════════════════════════════════════════════════════════════════

def _apply_crop(image: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim != 2:
        return a
    h, w   = a.shape
    t, b, l, r = crop
    t = max(0, min(int(t), h - 1))
    b = max(0, min(int(b), h - 1))
    l = max(0, min(int(l), w - 1))
    r = max(0, min(int(r), w - 1))
    return a[t: max(t + 1, h - b), l: max(l + 1, w - r)]


def _crop_axis(axis: Optional[np.ndarray], crop: Tuple[int, int, int, int], width: int) -> Optional[np.ndarray]:
    if axis is None:
        return None
    arr = np.asarray(axis, dtype=float).ravel()
    if arr.size != width:
        return arr
    l, r = crop[2], crop[3]
    if r > 0:
        arr = arr[: max(0, arr.size - r)]
    if l > 0:
        arr = arr[min(l, arr.size):]
    return arr


def _linecut_h(image: np.ndarray, row: int, width: int) -> Optional[np.ndarray]:
    a = np.asarray(image)
    if a.ndim != 2:
        return None
    h, _ = a.shape
    width = max(1, int(width))
    half  = width // 2
    r1 = max(0, int(row) - half)
    r2 = min(h, int(row) + half + (1 if width % 2 else 0))
    return None if r2 <= r1 else a[r1:r2, :].sum(axis=0)


# Points whose substrate signal is below this fraction of the largest |substrate|
# are set to NaN in ΔR/R0, so near-zero denominators do not blow up the colour scale.
RC_MIN_FRAC = 0.01


def _rc_ratio(num: np.ndarray, den: np.ndarray, min_frac: float = RC_MIN_FRAC) -> np.ndarray:
    """Element-wise num / den (num = R − R0, den = R0); NaN where |den| is tiny."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    a = np.abs(den)
    if not np.isfinite(a).any():
        return out
    thr = max(float(min_frac) * float(np.nanmax(a)), 0.0)
    ok = np.isfinite(den) & (a > thr)
    np.divide(num, den, out=out, where=ok)
    return out


def _lc_from_terms(terms, row_raw: int, width: int,
                   wl: Optional[np.ndarray] = None, flatten=None) -> Optional[np.ndarray]:
    """Linecut of Σ sign · (Σrows num) / (Σrows den); den None → plain row sum.
    ``flatten`` = (feature_ranges_nm, order): each ratio term is then divided by a
    smooth fit of its feature-free part (see ``_flatten_rc``)."""
    total = None
    for sign, num, den in terms:
        lc = _linecut_h(num, row_raw, width)
        if lc is None:
            return None
        if den is not None:
            lc_d = _linecut_h(den, row_raw, width)
            if lc_d is None:
                return None
            lc = _rc_ratio(lc, lc_d)
            if flatten is not None and wl is not None:
                lc = _flatten_rc(wl, lc, *flatten)[0]
        lc = float(sign) * np.asarray(lc, dtype=float)
        total = lc if total is None else total + lc
    return total


def _parse_ranges_nm(text: str) -> List[Tuple[float, float]]:
    """'690-710, 820-870' → [(690, 710), (820, 870)] in nm.
    A range whose ends are both < 10 is taken as eV (e.g. '1.95-2.0')."""
    out: List[Tuple[float, float]] = []
    for tok in re.split(r"[,;\n]+", str(text)):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(?:-|–|to|:)\s*([0-9]*\.?[0-9]+)\s*", tok)
        if not m:
            continue
        a, b = float(m.group(1)), float(m.group(2))
        if a < 10 and b < 10:
            if a <= 0 or b <= 0:
                continue
            a, b = EV_NM / a, EV_NM / b
        out.append((min(a, b), max(a, b)))
    return out


def _flatten_rc(wl: np.ndarray, rc: np.ndarray,
                ranges_nm: List[Tuple[float, float]], order: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a smooth multiplicative error from RC = R/R0 − 1.

    R/R0 measured = (1 + RC_true)·g(λ) with g smooth (focus / spot mismatch).
    g is a polynomial of ``order`` fitted to 1 + RC outside ``ranges_nm``
    (the feature regions); returns (RC_flat = (1 + RC)/g − 1, g − 1).
    Too few feature-free points → all NaN."""
    x = np.asarray(wl, dtype=float).ravel()
    y = 1.0 + np.asarray(rc, dtype=float).ravel()
    nan = np.full(y.shape, np.nan)
    if x.size != y.size:
        return nan, nan.copy()
    order = max(0, int(order))
    keep = np.isfinite(x) & np.isfinite(y)
    for lo, hi in ranges_nm:
        keep &= ~((x >= lo) & (x <= hi))
    if int(keep.sum()) < order + 2:
        return nan, nan.copy()
    xs = x[keep]
    c = 0.5 * (xs.min() + xs.max())
    s = 0.5 * (xs.max() - xs.min()) or 1.0
    P = np.polynomial.polynomial
    g = P.polyval((x - c) / s, P.polyfit((xs - c) / s, y[keep], order))
    flat = nan.copy()
    ok = np.isfinite(g) & (g > 0)
    np.divide(y, g, out=flat, where=ok)
    return flat - 1.0, g - 1.0


def _flatten_rc_rows(wl: np.ndarray, rc_img: np.ndarray,
                     ranges_nm: List[Tuple[float, float]], order: int) -> np.ndarray:
    """``_flatten_rc`` applied to every row of a (rows, λ) ΔR/R0 image."""
    out = np.full(rc_img.shape, np.nan)
    for i in range(rc_img.shape[0]):
        if np.isfinite(rc_img[i]).any():
            out[i] = _flatten_rc(wl, rc_img[i], ranges_nm, order)[0]
    return out


def _resample(wl_src: Optional[np.ndarray], lc: np.ndarray, wl_dst: Optional[np.ndarray]) -> np.ndarray:
    if wl_src is None or wl_dst is None:
        return lc
    src = np.asarray(wl_src, dtype=float).ravel()
    dst = np.asarray(wl_dst, dtype=float).ravel()
    if src.size == dst.size and np.allclose(src, dst):
        return lc
    order = np.argsort(src)
    return np.interp(dst, src[order], np.asarray(lc, dtype=float)[order], left=np.nan, right=np.nan)


def _deriv_vs_x(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    x   = np.asarray(x, dtype=float).ravel()
    y   = np.asarray(data, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    n   = x.size
    if y.ndim != 2 or n != y.shape[1] or n < 2:
        return out
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        prev = i - 1
        while prev >= 0 and (not np.isfinite(x[prev]) or x[prev] == x[i]):
            prev -= 1
        nxt = i + 1
        while nxt < n and (not np.isfinite(x[nxt]) or x[nxt] == x[i]):
            nxt += 1
        if prev >= 0 and nxt < n:
            d = x[nxt] - x[prev]
            if d:
                out[:, i] = (y[:, nxt] - y[:, prev]) / d
        elif prev >= 0:
            d = x[i] - x[prev]
            if d:
                out[:, i] = (y[:, i] - y[:, prev]) / d
        elif nxt < n:
            d = x[nxt] - x[i]
            if d:
                out[:, i] = (y[:, nxt] - y[:, i]) / d
    return out


def _deriv_dlogp(data: np.ndarray, powers_w: np.ndarray) -> np.ndarray:
    p    = np.asarray(powers_w, dtype=float).ravel()
    logp = np.full_like(p, np.nan)
    mask = p > 0
    logp[mask] = np.log10(p[mask])
    return _deriv_vs_x(data, logp)


def _deriv_vs_ev(m_ev: np.ndarray, y_ev: np.ndarray) -> np.ndarray:
    """Derivative along the energy (y) axis for a uniformly-spaced energy grid.

    m_ev : (n_ev, n_x)  — data on a uniform energy grid
    y_ev : (n_ev,)      — uniform energy axis (eV)
    Returns dI/dE with the same shape as m_ev.
    """
    y = np.asarray(y_ev, dtype=float).ravel()
    m = np.asarray(m_ev, dtype=float)
    if m.ndim != 2 or y.size != m.shape[0] or y.size < 3:
        return np.full_like(m, np.nan, dtype=float)
    dy = float(y[1] - y[0])
    if dy == 0.0:
        return np.full_like(m, np.nan, dtype=float)
    return np.gradient(m, dy, axis=0)


def _nm_to_ev(wl_nm: np.ndarray, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.asarray(wl_nm, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        ev = EV_NM / wl
    ok     = np.isfinite(ev)
    ev_ok  = ev[ok]
    arr_ok = arr[ok, :]
    if ev_ok.size < 2:
        return ev_ok, arr_ok
    order  = np.argsort(ev_ok)
    ev_s   = ev_ok[order]
    arr_s  = arr_ok[order, :]
    # Resample onto a uniform energy grid so ImageItem rect gives correct axis labels.
    # (eV = hc/λ is nonlinear in λ; without resampling, pixel positions would not
    #  match the linearly-scaled axis.)
    ev_u  = np.linspace(ev_s[0], ev_s[-1], ev_s.size)
    n_x   = arr_s.shape[1]
    arr_u = np.empty((ev_s.size, n_x), dtype=float)
    for j in range(n_x):
        col  = arr_s[:, j]
        mask = np.isfinite(col)
        if mask.sum() >= 2:
            arr_u[:, j] = np.interp(ev_u, ev_s[mask], col[mask])
        else:
            arr_u[:, j] = np.nan
    return ev_u, arr_u


def fwhm_nm_to_mev(center_nm: np.ndarray, fwhm_nm: np.ndarray) -> np.ndarray:
    c = np.asarray(center_nm, dtype=float)
    w = np.asarray(fwhm_nm,   dtype=float)
    out = np.full(np.broadcast(c, w).shape, np.nan, dtype=float)
    c, w = np.broadcast_arrays(c, w)
    half  = 0.5 * w
    valid = np.isfinite(c) & np.isfinite(w) & (w > 0) & (c > half) & (c + half > 0)
    if not np.any(valid):
        return out
    out[valid] = np.abs(EV_NM / (c[valid] - half[valid]) - EV_NM / (c[valid] + half[valid])) * 1e3
    return out


# ══════════════════════════════════════════════════════════════════════
# Peak-fitting utilities  (ported from v1)
# ══════════════════════════════════════════════════════════════════════

def _lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        return np.zeros_like(x)
    g  = float(gamma)
    dx = x - float(center)
    return float(amplitude) * g * g / (dx * dx + g * g)


def _baseline_from_tail(wl: np.ndarray, y: np.ndarray, lo: float = 940.0, hi: float = 960.0) -> float:
    x = np.asarray(wl, dtype=float).ravel()
    y = np.asarray(y,  dtype=float).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if mask.sum() >= 3:
        return float(np.nanmean(y[mask]))
    finite = y[np.isfinite(y)]
    return 0.0 if finite.size == 0 else float(np.nanpercentile(finite, 10.0))


def _noise_floor(y: np.ndarray) -> float:
    v = np.asarray(y, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 0.0
    diffs = np.diff(v)
    mad   = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad / np.sqrt(2.0))
    std = np.nanstd(v)
    return 0.0 if not np.isfinite(std) or std <= 0 else float(std)


def _peak_guess(x, y, lo, hi, fallback):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        return 0.0, float(fallback), 3.0
    xs = x[mask]; ys = y[mask]
    idx    = int(np.nanargmax(ys))
    center = float(xs[idx])
    bl     = float(np.nanpercentile(ys, 15.0))
    amp    = float(max(0.0, ys[idx] - bl))
    wt     = np.clip(ys - bl, 0.0, None)
    ws     = float(np.nansum(wt))
    sigma  = float(np.sqrt(np.nansum(wt * (xs - center) ** 2) / ws)) if ws > 0 else 3.0
    return amp, center, float(np.clip(0.5 * sigma, 0.5, 10.0))


def fit_lorentzian_peak_local(wl, intensity, center_guess, *, window_nm,
                               min_points=7, min_snr=2.0, min_gamma=0.08, max_gamma=10.0):
    x = np.asarray(wl,        dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < min_points:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x - center_guess) <= window_nm)
    if mask.sum() < min_points:
        return None
    xs = x[mask]; ys = y[mask]
    bl  = float(np.nanpercentile(ys, 10.0))
    yb  = ys - bl
    amp = float(np.nanmax(yb))
    if not np.isfinite(amp) or amp <= 0:
        return None
    if _noise_floor(ys) > 0 and amp < min_snr * _noise_floor(ys):
        return None
    idx     = int(np.nanargmax(yb))
    c_fit   = float(xs[idx])
    y_half  = 0.5 * amp
    left_   = np.where(yb[:idx] <= y_half)[0]
    right_  = np.where(yb[idx + 1:] <= y_half)[0]
    if left_.size > 0 and right_.size > 0:
        fwhm_g = max(2.0 * min_gamma, float(xs[idx + 1 + right_[0]]) - float(xs[left_[-1]]))
    else:
        fwhm_g = max(2.0 * min_gamma, 0.25 * window_nm)
    g_fit = float(np.clip(0.5 * fwhm_g, min_gamma, max_gamma))
    a_fit = amp
    if _curve_fit is not None:
        try:
            lo  = np.array([0.0,  center_guess - window_nm, min_gamma])
            hi  = np.array([max(1.0, 4.0 * amp), center_guess + window_nm, max_gamma])
            p0  = np.clip([a_fit, c_fit, g_fit], lo + 1e-9, hi - 1e-9)
            popt, _ = _curve_fit(lambda xx, a, m, g: _lorentzian(xx, a, m, g),
                                 xs, yb, p0=p0, bounds=(lo, hi), maxfev=8000)
            a_fit, c_fit, g_fit = [float(v) for v in popt]
        except Exception:
            pass
    if not (min_gamma <= g_fit <= max_gamma):
        return None
    if abs(c_fit - center_guess) > window_nm * 1.2:
        return None
    model = bl + _lorentzian(xs, a_fit, c_fit, g_fit)
    sst   = float(np.nansum((ys - np.nanmean(ys)) ** 2))
    r2    = 1.0 - float(np.nansum((ys - model) ** 2)) / sst if sst > 0 else 1.0
    if not np.isfinite(r2) or r2 < -0.3:
        return None
    return {"center_nm": c_fit, "gamma_nm": g_fit, "fwhm_nm": 2.0 * g_fit, "amplitude": max(0.0, a_fit)}


def fit_two_peak_linecut(wl, intensity, *, peak1_guess_nm, peak2_guess_nm,
                          peak1_window_nm=10.0, peak2_window_nm=5.0) -> dict:
    x_all = np.asarray(wl,        dtype=float).ravel()
    y_all = np.asarray(intensity, dtype=float).ravel()
    empty = {"peak_1": None, "peak_2": None, "offset": 0.0}
    if x_all.size != y_all.size or x_all.size < 8:
        return empty
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    if valid.sum() < 8:
        return empty
    x_all = x_all[valid]; y_all = y_all[valid]
    order = np.argsort(x_all)
    x_all = x_all[order]; y_all = y_all[order]

    bl    = _baseline_from_tail(x_all, y_all)
    y_sub = y_all - bl

    w1 = max(0.2, float(peak1_window_nm)); w2 = max(0.2, float(peak2_window_nm))
    p1_lo = peak1_guess_nm - w1; p1_hi = peak1_guess_nm + w1
    p2_lo = peak2_guess_nm - w2; p2_hi = peak2_guess_nm + w2
    fit_lo = min(p2_lo, p1_lo) - 6.0; fit_hi = max(p2_hi, p1_hi) + 30.0

    mask = (x_all >= fit_lo) & (x_all <= fit_hi)
    x = x_all[mask] if mask.sum() >= 20 else x_all
    y = y_sub[mask] if mask.sum() >= 20 else y_sub
    if x.size < 8:
        return {**empty, "offset": float(bl)}

    y_span = float(max(1.0, np.nanmax(y) - np.nanmin(y)))
    a1g, m1g, g1g = _peak_guess(x, y, p1_lo, p1_hi, peak1_guess_nm)
    a2g, m2g, g2g = _peak_guess(x, y, p2_lo, p2_hi, peak2_guess_nm)
    a1g = max(0.08 * y_span, a1g); a2g = max(0.02 * y_span, a2g)

    popt2 = popt1 = None
    if _curve_fit is not None:
        lo2 = [0.0, p1_lo, 0.4, 0.0, p2_lo, 0.4]
        hi2 = [4*y_span, p1_hi, 15.0, 4*y_span, p2_hi, 12.0]
        lo1 = [0.0, p1_lo, 0.4]; hi1 = [4*y_span, p1_hi, 15.0]
        try:
            p0  = np.clip([a1g, m1g, g1g, a2g, m2g, g2g], np.array(lo2)+1e-6, np.array(hi2)-1e-6)
            popt2, _ = _curve_fit(
                lambda xx, a1, m1, g1, a2, m2, g2: _lorentzian(xx,a1,m1,g1)+_lorentzian(xx,a2,m2,g2),
                x, y, p0=p0, bounds=(lo2, hi2), maxfev=20000)
        except Exception:
            popt2 = None
        try:
            p0  = np.clip([a1g, m1g, g1g], np.array(lo1)+1e-6, np.array(hi1)-1e-6)
            popt1, _ = _curve_fit(lambda xx, a, m, g: _lorentzian(xx,a,m,g),
                                   x, y, p0=p0, bounds=(lo1, hi1), maxfev=12000)
        except Exception:
            popt1 = None

    if popt2 is None and popt1 is None:
        p1 = fit_lorentzian_peak_local(x_all, y_sub, peak1_guess_nm, window_nm=10.0, min_snr=1.2)
        resid = np.array(y_sub, dtype=float)
        if p1 is not None:
            resid -= _lorentzian(x_all, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
        p2 = fit_lorentzian_peak_local(x_all, resid, peak2_guess_nm, window_nm=5.0, min_snr=1.5)
        if p1 is not None and not (p1_lo <= p1["center_nm"] <= p1_hi): p1 = None
        if p2 is not None and not (p2_lo <= p2["center_nm"] <= p2_hi): p2 = None
        return {"peak_1": p1, "peak_2": p2, "offset": float(bl)}

    # choose between 1-peak and 2-peak solutions
    use2 = False
    if popt2 is not None:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]
        y2 = _lorentzian(x,a1,m1,g1)+_lorentzian(x,a2,m2,g2)
        y1 = _lorentzian(x, *(float(v) for v in popt1)) if popt1 is not None else _lorentzian(x,a1,m1,g1)
        noise = _noise_floor(y - y2)
        rss1  = float(np.nansum((y - y1)**2))
        rss2  = float(np.nansum((y - y2)**2))
        imp   = (rss1 - rss2) / rss1 if rss1 > 0 else 0.0
        min_a2 = max(2.0*noise, 0.04*y_span, 0.06*max(a1, 1.0))
        use2  = (a2 >= min_a2 and m2 < m1 - 1.0 and p2_lo <= m2 <= p2_hi
                 and 0.4 <= g2 <= 12.0 and imp > 0.03)

    if use2:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]
    elif popt1 is not None:
        a1, m1, g1 = [float(v) for v in popt1]; a2, m2, g2 = 0.0, peak2_guess_nm, 3.0
    else:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]; a2 = 0.0

    p1 = ({"center_nm": m1, "gamma_nm": g1, "fwhm_nm": 2*g1, "amplitude": a1}
          if a1 > 0 and 0.4 <= g1 <= 20.0 and p1_lo <= m1 <= p1_hi else None)
    p2 = ({"center_nm": m2, "gamma_nm": g2, "fwhm_nm": 2*g2, "amplitude": a2}
          if use2 and a2 > 0 and 0.4 <= g2 <= 20.0 and p2_lo <= m2 <= p2_hi else None)
    return {"peak_1": p1, "peak_2": p2, "offset": float(bl)}


def fit_two_peak_map_sequential(wl_nm, map_data, *, ref_idx, peak1_guess_nm, peak2_guess_nm,
                                  max_shift_nm=2.0) -> dict:
    wl   = np.asarray(wl_nm,   dtype=float).ravel()
    data = np.asarray(map_data, dtype=float)
    n_x  = data.shape[1] if data.ndim == 2 else 0
    c1   = np.full(n_x, np.nan); fw1 = np.full(n_x, np.nan)
    gm1  = np.full(n_x, np.nan); a1  = np.full(n_x, np.nan)
    c2   = np.full(n_x, np.nan); fw2 = np.full(n_x, np.nan)
    gm2  = np.full(n_x, np.nan); a2  = np.full(n_x, np.nan)
    off  = np.full(n_x, np.nan)

    def _result():
        return {"center_1_nm": c1, "fwhm_1_nm": fw1, "gamma_1_nm": gm1, "amp_1": a1,
                "center_2_nm": c2, "fwhm_2_nm": fw2, "gamma_2_nm": gm2, "amp_2": a2, "offset": off}

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0:
        return _result()

    ref  = max(0, min(int(ref_idx), n_x - 1))
    step = max(0.2, float(max_shift_nm))

    def _store(j, fit):
        p1 = fit.get("peak_1"); p2 = fit.get("peak_2")
        off[j] = float(fit.get("offset", np.nan))
        if p1:
            c1[j] = p1.get("center_nm", np.nan); fw1[j] = p1.get("fwhm_nm", np.nan)
            gm1[j] = p1.get("gamma_nm", np.nan); a1[j]  = p1.get("amplitude", np.nan)
        if p2:
            c2[j] = p2.get("center_nm", np.nan); fw2[j] = p2.get("fwhm_nm", np.nan)
            gm2[j] = p2.get("gamma_nm", np.nan); a2[j]  = p2.get("amplitude", np.nan)

    fit_ref = fit_two_peak_linecut(wl, data[:, ref],
                                    peak1_guess_nm=peak1_guess_nm, peak2_guess_nm=peak2_guess_nm,
                                    peak1_window_nm=10.0, peak2_window_nm=5.0)
    _store(ref, fit_ref)
    g1r = c1[ref] if np.isfinite(c1[ref]) else float(peak1_guess_nm)
    g2r = c2[ref] if np.isfinite(c2[ref]) else float(peak2_guess_nm)

    def _walk(start, stop, inc, g1i, g2i):
        g1 = float(g1i); g2 = float(g2i)
        for j in range(start, stop, inc):
            fj = fit_two_peak_linecut(wl, data[:, j],
                                       peak1_guess_nm=g1, peak2_guess_nm=g2,
                                       peak1_window_nm=step, peak2_window_nm=step)
            _store(j, fj)
            if np.isfinite(c1[j]): g1 = float(c1[j])
            if np.isfinite(c2[j]): g2 = float(c2[j])

    _walk(ref + 1, n_x, +1, g1r, g2r)
    _walk(ref - 1, -1,  -1, g1r, g2r)
    return _result()


# ══════════════════════════════════════════════════════════════════════
# Reflectance dip fitting  (N simultaneous negative Lorentzians)
# ══════════════════════════════════════════════════════════════════════

def find_dip_local_min(wl, signal, dip_guesses_nm, *,
                       window_nm: float = 8.0,
                       max_shift_nm: float = 3.0,
                       smooth_sigma_nm: float = 1.5) -> dict:
    """Find local minima in a Gaussian-smoothed spectrum near each guess.

    Returns {"dips": [{"center_nm", "value_at_min"} | None, ...],
             "x_smooth": ndarray, "y_smooth": ndarray}
    None is returned for a dip when no local minimum exists within max_shift_nm.
    """
    x       = np.asarray(wl,     dtype=float).ravel()
    y       = np.asarray(signal, dtype=float).ravel()
    guesses = np.asarray(dip_guesses_nm, dtype=float).ravel()
    n_dips  = guesses.size
    empty   = {"dips": [None] * n_dips, "x_smooth": x, "y_smooth": y}

    if x.size != y.size or x.size < 5 or n_dips == 0:
        return empty
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return empty
    x = x[valid]; y = y[valid]
    order = np.argsort(x); x = x[order]; y = y[order]

    dx       = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    sigma_px = max(0.5, smooth_sigma_nm / max(dx, 1e-12))
    try:
        from scipy.ndimage import gaussian_filter1d
        y_sm = gaussian_filter1d(y, sigma=sigma_px)
    except Exception:
        y_sm = y.copy()

    results = []
    for g in guesses:
        g  = float(g)
        lo = g - float(max_shift_nm)
        hi = g + float(max_shift_nm)
        mask = (x >= lo) & (x <= hi)
        n_in = int(mask.sum())
        if n_in < 3:
            results.append(None)
            continue
        xs = x[mask]; ys = y_sm[mask]; yo = y[mask]
        mi = int(np.argmin(ys))
        # Reject boundary minima — likely the slope of a neighbouring feature
        if mi == 0 or mi == n_in - 1:
            results.append(None)
            continue
        results.append({"center_nm": float(xs[mi]),
                        "value_at_min": float(yo[mi])})
    return {"dips": results, "x_smooth": x, "y_smooth": y_sm}


def find_dip_map_sequential(wl_nm, map_data, *, ref_idx, dip_guesses_nm,
                            window_nm: float = 8.0,
                            max_shift_nm: float = 2.0,
                            smooth_sigma_nm: float = 1.5) -> dict:
    """Find reflectance dip positions sequentially across the sweep axis.

    Returns {"centers_nm": (n_dips, n_x), "amps": (n_dips, n_x)}
    where amps = signal value at the local minimum.  NaN where no dip found.
    """
    wl      = np.asarray(wl_nm,          dtype=float).ravel()
    data    = np.asarray(map_data,        dtype=float)
    guesses = np.asarray(dip_guesses_nm,  dtype=float).ravel()
    n_dips  = guesses.size
    n_x     = data.shape[1] if data.ndim == 2 else 0

    centers = np.full((n_dips, n_x), np.nan)
    amps    = np.full((n_dips, n_x), np.nan)

    def _result():
        return {"centers_nm": centers, "amps": amps}

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0 or n_dips == 0:
        return _result()

    ref  = max(0, min(int(ref_idx), n_x - 1))
    step = max(0.2, float(max_shift_nm))

    def _store(j, fit):
        for i, r in enumerate(fit["dips"]):
            if r is not None:
                centers[i, j] = r["center_nm"]
                amps[i, j]    = r["value_at_min"]

    cur_g   = guesses.copy()
    fit_ref = find_dip_local_min(wl, data[:, ref], cur_g,
                                 window_nm=window_nm, max_shift_nm=step,
                                 smooth_sigma_nm=smooth_sigma_nm)
    _store(ref, fit_ref)
    for i, r in enumerate(fit_ref["dips"]):
        if r is not None and np.isfinite(r["center_nm"]):
            cur_g[i] = r["center_nm"]

    def _walk(start, stop, inc, init_g):
        g = init_g.copy()
        for j in range(start, stop, inc):
            f = find_dip_local_min(wl, data[:, j], g,
                                   window_nm=window_nm, max_shift_nm=step,
                                   smooth_sigma_nm=smooth_sigma_nm)
            _store(j, f)
            for i, r in enumerate(f["dips"]):
                if r is not None and np.isfinite(r["center_nm"]):
                    g[i] = r["center_nm"]

    _walk(ref + 1, n_x, +1, cur_g)
    _walk(ref - 1, -1,  -1, cur_g)
    return _result()


# ══════════════════════════════════════════════════════════════════════
# NEW: Baseline correction
# ══════════════════════════════════════════════════════════════════════

def _baseline_correct(image: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Subtract per-row mean of [lo, hi] nm from every spatial row.

    image : (n_spatial, n_wavelength)
    wl    : (n_wavelength,)
    """
    img = np.array(image, dtype=float)
    wl  = np.asarray(wl,  dtype=float).ravel()
    if img.ndim != 2 or wl.size != img.shape[1]:
        return img
    mask = (wl >= float(lo)) & (wl <= float(hi))
    if not mask.any():
        return img
    bg = np.nanmean(img[:, mask], axis=1, keepdims=True)
    return img - bg


def _corrected_image(
    data_img: np.ndarray, data_wl: Optional[np.ndarray],
    sub_img:  Optional[np.ndarray], sub_wl: Optional[np.ndarray],
    *,
    use_baseline: bool, lo: float, hi: float,
    use_sub: bool,
) -> np.ndarray:
    d = np.array(data_img, dtype=float)
    if use_baseline and data_wl is not None:
        d = _baseline_correct(d, data_wl, lo, hi)
    if use_sub and sub_img is not None:
        s = np.array(sub_img, dtype=float)
        if use_baseline and sub_wl is not None:
            s = _baseline_correct(s, sub_wl, lo, hi)
        if s.shape == d.shape:
            d = d - s
    return d


# ══════════════════════════════════════════════════════════════════════
# SG smoothing
# ══════════════════════════════════════════════════════════════════════

def _sg_smooth_2d(data, win_sweep, ord_sweep, win_wl, ord_wl):
    if _savgol is None:
        return np.array(data, dtype=float)
    out = np.array(data, dtype=float)
    win_sweep += (win_sweep % 2 == 0)
    win_wl    += (win_wl    % 2 == 0)

    def _rows(arr, win, order):
        if win < 3 or order < 1 or order >= win or arr.shape[1] < win:
            return arr
        res = arr.copy()
        for i in range(arr.shape[0]):
            row  = arr[i, :]
            mask = np.isfinite(row)
            if mask.sum() < win:
                continue
            if mask.all():
                res[i, :] = _savgol(row, win, order, mode="mirror")
            else:
                xi     = np.arange(row.size)
                filled = np.interp(xi, xi[mask], row[mask])
                res[i, :] = _savgol(filled, win, order, mode="mirror")
                res[i, ~mask] = np.nan
        return res

    def _cols(arr, win, order):
        if win < 3 or order < 1 or order >= win or arr.shape[0] < win:
            return arr
        res = arr.copy()
        for j in range(arr.shape[1]):
            col  = arr[:, j]
            mask = np.isfinite(col)
            if mask.sum() < win:
                continue
            if mask.all():
                res[:, j] = _savgol(col, win, order, mode="mirror")
            else:
                xi     = np.arange(col.size)
                filled = np.interp(xi, xi[mask], col[mask])
                res[:, j] = _savgol(filled, win, order, mode="mirror")
                res[~mask, j] = np.nan
        return res

    out = _rows(out, win_sweep, ord_sweep)
    out = _cols(out, win_wl,    ord_wl)
    return out


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def _load_frames(data_dir: Path, paths: Optional[List[Path]] = None) -> List[PLFrame]:
    """Every .asc in ``data_dir``, or only ``paths`` when given."""
    frames: List[PLFrame] = []
    for path in (sorted(data_dir.glob("*.asc")) if paths is None else paths):
        try:
            meta, wl, image = _load_andor_ascii(path)
        except Exception:
            continue
        p_m  = _parse_float(meta, "power_w")
        vf_m = _parse_float(meta, "front_v_set")
        vb_m = _parse_float(meta, "back_v_set")
        p_n, vf_n, vb_n = _parse_from_name(path.name)
        angle_n = _parse_angle_from_name(path.name)
        true_power = p_n if p_n is not None else p_m
        frames.append(PLFrame(
            path=path,
            power_w    = true_power if true_power is not None else angle_n,
            v_front    = vf_n if vf_n is not None else vf_m,
            v_back     = vb_n if vb_n is not None else vb_m,
            wavelength_nm = wl,
            image      = image,
            meta       = meta,
            mtime      = float(path.stat().st_mtime),
            angle_deg  = angle_n,
            vf_named   = vf_n is not None,   # swept front → clean target in filename
            vb_named   = vb_n is not None,   # swept back  → clean target in filename
        ))
    return frames


def _build_channel(ch: _Channel, frames: List[PLFrame]) -> None:
    """Populate grid / powers / gates from frames list."""
    ch.frames = frames

    # A gate is "swept" only if its value appears in the file name (a clean
    # target). A gate that only ever comes from metadata is a FIXED gate whose
    # read-back is noisy — collapse it to a single representative value so read
    # noise cannot fragment it into many spurious grid keys.
    front_named = any(getattr(fr, "vf_named", False) for fr in frames)
    back_named  = any(getattr(fr, "vb_named", False) for fr in frames)

    def _collapsed(vals) -> float:
        v = [float(x) for x in vals if x is not None and np.isfinite(x)]
        return 0.0 if not v else float(round(float(np.median(v)), 3))

    fixed_front = None if front_named else _norm_key(_collapsed([fr.v_front for fr in frames]))
    fixed_back  = None if back_named  else _norm_key(_collapsed([fr.v_back  for fr in frames]))

    def _eff_vf(fr):
        return _norm_key(fr.v_front) if fixed_front is None else fixed_front

    def _eff_vb(fr):
        return _norm_key(fr.v_back) if fixed_back is None else fixed_back

    power_keys: set = set()
    gate_keys:  set = set()
    back_keys:  set = set()
    back_by_gate: Dict = {}

    for fr in frames:
        p  = _norm_key(fr.power_w)
        vf = _eff_vf(fr)
        vb = _eff_vb(fr)
        if p  is not None: power_keys.add(p)
        if vb is not None: back_keys.add(vb)
        if vf is not None:
            gate_keys.add(vf)
            if vb is not None:
                back_by_gate.setdefault(vf, []).append(vb)

    if not power_keys: power_keys = {0.0}
    if not gate_keys:  gate_keys  = {0.0}
    if not back_keys:  back_keys  = {0.0}

    ch.powers = np.array(sorted(power_keys), dtype=float)
    ch.gates  = np.array(sorted(gate_keys),  dtype=float)
    ch.backs  = np.array(sorted(back_keys),  dtype=float)
    ch.default_power_key = float(_norm_key(ch.powers[0]) or 0.0)
    ch.default_gate_key  = float(_norm_key(ch.gates[0])  or 0.0)
    ch.default_back_key  = float(_norm_key(ch.backs[0])  or 0.0)

    # Grid mode: the back gate is not a function of the front gate, i.e. at least
    # one front-gate value was measured at more than one back-gate value (this
    # also covers a pure back sweep at a single fixed front value).
    ch.is_grid = any(len(set(vbs)) > 1 for vbs in back_by_gate.values())

    def _key_vf(fr):
        v = _eff_vf(fr)
        return ch.default_gate_key if v is None else float(v)

    def _key_vb(fr):
        v = _eff_vb(fr)
        return ch.default_back_key if v is None else float(v)

    # 2-D grid keyed by (power, front) — used for non-grid data and as a fallback
    ch.grid = {}
    for fr in frames:
        key = (_ch_norm_power(ch, fr.power_w), _key_vf(fr))
        old = ch.grid.get(key)
        if old is None or fr.mtime > old.mtime:
            ch.grid[key] = fr

    # 3-D grid keyed by (power, front, back) — used for grid-mode data
    ch.grid3 = {}
    for fr in frames:
        key = (_ch_norm_power(ch, fr.power_w), _key_vf(fr), _key_vb(fr))
        old = ch.grid3.get(key)
        if old is None or fr.mtime > old.mtime:
            ch.grid3[key] = fr

    ch.can_power_mode = ch.powers.size > 1
    ch.can_gate_mode  = ch.gates.size  > 1 or ch.backs.size > 1
    ch.angle_mode = any(fr.angle_deg is not None for fr in frames)

    backs = np.full(ch.gates.shape, np.nan, dtype=float)
    for i, vf in enumerate(ch.gates):
        vals = back_by_gate.get(_norm_key(vf), [])
        if vals:
            backs[i] = float(np.median(vals))
    ch.backs_for_gate = backs

    if frames:
        ref = frames[0]
        ch.blank_image = np.zeros_like(ref.image, dtype=float)
        wl = ref.wavelength_nm
        ch.blank_wl = (
            np.asarray(wl, dtype=float).copy()
            if wl is not None
            else np.arange(ref.image.shape[1], dtype=float)
        )
    else:
        ch.blank_image = np.zeros((10, 10), dtype=float)
        ch.blank_wl    = np.arange(10, dtype=float)


def _ch_norm_power(ch: _Channel, p: Optional[float]) -> float:
    k = _norm_key(p)
    return ch.default_power_key if k is None else float(k)


def _ch_norm_gate(ch: _Channel, vf: Optional[float]) -> float:
    k = _norm_key(vf)
    return ch.default_gate_key if k is None else float(k)


def _ch_norm_back(ch: _Channel, vb: Optional[float]) -> float:
    k = _norm_key(vb)
    return ch.default_back_key if k is None else float(k)


def _frame_at(ch: _Channel, p_key: float, vf_key: float, vb_key: float) -> Optional[PLFrame]:
    """Fetch the frame at (power, front, back).

    Grid data is indexed by all three; non-grid data is indexed by (power, front)
    only (the back gate follows the front gate)."""
    if ch.is_grid:
        fr = ch.grid3.get((p_key, vf_key, vb_key))
        if fr is not None:
            return fr
        # fall back to the nearest back gate measured at this (power, front)
        cands = [(k, v) for k, v in ch.grid3.items() if k[0] == p_key and k[1] == vf_key]
        if cands:
            cands.sort(key=lambda kv: abs(kv[0][2] - vb_key))
            return cands[0][1]
        return None
    return ch.grid.get((p_key, vf_key))


def _ch_substrate_for(ch: _Channel, fr: PLFrame) -> Optional[PLFrame]:
    if ch.sub_single is not None:
        return ch.sub_single
    if not ch.sub_grid:
        return None
    p  = float(_norm_key(fr.power_w)  or ch.default_power_key)
    vf = float(_norm_key(fr.v_front)  or ch.default_gate_key)
    key = (p, vf)
    if key in ch.sub_grid:
        return ch.sub_grid[key]
    same_gate  = [(k, v) for k, v in ch.sub_grid.items() if k[1] == vf]
    if same_gate:
        same_gate.sort(key=lambda kv: abs(kv[0][0] - p))
        return same_gate[0][1]
    same_power = [(k, v) for k, v in ch.sub_grid.items() if k[0] == p]
    if same_power:
        same_power.sort(key=lambda kv: abs(kv[0][1] - vf))
        return same_power[0][1]
    return next(iter(ch.sub_grid.values()), None)


def _load_sub_folder(ch: _Channel, folder: Path) -> None:
    frames = _load_frames(folder)
    ch.sub_single = None
    ch.sub_grid   = {}
    if not frames:
        return
    default_p  = _norm_key(frames[0].power_w) or 0.0
    default_vf = _norm_key(frames[0].v_front)  or 0.0
    for fr in frames:
        p  = float(_norm_key(fr.power_w) or default_p)
        vf = float(_norm_key(fr.v_front) or default_vf)
        key = (p, vf)
        old = ch.sub_grid.get(key)
        if old is None or fr.mtime > old.mtime:
            ch.sub_grid[key] = fr


# ══════════════════════════════════════════════════════════════════════
# SingleFrameView  (pyqtgraph single-frame viewer with linecut)
# ══════════════════════════════════════════════════════════════════════

def _pg_cmap(name: str) -> pg.ColorMap:
    for source in ("matplotlib", None):
        try:
            return pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
        except Exception:
            pass
    return pg.colormap.get("viridis")


class SingleFrameView(QtWidgets.QWidget):
    """pyqtgraph-based PL image viewer with movable linecut selector."""

    linecut_changed = QtCore.pyqtSignal(int)  # display row (0 = top)
    feature_regions_edited = QtCore.pyqtSignal(object)  # [(lo_nm, hi_nm), ...]

    def __init__(self, title: str = "", *, default_linecut_width: int = 1, parent=None):
        super().__init__(parent)
        self._img_data: Optional[np.ndarray] = None
        self._wl:       Optional[np.ndarray] = None
        self._lc_terms = None  # ΔR/R0: [(sign, num, den)] → linecut = ratio of row sums
        self._flatten  = None  # (feature_ranges_nm, order) when flattening ΔR/R0
        self._h = self._w = 0
        self._build_ui(str(title), int(default_linecut_width))

    def _build_ui(self, title: str, lc_width: int) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(4)

        self._titleLbl = QtWidgets.QLabel(title)
        vlay.addWidget(self._titleLbl)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Linecut width:"))
        self.linecutWidthSpin = QtWidgets.QSpinBox()
        self.linecutWidthSpin.setRange(1, 1000)
        self.linecutWidthSpin.setValue(lc_width)
        self.linecutWidthSpin.setFixedWidth(60)
        ctrl.addWidget(self.linecutWidthSpin)
        ctrl.addStretch()
        vlay.addLayout(ctrl)

        self._glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self._glw)

        # Image plot
        self._imgPlot = self._glw.addPlot(row=0, col=0)
        self._imgPlot.getViewBox().invertY(True)
        self._imgPlot.setLabel("bottom", "Wavelength (nm)")
        self._imgPlot.setLabel("left", "Spatial pixel")
        self._imgItem = pg.ImageItem()
        self._imgPlot.addItem(self._imgItem)

        # Movable horizontal line (display row → horizontal linecut)
        self._hline = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen("y", width=1.5),
            hoverPen=pg.mkPen("y", width=2.5),
        )
        self._hline.setValue(0)
        self._imgPlot.addItem(self._hline)
        self._hline.sigPositionChangeFinished.connect(self._on_hline_moved)

        # Movable vertical line (wavelength column → vertical linecut)
        self._vline = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("c", width=1.5),
            hoverPen=pg.mkPen("c", width=2.5),
        )
        self._vline.setValue(0)
        self._imgPlot.addItem(self._vline)
        self._vline.sigPositionChangeFinished.connect(self._refresh_vlinecut)

        # Color bar (narrow)
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._imgItem)
        self._glw.addItem(self._hist, row=0, col=1)

        # Vertical linecut plot — intensity vs spatial pixel at cursor wavelength
        self._vlcPlot = self._glw.addPlot(row=0, col=2)
        self._vlcPlot.setLabel("bottom", "Intensity")
        self._vlcPlot.setLabel("left", "")
        self._vlcPlot.getViewBox().invertY(True)
        self._vlcPlot.setYLink(self._imgPlot)
        self._vlcCurve = self._vlcPlot.plot(pen=pg.mkPen("#ff7f0e", width=1.2))

        # Horizontal linecut plot — spectrum at cursor row
        self._lcPlot = self._glw.addPlot(row=1, col=0, colspan=3)
        self._lcPlot.setLabel("bottom", "Wavelength (nm)")
        self._lcPlot.setLabel("left", "Intensity")
        self._lcPlot.addLegend(offset=(-10, 10))
        self._lcCurve   = self._lcPlot.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), name="Linecut")
        self._fitTotal  = self._lcPlot.plot(
            pen=pg.mkPen("#ffb300", width=1.2, style=QtCore.Qt.DashLine), name="Fit total")
        self._fitPeak1  = self._lcPlot.plot(
            pen=pg.mkPen("#1f77b4", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 1")
        self._fitPeak2  = self._lcPlot.plot(
            pen=pg.mkPen("#d62728", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 2")
        # Dip mode: smoothed spectrum curve + vertical position lines
        self._smoothCurve = self._lcPlot.plot(
            pen=pg.mkPen("#aaaaaa", width=1.0, style=QtCore.Qt.DashLine),
            name="Smoothed")
        self._smoothCurve.setVisible(False)
        self._dipVLines: List[pg.InfiniteLine] = []
        # ΔR/R0 flatten mode: unflattened ratio, smooth fit (g − 1), feature regions
        self._rawRcCurve  = pg.PlotDataItem(pen=pg.mkPen("#9e9e9e", width=1.0))
        self._flatFitCurve = pg.PlotDataItem(
            pen=pg.mkPen("#2e7d32", width=1.5, style=QtCore.Qt.DashLine))
        for it in (self._rawRcCurve, self._flatFitCurve):
            it.setZValue(-5)
            self._lcPlot.addItem(it)
        self._flat_legend = False
        self._featRegions: List[pg.LinearRegionItem] = []

        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 2)
        self._glw.ci.layout.setColumnMaximumWidth(1, 65)   # narrow histogram
        self._glw.ci.layout.setColumnStretchFactor(0, 4)   # image (wide)
        self._glw.ci.layout.setColumnStretchFactor(2, 2)   # vertical linecut

        self.linecutWidthSpin.valueChanged.connect(lambda _: self._refresh_linecut())

    # ── public interface ──────────────────────────────────────────────

    def linecut_row(self) -> Optional[int]:
        if self._h == 0:
            return None
        return int(np.clip(round(self._hline.value()), 0, self._h - 1))

    def linecut_width(self) -> int:
        return int(self.linecutWidthSpin.value())

    def set_image(self, img: np.ndarray, wl: Optional[np.ndarray] = None,
                  lc_terms=None, flatten=None) -> None:
        """lc_terms: optional [(sign, num, den)] images; the linecut is then
        Σ sign·(Σrows num)/(Σrows den) instead of the row sum of ``img``.
        flatten: optional (feature_ranges_nm, order), see ``_flatten_rc``."""
        self._img_data = np.asarray(img, dtype=float)
        self._lc_terms = lc_terms
        self._flatten  = flatten if lc_terms else None
        self._h, self._w = self._img_data.shape
        self._wl = None if wl is None else np.asarray(wl, dtype=float).ravel()

        # flipud so display row 0 = raw row h-1 (matches v1 coordinate convention);
        # transpose to (width, height) as required by ImageItem.
        disp = np.flipud(self._img_data).T

        if self._wl is not None and self._wl.size == self._w:
            x0 = float(self._wl[0])
            x_span = float(self._wl[-1] - self._wl[0]) or 1.0
        else:
            x0, x_span = 0.0, float(self._w) or 1.0

        locked = getattr(self, "_axes_locked", False)
        disp_safe = disp if np.isfinite(disp).any() else np.zeros_like(disp)
        self._imgItem.setImage(disp_safe, autoLevels=not locked)
        self._imgItem.setRect(QtCore.QRectF(x0, 0, x_span, float(self._h)))
        if not locked:
            self._imgPlot.setXRange(x0, x0 + x_span, padding=0.01)
            self._imgPlot.setYRange(0, float(self._h), padding=0)
            mid = x0 + x_span / 2.0
            self._vline.setValue(mid)
        self._refresh_linecut()
        self._refresh_vlinecut()

    def set_image_title(self, text: str) -> None:
        self._titleLbl.setText(str(text))

    def set_crop(self, *args) -> None:
        pass  # crop is applied externally before set_image is called

    def clear_linecut_fit_overlay(self) -> None:
        self._fitTotal.setData([], [])
        self._fitPeak1.setData([], [])
        self._fitPeak2.setData([], [])

    def set_linecut_fit_overlay(self, x_axis, y_total, y1, y2) -> None:
        x = np.asarray(x_axis, dtype=float).ravel()
        if x.size == 0:
            self.clear_linecut_fit_overlay()
            return

        def _v(y):
            if y is None:
                return None
            a = np.asarray(y, dtype=float).ravel()
            return a if (a.size == x.size and np.isfinite(a).any()) else None

        yt, y1v, y2v = _v(y_total), _v(y1), _v(y2)
        self._fitTotal.setData(x, yt)  if yt  is not None else self._fitTotal.setData([], [])
        self._fitPeak1.setData(x, y1v) if y1v is not None else self._fitPeak1.setData([], [])
        self._fitPeak2.setData(x, y2v) if y2v is not None else self._fitPeak2.setData([], [])

    # ── internal ──────────────────────────────────────────────────────

    def _on_hline_moved(self) -> None:
        row_d = self.linecut_row()
        if row_d is not None:
            self.linecut_changed.emit(row_d)
        self._refresh_linecut()

    def _refresh_linecut(self) -> None:
        if self._img_data is None or self._h == 0:
            self._lcCurve.setData([], [])
            return
        row_d   = int(np.clip(round(self._hline.value()), 0, self._h - 1))
        row_raw = self._h - 1 - row_d  # convert display row (0=top) to raw row
        width   = self.linecut_width()
        if self._lc_terms:
            lc = _lc_from_terms(self._lc_terms, row_raw, width,
                                wl=self._wl, flatten=self._flatten)
        else:
            lc = _linecut_h(self._img_data, row_raw, width)
        self._rawRcCurve.setData([], [])
        self._flatFitCurve.setData([], [])
        if lc is None:
            self._lcCurve.setData([], [])
            self._show_flat_legend(False)
            return
        x = self._wl if (self._wl is not None and self._wl.size == lc.size) \
            else np.arange(lc.size, dtype=float)
        self._lcCurve.setData(x, lc)
        # Flatten mode, single term: also show the raw ratio and the smooth fit
        show = bool(self._flatten) and len(self._lc_terms or []) == 1 \
            and self._lc_terms[0][2] is not None
        if show:
            raw = _lc_from_terms(self._lc_terms, row_raw, width)
            if raw is not None and raw.size == x.size:
                fit = _flatten_rc(x, raw, *self._flatten)[1]
                self._rawRcCurve.setData(x, raw)
                self._flatFitCurve.setData(x, fit)
        self._show_flat_legend(show)

    def _show_flat_legend(self, on: bool) -> None:
        leg = self._lcPlot.legend
        if leg is None or on == self._flat_legend:
            return
        if on:
            leg.addItem(self._rawRcCurve, "R/R₀ − 1 (raw)")
            leg.addItem(self._flatFitCurve, "Smooth fit")
        else:
            leg.removeItem(self._rawRcCurve)
            leg.removeItem(self._flatFitCurve)
        self._flat_legend = on

    def set_feature_regions(self, ranges_nm: List[Tuple[float, float]], visible: bool) -> None:
        """Shaded, draggable feature regions on the linecut plot (flatten mode).
        Dragging one emits ``feature_regions_edited`` with the new list."""
        ranges_nm = list(ranges_nm) if visible else []
        if len(ranges_nm) != len(self._featRegions):
            for r in self._featRegions:
                self._lcPlot.removeItem(r)
            self._featRegions = []
            for _ in ranges_nm:
                r = pg.LinearRegionItem(brush=pg.mkBrush(255, 160, 0, 45),
                                        pen=pg.mkPen((230, 120, 0, 160)))
                r.setZValue(-10)
                r.sigRegionChangeFinished.connect(self._on_feat_region_moved)
                self._lcPlot.addItem(r)
                self._featRegions.append(r)
        for r, (lo, hi) in zip(self._featRegions, ranges_nm):
            r.blockSignals(True)
            r.setRegion((lo, hi))
            r.blockSignals(False)

    def _on_feat_region_moved(self, *_) -> None:
        self.feature_regions_edited.emit(
            [tuple(sorted(float(v) for v in r.getRegion())) for r in self._featRegions])

    def _refresh_vlinecut(self) -> None:
        if self._img_data is None or self._h == 0 or self._w == 0:
            self._vlcCurve.setData([], [])
            return
        pos = float(self._vline.value())
        if self._wl is not None and self._wl.size == self._w:
            col = int(np.argmin(np.abs(self._wl - pos)))
        else:
            col = int(np.clip(round(pos), 0, self._w - 1))
        col = max(0, min(col, self._w - 1))
        # flip to display order: display row 0 = data row (h-1)
        profile = self._img_data[::-1, col]
        rows    = np.arange(self._h, dtype=float)
        self._vlcCurve.setData(profile, rows)

    # ── dip overlay ───────────────────────────────────────────────────

    def clear_dip_overlay(self) -> None:
        self._smoothCurve.setVisible(False)
        for vl in self._dipVLines:
            vl.setVisible(False)

    def set_dip_vlines(self, positions_nm: list, *,
                       x_smooth=None, y_smooth=None) -> None:
        """Show smoothed spectrum and a vertical line per dip position."""
        if x_smooth is not None and y_smooth is not None:
            xs = np.asarray(x_smooth, dtype=float).ravel()
            ys = np.asarray(y_smooth, dtype=float).ravel()
            if xs.size == ys.size and xs.size > 0:
                self._smoothCurve.setData(xs, ys)
                self._smoothCurve.setVisible(True)
            else:
                self._smoothCurve.setVisible(False)
        else:
            self._smoothCurve.setVisible(False)

        n = len(positions_nm)
        while len(self._dipVLines) < n:
            i     = len(self._dipVLines)
            color = _DIP_COLORS[i % len(_DIP_COLORS)]
            vl = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen(color, width=1.5,
                                              style=QtCore.Qt.DashLine))
            self._lcPlot.addItem(vl)
            vl.setVisible(False)
            self._dipVLines.append(vl)
        for vl in self._dipVLines:
            vl.setVisible(False)
        for i, pos in enumerate(positions_nm):
            if pos is not None:
                try:
                    v = float(pos)
                    if np.isfinite(v):
                        self._dipVLines[i].setValue(v)
                        self._dipVLines[i].setVisible(True)
                except (TypeError, ValueError):
                    pass


# ══════════════════════════════════════════════════════════════════════
# MapPlotWidget  (pyqtgraph 2-D image map + row linecut)
# ══════════════════════════════════════════════════════════════════════

class MapPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, *, cmap_name: str, line_label: str, parent=None):
        super().__init__(title, parent)
        self._line_label     = line_label
        self._cmap_name      = cmap_name
        self._manual_sym_max = None
        self._x = self._y = self._z = None
        self._selected_y_idx = None
        self._top_x          = None
        self._top_label      = ""
        self._last_vline_x   = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addStretch()
        top_row.addWidget(self.exportBtn)
        layout.addLayout(top_row)

        self._glw = pg.GraphicsLayoutWidget()

        # Map image plot
        self._mapPlot = self._glw.addPlot(row=0, col=0)
        self._mapPlot.setLabel("bottom", "")
        self._mapPlot.setLabel("left", "")
        self._imgItem = pg.ImageItem()
        try:
            self._imgItem.setColorMap(_pg_cmap(self._cmap_name))
        except Exception:
            pass
        self._mapPlot.addItem(self._imgItem)

        # Histogram / colour control (narrow)
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._imgItem)
        self._glw.addItem(self._hist, row=0, col=1)

        # Horizontal selection line — draggable, selects y/energy row
        self._hline = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen("w", width=0.8),
            hoverPen=pg.mkPen("w", width=1.5),
        )
        self._mapPlot.addItem(self._hline)
        self._hline.sigPositionChangeFinished.connect(self._on_hline_moved)

        # Vertical cursor line — draggable, selects x/sweep column
        self._vline = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("c", width=0.8),
            hoverPen=pg.mkPen("c", width=1.5),
        )
        self._mapPlot.addItem(self._vline)
        self._vline.sigPositionChangeFinished.connect(self._on_vline_moved)

        # Vertical linecut plot — spectrum at cursor x/sweep value
        self._vlcPlot = self._glw.addPlot(row=0, col=2)
        self._vlcPlot.setLabel("bottom", self._line_label)
        self._vlcPlot.setLabel("left", "")
        self._vlcPlot.setYLink(self._mapPlot)
        self._vlcCurve = self._vlcPlot.plot(pen=pg.mkPen("#ff7f0e", width=1.2))
        self._vlcTitleItem = pg.TextItem("", anchor=(0, 1))
        self._vlcPlot.addItem(self._vlcTitleItem)

        # Horizontal linecut plot — trace at cursor y/energy value
        self._lcPlot = self._glw.addPlot(row=1, col=0, colspan=3)
        self._lcPlot.setLabel("bottom", "")
        self._lcPlot.setLabel("left", self._line_label)
        self._lcCurve = self._lcPlot.plot(pen=pg.mkPen("#1565c0", width=1.2))
        self._lcTitleItem = pg.TextItem("", anchor=(0, 1))
        self._lcPlot.addItem(self._lcTitleItem)

        self._glw.ci.layout.setRowStretchFactor(0, 3)
        self._glw.ci.layout.setRowStretchFactor(1, 1)
        self._glw.ci.layout.setColumnMaximumWidth(1, 65)   # narrow histogram
        self._glw.ci.layout.setColumnStretchFactor(0, 4)   # map (wide)
        self._glw.ci.layout.setColumnStretchFactor(2, 2)   # vertical linecut

        layout.addWidget(self._glw)

        self._mapPlot.scene().sigMouseClicked.connect(self._on_scene_click)
        self.exportBtn.clicked.connect(self._on_export)

    # ── public interface ──────────────────────────────────────────────

    def set_manual_symmetric_max(self, value: Optional[float]) -> None:
        self._manual_sym_max = None if (value is None or value <= 0) else float(value)

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x     = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    def set_map(self, x, y, z, *, x_label: str, y_label: str, symmetric: bool = False) -> None:
        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        self._z = np.asarray(z, dtype=float)

        if not (self._x.size and self._y.size and self._z.size):
            return

        n_x, n_y = self._x.size, self._y.size

        # ImageItem expects shape (n_x, n_y); z has shape (n_y, n_x)
        img_data = self._z.T.copy()

        x0 = float(self._x[0])
        x_span = float(self._x[-1] - self._x[0]) if n_x > 1 else 1.0
        y0 = float(self._y[0])
        y_span = float(self._y[-1] - self._y[0]) if n_y > 1 else 1.0

        locked = getattr(self, "_axes_locked", False)
        img_safe = img_data if np.isfinite(img_data).any() else np.zeros_like(img_data)
        self._imgItem.setImage(img_safe, autoLevels=not locked)
        self._imgItem.setRect(QtCore.QRectF(x0, y0, x_span, y_span))

        if not locked:
            finite = self._z[np.isfinite(self._z)]
            if finite.size:
                if symmetric:
                    vmax = float(self._manual_sym_max or np.nanpercentile(np.abs(finite), 98))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = max(float(np.nanmax(np.abs(finite))), 1.0)
                    self._imgItem.setLevels((-vmax, vmax))
                else:
                    vmin = float(np.nanpercentile(finite, 1))
                    vmax = float(np.nanpercentile(finite, 99))
                    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
                        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
                    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
                        vmin, vmax = 0.0, 1.0
                    self._imgItem.setLevels((vmin, vmax))

        self._mapPlot.setLabel("bottom", x_label)
        self._mapPlot.setLabel("left",   y_label)
        self._lcPlot.setLabel("bottom",  x_label)
        if not locked:
            self._mapPlot.setXRange(x0, x0 + x_span, padding=0.02)
            self._mapPlot.setYRange(y0, y0 + y_span, padding=0.02)

        if self._selected_y_idx is None:
            self._selected_y_idx = n_y // 2
        self._selected_y_idx = max(0, min(int(self._selected_y_idx), n_y - 1))
        self._hline.setValue(float(self._y[self._selected_y_idx]))
        self._update_linecut()
        self._apply_secondary_axis()

        if self._last_vline_x is not None:
            self.set_x_marker(self._last_vline_x)
        else:
            self._vline.setValue(float(self._x[0]))
            self._update_vlinecut()

    def set_x_marker(self, x: float) -> None:
        self._last_vline_x = float(x)
        self._vline.setValue(float(x))
        self._update_vlinecut()

    # ── internal ──────────────────────────────────────────────────────

    def _apply_secondary_axis(self) -> None:
        if self._top_x is None or self._x is None or self._top_x.size != self._x.size:
            self._mapPlot.showAxis("top", False)
            return
        self._mapPlot.showAxis("top", True)
        top_ax = self._mapPlot.getAxis("top")
        top_ax.setLabel(self._top_label)
        nticks = min(8, self._x.size)
        idx    = np.linspace(0, self._x.size - 1, nticks).astype(int)
        ticks  = [(float(self._x[i]),
                   f"{float(self._top_x[i]):.3g}" if np.isfinite(self._top_x[i]) else "")
                  for i in idx]
        top_ax.setTicks([ticks])

    def _update_linecut(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            return
        yv  = float(self._hline.value())
        idx = int(np.argmin(np.abs(self._y - yv)))
        idx = max(0, min(idx, self._y.size - 1))
        self._selected_y_idx = idx
        self._hline.setValue(float(self._y[idx]))
        self._lcCurve.setData(self._x, self._z[idx, :])
        self._lcTitleItem.setText(f"{float(self._y[idx]):.4g}")

    def _on_hline_moved(self) -> None:
        self._update_linecut()

    def _on_vline_moved(self) -> None:
        self._update_vlinecut()

    def _update_vlinecut(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            self._vlcCurve.setData([], [])
            return
        xv  = float(self._vline.value())
        idx = int(np.argmin(np.abs(self._x - xv)))
        idx = max(0, min(idx, self._x.size - 1))
        self._vline.setValue(float(self._x[idx]))
        self._vlcCurve.setData(self._z[:, idx], self._y)
        self._vlcTitleItem.setText(f"{float(self._x[idx]):.4g}")

    def _on_scene_click(self, event) -> None:
        if self._y is None or self._z is None:
            return
        pos = event.scenePos()
        if not self._mapPlot.sceneBoundingRect().contains(pos):
            return
        vb = self._mapPlot.getViewBox()
        pt = vb.mapSceneToView(pos)
        yv = float(pt.y())
        idx = int(np.argmin(np.abs(self._y - yv)))
        self._selected_y_idx = idx
        self._hline.setValue(float(self._y[idx]))
        self._update_linecut()

    def _on_export(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data.")
            return
        x, y, z = np.asarray(self._x), np.asarray(self._y), np.asarray(self._z)
        if z.shape != (y.size, x.size):
            QtWidgets.QMessageBox.warning(self, "Export CSV", "Shape mismatch.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        out = np.empty((y.size + 1, x.size + 1), dtype=float)
        out.fill(np.nan)
        out[0, 0]  = 0.0
        out[0, 1:] = x
        out[1:, 0] = y
        out[1:, 1:] = z
        np.savetxt(path, out, delimiter=",", fmt="%.10g")


# ══════════════════════════════════════════════════════════════════════
# FitTrendPlotWidget  (pyqtgraph peak fit trend over sweep axis)
# ══════════════════════════════════════════════════════════════════════

class FitTrendPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, y_label: str, parent=None):
        super().__init__(title, parent)
        self._y_label = y_label
        self._x_last  = self._y1_last = self._y2_last = None
        self._xl_last = "X"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addStretch()
        top_row.addWidget(self.exportBtn)
        layout.addLayout(top_row)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left",   self._y_label)
        self._plot.setLabel("bottom", "X")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(-10, 10))
        self._curve1 = self._plot.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#1f77b4", symbolPen=None, name="Peak 1")
        self._curve2 = self._plot.plot(
            pen=pg.mkPen("#d62728", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#d62728", symbolPen=None, name="Peak 2")
        layout.addWidget(self._plot)
        self.exportBtn.clicked.connect(self._on_export)

    def set_data(self, x, y1, y2, *, x_label: str,
                 label_1: str = "Peak 1", label_2: str = "Peak 2") -> None:
        x  = np.asarray(x,  dtype=float).ravel()
        y1 = np.asarray(y1, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel()
        n  = min(x.size, y1.size, y2.size)
        self._x_last  = x[:n].copy()
        self._y1_last = y1[:n].copy()
        self._y2_last = y2[:n].copy()
        self._xl_last = str(x_label)
        m1 = np.isfinite(x[:n]) & np.isfinite(y1[:n])
        m2 = np.isfinite(x[:n]) & np.isfinite(y2[:n])
        self._curve1.setData(x[:n][m1], y1[:n][m1])
        self._curve2.setData(x[:n][m2], y2[:n][m2])
        self._plot.setLabel("bottom", x_label)

    def clear_data(self, message: str, *, x_label: str = "X") -> None:
        self._x_last = self._y1_last = self._y2_last = None
        self._xl_last = str(x_label)
        self._curve1.setData([], [])
        self._curve2.setData([], [])
        self._plot.setLabel("bottom", x_label)
        self._plot.setTitle(message)

    def _on_export(self) -> None:
        if self._x_last is None:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data.")
            return
        x, y1, y2 = self._x_last, self._y1_last, self._y2_last
        n = min(x.size, y1.size, y2.size)
        if n == 0:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        hdr = f"{self._xl_last},peak1,peak2"
        np.savetxt(path, np.column_stack([x[:n], y1[:n], y2[:n]]),
                   delimiter=",", fmt="%.10g", header=hdr, comments="")


# ══════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════

class PLViewerV2(QtWidgets.QMainWindow):
    MODE_POWER = "Power dependence"
    MODE_GATE  = "Gate dependence"

    def __init__(self):
        super().__init__()
        # one _Channel per key: "" = non-pol, "LH", "RH"
        self._ch: Dict[str, _Channel] = {"": _Channel(), "LH": _Channel(), "RH": _Channel()}
        self._crop      = DEFAULT_CROP
        self._power_idx = 0
        self._front_idx = 0
        self._back_idx  = 0
        self._current_mode = self.MODE_POWER
        self._gate_axis    = "front"   # "front" | "back" — swept gate in gate mode
        self._cube_win     = None
        self._last_x: Optional[np.ndarray]      = None
        self._last_y_wl: Optional[np.ndarray]   = None
        self._last_map: Optional[np.ndarray]     = None
        self._last_x_label = ""
        self._fit_result      = None
        self._fit_map_version = -1
        self._map_version     = 0
        self._dip_fit_result      = None
        self._dip_fit_map_version = -1
        self._active_overlay      = "peak"   # "peak" | "dip"

        self._build_ui()
        self._refresh_all()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("PL Power/Gate Viewer  v2")
        self.resize(1900, 980)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._make_controls())

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.liveView = SingleFrameView(title="PL Image", default_linecut_width=1)
        self.liveView.linecut_changed.connect(lambda _: self._update_maps())
        self.liveView.linecutWidthSpin.valueChanged.connect(lambda _: self._update_maps())
        self.liveView.feature_regions_edited.connect(self._on_feat_regions_dragged)
        split.addWidget(self.liveView)
        self.mapView   = MapPlotWidget("Linecut map",   cmap_name="viridis",  line_label="I")
        self.derivView = MapPlotWidget("Derivative map", cmap_name="coolwarm", line_label="dI/dx")
        self.fitWidthView     = FitTrendPlotWidget("Lorentzian Width vs Sweep", "FWHM (meV)")
        self.fitEnergyView    = FitTrendPlotWidget("Peak Position vs Sweep",    "Energy (eV)")
        self.fitIntensityView = FitTrendPlotWidget("Peak Intensity vs Sweep",   "Fitted Intensity (a.u.)")
        self.dipPosPlot = pg.PlotWidget()
        self.dipPosPlot.setLabel("left",   "Energy (eV)")
        self.dipPosPlot.setLabel("bottom", "X")
        self.dipPosPlot.showGrid(x=True, y=True, alpha=0.25)
        self.dipPosPlot.addLegend(offset=(-10, 10))
        self._dipPosCurves: List[object] = []

        self.derivEView  = MapPlotWidget("dI/dE",    cmap_name="coolwarm", line_label="dI/dE")
        self.deriv2EView = MapPlotWidget("d²I/dE²", cmap_name="coolwarm", line_label="d²I/dE²")

        self.analysisTabs = QtWidgets.QTabWidget()
        self.analysisTabs.addTab(self.derivView,        "Derivative")
        self.analysisTabs.addTab(self.fitWidthView,     "Fit Width")
        self.analysisTabs.addTab(self.fitEnergyView,    "Peak Energy")
        self.analysisTabs.addTab(self.fitIntensityView, "Peak Intensity")
        self.analysisTabs.addTab(self.dipPosPlot,       "Dip Positions")
        self.analysisTabs.addTab(self.derivEView,       "dI/dE")
        self.analysisTabs.addTab(self.deriv2EView,      "d²I/dE²")
        split.addWidget(self.mapView)
        split.addWidget(self.analysisTabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        self.setCentralWidget(central)

    def _make_controls(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Controls")
        outer = QtWidgets.QHBoxLayout(box)
        outer.setContentsMargins(8, 8, 8, 6)
        outer.setSpacing(10)

        # ── Left column: data source + corrections ────────────────────
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(4)

        self.polCheck = QtWidgets.QCheckBox("Polarization dependence")
        left.addWidget(self.polCheck)

        self.srcStack = QtWidgets.QStackedWidget()
        self.srcStack.addWidget(self._make_nonpol_page())   # index 0
        self.srcStack.addWidget(self._make_pol_page())      # index 1
        left.addWidget(self.srcStack)

        corr = QtWidgets.QGroupBox("Corrections")
        cv = QtWidgets.QVBoxLayout(corr)
        cv.setContentsMargins(8, 4, 8, 4)
        cv.setSpacing(4)
        cl = QtWidgets.QHBoxLayout()
        cl.setSpacing(8)
        cv.addLayout(cl)

        self.baselineCheck = QtWidgets.QCheckBox("Baseline correction")
        cl.addWidget(self.baselineCheck)
        cl.addWidget(QtWidgets.QLabel("Range:"))
        self.bgLoSpin = QtWidgets.QDoubleSpinBox()
        self.bgLoSpin.setDecimals(1)
        self.bgLoSpin.setRange(200.0, 2000.0)
        self.bgLoSpin.setValue(900.0)
        self.bgLoSpin.setFixedWidth(75)
        self.bgLoSpin.setSuffix(" nm")
        cl.addWidget(self.bgLoSpin)
        cl.addWidget(QtWidgets.QLabel("to"))
        self.bgHiSpin = QtWidgets.QDoubleSpinBox()
        self.bgHiSpin.setDecimals(1)
        self.bgHiSpin.setRange(200.0, 2000.0)
        self.bgHiSpin.setValue(940.0)
        self.bgHiSpin.setFixedWidth(75)
        self.bgHiSpin.setSuffix(" nm")
        cl.addWidget(self.bgHiSpin)
        self.subCheck = QtWidgets.QCheckBox("Subtract substrate")
        cl.addWidget(self.subCheck)
        self.rcCheck = QtWidgets.QCheckBox("÷ substrate (ΔR/R₀)")
        self.rcCheck.setToolTip(
            "Also divide by the substrate: (R − R0) / R0.\n"
            "Linecuts and maps use the ratio of row sums over the linecut width;\n"
            "the 2-D image shows the per-pixel ratio.\n"
            f"Points where |R0| < {RC_MIN_FRAC:.0%} of its max are blanked (NaN).\n"
            "Turn on baseline correction so R0 has no dark offset.")
        self.rcCheck.setEnabled(False)
        cl.addWidget(self.rcCheck)
        self.lockAxesCheck = QtWidgets.QCheckBox("Lock axes")
        self.lockAxesCheck.setToolTip(
            "Freeze all axis ranges and colour scales so switching\n"
            "conditions does not rescale the plots.")
        cl.addWidget(self.lockAxesCheck)
        self.loadAngleCalibBtn = QtWidgets.QPushButton("Power calib…")
        self.loadAngleCalibBtn.setToolTip(
            "Load angle→power calibration (.xlsx or .csv) for data\n"
            "collected with a single rotation stage (angle-list mode)")
        self.loadAngleCalibBtn.setFixedWidth(110)
        cl.addWidget(self.loadAngleCalibBtn)
        self.angleCalibStatusLbl = QtWidgets.QLabel("No calib")
        self.angleCalibStatusLbl.setStyleSheet("color: #888; font-size: 8pt;")
        cl.addWidget(self.angleCalibStatusLbl)
        cl.addStretch()

        # Row 2: absolute ΔR/R0 — divide out a smooth fit of the feature-free part
        cl2 = QtWidgets.QHBoxLayout()
        cl2.setSpacing(8)
        cv.addLayout(cl2)
        self.flatCheck = QtWidgets.QCheckBox("Flatten ΔR/R₀")
        self.flatCheck.setToolTip(
            "Absolute RC: R/R0 = (1 + RC)·g(λ), with g a smooth error from focus /\n"
            "spot mismatch. g is a polynomial fitted to R/R0 outside the feature\n"
            "ranges; shown is (R/R0)/g − 1. Each spectrum (map column, image row)\n"
            "gets its own fit. The linecut panel shows the raw ratio (grey), the\n"
            "fit (green dashed) and the feature ranges (orange, draggable).")
        self.flatCheck.setEnabled(False)
        cl2.addWidget(self.flatCheck)
        cl2.addWidget(QtWidgets.QLabel("Features:"))
        self.featRangesEdit = QtWidgets.QLineEdit("820-870")
        self.featRangesEdit.setPlaceholderText("e.g. 690-710, 820-870  (or eV: 1.45-1.5)")
        self.featRangesEdit.setToolTip(
            "Wavelength ranges (nm) that contain features; they are left out of the\n"
            "smooth fit. Separate ranges with commas. A range with both ends < 10\n"
            "is read as eV. Features at the edge of the spectrum rely on\n"
            "extrapolating the fit, so keep the order low there.")
        self.featRangesEdit.setMinimumWidth(200)
        cl2.addWidget(self.featRangesEdit, 1)
        cl2.addWidget(QtWidgets.QLabel("Order:"))
        self.flatOrderSpin = QtWidgets.QSpinBox()
        self.flatOrderSpin.setRange(0, 8)
        self.flatOrderSpin.setValue(2)
        self.flatOrderSpin.setFixedWidth(50)
        self.flatOrderSpin.setToolTip("Polynomial order of the smooth fit g(λ)")
        cl2.addWidget(self.flatOrderSpin)
        cl2.addStretch()
        left.addWidget(corr)
        left.addStretch()
        outer.addLayout(left, 2)

        # ── Right column: mode/sliders + compact misc ─────────────────
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(4)

        slider_grid = QtWidgets.QGridLayout()
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(4)

        slider_grid.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems([self.MODE_POWER, self.MODE_GATE])
        self.modeCombo.setFixedWidth(150)
        slider_grid.addWidget(self.modeCombo, 0, 1)

        slider_grid.addWidget(QtWidgets.QLabel("Gate axis:"), 0, 2)
        self.gateAxisCombo = QtWidgets.QComboBox()
        self.gateAxisCombo.addItems(["Front gate", "Back gate"])
        self.gateAxisCombo.setFixedWidth(120)
        self.gateAxisCombo.setToolTip(
            "In Gate dependence mode, choose which gate is swept along the map\n"
            "x-axis; the other gate is held fixed by its slider (grid data only).")
        slider_grid.addWidget(self.gateAxisCombo, 0, 3)

        self.cubeBtn = QtWidgets.QPushButton("Print cube")
        self.cubeBtn.setFixedWidth(100)
        self.cubeBtn.setToolTip(
            "Open a 3-D volume: x = front gate, y = back gate, z = wavelength,\n"
            "colour = baseline-corrected linecut intensity (baseline transparent).")
        slider_grid.addWidget(self.cubeBtn, 0, 4)

        # Power / Front / Back sliders
        slider_grid.addWidget(QtWidgets.QLabel("Power:"), 1, 0)
        self.powerSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.powerSlider.setTracking(True)
        slider_grid.addWidget(self.powerSlider, 1, 1, 1, 3)
        self.powerLbl = QtWidgets.QLabel("--")
        self.powerLbl.setMinimumWidth(200)
        slider_grid.addWidget(self.powerLbl, 1, 4)

        slider_grid.addWidget(QtWidgets.QLabel("Front gate:"), 2, 0)
        self.frontSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frontSlider.setTracking(True)
        slider_grid.addWidget(self.frontSlider, 2, 1, 1, 3)
        self.frontLbl = QtWidgets.QLabel("--")
        slider_grid.addWidget(self.frontLbl, 2, 4)

        slider_grid.addWidget(QtWidgets.QLabel("Back gate:"), 3, 0)
        self.backSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.backSlider.setTracking(True)
        slider_grid.addWidget(self.backSlider, 3, 1, 1, 3)
        self.backLbl = QtWidgets.QLabel("--")
        slider_grid.addWidget(self.backLbl, 3, 4)

        right.addLayout(slider_grid)

        # Compact misc grid — 4 rows
        misc_grid = QtWidgets.QGridLayout()
        misc_grid.setHorizontalSpacing(6)
        misc_grid.setVerticalSpacing(3)

        # Row 0: Crop + Deriv max
        misc_grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 0, 0)
        self.cropT = QtWidgets.QSpinBox(); self.cropB = QtWidgets.QSpinBox()
        self.cropL = QtWidgets.QSpinBox(); self.cropR = QtWidgets.QSpinBox()
        for s in (self.cropT, self.cropB, self.cropL, self.cropR):
            s.setRange(0, 10000); s.setFixedWidth(58)
        self.cropT.setValue(DEFAULT_CROP[0]); self.cropB.setValue(DEFAULT_CROP[1])
        self.cropL.setValue(DEFAULT_CROP[2]); self.cropR.setValue(DEFAULT_CROP[3])
        self.cropApplyBtn = QtWidgets.QPushButton("Apply Crop")
        self.cropApplyBtn.setFixedWidth(88)
        misc_grid.addWidget(self.cropT,        0, 1)
        misc_grid.addWidget(self.cropB,        0, 2)
        misc_grid.addWidget(self.cropL,        0, 3)
        misc_grid.addWidget(self.cropR,        0, 4)
        misc_grid.addWidget(self.cropApplyBtn, 0, 5)
        misc_grid.addWidget(QtWidgets.QLabel("Deriv max:"), 0, 6)
        self.derivMaxSpin = QtWidgets.QDoubleSpinBox()
        self.derivMaxSpin.setDecimals(4)
        self.derivMaxSpin.setRange(0.0, 1e12)
        self.derivMaxSpin.setSpecialValueText("Auto")
        self.derivMaxSpin.setValue(0.0)
        self.derivMaxSpin.setFixedWidth(100)
        misc_grid.addWidget(self.derivMaxSpin, 0, 7)

        # Row 1: SG smooth + Reverse λ
        misc_grid.addWidget(QtWidgets.QLabel("SG smooth:"), 1, 0)
        self.sgSweepCheck = QtWidgets.QCheckBox("Sweep")
        self.sgSweepWin   = QtWidgets.QSpinBox(); self.sgSweepWin.setRange(3, 501); self.sgSweepWin.setSingleStep(2); self.sgSweepWin.setValue(5); self.sgSweepWin.setFixedWidth(55)
        self.sgSweepOrd   = QtWidgets.QSpinBox(); self.sgSweepOrd.setRange(1, 10);  self.sgSweepOrd.setValue(2);      self.sgSweepOrd.setFixedWidth(45)
        self.sgWlCheck    = QtWidgets.QCheckBox("Along λ")
        self.sgWlWin      = QtWidgets.QSpinBox(); self.sgWlWin.setRange(3, 501); self.sgWlWin.setSingleStep(2); self.sgWlWin.setValue(5); self.sgWlWin.setFixedWidth(55)
        self.sgWlOrd      = QtWidgets.QSpinBox(); self.sgWlOrd.setRange(1, 10);  self.sgWlOrd.setValue(2);      self.sgWlOrd.setFixedWidth(45)
        misc_grid.addWidget(self.sgSweepCheck, 1, 1)
        misc_grid.addWidget(self.sgSweepWin,   1, 2)
        misc_grid.addWidget(self.sgSweepOrd,   1, 3)
        misc_grid.addWidget(self.sgWlCheck,    1, 4)
        misc_grid.addWidget(self.sgWlWin,      1, 5)
        misc_grid.addWidget(self.sgWlOrd,      1, 6)
        self.reverseWlCheck = QtWidgets.QCheckBox("Reverse λ")
        misc_grid.addWidget(self.reverseWlCheck, 1, 7)

        # Row 2: Peak fitting (all inline with button)
        misc_grid.addWidget(QtWidgets.QLabel("Peak 1 (nm):"), 2, 0)
        self.peak1GuessEdit = QtWidgets.QLineEdit("855"); self.peak1GuessEdit.setFixedWidth(68)
        misc_grid.addWidget(self.peak1GuessEdit, 2, 1)
        misc_grid.addWidget(QtWidgets.QLabel("Peak 2 (nm):"), 2, 2)
        self.peak2GuessEdit = QtWidgets.QLineEdit("835"); self.peak2GuessEdit.setFixedWidth(68)
        misc_grid.addWidget(self.peak2GuessEdit, 2, 3)
        misc_grid.addWidget(QtWidgets.QLabel("Ref frame:"), 2, 4)
        self.refIndexSpin = QtWidgets.QSpinBox(); self.refIndexSpin.setRange(0, 0); self.refIndexSpin.setFixedWidth(58)
        misc_grid.addWidget(self.refIndexSpin, 2, 5)
        misc_grid.addWidget(QtWidgets.QLabel("Max shift:"), 2, 6)
        self.maxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.maxShiftSpin.setDecimals(2); self.maxShiftSpin.setRange(0.2, 20.0)
        self.maxShiftSpin.setSingleStep(0.2); self.maxShiftSpin.setValue(2.0); self.maxShiftSpin.setFixedWidth(68)
        misc_grid.addWidget(self.maxShiftSpin, 2, 7)
        self.fitPeaksBtn = QtWidgets.QPushButton("Fit Peaks")
        self.fitPeaksBtn.setFixedWidth(82)
        misc_grid.addWidget(self.fitPeaksBtn, 2, 8)

        # Row 3: Dip fitting (all inline with button)
        misc_grid.addWidget(QtWidgets.QLabel("Dip guesses (nm):"), 3, 0)
        self.dipGuessesEdit = QtWidgets.QLineEdit("")
        self.dipGuessesEdit.setPlaceholderText("e.g. 855, 860")
        self.dipGuessesEdit.setFixedWidth(140)
        misc_grid.addWidget(self.dipGuessesEdit, 3, 1, 1, 2)   # colspan 2
        misc_grid.addWidget(QtWidgets.QLabel("Window:"), 3, 3)
        self.dipWindowSpin = QtWidgets.QDoubleSpinBox()
        self.dipWindowSpin.setDecimals(1); self.dipWindowSpin.setRange(0.5, 100.0)
        self.dipWindowSpin.setSingleStep(0.5); self.dipWindowSpin.setValue(8.0); self.dipWindowSpin.setFixedWidth(60)
        misc_grid.addWidget(self.dipWindowSpin, 3, 4)
        misc_grid.addWidget(QtWidgets.QLabel("Max shift:"), 3, 5)
        self.dipMaxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.dipMaxShiftSpin.setDecimals(2); self.dipMaxShiftSpin.setRange(0.1, 20.0)
        self.dipMaxShiftSpin.setSingleStep(0.1); self.dipMaxShiftSpin.setValue(2.0); self.dipMaxShiftSpin.setFixedWidth(60)
        misc_grid.addWidget(self.dipMaxShiftSpin, 3, 6)
        misc_grid.addWidget(QtWidgets.QLabel("Smooth σ:"), 3, 7)
        self.dipSmoothSpin = QtWidgets.QDoubleSpinBox()
        self.dipSmoothSpin.setDecimals(2); self.dipSmoothSpin.setRange(0.0, 50.0)
        self.dipSmoothSpin.setSingleStep(0.1); self.dipSmoothSpin.setValue(0.5); self.dipSmoothSpin.setFixedWidth(58)
        misc_grid.addWidget(self.dipSmoothSpin, 3, 8)
        self.fitDipsBtn = QtWidgets.QPushButton("Fit Dips")
        self.fitDipsBtn.setFixedWidth(82)
        misc_grid.addWidget(self.fitDipsBtn, 3, 9)

        right.addLayout(misc_grid)
        right.addStretch()
        outer.addLayout(right, 3)

        # ── Wire signals ──────────────────────────────────────────────
        self.polCheck.toggled.connect(self._on_pol_toggled)
        self.baselineCheck.toggled.connect(lambda _: self._clear_caches_and_refresh())
        self.bgLoSpin.valueChanged.connect(lambda _: self._clear_caches_and_refresh())
        self.bgHiSpin.valueChanged.connect(lambda _: self._clear_caches_and_refresh())
        self.subCheck.toggled.connect(self.rcCheck.setEnabled)
        self.subCheck.toggled.connect(lambda _: self._sync_flat_enabled())
        self.rcCheck.toggled.connect(lambda _: self._sync_flat_enabled())
        self.subCheck.toggled.connect(lambda _: self._refresh_all())
        self.rcCheck.toggled.connect(lambda _: self._refresh_all())
        self.flatCheck.toggled.connect(lambda _: self._on_flatten_changed())
        self.featRangesEdit.editingFinished.connect(self._on_flatten_changed)
        self.flatOrderSpin.valueChanged.connect(lambda _: self._on_flatten_changed())
        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)
        self.gateAxisCombo.currentTextChanged.connect(self._on_gate_axis_changed)
        self.powerSlider.valueChanged.connect(self._on_power_changed)
        self.frontSlider.valueChanged.connect(self._on_front_changed)
        self.backSlider.valueChanged.connect(self._on_back_changed)
        self.cubeBtn.clicked.connect(self._on_print_cube)
        self.cropApplyBtn.clicked.connect(self._on_apply_crop)
        self.derivMaxSpin.valueChanged.connect(lambda _: self._update_maps())
        for w in (self.sgSweepCheck, self.sgWlCheck, self.sgSweepWin, self.sgSweepOrd, self.sgWlWin, self.sgWlOrd):
            sig = w.toggled if isinstance(w, QtWidgets.QCheckBox) else w.valueChanged
            sig.connect(lambda _: self._update_maps())
        self.reverseWlCheck.toggled.connect(self._on_reverse_wl)
        self.fitPeaksBtn.clicked.connect(self._on_fit_peaks_clicked)
        self.peak1GuessEdit.editingFinished.connect(self._refresh_all)
        self.peak2GuessEdit.editingFinished.connect(self._refresh_all)
        self.fitDipsBtn.clicked.connect(self._on_fit_dips_clicked)
        self.dipGuessesEdit.editingFinished.connect(self._update_maps)
        self.dipSmoothSpin.valueChanged.connect(lambda _: self._on_dip_smooth_changed())
        self.lockAxesCheck.toggled.connect(self._on_lock_axes_toggled)
        self.loadAngleCalibBtn.clicked.connect(self._on_load_angle_calib)
        # liveView signals connected later in _build_ui() after liveView is created

        return box

    def _make_nonpol_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        g    = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)

        g.addWidget(QtWidgets.QLabel("Data folder:"), 0, 0)
        self.npDataLbl = QtWidgets.QLabel("(none)")
        self.npDataLbl.setStyleSheet("font-style:italic; color:#555;")
        g.addWidget(self.npDataLbl, 0, 1)
        btn = QtWidgets.QPushButton("Browse…"); btn.setFixedWidth(80)
        btn.clicked.connect(lambda: self._browse_data(""))
        g.addWidget(btn, 0, 2)

        g.addWidget(QtWidgets.QLabel("Substrate:"), 1, 0)
        self.npSubLbl = QtWidgets.QLabel("(none)")
        self.npSubLbl.setStyleSheet("font-style:italic; color:#888;")
        g.addWidget(self.npSubLbl, 1, 1)
        bf = QtWidgets.QPushButton("File…");   bf.setFixedWidth(70)
        bfo= QtWidgets.QPushButton("Folder…"); bfo.setFixedWidth(80)
        bf.clicked.connect(lambda: self._browse_sub_file(""))
        bfo.clicked.connect(lambda: self._browse_sub_folder(""))
        g.addWidget(bf,  1, 2)
        g.addWidget(bfo, 1, 3)
        g.setColumnStretch(1, 1)
        return page

    def _make_pol_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        g    = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)

        def _row(label_text: str, key: str, row: int, lbl_attr: str, sub_attr: str) -> None:
            g.addWidget(QtWidgets.QLabel(label_text), row, 0)
            lbl = QtWidgets.QLabel("(none)")
            lbl.setStyleSheet("font-style:italic; color:#555;")
            setattr(self, lbl_attr, lbl)
            g.addWidget(lbl, row, 1)
            bd = QtWidgets.QPushButton("Data…");   bd.setFixedWidth(70)
            bd.clicked.connect(lambda checked, k=key: self._browse_data(k))
            g.addWidget(bd, row, 2)

            sub_lbl = QtWidgets.QLabel("(none)")
            sub_lbl.setStyleSheet("font-style:italic; color:#888;")
            setattr(self, sub_attr, sub_lbl)
            g.addWidget(QtWidgets.QLabel("Substrate:"), row + 1, 0)
            g.addWidget(sub_lbl, row + 1, 1)
            bf  = QtWidgets.QPushButton("File…");   bf.setFixedWidth(70)
            bfo = QtWidgets.QPushButton("Folder…"); bfo.setFixedWidth(80)
            bf.clicked.connect(lambda checked, k=key: self._browse_sub_file(k))
            bfo.clicked.connect(lambda checked, k=key: self._browse_sub_folder(k))
            g.addWidget(bf,  row + 1, 2)
            g.addWidget(bfo, row + 1, 3)

        g.addWidget(_bold_label("Left Hand (LH)"), 0, 0, 1, 4)
        _row("  Data folder:", "LH", 1, "lhDataLbl", "lhSubLbl")
        g.addWidget(_bold_label("Right Hand (RH)"), 3, 0, 1, 4)
        _row("  Data folder:", "RH", 4, "rhDataLbl", "rhSubLbl")

        # Display selector
        disp_row = QtWidgets.QHBoxLayout()
        disp_row.setSpacing(12)
        disp_row.addWidget(QtWidgets.QLabel("Display:"))
        self.polDispGroup = QtWidgets.QButtonGroup(self)
        self.polLHBtn   = QtWidgets.QRadioButton("Left Hand");  self.polLHBtn.setChecked(True)
        self.polRHBtn   = QtWidgets.QRadioButton("Right Hand")
        self.polDiffBtn = QtWidgets.QRadioButton("LH − RH")
        for i, rb in enumerate([self.polLHBtn, self.polRHBtn, self.polDiffBtn]):
            self.polDispGroup.addButton(rb, i)
            disp_row.addWidget(rb)
        disp_row.addStretch()
        g.addLayout(disp_row, 6, 0, 1, 5)

        self.polDispGroup.buttonClicked.connect(lambda _: self._refresh_all())
        g.setColumnStretch(1, 1)
        return page

    # ── Helper ────────────────────────────────────────────────────────

    def _current_ch_key(self) -> str:
        """Key of the primary channel to display (ignores DIFF, returns LH for it)."""
        if not self.polCheck.isChecked():
            return ""
        if self.polRHBtn.isChecked():
            return "RH"
        return "LH"

    def _is_diff_mode(self) -> bool:
        return self.polCheck.isChecked() and self.polDiffBtn.isChecked()

    def _primary_ch(self) -> _Channel:
        return self._ch[self._current_ch_key()]

    # ── Browse callbacks ──────────────────────────────────────────────

    # ── ΔR/R0 flatten ─────────────────────────────────────────────────

    def _sync_flat_enabled(self) -> None:
        self.flatCheck.setEnabled(self.subCheck.isChecked() and self.rcCheck.isChecked())

    def _flatten_params(self):
        """(feature_ranges_nm, order) when Flatten ΔR/R0 is active, else None."""
        if not (self.flatCheck.isEnabled() and self.flatCheck.isChecked()):
            return None
        return _parse_ranges_nm(self.featRangesEdit.text()), int(self.flatOrderSpin.value())

    def _on_flatten_changed(self) -> None:
        fl = self._flatten_params()
        self.liveView.set_feature_regions(fl[0] if fl else [], fl is not None)
        self._refresh_all()

    def _on_feat_regions_dragged(self, ranges) -> None:
        self.featRangesEdit.setText(", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in ranges))
        self._refresh_all()

    def _on_load_angle_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load angle→power calibration", "",
            "Excel / CSV (*.xlsx *.xls *.csv);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        try:
            angles, powers_w, extra_a, extra_p = _parse_angle_power_calib(p)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Calibration load error", f"{p.name}\n\n{exc}")
            return

        # Pairs sitting outside the labelled angle/power columns are the usual
        # cause of a calibration that loads but is missing points, so ask rather
        # than dropping them silently.
        if extra_a.size:
            shown = "\n".join(f"    {a:g}°  →  {w * 1e6:.4g} µW"
                              for a, w in zip(extra_a[:10], extra_p[:10]))
            more  = f"\n    … and {extra_a.size - 10} more" if extra_a.size > 10 else ""
            ans = QtWidgets.QMessageBox.question(
                self, "Extra calibration points",
                f"{p.name} has {extra_a.size} numeric pair(s) outside the "
                f"labelled 'angle'/'power' columns:\n\n{shown}{more}\n\n"
                "Include them in the calibration?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if ans == QtWidgets.QMessageBox.Yes:
                angles   = np.concatenate([angles, extra_a])
                powers_w = np.concatenate([powers_w, extra_p])

        angles, powers_w = _finalise_calib(angles, powers_w)
        if angles.size < 2:
            QtWidgets.QMessageBox.critical(
                self, "Calibration load error",
                f"{p.name}\n\nNeed at least 2 distinct angles; got {angles.size}.")
            return

        for ch in self._ch.values():
            ch._angle_calib_angles   = angles
            ch._angle_calib_powers_w = powers_w

        self.angleCalibStatusLbl.setText(f"{angles.size} pts: {p.name}")
        self.angleCalibStatusLbl.setToolTip(
            f"{p}\n{angles.size} points, {angles[0]:g}° – {angles[-1]:g}°, "
            f"{powers_w.min() * 1e6:.4g} – {powers_w.max() * 1e6:.4g} µW\n"
            "Angles outside this range are clamped to the nearest endpoint.")

        # The calibration only shows up on data whose file names carry a stage
        # angle; say so instead of leaving the user wondering why nothing moved.
        loaded = [ch for ch in self._ch.values() if ch.has_data]
        if loaded and not any(ch.angle_mode for ch in loaded):
            QtWidgets.QMessageBox.information(
                self, "Calibration loaded",
                f"Loaded {angles.size} calibration points from {p.name}.\n\n"
                "The data currently loaded has no stage angle in its file names "
                "(expected e.g. PL_A180deg_Vf0_Vb0.asc), so it is treated as a "
                "fixed-power measurement and the calibration will not be applied "
                "to its axes.")
        self._refresh_all()

    def _browse_data(self, key: str) -> None:
        ch  = self._ch[key]
        start = str(ch.data_dir) if ch.data_dir else ""
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data folder", start)
        if not chosen:
            return
        folder = Path(chosen)
        frames = _load_frames(folder)
        if not frames:
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc files found in:\n{folder}")
            return
        ch.clear_cache()
        ch.data_dir = folder
        _build_channel(ch, frames)
        self._update_data_label(key)
        self._reconfigure_sliders()
        self._refresh_all()

    def _browse_sub_file(self, key: str) -> None:
        ch = self._ch[key]
        start = str(ch.data_dir) if ch.data_dir else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select substrate file", start, "Andor ASCII (*.asc);;All files (*)"
        )
        if not path:
            return
        try:
            meta, wl, image = _load_andor_ascii(Path(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        p_n, vf_n, vb_n = _parse_from_name(Path(path).name)
        ch.sub_single = PLFrame(
            path=Path(path), power_w=p_n, v_front=vf_n, v_back=vb_n,
            wavelength_nm=wl, image=image, meta=meta,
            mtime=float(Path(path).stat().st_mtime),
        )
        ch.sub_grid = {}
        self._update_sub_label(key)
        self._refresh_all()

    def _browse_sub_folder(self, key: str) -> None:
        ch = self._ch[key]
        start = str(ch.data_dir) if ch.data_dir else ""
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select substrate folder", start)
        if not chosen:
            return
        folder = Path(chosen)
        _load_sub_folder(ch, folder)
        if not ch.sub_grid:
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc files found in:\n{folder}")
            return
        self._update_sub_label(key, folder=folder)
        self._refresh_all()

    def _update_data_label(self, key: str) -> None:
        ch  = self._ch[key]
        txt = str(ch.data_dir.name) if ch.data_dir else "(none)"
        if   key == "":   self.npDataLbl.setText(txt)
        elif key == "LH": self.lhDataLbl.setText(txt)
        elif key == "RH": self.rhDataLbl.setText(txt)

    def _update_sub_label(self, key: str, folder: Optional[Path] = None) -> None:
        ch = self._ch[key]
        if ch.sub_single is not None:
            txt = f"File: {ch.sub_single.path.name}"
        elif ch.sub_grid:
            n   = len(ch.sub_grid)
            name = folder.name if folder else "?"
            txt = f"Folder: {name} ({n} files)"
        else:
            txt = "(none)"
        if   key == "":   self.npSubLbl.setText(txt)
        elif key == "LH": self.lhSubLbl.setText(txt)
        elif key == "RH": self.rhSubLbl.setText(txt)

    # ── Axis locking ──────────────────────────────────────────────────

    def _on_lock_axes_toggled(self, checked: bool) -> None:
        """Freeze or unfreeze all plot axes and colour scales."""
        for w in (self.liveView, self.mapView, self.derivView, self.derivEView, self.deriv2EView):
            w._axes_locked = checked

        def _toggle(vb):
            if checked:
                vb.disableAutoRange()
            else:
                vb.enableAutoRange()

        _toggle(self.liveView._imgPlot.getViewBox())
        _toggle(self.liveView._vlcPlot.getViewBox())
        _toggle(self.liveView._lcPlot.getViewBox())
        for mw in (self.mapView, self.derivView, self.derivEView, self.deriv2EView):
            _toggle(mw._mapPlot.getViewBox())
            _toggle(mw._vlcPlot.getViewBox())
            _toggle(mw._lcPlot.getViewBox())
        for fw in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
            _toggle(fw._plot.getViewBox())
        _toggle(self.dipPosPlot.getViewBox())

        if not checked:
            self._refresh_all()

    # ── Polarization toggle ───────────────────────────────────────────

    def _on_pol_toggled(self, checked: bool) -> None:
        self.srcStack.setCurrentIndex(1 if checked else 0)
        self._reconfigure_sliders()
        self._refresh_all()

    # ── Crop ──────────────────────────────────────────────────────────

    def _on_apply_crop(self) -> None:
        self._crop = (
            int(self.cropT.value()), int(self.cropB.value()),
            int(self.cropL.value()), int(self.cropR.value()),
        )
        for ch in self._ch.values():
            ch.clear_cache()
        self.liveView.set_crop(*self._crop)
        self._refresh_all()

    def _on_reverse_wl(self) -> None:
        for ch in self._ch.values():
            ch.clear_cache()
        self._refresh_all()

    def _clear_caches_and_refresh(self) -> None:
        for ch in self._ch.values():
            ch.clear_cache()
        self._refresh_all()

    # ── Mode / slider configuration ───────────────────────────────────

    # ── Peak fitting ──────────────────────────────────────────────────

    def _read_peak_guesses(self) -> Tuple[float, float]:
        p1, p2 = 855.0, 835.0
        try:
            v = float(self.peak1GuessEdit.text().strip())
            if np.isfinite(v): p1 = v
        except Exception:
            pass
        try:
            v = float(self.peak2GuessEdit.text().strip())
            if np.isfinite(v): p2 = v
        except Exception:
            pass
        return p1, p2

    def _on_fit_peaks_clicked(self) -> None:
        self._active_overlay = "peak"
        self.liveView.clear_dip_overlay()
        if self._last_x is None or self._last_y_wl is None or self._last_map is None:
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data("No map data", x_label=self._last_x_label or "X")
            return
        p1, p2 = self._read_peak_guesses()
        ref_idx    = int(self.refIndexSpin.value())
        max_shift  = float(self.maxShiftSpin.value())
        fit = fit_two_peak_map_sequential(
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map,  dtype=float),
            ref_idx=ref_idx, peak1_guess_nm=p1, peak2_guess_nm=p2, max_shift_nm=max_shift,
        )
        self._fit_result      = fit
        self._fit_map_version = int(self._map_version)
        self._update_fit_trend_tabs(
            np.asarray(self._last_x,    dtype=float),
            str(self._last_x_label or "X"),
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map,  dtype=float),
        )

    def _update_fit_trend_tabs(self, x, x_label, y_wl, map_data) -> None:
        if map_data.ndim != 2 or y_wl.size != map_data.shape[0]:
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data("No data", x_label=x_label)
            return
        if self._fit_result is None or int(self._fit_map_version) != int(self._map_version):
            msg = "Click 'Fit Peaks' to run / refresh"
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView):
                w.clear_data(msg, x_label=x_label)
            return

        fit = self._fit_result
        c1  = np.asarray(fit.get("center_1_nm", []), dtype=float)
        w1  = np.asarray(fit.get("fwhm_1_nm",   []), dtype=float)
        a1  = np.asarray(fit.get("amp_1",        []), dtype=float)
        c2  = np.asarray(fit.get("center_2_nm", []), dtype=float)
        w2  = np.asarray(fit.get("fwhm_2_nm",   []), dtype=float)
        a2  = np.asarray(fit.get("amp_2",        []), dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = EV_NM / c1; e2 = EV_NM / c2
        e1[~np.isfinite(e1)] = np.nan; e2[~np.isfinite(e2)] = np.nan

        self.fitWidthView.set_data(     x, fwhm_nm_to_mev(c1, w1), fwhm_nm_to_mev(c2, w2),
                                        x_label=x_label, label_1="Peak 1", label_2="Peak 2")
        self.fitEnergyView.set_data(    x, e1, e2,
                                        x_label=x_label, label_1="Peak 1", label_2="Peak 2")
        self.fitIntensityView.set_data( x, a1, a2,
                                        x_label=x_label, label_1="Peak 1", label_2="Peak 2")

    def _update_live_linecut_fit_overlay(self) -> None:
        ch_key = self._current_ch_key()
        ch     = self._ch[ch_key]
        fr     = self._get_frame(ch)
        if fr is None:
            self.liveView.clear_linecut_fit_overlay(); return
        lc, wl, _ = self._corrected_linecut(ch, fr, self.liveView.linecut_row(),
                                            int(self.liveView.linecut_width()))
        if lc is None:
            self.liveView.clear_linecut_fit_overlay(); return
        wl  = np.asarray(wl, dtype=float).ravel()
        lc  = np.asarray(lc, dtype=float).ravel()
        if wl.size != lc.size or wl.size < 8:
            self.liveView.clear_linecut_fit_overlay(); return
        p1g, p2g = self._read_peak_guesses()
        fit = fit_two_peak_linecut(wl, lc, peak1_guess_nm=p1g, peak2_guess_nm=p2g)
        p1  = fit.get("peak_1"); p2 = fit.get("peak_2")
        if p1 is None and p2 is None:
            self.liveView.clear_linecut_fit_overlay(); return
        bl = np.full_like(wl, float(fit.get("offset", 0.0)))
        g1 = np.zeros_like(wl); g2 = np.zeros_like(wl)
        if p1: g1 = _lorentzian(wl, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
        if p2: g2 = _lorentzian(wl, p2["amplitude"], p2["center_nm"], p2["gamma_nm"])
        y_total = bl + g1 + g2
        y1 = (bl + g1) if p1 else None
        y2 = (bl + g2) if p2 else None
        self.liveView.set_linecut_fit_overlay(wl, y_total, y1, y2)

    def _on_mode_changed(self, text: str) -> None:
        self._current_mode = text
        self._reconfigure_sliders()
        self._refresh_all()

    def _on_gate_axis_changed(self, text: str) -> None:
        self._gate_axis = "back" if "back" in text.lower() else "front"
        self._refresh_all()

    def _on_power_changed(self, v: int) -> None:
        self._power_idx = int(v)
        self._refresh_all()

    def _on_front_changed(self, v: int) -> None:
        self._front_idx = int(v)
        ch = self._primary_ch()
        if not ch.is_grid:
            self._sync_back_to_front(ch)   # non-grid: back follows front
        self._refresh_all()

    def _on_back_changed(self, v: int) -> None:
        self._back_idx = int(v)
        self._refresh_all()

    def _cur_power_key(self, ch: _Channel) -> float:
        return _ch_norm_power(ch, float(ch.powers[self._power_idx])) if ch.powers.size else ch.default_power_key

    def _cur_front_key(self, ch: _Channel) -> float:
        return _ch_norm_gate(ch, float(ch.gates[self._front_idx])) if ch.gates.size else ch.default_gate_key

    def _cur_back_key(self, ch: _Channel) -> float:
        return _ch_norm_back(ch, float(ch.backs[self._back_idx])) if ch.backs.size else ch.default_back_key

    def _sync_back_to_front(self, ch: _Channel) -> None:
        """Non-grid data: set the back slider to the back value paired with the
        currently-selected front gate, and keep it read-only."""
        if ch.gates.size == 0 or ch.backs.size == 0:
            return
        fi = max(0, min(self._front_idx, ch.gates.size - 1))
        vb = ch.backs_for_gate[fi] if fi < ch.backs_for_gate.size else np.nan
        if np.isfinite(vb):
            bi = int(np.argmin(np.abs(ch.backs - float(vb))))
            self._back_idx = bi
            self.backSlider.blockSignals(True)
            self.backSlider.setValue(bi)
            self.backSlider.blockSignals(False)

    def _reconfigure_sliders(self) -> None:
        ch = self._primary_ch()
        for s in (self.powerSlider, self.frontSlider, self.backSlider):
            s.blockSignals(True)
        if not ch.has_data:
            for s in (self.powerSlider, self.frontSlider, self.backSlider):
                s.setRange(0, 0)
                s.blockSignals(False)
            self.gateAxisCombo.setEnabled(False)
            return

        # auto-select mode if only one axis is available
        if ch.can_gate_mode and not ch.can_power_mode:
            self._current_mode = self.MODE_GATE
        elif ch.can_power_mode and not ch.can_gate_mode:
            self._current_mode = self.MODE_POWER
        self.modeCombo.blockSignals(True)
        self.modeCombo.setCurrentText(self._current_mode)
        self.modeCombo.blockSignals(False)

        self._power_idx = max(0, min(self._power_idx, max(0, ch.powers.size - 1)))
        self._front_idx = max(0, min(self._front_idx, max(0, ch.gates.size  - 1)))
        self._back_idx  = max(0, min(self._back_idx,  max(0, ch.backs.size   - 1)))
        self.powerSlider.setRange(0, max(0, ch.powers.size - 1)); self.powerSlider.setValue(self._power_idx)
        self.frontSlider.setRange(0, max(0, ch.gates.size  - 1)); self.frontSlider.setValue(self._front_idx)
        self.backSlider.setRange(0, max(0, ch.backs.size   - 1)); self.backSlider.setValue(self._back_idx)
        for s in (self.powerSlider, self.frontSlider, self.backSlider):
            s.blockSignals(False)

        # gate-axis selector only meaningful for grid data in gate mode
        self.gateAxisCombo.setEnabled(ch.is_grid and self._current_mode == self.MODE_GATE)
        if not ch.is_grid:
            self._gate_axis = "front"
        elif ch.gates.size <= 1 and ch.backs.size > 1:
            self._gate_axis = "back"    # only the back gate was swept
        elif ch.backs.size <= 1 and ch.gates.size > 1:
            self._gate_axis = "front"
        self.gateAxisCombo.blockSignals(True)
        self.gateAxisCombo.setCurrentText("Back gate" if self._gate_axis == "back" else "Front gate")
        self.gateAxisCombo.blockSignals(False)

        self.powerSlider.setEnabled(ch.powers.size > 1)
        self.frontSlider.setEnabled(ch.gates.size  > 1)
        # the back gate is user-driven only for grid data; otherwise it follows front
        self.backSlider.setEnabled(ch.is_grid and ch.backs.size > 1)
        if not ch.is_grid:
            self._sync_back_to_front(ch)

    # ── Per-frame helpers ─────────────────────────────────────────────

    def _get_cropped_raw(self, ch: _Channel, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cropped_image, cropped_wl) — cached, no corrections applied."""
        if fr.path in ch._img_cache:
            return ch._img_cache[fr.path]
        img = _apply_crop(fr.image, self._crop)
        wl  = fr.wavelength_nm
        if wl is None:
            wl = np.arange(fr.image.shape[1], dtype=float)
        wl_c = _crop_axis(wl, self._crop, fr.image.shape[1])
        if self.reverseWlCheck.isChecked():
            img = img[:, ::-1]
        ch._img_cache[fr.path] = (img, wl_c)
        return img, wl_c

    def _corrected_linecut(self, ch: _Channel, fr: PLFrame, row_disp: Optional[int],
                           width: int) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
        """(linecut, wl, is_rc) at display row ``row_disp`` (None → middle row).
        With ÷ substrate on, the linecut is (ΣR − ΣR0)/ΣR0 over the rows
        (then flattened when Flatten ΔR/R0 is on)."""
        num, den, wl = self._get_corrected_parts(ch, fr)
        h = num.shape[0]
        row_d   = h // 2 if row_disp is None else max(0, min(int(row_disp), h - 1))
        row_raw = h - 1 - row_d
        lc = _lc_from_terms([(1.0, num, den)], row_raw, width,
                            wl=wl, flatten=self._flatten_params())
        return lc, wl, den is not None

    def _get_corrected_parts(self, ch: _Channel, fr: PLFrame
                             ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Return (corrected_image, substrate_or_None, wl).  The second item is the
        (baseline-corrected) substrate image R0 only when ÷ substrate is active."""
        img, wl = self._get_cropped_raw(ch, fr)
        do_bl  = self.baselineCheck.isChecked()
        do_sub = self.subCheck.isChecked() and ch.has_substrate
        if not do_bl and not do_sub:
            return img, None, wl
        sub_fr = _ch_substrate_for(ch, fr) if do_sub else None
        sub_img = sub_wl = None
        if sub_fr is not None:
            sub_img_raw = _apply_crop(sub_fr.image, self._crop)
            sub_wl_raw  = sub_fr.wavelength_nm
            if sub_wl_raw is None:
                sub_wl_raw = np.arange(sub_fr.image.shape[1], dtype=float)
            sub_wl  = _crop_axis(sub_wl_raw, self._crop, sub_fr.image.shape[1])
            sub_img = sub_img_raw
            if self.reverseWlCheck.isChecked():
                sub_img = sub_img[:, ::-1]
        lo = float(self.bgLoSpin.value())
        hi = float(self.bgHiSpin.value())
        corrected = _corrected_image(
            img, wl, sub_img, sub_wl,
            use_baseline=do_bl, lo=lo, hi=hi,
            use_sub=do_sub and sub_img is not None,
        )
        den = None
        if (do_sub and sub_img is not None and self.rcCheck.isChecked()
                and np.shape(sub_img) == np.shape(img)):
            den = _corrected_image(sub_img, sub_wl, None, None,
                                   use_baseline=do_bl, lo=lo, hi=hi, use_sub=False)
        return corrected, den, wl

    def _get_frame(self, ch: _Channel) -> Optional[PLFrame]:
        """Current frame to display, given the power/front/back sliders."""
        if not ch.has_data:
            return None
        return _frame_at(ch, self._cur_power_key(ch),
                         self._cur_front_key(ch), self._cur_back_key(ch))

    # ── Refresh ───────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._update_single_frame_view()
        self._update_maps()

    def _update_single_frame_view(self) -> None:
        ch_key = self._current_ch_key()
        ch     = self._ch[ch_key]
        fr     = self._get_frame(ch)

        if fr is None and not ch.has_data:
            blank = np.zeros((10, 10), dtype=float)
            self.liveView.set_image(blank)
            self.liveView.set_image_title("No data loaded")
            self.powerLbl.setText("--")
            self.frontLbl.setText("--")
            self.backLbl.setText("--")
            return

        if fr is None:
            blank = np.zeros_like(ch.blank_image, dtype=float)
            self.liveView.set_image(blank, ch.blank_wl)
            self.liveView.set_image_title("(missing frame)")
            return

        fl = self._flatten_params()

        def _rc_img(num_, den_, wl_):
            if den_ is None:
                return num_
            r = _rc_ratio(num_, den_)
            return r if fl is None else _flatten_rc_rows(wl_, r, *fl)

        num, den, wl = self._get_corrected_parts(ch, fr)
        img = _rc_img(num, den, wl)
        terms = [(1.0, num, den)]

        if self._is_diff_mode():
            ch_rh = self._ch["RH"]
            fr_rh = self._get_frame(ch_rh)
            if fr_rh is not None:
                num_rh, den_rh, wl_rh = self._get_corrected_parts(ch_rh, fr_rh)
                img_rh = _rc_img(num_rh, den_rh, wl_rh)
                if img.shape == img_rh.shape:
                    img = img - img_rh
                    terms.append((-1.0, num_rh, den_rh))

        # ΔR/R0: the linecut panel uses the ratio of row sums, like the maps
        rc = any(t[2] is not None for t in terms)
        self.liveView.set_image(img, wl, lc_terms=terms if rc else None,
                                flatten=fl if rc else None)
        self.liveView.set_feature_regions(fl[0] if fl else [], bool(fl) and rc)

        def _f(v): return "--" if v is None or not np.isfinite(float(v)) else f"{float(v):.4g}"
        suffix = " [LH−RH]" if self._is_diff_mode() else (f" [{ch_key}]" if ch_key else "")

        if ch.angle_mode:
            angle_val = float(fr.power_w) if fr.power_w is not None else np.nan
            pw = None
            if ch._angle_calib_angles is not None and np.isfinite(angle_val):
                pw = float(np.interp(angle_val, ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6)
            self.liveView.set_image_title(
                (f"A={_f(angle_val)} °  P={_f(pw)} µW" if pw is not None else f"A={_f(angle_val)} °")
                + f"  Vf={_f(fr.v_front)} V  Vb={_f(fr.v_back)} V{suffix}")
            power_base = f"Angle: {_f(angle_val)} °" + (f"  ({_f(pw)} µW)" if pw is not None else "")
        else:
            p_uW = float(fr.power_w * 1e6) if fr.power_w is not None else np.nan
            self.liveView.set_image_title(
                f"P={_f(p_uW)} µW  Vf={_f(fr.v_front)} V  Vb={_f(fr.v_back)} V{suffix}")
            power_base = f"Power: {_f(p_uW)} µW"

        # Tag which quantity is the swept (map x-axis) variable
        if self._current_mode == self.MODE_GATE and self._gate_axis == "back" and ch.is_grid:
            swept = "back"
        elif self._current_mode == self.MODE_GATE:
            swept = "front"
        else:
            swept = "power"

        def _tag(name: str) -> str:
            return "   ← swept" if name == swept else "   (fixed)"

        self.powerLbl.setText(power_base + _tag("power"))
        self.frontLbl.setText(f"Front gate: {_f(fr.v_front)} V" + _tag("front"))
        self.backLbl.setText(f"Back gate: {_f(fr.v_back)} V" + _tag("back"))

    def _sg_apply(self, m: np.ndarray) -> np.ndarray:
        ws = int(self.sgSweepWin.value()) if self.sgSweepCheck.isChecked() else 0
        os = int(self.sgSweepOrd.value()) if self.sgSweepCheck.isChecked() else 1
        ww = int(self.sgWlWin.value())    if self.sgWlCheck.isChecked()    else 0
        ow = int(self.sgWlOrd.value())    if self.sgWlCheck.isChecked()    else 1
        return m if (ws < 3 and ww < 3) else _sg_smooth_2d(m, ws, os, ww, ow)

    def _build_map(self, ch: _Channel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, wl_ref, map_data) — shape map: (n_wl, n_x)."""
        if not ch.has_data:
            empty = np.zeros((10, 1), dtype=float)
            return np.array([0.0]), np.arange(10, dtype=float), empty

        row_disp_raw = self.liveView.linecut_row()  # None if no cursor set

        p_key  = self._cur_power_key(ch)
        vf_key = self._cur_front_key(ch)
        vb_key = self._cur_back_key(ch)

        gate_back = (self._current_mode == self.MODE_GATE
                     and self._gate_axis == "back" and ch.is_grid)

        if gate_back:
            sweep_vals = ch.backs
            x_axis     = ch.backs
            def _frame_for(val):
                return _frame_at(ch, p_key, vf_key, _ch_norm_back(ch, float(val)))
        elif self._current_mode == self.MODE_GATE:
            sweep_vals = ch.gates
            x_axis     = ch.gates
            def _frame_for(val):
                return _frame_at(ch, p_key, _ch_norm_gate(ch, float(val)), vb_key)
        else:
            sweep_vals = ch.powers
            x_axis     = ch.powers if ch.angle_mode else ch.powers * 1e6
            def _frame_for(val):
                return _frame_at(ch, _ch_norm_power(ch, float(val)), vf_key, vb_key)

        # Find reference wavelength axis from first available frame
        wl_ref = None
        for val in sweep_vals:
            fr = _frame_for(val)
            if fr is not None:
                _, wl_ref = self._get_cropped_raw(ch, fr)
                break
        if wl_ref is None:
            wl_ref = ch.blank_wl

        wl_ref = np.asarray(wl_ref, dtype=float)
        width  = int(self.liveView.linecut_width())
        cols   = []

        for val in sweep_vals:
            fr = _frame_for(val)

            if fr is None:
                cols.append(np.full(wl_ref.shape, np.nan, dtype=float))
                continue
            # linecut_row() is in display space (flipud); converted to raw row inside
            lc, wl, _ = self._corrected_linecut(ch, fr, row_disp_raw, width)
            if lc is None:
                cols.append(np.full(wl_ref.shape, np.nan, dtype=float))
                continue
            cols.append(np.asarray(_resample(wl, lc, wl_ref), dtype=float))

        map_data = np.stack(cols, axis=1) if cols else np.full((wl_ref.size, x_axis.size), np.nan)
        return x_axis, wl_ref, map_data

    def _update_maps(self) -> None:
        ch_key = self._current_ch_key()
        ch     = self._ch[ch_key]

        if not ch.has_data:
            return

        x, wl_nm, m = self._build_map(ch)

        if self._is_diff_mode():
            ch_rh = self._ch["RH"]
            if ch_rh.has_data:
                x_rh, wl_rh, m_rh = self._build_map(ch_rh)
                # resample RH onto LH wavelength axis, then subtract
                if m_rh.shape[1] == m.shape[1]:
                    # resample each column (sweep index) of RH map onto LH wavelength axis
                    m_rh_res = np.column_stack([
                        _resample(wl_rh, m_rh[:, j], wl_nm) for j in range(m_rh.shape[1])
                    ])
                    m = m - m_rh_res

        # Single-wheel (angle) power dependence with a calibration: order the
        # sweep columns by ascending calibrated power rather than by stage angle
        # (angle and power are not monotonically related).
        col_order = None
        if (self._current_mode != self.MODE_GATE and ch.angle_mode
                and ch._angle_calib_angles is not None and ch.powers.size > 1):
            pw_cal = np.interp(ch.powers, ch._angle_calib_angles, ch._angle_calib_powers_w)
            order  = np.argsort(pw_cal, kind="stable")
            if not np.array_equal(order, np.arange(order.size)):
                col_order = order
                m = m[:, col_order]

        self._map_version += 1
        self._last_y_wl   = np.asarray(wl_nm, dtype=float)
        self._last_map    = np.asarray(m, dtype=float)

        m_smooth = self._sg_apply(m)

        sec_axis = None
        if self._current_mode == self.MODE_GATE:
            deriv = _deriv_vs_x(m_smooth, x)
            if self._gate_axis == "back" and ch.is_grid:
                x_label = "Back gate (V)"
                marker  = float(ch.backs[self._back_idx]) if ch.backs.size else 0.0
                self.mapView.setTitle("Linecut vs Back Gate")
                self.derivView.setTitle("dI/dVb")
            else:
                x_label = "Front gate (V)"
                marker  = float(ch.gates[self._front_idx]) if ch.gates.size else 0.0
                if not ch.is_grid:
                    sec_axis = ch.backs_for_gate
                self.mapView.setTitle("Linecut vs Front Gate")
                self.derivView.setTitle("dI/dVf")
            self.mapView.set_secondary_axis(sec_axis, label="Back gate (V)")
            self.derivView.set_secondary_axis(sec_axis, label="Back gate (V)")
        elif ch.angle_mode:
            if ch._angle_calib_angles is not None:
                x_pw = np.interp(ch.powers, ch._angle_calib_angles, ch._angle_calib_powers_w)
                if col_order is not None:      # match the reordered map columns
                    x_pw = x_pw[col_order]
                x      = x_pw * 1e6
                deriv  = _deriv_dlogp(m_smooth, x_pw)
                x_label = "Power (µW)"
                marker  = float(np.interp(ch.powers[self._power_idx], ch._angle_calib_angles,
                                          ch._angle_calib_powers_w) * 1e6) if ch.powers.size else 0.0
                self.mapView.setTitle("Linecut vs Power")
                self.derivView.setTitle("dI/dlog₁₀P")
            else:
                x      = ch.powers.copy()
                deriv  = _deriv_vs_x(m_smooth, x)
                x_label = "Stage A angle (°)"
                marker  = float(ch.powers[self._power_idx]) if ch.powers.size else 0.0
                self.mapView.setTitle("Linecut vs Angle")
                self.derivView.setTitle("dI/dAngle")
            self.mapView.set_secondary_axis(None)
            self.derivView.set_secondary_axis(None)
        else:
            deriv    = _deriv_dlogp(m_smooth, ch.powers)
            x_label  = "Power (µW)"
            marker   = float(ch.powers[self._power_idx] * 1e6) if ch.powers.size else 0.0
            self.mapView.set_secondary_axis(None)
            self.derivView.set_secondary_axis(None)
            self.mapView.setTitle("Linecut vs Power")
            self.derivView.setTitle("dI/dlog₁₀P")

        self._last_x_label = x_label
        self._last_x       = np.asarray(x, dtype=float)

        # convert nm → eV for display (uniform energy grid)
        y_ev, m_ev    = _nm_to_ev(wl_nm, m_smooth)
        _,    d_ev    = _nm_to_ev(wl_nm, deriv)

        # energy-axis derivatives (along y)
        de_ev  = _deriv_vs_ev(m_ev, y_ev)
        d2e_ev = _deriv_vs_ev(de_ev, y_ev)

        max_abs = float(self.derivMaxSpin.value())
        self.derivView.set_manual_symmetric_max(max_abs if max_abs > 0 else None)
        self.derivEView.set_manual_symmetric_max(max_abs if max_abs > 0 else None)
        self.deriv2EView.set_manual_symmetric_max(max_abs if max_abs > 0 else None)

        if self._current_mode == self.MODE_GATE:
            self.derivEView.set_secondary_axis(sec_axis,  label="Back gate (V)")
            self.deriv2EView.set_secondary_axis(sec_axis, label="Back gate (V)")
        else:
            self.derivEView.set_secondary_axis(None)
            self.deriv2EView.set_secondary_axis(None)

        self.mapView.set_map(    x, y_ev, m_ev,   x_label=x_label, y_label="Energy (eV)", symmetric=False)
        self.derivView.set_map(  x, y_ev, d_ev,   x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.derivEView.set_map( x, y_ev, de_ev,  x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.deriv2EView.set_map(x, y_ev, d2e_ev, x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.mapView.set_x_marker(marker)
        self.derivView.set_x_marker(marker)
        self.derivEView.set_x_marker(marker)
        self.deriv2EView.set_x_marker(marker)

        self.refIndexSpin.setRange(0, max(0, int(x.size) - 1))
        if self._active_overlay == "dip":
            self._update_dip_trend_tabs(x, x_label)
            self.liveView.clear_linecut_fit_overlay()
            self._update_live_dip_overlay()
        else:
            self._update_fit_trend_tabs(x, x_label,
                                        np.asarray(wl_nm, dtype=float),
                                        np.asarray(m,     dtype=float))
            self.liveView.clear_dip_overlay()
            self._update_live_linecut_fit_overlay()
        self._update_dip_positions_tab(x, x_label)

    # ── Dip fitting ───────────────────────────────────────────────────

    def _read_dip_guesses(self) -> list:
        vals = []
        for tok in self.dipGuessesEdit.text().replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = float(tok)
                if np.isfinite(v) and v > 0:
                    vals.append(v)
            except ValueError:
                pass
        return vals

    def _on_fit_dips_clicked(self) -> None:
        self._active_overlay = "dip"
        self.liveView.clear_linecut_fit_overlay()
        x_label = str(self._last_x_label or "X")
        if self._last_x is None or self._last_y_wl is None or self._last_map is None:
            self.dipPosPlot.setTitle("No map data loaded")
            return
        guesses = self._read_dip_guesses()
        if not guesses:
            self.dipPosPlot.setTitle("Enter dip guesses (nm) above")
            return
        fit = find_dip_map_sequential(
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map,  dtype=float),
            ref_idx        = int(self.refIndexSpin.value()),
            dip_guesses_nm = np.array(guesses, dtype=float),
            window_nm       = float(self.dipWindowSpin.value()),
            max_shift_nm    = float(self.dipMaxShiftSpin.value()),
            smooth_sigma_nm = float(self.dipSmoothSpin.value()),
        )
        self._dip_fit_result      = fit
        self._dip_fit_map_version = int(self._map_version)
        x = np.asarray(self._last_x, dtype=float)
        self._update_dip_positions_tab(x, x_label)
        self._update_dip_trend_tabs(x, x_label)
        self._update_live_dip_overlay()

    def _update_dip_positions_tab(self, x, x_label: str) -> None:
        if self._dip_fit_result is None or \
                int(self._dip_fit_map_version) != int(self._map_version):
            self.dipPosPlot.setTitle("Click 'Fit Dips' to run / refresh")
            for c in self._dipPosCurves:
                c.setData([], [])
            return
        centers = np.asarray(self._dip_fit_result["centers_nm"])  # (n_dips, n_x)
        n_dips  = centers.shape[0]
        while len(self._dipPosCurves) < n_dips:
            i     = len(self._dipPosCurves)
            color = _DIP_COLORS[i % len(_DIP_COLORS)]
            c = self.dipPosPlot.plot(
                pen=pg.mkPen(color, width=1.2), symbol="o",
                symbolSize=4, symbolBrush=color, symbolPen=None,
                name=f"Dip {i + 1}")
            self._dipPosCurves.append(c)
        for c in self._dipPosCurves:
            c.setData([], [])
        for i in range(n_dips):
            c_nm = centers[i, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                c_ev = EV_NM / c_nm
            c_ev[~np.isfinite(c_ev)] = np.nan
            valid = np.isfinite(x) & np.isfinite(c_ev)
            if valid.any():
                self._dipPosCurves[i].setData(x[valid], c_ev[valid])
        self.dipPosPlot.setLabel("bottom", x_label)
        self.dipPosPlot.setTitle("Dip positions vs sweep")

    def _update_dip_trend_tabs(self, x, x_label: str) -> None:
        """Populate FitTrend tabs from dip fit result (energy + intensity only)."""
        msg = "Click 'Fit Dips' to run / refresh"
        if self._dip_fit_result is None or \
                int(self._dip_fit_map_version) != int(self._map_version):
            self.fitWidthView.clear_data(msg, x_label=x_label)
            self.fitEnergyView.clear_data(msg, x_label=x_label)
            self.fitIntensityView.clear_data(msg, x_label=x_label)
            return

        centers = np.asarray(self._dip_fit_result["centers_nm"])  # (n_dips, n_x)
        amps    = np.asarray(self._dip_fit_result["amps"])         # (n_dips, n_x)

        def _row(arr, i):
            if i < arr.shape[0]:
                return arr[i, :].copy()
            return np.full(x.size, np.nan)

        c1 = _row(centers, 0); c2 = _row(centers, 1)
        a1 = _row(amps,    0); a2 = _row(amps,    1)

        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = EV_NM / c1; e2 = EV_NM / c2
        e1[~np.isfinite(e1)] = np.nan
        e2[~np.isfinite(e2)] = np.nan

        self.fitWidthView.clear_data("N/A in dip mode", x_label=x_label)
        self.fitEnergyView.set_data(x, e1, e2,
                                    x_label=x_label,
                                    label_1="Dip 1", label_2="Dip 2")
        self.fitIntensityView.set_data(x, a1, a2,
                                       x_label=x_label,
                                       label_1="Dip 1", label_2="Dip 2")

    def _on_dip_smooth_changed(self) -> None:
        if self._active_overlay == "dip":
            self._update_live_dip_overlay()

    def _update_live_dip_overlay(self) -> None:
        guesses = self._read_dip_guesses()
        if not guesses:
            self.liveView.clear_dip_overlay()
            return
        ch_key = self._current_ch_key()
        ch     = self._ch[ch_key]
        fr     = self._get_frame(ch)
        if fr is None:
            self.liveView.clear_dip_overlay()
            return
        lc, wl, _ = self._corrected_linecut(ch, fr, self.liveView.linecut_row(),
                                            int(self.liveView.linecut_width()))
        if lc is None:
            self.liveView.clear_dip_overlay()
            return
        wl = np.asarray(wl, dtype=float).ravel()
        lc = np.asarray(lc, dtype=float).ravel()
        if wl.size != lc.size or wl.size < 8:
            self.liveView.clear_dip_overlay()
            return
        fit = find_dip_local_min(
            wl, lc, np.array(guesses, dtype=float),
            window_nm       = float(self.dipWindowSpin.value()),
            max_shift_nm    = float(self.dipMaxShiftSpin.value()),
            smooth_sigma_nm = float(self.dipSmoothSpin.value()),
        )
        positions = [r["center_nm"] if r is not None else None
                     for r in fit["dips"]]
        if not any(p is not None for p in positions):
            self.liveView.clear_dip_overlay()
            return
        self.liveView.set_dip_vlines(positions,
                                     x_smooth=fit.get("x_smooth"),
                                     y_smooth=fit.get("y_smooth"))

    # ── 3-D cube ──────────────────────────────────────────────────────

    def _baseline_linecut(self, lc: np.ndarray, wl: np.ndarray) -> np.ndarray:
        """Baseline-correct a 1-D linecut by subtracting its mean over the
        [bgLo, bgHi] nm range (same convention as the 2-D baseline correction;
        summing baseline-corrected rows is equivalent to baselining the sum)."""
        lc = np.asarray(lc, dtype=float).ravel()
        wl = np.asarray(wl, dtype=float).ravel()
        if lc.size != wl.size or lc.size == 0:
            return lc
        lo = float(self.bgLoSpin.value())
        hi = float(self.bgHiSpin.value())
        mask = (wl >= lo) & (wl <= hi)
        if mask.any():
            return lc - float(np.nanmean(lc[mask]))
        return lc

    def _collect_cube_data(self, ch: _Channel):
        """Build (fronts, backs, wl_ref, vol[nf, nb, nwl]) of baseline-corrected
        linecuts at the current fixed power and linecut row."""
        fronts, backs = ch.gates, ch.backs
        if fronts.size == 0 or backs.size == 0:
            return None
        p_key    = self._cur_power_key(ch)
        row_disp = self.liveView.linecut_row()
        width    = int(self.liveView.linecut_width())

        wl_ref = None
        for vf in fronts:
            for vb in backs:
                fr = _frame_at(ch, p_key, _ch_norm_gate(ch, float(vf)), _ch_norm_back(ch, float(vb)))
                if fr is not None:
                    _, wl_ref = self._get_cropped_raw(ch, fr)
                    break
            if wl_ref is not None:
                break
        if wl_ref is None:
            return None
        wl_ref = np.asarray(wl_ref, dtype=float)
        vol = np.full((fronts.size, backs.size, wl_ref.size), np.nan, dtype=float)

        for i, vf in enumerate(fronts):
            for j, vb in enumerate(backs):
                fr = _frame_at(ch, p_key, _ch_norm_gate(ch, float(vf)), _ch_norm_back(ch, float(vb)))
                if fr is None:
                    continue
                lc, wl, is_rc = self._corrected_linecut(ch, fr, row_disp, width)
                if lc is None:
                    continue
                if not is_rc:  # ΔR/R0 is already a ratio of baselined sums
                    lc = self._baseline_linecut(lc, wl)
                vol[i, j, :] = np.asarray(_resample(wl, lc, wl_ref), dtype=float)
        return fronts.copy(), backs.copy(), wl_ref, vol

    def _on_print_cube(self) -> None:
        ch = self._primary_ch()
        if not ch.has_data:
            QtWidgets.QMessageBox.information(self, "Print cube", "Load data first.")
            return
        try:
            import pyqtgraph.opengl  # noqa: F401
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Print cube",
                f"The 3-D cube needs PyOpenGL:\n{exc}\n\nInstall with:  pip install PyOpenGL")
            return
        data = self._collect_cube_data(ch)
        if data is None:
            QtWidgets.QMessageBox.warning(self, "Print cube",
                                          "No frames available for the current power.")
            return
        fronts, backs, wl, vol = data
        if not np.isfinite(vol).any():
            QtWidgets.QMessageBox.warning(self, "Print cube", "All linecuts are empty.")
            return
        p_key = self._cur_power_key(ch)
        ptxt  = f"angle {p_key:g}" if ch.angle_mode else f"P={p_key * 1e6:.4g} µW"
        row   = self.liveView.linecut_row()
        title = f"PL cube — {ptxt}, linecut row {row}"
        try:
            self._cube_win = _CubeWindow(fronts, backs, wl, vol, title=title,
                                         cmap_name="viridis", parent=self)
            self._cube_win.show()
            self._cube_win.raise_()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Print cube",
                                          f"Could not open the 3-D view:\n{exc}")


# ── tiny helper ───────────────────────────────────────────────────────

def _bold_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(f"<b>{text}</b>")
    return lbl


# ══════════════════════════════════════════════════════════════════════
# 3-D cube window (front gate × back gate × wavelength, colour = intensity)
# ══════════════════════════════════════════════════════════════════════

class _CubeWindow(QtWidgets.QMainWindow):
    """Volumetric render of baseline-corrected linecuts over the gate grid.

    x = front gate, y = back gate, z = wavelength, colour = intensity.
    Opacity ramps from transparent at the (corrected) baseline to opaque at the
    highest intensity, so the baseline background disappears.
    """

    _L = 10.0  # visual edge length per axis

    def __init__(self, fronts, backs, wl, vol, *, title: str = "",
                 cmap_name: str = "viridis", parent=None):
        super().__init__(parent)
        import pyqtgraph.opengl as gl
        self._gl   = gl
        self.fronts = np.asarray(fronts, dtype=float).ravel()
        self.backs  = np.asarray(backs,  dtype=float).ravel()
        self.wl     = np.asarray(wl,     dtype=float).ravel()
        self.vol    = np.asarray(vol,    dtype=float)          # (nf, nb, nz)
        self._cmap_name = cmap_name
        self._vol_item  = None
        self.setWindowTitle(title or "PL cube")
        self.resize(920, 780)
        self._build_ui()
        self._add_static_items()
        self._rebuild_volume()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Opacity:"))
        self.opacitySlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacitySlider.setRange(1, 100)
        self.opacitySlider.setValue(45)
        self.opacitySlider.setFixedWidth(170)
        ctrl.addWidget(self.opacitySlider)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Transparent below:"))
        self.threshSpin = QtWidgets.QDoubleSpinBox()
        self.threshSpin.setRange(0.0, 99.0)
        self.threshSpin.setValue(2.0)
        self.threshSpin.setSuffix(" %")
        self.threshSpin.setFixedWidth(70)
        self.threshSpin.setToolTip("Intensity (as % of the colour-scale max) below\n"
                                   "which voxels are fully transparent.")
        ctrl.addWidget(self.threshSpin)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Opacity γ:"))
        self.gammaSpin = QtWidgets.QDoubleSpinBox()
        self.gammaSpin.setRange(0.1, 5.0)
        self.gammaSpin.setSingleStep(0.1)
        self.gammaSpin.setValue(1.5)
        self.gammaSpin.setFixedWidth(60)
        self.gammaSpin.setToolTip("Opacity curve: >1 suppresses low intensities\n"
                                  "(more transparent), <1 lifts them.")
        ctrl.addWidget(self.gammaSpin)
        ctrl.addStretch()
        v.addLayout(ctrl)

        self.glView = self._gl.GLViewWidget()
        self.glView.setBackgroundColor("k")
        v.addWidget(self.glView, 1)

        def _rng(a):
            a = a[np.isfinite(a)]
            return (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
        f0, f1 = _rng(self.fronts); b0, b1 = _rng(self.backs); w0, w1 = _rng(self.wl)
        self.infoLbl = QtWidgets.QLabel(
            f"X front gate {f0:g}…{f1:g} V   |   "
            f"Y back gate {b0:g}…{b1:g} V   |   "
            f"Z wavelength {w0:.1f}…{w1:.1f} nm   |   colour = intensity")
        self.infoLbl.setStyleSheet("color:#aaa; font-size:8pt;")
        v.addWidget(self.infoLbl)
        self.setCentralWidget(central)

        self.opacitySlider.valueChanged.connect(lambda _: self._rebuild_volume())
        self.threshSpin.valueChanged.connect(lambda _: self._rebuild_volume())
        self.gammaSpin.valueChanged.connect(lambda _: self._rebuild_volume())

    def _scales(self):
        nf, nb, nz = self.vol.shape
        L = self._L
        sx = L / max(1, nf - 1) if nf > 1 else L
        sy = L / max(1, nb - 1) if nb > 1 else L
        sz = L / max(1, nz - 1) if nz > 1 else L
        return sx, sy, sz

    def _add_static_items(self) -> None:
        gl = self._gl
        L  = self._L
        ax = gl.GLAxisItem()
        ax.setSize(L, L, L)
        self.glView.addItem(ax)

        def _lbl(text, pos, color=(200, 200, 200, 255)):
            try:
                t = gl.GLTextItem(pos=np.array(pos, dtype=float), text=text, color=color)
                self.glView.addItem(t)
            except Exception:
                pass

        def _rng(a):
            a = a[np.isfinite(a)]
            return (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
        f0, f1 = _rng(self.fronts); b0, b1 = _rng(self.backs); w0, w1 = _rng(self.wl)
        _lbl(f"Vf {f1:g}", (L, 0, 0), (255, 120, 120, 255))
        _lbl(f"Vb {b1:g}", (0, L, 0), (120, 255, 120, 255))
        _lbl(f"λ {w1:.0f}", (0, 0, L), (120, 160, 255, 255))
        _lbl(f"Vf {f0:g} / Vb {b0:g} / λ {w0:.0f}", (0, 0, 0))

        self.glView.opts["center"] = pg.Vector(L / 2, L / 2, L / 2)
        self.glView.setCameraPosition(distance=2.4 * L, elevation=22, azimuth=-60)

    def _rebuild_volume(self) -> None:
        gl = self._gl
        nf, nb, nz = self.vol.shape
        finite = self.vol[np.isfinite(self.vol)]
        if finite.size == 0:
            return
        vhi = float(np.nanpercentile(finite, 99.5))
        if not np.isfinite(vhi) or vhi <= 0:
            vhi = float(np.nanmax(finite))
        if not np.isfinite(vhi) or vhi <= 0:
            vhi = 1.0

        # baseline (≈0 after correction) → 0 → transparent; peak → 1 → opaque
        norm = np.clip(np.nan_to_num(self.vol, nan=0.0) / vhi, 0.0, 1.0)
        thr = float(self.threshSpin.value()) / 100.0
        if thr > 0:
            norm = np.clip((norm - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
        gamma = float(self.gammaSpin.value())
        alpha = norm ** gamma

        lut = _pg_cmap(self._cmap_name).getLookupTable(0.0, 1.0, 256)  # (256, 3) ubyte
        idx = np.clip(norm * 255.0, 0, 255).astype(int)
        rgba = np.empty((nf, nb, nz, 4), dtype=np.ubyte)
        rgba[..., 0] = lut[idx, 0]
        rgba[..., 1] = lut[idx, 1]
        rgba[..., 2] = lut[idx, 2]
        gain = float(self.opacitySlider.value()) / 100.0
        rgba[..., 3] = np.clip(alpha * 255.0 * gain, 0, 255).astype(np.ubyte)

        if self._vol_item is not None:
            self.glView.removeItem(self._vol_item)
            self._vol_item = None
        vi = gl.GLVolumeItem(rgba, smooth=True, glOptions="translucent")
        sx, sy, sz = self._scales()
        vi.scale(sx, sy, sz)
        self.glView.addItem(vi)
        self._vol_item = vi


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#   TRPL  (SPAD23 or SPAD93)  —  everything below is new in this file
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

import csv
import threading
import time
import warnings
from collections import OrderedDict

# ── SPAD geometry: SPAD23 or SPAD93, picked per TRPL file from its number of pixel columns ──
class SpadGeometry:
    """Pixel layout of one detector (coordinates in µm, the 2D-fit pitches, how the map draws it)."""

    PICK_RADIUS_UM = 11.5        # half the 23 µm nearest-neighbour distance (both detectors)

    def __init__(self, name, x_um, y_um, center_pixel, pitch_x, pitch_y, invert_y, pad_um, dot_um=None):
        self.name, self.center_pixel = name, int(center_pixel)
        self.x_um, self.y_um = np.asarray(x_um, float), np.asarray(y_um, float)
        self.num_pixels = int(self.x_um.size)
        self.pitch_x, self.pitch_y = float(pitch_x), float(pitch_y)   # 2D fit: bounds and start widths
        self.invert_y = bool(invert_y)
        self.x_range = (float(self.x_um.min()) - pad_um, float(self.x_um.max()) + pad_um)
        self.y_range = (float(self.y_um.min()) - pad_um, float(self.y_um.max()) + pad_um)
        self.dot_um = dot_um     # None: fixed-size dots (screen pixels), else dots scale with the view

    @property
    def label(self) -> str:
        return f"{self.name}, {self.num_pixels} px"


def _spad23_layout():
    """Same as spad23/spad23_geometry.py: rows 0-4 | 5-8 | 9-13 | 14-17 | 18-22; x = 23 µm * k
    (+11.5 µm in the 4-pixel rows), y = 19.92 µm * row, row 0 drawn at the top."""
    rows = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17), (18, 19, 20, 21, 22))
    x, y = np.zeros(23), np.zeros(23)
    for r, ids in enumerate(rows):
        for k, p in enumerate(ids):
            x[p] = (0.5 * 23.0 if len(ids) < 5 else 0.0) + k * 23.0
            y[p] = r * 19.92
    return x, y


def _spad93_layout():
    """Same as spad93/spad93_geometry.py: 17 columns 19.92 µm apart, active sites 23 µm apart along y
    (odd columns shifted by half a site), numbered column by column from the bottom; y points up."""
    active = ((3, 5), (2, 5), (2, 6), (2, 5), (2, 6), (1, 6), (1, 7), (0, 7), (0, 8),
              (0, 7), (1, 7), (1, 6), (2, 6), (2, 5), (2, 6), (2, 5), (3, 5))
    x, y = [], []
    for col, (first, last) in enumerate(active):
        for site in range(first, last + 1):
            x.append(col * 19.92)
            y.append((site + 0.5 * (col % 2)) * 23.0)
    return np.array(x), np.array(y)


SPAD23_GEOM = SpadGeometry("SPAD23", *_spad23_layout(), center_pixel=11, pitch_x=23.0, pitch_y=19.92,
                           invert_y=True, pad_um=15.0)
SPAD93_GEOM = SpadGeometry("SPAD93", *_spad93_layout(), center_pixel=46, pitch_x=19.92, pitch_y=23.0,
                           invert_y=False, pad_um=15.0, dot_um=19.0)
GEOMETRIES = {g.num_pixels: g for g in (SPAD23_GEOM, SPAD93_GEOM)}


def geometry_for(n_pixels: int) -> SpadGeometry:
    g = GEOMETRIES.get(int(n_pixels))
    if g is None:
        raise ValueError(f"{n_pixels} pixel columns: expected 23 (SPAD23) or 93 (SPAD93)")
    return g

MIN_FIT_PIXELS_2D = 6
MIN_SNR           = 3.0          # amplitude / its standard error below this = "low signal" (flagged)
UM2_NS_TO_CM2_S   = 10.0         # 1 µm²/ns = 10 cm²/s
TRPL_CACHE_FILES  = 48

# ── pump 2 (filter flipper) ───────────────────────────────────────────
P2_OFF, P2_ON, P2_NONE = "off", "on", "none"
P2_ORDER  = (P2_OFF, P2_ON, P2_NONE)
P2_LABELS = {P2_OFF: "Pump 2 off", P2_ON: "Pump 2 on", P2_NONE: "No pump-2 flag"}
_P2_TAG_RE = re.compile(r"_(?P<tag>P2off|P2on)(?:_\d+)?$")
_RETAKE_RE = re.compile(r"_\d+$")
_BARE_POWER_RE = re.compile(
    r"(?:^|_)(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?P<unit>nW|uW|mW|W)(?=_|\.|$)")


# ══════════════════════════════════════════════════════════════════════
# Folder scanning: PL / TRPL files per pump-2 state
# ══════════════════════════════════════════════════════════════════════
#   <base>[_Vf..][_Vb..][_P2off|_P2on][_N].asc   and   TRPL_<same>.csv
# The .asc header also says "pump2: on/off" and the flipper position; the
# TRPL csv has no header, so it takes its name tag or its PL file's state.

def _pump2_state(name: str) -> Optional[str]:
    m = _P2_TAG_RE.search(Path(name).stem)
    if m is None:
        return None
    return P2_ON if m.group("tag") == "P2on" else P2_OFF


def _pair_stem(name: str) -> str:
    """Name shared by the PL and TRPL files (and the off / on files) of one sweep point."""
    stem = Path(name).stem
    if stem.startswith("TRPL_"):
        stem = stem[len("TRPL_"):]
    m = _P2_TAG_RE.search(stem)
    return stem[:m.start()] if m is not None else _RETAKE_RE.sub("", stem)


def _header_pump2(meta: dict) -> Optional[str]:
    v = str(meta.get("pump2", "")).strip().lower()
    return P2_ON if v in ("on", "true", "1") else P2_OFF if v in ("off", "false", "0") else None


class FolderScan:
    def __init__(self, folder: Path):
        self.folder  = Path(folder)
        self.asc:  Dict[str, List[Path]]      = {}   # state -> .asc files
        self.trpl: Dict[str, Dict[str, Path]] = {}   # state -> stem -> TRPL csv
        self.flipper: Dict[str, dict]         = {}   # state -> flipper header fields
        self.session: dict                    = {}

    def states(self) -> List[str]:
        return [s for s in P2_ORDER if self.asc.get(s) or self.trpl.get(s)]

    def trpl_paths(self) -> List[Path]:
        return [p for s in P2_ORDER for p in self.trpl.get(s, {}).values()]

    def describe(self, state: str) -> str:
        n_pl, n_tr = len(self.asc.get(state, [])), len(self.trpl.get(state, {}))
        text = f"{n_pl} PL, {n_tr} TRPL"
        if n_pl and n_tr:
            stems = {_pair_stem(p.name) for p in self.asc[state]}
            text += f" ({len(stems & set(self.trpl[state]))} matched)"
        fl = self.flipper.get(state) or {}
        if fl.get("flipper_position"):
            text += f"; flipper at position {fl['flipper_position']}"
            if fl.get("flipper_serial"):
                text += f" (S/N {fl['flipper_serial']})"
        return text


def scan_folder(folder: Path) -> FolderScan:
    scan = FolderScan(folder)
    scan.session = _read_session(scan.folder)
    stem_state: Dict[str, str] = {}
    for path in sorted(scan.folder.glob("*.asc")):
        state = _pump2_state(path.name)
        if state is None or state not in scan.flipper:
            try:
                meta = _parse_header(path)
            except OSError:
                meta = {}
            if state is None:
                state = _header_pump2(meta) or P2_NONE
            if state != P2_NONE and state not in scan.flipper:
                scan.flipper[state] = {k: meta[k] for k in ("flipper_position", "flipper_serial") if k in meta}
        scan.asc.setdefault(state, []).append(path)
        stem_state[_pair_stem(path.name)] = state
    for path in sorted(scan.folder.glob("TRPL_*.csv")):
        stem  = _pair_stem(path.name)
        state = _pump2_state(path.name) or stem_state.get(stem, P2_NONE)
        old   = scan.trpl.setdefault(state, {}).get(stem)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:      # retakes: newest wins
            scan.trpl[state][stem] = path
    return scan


def _read_session(folder: Path) -> dict:
    """TRPL histogram range / reference and alignment from the measuring GUI's session.md."""
    out: dict = {}
    try:
        text = (Path(folder) / "session.md").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        body = line.lstrip("- ").strip()
        if ":" not in body:
            continue
        key, val = (s.strip() for s in body.split(":", 1))
        low = key.lower()
        if low.startswith("histogram start/end"):
            try:
                vals = [float(v) for v in val.split(",")]
            except ValueError:
                continue
            if len(vals) >= 2 and vals[1] > vals[0]:
                out["hist_ns"] = (vals[0], vals[1])
            if len(vals) >= 3:
                out["ref_ns"] = vals[2]
            out["hist_after_t0"] = "t0 = 0" in low      # newer GUIs: times after time zero (else raw)
        elif low == "reference":
            out["ref_auto"] = not val.lower().startswith("manual")
        elif low.startswith("time-zero alignment"):
            out["align"] = val.lower().startswith("on")
        elif low.startswith("di/dp"):
            out["didp"] = val
        elif low.startswith("acquisition (s)"):
            try:
                out["acq_s"] = float(val)
            except ValueError:
                pass
        elif low.startswith("background subtracted before the 2d fit"):
            v = val.lower()
            out["background"] = "dark_ref" if "dark" in v else "pre_t0" if "pre-t0" in v else "none"
    return out


DYN_LOG_NAME = "dynamic_trpl_log.csv"   # the measuring GUI's dynamic TRPL time (not "trpl_*": TRPL_*.csv globs ignore case)


def read_dynamic_log(folder: Path) -> Dict[str, dict]:
    """``{TRPL file name: {total_s, reached, peak, target, n_chunks, reason}}`` from the folder's
    dynamic_trpl_log.csv ({} = fixed TRPL time: every file has session.md's 'Acquisition (s)')."""
    out: Dict[str, dict] = {}
    path = Path(folder) / DYN_LOG_NAME
    if not path.is_file():
        return out
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("file") or "").strip()
                if not name:
                    continue                     # aborted spot: no summed file
                try:
                    out[name] = {"total_s": float(row["total_s"]), "reached": row.get("reached", "") == "yes",
                                 "peak": float(row.get("peak_counts") or "nan"),
                                 "target": float(row.get("target_counts") or "nan"),
                                 "n_chunks": int(row.get("n_chunks") or 0), "reason": row.get("reason", ""),
                                 "pump2": row.get("pump2", "")}
                except (KeyError, ValueError):
                    continue
    except OSError:
        return {}
    return out


_TRPL_PATH_KEY = "_trpl_path"


def _load_state_frames(scan: FolderScan, state: str) -> List[PLFrame]:
    """PL frames of one pump-2 state; 1×1 stub frames from the TRPL names if there is no PL."""
    paths = scan.asc.get(state, [])
    if paths:
        return _load_frames(scan.folder, paths)
    frames = []
    for path in sorted(scan.trpl.get(state, {}).values()):
        power, vf, vb = _parse_from_name(path.name)
        angle = _parse_angle_from_name(path.name)
        if power is None and angle is None:
            m = _BARE_POWER_RE.search(path.name)
            if m and _tok(m.group("val")) is not None:
                power = _tok(m.group("val")) * {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9}[m.group("unit").lower()]
        frames.append(PLFrame(path=path, power_w=power if power is not None else angle, v_front=vf, v_back=vb,
                              wavelength_nm=None, image=np.zeros((1, 1)), meta={_TRPL_PATH_KEY: path},
                              mtime=float(path.stat().st_mtime), angle_deg=angle,
                              vf_named=vf is not None, vb_named=vb is not None))
    return frames


def _trpl_path_for(scan: Optional[FolderScan], state: str, fr: Optional[PLFrame]) -> Optional[Path]:
    if fr is None:
        return None
    if isinstance(fr.meta, dict) and fr.meta.get(_TRPL_PATH_KEY) is not None:
        return Path(fr.meta[_TRPL_PATH_KEY])
    return None if scan is None else scan.trpl.get(state, {}).get(_pair_stem(fr.path.name))


# ══════════════════════════════════════════════════════════════════════
# TRPL files
# ══════════════════════════════════════════════════════════════════════

class DarkReference:
    """Dark reference (beam blocked) saved by the measuring GUI as DARK_TRPL_*.csv.

    Same as spad23/spad93_analysis.DarkReference: dark counts are uniform in time, so they are kept per
    pixel as counts per bin per second and scaled to the data's acquisition time and bin width.
    """

    def __init__(self, rate_per_bin, bin_width_ps: float, measurement_ms: float, n_bins: int, path: str,
                 taken_at: str = "", detector: str = ""):
        self.rate_per_bin = np.asarray(rate_per_bin, dtype=np.float64)
        self.bin_width_ps, self.measurement_ms, self.n_bins = float(bin_width_ps), float(measurement_ms), int(n_bins)
        self.path, self.taken_at, self.detector = str(path), taken_at, detector

    @property
    def num_pixels(self) -> int:
        return int(self.rate_per_bin.size)

    def per_bin(self, acq_s: float, bin_ps: float) -> np.ndarray:
        """Expected dark counts per bin of width ``bin_ps`` in an acquisition of ``acq_s`` seconds."""
        return self.rate_per_bin * float(acq_s) * float(bin_ps) / self.bin_width_ps

    def describe(self) -> str:
        cps = self.rate_per_bin * max(self.n_bins, 1)
        return (f"{Path(self.path).name}: {self.detector or 'dark'}, {self.measurement_ms / 1000.0:g} s, taken "
                f"{self.taken_at.replace('T', ' ')}, {np.mean(cps):.3g} cps per pixel on average")


def _csv_header(path) -> dict:
    """``# key: value`` rows at the top of a csv (values as float / bool where possible)."""
    out = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" not in body:
                continue
            k, v = (s.strip() for s in body.split(":", 1))
            if v in ("True", "False"):
                out[k] = v == "True"
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def load_dark_reference(path) -> DarkReference:
    meta = _csv_header(path)
    if meta.get("dark_reference") is not True:
        raise ValueError("not a dark reference (no 'dark_reference: True' header row)")
    ms = meta.get("measurement_ms")
    if not isinstance(ms, float) or not ms > 0:
        raise ValueError("no acquisition time (measurement_ms) in the header")
    data = load_trpl_csv(path)
    bin_ps = meta.get("bin_width_ps")
    if not isinstance(bin_ps, float):
        bin_ps = float(np.median(np.diff(data.time_ns))) * 1000.0
    rate = data.counts.astype(np.float64).mean(axis=0) / (ms / 1000.0)
    return DarkReference(rate, bin_ps, ms, data.counts.shape[0], str(path), str(meta.get("taken_at", "")),
                         str(meta.get("detector", "")))


def dark_compare_text(pre_t0, dark, mask=None) -> str:
    """How much of the pre-t0 level is not dark (signal left over from earlier pulses)."""
    if pre_t0 is None or dark is None or np.size(pre_t0) != np.size(dark):
        return ""
    p, d = np.asarray(pre_t0, float), np.asarray(dark, float)
    if mask is not None:
        p, d = p[mask], d[mask]
    sp, sd = float(p.sum()), float(d.sum())
    if not sp > 0:
        return ""
    frac = (sp - sd) / sp
    levels = f"pre-t0 level {sp / p.size:.3g} vs dark {sd / d.size:.3g} counts/bin per pixel"
    below = int(np.sum(p < 0.8 * d))
    if frac < -0.1 or below > 0.5 * p.size:
        return (f"{levels}: the pre-t0 level is BELOW the dark in {below} px - stale dark reference "
                "or a different acquisition time / bin width than assumed?")
    return f"{levels}: {100.0 * max(frac, 0.0):.0f}% of the pre-t0 level is not dark (signal from earlier pulses)"


class TrplData:
    __slots__ = ("path", "time_ns", "counts", "geom")

    def __init__(self, path: str, time_ns: np.ndarray, counts: np.ndarray):
        self.path    = path
        self.time_ns = time_ns       # (bins,)
        self.counts  = counts        # (bins, pixels) int: 23 (SPAD23) or 93 (SPAD93)
        self.geom    = geometry_for(counts.shape[1])


def load_trpl_csv(path) -> TrplData:
    """TRPL csv: time_ns (or time_ps), pixel_0 … pixel_22 (SPAD23) or … pixel_92 (SPAD93).

    The header may be a '#' line.  The number of pixel columns picks the detector.
    """
    import pandas as pd
    names = None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if "," in body and ("pixel_0" in body or "time_" in body) and ":" not in body:
                names = [c.strip() for c in body.split(",")]
    df = pd.read_csv(path, comment="#", encoding="utf-8-sig", skipinitialspace=True,
                     header=None if names else "infer", names=names)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c.startswith("time")), None)
    if tcol is None:
        raise ValueError(f"no time column in {Path(path).name}")
    scale = 1e-3 if "ps" in tcol else 1.0
    n_pix = sum(1 for c in df.columns if re.fullmatch(r"pixel_\d+", c))
    try:
        geometry_for(n_pix)
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: {exc}") from None
    pcols = [f"pixel_{i}" for i in range(n_pix)]
    missing = [c for c in pcols if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing[:3]}")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df[tcol].notna()]
    t = df[tcol].to_numpy(dtype=np.float64) * scale
    c = df[pcols].fillna(0).to_numpy(dtype=np.float64).round().astype(np.int64)
    if t.size < 2:
        raise ValueError(f"{Path(path).name}: no data rows")
    return TrplData(str(path), t, c)


def rebin_time(counts: np.ndarray, time_ns: np.ndarray, factor: int):
    """Sum ``factor`` consecutive time bins (a short last group is summed too)."""
    factor = max(1, int(factor))
    if factor == 1 or counts.shape[0] == 0:
        return counts, time_ns
    n = counts.shape[0]
    n_groups = int(np.ceil(n / factor))
    pad = n_groups * factor - n
    c = counts
    if pad:
        c = np.concatenate([counts, np.zeros((pad, counts.shape[1]), dtype=counts.dtype)], axis=0)
    return c.reshape(n_groups, factor, counts.shape[1]).sum(axis=1), time_ns[::factor][:n_groups]


def roll_bins(counts: np.ndarray, shifts) -> np.ndarray:
    """Circular per-pixel shift by whole bins."""
    out = np.array(counts, copy=True)
    for p, s in enumerate(shifts):
        if int(s):
            out[:, p] = np.roll(counts[:, p], int(s))
    return out


def pre_pulse_window(time_ns: np.ndarray, total: np.ndarray, margin_ns: float = 1.0, margin_bins: int = 10):
    """Window from the first bin to just before the rise of ``total`` (None if under 3 bins)."""
    t = np.asarray(time_ns, dtype=np.float64)
    if t.size < 4:
        return None
    bin_ns = float(np.median(np.diff(t)))
    t_end  = float(t[int(np.argmax(total))]) - max(margin_ns, margin_bins * bin_ns)
    if int(np.count_nonzero(t <= t_end)) < 3:
        return None
    return float(t[0]), t_end


def range_bins(t_ns: np.ndarray, t_range: Optional[Tuple[float, float]]) -> Tuple[int, int]:
    n = int(t_ns.size)
    if not n or t_range is None:
        return 0, max(n - 1, 0)
    t0, t1 = sorted(t_range)
    r0 = int(np.clip(np.searchsorted(t_ns, t0 - 1e-9, side="left"), 0, n - 1))
    r1 = int(np.clip(np.searchsorted(t_ns, t1 + 1e-9, side="right") - 1, r0, n - 1))
    return r0, r1


# ══════════════════════════════════════════════════════════════════════
# Time-zero alignment on each pixel's first peak  (copied from the SPAD23 / SPAD93 v2 GUIs)
# ══════════════════════════════════════════════════════════════════════
#   1. coarse: circular cross-correlation of every trace with a template built
#      from the data itself, which finds large offsets whichever peak is higher;
#   2. fine: the rising edge (half height) of the template's first peak in every pixel;
#   3. every pixel is shifted by whole bins (circularly) so its edge lands on the
#      median edge of the largest group of pixels.
# Across a sweep the whole point is then moved to the sweep's time zero (the
# circular median of all points), so a time range / reference in ns means the
# same delay at every point.

T0_SMOOTH_NS = 0.3
T0_SEARCH_NS = 1.5
T0_EDGE_LEVEL = 0.5
T0_MIN_SNR = 10.0
T0_SECOND_MIN_REL = 0.15
T0_PEAK_SEP_NS = 3.0
T0_CLUSTER_NS = 1.0
T0_EDGE, T0_COARSE, T0_NONE = 0, 1, 2


class TimeZero:
    __slots__ = ("shift_bins", "status", "bin_ns", "target_ns", "second_delay_ns", "message")

    def __init__(self, shift_bins, status, bin_ns, target_ns, second_delay_ns, message):
        self.shift_bins      = shift_bins      # per pixel, raw bins (+ = later)
        self.status          = status          # per pixel, T0_* code
        self.bin_ns          = bin_ns
        self.target_ns       = target_ns       # where every first-peak edge is put (from bin 0)
        self.second_delay_ns = second_delay_ns
        self.message         = message


def _circular_boxcar(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y.astype(np.float64)
    ker = np.zeros(y.shape[0])
    ker[:k] = 1.0 / k
    ker = np.roll(ker, -(k // 2))
    return np.real(np.fft.ifft(np.fft.fft(y, axis=0) * np.fft.fft(ker)[:, None], axis=0))


def _template_peaks(tmpl: np.ndarray, dt: float, m: int) -> List[int]:
    n = tmpl.size
    w = max(int(round(0.5 / dt)), 1)
    sep = max(int(round(T0_PEAK_SEP_NS / dt)), 2)
    local_max = np.ones(n, dtype=bool)
    for s in range(1, w + 1):
        local_max &= (tmpl >= np.roll(tmpl, s)) & (tmpl >= np.roll(tmpl, -s))
    cand = np.flatnonzero(local_max & (tmpl >= T0_SECOND_MIN_REL * tmpl[m]))
    out = [m]
    for j in cand:
        j = int(j)
        if min((j - m) % n, (m - j) % n) <= sep:
            continue
        lo, hi = (j, m) if (m - j) % n < n // 2 else (m, j)
        seg = tmpl[(lo + np.arange((hi - lo) % n + 1)) % n]
        if seg.min() <= 0.7 * min(tmpl[j], tmpl[m]):
            out.append(j)
    return out


def _circ_dist(a, b, period: float):
    return (np.asarray(a) - b + period / 2.0) % period - period / 2.0


def align_first_peaks(counts: np.ndarray, time_ns: np.ndarray, edge_level: float = T0_EDGE_LEVEL) -> TimeZero:
    c = np.asarray(counts)
    t = np.asarray(time_ns, dtype=np.float64)
    n, n_pix = c.shape
    if n < 8:
        raise ValueError("Too few time bins.")
    dt = float(np.median(np.diff(t)))
    period = n * dt
    k = max(1, int(round(T0_SMOOTH_NS / dt))) | 1
    bg = np.median(c, axis=0).astype(np.float64)
    y = _circular_boxcar(c.astype(np.float64) - bg, k)
    norm = np.sqrt((y**2).sum(axis=0))
    yn = y / np.where(norm > 0, norm, 1.0)

    tmpl = np.median(np.stack([np.roll(yn[:, p], -int(np.argmax(yn[:, p]))) for p in range(n_pix)], axis=1), axis=1)
    F = np.fft.fft(yn, axis=0)
    lag = np.zeros(n_pix, dtype=np.int64)
    corr = np.zeros(n_pix)
    for _ in range(3):
        tn = np.sqrt((tmpl**2).sum())
        xc = np.real(np.fft.ifft(F * np.conj(np.fft.fft(tmpl))[:, None], axis=0))
        lag = np.argmax(xc, axis=0)
        corr = xc[lag, np.arange(n_pix)] / (tn if tn > 0 else 1.0)
        good = corr > 0.3
        if not good.any():
            break
        tmpl = np.mean(np.stack([np.roll(yn[:, p], -int(lag[p])) for p in np.flatnonzero(good)], axis=1), axis=1)

    m = int(np.argmax(tmpl))
    peaks = _template_peaks(tmpl, dt, m)
    before = [(int((m - j) % n), j) for j in peaks if 0 < (m - j) % n < n // 2]
    first = max(before)[1] if before else m
    after = sorted(int((j - first) % n) for j in peaks if j != first)
    second_delay = after[0] * dt if after and after[0] < n // 2 else float("nan")

    win = max(int(round(T0_SEARCH_NS / dt)), 2)
    status = np.full(n_pix, T0_NONE, dtype=np.int64)
    pos = np.full(n_pix, np.nan)
    for p in range(n_pix):
        if corr[p] <= 0.3:
            continue
        center = (first + int(lag[p])) % n
        idx = (center + np.arange(-win, win + 1)) % n
        seg = y[idx, p]
        i = int(np.argmax(seg))
        h = float(seg[i])
        pos[p] = center * dt
        status[p] = T0_COARSE
        if h < T0_MIN_SNR * np.sqrt(max(bg[p], 1.0) / k):
            continue
        g, steps = int(idx[i]), 0
        while y[(g - 1) % n, p] >= edge_level * h and steps < 4 * win:
            g -= 1
            steps += 1
        a, b = float(y[(g - 1) % n, p]), float(y[g % n, p])
        frac = (edge_level * h - a) / (b - a) if b != a else 1.0
        pos[p] = ((g - 1 + frac) * dt) % period
        status[p] = T0_EDGE
    if np.any(status == T0_COARSE):
        tm = float(tmpl[first])
        g, steps = first, 0
        while tmpl[(g - 1) % n] >= edge_level * tm and steps < 4 * win:
            g -= 1
            steps += 1
        a, b = float(tmpl[(g - 1) % n]), float(tmpl[g % n])
        offset = (g - 1 + ((edge_level * tm - a) / (b - a) if b != a else 1.0) - first) * dt
        coarse = status == T0_COARSE
        pos[coarse] = (pos[coarse] + offset) % period

    have = np.flatnonzero(np.isfinite(pos))
    shift = np.zeros(n_pix, dtype=np.int64)
    if have.size == 0:
        return TimeZero(shift, status, dt, float("nan"), second_delay,
                        "Time-zero alignment: no pixel has a usable first peak; nothing shifted.")
    ref = pos[have[status[have] == T0_EDGE]] if np.any(status[have] == T0_EDGE) else pos[have]
    counts_near = [int(np.sum(np.abs(_circ_dist(ref, v, period)) <= T0_CLUSTER_NS)) for v in ref]
    v0 = float(ref[int(np.argmax(counts_near))])
    group = ref[np.abs(_circ_dist(ref, v0, period)) <= T0_CLUSTER_NS]
    target = float((v0 + np.median(_circ_dist(group, v0, period))) % period)
    for p in have:
        shift[p] = int(np.round(_circ_dist(target, pos[p], period) / dt))

    s_ns = shift * dt
    msg = (f"Time zero: first-peak edge of every pixel put at {target:.2f} ns"
           + (f" (second peak {second_delay:.1f} ns later)" if np.isfinite(second_delay) else "")
           + f"; {int((shift != 0).sum())} px shifted")
    if shift.any():
        msg += f" (up to {np.abs(s_ns).max():.1f} ns)"
    n_coarse, n_none = int(np.sum(status == T0_COARSE)), int(np.sum(status == T0_NONE))
    if n_coarse:
        msg += f"; {n_coarse} px by cross-correlation only (weak first peak)"
    if n_none:
        msg += f"; {n_none} px without usable signal left alone"
    return TimeZero(shift, status, dt, target, second_delay, msg + ".")


def move_to_t0(tz: TimeZero, t_ns: np.ndarray, t0_ref: Optional[float]) -> TimeZero:
    """``tz`` with the whole point moved by whole bins so its time zero lands on ``t0_ref``."""
    if t0_ref is None or not np.isfinite(t0_ref) or not np.isfinite(tz.target_ns):
        return tz
    period = t_ns.size * tz.bin_ns
    t0 = float(t_ns[0]) + tz.target_ns
    whole = int(np.round(((t0_ref - t0 + period / 2.0) % period - period / 2.0) / tz.bin_ns))
    if not whole:
        return tz
    return TimeZero(tz.shift_bins + whole, tz.status, tz.bin_ns, (tz.target_ns + whole * tz.bin_ns) % period,
                    tz.second_delay_ns, tz.message + f" Whole point moved {whole * tz.bin_ns:+.2f} ns to the "
                                                     f"sweep's time zero ({t0_ref:.2f} ns).")


def common_time_zero(t0s, period_ns: float) -> float:
    """Circular median of the points' time zeros (NaN when none is finite)."""
    v = np.asarray([x for x in t0s if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    d = (v - v[0] + period_ns / 2.0) % period_ns - period_ns / 2.0
    return float(v[0] + np.median(d))


# With the alignment on, the views show t = raw time - offset: t = 0 ns at each point's signal peak
# (brightest pixel, found after the first-peak alignment), on the bin grid.
T0_REF_BINS = 5      # default reference frame: this many (shown) bins after time zero


def t0_display_offset(time_ns: np.ndarray, target_ns: Optional[float]) -> Optional[float]:
    """Raw time (ns) of the bin holding time zero ``target_ns`` (from bin 0); None without one."""
    t = np.asarray(time_ns, dtype=np.float64)
    if t.size < 2 or target_ns is None or not np.isfinite(target_ns):
        return None
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    return float(t[int(np.round(float(target_ns) / dt)) % t.size])


T0_PEAK_SMOOTH_NS = 0.5  # boxcar over which the brightest pixel's maximum is taken (the time zero shown)


def peak_display_offset(counts: np.ndarray, time_ns: np.ndarray, excluded=()) -> Optional[float]:
    """Raw time (ns) of the signal peak: the maximum of the brightest pixel (each pixel minus its median,
    smoothed over T0_PEAK_SMOOTH_NS).  With the time-zero alignment on, the views put t = 0 here (the
    pixels themselves stay aligned on their first-peak edge).  None without signal."""
    c = np.asarray(counts, dtype=np.float64)
    t = np.asarray(time_ns, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 4 or t.size != c.shape[0]:
        return None
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    k = max(1, int(round(T0_PEAK_SMOOTH_NS / dt))) | 1
    y = c - np.median(c, axis=0)
    if k > 1:
        cs = np.cumsum(np.vstack([np.zeros((1, y.shape[1])), y]), axis=0)
        y = (cs[k:] - cs[:-k]) / k  # row i = mean of rows i .. i+k-1 (centre i + k // 2)
    ex = [int(p) for p in excluded if 0 <= int(p) < y.shape[1]]
    if ex:
        y[:, ex] = -np.inf
    p = int(np.argmax(y.max(axis=0)))
    i = int(np.argmax(y[:, p]))
    if not y[i, p] > 0:
        return None
    return float(t[i + k // 2])


def view_offset(time_ns: np.ndarray, zero_ns: Optional[float]) -> Optional[float]:
    """Raw time of the bin holding ``zero_ns`` (a raw time, circular): subtracted, it puts t = 0 there."""
    t = np.asarray(time_ns, dtype=np.float64)
    if t.size < 2 or zero_ns is None or not np.isfinite(zero_ns):
        return None
    period = t.size * float(np.median(np.diff(t)))
    return t0_display_offset(t, (float(zero_ns) - float(t[0])) % period)


def t0_ref_index(view_time_ns: np.ndarray, n_bins: int = T0_REF_BINS) -> int:
    """Index of the default reference frame: ``n_bins`` bins after the first bin at or after t = 0."""
    t = np.asarray(view_time_ns, dtype=np.float64)
    if t.size == 0:
        return 0
    return int(min(int(np.searchsorted(t, -1e-9, side="left")) + int(n_bins), t.size - 1))


class TrplStore:
    """Thread-safe cache of TRPL files (least recently used out first) and their time-zero alignments."""

    def __init__(self, max_files: int = TRPL_CACHE_FILES):
        self._lock  = threading.Lock()
        self._max   = max_files
        self._files: "OrderedDict[str, TrplData]" = OrderedDict()
        self._align: Dict[str, Tuple[Optional[TimeZero], float, float, str]] = {}  # tz, t_first, period, error
        self._delay: Dict[str, Optional[float]] = {}      # signal-peak delay after the time zero, per file
        self._sweep_delay: Dict[float, Optional[float]] = {}  # common_t0 -> median delay

    def clear(self) -> None:
        with self._lock:
            self._files.clear()
            self._align.clear()
            self._delay.clear()
            self._sweep_delay.clear()

    def load(self, path) -> TrplData:
        key = str(path)
        with self._lock:
            d = self._files.get(key)
            if d is not None:
                self._files.move_to_end(key)
                return d
        d = load_trpl_csv(key)
        with self._lock:
            self._files[key] = d
            while len(self._files) > self._max:
                self._files.popitem(last=False)
        return d

    def has_alignment(self, path) -> bool:
        with self._lock:
            return str(path) in self._align

    def alignment(self, path, data: Optional[TrplData] = None):
        key = str(path)
        with self._lock:
            info = self._align.get(key)
        if info is not None:
            return info
        data = data if data is not None else self.load(key)
        t = data.time_ns
        period = t.size * float(np.median(np.diff(t)))
        try:
            info = (align_first_peaks(data.counts, t), float(t[0]), period, "")
        except ValueError as exc:
            info = (None, float(t[0]), period, str(exc))
        delay = None    # signal peak (brightest pixel) after the file's own time zero (first-peak edge)
        if info[0] is not None and np.isfinite(info[0].target_ns):
            pk = peak_display_offset(roll_bins(data.counts, info[0].shift_bins), t)
            if pk is not None:
                delay = float((pk - (float(t[0]) + info[0].target_ns) + period / 2.0) % period - period / 2.0)
        with self._lock:
            self._align[key] = info
            self._delay[key] = delay
        return info

    def zero_of(self, data: TrplData, tz: Optional["TimeZero"], t0_ref: Optional[float]) -> Optional[float]:
        """Raw time shown as t = 0 (alignment on): this point's own signal peak (brightest pixel), so every
        point has its peak at 0 even where the first-peak alignment locked onto another feature; a point
        without a clear peak uses the sweep's median delay after its time zero.  None if neither is known."""
        if tz is None or not np.isfinite(tz.target_ns):
            return None
        self.alignment(data.path, data)
        with self._lock:
            d = self._delay.get(str(data.path))
            if d is None and t0_ref is not None and np.isfinite(t0_ref):
                d = self._sweep_delay.get(round(float(t0_ref), 9))
        return None if d is None else float(data.time_ns[0]) + tz.target_ns + d

    def common_t0(self, paths) -> Optional[float]:
        infos = [self.alignment(p) for p in paths]
        t0s = [t_first + tz.target_ns for tz, t_first, _, _ in infos if tz is not None]
        if not t0s:
            return None
        period = float(np.median([per for tz, _, per, _ in infos if tz is not None]))
        t0 = common_time_zero(t0s, period)
        if not np.isfinite(t0):
            return None
        with self._lock:
            ds = [self._delay.get(str(p)) for p in paths]
            ds = [d for d in ds if d is not None]
            self._sweep_delay[round(float(t0), 9)] = float(np.median(ds)) if ds else None
        return t0


class ViewSettings:
    """How a TRPL file becomes the counts that are shown and fitted."""
    __slots__ = ("align", "t0_ref", "rebin")

    def __init__(self, align: bool, t0_ref: Optional[float], rebin: int):
        self.align, self.t0_ref, self.rebin = bool(align), t0_ref, max(1, int(rebin))

    def key(self):
        return (self.align, self.t0_ref if self.align else None, self.rebin)


def point_view(store: TrplStore, data: TrplData, vs: ViewSettings):
    """(time_ns, counts [bins, pixels], TimeZero or None, note) of one point.

    With the alignment on, t = 0 ns is the signal peak (TrplStore.zero_of; the pixels stay aligned on
    their first-peak edge)."""
    tz, note = None, ""
    counts = data.counts
    t_raw = data.time_ns
    if vs.align:
        tz0, _, _, err = store.alignment(data.path, data)
        if tz0 is None:
            note = f"time-zero alignment failed ({err}); raw data"
        else:
            tz = move_to_t0(tz0, data.time_ns, vs.t0_ref)
            counts = roll_bins(counts, tz.shift_bins)
        off = view_offset(data.time_ns, store.zero_of(data, tz, vs.t0_ref))
        if off is not None:
            t_raw = data.time_ns - off
            if vs.rebin > 1:            # rebinned groups start at t = 0 (drop < rebin early bins)
                s0 = int(np.argmin(np.abs(t_raw))) % vs.rebin
                counts, t_raw = counts[s0:], t_raw[s0:]
    c, t = rebin_time(counts, t_raw, vs.rebin)
    return t, c, tz, note


# ══════════════════════════════════════════════════════════════════════
# 2D Gaussian fits and diffusion  (fits copied from the SPAD23 / SPAD93 v2 GUIs, method 1)
# ══════════════════════════════════════════════════════════════════════
#   Fit All: the reference frame is fitted, then every later frame starting from
#   the previous fit with small bounds.  sigma_eq² = (sx² + sy²)/2 is the
#   per-axis variance, so for diffusion  sigma² = sigma0² + 2 D (t − t_ref).

def gauss2d(coords, A, x0, y0, sx, sy, offset):
    xg, yg = coords
    return A * np.exp(-(((xg - x0) ** 2) / (2.0 * sx**2) + ((yg - y0) ** 2) / (2.0 * sy**2))) + offset


class FitResult:
    __slots__ = ("popt", "sigma_eq", "snr", "at_bound")

    def __init__(self, popt, sigma_eq, snr, at_bound):
        self.popt, self.sigma_eq, self.snr, self.at_bound = popt, sigma_eq, snr, at_bound

    @property
    def quality_ok(self) -> bool:
        return self.snr >= MIN_SNR


def _clip_into(p0, lower, upper) -> List[float]:
    out = []
    for v, lo, hi in zip(p0, lower, upper):
        v = float(v)
        if np.isfinite(lo) and np.isfinite(hi):
            span = hi - lo
            v = min(max(v, lo + 1e-6 * span), hi - 1e-6 * span)
        elif np.isfinite(lo):
            v = max(v, lo + 1e-9)
        elif np.isfinite(hi):
            v = min(v, hi - 1e-9)
        out.append(v)
    return out


def _hit_bound(p, lower, upper, rel: float = 1e-3) -> bool:
    for v, lo, hi in zip(p, lower, upper):
        if np.isfinite(lo) and np.isfinite(hi):
            tol = rel * (hi - lo)
            if v - lo <= tol or hi - v <= tol:
                return True
    return False


def _fit_gauss2d(x, y, z, p0, lower, upper, fix_offset: Optional[float], maxfev: int):
    """curve_fit of gauss2d with bounds; the offset held at ``fix_offset`` if given → (popt, pcov)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fix_offset is None:
            popt, pcov = _curve_fit(gauss2d, (x, y), z, p0=_clip_into(p0, lower, upper),
                                    bounds=(lower, upper), maxfev=maxfev)
            return np.asarray(popt, float), np.asarray(pcov, float)
        off = float(fix_offset)
        popt, pcov = _curve_fit(lambda xy, A, x0, y0, sx, sy: gauss2d(xy, A, x0, y0, sx, sy, off), (x, y), z,
                                p0=_clip_into(p0[:5], lower[:5], upper[:5]), bounds=(lower[:5], upper[:5]),
                                maxfev=maxfev)
    full = np.zeros((6, 6))
    full[:5, :5] = pcov
    return np.append(np.asarray(popt, float), off), full


def _make_fit(z, popt, pcov, lower, upper, n_free) -> FitResult:
    dof = int(z.size) - n_free
    var = float(pcov[0, 0]) if np.size(pcov) else float("nan")
    snr = float("inf") if dof <= 0 else (0.0 if not np.isfinite(var) or var <= 0 else float(popt[0] / np.sqrt(var)))
    return FitResult(popt, float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0)), snr, _hit_bound(popt, lower, upper))


def fit_gaussian_2d(x, y, z, pitch_x: float, pitch_y: float,
                    fix_offset: Optional[float] = None) -> Optional[FitResult]:
    """Free 2D Gaussian from moment start values (the reference frame); pitches of the detector."""
    if _curve_fit is None or z.size < MIN_FIT_PIXELS_2D or np.allclose(z, z[0]):
        return None
    span  = max(float(np.ptp(x)), float(np.ptp(y)), pitch_x, pitch_y)
    lower = [0.0, float(np.min(x) - pitch_x), float(np.min(y) - pitch_y), 0.5, 0.5, -np.inf]
    upper = [np.inf, float(np.max(x) + pitch_x), float(np.max(y) + pitch_y), 2.0 * span, 2.0 * span, np.inf]
    bg = float(np.percentile(z, 10)) if fix_offset is None else float(fix_offset)
    w = np.clip(z - bg, 0.0, None)
    tot = float(w.sum())
    amp = max(float(np.max(z) - bg), 1e-6)
    if tot <= 0:
        guess = [amp, float(np.mean(x)), float(np.mean(y)), 20.0, 20.0, bg]
    else:
        x0, y0 = float((w * x).sum() / tot), float((w * y).sum() / tot)
        guess = [amp, x0, y0,
                 float(np.sqrt(max((w * (x - x0) ** 2).sum() / tot, (0.5 * pitch_x) ** 2))),
                 float(np.sqrt(max((w * (y - y0) ** 2).sum() / tot, (0.5 * pitch_y) ** 2))), bg]
    zs = max(float(np.max(np.abs(z))), 1e-300)        # fit z / max|z|: much faster on noisy frames
    guess[0] /= zs
    guess[5] /= zs
    try:
        popt, pcov = _fit_gauss2d(x, y, z / zs, guess, lower, upper,
                                  None if fix_offset is None else float(fix_offset) / zs, 20000)
    except Exception:
        return None
    popt[[0, 5]] *= zs
    return _make_fit(z, popt, pcov * zs**2, lower, upper, 6 if fix_offset is None else 5)


def fit_gaussian_2d_next(x, y, z, prev: np.ndarray, tightness: float, pitch_x: float, pitch_y: float,
                         fix_offset: Optional[float] = None) -> Optional[FitResult]:
    """Next TRPL frame from the previous frame's fit, with small bounds around it."""
    if _curve_fit is None or np.allclose(z, z[0]):
        return None
    A, x0, y0, sx, sy, off = [float(v) for v in prev]
    sx, sy = max(sx, 1e-3), max(sy, 1e-3)
    dA, dx, dy = max(abs(A) * 0.18 * tightness, 1e-3), max(pitch_x * 0.18 * tightness, 0.8), \
        max(pitch_y * 0.18 * tightness, 0.8)
    dsx, dsy, doff = max(sx * 0.18 * tightness, 0.5), max(sy * 0.18 * tightness, 0.5), \
        max(abs(off) * 0.25 * tightness, 0.5)
    lower = [max(0.0, A - dA), x0 - dx, y0 - dy, max(1e-3, sx - dsx), max(1e-3, sy - dsy), off - doff]
    upper = [A + dA, x0 + dx, y0 + dy, sx + dsx, sy + dsy, off + doff]
    if fix_offset is not None:
        lower[5], upper[5] = -np.inf, np.inf
    try:
        popt, pcov = _fit_gauss2d(x, y, z, [A, x0, y0, sx, sy, off], lower, upper, fix_offset, 20000)
    except Exception:
        return None
    return _make_fit(z, popt, pcov, lower, upper, 6 if fix_offset is None else 5)


class FitSettings:
    """The TRPL settings a diffusion fit depends on (also used to tell whether a fit is still valid)."""
    __slots__ = ("view", "t_range", "ref_ns", "dark", "pre_window", "fix_offset", "excluded", "dark_ref", "dark_id",
                 "dark_per_s")

    def __init__(self, view: ViewSettings, t_range, ref_ns, dark, pre_window, fix_offset, excluded,
                 dark_ref: Optional[np.ndarray] = None, dark_id: str = "", dark_per_s: bool = False):
        """``dark`` = subtract the pre-t0 offset; ``dark_ref`` = a dark reference per pixel (counts per
        view bin), subtracted instead; ``dark_id`` names it (file, scale) for the cache key.
        ``dark_per_s``: ``dark_ref`` is per second of acquisition, scaled by each file's own time
        (dynamic TRPL time: every spot has its own)."""
        self.view, self.t_range, self.ref_ns = view, t_range, float(ref_ns)
        self.dark, self.pre_window, self.fix_offset = bool(dark), pre_window, bool(fix_offset)
        self.excluded = tuple(sorted(excluded))
        self.dark_ref, self.dark_id = dark_ref, (dark_id if dark_ref is not None else "")
        self.dark_per_s = bool(dark_per_s) and dark_ref is not None

    def key(self):
        return (self.view.key(), self.t_range, round(self.ref_ns, 6), self.dark, self.pre_window,
                self.fix_offset, self.excluded, self.dark_id)

    def mask(self, n_pixels: int) -> np.ndarray:
        m = np.ones(int(n_pixels), dtype=bool)
        ex = [p for p in self.excluded if p < n_pixels]
        if ex:
            m[ex] = False
        return m

    def describe(self) -> str:
        rng = "all bins" if self.t_range is None else f"{self.t_range[0]:g}–{self.t_range[1]:g} ns"
        bg = ("pre-t0 offset subtracted (" + ("auto window" if self.pre_window is None else
              f"{self.pre_window[0]:g}–{self.pre_window[1]:g} ns") + ")") if self.dark else "no pre-t0 offset"
        if self.dark_ref is not None:
            bg = f"dark reference subtracted ({self.dark_id.split('|')[0]})"
        return (f"sequential 2D Gaussian, reference {self.ref_ns:.3f} ns, range {rng}, rebin ×{self.view.rebin}, "
                f"{'aligned' if self.view.align else 'not aligned'}, {bg}, Gaussian offset "
                f"{'fixed at 0' if self.fix_offset else 'fitted'}, {len(self.excluded)} px excluded")


class PointDiffusion:
    __slots__ = ("t_ns", "sigma2", "flagged", "ref_ns", "failures", "note")

    def __init__(self, t_ns, sigma2, flagged, ref_ns, failures=0, note=""):
        self.t_ns, self.sigma2, self.flagged = t_ns, sigma2, flagged   # flagged: at a bound or low signal
        self.ref_ns, self.failures, self.note = ref_ns, failures, note


def fit_point(t: np.ndarray, counts: np.ndarray, s: FitSettings, cancelled=None,
              acq_s: Optional[float] = None) -> PointDiffusion:
    """Fit All of one point: reference frame, then every later frame of the range (raises ValueError).

    The detector (pixel layout, fit pitches) follows the number of pixel columns of ``counts``.
    ``acq_s``: this file's acquisition time (needed when the dark reference is per second).
    """
    geom = geometry_for(counts.shape[1])
    r0, r1 = range_bins(t, s.t_range)
    ref = int(np.argmin(np.abs(t - s.ref_ns)))
    if max(r0, ref) > r1:
        raise ValueError("the time range has no frames after the reference")
    mask = s.mask(geom.num_pixels)
    if int(mask.sum()) < MIN_FIT_PIXELS_2D:
        raise ValueError(f"not enough enabled pixels (need ≥ {MIN_FIT_PIXELS_2D})")
    frames, note = np.asarray(counts[: r1 + 1], dtype=np.float64), ""
    if s.dark_ref is not None:
        if np.size(s.dark_ref) != counts.shape[1]:
            raise ValueError(f"the dark reference has {np.size(s.dark_ref)} pixels, the data {counts.shape[1]}")
        dark = np.asarray(s.dark_ref, dtype=np.float64)
        if s.dark_per_s:
            if acq_s is None or not np.isfinite(acq_s) or acq_s <= 0:
                raise ValueError("no acquisition time for this file (not in dynamic_trpl_log.csv): "
                                 "the dark reference cannot be scaled")
            dark = dark * float(acq_s)
        frames = frames - dark
    elif s.dark:
        win = s.pre_window if s.pre_window is not None else pre_pulse_window(t, counts.sum(axis=1))
        sel = np.zeros(t.size, bool) if win is None else (t >= win[0]) & (t <= win[1])
        if sel.any():
            frames = frames - counts[sel].mean(axis=0)
        else:
            note = "pre-t0 window empty: no offset subtracted"
    fix = 0.0 if s.fix_offset else None
    xm, ym = geom.x_um[mask], geom.y_um[mask]
    px, py = geom.pitch_x, geom.pitch_y
    prev = fit_gaussian_2d(xm, ym, frames[ref][mask], px, py, fix_offset=fix)
    if prev is None:
        raise ValueError("could not fit the reference frame")
    results = {ref: prev}
    failures = 0
    for i in range(max(r0, ref) + 1, r1 + 1):
        if cancelled is not None and cancelled():
            break
        z = frames[i][mask]
        fit = fit_gaussian_2d_next(xm, ym, z, prev.popt, 1.0, px, py, fix)
        if fit is None:
            fit = fit_gaussian_2d_next(xm, ym, z, prev.popt, 1.8, px, py, fix)
        if fit is None:
            failures += 1
        else:
            results[i] = prev = fit
    idx = sorted(results)
    fits = [results[i] for i in idx]
    return PointDiffusion(t[idx], np.array([f.sigma_eq ** 2 for f in fits]),
                          np.array([bool(f.at_bound) or not f.quality_ok for f in fits]),
                          float(t[ref]), failures, note)


def fit_line(pd_: PointDiffusion, window: Tuple[float, Optional[float]], use_flagged: bool = False):
    """Line sigma² = sigma0² + slope·(t − t_ref) over ``window`` (ns after the reference; end None = all).

    Returns dict(D, D_err [cm²/s], s0, s0_err [µm²], slope, n) or None (fewer than 3 bins).
    """
    tr = pd_.t_ns - pd_.ref_ns
    ok = np.isfinite(pd_.sigma2) & (tr >= window[0] - 1e-9)
    if window[1] is not None:
        ok &= tr <= window[1] + 1e-9
    if not use_flagged:
        ok &= ~pd_.flagged
    n = int(ok.sum())
    if n < 3:
        return None
    x, y = tr[ok], pd_.sigma2[ok]
    xm, ym = float(x.mean()), float(y.mean())
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return None
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    icpt = ym - slope * xm
    s2 = float(((y - icpt - slope * x) ** 2).sum()) / (n - 2) if n > 2 else float("nan")
    k = UM2_NS_TO_CM2_S / 2.0
    return {"D": slope * k, "D_err": float(np.sqrt(s2 / sxx)) * k, "s0": icpt,
            "s0_err": float(np.sqrt(s2 * (1.0 / n + xm ** 2 / sxx))), "slope": slope, "n": n}


def process_sweep(store: TrplStore, paths, vs: ViewSettings, fit: Optional[FitSettings] = None,
                  progress=None, cancelled=None, acq_of=None):
    """Counts of every point on a common time axis, and/or (``fit``) its Fit All.
    ``acq_of(path)``: a file's acquisition time (s), for a per-second dark reference.

    Returns dict(t, traces [points, bins, pixels] float32 NaN = missing, fits, notes, cancelled).
    The pixel count is that of the first point; points of another detector are left out (noted).
    """
    stop = cancelled or (lambda: False)
    views, fits, notes = [], [], []
    for k, path in enumerate(paths):
        if stop():
            return {"cancelled": True}
        view, pdf = None, None
        if path is not None:
            try:
                t, c, _, note = point_view(store, store.load(path), vs)
                view = (t, c)
                if note:
                    notes.append(f"{Path(path).name}: {note}")
                if fit is not None:
                    pdf = fit_point(t, c, fit, cancelled=stop,
                                    acq_s=acq_of(path) if acq_of is not None else None)
            except Exception as exc:
                notes.append(f"{Path(path).name}: {exc}")
        views.append(None if fit is not None else view)
        fits.append(pdf)
        if progress is not None:
            progress(k + 1, len(paths))
    out = {"fits": fits if fit is not None else None, "notes": notes, "cancelled": False, "t": None, "traces": None}
    first = next((v for v in views if v is not None), None)
    if fit is None and first is not None:
        t_ref, n_pix = first[0], first[1].shape[1]
        tr = np.full((len(paths), t_ref.size, n_pix), np.nan, dtype=np.float32)
        for k, v in enumerate(views):
            if v is None:
                continue
            t, c = v
            if c.shape[1] != n_pix:
                notes.append(f"{Path(paths[k]).name}: {c.shape[1]} pixels, the sweep has {n_pix} (other detector): left out")
                continue
            if t.size == t_ref.size and np.allclose(t, t_ref):
                tr[k] = c
            else:
                for p in range(n_pix):
                    tr[k, :, p] = np.interp(t_ref, t, c[:, p], left=np.nan, right=np.nan)
        out["t"], out["traces"] = t_ref, tr
    return out


# ══════════════════════════════════════════════════════════════════════
# Worker thread (alignment, sweep loading and fits never run in the GUI thread)
# ══════════════════════════════════════════════════════════════════════

class _Job(QtCore.QThread):
    """Runs fn(progress, cancelled); done(job_id, result, error text)."""
    progress = QtCore.pyqtSignal(int, int)
    done     = QtCore.pyqtSignal(int, object, str)

    def __init__(self, job_id: int, fn, parent=None):
        super().__init__(parent)
        self.job_id  = int(job_id)
        self._fn     = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            res = self._fn(self.progress.emit, lambda: self._cancel)
            self.done.emit(self.job_id, res, "")
        except Exception as exc:
            self.done.emit(self.job_id, None, str(exc))


# ══════════════════════════════════════════════════════════════════════
# TRPL plots
# ══════════════════════════════════════════════════════════════════════

def _style(plot, left: str = "", bottom: str = "") -> None:
    for ax in ("bottom", "left"):
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setLabel("left", left)
    plot.setLabel("bottom", bottom)


_TRACE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#e377c2"]


class PixelMapPlot(pg.PlotWidget):
    """The SPAD pixels (SPAD23 or SPAD93, ``set_geometry``) as coloured dots.

    Left click: select pixel; right click: exclude pixel.
    """
    pixel_clicked = QtCore.pyqtSignal(int, int)      # pixel, 1 = left / 2 = right

    def __init__(self, parent=None):
        super().__init__(parent)
        pi = self.getPlotItem()
        _style(pi, "y (µm)", "x (µm)")
        pi.showGrid(x=False, y=False)
        pi.getViewBox().setAspectLocked(True)
        pi.getViewBox().setMouseEnabled(False, False)
        pi.hideButtons()
        self._cmap = _pg_cmap("inferno")
        self.geom: Optional[SpadGeometry] = None
        self.dots = self.sel = self.excl = None
        self.labels = []
        self.set_geometry(SPAD23_GEOM)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_geometry(self, geom: SpadGeometry) -> None:
        """Rebuild the dots and labels for this detector's layout."""
        if geom is self.geom:
            return
        pi = self.getPlotItem()
        for it in [self.dots, self.sel, self.excl, *self.labels]:
            if it is not None:
                pi.removeItem(it)
        self.geom = g = geom
        if g.dot_um is None:    # SPAD23: fixed-size dots, as before
            dot = dict(size=34, pxMode=True)
            ring = dict(size=42, pxMode=True)
            cross = dict(size=20, pxMode=True)
            font = None
        else:                   # SPAD93: dots in µm, so 93 of them fit at any widget size
            dot = dict(size=g.dot_um, pxMode=False)
            ring = dict(size=1.25 * g.dot_um, pxMode=False)
            cross = dict(size=0.6 * g.dot_um, pxMode=False)
            font = 7
        self.dots = pg.ScatterPlotItem(x=g.x_um, y=g.y_um, pen=pg.mkPen("#374151", width=1), **dot)
        self.sel = pg.ScatterPlotItem(pen=pg.mkPen("#06b6d4", width=3), brush=None, **ring)
        self.excl = pg.ScatterPlotItem(symbol="x", pen=pg.mkPen("#ef4444", width=2),
                                       brush=pg.mkBrush("#ef4444"), **cross)
        for it in (self.dots, self.sel, self.excl):
            pi.addItem(it)
        self.labels = []
        for p in range(g.num_pixels):
            t = pg.TextItem(str(p), anchor=(0.5, 0.5), color="w")
            if font is not None:
                f = t.textItem.font()
                f.setPointSize(font)
                t.setFont(f)
            t.setPos(g.x_um[p], g.y_um[p])
            pi.addItem(t)
            self.labels.append(t)
        pi.getViewBox().invertY(g.invert_y)
        pi.setXRange(*g.x_range, padding=0)
        pi.setYRange(*g.y_range, padding=0)

    def set_values(self, v: np.ndarray, excluded) -> None:
        v = np.asarray(v, dtype=float).copy()
        if excluded:
            v[list(excluded)] = np.nan
        fin = v[np.isfinite(v)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        hi = hi if hi > lo else lo + 1.0
        norm = np.clip((v - lo) / (hi - lo), 0, 1)
        brushes = [pg.mkBrush("#d1d5db") if not np.isfinite(x) else pg.mkBrush(self._cmap.map(float(x), mode="qcolor"))
                   for x in norm]
        self.dots.setBrush(brushes)
        for p, t in enumerate(self.labels):
            t.setColor("#111827" if np.isfinite(norm[p]) and norm[p] > 0.6 else "#f9fafb")
        self.getPlotItem().setTitle(f"counts {lo:.4g} … {hi:.4g}")

    def set_selected(self, pix, excluded) -> None:
        g = self.geom
        ids = np.asarray(sorted(p for p in pix if p < g.num_pixels), dtype=int)
        self.sel.setData(x=g.x_um[ids], y=g.y_um[ids])
        ex = np.asarray(sorted(p for p in excluded if p < g.num_pixels), dtype=int)
        self.excl.setData(x=g.x_um[ex], y=g.y_um[ex])

    def _on_click(self, ev) -> None:
        vb = self.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = vb.mapSceneToView(ev.scenePos())
        d = np.hypot(self.geom.x_um - pt.x(), self.geom.y_um - pt.y())
        p = int(np.argmin(d))
        if d[p] <= SpadGeometry.PICK_RADIUS_UM:
            self.pixel_clicked.emit(p, 2 if ev.button() == QtCore.Qt.RightButton else 1)


# ══════════════════════════════════════════════════════════════════════
# TRPL panel: settings + "Point" and "Sweep" tabs
# ══════════════════════════════════════════════════════════════════════

class TrplPanel(QtWidgets.QWidget):
    status          = QtCore.pyqtSignal(str)
    align_requested = QtCore.pyqtSignal()     # alignment switched on: align all TRPL files of the folder
    sync_requested  = QtCore.pyqtSignal()     # the Sweep tab was opened: send the current sweep line

    def __init__(self, store: TrplStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.folder: Optional[Path] = None
        self.t0_ref: Optional[float] = None      # sweep time zero (alignment on)
        self.t0_ready = True
        # the time range / reference / pre-t0 window are in the shown coordinates: after time zero (aligned)
        # or raw; they are converted when the view switches between the two (see _sync_time_frame)
        self._times_after_t0 = False
        self._last_off: Optional[float] = None   # raw time of t = 0 at the last aligned point
        self.geom = SPAD23_GEOM                 # detector of the loaded data (set per point)
        self.sel = {self.geom.center_pixel}
        self.excluded: set = set()
        self._dark: Optional[DarkReference] = None     # the folder's dark reference (beam blocked)
        self._dark_err = ""
        self._acq_s: Optional[float] = None             # TRPL acquisition time (session.md)
        self._dyn: Dict[str, dict] = {}                 # dynamic TRPL time: per-file time / flag (log)
        self._bg_fallback = False                       # dark wanted but not usable yet: pre-t0 for now
        # current point
        self._path: Optional[str] = None
        self._label = ""
        self._t = np.zeros(0)
        self._c = np.zeros((0, self.geom.num_pixels))
        self._tz: Optional[TimeZero] = None
        self._view_key = None
        self._point_fit: Optional[PointDiffusion] = None
        self._point_fit_key = None
        # sweep line
        self._points: List[Tuple[float, str, Optional[Path]]] = []
        self._points_all: List[Tuple[float, str, Optional[Path]]] = []   # before 'Skip below-target spots'
        self._x_label = "X"
        self._marker: Optional[float] = None
        self._context = ""
        self._sweep = None
        self._sweep_key = None
        self._fits = None
        self._fits_key = None
        self._fits_settings: Optional[FitSettings] = None
        self._fit_notes: List[str] = []
        # threads
        self._jobs: List[_Job] = []
        self._job_id = 0
        self._map_job = self._fit_job = self._point_job = None
        self._build_ui()
        self._map_timer = QtCore.QTimer(self)
        self._map_timer.setSingleShot(True)
        self._map_timer.setInterval(250)
        self._map_timer.timeout.connect(self._start_map_job)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        box = QtWidgets.QGroupBox("TRPL settings (shared by Point and Sweep)")
        g = QtWidgets.QGridLayout(box)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        g.setContentsMargins(8, 8, 8, 8)

        def spin(lo, hi, dec, suffix="", width=92):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.setFixedWidth(width)
            return s

        self.alignCheck = QtWidgets.QCheckBox("Align time zero")
        self.alignCheck.setToolTip(
            "Put every pixel's time zero on the rising edge of its first peak (as in the SPAD v2 GUIs),\n"
            "then move each point as a whole to the sweep's time zero (circular median of all TRPL files),\n"
            "so a time range or reference in ns means the same delay at every point. The time axis is then\n"
            "shifted so t = 0 ns is each point's signal peak (brightest pixel); the time range,\n"
            "reference and pre-t0 window are converted when\n"
            "this is switched. View and fits only.")
        self.rebinSpin = QtWidgets.QSpinBox()
        self.rebinSpin.setRange(1, 1000)
        self.rebinSpin.setPrefix("×")
        self.t0Spin, self.t1Spin = spin(-1e6, 1e6, 3, " ns"), spin(-1e6, 1e6, 3, " ns")
        self.fullRangeCheck = QtWidgets.QCheckBox("All bins")
        self.fullRangeCheck.setChecked(True)
        self.refSpin = spin(-1e6, 1e6, 3, " ns")
        self.refSpin.setToolTip("Reference frame of the fits (σ² is fitted from here on); t_ref of D.")
        self.refCursorBtn = QtWidgets.QPushButton("= cursor")
        self.refCursorBtn.setFixedWidth(70)
        self.refAutoCheck = QtWidgets.QCheckBox(f"t0+{T0_REF_BINS}")
        self.refAutoCheck.setChecked(True)
        self.refAutoCheck.setToolTip(
            f"With 'Align time zero' on: the reference follows the signal peak (t = 0) + {T0_REF_BINS} bins (shown bins, after\n"
            "rebinning). Typing a reference or '= cursor' switches this off.")
        self.logCheck = QtWidgets.QCheckBox("Log y")
        g.addWidget(self.alignCheck, 0, 0)
        g.addWidget(QtWidgets.QLabel("Rebin"), 0, 1)
        g.addWidget(self.rebinSpin, 0, 2)
        g.addWidget(QtWidgets.QLabel("Time range"), 0, 3)
        g.addWidget(self.t0Spin, 0, 4)
        g.addWidget(QtWidgets.QLabel("to"), 0, 5)
        g.addWidget(self.t1Spin, 0, 6)
        g.addWidget(self.fullRangeCheck, 0, 7)
        g.addWidget(QtWidgets.QLabel("Reference"), 0, 8)
        g.addWidget(self.refSpin, 0, 9)
        rb = QtWidgets.QHBoxLayout()
        rb.setSpacing(4)
        rb.addWidget(self.refCursorBtn)
        rb.addWidget(self.refAutoCheck)
        g.addLayout(rb, 0, 10)
        g.addWidget(self.logCheck, 0, 11)

        self.bgCombo = QtWidgets.QComboBox()
        self.bgCombo.addItem("Subtract pre-t0 offset", "pre_t0")
        self.bgCombo.addItem("Subtract dark reference", "dark_ref")
        self.bgCombo.addItem("No background", "none")
        self.bgCombo.setToolTip(
            "Per-pixel background subtracted before the 2D fits.\n"
            "Pre-t0 offset: each pixel's mean count per bin over the pre-t0 window (dark counts plus any signal\n"
            "left over from earlier pulses).\n"
            "Dark reference: the folder's DARK_TRPL_*.csv (beam blocked), scaled to the acquisition time in\n"
            "session.md (dynamic TRPL time: each file's own time from dynamic_trpl_log.csv) and the bin width shown.")
        self.autoWinCheck = QtWidgets.QCheckBox("auto window")
        self.autoWinCheck.setChecked(True)
        self.w0Spin, self.w1Spin = spin(-1e6, 1e6, 3, " ns"), spin(-1e6, 1e6, 3, " ns")
        self.fixOffsetCheck = QtWidgets.QCheckBox("Fix Gaussian offset at 0")
        self.exclEdit = QtWidgets.QLineEdit()
        self.exclEdit.setPlaceholderText("e.g. 53, 7")
        self.exclEdit.setFixedWidth(90)
        self.exclEdit.setToolTip("Pixels left out of the fits (or right-click them on the map).")
        self.d0Spin, self.d1Spin = spin(0, 1e6, 3, " ns", 86), spin(0, 1e6, 3, " ns", 86)
        self.d1Spin.setSpecialValueText("end")
        self.flaggedCheck = QtWidgets.QCheckBox("Include flagged fits")
        self.flaggedCheck.setToolTip("Also use fits at a bound or with low signal (hollow points) in the σ² line.")
        g.addWidget(self.bgCombo, 1, 0)
        g.addWidget(self.autoWinCheck, 1, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("Pre-t0 window"), 1, 3)
        g.addWidget(self.w0Spin, 1, 4)
        g.addWidget(QtWidgets.QLabel("to"), 1, 5)
        g.addWidget(self.w1Spin, 1, 6)
        g.addWidget(self.fixOffsetCheck, 1, 7)
        g.addWidget(QtWidgets.QLabel("Excluded px"), 1, 8)
        g.addWidget(self.exclEdit, 1, 9)
        dl = QtWidgets.QHBoxLayout()
        dl.addWidget(QtWidgets.QLabel("σ² line"))
        dl.addWidget(self.d0Spin)
        dl.addWidget(QtWidgets.QLabel("to"))
        dl.addWidget(self.d1Spin)
        dl.addWidget(QtWidgets.QLabel("after ref."))
        dl.addWidget(self.flaggedCheck)
        dl.addStretch()
        g.addLayout(dl, 1, 10, 1, 4)
        self.infoLbl = QtWidgets.QLabel("Load a PL + TRPL folder.")
        self.infoLbl.setStyleSheet("color:#4b5563")
        self.infoLbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        g.addWidget(self.infoLbl, 2, 0, 1, 14)
        g.setColumnStretch(13, 1)
        root.addWidget(box)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._make_point_page(), "Point")
        self.tabs.addTab(self._make_sweep_page(), "Sweep")
        root.addWidget(self.tabs, 1)

        self.bgCombo.currentIndexChanged.connect(lambda _i: self._on_bg_changed())
        for w in (self.alignCheck, self.logCheck, self.autoWinCheck, self.fixOffsetCheck,
                  self.fullRangeCheck, self.flaggedCheck):
            w.toggled.connect(lambda _v, w=w: self._on_setting(w))
        self.refSpin.valueChanged.connect(lambda _v: self._set_ref_auto(False))   # typed: no longer t0 + N
        self.refAutoCheck.toggled.connect(self._on_ref_auto)
        for w in (self.rebinSpin, self.t0Spin, self.t1Spin, self.refSpin, self.w0Spin, self.w1Spin,
                  self.d0Spin, self.d1Spin):
            w.valueChanged.connect(lambda _v, w=w: self._on_setting(w))
        self.exclEdit.editingFinished.connect(self._on_excluded_edited)
        self.refCursorBtn.clicked.connect(lambda: self.refSpin.setValue(self._cursor_time()))
        self.tabs.currentChanged.connect(lambda _i: self._on_tab())

    def _make_point_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.pointLbl = QtWidgets.QLabel("No point.")
        self.pointLbl.setWordWrap(True)
        lv.addWidget(self.pointLbl)
        self.pixMap = PixelMapPlot()
        lv.addWidget(self.pixMap, 1)
        sl = QtWidgets.QHBoxLayout()
        sl.addWidget(QtWidgets.QLabel("Time bin"))
        self.timeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl.addWidget(self.timeSlider, 1)
        self.timeLbl = QtWidgets.QLabel("t = --")
        self.timeLbl.setMinimumWidth(110)
        sl.addWidget(self.timeLbl)
        lv.addLayout(sl)
        hint = QtWidgets.QLabel("Left click: select / deselect pixel · right click: exclude pixel from the fits")
        hint.setStyleSheet("color:#6b7280")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.decayPlot = pg.PlotWidget()
        _style(self.decayPlot.getPlotItem(), "Counts", "Time (ns)")
        self.decayPlot.addLegend(offset=(-10, 10))
        self._decayCurves: List[object] = []
        self._cursorLine = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#111827", width=1,
                                                                                style=QtCore.Qt.DashLine))
        self._refLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#059669", width=1.5))
        self._region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(6, 182, 212, 25))
        for it in (self._region, self._refLine, self._cursorLine):
            self.decayPlot.addItem(it, ignoreBounds=True)
        self._cursorLine.sigPositionChangeFinished.connect(
            lambda: self._set_cursor_time(float(self._cursorLine.value())))
        right.addWidget(self.decayPlot)

        fitw = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(fitw)
        fv.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.fitPointBtn = QtWidgets.QPushButton("Fit point")
        self.fitPointBtn.setFixedWidth(90)
        self.fitPointBtn.setToolTip("Sequential 2D Gaussian fit of every frame of the time range from the reference,\n"
                                    "then D from a line through σ²(t).")
        row.addWidget(self.fitPointBtn)
        self.pointFitLbl = QtWidgets.QLabel("Set the reference and click 'Fit point'.")
        self.pointFitLbl.setWordWrap(True)
        row.addWidget(self.pointFitLbl, 1)
        fv.addLayout(row)
        self.sigPlot = pg.PlotWidget()
        _style(self.sigPlot.getPlotItem(), "σ_eq² (µm²)", "t − t_ref (ns)")
        self._sigGood = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush("#1d4ed8"))
        self._sigBad = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#1d4ed8", width=1.2), brush=None)
        self._sigLine = self.sigPlot.plot(pen=pg.mkPen("#dc2626", width=1.6, style=QtCore.Qt.DashLine))
        self.sigPlot.addItem(self._sigGood)
        self.sigPlot.addItem(self._sigBad)
        fv.addWidget(self.sigPlot, 1)
        right.addWidget(fitw)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        h.addWidget(split)

        self.pixMap.pixel_clicked.connect(self._on_pixel_clicked)
        self.timeSlider.valueChanged.connect(lambda _v: self._update_cursor())
        self.fitPointBtn.clicked.connect(self._on_fit_point)
        return page

    def _make_sweep_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.normCheck = QtWidgets.QCheckBox("Normalize each point")
        self.mapLogCheck = QtWidgets.QCheckBox("Log₁₀ colour")
        self.fitSweepBtn = QtWidgets.QPushButton("Fit sweep")
        self.fitSweepBtn.setToolTip("'Fit point' for every point of the current sweep line (in the background),\n"
                                    "then D and σ0² versus the swept power / gate.")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.exportBtn = QtWidgets.QPushButton("Export D…")
        for b in (self.fitSweepBtn, self.cancelBtn, self.exportBtn):
            b.setFixedWidth(90)
        self.cancelBtn.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(160)
        self.perSecCheck = QtWidgets.QCheckBox("÷ acq time")
        self.perSecCheck.setToolTip("Dynamic TRPL time: show counts per second, so spots with different times\n"
                                    "compare fairly (decay map and counts vs sweep; the fits do not change).")
        self.skipLowCheck = QtWidgets.QCheckBox("Skip below-target spots")
        self.skipLowCheck.setToolTip("Dynamic TRPL time: leave out the spots that hit the max time before their\n"
                                     "peak reached the target (red rings in the plots).")
        self.perSecCheck.setVisible(False)
        self.skipLowCheck.setVisible(False)
        for w in (self.normCheck, self.mapLogCheck, self.perSecCheck, self.skipLowCheck, self.fitSweepBtn,
                  self.cancelBtn, self.progress, self.exportBtn):
            row.addWidget(w)
        self.sweepLbl = QtWidgets.QLabel("")
        self.sweepLbl.setStyleSheet("color:#4b5563")
        self.sweepLbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row.addWidget(self.sweepLbl, 1)
        v.addLayout(row)
        self.fitLbl = QtWidgets.QLabel("")
        self.fitLbl.setWordWrap(True)
        v.addWidget(self.fitLbl)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.decayMap = MapPlotWidget("Decay vs sweep", cmap_name="viridis", line_label="Counts")
        split.addWidget(self.decayMap)
        tabs = QtWidgets.QTabWidget()
        glw = pg.GraphicsLayoutWidget()
        self.pD = glw.addPlot(row=0, col=0)
        self.pS0 = glw.addPlot(row=1, col=0)
        self.pS0.setXLink(self.pD)
        _style(self.pD, "D (cm²/s)")
        _style(self.pS0, "σ₀² (µm²)")
        self.dCurve = self.pD.plot(pen=pg.mkPen("#1d4ed8", width=1.2), symbol="o", symbolSize=5,
                                   symbolBrush="#1d4ed8", symbolPen=None)
        self.dErr = pg.ErrorBarItem(pen=pg.mkPen("#1d4ed8", width=1), beam=0.0)
        self.s0Curve = self.pS0.plot(pen=pg.mkPen("#059669", width=1.2), symbol="o", symbolSize=5,
                                     symbolBrush="#059669", symbolPen=None)
        self.s0Err = pg.ErrorBarItem(pen=pg.mkPen("#059669", width=1), beam=0.0)
        self.pD.addItem(self.dErr)
        self.pS0.addItem(self.s0Err)
        ring = dict(size=11, brush=None, pen=pg.mkPen("#dc2626", width=1.6))
        self.dLow, self.s0Low = pg.ScatterPlotItem(**ring), pg.ScatterPlotItem(**ring)
        self.pD.addItem(self.dLow)
        self.pS0.addItem(self.s0Low)
        tabs.addTab(glw, "D vs sweep")
        self.sigAll = pg.PlotWidget()
        _style(self.sigAll.getPlotItem(), "σ_eq² (µm²)", "t − t_ref (ns)")
        self._sigAllItems: List[object] = []
        tabs.addTab(self.sigAll, "σ²(t) curves")
        self.cntPlot = pg.PlotWidget()
        _style(self.cntPlot.getPlotItem(), "Counts in the time range")
        self.cntCurve = self.cntPlot.plot(pen=pg.mkPen("#d62728", width=1.2), symbol="o", symbolSize=5,
                                          symbolBrush="#d62728", symbolPen=None)
        self.cntLow = pg.ScatterPlotItem(size=11, brush=None, pen=pg.mkPen("#dc2626", width=1.6))
        self.cntPlot.addItem(self.cntLow)
        tabs.addTab(self.cntPlot, "Counts vs sweep")
        self._markers = []
        for p in (self.pD, self.pS0, self.cntPlot.getPlotItem()):
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#06b6d4", width=1.2))
            p.addItem(ln, ignoreBounds=True)
            self._markers.append(ln)
        split.addWidget(tabs)
        v.addWidget(split, 1)

        self.normCheck.toggled.connect(lambda _v: self._redraw_map())
        self.mapLogCheck.toggled.connect(lambda _v: self._redraw_map())
        self.perSecCheck.toggled.connect(lambda _v: self._redraw_map())
        self.skipLowCheck.toggled.connect(lambda _v: self._apply_skip_low())
        self.fitSweepBtn.clicked.connect(self._on_fit_sweep)
        self.cancelBtn.clicked.connect(self.cancel)
        self.exportBtn.clicked.connect(self._on_export)
        self._sweep_page = page
        return page

    # ── settings ──────────────────────────────────────────────────────

    def view_settings(self) -> ViewSettings:
        return ViewSettings(self.alignCheck.isChecked(), self.t0_ref, self.rebinSpin.value())

    def time_range(self) -> Optional[Tuple[float, float]]:
        if self.fullRangeCheck.isChecked():
            return None
        return tuple(sorted((float(self.t0Spin.value()), float(self.t1Spin.value()))))

    def fit_settings(self) -> FitSettings:
        win = None if self.autoWinCheck.isChecked() else tuple(sorted((self.w0Spin.value(), self.w1Spin.value())))
        key = self.bgCombo.currentData()
        dark, dark_id = (None, "")
        if key == "dark_ref":
            dark, dark_id = self._dark_scaled()
        return FitSettings(self.view_settings(), self.time_range(), self.refSpin.value(), key == "pre_t0",
                           win, self.fixOffsetCheck.isChecked(), self.excluded, dark, dark_id,
                           dark_per_s=bool(self._dyn))

    # ── dynamic TRPL time (dynamic_trpl_log.csv) ──────────────────────
    def _acq_of(self, path) -> Optional[float]:
        """Acquisition time (s) of one TRPL file: its own (dynamic) or session.md's (fixed)."""
        if path is None:
            return None
        rec = self._dyn.get(Path(path).name)
        return rec["total_s"] if rec is not None else self._acq_s

    def _below_target(self, path) -> bool:
        rec = self._dyn.get(Path(path).name) if path is not None else None
        return rec is not None and not rec["reached"]

    def _dyn_text(self, path) -> str:
        rec = self._dyn.get(Path(path).name) if path is not None else None
        if rec is None:
            return ""
        txt = (f"dynamic: {rec['total_s']:g} s in {rec['n_chunks']} chunk(s), peak {rec['peak']:.0f}/"
               f"{rec['target']:g}")
        return txt + ("" if rec["reached"] else "  — BELOW TARGET (max time)")

    def _apply_skip_low(self) -> None:
        """'Skip below-target spots': those points get no TRPL file (left out of the map and the fits)."""
        if self.skipLowCheck.isChecked() and self._dyn:
            pts = [(x, lab, None if self._below_target(pth) else pth) for x, lab, pth in self._points_all]
        else:
            pts = list(self._points_all)
        changed = [str(p[2]) for p in pts] != [str(p[2]) for p in self._points]
        self._points = pts
        if changed:
            self._schedule_map()
            self._redraw_fits()

    # ── dark reference (DARK_TRPL_*.csv of the folder) ─────────────────
    def _dark_reason(self) -> str:
        """Why the dark reference cannot be used (empty = it can)."""
        if self._dark is None:
            return self._dark_err or "no DARK_TRPL_*.csv in this folder"
        if self._acq_s is None and not self._dyn:
            return "session.md has no 'Acquisition (s)': the dark cannot be scaled"
        if self._dark.num_pixels != self.geom.num_pixels:
            return f"the dark reference has {self._dark.num_pixels} pixels, the data {self.geom.num_pixels}"
        return ""

    def _dark_scaled(self) -> Tuple[Optional[np.ndarray], str]:
        """The dark reference in counts per bin of the current view, and an id for the fit cache."""
        if self._dark_reason() or self._t.size < 2:
            return None, ""
        bin_ps = float(np.median(np.diff(self._t))) * 1000.0
        if self._dyn:   # per second: fit_point scales it by each file's own time
            return self._dark.per_bin(1.0, bin_ps), f"{Path(self._dark.path).name}|per_s|{bin_ps:.6g}"
        return self._dark.per_bin(self._acq_s, bin_ps), f"{Path(self._dark.path).name}|{self._acq_s:g}|{bin_ps:.6g}"

    def _on_bg_changed(self) -> None:
        self._bg_fallback = False  # chosen by the user: keep it
        self._on_setting(self.bgCombo)

    def _update_dark_item(self) -> None:
        """Dark choice enabled only when usable; an automatic fallback to the pre-t0 offset is undone
        once the dark becomes usable (e.g. after the first point of a new folder has loaded)."""
        i = self.bgCombo.findData("dark_ref")
        why = self._dark_reason()
        item = self.bgCombo.model().item(i)
        item.setEnabled(not why)
        item.setToolTip(why or self._dark.describe())
        self.bgCombo.blockSignals(True)
        if why and self.bgCombo.currentIndex() == i:
            self.bgCombo.setCurrentIndex(self.bgCombo.findData("pre_t0"))
            self._bg_fallback = True
        elif not why and self._bg_fallback:
            self.bgCombo.setCurrentIndex(i)
            self._bg_fallback = False
        self.bgCombo.blockSignals(False)

    def _line_window(self):
        d1 = float(self.d1Spin.value())
        return float(self.d0Spin.value()), (None if d1 <= 0 else d1)

    def apply_session(self, session: dict) -> None:
        """Alignment, time range and reference as they were in the measuring GUI."""
        widgets = (self.alignCheck, self.t0Spin, self.t1Spin, self.refSpin, self.fullRangeCheck, self.refAutoCheck)
        for w in widgets:
            w.blockSignals(True)
        if "align" in session:
            self.alignCheck.setChecked(bool(session["align"]))
        if "hist_ns" in session:
            self.t0Spin.setValue(session["hist_ns"][0])
            self.t1Spin.setValue(session["hist_ns"][1])
            self.fullRangeCheck.setChecked(False)
        if "ref_ns" in session:
            self.refSpin.setValue(session["ref_ns"])
        if "hist_ns" in session or "ref_ns" in session:
            # older GUIs saved raw times: converted at the first aligned point (_sync_time_frame)
            self._times_after_t0 = bool(session.get("hist_after_t0", False))
        self.refAutoCheck.setChecked(bool(session.get("ref_auto", True)))
        for w in widgets:
            w.blockSignals(False)

    def _set_ref_auto(self, on: bool) -> None:
        self.refAutoCheck.blockSignals(True)
        self.refAutoCheck.setChecked(bool(on))
        self.refAutoCheck.blockSignals(False)

    def _on_ref_auto(self, on: bool) -> None:
        if on and self._apply_auto_ref():
            self._on_setting(self.refSpin)

    def _apply_auto_ref(self) -> bool:
        """Reference = time zero + T0_REF_BINS shown bins (alignment on, 't0+N' checked); True if it changed."""
        if not (self.refAutoCheck.isChecked() and self.alignCheck.isChecked() and self._times_after_t0
                and self._t.size):
            return False
        v = round(float(self._t[t0_ref_index(self._t)]), 6)
        if abs(float(self.refSpin.value()) - v) < 1e-9:
            return False
        self.refSpin.blockSignals(True)
        self.refSpin.setValue(v)
        self.refSpin.blockSignals(False)
        return True

    def _sync_time_frame(self, off: Optional[float]) -> None:
        """The view switches between raw times (off None) and times after time zero: move the time range,
        reference and manual pre-t0 window by the offset so they stay on the same data."""
        after = off is not None
        if after:
            self._last_off = off
        if after == self._times_after_t0:
            return
        d = -float(off) if after else float(self._last_off or 0.0)
        spins = [self.t0Spin, self.t1Spin, self.refSpin]
        if not self.autoWinCheck.isChecked():
            spins += [self.w0Spin, self.w1Spin]
        for sp in spins:
            sp.blockSignals(True)
            sp.setValue(sp.value() + d)
            sp.blockSignals(False)
        self._times_after_t0 = after

    def _on_setting(self, w) -> None:
        if w in (self.alignCheck, self.rebinSpin):
            if w is self.alignCheck:
                self.align_requested.emit()
            self._reload_point()
            self._schedule_map()
        elif w in (self.d0Spin, self.d1Spin, self.flaggedCheck):
            self._draw_point_fit()
            self._redraw_fits()
            return
        elif w in (self.t0Spin, self.t1Spin, self.fullRangeCheck):
            self._redraw_point()
            self._redraw_map()
        else:
            self._redraw_point()
        self._draw_point_fit()
        self._redraw_fits()

    def _on_excluded_edited(self) -> None:
        ex = set()
        for tok in self.exclEdit.text().replace(";", ",").replace(" ", ",").split(","):
            if tok.strip().isdigit() and int(tok) < self.geom.num_pixels:
                ex.add(int(tok))
        self._set_excluded(ex)

    def _set_geometry(self, geom: SpadGeometry) -> None:
        """Data of another detector: map rebuilt, selection back to its center pixel, exclusions cleared."""
        if geom is self.geom:
            return
        self.geom = geom
        self.sel = {geom.center_pixel}
        self.excluded = set()
        self.exclEdit.setText("")
        self.pixMap.set_geometry(geom)
        self._point_fit = self._point_fit_key = None
        self.status.emit(f"TRPL data from the {geom.label}: pixel map, selection and exclusions reset.")

    def _set_excluded(self, ex: set) -> None:
        self.excluded = set(ex)
        self.exclEdit.setText(", ".join(str(p) for p in sorted(self.excluded)))
        self._redraw_point()
        self._draw_point_fit()
        self._redraw_fits()

    # ── data in ───────────────────────────────────────────────────────

    def set_folder(self, folder: Path, session: dict) -> None:
        self.folder = folder
        self._path = None
        self._sweep = self._sweep_key = None
        self._fits = self._fits_key = None
        self._point_fit = self._point_fit_key = None
        self.apply_session(session)
        # dark reference: the newest DARK_TRPL_*.csv the measuring GUI copied into the folder
        self._acq_s = session.get("acq_s")
        self._dyn = read_dynamic_log(folder)
        for w in (self.perSecCheck, self.skipLowCheck):
            w.setVisible(bool(self._dyn))
        self.skipLowCheck.blockSignals(True)
        self.skipLowCheck.setChecked(False)
        self.skipLowCheck.blockSignals(False)
        self.perSecCheck.blockSignals(True)
        times = {round(r["total_s"], 6) for r in self._dyn.values()}
        self.perSecCheck.setChecked(len(times) > 1)       # times differ: counts per second by default
        self.perSecCheck.blockSignals(False)
        if self._dyn:
            spots = [r for r in self._dyn.values() if r["pump2"] != "on"]
            low = sum(not r["reached"] for r in spots)
            self.status.emit(f"Dynamic TRPL time: {len(spots)} spot(s), {low} below target "
                             f"(red rings in the Sweep tab); the dark reference is scaled per file.")
        self._dark, self._dark_err = None, ""
        darks = sorted(Path(folder).glob("DARK_TRPL_*.csv"), key=lambda p: p.stat().st_mtime)
        if darks:
            try:
                self._dark = load_dark_reference(darks[-1])
            except Exception as exc:
                self._dark_err = f"{darks[-1].name}: {exc}"
        want = session.get("background") or ("dark_ref" if self._dark is not None else "pre_t0")
        self.bgCombo.blockSignals(True)
        self.bgCombo.setCurrentIndex(max(0, self.bgCombo.findData(want)))
        self.bgCombo.blockSignals(False)
        self._bg_fallback = False
        self._update_dark_item()

    def set_sweep_t0(self, t0: Optional[float], ready: bool) -> None:
        changed = t0 != self.t0_ref
        self.t0_ref, self.t0_ready = t0, bool(ready)
        if changed and self.alignCheck.isChecked():
            self._reload_point()
        self._schedule_map()

    def show_point(self, path: Optional[Path], label: str) -> None:
        if str(path) == str(self._path) and self._view_key == self.view_settings().key():
            self._label = label
            return
        self._path, self._label = (None if path is None else str(path)), label
        self._reload_point()

    def set_points(self, points, x_label: str, marker: Optional[float], context: str) -> None:
        self._points_all = list(points)
        if self.skipLowCheck.isChecked() and self._dyn:
            points = [(x, lab, None if self._below_target(pth) else pth) for x, lab, pth in points]
        changed = [(p[0], str(p[2])) for p in points] != [(p[0], str(p[2])) for p in self._points]
        self._points, self._x_label, self._marker, self._context = list(points), x_label, marker, context
        if changed:
            self._schedule_map()
        if self.tabs.currentWidget() is self._sweep_page:
            self._redraw_fits()

    # ── point ─────────────────────────────────────────────────────────

    def _reload_point(self) -> None:
        vs = self.view_settings()
        self._view_key = vs.key()
        self._tz = None
        if self._path is None:
            self._t, self._c = np.zeros(0), np.zeros((0, self.geom.num_pixels))
            self.pointLbl.setText(f"No TRPL file at this point ({self._label}).")
            self.pointLbl.setStyleSheet("color:#b45309")
        else:
            try:
                data = self.store.load(self._path)
                self._set_geometry(data.geom)
                self._t, self._c, self._tz, note = point_view(self.store, data, vs)
                off = view_offset(data.time_ns, self.store.zero_of(data, self._tz, vs.t0_ref)) if vs.align else None
                # aligned: converted once the sweep's time zero is known (a point's own differs by a few ns)
                if not vs.align or (off is not None and self.t0_ready):
                    self._sync_time_frame(off)
                self._apply_auto_ref()
                dyn = self._dyn_text(self._path)
                self.pointLbl.setStyleSheet("color:#b91c1c" if self._below_target(self._path) else "")
                self.pointLbl.setText(f"{self._label}  —  {Path(self._path).name}  [{data.geom.label}]"
                                      + (f"  ({note})" if note else "") + (f"  —  {dyn}" if dyn else ""))
            except Exception as exc:
                self._t, self._c = np.zeros(0), np.zeros((0, self.geom.num_pixels))
                self.pointLbl.setStyleSheet("color:#b91c1c")
                self.pointLbl.setText(f"Could not load {Path(self._path).name}: {exc}")
        n = self._t.size
        old_t = self._cursor_time() if self.timeSlider.maximum() > 0 else None
        self.timeSlider.blockSignals(True)
        self.timeSlider.setRange(0, max(n - 1, 0))
        if n and (old_t is None or not (self._t[0] <= old_t <= self._t[-1])):
            self.timeSlider.setValue(int(np.argmax(self._c.sum(axis=1))))
        elif n:
            self.timeSlider.setValue(int(np.argmin(np.abs(self._t - old_t))))
        self.timeSlider.blockSignals(False)
        if self.alignCheck.isChecked():
            self.infoLbl.setText((self._tz.message + "  Time axis: t = 0 at the signal peak.") if self._tz is not None else
                                 ("Waiting for the alignment of all TRPL files …" if not self.t0_ready else ""))
        else:
            self.infoLbl.setText(f"{n} bins of {np.median(np.diff(self._t)):.3g} ns" if n > 1 else "")
        self._update_dark_item()
        dark, _ = self._dark_scaled()
        if dark is not None and self._dyn:
            acq = self._acq_of(self._path)
            dark = dark * acq if acq is not None else None
        if dark is not None and n > 3:
            win = pre_pulse_window(self._t, self._c.sum(axis=1))
            sel = (self._t >= win[0]) & (self._t <= win[1]) if win is not None else np.zeros(n, bool)
            if sel.any():
                m = np.ones(self.geom.num_pixels, bool)
                m[[p for p in self.excluded if p < m.size]] = False
                cmp = dark_compare_text(self._c[sel].mean(axis=0), dark, m)
                if cmp:
                    self.infoLbl.setText((self.infoLbl.text() + "  |  " if self.infoLbl.text() else "") + cmp)
        self._redraw_point()
        self._draw_point_fit()

    def _cursor_time(self) -> float:
        if not self._t.size:
            return 0.0
        return float(self._t[int(np.clip(self.timeSlider.value(), 0, self._t.size - 1))])

    def _set_cursor_time(self, t: float) -> None:
        if self._t.size:
            self.timeSlider.setValue(int(np.argmin(np.abs(self._t - t))))

    def _redraw_point(self) -> None:
        pi = self.decayPlot.getPlotItem()
        for c in self._decayCurves:
            pi.removeItem(c)
        self._decayCurves = []
        pi.legend.clear()
        self.pixMap.set_selected(self.sel, self.excluded)
        if not self._t.size:
            self.pixMap.set_values(np.full(self.geom.num_pixels, np.nan), self.excluded)
            return
        r0, r1 = range_bins(self._t, self.time_range())
        t = self._t[r0:r1 + 1]
        log = self.logCheck.isChecked()
        pi.setLogMode(y=log)
        for k, p in enumerate(sorted(self.sel)):
            y = self._c[r0:r1 + 1, p].astype(float)
            if log:
                y = np.clip(y, 0.5, None)
            c = pi.plot(t, y, pen=pg.mkPen(_TRACE_COLORS[k % len(_TRACE_COLORS)], width=1.1), name=f"px {p}")
            self._decayCurves.append(c)
        self._region.setRegion((float(t[0]), float(t[-1])))
        self._refLine.setValue(float(self.refSpin.value()))
        pi.setXRange(float(t[0]), float(t[-1]), padding=0.01)
        self._update_cursor()

    def _update_cursor(self) -> None:
        if not self._t.size:
            return
        i = int(np.clip(self.timeSlider.value(), 0, self._t.size - 1))
        self.timeLbl.setText(f"t = {self._t[i]:.3f} ns")
        self._cursorLine.setValue(float(self._t[i]))
        self.pixMap.set_values(self._c[i].astype(float), self.excluded)

    def _on_pixel_clicked(self, p: int, button: int) -> None:
        if button == 2:
            self._set_excluded(self.excluded ^ {p})
            return
        if p in self.sel and len(self.sel) > 1:
            self.sel.discard(p)
        else:
            self.sel.add(p)
        self._redraw_point()
        self._redraw_map()

    def _on_fit_point(self) -> None:
        if not self._t.size or self._point_job is not None:
            return
        s = self.fit_settings()
        t, c, key = self._t, self._c, (self._path, s.key())
        self.pointFitLbl.setText("Fitting …")
        acq = self._acq_of(self._path)
        self._point_job = self._run(lambda prog, cancelled: fit_point(t, c, s, cancelled, acq_s=acq),
                                    lambda res, err: self._on_point_fit_done(res, err, key))

    def _on_point_fit_done(self, res, err: str, key) -> None:
        self._point_job = None
        if err:
            self.pointFitLbl.setText(f"Fit failed: {err}")
            return
        self._point_fit, self._point_fit_key = res, key
        self._draw_point_fit()

    def _draw_point_fit(self) -> None:
        f = self._point_fit
        if f is None or self._point_fit_key != (self._path, self.fit_settings().key()):
            self._sigGood.setData([], [])
            self._sigBad.setData([], [])
            self._sigLine.setData([], [])
            if f is not None:
                self.pointFitLbl.setText("Point or settings changed: click 'Fit point'.")
            return
        tr = f.t_ns - f.ref_ns
        self._sigGood.setData(tr[~f.flagged], f.sigma2[~f.flagged])
        self._sigBad.setData(tr[f.flagged], f.sigma2[f.flagged])
        ln = fit_line(f, self._line_window(), self.flaggedCheck.isChecked())
        text = (f"{f.t_ns.size} frames fitted ({int(f.flagged.sum())} flagged = hollow: at a bound or low signal), "
                f"{f.failures} failed. ")
        if ln is None:
            self._sigLine.setData([], [])
            text += "Fewer than 3 usable bins in the σ² line window."
        else:
            hi = float(tr.max()) if self._line_window()[1] is None else min(self._line_window()[1], float(tr.max()))
            xx = np.array([self._line_window()[0], hi])
            self._sigLine.setData(xx, ln["s0"] + ln["slope"] * xx)
            text += (f"<b>D = {ln['D']:.3g} ± {ln['D_err']:.2g} cm²/s</b>, σ₀² = {ln['s0']:.3g} ± "
                     f"{ln['s0_err']:.2g} µm² ({ln['n']} bins)")
        if f.note:
            text += f"  [{f.note}]"
        self.pointFitLbl.setText(text)

    # ── sweep ─────────────────────────────────────────────────────────

    def _sweep_key_now(self):
        return (tuple(str(p[2]) for p in self._points), self.view_settings().key())

    def _schedule_map(self) -> None:
        if self.tabs.currentWidget() is self._sweep_page and self.isVisible():
            self._map_timer.start()

    def _on_tab(self) -> None:
        if self.tabs.currentWidget() is self._sweep_page:
            self.sync_requested.emit()
            self._map_timer.start()
            self._redraw_fits()

    def _start_map_job(self) -> None:
        key = self._sweep_key_now()
        if key == self._sweep_key:
            self._redraw_map()
            return
        if not any(p[2] is not None for p in self._points):
            self.sweepLbl.setText("No TRPL files on this sweep line.")
            return
        if self.alignCheck.isChecked() and not self.t0_ready:
            self.sweepLbl.setText("Waiting for the time-zero alignment of all TRPL files …")
            return
        if self._map_job is not None:
            self._map_job.cancel()
        paths, vs = [p[2] for p in self._points], self.view_settings()
        self._map_job = self._run(lambda prog, cancelled: process_sweep(self.store, paths, vs, None, prog, cancelled),
                                  lambda res, err, j=None: self._on_map_done(res, err, key),
                                  progress=lambda d, n: self.sweepLbl.setText(f"Loading TRPL files {d}/{n} …"))

    def _on_map_done(self, res, err: str, key) -> None:
        if key != self._sweep_key_now():
            return                                       # the sweep line changed meanwhile
        self._map_job = None
        if err or res is None or res.get("cancelled"):
            self.sweepLbl.setText(f"Decay map failed: {err}" if err else "")
            return
        self._sweep, self._sweep_key = res, key
        if res.get("traces") is not None and res["traces"].shape[2] != self.geom.num_pixels:
            self._set_geometry(geometry_for(res["traces"].shape[2]))
            self._redraw_point()
        if res["notes"]:
            self.status.emit("; ".join(res["notes"][:3]))
        self._redraw_map()

    def _decays(self):
        res = self._sweep
        if res is None or res["traces"] is None or res["traces"].shape[0] != len(self._points):
            return None
        r0, r1 = range_bins(res["t"], self.time_range())
        sel = [p for p in sorted(self.sel) if p < res["traces"].shape[2]]
        z = res["traces"][:, r0:r1 + 1, :][:, :, sel].sum(axis=2, dtype=np.float64)
        return np.array([p[0] for p in self._points], float), res["t"][r0:r1 + 1], z

    def _redraw_map(self) -> None:
        d = self._decays()
        if d is None:
            return
        x, t, z = d
        per_s = self.perSecCheck.isChecked() and bool(self._dyn)
        if per_s:
            acq = np.array([np.nan if self._acq_of(p[2]) is None else self._acq_of(p[2]) for p in self._points], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = z / acq[:, None]
        have = np.isfinite(z).any(axis=1)
        counts = np.where(have, np.nansum(z, axis=1), np.nan)
        zz = z.copy()
        if self.normCheck.isChecked():
            peak = np.nanmax(np.where(np.isfinite(zz), zz, -np.inf), axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                zz = np.where(peak > 0, zz / peak, np.nan)
        if self.mapLogCheck.isChecked():
            zz = np.log10(np.clip(zz, 1e-3 if self.normCheck.isChecked() else 0.5, None))
        px = ", ".join(str(p) for p in sorted(self.sel))
        self.decayMap.setTitle(f"Decay vs sweep — px {px}" + (f" — {self._context}" if self._context else ""))
        self.decayMap.set_map(x, t, zz.T, x_label=self._x_label, y_label="Time (ns)")
        self.cntCurve.setData(x[have], counts[have])
        low = np.array([self._below_target(p[2]) for p in self._points], bool) & have
        self.cntLow.setData(x[low], counts[low])
        self.cntPlot.setLabel("bottom", self._x_label)
        self.cntPlot.setLabel("left", "Counts / s in the time range" if per_s else "Counts in the time range")
        self.cntPlot.setTitle(f"Counts{' / s' if per_s else ''} of px {px} in {t[0]:.2f}–{t[-1]:.2f} ns")
        text = f"{int(have.sum())}/{len(self._points)} points with TRPL, {t.size} bins"
        if self._dyn:
            n_low = sum(self._below_target(p[2]) for p in self._points_all)
            text += (f"; dynamic time, {n_low} below target"
                     + (" (skipped)" if self.skipLowCheck.isChecked() and n_low else "")
                     + ("; counts / s" if per_s else ""))
        self.sweepLbl.setText(text)
        self._draw_markers()

    def _draw_markers(self) -> None:
        for ln in self._markers:
            ln.setVisible(self._marker is not None)
            if self._marker is not None:
                ln.setValue(float(self._marker))
        if self._marker is not None and self._sweep is not None:
            self.decayMap.set_x_marker(self._marker)

    def _on_fit_sweep(self) -> None:
        if self._fit_job is not None:
            return
        if not any(p[2] is not None for p in self._points):
            QtWidgets.QMessageBox.information(self, "Fit sweep", "No TRPL files on this sweep line.")
            return
        if self.alignCheck.isChecked() and not self.t0_ready:
            QtWidgets.QMessageBox.information(self, "Fit sweep", "The time-zero alignment is still running.")
            return
        s = self.fit_settings()
        paths, key = [p[2] for p in self._points], (self._sweep_key_now(), s.key())
        self.fitLbl.setText(f"Fitting {sum(p is not None for p in paths)} points: {s.describe()} …")
        self.fitSweepBtn.setEnabled(False)
        self.cancelBtn.setEnabled(True)
        acq_of = self._acq_of
        self._fit_job = self._run(lambda prog, cancelled: process_sweep(self.store, paths, s.view, s, prog, cancelled,
                                                                        acq_of=acq_of),
                                  lambda res, err: self._on_fits_done(res, err, key, s),
                                  progress=self._on_fit_progress)

    def _on_fit_progress(self, d: int, n: int) -> None:
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(d)

    def _on_fits_done(self, res, err: str, key, s: FitSettings) -> None:
        self._fit_job = None
        self.fitSweepBtn.setEnabled(True)
        self.cancelBtn.setEnabled(False)
        if err or res is None or res.get("cancelled"):
            self.fitLbl.setText(f"Fit sweep failed: {err}" if err else "Fit sweep cancelled.")
            return
        self._fits, self._fits_key, self._fits_settings, self._fit_notes = res["fits"], key, s, res["notes"]
        self._redraw_fits()

    def _valid_fits(self):
        if self._fits is None or self._fits_key is None or self._fits_key[0] != self._sweep_key_now():
            return None
        return self._fits

    def _lines(self, fits):
        win, fl = self._line_window(), self.flaggedCheck.isChecked()
        return [None if f is None else fit_line(f, win, fl) for f in fits]

    def _redraw_fits(self) -> None:
        for it in self._sigAllItems:
            self.sigAll.removeItem(it)
        self._sigAllItems = []
        for c, e in ((self.dCurve, self.dErr), (self.s0Curve, self.s0Err)):
            c.setData([], [])
            e.setData(x=np.zeros(0), y=np.zeros(0), top=np.zeros(0), bottom=np.zeros(0))
        self.dLow.setData([], [])
        self.s0Low.setData([], [])
        self.pS0.setLabel("bottom", self._x_label)
        self._draw_markers()
        fits = self._valid_fits()
        if fits is None:
            self.fitLbl.setText("Click 'Fit sweep' to fit every point of this sweep line."
                                if self._fits is None else "The sweep line changed: click 'Fit sweep'.")
            return
        x = np.array([p[0] for p in self._points], float)
        lines = self._lines(fits)
        for curve, err, key, ekey in ((self.dCurve, self.dErr, "D", "D_err"), (self.s0Curve, self.s0Err, "s0", "s0_err")):
            y = np.array([np.nan if ln is None else ln[key] for ln in lines])
            ye = np.array([np.nan if ln is None else ln[ekey] for ln in lines])
            ok = np.isfinite(y)
            curve.setData(x[ok], y[ok])
            eo = ok & np.isfinite(ye)
            err.setData(x=x[eo], y=y[eo], top=ye[eo], bottom=ye[eo])
            low = ok & np.array([self._below_target(p[2]) for p in self._points], bool)
            (self.dLow if key == "D" else self.s0Low).setData(x[low], y[low])
        lo, hi = self._line_window()
        self.pD.setTitle(f"σ² = σ₀² + 2D(t − t_ref) over {lo:g}–{'end' if hi is None else f'{hi:g}'} ns")
        cmap = _pg_cmap("viridis")
        use_fl = self.flaggedCheck.isChecked()
        for k, f in enumerate(fits):
            if f is None or not f.t_ns.size:
                continue
            ok = np.ones(f.t_ns.size, bool) if use_fl else ~f.flagged
            col = cmap.map(k / max(len(fits) - 1, 1), mode="qcolor")
            self._sigAllItems.append(self.sigAll.plot(f.t_ns[ok] - f.ref_ns, f.sigma2[ok], pen=pg.mkPen(col, width=1)))
        self.sigAll.setTitle(f"σ²(t) of every point (colour: first → last of {self._x_label})")
        n_bins = sum(f.t_ns.size for f in fits if f is not None)
        n_flag = sum(int(f.flagged.sum()) for f in fits if f is not None)
        text = (f"{sum(f is not None for f in fits)}/{len(fits)} points fitted ({n_flag}/{n_bins} bins flagged), "
                f"{sum(ln is not None for ln in lines)} with a σ² line. {self._fits_settings.describe()}.")
        if self._fits_key[1] != self.fit_settings().key():
            text += "  <b>Settings changed since this fit</b>: click 'Fit sweep' to refit."
        if self._fit_notes:
            text += "  " + "; ".join(self._fit_notes[:2])
        n_low = sum(self._below_target(p[2]) for p in self._points)
        if n_low:
            text += f"  <span style='color:#dc2626'>{n_low} point(s) below the count target (red rings).</span>"
        self.fitLbl.setText(text)

    def _on_export(self) -> None:
        fits = self._valid_fits()
        if fits is None:
            QtWidgets.QMessageBox.information(self, "Export", "No sweep fit: click 'Fit sweep' first.")
            return
        default = str((self.folder or Path.home()) / "trpl_diffusion_vs_sweep.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export D vs sweep", default, "CSV (*.csv)")
        if not path:
            return
        x = np.array([p[0] for p in self._points], float)
        lines = self._lines(fits)
        d = self._decays()
        counts = np.full(x.size, np.nan) if d is None else np.where(np.isfinite(d[2]).any(axis=1),
                                                                     np.nansum(d[2], axis=1), np.nan)
        lo, hi = self._line_window()
        head = [f"folder: {self.folder}", f"sweep: {self._context}", f"x: {self._x_label}",
                f"fit: {self._fits_settings.describe()}",
                f"sigma2 line: {lo:g} to {'end' if hi is None else f'{hi:g}'} ns after the reference, "
                f"flagged fits {'included' if self.flaggedCheck.isChecked() else 'left out'}",
                "D from sigma^2 = sigma0^2 + 2 D (t - t_ref); 1 um^2/ns = 10 cm^2/s"]

        def col(key):
            return [np.nan if ln is None else ln[key] for ln in lines]

        table = np.column_stack([x, col("D"), col("D_err"), col("s0"), col("s0_err"), col("n"),
                                 [np.nan if f is None else f.failures for f in fits], counts])
        cols = "x,D_cm2_s,D_err_cm2_s,sigma0_2_um2,sigma0_2_err_um2,n_bins,fit_failures,counts_in_range"
        if self._dyn:   # dynamic TRPL time: each point's own time and whether it reached the target
            acq = [np.nan if self._acq_of(p[2]) is None else self._acq_of(p[2]) for p in self._points]
            low = [np.nan if p[2] is None else float(self._below_target(p[2])) for p in self._points]
            table = np.column_stack([table, acq, low])
            cols += ",acq_s,below_target"
            head.append("dynamic TRPL time (dynamic_trpl_log.csv): counts_in_range "
                        + ("per second" if self.perSecCheck.isChecked() else "raw"))
        try:
            np.savetxt(path, table, delimiter=",", fmt="%.10g", comments="# ",
                       header="\n".join(head) + "\n" + cols)
            rows = [(k, x[k], t, t - f.ref_ns, s2, fl) for k, f in enumerate(fits) if f is not None
                    for t, s2, fl in zip(f.t_ns, f.sigma2, f.flagged)]
            path2 = os.path.splitext(path)[0] + "_sigma2_t.csv"
            np.savetxt(path2, np.array(rows, dtype=float).reshape(-1, 6), delimiter=",", fmt="%.10g", comments="# ",
                       header="\n".join(head[:4]) + "\npoint,x,t_ns,t_minus_ref_ns,sigma2_um2,flagged")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Saved {path} and {os.path.basename(path2)}")

    # ── threads ───────────────────────────────────────────────────────

    def _run(self, fn, on_done, progress=None) -> _Job:
        self._job_id += 1
        job = _Job(self._job_id, fn, self)
        job.done.connect(lambda _id, res, err: on_done(res, err))
        if progress is not None:
            job.progress.connect(progress)
        job.finished.connect(lambda job=job: self._on_finished(job))
        self._jobs.append(job)
        job.start()
        return job

    def _on_finished(self, job: _Job) -> None:
        job.wait(1000)      # finished is emitted just before the thread really ends
        if job in self._jobs:
            self._jobs.remove(job)
        job.deleteLater()

    @property
    def busy(self) -> bool:
        return bool(self._jobs)

    def cancel(self) -> None:
        for j in self._jobs:
            j.cancel()


# ══════════════════════════════════════════════════════════════════════
# Main window: the PL viewer v2 + pump-2 selector + TRPL tab
# ══════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.4g}" if np.isfinite(v) else "--"


class PLTRPLViewerV2(PLViewerV2):
    """PLViewerV2 unchanged for the PL; adds the pump-2 (flipper) selector and the TRPL tab."""

    def __init__(self):
        self._ready = False
        self.store = TrplStore()
        self._scans: Dict[str, FolderScan] = {}
        self._frames_cache: Dict[Tuple[str, str], List[PLFrame]] = {}
        self._trpl_only: Dict[str, bool] = {}
        self._pump2 = P2_OFF
        self._sweep_t0: Optional[float] = None
        self._align_job = None
        self._jobs: List[_Job] = []
        self._close_t0: Optional[float] = None
        super().__init__()
        self._ready = True
        self._refresh_all()

    def _build_ui(self) -> None:
        super()._build_ui()
        self.setWindowTitle("PL + TRPL Power/Gate Viewer  v2")
        root = self.centralWidget().layout()
        controls, split = root.itemAt(0).widget(), root.itemAt(1).widget()
        root.removeWidget(split)

        left = controls.layout().itemAt(0).layout()        # data-source column of the PL viewer
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Pump 2 (flipper):"))
        self.pump2Combo = QtWidgets.QComboBox()
        self.pump2Combo.setFixedWidth(150)
        self.pump2Combo.setEnabled(False)
        self.pump2Combo.setToolTip("Which pump-2 state to show: files tagged _P2off / _P2on (dI/dP mode),\n"
                                   "or 'pump2: off/on' in the .asc header.")
        row.addWidget(self.pump2Combo)
        self.pump2Lbl = QtWidgets.QLabel("")
        self.pump2Lbl.setStyleSheet("color:#555; font-size: 8pt;")
        row.addWidget(self.pump2Lbl, 1)
        left.insertLayout(1, row)

        self.trpl = TrplPanel(self.store)
        self.viewTabs = QtWidgets.QTabWidget()
        self.viewTabs.addTab(split, "PL")
        self.viewTabs.addTab(self.trpl, "TRPL")
        root.addWidget(self.viewTabs, 1)

        self.pump2Combo.currentIndexChanged.connect(self._on_pump2_changed)
        self.viewTabs.currentChanged.connect(lambda _i: self._sync_trpl())
        self.trpl.status.connect(lambda t: self.statusBar().showMessage(t, 15000))
        self.trpl.align_requested.connect(self._start_alignment)
        self.trpl.sync_requested.connect(self._sync_trpl)

    # ── loading ───────────────────────────────────────────────────────

    def _browse_data(self, key: str) -> None:
        ch = self._ch[key]
        start = str(ch.data_dir) if ch.data_dir else ""
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select PL + TRPL data folder", start)
        if chosen:
            self.load_folder(Path(chosen), key)

    def load_folder(self, folder, key: str = "") -> bool:
        scan = scan_folder(Path(folder))
        if not scan.states():
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc or TRPL_*.csv files in:\n{folder}")
            return False
        self._frames_cache = {k: v for k, v in self._frames_cache.items() if k[0] != key}
        self._scans[key] = scan
        if all(s.folder == scan.folder for s in self._scans.values()):
            self.store.clear()
        self._sweep_t0 = None
        self._sync_pump2_combo()
        self._apply_pump2(key)
        self._reconfigure_sliders()
        if key == self._current_ch_key():
            self.trpl.set_folder(scan.folder, scan.session)
        self._start_alignment()
        self._refresh_all()
        st = self._state_for(scan)
        self.statusBar().showMessage(f"Loaded {scan.folder.name}: {P2_LABELS[st]}, {scan.describe(st)}", 15000)
        return True

    def _state_for(self, scan: FolderScan) -> str:
        states = scan.states()
        return self._pump2 if self._pump2 in states else states[0]

    def _sync_pump2_combo(self) -> None:
        states = [s for s in P2_ORDER if any(s in sc.states() for sc in self._scans.values())]
        if states and self._pump2 not in states:
            self._pump2 = states[0]
        self.pump2Combo.blockSignals(True)
        self.pump2Combo.clear()
        for s in states:
            self.pump2Combo.addItem(P2_LABELS[s], s)
        self.pump2Combo.setCurrentIndex(max(0, self.pump2Combo.findData(self._pump2)))
        self.pump2Combo.blockSignals(False)
        self.pump2Combo.setEnabled(len(states) > 1)
        self._update_pump2_label()

    def _update_pump2_label(self) -> None:
        scan = self._scans.get(self._current_ch_key())
        if scan is None:
            self.pump2Lbl.setText("")
            return
        st = self._state_for(scan)
        text = scan.describe(st) + ("; files carry no pump-2 flag" if st == P2_NONE else "")
        others = [P2_LABELS[s] for s in scan.states() if s != st]
        self.pump2Lbl.setText(text + (f"  |  also in folder: {', '.join(others)}" if others else ""))
        self.pump2Lbl.setToolTip(f"{scan.folder}\n{scan.session.get('didp', '')}")

    def _apply_pump2(self, key: str) -> None:
        scan = self._scans.get(key)
        if scan is None:
            return
        st = self._state_for(scan)
        frames = self._frames_cache.get((key, st))
        if frames is None:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                frames = _load_state_frames(scan, st)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._frames_cache[(key, st)] = frames
        ch = self._ch[key]
        ch.clear_cache()
        ch.data_dir = scan.folder
        _build_channel(ch, frames)
        self._trpl_only[key] = not scan.asc.get(st)
        self._update_data_label(key)

    def _on_pump2_changed(self, _i: int) -> None:
        st = self.pump2Combo.currentData()
        if st is None or st == self._pump2:
            return
        self._pump2 = str(st)
        for key in list(self._scans):
            self._apply_pump2(key)
        self._reconfigure_sliders()
        self._update_pump2_label()
        self._refresh_all()

    # ── time-zero alignment of all TRPL files (background) ─────────────

    def _start_alignment(self) -> None:
        if self._align_job is not None:
            self._align_job.cancel()
            self._align_job = None
        paths = [p for scan in self._scans.values() for p in scan.trpl_paths()]
        if not self.trpl.alignCheck.isChecked() or not paths:
            return
        if all(self.store.has_alignment(p) for p in paths):
            self._sweep_t0 = self.store.common_t0(paths)
            self.trpl.set_sweep_t0(self._sweep_t0, True)
            return
        self.trpl.set_sweep_t0(self._sweep_t0, False)

        def work(progress, cancelled):
            for k, p in enumerate(paths):
                if cancelled():
                    return "cancelled"
                try:
                    self.store.alignment(p)
                except Exception:
                    pass
                progress(k + 1, len(paths))
            return self.store.common_t0(paths)

        job = _Job(0, work, self)
        job.progress.connect(lambda d, n: self.statusBar().showMessage(f"Time-zero alignment: {d}/{n} TRPL files"))
        job.done.connect(lambda _i, res, err, job=job: self._on_alignment_done(job, res, err))
        job.finished.connect(lambda job=job: self._on_job_finished(job))
        self._jobs.append(job)
        self._align_job = job
        job.start()

    def _on_alignment_done(self, job, t0, err: str) -> None:
        if job is not self._align_job or t0 == "cancelled":
            return
        self._align_job = None
        self._sweep_t0 = t0 if not err else None
        self.trpl.set_sweep_t0(self._sweep_t0, True)
        self.statusBar().showMessage(f"Time-zero alignment failed: {err}" if err else
                                     f"Time-zero alignment done: sweep time zero {_fmt(t0)} ns", 15000)

    def _on_job_finished(self, job) -> None:
        job.wait(1000)
        if job in self._jobs:
            self._jobs.remove(job)
        job.deleteLater()

    # ── refreshing ────────────────────────────────────────────────────

    def _is_trpl_only(self) -> bool:
        return bool(self._trpl_only.get(self._current_ch_key()))

    def _update_maps(self) -> None:
        if getattr(self, "_ready", False) and self._is_trpl_only():
            return          # no PL in this folder
        super()._update_maps()

    def _update_single_frame_view(self) -> None:
        super()._update_single_frame_view()
        scan = self._scans.get(self._current_ch_key())
        if scan is None:
            return
        st = self._state_for(scan)
        title = self.liveView._titleLbl.text()
        if st != P2_NONE:
            title += f"   [{P2_LABELS[st]}]"
        if self._is_trpl_only():
            title += "   (no PL in this folder)"
        self.liveView.set_image_title(title)

    def _refresh_all(self) -> None:
        super()._refresh_all()
        self._sync_trpl()

    def _point_label(self, ch: _Channel, fr: Optional[PLFrame]) -> str:
        if fr is None:
            return "no point"
        if ch.angle_mode:
            text = f"A = {_fmt(fr.power_w)}°"
            if ch._angle_calib_angles is not None and fr.power_w is not None:
                text += f" ({_fmt(np.interp(float(fr.power_w), ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6)} µW)"
        else:
            text = f"P = {_fmt(None if fr.power_w is None else fr.power_w * 1e6)} µW"
        text += f", Vf = {_fmt(fr.v_front)} V, Vb = {_fmt(fr.v_back)} V"
        scan = self._scans.get(self._current_ch_key())
        if scan is not None and self._state_for(scan) != P2_NONE:
            text += f", {P2_LABELS[self._state_for(scan)].lower()}"
        return text

    def _sweep_points(self, ch: _Channel):
        """(x, x label, frames, marker x) of the current sweep line, in the same order as the PL map."""
        p_key, vf_key, vb_key = self._cur_power_key(ch), self._cur_front_key(ch), self._cur_back_key(ch)
        order = None
        if self._current_mode == self.MODE_GATE and self._gate_axis == "back" and ch.is_grid:
            x, label = ch.backs.astype(float), "Back gate (V)"
            frames = [_frame_at(ch, p_key, vf_key, _ch_norm_back(ch, float(v))) for v in ch.backs]
            marker = float(ch.backs[self._back_idx]) if ch.backs.size else None
        elif self._current_mode == self.MODE_GATE:
            x, label = ch.gates.astype(float), "Front gate (V)"
            frames = [_frame_at(ch, p_key, _ch_norm_gate(ch, float(v)), vb_key) for v in ch.gates]
            marker = float(ch.gates[self._front_idx]) if ch.gates.size else None
        else:
            frames = [_frame_at(ch, _ch_norm_power(ch, float(v)), vf_key, vb_key) for v in ch.powers]
            if ch.angle_mode and ch._angle_calib_angles is not None:
                x = np.interp(ch.powers, ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6
                label, order = "Power (µW)", np.argsort(x, kind="stable")
            elif ch.angle_mode:
                x, label = ch.powers.astype(float), "Stage A angle (°)"
            else:
                x, label = ch.powers * 1e6, "Power (µW)"
            marker = float(x[self._power_idx]) if x.size else None
        if order is not None:
            x, frames = x[order], [frames[i] for i in order]
        return np.asarray(x, float), label, frames, marker

    def _sweep_context(self, ch: _Channel, fr: Optional[PLFrame]) -> str:
        parts = []
        scan = self._scans.get(self._current_ch_key())
        if scan is not None and self._state_for(scan) != P2_NONE:
            parts.append(P2_LABELS[self._state_for(scan)].lower())
        if fr is not None:
            if self._current_mode == self.MODE_GATE:
                parts.append(f"A = {_fmt(fr.power_w)}°" if ch.angle_mode else
                             f"P = {_fmt(None if fr.power_w is None else fr.power_w * 1e6)} µW")
            else:
                parts.append(f"Vf = {_fmt(fr.v_front)} V, Vb = {_fmt(fr.v_back)} V")
        return ", ".join(parts)

    def _sync_trpl(self) -> None:
        """Send the current point and sweep line to the TRPL tab (only while it is shown)."""
        if not getattr(self, "_ready", False) or self.viewTabs.currentWidget() is not self.trpl:
            return
        key = self._current_ch_key()
        ch, scan = self._ch[key], self._scans.get(key)
        if scan is None or not ch.has_data:
            return
        st = self._state_for(scan)
        fr = self._get_frame(ch)
        self.trpl.show_point(_trpl_path_for(scan, st, fr), self._point_label(ch, fr))
        x, label, frames, marker = self._sweep_points(ch)
        self.trpl.set_points([(float(xv), self._point_label(ch, f), _trpl_path_for(scan, st, f))
                              for xv, f in zip(x, frames)], label, marker, self._sweep_context(ch, fr))

    # ── closing ───────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        # let running alignments / fits stop, so no QThread is destroyed while running
        if not (self._jobs or self.trpl.busy):
            event.accept()
            return
        if self._close_t0 is None:
            self._close_t0 = time.monotonic()
            for j in self._jobs:
                j.cancel()
            self.trpl.cancel()
        if time.monotonic() - self._close_t0 > 15.0:
            event.accept()
            return
        event.ignore()
        QtCore.QTimer.singleShot(200, self.close)


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="PL + TRPL power / gate viewer v2")
    ap.add_argument("--folder", default="", help="PL + TRPL data folder to open at startup")
    args = ap.parse_args()
    pg.setConfigOptions(background="w", foreground="k")
    app = QtWidgets.QApplication(sys.argv)
    win = PLTRPLViewerV2()
    win.show()
    if args.folder:
        QtCore.QTimer.singleShot(0, lambda: win.load_folder(args.folder))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
