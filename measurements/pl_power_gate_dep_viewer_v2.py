"""
pl_power_gate_dep_viewer_v2.py

PL power / gate dependence offline viewer  —  version 2.

Changes vs v1
─────────────
• Window opens immediately (no upfront folder dialog)
• Baseline correction: per-row subtract mean in a user-chosen λ range,
  applied independently to data and substrate before data − substrate
• "Substrate" replaces generic "Background"
• Polarization mode: separate data + substrate for LH and RH;
  switch display between LH / RH / LH−RH
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

def _load_frames(data_dir: Path) -> List[PLFrame]:
    frames: List[PLFrame] = []
    for path in sorted(data_dir.glob("*.asc")):
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

    def __init__(self, title: str = "", *, default_linecut_width: int = 1, parent=None):
        super().__init__(parent)
        self._img_data: Optional[np.ndarray] = None
        self._wl:       Optional[np.ndarray] = None
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

    def set_image(self, img: np.ndarray, wl: Optional[np.ndarray] = None) -> None:
        self._img_data = np.asarray(img, dtype=float)
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
        lc = _linecut_h(self._img_data, row_raw, self.linecut_width())
        if lc is None:
            self._lcCurve.setData([], [])
            return
        x = self._wl if (self._wl is not None and self._wl.size == lc.size) \
            else np.arange(lc.size, dtype=float)
        self._lcCurve.setData(x, lc)

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
        cl = QtWidgets.QHBoxLayout(corr)
        cl.setContentsMargins(8, 4, 8, 4)
        cl.setSpacing(8)

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
        self.subCheck.toggled.connect(lambda _: self._refresh_all())
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
        img, wl = self._get_corrected(ch, fr)
        h = img.shape[0]
        row_d   = self.liveView.linecut_row()
        row_d   = h // 2 if row_d is None else max(0, min(int(row_d), h - 1))
        row_raw = h - 1 - row_d
        width   = int(self.liveView.linecut_width())
        lc = _linecut_h(img, row_raw, width)
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

    def _get_corrected(self, ch: _Channel, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (corrected_image, wl) applying baseline and substrate corrections."""
        img, wl = self._get_cropped_raw(ch, fr)
        do_bl  = self.baselineCheck.isChecked()
        do_sub = self.subCheck.isChecked() and ch.has_substrate
        if not do_bl and not do_sub:
            return img, wl
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
        return corrected, wl

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

        img, wl = self._get_corrected(ch, fr)

        if self._is_diff_mode():
            ch_rh = self._ch["RH"]
            fr_rh = self._get_frame(ch_rh)
            if fr_rh is not None:
                img_rh, wl_rh = self._get_corrected(ch_rh, fr_rh)
                if img.shape == img_rh.shape:
                    img = img - img_rh

        self.liveView.set_image(img, wl)

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
            img, wl = self._get_corrected(ch, fr)
            h = img.shape[0]
            # linecut_row() is in display space (flipud); convert to raw row
            row_d   = int(row_disp_raw) if row_disp_raw is not None else h // 2
            row_d   = max(0, min(row_d, h - 1))
            row_raw = h - 1 - row_d
            lc = _linecut_h(img, row_raw, width)
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
        img, wl = self._get_corrected(ch, fr)
        h       = img.shape[0]
        row_d   = self.liveView.linecut_row()
        row_d   = h // 2 if row_d is None else max(0, min(int(row_d), h - 1))
        row_raw = h - 1 - row_d
        lc = _linecut_h(img, row_raw, int(self.liveView.linecut_width()))
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
                img, wl = self._get_corrected(ch, fr)
                h = img.shape[0]
                row_d   = h // 2 if row_disp is None else max(0, min(int(row_disp), h - 1))
                row_raw = h - 1 - row_d
                lc = _linecut_h(img, row_raw, width)
                if lc is None:
                    continue
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
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = PLViewerV2()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
