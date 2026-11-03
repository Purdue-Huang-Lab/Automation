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

# DPI fix — must run before Qt is imported so Windows reports DPR=1
import ctypes as _ctypes
try:
    _ctypes.windll.user32.SetProcessDpiAwarenessContext(_ctypes.c_ssize_t(-1))
except Exception:
    pass
import os as _os
_os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
_os.environ.setdefault("QT_SCALE_FACTOR", "1")

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")


def _fix_pg_axes(pw: "pg.PlotWidget") -> None:
    """Set explicit black pens on both axes so labels stay visible at any DPI."""
    for ax in ("bottom", "left"):
        pw.getAxis(ax).setPen(pg.mkPen("k"))
        pw.getAxis(ax).setTextPen(pg.mkPen("k"))


def _apply_top_axis(plot_item, x: Optional[np.ndarray], x2: Optional[np.ndarray], label: str = "") -> None:
    """Show/hide a top AxisItem on `plot_item` with ticks labelled from `x2`.

    `plot_item` may be a pg.PlotItem or pg.PlotWidget (both expose showAxis/getAxis).
    Ticks sit at up to 8 evenly-spaced positions along `x`; hides the axis if
    x2 is None or its size doesn't match x.
    """
    x_arr = None if x is None else np.asarray(x, dtype=float).ravel()
    x2_arr = None if x2 is None else np.asarray(x2, dtype=float).ravel()
    if x_arr is None or x2_arr is None or x2_arr.size != x_arr.size or x_arr.size == 0:
        plot_item.showAxis("top", False)
        return
    plot_item.showAxis("top", True)
    top_ax = plot_item.getAxis("top")
    top_ax.setPen(pg.mkPen("k"))
    top_ax.setTextPen(pg.mkPen("k"))
    top_ax.setLabel(label)
    nticks = min(8, x_arr.size)
    idx = np.linspace(0, x_arr.size - 1, nticks).astype(int)
    ticks = [(float(x_arr[i]),
              f"{float(x2_arr[i]):.3g}" if np.isfinite(x2_arr[i]) else "")
             for i in idx]
    top_ax.setTicks([ticks])


# Sequential ramps for all-fixed-axis overlays (light → dark = increasing value).
# Anchors chosen so the lightest shade is still readable on white.
_OVL_BLUE  = ("#9ec7e8", "#08306b")   # RP (peak 1)
_OVL_RED   = ("#f9a97f", "#7f0009")   # AP (peak 2)
_OVL_GREEN = ("#a3d9a5", "#00441b")   # total


def _shade_pens(anchors: tuple, n: int, width: float = 1.1) -> list:
    """Return n pens interpolating anchors[0] → anchors[1]."""
    lc = pg.mkColor(anchors[0])
    dc = pg.mkColor(anchors[1])
    lo = np.array([lc.red(), lc.green(), lc.blue()], dtype=float)
    hi = np.array([dc.red(), dc.green(), dc.blue()], dtype=float)
    pens = []
    for i in range(max(1, n)):
        t = i / (n - 1) if n > 1 else 1.0
        r, g, b = (lo + (hi - lo) * t).astype(int)
        pens.append(pg.mkPen(QtGui.QColor(int(r), int(g), int(b)), width=width))
    return pens


try:
    from scipy.signal import savgol_filter as _savgol
except ImportError:
    _savgol = None

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigCanvasAgg
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

DEFAULT_CROP = (50, 50, 400, 400)
EV_NM = 1239.841984
GRID_DECIMALS = 9
_DIP_COLORS = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

# ── Sample calibration (single spot) — gate/power → E_F, n_ex ──────────
# Viewer's front gate (Vf) is this calibration's Vg (back gate unconnected).
CAL_V0_V = 0.5                # gate threshold [V]
CAL_EF_MEV_PER_V = 2.06       # E_F slope [meV / V]
CAL_K_CM2_PER_W = 4.21e20     # n_ex = eta * K * A(Vg) * P   [cm^-2 per pulse]
CAL_EF_LABEL = "E_F (meV)"
CAL_NEX_LABEL = "n_ex/η (10¹² cm⁻²/pulse)"


def _cal_ef_mev(vg) -> np.ndarray:
    """E_F(Vg) in meV; 0 below the gate threshold."""
    return CAL_EF_MEV_PER_V * np.clip(np.asarray(vg, dtype=float) - CAL_V0_V, 0.0, None)


def _cal_A(vg) -> np.ndarray:
    vg = np.asarray(vg, dtype=float)
    return np.where(vg < 4.0, 5.71e-4 - 6.6e-5 * vg, 3.07e-4)


def _cal_nex_1e12(vg, p_w) -> np.ndarray:
    """n_ex/eta in units of 10^12 cm^-2 per pulse, at fixed Vg and power P [W]."""
    return CAL_K_CM2_PER_W * _cal_A(vg) * np.asarray(p_w, dtype=float) / 1e12


def _cal_nex_cm2(vg, p_w) -> np.ndarray:
    """n_ex/eta in raw cm^-2 per pulse (eta=1 assumed), at fixed Vg and power P [W]."""
    return CAL_K_CM2_PER_W * _cal_A(vg) * np.asarray(p_w, dtype=float)


def _cal_rb_nm(vg: float, k_uW: float) -> float:
    """Bimolecular recombination radius R_b (nm, eta=1 assumed) from a Hill-fit
    saturation power K [uW] at fixed gate Vg: n_sat = n_ex(Vg, K), R_b = 1/sqrt(pi*n_sat)."""
    if not np.isfinite(k_uW) or k_uW <= 0:
        return float("nan")
    n_sat_cm2 = float(_cal_nex_cm2(vg, k_uW * 1e-6))
    if not np.isfinite(n_sat_cm2) or n_sat_cm2 <= 0:
        return float("nan")
    return 1.0 / np.sqrt(np.pi * n_sat_cm2) * 1e7   # cm -> nm


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════

class PLFrame:
    __slots__ = ("path", "power_w", "v_front", "v_back",
                 "wavelength_nm", "image", "meta", "mtime", "angle_deg")

    def __init__(self, path, power_w, v_front, v_back, wavelength_nm, image, meta, mtime, angle_deg=None):
        self.path = path
        self.power_w = power_w
        self.v_front = v_front
        self.v_back = v_back
        self.wavelength_nm = wavelength_nm
        self.image = image
        self.meta = meta
        self.mtime = mtime
        self.angle_deg = angle_deg


class _Channel:
    """All state for one measurement channel (data frames + optional substrate)."""

    def __init__(self):
        self.data_dir: Optional[Path] = None
        self.frames:   List[PLFrame] = []
        self.grid:     Dict[Tuple[float, float], PLFrame] = {}
        self.powers:   np.ndarray = np.array([], dtype=float)
        self.gates:    np.ndarray = np.array([], dtype=float)
        self.backs_for_gate: np.ndarray = np.array([], dtype=float)
        self.default_power_key: float = 0.0
        self.default_gate_key:  float = 0.0
        self.can_power_mode: bool = False
        self.can_gate_mode:  bool = False
        self.blank_image: Optional[np.ndarray] = None
        self.blank_wl:    Optional[np.ndarray] = None
        self.sub_single: Optional[PLFrame] = None
        self.sub_grid:   Dict[Tuple[float, float], PLFrame] = {}
        self._img_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
        self.angle_mode: bool = False
        self._angle_calib_angles: Optional[np.ndarray] = None
        self._angle_calib_powers_w: Optional[np.ndarray] = None

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
    wl = None
    cols = str(meta.get("columns", "")).lower()
    if data.shape[1] > 1 and "wavelength" in cols:
        wl = data[:, 0]
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
_VF_RE = re.compile(
    r"(?:^|_)Vf(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")
_VB_RE = re.compile(
    r"(?:^|_)Vb(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")
_ANGLE_RE = re.compile(
    r"(?:^|_)A(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?)deg(?=_|\.|$)")


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
        val = _tok(pm.group("val"))
        scale = {"w": 1.0, "mw": 1e-3, "uw": 1e-6,
                 "nw": 1e-9}.get(pm.group("unit").lower())
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
    h, w = a.shape
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
    half = width // 2
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
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(data, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    n = x.size
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
    p = np.asarray(powers_w, dtype=float).ravel()
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
    ok = np.isfinite(ev)
    ev_ok = ev[ok]
    arr_ok = arr[ok, :]
    if ev_ok.size < 2:
        return ev_ok, arr_ok
    order = np.argsort(ev_ok)
    ev_s = ev_ok[order]
    arr_s = arr_ok[order, :]
    # Resample onto a uniform energy grid so ImageItem rect gives correct axis labels.
    # (eV = hc/λ is nonlinear in λ; without resampling, pixel positions would not
    #  match the linearly-scaled axis.)
    ev_u = np.linspace(ev_s[0], ev_s[-1], ev_s.size)
    n_x = arr_s.shape[1]
    arr_u = np.empty((ev_s.size, n_x), dtype=float)
    for j in range(n_x):
        col = arr_s[:, j]
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
    half = 0.5 * w
    valid = np.isfinite(c) & np.isfinite(w) & (
        w > 0) & (c > half) & (c + half > 0)
    if not np.any(valid):
        return out
    out[valid] = np.abs(EV_NM / (c[valid] - half[valid]) -
                        EV_NM / (c[valid] + half[valid])) * 1e3
    return out


# ══════════════════════════════════════════════════════════════════════
# Peak-fitting utilities  (ported from v1)
# ══════════════════════════════════════════════════════════════════════

def _lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        return np.zeros_like(x)
    g = float(gamma)
    dx = x - float(center)
    return float(amplitude) * g * g / (dx * dx + g * g)


def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(x)
    dx = x - float(center)
    return float(amplitude) * np.exp(-0.5 * dx * dx / (float(sigma) * float(sigma)))


_FWHM_GAUSS = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


def _peak_model(fit_model: str):
    """Return (func, fwhm_factor) for the selected fit model."""
    if fit_model == "gaussian":
        return _gaussian, _FWHM_GAUSS
    return _lorentzian, 2.0


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
    mad = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad / np.sqrt(2.0))
    std = np.nanstd(v)
    return 0.0 if not np.isfinite(std) or std <= 0 else float(std)


def _peak_guess(x, y, lo, hi, fallback):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        return 0.0, float(fallback), 3.0
    xs = x[mask]
    ys = y[mask]
    idx = int(np.nanargmax(ys))
    center = float(xs[idx])
    bl = float(np.nanpercentile(ys, 15.0))
    amp = float(max(0.0, ys[idx] - bl))
    wt = np.clip(ys - bl, 0.0, None)
    ws = float(np.nansum(wt))
    sigma = float(np.sqrt(np.nansum(wt * (xs - center) ** 2) / ws)
                  ) if ws > 0 else 3.0
    return amp, center, float(np.clip(0.5 * sigma, 0.5, 10.0))


def fit_lorentzian_peak_local(wl, intensity, center_guess, *, window_nm,
                              min_points=7, min_snr=2.0, min_gamma=0.08, max_gamma=10.0,
                              fit_model="lorentzian"):
    x = np.asarray(wl,        dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < min_points:
        return None
    mask = np.isfinite(x) & np.isfinite(y) & (
        np.abs(x - center_guess) <= window_nm)
    if mask.sum() < min_points:
        return None
    xs = x[mask]
    ys = y[mask]
    bl = float(np.nanpercentile(ys, 10.0))
    yb = ys - bl
    amp = float(np.nanmax(yb))
    if not np.isfinite(amp) or amp <= 0:
        return None
    if _noise_floor(ys) > 0 and amp < min_snr * _noise_floor(ys):
        return None
    idx = int(np.nanargmax(yb))
    c_fit = float(xs[idx])
    y_half = 0.5 * amp
    left_ = np.where(yb[:idx] <= y_half)[0]
    right_ = np.where(yb[idx + 1:] <= y_half)[0]
    if left_.size > 0 and right_.size > 0:
        fwhm_g = max(2.0 * min_gamma,
                     float(xs[idx + 1 + right_[0]]) - float(xs[left_[-1]]))
    else:
        fwhm_g = max(2.0 * min_gamma, 0.25 * window_nm)
    g_fit = float(np.clip(0.5 * fwhm_g, min_gamma, max_gamma))
    a_fit = amp
    mfunc, fwhm_fac = _peak_model(fit_model)
    if _curve_fit is not None:
        try:
            lo = np.array([0.0,  center_guess - window_nm, min_gamma])
            hi = np.array(
                [max(1.0, 4.0 * amp), center_guess + window_nm, max_gamma])
            p0 = np.clip([a_fit, c_fit, g_fit], lo + 1e-9, hi - 1e-9)
            popt, _ = _curve_fit(lambda xx, a, m, g: mfunc(xx, a, m, g),
                                 xs, yb, p0=p0, bounds=(lo, hi), maxfev=8000)
            a_fit, c_fit, g_fit = [float(v) for v in popt]
        except Exception:
            pass
    if not (min_gamma <= g_fit <= max_gamma):
        return None
    if abs(c_fit - center_guess) > window_nm * 1.2:
        return None
    model = bl + mfunc(xs, a_fit, c_fit, g_fit)
    sst = float(np.nansum((ys - np.nanmean(ys)) ** 2))
    r2 = 1.0 - float(np.nansum((ys - model) ** 2)) / sst if sst > 0 else 1.0
    if not np.isfinite(r2) or r2 < -0.3:
        return None
    return {"center_nm": c_fit, "gamma_nm": g_fit, "fwhm_nm": fwhm_fac * g_fit, "amplitude": max(0.0, a_fit)}


def fit_two_peak_linecut(wl, intensity, *, peak1_guess_nm, peak2_guess_nm,
                         peak1_lo=None, peak1_hi=None, peak2_lo=None, peak2_hi=None,
                         peak1_window_nm=10.0, peak2_window_nm=5.0,
                         peak1_gamma_lo=None, peak1_gamma_hi=None,
                         peak2_gamma_lo=None, peak2_gamma_hi=None,
                         fit_model="lorentzian") -> dict:
    x_all = np.asarray(wl,        dtype=float).ravel()
    y_all = np.asarray(intensity, dtype=float).ravel()
    empty = {"peak_1": None, "peak_2": None, "offset": 0.0}
    if x_all.size != y_all.size or x_all.size < 8:
        return empty
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    if valid.sum() < 8:
        return empty
    x_all = x_all[valid]
    y_all = y_all[valid]
    order = np.argsort(x_all)
    x_all = x_all[order]
    y_all = y_all[order]

    bl = _baseline_from_tail(x_all, y_all)
    y_sub = y_all - bl

    # Resolve peak ranges — explicit range takes priority over guess±window
    if peak1_lo is not None and peak1_hi is not None:
        p1_lo, p1_hi = float(peak1_lo), float(peak1_hi)
        if not (p1_lo <= float(peak1_guess_nm) <= p1_hi):
            peak1_guess_nm = 0.5 * (p1_lo + p1_hi)
    else:
        w1 = max(0.2, float(peak1_window_nm))
        p1_lo = float(peak1_guess_nm) - w1
        p1_hi = float(peak1_guess_nm) + w1

    if peak2_lo is not None and peak2_hi is not None:
        p2_lo, p2_hi = float(peak2_lo), float(peak2_hi)
        if not (p2_lo <= float(peak2_guess_nm) <= p2_hi):
            peak2_guess_nm = 0.5 * (p2_lo + p2_hi)
    else:
        w2 = max(0.2, float(peak2_window_nm))
        p2_lo = float(peak2_guess_nm) - w2
        p2_hi = float(peak2_guess_nm) + w2

    fit_lo = min(p2_lo, p1_lo) - 6.0
    fit_hi = max(p2_hi, p1_hi) + 30.0

    mask = (x_all >= fit_lo) & (x_all <= fit_hi)
    x = x_all[mask] if mask.sum() >= 20 else x_all
    y = y_sub[mask] if mask.sum() >= 20 else y_sub
    if x.size < 8:
        return {**empty, "offset": float(bl)}

    mfunc, fwhm_fac = _peak_model(fit_model)

    # Width bounds — three-layer priority (tightest wins):
    #   1. Sequential propagation bounds passed by the caller (peak*_gamma_lo/hi)
    #   2. Range-derived natural cap: FWHM ≤ range width → gamma ≤ range_width / fwhm_fac
    #   3. Hard global floor/ceiling (0.4 / 15 nm)
    _range1_w = max(1.0, p1_hi - p1_lo)
    _range2_w = max(1.0, p2_hi - p2_lo)
    _cap1 = min(15.0, _range1_w / fwhm_fac)   # FWHM ≤ range width
    _cap2 = min(12.0, _range2_w / fwhm_fac)
    _glo1 = max(0.4, float(peak1_gamma_lo)
                ) if peak1_gamma_lo is not None else 0.4
    _ghi1 = min(_cap1, float(peak1_gamma_hi)
                ) if peak1_gamma_hi is not None else _cap1
    _glo2 = max(0.4, float(peak2_gamma_lo)
                ) if peak2_gamma_lo is not None else 0.4
    _ghi2 = min(_cap2, float(peak2_gamma_hi)
                ) if peak2_gamma_hi is not None else _cap2
    _ghi1 = max(_ghi1, _glo1 + 0.05)   # ensure lo < hi after tightening
    _ghi2 = max(_ghi2, _glo2 + 0.05)

    y_span = float(max(1.0, np.nanmax(y) - np.nanmin(y)))
    a1g, m1g, g1g = _peak_guess(x, y, p1_lo, p1_hi, peak1_guess_nm)
    a2g, m2g, g2g = _peak_guess(x, y, p2_lo, p2_hi, peak2_guess_nm)
    a1g = max(0.08 * y_span, a1g)
    a2g = max(0.02 * y_span, a2g)
    g1g = float(np.clip(g1g, _glo1, _ghi1))
    g2g = float(np.clip(g2g, _glo2, _ghi2))

    popt2 = popt1 = None
    if _curve_fit is not None:
        lo2 = [0.0, p1_lo, _glo1, 0.0, p2_lo, _glo2]
        hi2 = [4*y_span, p1_hi, _ghi1, 4*y_span, p2_hi, _ghi2]
        lo1 = [0.0, p1_lo, _glo1]
        hi1 = [4*y_span, p1_hi, _ghi1]
        try:
            p0 = np.clip([a1g, m1g, g1g, a2g, m2g, g2g],
                         np.array(lo2)+1e-6, np.array(hi2)-1e-6)
            popt2, _ = _curve_fit(
                lambda xx, a1, m1, g1, a2, m2, g2: mfunc(
                    xx, a1, m1, g1)+mfunc(xx, a2, m2, g2),
                x, y, p0=p0, bounds=(lo2, hi2), maxfev=20000)
        except Exception:
            popt2 = None
        try:
            p0 = np.clip([a1g, m1g, g1g], np.array(
                lo1)+1e-6, np.array(hi1)-1e-6)
            popt1, _ = _curve_fit(lambda xx, a, m, g: mfunc(xx, a, m, g),
                                  x, y, p0=p0, bounds=(lo1, hi1), maxfev=12000)
        except Exception:
            popt1 = None

    if popt2 is None and popt1 is None:
        fb_win1 = max(0.5*(p1_hi - p1_lo), 2.0)
        fb_win2 = max(0.5*(p2_hi - p2_lo), 2.0)
        p1 = fit_lorentzian_peak_local(x_all, y_sub, float(peak1_guess_nm),
                                       window_nm=fb_win1, min_snr=1.2, fit_model=fit_model,
                                       min_gamma=_glo1, max_gamma=_ghi1)
        resid = np.array(y_sub, dtype=float)
        if p1 is not None:
            resid -= mfunc(x_all, p1["amplitude"],
                           p1["center_nm"], p1["gamma_nm"])
        p2 = fit_lorentzian_peak_local(x_all, resid, float(peak2_guess_nm),
                                       window_nm=fb_win2, min_snr=1.5, fit_model=fit_model,
                                       min_gamma=_glo2, max_gamma=_ghi2)
        if p1 is not None and not (p1_lo <= p1["center_nm"] <= p1_hi):
            p1 = None
        if p2 is not None and not (p2_lo <= p2["center_nm"] <= p2_hi):
            p2 = None
        return {"peak_1": p1, "peak_2": p2, "offset": float(bl)}

    # choose between 1-peak and 2-peak solutions
    use2 = False
    if popt2 is not None:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]
        y2 = mfunc(x, a1, m1, g1)+mfunc(x, a2, m2, g2)
        y1 = mfunc(x, *(float(v) for v in popt1)
                   ) if popt1 is not None else mfunc(x, a1, m1, g1)
        noise = _noise_floor(y - y2)
        rss1 = float(np.nansum((y - y1)**2))
        rss2 = float(np.nansum((y - y2)**2))
        imp = (rss1 - rss2) / rss1 if rss1 > 0 else 0.0
        min_a2 = max(2.0*noise, 0.04*y_span, 0.06*max(a1, 1.0))
        use2 = (a2 >= min_a2 and p2_lo <= m2 <= p2_hi
                and _glo2 <= g2 <= _ghi2 and imp > 0.03)

    if use2:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]
    elif popt1 is not None:
        a1, m1, g1 = [float(v) for v in popt1]
        a2, m2, g2 = 0.0, float(peak2_guess_nm), 3.0
    else:
        a1, m1, g1, a2, m2, g2 = [float(v) for v in popt2]
        a2 = 0.0

    p1 = ({"center_nm": m1, "gamma_nm": g1, "fwhm_nm": fwhm_fac*g1, "amplitude": a1}
          if a1 > 0 and _glo1 <= g1 <= _ghi1 and p1_lo <= m1 <= p1_hi else None)
    p2 = ({"center_nm": m2, "gamma_nm": g2, "fwhm_nm": fwhm_fac*g2, "amplitude": a2}
          if use2 and a2 > 0 and _glo2 <= g2 <= _ghi2 and p2_lo <= m2 <= p2_hi else None)
    return {"peak_1": p1, "peak_2": p2, "offset": float(bl)}


def fit_two_peak_map_sequential(wl_nm, map_data, *, ref_idx, peak1_guess_nm, peak2_guess_nm,
                                max_shift_nm=2.0, max_width_ratio=2.0,
                                peak1_lo=None, peak1_hi=None, peak2_lo=None, peak2_hi=None,
                                fit_model="lorentzian") -> dict:
    wl = np.asarray(wl_nm,   dtype=float).ravel()
    data = np.asarray(map_data, dtype=float)
    n_x = data.shape[1] if data.ndim == 2 else 0
    c1 = np.full(n_x, np.nan)
    fw1 = np.full(n_x, np.nan)
    gm1 = np.full(n_x, np.nan)
    a1 = np.full(n_x, np.nan)
    c2 = np.full(n_x, np.nan)
    fw2 = np.full(n_x, np.nan)
    gm2 = np.full(n_x, np.nan)
    a2 = np.full(n_x, np.nan)
    off = np.full(n_x, np.nan)

    def _result():
        return {"center_1_nm": c1, "fwhm_1_nm": fw1, "gamma_1_nm": gm1, "amp_1": a1,
                "center_2_nm": c2, "fwhm_2_nm": fw2, "gamma_2_nm": gm2, "amp_2": a2, "offset": off}

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0:
        return _result()

    ref = max(0, min(int(ref_idx), n_x - 1))
    step = max(0.2, float(max_shift_nm))

    def _store(j, fit):
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        off[j] = float(fit.get("offset", np.nan))
        if p1:
            c1[j] = p1.get("center_nm", np.nan)
            fw1[j] = p1.get("fwhm_nm", np.nan)
            gm1[j] = p1.get("gamma_nm", np.nan)
            a1[j] = p1.get("amplitude", np.nan)
        if p2:
            c2[j] = p2.get("center_nm", np.nan)
            fw2[j] = p2.get("fwhm_nm", np.nan)
            gm2[j] = p2.get("gamma_nm", np.nan)
            a2[j] = p2.get("amplitude", np.nan)

    _kw = dict(peak1_lo=peak1_lo, peak1_hi=peak1_hi,
               peak2_lo=peak2_lo, peak2_hi=peak2_hi,
               fit_model=fit_model)

    fit_ref = fit_two_peak_linecut(wl, data[:, ref],
                                   peak1_guess_nm=peak1_guess_nm, peak2_guess_nm=peak2_guess_nm,
                                   peak1_window_nm=10.0, peak2_window_nm=5.0, **_kw)
    _store(ref, fit_ref)
    g1r = c1[ref] if np.isfinite(c1[ref]) else float(peak1_guess_nm)
    g2r = c2[ref] if np.isfinite(c2[ref]) else float(peak2_guess_nm)
    gw1r = gm1[ref] if np.isfinite(gm1[ref]) and gm1[ref] > 0 else np.nan
    gw2r = gm2[ref] if np.isfinite(gm2[ref]) and gm2[ref] > 0 else np.nan

    _use_width_ratio = max_width_ratio > 1.0

    # SNR gates for width continuity: a weak peak carries almost no width
    # information, so its fitted gamma must not steer the propagated bounds.
    _SNR_HOLD   = 4.0   # signal below this ×noise → freeze width bounds (ratio 1.2)
    _SNR_UPDATE = 3.0   # fitted amp below this ×noise → don't propagate its width
    _WEAK_RATIO = 1.2

    def _walk(start, stop, inc, g1i, g2i, gw1i, gw2i):
        g1 = float(g1i)
        g2 = float(g2i)
        gw1_cur = float(gw1i) if np.isfinite(gw1i) and gw1i > 0 else np.nan
        gw2_cur = float(gw2i) if np.isfinite(gw2i) and gw2i > 0 else np.nan
        for j in range(start, stop, inc):
            spec    = data[:, j]
            noise_j = _noise_floor(spec)

            def _sig_est(center):
                """Peak height above baseline near `center` (pre-fit, raw data)."""
                m = np.isfinite(wl) & np.isfinite(spec) & \
                    (np.abs(wl - center) <= max(2.0 * step, 3.0))
                if m.sum() < 3:
                    return np.inf          # can't tell — don't tighten
                fin = spec[np.isfinite(spec)]
                base = float(np.nanpercentile(fin, 10.0)) if fin.size else 0.0
                return float(np.nanmax(spec[m]) - base)

            # Weak-signal frames: hold the propagated width nearly fixed instead
            # of the loose max_width_ratio, so noise can't collapse the width.
            wr1 = max_width_ratio if (noise_j <= 0 or _sig_est(g1) >= _SNR_HOLD * noise_j) \
                else _WEAK_RATIO
            wr2 = max_width_ratio if (noise_j <= 0 or _sig_est(g2) >= _SNR_HOLD * noise_j) \
                else _WEAK_RATIO

            p1glo = p1ghi = p2glo = p2ghi = None
            if _use_width_ratio and np.isfinite(gw1_cur):
                p1glo = gw1_cur / wr1
                p1ghi = gw1_cur * wr1
            if _use_width_ratio and np.isfinite(gw2_cur):
                p2glo = gw2_cur / wr2
                p2ghi = gw2_cur * wr2
            fj = fit_two_peak_linecut(wl, data[:, j],
                                      peak1_guess_nm=g1, peak2_guess_nm=g2,
                                      peak1_window_nm=step, peak2_window_nm=step,
                                      peak1_gamma_lo=p1glo, peak1_gamma_hi=p1ghi,
                                      peak2_gamma_lo=p2glo, peak2_gamma_hi=p2ghi,
                                      **_kw)
            _store(j, fj)
            if np.isfinite(c1[j]):
                g1 = float(c1[j])
            if np.isfinite(c2[j]):
                g2 = float(c2[j])
            # Confidence-gated width propagation: only widths fitted from peaks
            # well above the noise floor update the trusted width — prevents
            # weak/noisy frames from ratcheting the bounds toward collapse.
            if (np.isfinite(gm1[j]) and gm1[j] > 0
                    and (noise_j <= 0 or a1[j] >= _SNR_UPDATE * noise_j)):
                gw1_cur = float(gm1[j])
            if (np.isfinite(gm2[j]) and gm2[j] > 0
                    and (noise_j <= 0 or a2[j] >= _SNR_UPDATE * noise_j)):
                gw2_cur = float(gm2[j])

    _walk(ref + 1, n_x, +1, g1r, g2r, gw1r, gw2r)
    _walk(ref - 1, -1,  -1, g1r, g2r, gw1r, gw2r)
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
    x = np.asarray(wl,     dtype=float).ravel()
    y = np.asarray(signal, dtype=float).ravel()
    guesses = np.asarray(dip_guesses_nm, dtype=float).ravel()
    n_dips = guesses.size
    empty = {"dips": [None] * n_dips, "x_smooth": x, "y_smooth": y}

    if x.size != y.size or x.size < 5 or n_dips == 0:
        return empty
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return empty
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    sigma_px = max(0.5, smooth_sigma_nm / max(dx, 1e-12))
    try:
        from scipy.ndimage import gaussian_filter1d
        y_sm = gaussian_filter1d(y, sigma=sigma_px)
    except Exception:
        y_sm = y.copy()

    results = []
    for g in guesses:
        g = float(g)
        lo = g - float(max_shift_nm)
        hi = g + float(max_shift_nm)
        mask = (x >= lo) & (x <= hi)
        n_in = int(mask.sum())
        if n_in < 3:
            results.append(None)
            continue
        xs = x[mask]
        ys = y_sm[mask]
        yo = y[mask]
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
    wl = np.asarray(wl_nm,          dtype=float).ravel()
    data = np.asarray(map_data,        dtype=float)
    guesses = np.asarray(dip_guesses_nm,  dtype=float).ravel()
    n_dips = guesses.size
    n_x = data.shape[1] if data.ndim == 2 else 0

    centers = np.full((n_dips, n_x), np.nan)
    amps = np.full((n_dips, n_x), np.nan)

    def _result():
        return {"centers_nm": centers, "amps": amps}

    if data.ndim != 2 or wl.size != data.shape[0] or n_x == 0 or n_dips == 0:
        return _result()

    ref = max(0, min(int(ref_idx), n_x - 1))
    step = max(0.2, float(max_shift_nm))

    def _store(j, fit):
        for i, r in enumerate(fit["dips"]):
            if r is not None:
                centers[i, j] = r["center_nm"]
                amps[i, j] = r["value_at_min"]

    cur_g = guesses.copy()
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
    wl = np.asarray(wl,  dtype=float).ravel()
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
    win_wl += (win_wl % 2 == 0)

    def _rows(arr, win, order):
        if win < 3 or order < 1 or order >= win or arr.shape[1] < win:
            return arr
        res = arr.copy()
        for i in range(arr.shape[0]):
            row = arr[i, :]
            mask = np.isfinite(row)
            if mask.sum() < win:
                continue
            if mask.all():
                res[i, :] = _savgol(row, win, order, mode="mirror")
            else:
                xi = np.arange(row.size)
                filled = np.interp(xi, xi[mask], row[mask])
                res[i, :] = _savgol(filled, win, order, mode="mirror")
                res[i, ~mask] = np.nan
        return res

    def _cols(arr, win, order):
        if win < 3 or order < 1 or order >= win or arr.shape[0] < win:
            return arr
        res = arr.copy()
        for j in range(arr.shape[1]):
            col = arr[:, j]
            mask = np.isfinite(col)
            if mask.sum() < win:
                continue
            if mask.all():
                res[:, j] = _savgol(col, win, order, mode="mirror")
            else:
                xi = np.arange(col.size)
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
        p_m = _parse_float(meta, "power_w")
        vf_m = _parse_float(meta, "front_v_set")
        vb_m = _parse_float(meta, "back_v_set")
        p_n, vf_n, vb_n = _parse_from_name(path.name)
        angle_n = _parse_angle_from_name(path.name)
        true_power = p_n if p_n is not None else p_m
        frames.append(PLFrame(
            path=path,
            power_w=true_power if true_power is not None else angle_n,
            v_front=vf_n if vf_n is not None else vf_m,
            v_back=vb_n if vb_n is not None else vb_m,
            wavelength_nm=wl,
            image=image,
            meta=meta,
            mtime=float(path.stat().st_mtime),
            angle_deg=angle_n,
        ))
    return frames


def _build_channel(ch: _Channel, frames: List[PLFrame]) -> None:
    """Populate grid / powers / gates from frames list."""
    ch.frames = frames
    power_keys: set = set()
    gate_keys:  set = set()
    back_by_gate: Dict = {}

    for fr in frames:
        p = _norm_key(fr.power_w)
        vf = _norm_key(fr.v_front)
        vb = _norm_key(fr.v_back)
        if p is not None:
            power_keys.add(p)
        if vf is not None:
            gate_keys.add(vf)
            if vb is not None:
                back_by_gate.setdefault(vf, []).append(float(vb))

    if not power_keys:
        power_keys = {0.0}
    if not gate_keys:
        gate_keys = {0.0}

    ch.powers = np.array(sorted(power_keys), dtype=float)
    ch.gates = np.array(sorted(gate_keys),  dtype=float)
    ch.default_power_key = float(_norm_key(ch.powers[0]) or 0.0)
    ch.default_gate_key = float(_norm_key(ch.gates[0]) or 0.0)

    ch.grid = {}
    for fr in frames:
        p = _ch_norm_power(ch, fr.power_w)
        vf = _ch_norm_gate(ch, fr.v_front)
        key = (p, vf)
        old = ch.grid.get(key)
        if old is None or fr.mtime > old.mtime:
            ch.grid[key] = fr

    ch.can_power_mode = ch.powers.size > 1
    ch.can_gate_mode = ch.gates.size > 1
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
        ch.blank_wl = np.arange(10, dtype=float)


def _ch_norm_power(ch: _Channel, p: Optional[float]) -> float:
    k = _norm_key(p)
    return ch.default_power_key if k is None else float(k)


def _ch_norm_gate(ch: _Channel, vf: Optional[float]) -> float:
    k = _norm_key(vf)
    return ch.default_gate_key if k is None else float(k)


def _ch_substrate_for(ch: _Channel, fr: PLFrame) -> Optional[PLFrame]:
    if ch.sub_single is not None:
        return ch.sub_single
    if not ch.sub_grid:
        return None
    p = float(_norm_key(fr.power_w) or ch.default_power_key)
    vf = float(_norm_key(fr.v_front) or ch.default_gate_key)
    key = (p, vf)
    if key in ch.sub_grid:
        return ch.sub_grid[key]
    same_gate = [(k, v) for k, v in ch.sub_grid.items() if k[1] == vf]
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
    ch.sub_grid = {}
    if not frames:
        return
    default_p = _norm_key(frames[0].power_w) or 0.0
    default_vf = _norm_key(frames[0].v_front) or 0.0
    for fr in frames:
        p = float(_norm_key(fr.power_w) or default_p)
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
        self._lcCurve = self._lcPlot.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), name="Linecut")
        self._fitTotal = self._lcPlot.plot(
            pen=pg.mkPen("#ffb300", width=1.2, style=QtCore.Qt.DashLine), name="Fit total")
        self._fitPeak1 = self._lcPlot.plot(
            pen=pg.mkPen("#1f77b4", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 1")
        self._fitPeak2 = self._lcPlot.plot(
            pen=pg.mkPen("#d62728", width=1.0, style=QtCore.Qt.DotLine), name="Fit peak 2")
        # SG-smooth / dip-smooth overlay in linecut
        self._smoothCurve = self._lcPlot.plot(
            pen=pg.mkPen("#aaaaaa", width=1.0, style=QtCore.Qt.DashLine),
            name="Smoothed")
        self._smoothCurve.setVisible(False)
        self._dipVLines: List[pg.InfiniteLine] = []

        # Peak range shading (semi-transparent, non-movable)
        self._rangeRegion1 = pg.LinearRegionItem(
            values=(720, 735), movable=False,
            brush=pg.mkBrush(30, 120, 255, 35), pen=pg.mkPen(None))
        self._rangeRegion1.setVisible(False)
        self._lcPlot.addItem(self._rangeRegion1)
        self._rangeRegion2 = pg.LinearRegionItem(
            values=(735, 745), movable=False,
            brush=pg.mkBrush(255, 80, 50, 35), pen=pg.mkPen(None))
        self._rangeRegion2.setVisible(False)
        self._lcPlot.addItem(self._rangeRegion2)

        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 2)
        self._glw.ci.layout.setColumnMaximumWidth(1, 65)   # narrow histogram
        self._glw.ci.layout.setColumnStretchFactor(0, 4)   # image (wide)
        self._glw.ci.layout.setColumnStretchFactor(2, 2)   # vertical linecut

        self.linecutWidthSpin.valueChanged.connect(
            lambda _: self._refresh_linecut())

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
        self._fitTotal.setData(
            x, yt) if yt is not None else self._fitTotal.setData([], [])
        self._fitPeak1.setData(
            x, y1v) if y1v is not None else self._fitPeak1.setData([], [])
        self._fitPeak2.setData(
            x, y2v) if y2v is not None else self._fitPeak2.setData([], [])

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
        row_d = int(np.clip(round(self._hline.value()), 0, self._h - 1))
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
        rows = np.arange(self._h, dtype=float)
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
            i = len(self._dipVLines)
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

    def set_smooth_linecut(self, x, y) -> None:
        xs = np.asarray(x, dtype=float).ravel()
        ys = np.asarray(y, dtype=float).ravel()
        if xs.size == ys.size and xs.size > 0 and np.isfinite(ys).any():
            self._smoothCurve.setData(xs, ys)
            self._smoothCurve.setVisible(True)
        else:
            self._smoothCurve.setVisible(False)

    def clear_smooth_linecut(self) -> None:
        self._smoothCurve.setVisible(False)

    def set_peak_ranges(self, lo1: float, hi1: float, lo2: float, hi2: float) -> None:
        self._rangeRegion1.setRegion((lo1, hi1))
        self._rangeRegion1.setVisible(True)
        self._rangeRegion2.setRegion((lo2, hi2))
        self._rangeRegion2.setVisible(True)

    def clear_peak_ranges(self) -> None:
        self._rangeRegion1.setVisible(False)
        self._rangeRegion2.setVisible(False)


# ══════════════════════════════════════════════════════════════════════
# MapPlotWidget  (pyqtgraph 2-D image map + row linecut)
# ══════════════════════════════════════════════════════════════════════

class MapPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, *, cmap_name: str, line_label: str, parent=None):
        super().__init__(title, parent)
        self._line_label = line_label
        self._cmap_name = cmap_name
        self._manual_sym_max = None
        self._x = self._y = self._z = None
        self._selected_y_idx = None
        self._top_x = None
        self._top_label = ""
        self._last_vline_x = None
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
            pen=pg.mkPen("k", width=0.8),
            hoverPen=pg.mkPen("k", width=1.5),
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
        self._manual_sym_max = None if (
            value is None or value <= 0) else float(value)

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(
            x2, dtype=float).ravel()
        self._top_label = str(label or "")

    def set_map(self, x, y, z, *, x_label: str, y_label: str, symmetric: bool = False,
                n_contours: int = 0, contour_lo_frac: float = 0.01) -> None:
        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        self._z = np.asarray(z, dtype=float)

        if not (self._x.size and self._y.size and self._z.size):
            return

        n_x, n_y = self._x.size, self._y.size

        # ImageItem expects shape (n_x, n_y); z has shape (n_y, n_x)
        img_data = self._z.T.copy()

        # Contour fill:
        #  1. Normalise the map to [0, 1]  (min→0, 99th-pct→1)
        #  2. Lo% threshold (contour_lo_frac) is in that normalised space:
        #     everything below it gets the minimum colour (value 0)
        #  3. [lo_frac, 1] is divided into exactly n_contours equal bands that
        #     span the full colormap, so setLevels is always (0, 1)
        _contour_range: Optional[Tuple[float, float]] = None
        if n_contours >= 2:
            finite_c = img_data[np.isfinite(img_data)]
            if finite_c.size > 0:
                if symmetric:
                    vm = float(self._manual_sym_max or np.nanpercentile(
                        np.abs(finite_c), 98))
                    if not np.isfinite(vm) or vm <= 0:
                        vm = max(float(np.nanmax(np.abs(finite_c))), 1.0)
                    z_min_n, z_max_n = -vm, vm
                else:
                    z_min_n = float(np.nanmin(finite_c))
                    z_max_n = float(np.nanpercentile(finite_c, 99))
                    if not np.isfinite(z_max_n) or z_max_n <= z_min_n:
                        z_max_n = float(np.nanmax(finite_c))
                if np.isfinite(z_min_n) and np.isfinite(z_max_n) and z_max_n > z_min_n:
                    # Pre-smooth for clean band boundaries (before normalising)
                    try:
                        from scipy.ndimage import gaussian_filter
                        _nm_pre = ~np.isfinite(img_data)
                        _fill = img_data.copy()
                        if _nm_pre.any():
                            _fill[_nm_pre] = float(np.nanmean(img_data))
                        img_sm = gaussian_filter(_fill, sigma=1.0)
                        img_sm[_nm_pre] = np.nan
                    except Exception:
                        img_sm = img_data
                    # Normalise to [0, 1]
                    img_norm = (img_sm - z_min_n) / (z_max_n - z_min_n)
                    vlo = float(np.clip(contour_lo_frac, 0.0, 0.99))
                    edges = np.linspace(vlo, 1.0, n_contours + 1)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    nan_mask = ~np.isfinite(img_data)
                    below_mask = img_norm < vlo
                    band_idx = np.clip(np.digitize(
                        img_norm, edges[1:-1]), 0, n_contours - 1)
                    img_data = centers[band_idx].astype(float)
                    # below threshold → min colour
                    img_data[below_mask & ~nan_mask] = 0.0
                    img_data[nan_mask] = np.nan
                    _contour_range = (0.0, 1.0)

        x0 = float(self._x[0])
        x_span = float(self._x[-1] - self._x[0]) if n_x > 1 else 1.0
        y0 = float(self._y[0])
        y_span = float(self._y[-1] - self._y[0]) if n_y > 1 else 1.0

        locked = getattr(self, "_axes_locked", False)
        img_safe = img_data if np.isfinite(
            img_data).any() else np.zeros_like(img_data)
        self._imgItem.setImage(img_safe, autoLevels=not locked)
        self._imgItem.setRect(QtCore.QRectF(x0, y0, x_span, y_span))

        if not locked:
            if _contour_range is not None:
                # Contour mode: align levels to the quantised band range so the
                # full colormap is used (dark = vlo, bright = vhi).
                self._imgItem.setLevels(_contour_range)
            else:
                finite = self._z[np.isfinite(self._z)]
                if finite.size:
                    if symmetric:
                        vmax = float(self._manual_sym_max or np.nanpercentile(
                            np.abs(finite), 98))
                        if not np.isfinite(vmax) or vmax <= 0:
                            vmax = max(float(np.nanmax(np.abs(finite))), 1.0)
                        self._imgItem.setLevels((-vmax, vmax))
                    else:
                        vmin = float(np.nanpercentile(finite, 1))
                        vmax = float(np.nanpercentile(finite, 99))
                        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
                            vmin, vmax = float(np.nanmin(finite)), float(
                                np.nanmax(finite))
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
        _apply_top_axis(self._mapPlot, self._x, self._top_x, self._top_label)

    def _update_linecut(self) -> None:
        if self._x is None or self._y is None or self._z is None:
            return
        yv = float(self._hline.value())
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
        xv = float(self._vline.value())
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
            QtWidgets.QMessageBox.warning(
                self, "Export CSV", "Shape mismatch.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        out = np.empty((y.size + 1, x.size + 1), dtype=float)
        out.fill(np.nan)
        out[0, 0] = 0.0
        out[0, 1:] = x
        out[1:, 0] = y
        out[1:, 1:] = z
        np.savetxt(path, out, delimiter=",", fmt="%.10g")


# ══════════════════════════════════════════════════════════════════════
# FitTrendPlotWidget  (pyqtgraph peak fit trend over sweep axis)
# ══════════════════════════════════════════════════════════════════════

class FitTrendPlotWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str, y_label: str, parent=None,
                 *, show_diff: bool = False, diff_label: str = "RP − AP"):
        super().__init__(title, parent)
        self._y_label = y_label
        self._show_diff = bool(show_diff)
        self._diff_label = diff_label
        self._x_last = self._y1_last = self._y2_last = None
        self._xl_last = "X"
        self._top_x = None
        self._top_label = ""
        self._build_ui()

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.setFixedWidth(110)
        self.showAllChk = QtWidgets.QCheckBox("Show all fixed values")
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.showAllChk)
        top_row.addStretch()
        top_row.addWidget(self.exportBtn)
        layout.addLayout(top_row)

        self._plot = pg.PlotWidget()
        _fix_pg_axes(self._plot)
        self._plot.setLabel("left",   self._y_label)
        self._plot.setLabel("bottom", "X")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(-10, 10))
        self._curve1 = self._plot.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#1f77b4", symbolPen=None, name="RP")
        self._curve2 = self._plot.plot(
            pen=pg.mkPen("#d62728", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#d62728", symbolPen=None, name="AP")
        self._ovl_items: list = []
        layout.addWidget(self._plot)

        # Optional second pane: RP − AP difference vs main axis
        self._plot_diff = None
        self._curve_diff = None
        if self._show_diff:
            self._plot_diff = pg.PlotWidget()
            _fix_pg_axes(self._plot_diff)
            self._plot_diff.setLabel("left",   self._diff_label)
            self._plot_diff.setLabel("bottom", "X")
            self._plot_diff.showGrid(x=True, y=True, alpha=0.25)
            self._plot_diff.setXLink(self._plot)
            self._curve_diff = self._plot_diff.plot(
                pen=pg.mkPen("#6a3d9a", width=1.2), symbol="o",
                symbolSize=4, symbolBrush="#6a3d9a", symbolPen=None)
            layout.addWidget(self._plot_diff)

        self.exportBtn.clicked.connect(self._on_export)

    def set_overlays(self, x, series) -> None:
        """Overlay (y1, y2) curves for every fixed-axis value.

        series: list of (y1, y2) arrays, light→dark = increasing fixed value.
        RP curves in blues, AP curves in reds.
        """
        self.clear_overlays()
        x = np.asarray(x, dtype=float).ravel()
        n = len(series)
        pens1 = _shade_pens(_OVL_BLUE, n)
        pens2 = _shade_pens(_OVL_RED,  n)
        for (y1, y2), p1, p2 in zip(series, pens1, pens2):
            for y, pen in ((y1, p1), (y2, p2)):
                y = np.asarray(y, dtype=float).ravel()
                nn = min(x.size, y.size)
                m = np.isfinite(x[:nn]) & np.isfinite(y[:nn])
                c = self._plot.plot(x[:nn][m], y[:nn][m], pen=pen)
                c.setZValue(-5)   # behind the main (current fixed) curves
                self._ovl_items.append(c)

    def clear_overlays(self) -> None:
        for c in self._ovl_items:
            self._plot.removeItem(c)
        self._ovl_items = []

    def set_data(self, x, y1, y2, *, x_label: str,
                 label_1: str = "RP", label_2: str = "AP") -> None:
        x = np.asarray(x,  dtype=float).ravel()
        y1 = np.asarray(y1, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel()
        n = min(x.size, y1.size, y2.size)
        self._x_last = x[:n].copy()
        self._y1_last = y1[:n].copy()
        self._y2_last = y2[:n].copy()
        self._xl_last = str(x_label)
        m1 = np.isfinite(x[:n]) & np.isfinite(y1[:n])
        m2 = np.isfinite(x[:n]) & np.isfinite(y2[:n])
        self._curve1.setData(x[:n][m1], y1[:n][m1])
        self._curve2.setData(x[:n][m2], y2[:n][m2])
        self._plot.setLabel("bottom", x_label)
        _apply_top_axis(self._plot.plotItem, self._x_last, self._top_x, self._top_label)
        if self._curve_diff is not None:
            diff = y1[:n] - y2[:n]
            md = np.isfinite(x[:n]) & np.isfinite(diff)
            self._curve_diff.setData(x[:n][md], diff[md])
            self._plot_diff.setLabel("bottom", x_label)

    def clear_data(self, message: str, *, x_label: str = "X") -> None:
        self._x_last = self._y1_last = self._y2_last = None
        self._xl_last = str(x_label)
        self._curve1.setData([], [])
        self._curve2.setData([], [])
        self._plot.setLabel("bottom", x_label)
        self._plot.setTitle(message)
        _apply_top_axis(self._plot.plotItem, None, None, "")
        if self._curve_diff is not None:
            self._curve_diff.setData([], [])
            self._plot_diff.setLabel("bottom", x_label)

    def _on_export(self) -> None:
        if self._x_last is None:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data.")
            return
        x, y1, y2 = self._x_last, self._y1_last, self._y2_last
        n = min(x.size, y1.size, y2.size)
        if n == 0:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        if self._show_diff:
            hdr = f"{self._xl_last},RP,AP,RP-AP"
            cols = [x[:n], y1[:n], y2[:n], y1[:n] - y2[:n]]
        else:
            hdr = f"{self._xl_last},RP,AP"
            cols = [x[:n], y1[:n], y2[:n]]
        np.savetxt(path, np.column_stack(cols),
                   delimiter=",", fmt="%.10g", header=hdr, comments="")


# ══════════════════════════════════════════════════════════════════════
# ── AUC two-pane widget ───────────────────────────────────────────────────────

class FitAucWidget(QtWidgets.QGroupBox):
    """Two stacked plots: per-peak metric (top) and peak1+peak2 sum (bottom).

    In power-dependence mode runs a Hill / saturation fit on each curve and
    overlays the fit (dashed) and the low-density asymptote (dotted).
    """

    _DASH = QtCore.Qt.DashLine
    _DOT = QtCore.Qt.DotLine

    def __init__(self, title: str = "Peak AUC vs Sweep",
                 y_label: str = "AUC (a.u.·nm)",
                 total_label: str = "Total AUC (a.u.·nm)",
                 parent=None):
        super().__init__(title, parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.showAllChk = QtWidgets.QCheckBox("Show all fixed values")
        self.copyArrayBtn = QtWidgets.QPushButton("Copy Array")
        self.copyArrayBtn.setToolTip(
            "Copy power, n_ex, RP/AP data + saturation fit + asymptote to clipboard\n"
            "(tab-separated, paste directly into Origin)")
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.showAllChk)
        top_row.addStretch()
        top_row.addWidget(self.copyArrayBtn)
        layout.addLayout(top_row)

        self._plot_indiv = pg.PlotWidget()
        _fix_pg_axes(self._plot_indiv)
        self._plot_indiv.setLabel("left", y_label)
        self._plot_indiv.setLabel("bottom", "X")
        self._plot_indiv.showGrid(x=True, y=True, alpha=0.25)
        self._plot_indiv.addLegend(offset=(-10, 10))
        self._c1 = self._plot_indiv.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#1f77b4", symbolPen=None, name="RP")
        self._c2 = self._plot_indiv.plot(
            pen=pg.mkPen("#d62728", width=1.2), symbol="o",
            symbolSize=4, symbolBrush="#d62728", symbolPen=None, name="AP")
        self._fit1 = self._plot_indiv.plot(
            pen=pg.mkPen("#1f77b4", width=1.2, style=self._DASH))
        self._asym1 = self._plot_indiv.plot(
            pen=pg.mkPen("#1f77b4", width=1.0, style=self._DOT))
        self._fit2 = self._plot_indiv.plot(
            pen=pg.mkPen("#d62728", width=1.2, style=self._DASH))
        self._asym2 = self._plot_indiv.plot(
            pen=pg.mkPen("#d62728", width=1.0, style=self._DOT))

        self._plot_total = pg.PlotWidget()
        _fix_pg_axes(self._plot_total)
        self._plot_total.setLabel("left", total_label)
        self._plot_total.setLabel("bottom", "X")
        self._plot_total.showGrid(x=True, y=True, alpha=0.25)
        self._ct = self._plot_total.plot(
            pen=pg.mkPen("#2ca02c", width=1.4), symbol="o",
            symbolSize=4, symbolBrush="#2ca02c", symbolPen=None)
        self._fit_t = self._plot_total.plot(
            pen=pg.mkPen("#2ca02c", width=1.2, style=self._DASH))
        self._asym_t = self._plot_total.plot(
            pen=pg.mkPen("#2ca02c", width=1.0, style=self._DOT))

        layout.addWidget(self._plot_indiv)
        layout.addWidget(self._plot_total)

        self._ovl_items_indiv: list = []
        self._ovl_items_total: list = []
        self._ovl_ys_indiv: list = []   # arrays for range pinning
        self._ovl_ys_total: list = []
        self._top_x = None
        self._top_label = ""
        self._x_last = None
        self._y1_last = None
        self._y2_last = None

        self.copyArrayBtn.clicked.connect(self._on_copy_array)

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    # ------------------------------------------------------------------ #
    def set_overlays(self, x, series) -> None:
        """Overlay curves for every fixed-axis value.

        series: list of (y1, y2, y_total); RP blues / AP reds on the top pane,
        totals in greens on the bottom pane; light→dark = increasing fixed value.
        """
        self.clear_overlays()
        x = np.asarray(x, dtype=float).ravel()
        n = len(series)
        pens1 = _shade_pens(_OVL_BLUE,  n)
        pens2 = _shade_pens(_OVL_RED,   n)
        penst = _shade_pens(_OVL_GREEN, n)
        for (y1, y2, yt), p1, p2, pt in zip(series, pens1, pens2, penst):
            for y, pen, pw, items, ys in (
                    (y1, p1, self._plot_indiv, self._ovl_items_indiv, self._ovl_ys_indiv),
                    (y2, p2, self._plot_indiv, self._ovl_items_indiv, self._ovl_ys_indiv),
                    (yt, pt, self._plot_total, self._ovl_items_total, self._ovl_ys_total)):
                y = np.asarray(y, dtype=float).ravel()
                nn = min(x.size, y.size)
                m = np.isfinite(x[:nn]) & np.isfinite(y[:nn])
                c = pw.plot(x[:nn][m], y[:nn][m], pen=pen)
                c.setZValue(-5)
                items.append(c)
                ys.append(y[:nn][m])
        # widen the view so overlays and the main curves are all visible
        d1 = [d for c in (self._c1, self._c2) for d in [c.getData()[1]] if d is not None]
        dt = [d for d in [self._ct.getData()[1]] if d is not None]
        if self._ovl_ys_indiv:
            self._pin_range(self._plot_indiv, x, *(self._ovl_ys_indiv + d1))
        if self._ovl_ys_total:
            self._pin_range(self._plot_total, x, *(self._ovl_ys_total + dt))

    def clear_overlays(self) -> None:
        for c in self._ovl_items_indiv:
            self._plot_indiv.removeItem(c)
        for c in self._ovl_items_total:
            self._plot_total.removeItem(c)
        self._ovl_items_indiv = []
        self._ovl_items_total = []
        self._ovl_ys_indiv = []
        self._ovl_ys_total = []

    # ------------------------------------------------------------------ #
    # Hill / saturation fitting (same functional form as saturation_fit.py)
    # f(x) = Vmax * x^n / (K^n + x^n) + c
    # low-density asymptote: Vmax * (x/K)^n + c
    @staticmethod
    def _hill(x, Vmax, n, K, c):
        return Vmax * x**n / (K**n + x**n) + c

    def _fit_one(self, x, y, fit_curve, asym_curve):
        """Fit Hill model; update curves. Returns popt or None."""
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() < 4:
            fit_curve.setData([], [])
            asym_curve.setData([], [])
            return None
        xm, ym = x[m], y[m]
        c0 = float(np.nanmin(ym))
        Vm0 = max(float(np.nanmax(ym)) - c0, 1e-10)
        K0 = float(np.median(xm))
        p0 = [Vm0, 0.8, K0, c0]
        lo = [0.0, 0.3, float(xm.min()) * 0.01, 0.0]
        hi = [Vm0 * 20, 1.0, float(xm.max()) * 10, c0 * 2 + 1.0]
        try:
            from scipy.optimize import curve_fit as _cf
            popt, _ = _cf(self._hill, xm, ym, p0=p0,
                          bounds=(lo, hi), maxfev=20000)
        except Exception:
            fit_curve.setData([], [])
            asym_curve.setData([], [])
            return None
        Vmax, n_f, K_f, c_f = popt
        xx = np.linspace(float(xm.min()), float(xm.max()), 400)
        fit_curve.setData(xx,  self._hill(xx, *popt))
        asym_curve.setData(xx, Vmax * (xx / K_f)**n_f + c_f)
        return popt

    def _run_sat_fits(self, x, y1, y2, total, vg: Optional[float] = None):
        p1 = self._fit_one(x, y1,    self._fit1,  self._asym1)
        p2 = self._fit_one(x, y2,    self._fit2,  self._asym2)
        pt = self._fit_one(x, total, self._fit_t, self._asym_t)

        def _label(p):
            if p is None:
                return "no fit"
            Vm, nf, Kf, cf = p
            if vg is not None and np.isfinite(vg):
                rb = _cal_rb_nm(vg, Kf)
                if np.isfinite(rb):
                    return f"Vmax={Vm:.3g}  n={nf:.2f}  R_b={rb:.3g} nm"
            return f"Vmax={Vm:.3g}  n={nf:.2f}  K={Kf:.3g}"

        parts = []
        if p1 is not None:
            parts.append(f"RP: {_label(p1)}")
        if p2 is not None:
            parts.append(f"AP: {_label(p2)}")
        self._plot_indiv.setTitle(" | ".join(parts) if parts else "")
        self._plot_total.setTitle(_label(pt) if pt is not None else "")

    def _clear_sat_fits(self):
        for c in (self._fit1, self._asym1, self._fit2, self._asym2,
                  self._fit_t, self._asym_t):
            c.setData([], [])

    def _on_copy_array(self) -> None:
        """Copy Power, n_ex, RP/AP data + saturation fit + asymptote to the
        clipboard as tab-separated text, ready to paste into Origin."""
        if self._x_last is None or self._x_last.size == 0:
            QtWidgets.QMessageBox.warning(self, "Copy Array", "No data.")
            return
        x = self._x_last
        y1 = self._y1_last if self._y1_last is not None else np.full_like(x, np.nan)
        y2 = self._y2_last if self._y2_last is not None else np.full_like(x, np.nan)
        nex_label = self._top_label or "n_ex"

        # n_ex is linear in power at fixed gate: derive the power->n_ex ratio
        # from the data points, then reuse it for the fit curve's own x-grid.
        if self._top_x is not None and self._top_x.size == x.size:
            finite = np.isfinite(x) & (x != 0) & np.isfinite(self._top_x)
            ratio = float(np.median(self._top_x[finite] / x[finite])) if finite.any() else np.nan
        else:
            ratio = np.nan
        nex_data = x * ratio if np.isfinite(ratio) else np.full_like(x, np.nan)

        fx1, fy1 = self._fit1.getData()
        _,   ay1 = self._asym1.getData()
        fx2, fy2 = self._fit2.getData()
        _,   ay2 = self._asym2.getData()
        fx1 = np.asarray([] if fx1 is None else fx1, dtype=float)
        fx2 = np.asarray([] if fx2 is None else fx2, dtype=float)
        fit_x = fx1 if fx1.size >= fx2.size else fx2
        fit_nex = fit_x * ratio if np.isfinite(ratio) else np.full_like(fit_x, np.nan)

        n_rows = max(x.size, fit_x.size)

        def _pad(arr):
            arr = np.asarray([] if arr is None else arr, dtype=float)
            return np.pad(arr, (0, max(0, n_rows - arr.size)), constant_values=np.nan)[:n_rows]

        cols = [x, nex_data, y1, y2, fit_x, fit_nex, fy1, ay1, fy2, ay2]
        header = ["Power (µW)", nex_label, "RP_data", "AP_data",
                  "Power_fit (µW)", f"{nex_label}_fit",
                  "RP_fit", "RP_asymptote", "AP_fit", "AP_asymptote"]
        lines = ["\t".join(header)]
        for row in zip(*(_pad(c) for c in cols)):
            lines.append("\t".join("" if not np.isfinite(v) else f"{v:.10g}" for v in row))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    @staticmethod
    def _pin_range(pw, xs, *ys):
        """Set view range of pw based on (xs, ys) data only; fit curves are ignored."""
        xv = np.asarray(xs, float).ravel()
        xv = xv[np.isfinite(xv)]
        yv = np.concatenate([np.asarray(y, float).ravel() for y in ys])
        yv = yv[np.isfinite(yv)]
        if xv.size == 0 or yv.size == 0:
            pw.enableAutoRange()
            return
        x0, x1 = float(xv.min()), float(xv.max())
        y0, y1 = float(yv.min()), float(yv.max())
        xp = (x1 - x0) * 0.05 or max(abs(x0) * 0.05, 1e-6)
        yp = (y1 - y0) * 0.10 or max(abs(y0) * 0.10, 1e-6)
        pw.setXRange(x0 - xp, x1 + xp, padding=0)
        pw.setYRange(y0 - yp, y1 + yp, padding=0)

    # ------------------------------------------------------------------ #
    def set_data(self, x, auc1, auc2, *, x_label: str,
                 do_saturation_fit: bool = False, vg: Optional[float] = None) -> None:
        x = np.asarray(x,    dtype=float).ravel()
        auc1 = np.asarray(auc1, dtype=float).ravel()
        auc2 = np.asarray(auc2, dtype=float).ravel()
        n = min(x.size, auc1.size, auc2.size)
        x = x[:n]
        auc1 = auc1[:n]
        auc2 = auc2[:n]
        self._x_last = x.copy()
        self._y1_last = auc1.copy()
        self._y2_last = auc2.copy()
        total = np.where(
            np.isfinite(auc1) | np.isfinite(auc2),
            np.nansum(np.column_stack([auc1, auc2]), axis=1).astype(float),
            np.nan,
        )
        m1 = np.isfinite(x) & np.isfinite(auc1)
        m2 = np.isfinite(x) & np.isfinite(auc2)
        mt = np.isfinite(x) & np.isfinite(total)
        self._c1.setData(x[m1], auc1[m1])
        self._c2.setData(x[m2], auc2[m2])
        self._ct.setData(x[mt], total[mt])
        _apply_top_axis(self._plot_indiv.plotItem, x, self._top_x, self._top_label)
        for pw in (self._plot_indiv, self._plot_total):
            pw.setLabel("bottom", x_label)
            if not do_saturation_fit:
                pw.setTitle("")
        if do_saturation_fit:
            self._run_sat_fits(x, auc1, auc2, total, vg=vg)
        else:
            self._clear_sat_fits()
        # Set view to data bounds only — fit/asymptote may extend far outside.
        # Overlay curves (all fixed values), when present, stay in view too.
        self._pin_range(self._plot_indiv, np.concatenate([x[m1], x[m2]]),
                        auc1[m1], auc2[m2], *self._ovl_ys_indiv)
        self._pin_range(self._plot_total, x[mt], total[mt],
                        *self._ovl_ys_total)

    def clear_data(self, message: str, *, x_label: str = "X") -> None:
        self._x_last = None
        self._y1_last = None
        self._y2_last = None
        self._c1.setData([], [])
        self._c2.setData([], [])
        self._ct.setData([], [])
        self._clear_sat_fits()
        _apply_top_axis(self._plot_indiv.plotItem, None, None, "")
        for pw in (self._plot_indiv, self._plot_total):
            pw.setLabel("bottom", x_label)
            pw.setTitle(message)

    def all_viewboxes(self):
        return [self._plot_indiv.getViewBox(), self._plot_total.getViewBox()]


class FitSatDevWidget(QtWidgets.QGroupBox):
    """Two panes: (V−c)·(1−θ) showing the non-linear signal contribution.

    Top  — intensity-based (V = fitted amplitude)
    Bottom — AUC-based     (V = amp × gamma × π or √2π)

    θ = (V−c)/Vmax is the fractional saturation (0=linear, 1=saturated).
    (V−c)·(1−θ) = S·(1−S/Vmax): 0 at low power, peaks at half-saturation, 0 at full saturation.
    Only populated in power-dependence mode.
    """

    def __init__(self, parent=None):
        super().__init__("Saturation Deviation  (V−c)·(1−θ)", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        def _make_pw(y_label):
            pw = pg.PlotWidget()
            _fix_pg_axes(pw)
            pw.setLabel("left", y_label)
            pw.setLabel("bottom", "X")
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.addLegend(offset=(-10, 10))
            c1 = pw.plot(pen=pg.mkPen("#1f77b4", width=1.2), symbol="o",
                         symbolSize=4, symbolBrush="#1f77b4", symbolPen=None, name="RP")
            c2 = pw.plot(pen=pg.mkPen("#d62728", width=1.2), symbol="o",
                         symbolSize=4, symbolBrush="#d62728", symbolPen=None, name="AP")
            return pw, c1, c2

        self._plot_int, self._ci1, self._ci2 = _make_pw(
            "(V−c)·(1−θ)  intensity")
        self._plot_auc, self._ca1, self._ca2 = _make_pw("(V−c)·(1−θ)  AUC")
        layout.addWidget(self._plot_int)
        layout.addWidget(self._plot_auc)

        self._top_x = None
        self._top_label = ""

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hill(x, Vmax, n, K, c):
        return Vmax * x**n / (K**n + x**n) + c

    def _compute_sat_dev(self, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Fit Hill to (x, y) and return (y−c)·(1−θ) at all x points."""
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() < 4:
            return None
        xm, ym = x[m], y[m]
        c0 = float(np.nanmin(ym))
        Vm0 = max(float(np.nanmax(ym)) - c0, 1e-10)
        K0 = float(np.median(xm))
        try:
            from scipy.optimize import curve_fit as _cf
            popt, _ = _cf(
                self._hill, xm, ym,
                p0=[Vm0, 0.8, K0, c0],
                bounds=([0.0, 0.3, float(xm.min()) * 0.01, 0.0],
                        [Vm0 * 20, 1.0, float(xm.max()) * 10, c0 * 2 + 1.0]),
                maxfev=20000,
            )
        except Exception:
            return None
        Vmax, n_f, K_f, c_f = popt
        V_above_c = y - c_f                     # S = V − c  ∈ [0, Vmax]
        # θ = (V−c)/Vmax: 0=linear, 1=saturated
        theta = V_above_c / Vmax
        # peaks at half-saturation, 0 at both extremes
        return V_above_c * (1.0 - theta)

    @staticmethod
    def _pin_range(pw, xs, *ys):
        xv = np.asarray(xs, float).ravel()
        xv = xv[np.isfinite(xv)]
        yv = np.concatenate([np.asarray(y, float).ravel() for y in ys])
        yv = yv[np.isfinite(yv)]
        if xv.size == 0 or yv.size == 0:
            pw.enableAutoRange()
            return
        x0, x1 = float(xv.min()), float(xv.max())
        y0, y1 = float(yv.min()), float(yv.max())
        xp = (x1 - x0) * 0.05 or max(abs(x0) * 0.05, 1e-6)
        yp = (y1 - y0) * 0.10 or max(abs(y0) * 0.10, 1e-6)
        pw.setXRange(x0 - xp, x1 + xp, padding=0)
        pw.setYRange(y0 - yp, y1 + yp, padding=0)

    # ------------------------------------------------------------------ #
    def set_data(self, x, a1, a2, auc1, auc2, *,
                 x_label: str, do_sat_dev: bool = False) -> None:
        x = np.asarray(x,    float).ravel()
        a1 = np.asarray(a1,   float).ravel()
        a2 = np.asarray(a2,   float).ravel()
        auc1 = np.asarray(auc1, float).ravel()
        auc2 = np.asarray(auc2, float).ravel()
        n = min(x.size, a1.size, a2.size, auc1.size, auc2.size)
        x = x[:n]
        a1 = a1[:n]
        a2 = a2[:n]
        auc1 = auc1[:n]
        auc2 = auc2[:n]

        for pw in (self._plot_int, self._plot_auc):
            pw.setLabel("bottom", x_label)
        _apply_top_axis(self._plot_int.plotItem, x, self._top_x, self._top_label)

        if not do_sat_dev:
            for c in (self._ci1, self._ci2, self._ca1, self._ca2):
                c.setData([], [])
            for pw in (self._plot_int, self._plot_auc):
                pw.setTitle("Power mode only")
            return

        for pw in (self._plot_int, self._plot_auc):
            pw.setTitle("")

        x_pos = np.clip(x, 1e-20, None)

        def _plot(curve, pw_partner, yv, other_yv, label_partner=None):
            sd = self._compute_sat_dev(x_pos, yv)
            if sd is not None:
                m = np.isfinite(x) & np.isfinite(sd)
                curve.setData(x[m], sd[m])
            else:
                curve.setData([], [])
            return sd

        sd_i1 = _plot(self._ci1, self._plot_int, a1,   a2)
        sd_i2 = _plot(self._ci2, self._plot_int, a2,   a1)
        sd_a1 = _plot(self._ca1, self._plot_auc, auc1, auc2)
        sd_a2 = _plot(self._ca2, self._plot_auc, auc2, auc1)

        # Pin ranges to data only
        i_ys = [s[np.isfinite(s)] for s in (sd_i1, sd_i2) if s is not None]
        a_ys = [s[np.isfinite(s)] for s in (sd_a1, sd_a2) if s is not None]
        if i_ys:
            self._pin_range(self._plot_int, x, *i_ys)
        if a_ys:
            self._pin_range(self._plot_auc, x, *a_ys)

    def clear_data(self, message: str, *, x_label: str = "X") -> None:
        for c in (self._ci1, self._ci2, self._ca1, self._ca2):
            c.setData([], [])
        _apply_top_axis(self._plot_int.plotItem, None, None, "")
        for pw in (self._plot_int, self._plot_auc):
            pw.setLabel("bottom", x_label)
            pw.setTitle(message)

    def all_viewboxes(self):
        return [self._plot_int.getViewBox(), self._plot_auc.getViewBox()]


# ── Intensity / AUC fraction widget ───────────────────────────────────────────

class FitRatioWidget(QtWidgets.QGroupBox):
    """Two panes showing each peak's share of the total signal.

    Top    — intensity ratio: P1/(P1+P2) and P2/(P1+P2)
    Bottom — AUC ratio:       same with AUC values
    Only populated in power-dependence mode.
    """
    _C1 = (0.298, 0.447, 0.690)   # blue
    _C2 = (0.769, 0.306, 0.322)   # red

    def __init__(self, parent=None):
        super().__init__("Peak Fraction", parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        def _make_pw(title):
            pw = pg.PlotWidget(title=title)
            _fix_pg_axes(pw)
            pw.setLabel("left", "Fraction")
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setYRange(0, 1)
            return pw

        self._plot_int = _make_pw("Intensity fraction")
        self._plot_auc = _make_pw("AUC fraction")
        lay.addWidget(self._plot_int)
        lay.addWidget(self._plot_auc)

        def _pen(rgb): return pg.mkPen(color=tuple(int(c*255) for c in rgb), width=1.5)
        def _curves(pw):
            c1 = pw.plot([], [], pen=_pen(self._C1), name="RP")
            c2 = pw.plot([], [], pen=_pen(self._C2), name="AP")
            pw.addLegend(offset=(5, 5))
            return c1, c2

        self._ci1, self._ci2 = _curves(self._plot_int)
        self._ca1, self._ca2 = _curves(self._plot_auc)

        self._top_x = None
        self._top_label = ""

    def set_secondary_axis(self, x2: Optional[np.ndarray], label: str = "") -> None:
        self._top_x = None if x2 is None else np.asarray(x2, dtype=float).ravel()
        self._top_label = str(label or "")

    def set_data(self, x, a1, a2, auc1, auc2, *, x_label):
        for pw in (self._plot_int, self._plot_auc):
            pw.setLabel("bottom", x_label)
            pw.setTitle(pw.plotItem.titleLabel.text)

        xf = np.asarray(x, dtype=float)
        _apply_top_axis(self._plot_int.plotItem, xf, self._top_x, self._top_label)

        def _ratio(v1, v2):
            tot = v1 + v2
            with np.errstate(invalid="ignore", divide="ignore"):
                r1 = np.where(tot > 0, v1 / tot, np.nan)
                r2 = np.where(tot > 0, v2 / tot, np.nan)
            return r1, r2

        r1i, r2i = _ratio(np.asarray(a1,    dtype=float),
                          np.asarray(a2,    dtype=float))
        r1a, r2a = _ratio(np.asarray(auc1, dtype=float),
                          np.asarray(auc2, dtype=float))

        for c, r in ((self._ci1, r1i), (self._ci2, r2i),
                     (self._ca1, r1a), (self._ca2, r2a)):
            ok = np.isfinite(r)
            c.setData(xf[ok], r[ok])

        for pw in (self._plot_int, self._plot_auc):
            pw.setXRange(float(np.nanmin(xf)), float(np.nanmax(xf)), padding=0.02)

    def clear_data(self, message, *, x_label="X"):
        for c in (self._ci1, self._ci2, self._ca1, self._ca2):
            c.setData([], [])
        _apply_top_axis(self._plot_int.plotItem, None, None, "")
        for pw in (self._plot_int, self._plot_auc):
            pw.setLabel("bottom", x_label)
            pw.setTitle(message)

    def all_viewboxes(self):
        return [self._plot_int.getViewBox(), self._plot_auc.getViewBox()]


# ── GIF frame selector ────────────────────────────────────────────────────────

def _centers_to_edges(c: np.ndarray) -> np.ndarray:
    """Convert N cell-centre values to N+1 edge values (midpoint rule, extrapolated at ends)."""
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    edges = np.empty(c.size + 1)
    edges[1:-1] = (c[:-1] + c[1:]) * 0.5
    edges[0] = c[0] - (c[1] - c[0]) * 0.5
    edges[-1] = c[-1] + (c[-1] - c[-2]) * 0.5
    return edges


class SatMapWidget(QtWidgets.QGroupBox):
    """Three θ = (y−c)/Vmax colormaps (RP, AP, Total) from Hill fits across all gates.

    Uses PColorMeshItem so non-uniform power/gate spacing is rendered correctly.
    """

    def __init__(self, parent=None):
        super().__init__("Saturation Maps  θ = (signal−c) / Vmax", parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── controls row ──
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Metric:"))
        self.metricCombo = QtWidgets.QComboBox()
        self.metricCombo.addItems(["AUC", "Intensity"])
        ctrl.addWidget(self.metricCombo)
        self.fitAllBtn = QtWidgets.QPushButton("Fit All Gates")
        ctrl.addWidget(self.fitAllBtn)
        self._statusLbl = QtWidgets.QLabel("")
        ctrl.addWidget(self._statusLbl, 1)
        outer.addLayout(ctrl)

        # ── 3 stacked colormaps ──
        try:
            _cmap = pg.colormap.get("plasma")
        except Exception:
            _cmap = pg.colormap.get("hot")

        self._pws = []
        self._items = []   # PColorMeshItem or ImageItem
        self._cmap = _cmap
        for title in ("RP  θ", "AP  θ", "Total  θ"):
            pw = pg.PlotWidget()
            _fix_pg_axes(pw)
            pw.setLabel("bottom", "Gate (V)")
            pw.setLabel("left",   "Power (µW)")
            pw.setTitle(title)
            try:
                item = pg.PColorMeshItem(colorMap=_cmap)
                item.setLevels((0.0, 1.0))
            except AttributeError:
                # pyqtgraph < 0.12.4 fallback — spacing will be uniform
                item = pg.ImageItem()
                item.setColorMap(_cmap)
            pw.addItem(item)
            self._pws.append(pw)
            self._items.append(item)
            outer.addWidget(pw)

    def set_data(self, gates: np.ndarray, powers_uW: np.ndarray,
                 theta1: np.ndarray, theta2: np.ndarray, theta_t: np.ndarray,
                 status: str = "") -> None:
        self._statusLbl.setText(status)
        g_edges = _centers_to_edges(gates)
        p_edges = _centers_to_edges(np.clip(powers_uW, 1e-20, None))
        # y-coordinates in log10 space so non-uniform power spacing renders correctly
        p_edges_log = np.log10(p_edges)
        # shape (n_powers+1, n_gates+1)
        x_e, y_e = np.meshgrid(g_edges, p_edges_log)

        # Build y-axis tick labels: one per actual power value
        p_safe = np.clip(powers_uW, 1e-20, None)
        ticks = [(float(np.log10(p)), f"{p:.3g}") for p in p_safe]

        for item, theta in zip(self._items, (theta1, theta2, theta_t)):
            clipped = np.clip(theta, -0.1, 1.1)   # shape (n_powers, n_gates)
            if isinstance(item, pg.ImageItem):
                item.setImage(clipped, levels=(0.0, 1.0))
                g0, g1 = float(g_edges[0]), float(g_edges[-1])
                lp0, lp1 = float(p_edges_log[0]), float(p_edges_log[-1])
                item.setRect(QtCore.QRectF(g0, lp0, g1 - g0, lp1 - lp0))
            else:
                item.setData(x_e, y_e, clipped)

        for pw in self._pws:
            pw.getAxis("left").setTicks([ticks, []])
            pw.getAxis("left").setLabel("Power (µW)")

    def clear(self, message: str = "") -> None:
        self._statusLbl.setText(message)
        for item in self._items:
            if isinstance(item, pg.ImageItem):
                item.clear()
            else:
                try:
                    item.setData(np.zeros((2, 2)), np.zeros((2, 2)),
                                 np.zeros((1, 1)))
                except Exception:
                    pass


class GateNormMapWidget(QtWidgets.QGroupBox):
    """Three colormaps of fitted intensity vs (power, gate voltage).

    x = power (µW, log10-spaced like the θ maps), y = gate voltage,
    colour = fitted peak intensity normalized to [0, 1] within each power
    (i.e. across the gate axis at fixed power).
    Populated by the same "Fit All Gates" sweep as the θ Maps tab.
    """

    def __init__(self, parent=None):
        super().__init__("Intensity vs Gate  (normalized per power)", parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        self.fitAllBtn = QtWidgets.QPushButton("Fit All Gates")
        ctrl.addWidget(self.fitAllBtn)
        self._statusLbl = QtWidgets.QLabel("")
        ctrl.addWidget(self._statusLbl, 1)
        outer.addLayout(ctrl)

        # Diverging RdBu: red at 0, ~white at 0.5, blue at 1 — the white
        # midpoint marks where a gate sits halfway between that power's extremes.
        try:
            _cmap = pg.colormap.get("RdBu", source="matplotlib")
        except Exception:
            _cmap = pg.ColorMap(
                pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                color=np.array([[178,  24,  43, 255],    # red
                                [244, 165, 130, 255],    # light red
                                [247, 247, 247, 255],    # white
                                [146, 197, 222, 255],    # light blue
                                [ 33, 102, 172, 255]],   # blue
                               dtype=np.ubyte))

        self._pws = []
        self._items = []
        self._crossScatters = []   # P1/P2 intensity crossing markers, one per plot
        for title in ("RP intensity (norm.)", "AP intensity (norm.)",
                      "Total intensity (norm.)"):
            pw = pg.PlotWidget()
            _fix_pg_axes(pw)
            pw.setLabel("bottom", "Power (µW)")
            pw.setLabel("left",   "Gate (V)")
            pw.setTitle(title)
            try:
                item = pg.PColorMeshItem(colorMap=_cmap)
                item.setLevels((0.0, 1.0))
            except AttributeError:
                item = pg.ImageItem()
                item.setColorMap(_cmap)
            pw.addItem(item)
            sc = pg.ScatterPlotItem(
                symbol="o", size=7,
                brush=pg.mkBrush(255, 255, 255, 230),
                pen=pg.mkPen("k", width=1.2))
            sc.setZValue(10)
            pw.addItem(sc)
            self._pws.append(pw)
            self._items.append(item)
            self._crossScatters.append(sc)
            outer.addWidget(pw)

    @staticmethod
    def _norm_per_power(m: np.ndarray) -> np.ndarray:
        """Normalize each power row (axis 1 = gates) to [0, 1]."""
        m = np.asarray(m, dtype=float)
        out = np.full_like(m, np.nan)
        for p in range(m.shape[0]):
            row = m[p]
            fin = row[np.isfinite(row)]
            if fin.size < 2:
                continue
            lo, hi = float(np.min(fin)), float(np.max(fin))
            if hi > lo:
                out[p] = (row - lo) / (hi - lo)
            else:
                out[p] = 0.5
        return out

    def set_data(self, gates: np.ndarray, powers_uW: np.ndarray,
                 i1: np.ndarray, i2: np.ndarray, i_t: np.ndarray,
                 status: str = "") -> None:
        """i1/i2/i_t shape (n_powers, n_gates) — raw fitted intensities."""
        self._statusLbl.setText(status)
        g_edges = _centers_to_edges(np.asarray(gates, dtype=float))
        p_safe  = np.clip(np.asarray(powers_uW, dtype=float), 1e-20, None)
        p_edges_log = np.log10(_centers_to_edges(p_safe))
        # x = power (log), y = gate → grids shape (n_gates+1, n_powers+1)
        x_e, y_e = np.meshgrid(p_edges_log, g_edges)

        ticks = [(float(np.log10(p)), f"{p:.3g}") for p in p_safe]

        for item, mat in zip(self._items, (i1, i2, i_t)):
            norm = self._norm_per_power(mat)          # (n_powers, n_gates)
            z = norm.T                                # (n_gates, n_powers)
            clipped = np.clip(z, 0.0, 1.0)
            if isinstance(item, pg.ImageItem):
                item.setImage(clipped, levels=(0.0, 1.0))
                lp0, lp1 = float(p_edges_log[0]), float(p_edges_log[-1])
                g0, g1 = float(g_edges[0]), float(g_edges[-1])
                item.setRect(QtCore.QRectF(lp0, g0, lp1 - lp0, g1 - g0))
            else:
                item.setData(x_e, y_e, clipped)

        for pw in self._pws:
            pw.getAxis("bottom").setTicks([ticks, []])
            pw.getAxis("bottom").setLabel("Power (µW)")

        # P1/P2 crossing points: for each power, gates where the raw fitted
        # intensities I1(V) and I2(V) intersect (sign change of I1−I2,
        # linearly interpolated in V).
        g = np.asarray(gates, dtype=float)
        xs_pts: list = []
        ys_pts: list = []
        i1a = np.asarray(i1, dtype=float)
        i2a = np.asarray(i2, dtype=float)
        for p in range(min(i1a.shape[0], p_safe.size)):
            d  = i1a[p] - i2a[p]
            lx = float(np.log10(p_safe[p]))
            for j in range(g.size - 1):
                d0, d1 = d[j], d[j + 1]
                if not (np.isfinite(d0) and np.isfinite(d1)):
                    continue
                if d0 == 0.0:
                    xs_pts.append(lx)
                    ys_pts.append(float(g[j]))
                elif d0 * d1 < 0:
                    v = g[j] + (g[j + 1] - g[j]) * d0 / (d0 - d1)
                    xs_pts.append(lx)
                    ys_pts.append(float(v))
            if g.size and np.isfinite(d[-1]) and d[-1] == 0.0:
                xs_pts.append(lx)
                ys_pts.append(float(g[-1]))
        for sc in self._crossScatters:
            sc.setData(xs_pts, ys_pts)

    def clear(self, message: str = "") -> None:
        self._statusLbl.setText(message)
        for sc in self._crossScatters:
            sc.setData([], [])
        for item in self._items:
            if isinstance(item, pg.ImageItem):
                item.clear()
            else:
                try:
                    item.setData(np.zeros((2, 2)), np.zeros((2, 2)),
                                 np.zeros((1, 1)))
                except Exception:
                    pass


class RbEFWidget(QtWidgets.QGroupBox):
    """R_b (bimolecular recombination radius, nm) vs E_F, from a Hill fit of
    fitted peak intensity vs power at each gate in the sweep.

    R_b = 1/sqrt(pi * n_ex(Vg, K)), K = fitted Hill saturation power (eta=1 assumed).
    Populated by the "Fit All Gates" sweep (same pattern as theta Maps / Gate Norm).
    """

    def __init__(self, parent=None):
        super().__init__("R_b vs E_F", parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        self.fitAllBtn = QtWidgets.QPushButton("Fit All Gates")
        ctrl.addWidget(self.fitAllBtn)
        self._statusLbl = QtWidgets.QLabel("")
        ctrl.addWidget(self._statusLbl, 1)
        outer.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        _fix_pg_axes(self._plot)
        self._plot.setLabel("left",   "R_b (nm)")
        self._plot.setLabel("bottom", CAL_EF_LABEL)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(-10, 10))
        self._c1 = self._plot.plot(
            pen=pg.mkPen("#1f77b4", width=1.2), symbol="o",
            symbolSize=5, symbolBrush="#1f77b4", symbolPen=None, name="RP")
        self._c2 = self._plot.plot(
            pen=pg.mkPen("#d62728", width=1.2), symbol="o",
            symbolSize=5, symbolBrush="#d62728", symbolPen=None, name="AP")
        outer.addWidget(self._plot)

    def set_data(self, ef_mev: np.ndarray, rb1: np.ndarray, rb2: np.ndarray,
                 status: str = "") -> None:
        self._statusLbl.setText(status)
        x = np.asarray(ef_mev, dtype=float).ravel()
        rb1 = np.asarray(rb1, dtype=float).ravel()
        rb2 = np.asarray(rb2, dtype=float).ravel()
        order = np.argsort(x)
        x, rb1, rb2 = x[order], rb1[order], rb2[order]
        m1 = np.isfinite(x) & np.isfinite(rb1)
        m2 = np.isfinite(x) & np.isfinite(rb2)
        self._c1.setData(x[m1], rb1[m1])
        self._c2.setData(x[m2], rb2[m2])

    def clear(self, message: str = "") -> None:
        self._statusLbl.setText(message)
        self._c1.setData([], [])
        self._c2.setData([], [])


# ── GIF frame selector ────────────────────────────────────────────────────────

class GifFrameSelectDialog(QtWidgets.QDialog):
    """Pre-flight dialog: reorder / remove fixed-axis frames before the GIF is rendered."""

    def __init__(self, frame_labels: list, y_min: float = 1.4, y_max: float = 1.8,
                 parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Select GIF frames")
        self.resize(380, 540)

        hint = QtWidgets.QLabel(
            "Drag rows to reorder  ·  select + Delete to remove:")
        hint.setWordWrap(True)

        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        for idx, label in enumerate(frame_labels):
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, idx)
            self._list.addItem(item)

        btn_up = QtWidgets.QPushButton("↑ Up")
        btn_dn = QtWidgets.QPushButton("↓ Down")
        btn_del = QtWidgets.QPushButton("Delete")
        btn_up.clicked.connect(self._move_up)
        btn_dn.clicked.connect(self._move_dn)
        btn_del.clicked.connect(self._delete_selected)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(btn_up)
        btn_row.addWidget(btn_dn)
        btn_row.addWidget(btn_del)
        btn_row.addStretch()

        # Energy range row
        self._eLoSpin = QtWidgets.QDoubleSpinBox()
        self._eHiSpin = QtWidgets.QDoubleSpinBox()
        for sp in (self._eLoSpin, self._eHiSpin):
            sp.setRange(0.5, 5.0)
            sp.setDecimals(4)
            sp.setSingleStep(0.01)
            sp.setSuffix(" eV")
        self._eLoSpin.setValue(y_min)
        self._eHiSpin.setValue(y_max)
        e_row = QtWidgets.QHBoxLayout()
        e_row.addWidget(QtWidgets.QLabel("Energy:"))
        e_row.addWidget(self._eLoSpin)
        e_row.addWidget(QtWidgets.QLabel("–"))
        e_row.addWidget(self._eHiSpin)
        e_row.addStretch()

        btn_ok = QtWidgets.QPushButton("Create GIF")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        ok_row = QtWidgets.QHBoxLayout()
        ok_row.addStretch()
        ok_row.addWidget(btn_ok)
        ok_row.addWidget(btn_cancel)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self._list)
        lay.addLayout(btn_row)
        lay.addLayout(e_row)
        lay.addLayout(ok_row)

    # ── keyboard shortcut: Delete key removes selected rows ──────────────

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self._delete_selected()
        else:
            super().keyPressEvent(event)

    def _move_up(self):
        rows = sorted({self._list.row(it)
                      for it in self._list.selectedItems()})
        for row in rows:
            if row == 0:
                continue
            item = self._list.takeItem(row)
            self._list.insertItem(row - 1, item)
            item.setSelected(True)

    def _move_dn(self):
        rows = sorted({self._list.row(it)
                      for it in self._list.selectedItems()}, reverse=True)
        for row in rows:
            if row == self._list.count() - 1:
                continue
            item = self._list.takeItem(row)
            self._list.insertItem(row + 1, item)
            item.setSelected(True)

    def _delete_selected(self):
        for item in list(self._list.selectedItems()):
            self._list.takeItem(self._list.row(item))

    def original_indices(self) -> list:
        """Current list order as a list of original frame indices."""
        return [self._list.item(i).data(QtCore.Qt.UserRole)
                for i in range(self._list.count())]

    def energy_range(self) -> Tuple[float, float]:
        lo = float(self._eLoSpin.value())
        hi = float(self._eHiSpin.value())
        return (min(lo, hi), max(lo, hi))


# Main window
# ══════════════════════════════════════════════════════════════════════

def _mpl_top_axis(ax, x: Optional[np.ndarray], x2: Optional[np.ndarray], label: str = ""):
    """Add/replace a secondary top x-axis on `ax` with ticks labelled from x2.

    Returns the new secondary-axis object (or None if x2 is unavailable);
    caller is responsible for tracking/removing it before the next redraw.
    """
    x_arr = None if x is None else np.asarray(x, dtype=float).ravel()
    x2_arr = None if x2 is None else np.asarray(x2, dtype=float).ravel()
    if x_arr is None or x2_arr is None or x2_arr.size != x_arr.size or x_arr.size == 0:
        return None
    sec = ax.secondary_xaxis("top")
    nticks = min(8, x_arr.size)
    idx = np.linspace(0, x_arr.size - 1, nticks).astype(int)
    sec.set_xticks(x_arr[idx])
    sec.set_xticklabels(
        [f"{float(x2_arr[i]):.3g}" if np.isfinite(x2_arr[i]) else "" for i in idx])
    sec.set_xlabel(label, fontsize=8)
    sec.tick_params(labelsize=7)
    return sec


class MatplotlibContourDialog(QtWidgets.QDialog):
    """Separate window showing a proper matplotlib contourf of the PL map.

    Call update_plot() whenever the main window's data or contour parameters
    change.  The dialog owns its figure and redraws in-place on each call.
    """

    def __init__(self, parent=None, title: str = "Contour plot (matplotlib)"):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self.resize(700, 850)
        if not _MPL_OK:
            lbl = QtWidgets.QLabel("matplotlib not available.")
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(lbl)
            self._canvas = None
            return
        from matplotlib.gridspec import GridSpec
        self._fig = Figure(dpi=100)
        gs = GridSpec(2, 1, figure=self._fig, hspace=0.45)
        self._ax       = self._fig.add_subplot(gs[0])
        self._ax_deriv = self._fig.add_subplot(gs[1])
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self._canvas)
        self._secax_ax = None
        self._secax_deriv = None

    # ------------------------------------------------------------------
    def update_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    *, x_label: str = "", y_label: str = "",
                    n_contours: int = 15, lo_pct: float = 1.0,
                    z_deriv: Optional[np.ndarray] = None,
                    x2: Optional[np.ndarray] = None, x2_label: str = "") -> None:
        if self._canvas is None:
            return
        if self._secax_ax is not None:
            self._secax_ax.remove()
            self._secax_ax = None
        if self._secax_deriv is not None:
            self._secax_deriv.remove()
            self._secax_deriv = None
        lo_frac = float(np.clip(lo_pct / 100.0, 0.0, 0.99))
        n_lev   = max(2, int(n_contours))
        levels  = np.linspace(lo_frac, 1.0, n_lev + 1)
        X, Y    = np.meshgrid(x, y)

        # ── PL map ───────────────────────────────────────────────────────────
        self._ax.clear()
        finite = z[np.isfinite(z)]
        if finite.size > 0:
            z_min = float(np.nanmin(finite))
            z_max = float(np.nanpercentile(finite, 99.0))
            if not np.isfinite(z_max) or z_max <= z_min:
                z_max = float(np.nanmax(finite))
            z_disp = (z - z_min) / (z_max - z_min)
            self._ax.contourf(X, Y, z_disp, levels=levels,
                              cmap="viridis", extend="both")
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        self._ax.set_title("PL map", fontsize=9)
        self._secax_ax = _mpl_top_axis(self._ax, x, x2, x2_label)

        # ── Derivative map ────────────────────────────────────────────────────
        self._ax_deriv.clear()
        if z_deriv is not None:
            fin_d = z_deriv[np.isfinite(z_deriv)]
            if fin_d.size > 0:
                abs_max = float(np.nanpercentile(np.abs(fin_d), 99.0))
                if abs_max <= 0:
                    abs_max = float(np.nanmax(np.abs(fin_d))) or 1.0
                d_levels = np.linspace(-1.0, 1.0, n_lev + 1)
                z_d_norm = np.clip(z_deriv / abs_max, -1.0, 1.0)
                self._ax_deriv.contourf(X, Y, z_d_norm, levels=d_levels,
                                        cmap="coolwarm", extend="both")
            self._ax_deriv.set_xlabel(x_label)
            self._ax_deriv.set_ylabel(y_label)
            self._ax_deriv.set_title("Derivative", fontsize=9)
            self._secax_deriv = _mpl_top_axis(self._ax_deriv, x, x2, x2_label)
        else:
            self._ax_deriv.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=self._ax_deriv.transAxes, fontsize=9)
            self._ax_deriv.set_axis_off()

        self._fig.tight_layout()
        self._canvas.draw_idle()


class MatplotlibFitDialog(QtWidgets.QDialog):
    """Fitted-peak map, θ-normalised map, and per-peak fit-quality subplot."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Fitted-peak map (matplotlib)")
        self.resize(700, 900)
        if not _MPL_OK:
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(QtWidgets.QLabel("matplotlib not available."))
            self._canvas = None
            return
        from matplotlib.gridspec import GridSpec
        self._fig = Figure()
        gs = GridSpec(3, 1, figure=self._fig,
                      height_ratios=[3, 3, 1], hspace=0.55)
        self._ax_map = self._fig.add_subplot(gs[0])
        self._ax_norm = self._fig.add_subplot(gs[1])
        self._ax_qual = self._fig.add_subplot(gs[2])
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._shared_cbar = None   # one colorbar shared by both maps, built once

        # Compact peak selector (thin bar, no groupbox chrome)
        self._rb_p1 = QtWidgets.QRadioButton("RP")
        self._rb_p2 = QtWidgets.QRadioButton("AP")
        self._rb_both = QtWidgets.QRadioButton("RP + AP")
        self._rb_both.setChecked(True)
        self._rb_p1.toggled.connect(self._redraw_maps)
        self._rb_p2.toggled.connect(self._redraw_maps)
        self._rb_both.toggled.connect(self._redraw_maps)
        _rbar = QtWidgets.QWidget()
        _rbar.setFixedHeight(24)
        _rlay = QtWidgets.QHBoxLayout(_rbar)
        _rlay.setContentsMargins(4, 0, 4, 0)
        _rlay.setSpacing(10)
        _rlay.addWidget(QtWidgets.QLabel("Norm map:"))
        for rb in (self._rb_p1, self._rb_p2, self._rb_both):
            _rlay.addWidget(rb)
        _rlay.addStretch()

        # Cached data for _redraw_maps
        self._x_norm     = None
        self._y_fit      = None
        self._z_fit      = None
        self._z_fit_p1   = None
        self._z_fit_p2   = None
        self._y_norm     = None
        self._z_norm_p1  = None
        self._z_norm_p2  = None
        self._z_norm_sum = None
        self._n_contours = 15
        self._lo_frac    = 0.0
        self._x_label    = ""
        self._y_label    = ""
        self._x2         = None
        self._x2_label   = ""
        self._secax_map  = None
        self._secax_norm = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(_rbar)
        lay.addWidget(self._canvas, stretch=1)

    def update_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    *, x_label: str = "", y_label: str = "",
                    n_contours: int = 15, lo_pct: float = 1.0,
                    q1: Optional[np.ndarray] = None,
                    q2: Optional[np.ndarray] = None,
                    z_fit_p1: Optional[np.ndarray] = None,
                    z_fit_p2: Optional[np.ndarray] = None,
                    y_norm: Optional[np.ndarray] = None,
                    z_norm_p1: Optional[np.ndarray] = None,
                    z_norm_p2: Optional[np.ndarray] = None,
                    z_norm_sum: Optional[np.ndarray] = None,
                    x2: Optional[np.ndarray] = None, x2_label: str = "") -> None:
        if self._canvas is None:
            return
        self._ax_qual.clear()

        # Store all data for _redraw_maps
        lo_frac = float(np.clip(lo_pct / 100.0, 0.0, 0.99))
        self._x_norm     = x
        self._y_fit      = y
        self._z_fit      = z
        self._z_fit_p1   = z_fit_p1
        self._z_fit_p2   = z_fit_p2
        self._y_norm     = y_norm
        self._z_norm_p1  = z_norm_p1
        self._z_norm_p2  = z_norm_p2
        self._z_norm_sum = z_norm_sum
        self._n_contours = n_contours
        self._lo_frac    = lo_frac
        self._x_label    = x_label
        self._y_label    = y_label
        self._x2         = x2
        self._x2_label   = x2_label
        self._redraw_maps(draw=False)

        # ── quality subplot ──────────────────────────────────────────────────────
        _COLORS = {"RP": "#4c9be8", "AP": "#f5a623"}
        has_q = False
        for arr, label in ((q1, "RP"), (q2, "AP")):
            if arr is not None and np.any(np.isfinite(arr)):
                self._ax_qual.plot(x, arr, label=label,
                                   color=_COLORS[label], linewidth=1.4)
                has_q = True
        if has_q:
            self._ax_qual.axhline(1.0, color="gray", linestyle="--",
                                  linewidth=0.8, alpha=0.7)
            self._ax_qual.set_xlabel(x_label)
            self._ax_qual.set_ylabel("RMS resid. / σ")
            self._ax_qual.legend(fontsize=8, loc="upper right")
        self._fig.tight_layout()
        self._canvas.draw_idle()

    _SAT_CMAP = None   # class-level cache, built once

    @classmethod
    def _get_sat_cmap(cls):
        if cls._SAT_CMAP is None:
            from matplotlib.colors import LinearSegmentedColormap as _LSC
            cls._SAT_CMAP = _LSC.from_list("sat_dev", [
                (0.000, (0.00, 0.18, 0.75)),   # blue
                # blue (flat — keeps blue dominant)
                (0.120, (0.00, 0.18, 0.75)),
                (0.149, (1.00, 1.00, 1.00)),   # white band lo
                (0.151, (1.00, 1.00, 1.00)),   # white band hi
                # yellow (fast transition after white)
                (0.180, (1.00, 1.00, 0.00)),
                (0.500, (1.00, 0.45, 0.00)),   # orange
                (0.750, (0.90, 0.00, 0.00)),   # red
                (1.000, (0.00, 0.00, 0.00)),   # black
            ])
        return cls._SAT_CMAP

    def _redraw_maps(self, draw: bool = True) -> None:
        """Redraw both ax_map and ax_norm based on the current peak radio selection."""
        if self._canvas is None:
            return
        if self._secax_map is not None:
            self._secax_map.remove()
            self._secax_map = None
        if self._secax_norm is not None:
            self._secax_norm.remove()
            self._secax_norm = None

        p1_on   = self._rb_p1.isChecked()
        p2_on   = self._rb_p2.isChecked()
        both_on = self._rb_both.isChecked()

        # ── pick arrays for each subplot ─────────────────────────────────────────
        if p1_on:
            z_fit   = self._z_fit_p1
            z_norm  = self._z_norm_p1
            fit_title  = "Reconstructed fit — P1"
            norm_title = r"$(V_{P1}-c_{P1})(1-\theta_{P1})$"
        elif p2_on:
            z_fit   = self._z_fit_p2
            z_norm  = self._z_norm_p2
            fit_title  = "Reconstructed fit — P2"
            norm_title = r"$(V_{P2}-c_{P2})(1-\theta_{P2})$"
        else:
            z_fit   = getattr(self, "_z_fit", None)
            z_norm  = self._z_norm_sum
            fit_title  = "Reconstructed fit — P1 + P2"
            norm_title = (r"$(V_{P1}-c_{P1})(1-\theta_{P1})"
                          r"+(V_{P2}-c_{P2})(1-\theta_{P2})$")

        x      = self._x_norm
        n_lev  = max(2, int(self._n_contours))
        levels = np.linspace(self._lo_frac, 1.0, n_lev + 1)

        # ── ax_map: reconstructed fit ─────────────────────────────────────────────
        self._ax_map.clear()
        y_fit = getattr(self, "_y_fit", None)
        if z_fit is not None and y_fit is not None and x is not None:
            fin = z_fit[np.isfinite(z_fit)]
            if fin.size > 0:
                zf_min = float(np.nanmin(fin))
                zf_max = float(np.nanpercentile(fin, 99.0))
                if not np.isfinite(zf_max) or zf_max <= zf_min:
                    zf_max = float(np.nanmax(fin))
                z_disp = (z_fit - zf_min) / (zf_max - zf_min)
                X, Y = np.meshgrid(x, y_fit)
                self._ax_map.contourf(X, Y, z_disp, levels=levels,
                                      cmap="viridis", extend="both")
        self._ax_map.set_xlabel(self._x_label)
        self._ax_map.set_ylabel(self._y_label)
        self._ax_map.set_title(fit_title, fontsize=9)
        self._secax_map = _mpl_top_axis(self._ax_map, x, self._x2, self._x2_label)

        # ── ax_norm: (V−c)(1−θ) map ───────────────────────────────────────────────
        self._ax_norm.clear()
        y_norm = self._y_norm
        if z_norm is not None and y_norm is not None and x is not None:
            fin_n = z_norm[np.isfinite(z_norm)]
            if fin_n.size > 0:
                zn_min = float(np.nanmin(fin_n))
                zn_max = float(np.nanpercentile(fin_n, 99.0))
                if not np.isfinite(zn_max) or zn_max <= zn_min:
                    zn_max = float(np.nanmax(fin_n))
                z_disp = (z_norm - zn_min) / (zn_max - zn_min)
                Xn, Yn = np.meshgrid(x, y_norm)
                self._ax_norm.contourf(Xn, Yn, z_disp, levels=levels,
                                       cmap="viridis", extend="both")
                if self._shared_cbar is None:
                    import matplotlib.cm as _mcm
                    import matplotlib.colors as _mc
                    _sm = _mcm.ScalarMappable(
                        cmap="viridis", norm=_mc.Normalize(vmin=0, vmax=1))
                    _sm.set_array([])
                    self._shared_cbar = self._fig.colorbar(
                        _sm, ax=[self._ax_map, self._ax_norm],
                        fraction=0.046, pad=0.04)
            self._ax_norm.set_xlabel(self._x_label)
            self._ax_norm.set_ylabel(self._y_label)
            self._ax_norm.set_title(norm_title, fontsize=9)
            self._secax_norm = _mpl_top_axis(self._ax_norm, x, self._x2, self._x2_label)
        else:
            self._ax_norm.text(0.5, 0.5, "N/A (power mode + fit required)",
                               ha="center", va="center",
                               transform=self._ax_norm.transAxes, fontsize=9)
            self._ax_norm.set_axis_off()

        if draw:
            self._fig.tight_layout()
            self._canvas.draw_idle()


class PLViewerV2(QtWidgets.QMainWindow):
    MODE_POWER = "Power dependence"
    MODE_GATE = "Gate dependence"

    def __init__(self):
        super().__init__()
        # one _Channel per key: "" = non-pol, "LH", "RH"
        self._ch: Dict[str, _Channel] = {
            "": _Channel(), "LH": _Channel(), "RH": _Channel()}
        self._crop = DEFAULT_CROP
        self._main_idx = 0
        self._fixed_idx = 0
        self._current_mode = self.MODE_POWER
        self._last_x: Optional[np.ndarray] = None
        self._last_x2: Optional[np.ndarray] = None
        self._last_x2_label = ""
        self._last_y_wl: Optional[np.ndarray] = None
        self._last_map: Optional[np.ndarray] = None
        self._last_x_label = ""
        self._fit_result = None
        self._fit_map_version = -1
        self._map_version = 0
        self._dip_fit_result = None
        self._dip_fit_map_version = -1
        self._active_overlay = "peak"   # "peak" | "dip"
        self._sat_norm_cache: Optional[tuple] = None
        self._mplDialog:    Optional["MatplotlibContourDialog"] = None
        self._mplFitDialog: Optional["MatplotlibFitDialog"] = None
        # last y_ev used in mapView
        self._last_map_ev: Optional[np.ndarray] = None
        self._last_z_ev:   Optional[np.ndarray] = None
        self._last_d_ev:   Optional[np.ndarray] = None

        self._build_ui()
        self._refresh_all()
        self._auto_load_defaults()

    # ── Startup auto-load ─────────────────────────────────────────────

    def _auto_load_defaults(self) -> None:
        """Silently load default data folder and power calibration if they exist."""
        _base = Path(
            __file__).parent.parent.parent   # OneDrive - purdue.edu root
        _data_folder = (_base / "TMDC Gated" / "260615_mose2_polaron_pl"
                        / "pl_20260615_112546_polaron_mose2_power_gate_dep")
        _calib_file = _base / "TMDC Gated" / \
            "260615_mose2_polaron_pl" / "20260615power.xlsx"

        loaded_any = False

        if _data_folder.is_dir():
            try:
                ch = self._ch[""]
                frames = _load_frames(_data_folder)
                if frames:
                    ch.clear_cache()
                    ch.data_dir = _data_folder
                    _build_channel(ch, frames)
                    self._update_data_label("")
                    self._reconfigure_sliders()
                    loaded_any = True
            except Exception:
                pass

        if _calib_file.is_file():
            try:
                import pandas as pd
                p = _calib_file
                calib_angles = calib_powers_w = None
                for h in (0, 1, None):
                    try:
                        df = (pd.read_excel(p, header=h) if p.suffix.lower() in (".xlsx", ".xls")
                              else pd.read_csv(p, header=h))
                        cols = {str(c).lower().strip(): c for c in df.columns}
                        angle_col = next(
                            (cols[k] for k in cols if "angle" in k), None)
                        power_col = next(
                            (cols[k] for k in cols if "power" in k), None)
                        if angle_col is None or power_col is None:
                            continue
                        angles = pd.to_numeric(df[angle_col], errors="coerce")
                        powers = pd.to_numeric(df[power_col], errors="coerce")
                        valid = angles.notna() & powers.notna()
                        if valid.sum() < 2:
                            continue
                        angles = np.asarray(angles[valid], dtype=float)
                        powers = np.asarray(powers[valid], dtype=float)
                        pc = str(power_col).lower()
                        scale = (1e-3 if "mw" in pc else 1e-9 if "nw" in pc
                                 else 1.0 if ("/w" in pc and "mw" not in pc
                                              and "nw" not in pc and "uw" not in pc)
                                 else 1e-6)
                        order = np.argsort(angles)
                        calib_angles = angles[order]
                        calib_powers_w = powers[order] * scale
                        break
                    except Exception:
                        continue
                if calib_angles is not None:
                    for ch in self._ch.values():
                        ch._angle_calib_angles = calib_angles
                        ch._angle_calib_powers_w = calib_powers_w
                    self.angleCalibStatusLbl.setText(
                        f"{len(calib_angles)} pts: {p.name}")
                    loaded_any = True
            except Exception:
                pass

        if loaded_any:
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
        self.liveView = SingleFrameView(
            title="PL Image", default_linecut_width=1)
        self.liveView.linecut_changed.connect(lambda _: self._update_maps())
        self.liveView.linecutWidthSpin.valueChanged.connect(
            lambda _: self._update_maps())
        split.addWidget(self.liveView)
        self.mapView = MapPlotWidget(
            "Linecut map",   cmap_name="viridis",  line_label="I")
        self.derivView = MapPlotWidget(
            "Derivative map", cmap_name="coolwarm", line_label="dI/dx")
        self.fitWidthView = FitTrendPlotWidget(
            "Lorentzian Width vs Sweep", "FWHM (meV)")
        self.fitEnergyView = FitTrendPlotWidget(
            "Peak Position vs Sweep",    "Energy (eV)",
            show_diff=True, diff_label="RP − AP (eV)")
        self.fitIntensityView = FitAucWidget("Peak Intensity vs Sweep",
                                             "Fitted Intensity (a.u.)",
                                             "Total Intensity (a.u.)")
        self.fitAucView   = FitAucWidget()
        for _w, _kind in ((self.fitEnergyView,    "energy"),
                          (self.fitIntensityView, "intensity"),
                          (self.fitAucView,       "auc")):
            _w.showAllChk.toggled.connect(
                lambda on, w=_w, k=_kind: self._on_show_all_fixed(w, k, on))
        self.satDevView   = FitSatDevWidget()
        self.fitRatioView = FitRatioWidget()
        self.satMapView = SatMapWidget()
        self.satMapView.fitAllBtn.clicked.connect(self._on_fit_all_sat_map)
        self.gateNormMapView = GateNormMapWidget()
        self.gateNormMapView.fitAllBtn.clicked.connect(self._on_fit_all_sat_map)
        self.rbEFView = RbEFWidget()
        self.rbEFView.fitAllBtn.clicked.connect(self._on_fit_rb_vs_ef)
        self.dipPosPlot = pg.PlotWidget()
        _fix_pg_axes(self.dipPosPlot)
        self.dipPosPlot.setLabel("left",   "Energy (eV)")
        self.dipPosPlot.setLabel("bottom", "X")
        self.dipPosPlot.showGrid(x=True, y=True, alpha=0.25)
        self.dipPosPlot.addLegend(offset=(-10, 10))
        self._dipPosCurves: List[object] = []

        self.derivEView = MapPlotWidget(
            "dI/dE",    cmap_name="coolwarm", line_label="dI/dE")
        self.deriv2EView = MapPlotWidget(
            "d²I/dE²", cmap_name="coolwarm", line_label="d²I/dE²")

        self.analysisTabs = QtWidgets.QTabWidget()
        self.analysisTabs.addTab(self.derivView,        "Derivative")
        self.analysisTabs.addTab(self.fitWidthView,     "Fit Width")
        self.analysisTabs.addTab(self.fitEnergyView,    "Peak Energy")
        self.analysisTabs.addTab(self.fitIntensityView, "Peak Intensity")
        self.analysisTabs.addTab(self.fitAucView,       "Peak AUC")
        self.analysisTabs.addTab(self.satDevView,       "(V−c)(1−θ)")
        self.analysisTabs.addTab(self.fitRatioView,     "Peak Fraction")
        self.analysisTabs.addTab(self.satMapView,       "θ Maps")
        self.analysisTabs.addTab(self.gateNormMapView,  "Gate Norm")
        self.analysisTabs.addTab(self.rbEFView,         "R_b vs E_F")
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
        self.baselineCheck.setChecked(True)
        cl.addWidget(self.baselineCheck)
        cl.addWidget(QtWidgets.QLabel("Range:"))
        self.bgLoSpin = QtWidgets.QDoubleSpinBox()
        self.bgLoSpin.setDecimals(1)
        self.bgLoSpin.setRange(200.0, 2000.0)
        self.bgLoSpin.setValue(600.0)
        self.bgLoSpin.setFixedWidth(75)
        self.bgLoSpin.setSuffix(" nm")
        cl.addWidget(self.bgLoSpin)
        cl.addWidget(QtWidgets.QLabel("to"))
        self.bgHiSpin = QtWidgets.QDoubleSpinBox()
        self.bgHiSpin.setDecimals(1)
        self.bgHiSpin.setRange(200.0, 2000.0)
        self.bgHiSpin.setValue(650.0)
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
        self.modeCombo.setFixedWidth(160)
        slider_grid.addWidget(self.modeCombo, 0, 1)

        slider_grid.addWidget(QtWidgets.QLabel("Main:"), 0, 2)
        self.mainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mainSlider.setTracking(True)
        slider_grid.addWidget(self.mainSlider, 0, 3)
        self.mainLbl = QtWidgets.QLabel("--")
        self.mainLbl.setMinimumWidth(200)
        slider_grid.addWidget(self.mainLbl, 0, 4)

        slider_grid.addWidget(QtWidgets.QLabel("Fixed:"), 1, 2)
        self.fixedSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fixedSlider.setTracking(True)
        slider_grid.addWidget(self.fixedSlider, 1, 3)
        self.fixedLbl = QtWidgets.QLabel("--")
        slider_grid.addWidget(self.fixedLbl, 1, 4)

        self.saveGifBtn = QtWidgets.QPushButton("Save GIF…")
        self.saveGifBtn.setFixedWidth(110)
        self.saveGifBtn.setToolTip(
            "Animate the linecut colormap over all values of the fixed axis\n"
            "and save as an animated GIF (requires Pillow)."
        )
        slider_grid.addWidget(self.saveGifBtn, 2, 0, 1, 2)

        right.addLayout(slider_grid)

        # Compact misc grid — 4 rows
        misc_grid = QtWidgets.QGridLayout()
        misc_grid.setHorizontalSpacing(6)
        misc_grid.setVerticalSpacing(3)

        # Row 0: Crop + Deriv max
        misc_grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 0, 0)
        self.cropT = QtWidgets.QSpinBox()
        self.cropB = QtWidgets.QSpinBox()
        self.cropL = QtWidgets.QSpinBox()
        self.cropR = QtWidgets.QSpinBox()
        for s in (self.cropT, self.cropB, self.cropL, self.cropR):
            s.setRange(0, 10000)
            s.setFixedWidth(58)
        self.cropT.setValue(DEFAULT_CROP[0])
        self.cropB.setValue(DEFAULT_CROP[1])
        self.cropL.setValue(DEFAULT_CROP[2])
        self.cropR.setValue(DEFAULT_CROP[3])
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
        misc_grid.addWidget(QtWidgets.QLabel("Contours:"), 0, 8)
        self.mapContourSpin = QtWidgets.QSpinBox()
        self.mapContourSpin.setRange(0, 64)
        self.mapContourSpin.setValue(0)
        self.mapContourSpin.setSpecialValueText("Off")
        self.mapContourSpin.setFixedWidth(55)
        self.mapContourSpin.setToolTip(
            "Number of filled contour bands for the linecut map.\n"
            "0 = smooth gradient (off).  Try 10–20 for a contourf look.")
        misc_grid.addWidget(self.mapContourSpin, 0, 9)
        misc_grid.addWidget(QtWidgets.QLabel("Lo %:"), 0, 10)
        self.contourLoPctSpin = QtWidgets.QDoubleSpinBox()
        self.contourLoPctSpin.setRange(0.0, 49.0)
        self.contourLoPctSpin.setDecimals(1)
        self.contourLoPctSpin.setSingleStep(1.0)
        self.contourLoPctSpin.setValue(1.0)
        self.contourLoPctSpin.setFixedWidth(58)
        self.contourLoPctSpin.setToolTip(
            "The map is normalised to [0, 1] (min→0, 99th-pct→1).\n"
            "Everything below Lo%/100 in that space → minimum colour.\n"
            "The remaining range [Lo%/100, 1] is divided into N contour levels.\n"
            "e.g. Lo % = 30 → dark background up to 30 % of max,\n"
            "then N equal colour steps from 30 % to 100 %.")
        misc_grid.addWidget(self.contourLoPctSpin, 0, 11)
        self.contourWinBtn = QtWidgets.QPushButton("Contour win.")
        self.contourWinBtn.setFixedWidth(90)
        self.contourWinBtn.setToolTip(
            "Open a separate matplotlib contourf window synced to these controls.")
        misc_grid.addWidget(self.contourWinBtn, 0, 12)
        self.gifBtn = QtWidgets.QPushButton("Save GIF")
        self.gifBtn.setFixedWidth(75)
        self.gifBtn.setToolTip(
            "Save an animated GIF sweeping through the map axis at 2 fps.\n"
            "Each frame shows the sweep value and the selected energy as text.")
        misc_grid.addWidget(self.gifBtn, 0, 13)
        self.fitWinBtn = QtWidgets.QPushButton("Fit win.")
        self.fitWinBtn.setFixedWidth(68)
        self.fitWinBtn.setToolTip(
            "Open a separate matplotlib window showing the reconstructed\n"
            "fitted-peak map (model, not raw data).\n"
            "Run 'Fit Peaks' first.")
        misc_grid.addWidget(self.fitWinBtn, 0, 14)

        # Row 1: SG smooth + Reverse λ
        misc_grid.addWidget(QtWidgets.QLabel("SG smooth:"), 1, 0)
        self.sgSweepCheck = QtWidgets.QCheckBox("Sweep")
        self.sgSweepWin = QtWidgets.QSpinBox()
        self.sgSweepWin.setRange(3, 501)
        self.sgSweepWin.setSingleStep(2)
        self.sgSweepWin.setValue(5)
        self.sgSweepWin.setFixedWidth(55)
        self.sgSweepOrd = QtWidgets.QSpinBox()
        self.sgSweepOrd.setRange(1, 10)
        self.sgSweepOrd.setValue(2)
        self.sgSweepOrd.setFixedWidth(45)
        self.sgWlCheck = QtWidgets.QCheckBox("Along λ")
        self.sgWlCheck.setChecked(True)
        self.sgWlWin = QtWidgets.QSpinBox()
        self.sgWlWin.setRange(3, 501)
        self.sgWlWin.setSingleStep(2)
        self.sgWlWin.setValue(11)
        self.sgWlWin.setFixedWidth(55)
        self.sgWlOrd = QtWidgets.QSpinBox()
        self.sgWlOrd.setRange(1, 10)
        self.sgWlOrd.setValue(2)
        self.sgWlOrd.setFixedWidth(45)
        misc_grid.addWidget(self.sgSweepCheck, 1, 1)
        misc_grid.addWidget(self.sgSweepWin,   1, 2)
        misc_grid.addWidget(self.sgSweepOrd,   1, 3)
        misc_grid.addWidget(self.sgWlCheck,    1, 4)
        misc_grid.addWidget(self.sgWlWin,      1, 5)
        misc_grid.addWidget(self.sgWlOrd,      1, 6)
        self.reverseWlCheck = QtWidgets.QCheckBox("Reverse λ")
        self.reverseWlCheck.setChecked(True)
        misc_grid.addWidget(self.reverseWlCheck, 1, 7)

        # Row 2: Peak fitting
        misc_grid.addWidget(QtWidgets.QLabel("RP (nm):"), 2, 0)
        self.peak1GuessEdit = QtWidgets.QLineEdit("760")
        self.peak1GuessEdit.setFixedWidth(52)
        misc_grid.addWidget(self.peak1GuessEdit, 2, 1)
        misc_grid.addWidget(QtWidgets.QLabel("AP (nm):"), 2, 2)
        self.peak2GuessEdit = QtWidgets.QLineEdit("770")
        self.peak2GuessEdit.setFixedWidth(52)
        misc_grid.addWidget(self.peak2GuessEdit, 2, 3)
        misc_grid.addWidget(QtWidgets.QLabel("Model:"), 2, 4)
        self.fitModelCombo = QtWidgets.QComboBox()
        self.fitModelCombo.addItems(["Lorentzian", "Gaussian"])
        self.fitModelCombo.setFixedWidth(100)
        misc_grid.addWidget(self.fitModelCombo, 2, 5)
        misc_grid.addWidget(QtWidgets.QLabel("Ref frame:"), 2, 6)
        self.refIndexSpin = QtWidgets.QSpinBox()
        self.refIndexSpin.setRange(0, 0)
        self.refIndexSpin.setFixedWidth(52)
        misc_grid.addWidget(self.refIndexSpin, 2, 7)
        misc_grid.addWidget(QtWidgets.QLabel("Max shift:"), 2, 8)
        self.maxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.maxShiftSpin.setDecimals(2)
        self.maxShiftSpin.setRange(0.2, 20.0)
        self.maxShiftSpin.setSingleStep(0.2)
        self.maxShiftSpin.setValue(2.0)
        self.maxShiftSpin.setFixedWidth(62)
        misc_grid.addWidget(self.maxShiftSpin, 2, 9)
        self.fitPeaksBtn = QtWidgets.QPushButton("Fit Peaks")
        self.fitPeaksBtn.setFixedWidth(82)
        misc_grid.addWidget(self.fitPeaksBtn, 2, 10)

        # Row 3: Peak ranges (shaded in linecut plot)
        misc_grid.addWidget(QtWidgets.QLabel("RP range:"), 3, 0)
        self.peak1LoSpin = QtWidgets.QDoubleSpinBox()
        self.peak1LoSpin.setRange(200.0, 2000.0)
        self.peak1LoSpin.setDecimals(1)
        self.peak1LoSpin.setValue(755.0)
        self.peak1LoSpin.setSuffix(" nm")
        self.peak1LoSpin.setFixedWidth(72)
        misc_grid.addWidget(self.peak1LoSpin, 3, 1)
        misc_grid.addWidget(QtWidgets.QLabel("–"), 3, 2)
        self.peak1HiSpin = QtWidgets.QDoubleSpinBox()
        self.peak1HiSpin.setRange(200.0, 2000.0)
        self.peak1HiSpin.setDecimals(1)
        self.peak1HiSpin.setValue(765.0)
        self.peak1HiSpin.setSuffix(" nm")
        self.peak1HiSpin.setFixedWidth(72)
        misc_grid.addWidget(self.peak1HiSpin, 3, 3)
        misc_grid.addWidget(QtWidgets.QLabel("AP range:"), 3, 4)
        self.peak2LoSpin = QtWidgets.QDoubleSpinBox()
        self.peak2LoSpin.setRange(200.0, 2000.0)
        self.peak2LoSpin.setDecimals(1)
        self.peak2LoSpin.setValue(765.0)
        self.peak2LoSpin.setSuffix(" nm")
        self.peak2LoSpin.setFixedWidth(72)
        misc_grid.addWidget(self.peak2LoSpin, 3, 5)
        misc_grid.addWidget(QtWidgets.QLabel("–"), 3, 6)
        self.peak2HiSpin = QtWidgets.QDoubleSpinBox()
        self.peak2HiSpin.setRange(200.0, 2000.0)
        self.peak2HiSpin.setDecimals(1)
        self.peak2HiSpin.setValue(780.0)
        self.peak2HiSpin.setSuffix(" nm")
        self.peak2HiSpin.setFixedWidth(72)
        misc_grid.addWidget(self.peak2HiSpin, 3, 7)
        misc_grid.addWidget(QtWidgets.QLabel("W ratio:"), 3, 8)
        self.maxWidthRatioSpin = QtWidgets.QDoubleSpinBox()
        self.maxWidthRatioSpin.setRange(1.1, 10.0)
        self.maxWidthRatioSpin.setDecimals(1)
        self.maxWidthRatioSpin.setSingleStep(0.1)
        self.maxWidthRatioSpin.setValue(2.0)
        self.maxWidthRatioSpin.setFixedWidth(55)
        self.maxWidthRatioSpin.setToolTip(
            "Max allowed width ratio between adjacent sweep frames.\n"
            "Prevents one peak from suddenly going broad while the other stays narrow.\n"
            "Increase if fit fails; decrease for stronger smoothness enforcement.")
        misc_grid.addWidget(self.maxWidthRatioSpin, 3, 9)

        # Row 4: Dip fitting
        misc_grid.addWidget(QtWidgets.QLabel("Dip guesses (nm):"), 4, 0)
        self.dipGuessesEdit = QtWidgets.QLineEdit("")
        self.dipGuessesEdit.setPlaceholderText("e.g. 855, 860")
        self.dipGuessesEdit.setFixedWidth(140)
        misc_grid.addWidget(self.dipGuessesEdit, 4, 1, 1, 2)   # colspan 2
        misc_grid.addWidget(QtWidgets.QLabel("Window:"), 4, 3)
        self.dipWindowSpin = QtWidgets.QDoubleSpinBox()
        self.dipWindowSpin.setDecimals(1)
        self.dipWindowSpin.setRange(0.5, 100.0)
        self.dipWindowSpin.setSingleStep(0.5)
        self.dipWindowSpin.setValue(8.0)
        self.dipWindowSpin.setFixedWidth(60)
        misc_grid.addWidget(self.dipWindowSpin, 4, 4)
        misc_grid.addWidget(QtWidgets.QLabel("Max shift:"), 4, 5)
        self.dipMaxShiftSpin = QtWidgets.QDoubleSpinBox()
        self.dipMaxShiftSpin.setDecimals(2)
        self.dipMaxShiftSpin.setRange(0.1, 20.0)
        self.dipMaxShiftSpin.setSingleStep(0.1)
        self.dipMaxShiftSpin.setValue(2.0)
        self.dipMaxShiftSpin.setFixedWidth(60)
        misc_grid.addWidget(self.dipMaxShiftSpin, 4, 6)
        misc_grid.addWidget(QtWidgets.QLabel("Smooth σ:"), 4, 7)
        self.dipSmoothSpin = QtWidgets.QDoubleSpinBox()
        self.dipSmoothSpin.setDecimals(2)
        self.dipSmoothSpin.setRange(0.0, 50.0)
        self.dipSmoothSpin.setSingleStep(0.1)
        self.dipSmoothSpin.setValue(0.5)
        self.dipSmoothSpin.setFixedWidth(58)
        misc_grid.addWidget(self.dipSmoothSpin, 4, 8)
        self.fitDipsBtn = QtWidgets.QPushButton("Fit Dips")
        self.fitDipsBtn.setFixedWidth(82)
        misc_grid.addWidget(self.fitDipsBtn, 4, 9)

        right.addLayout(misc_grid)
        right.addStretch()
        outer.addLayout(right, 3)

        # ── Wire signals ──────────────────────────────────────────────
        self.polCheck.toggled.connect(self._on_pol_toggled)
        self.baselineCheck.toggled.connect(
            lambda _: self._clear_caches_and_refresh())
        self.bgLoSpin.valueChanged.connect(
            lambda _: self._clear_caches_and_refresh())
        self.bgHiSpin.valueChanged.connect(
            lambda _: self._clear_caches_and_refresh())
        self.subCheck.toggled.connect(lambda _: self._refresh_all())
        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)
        self.mainSlider.valueChanged.connect(self._on_main_changed)
        self.fixedSlider.valueChanged.connect(self._on_fixed_changed)
        self.cropApplyBtn.clicked.connect(self._on_apply_crop)
        self.derivMaxSpin.valueChanged.connect(lambda _: self._update_maps())
        self.mapContourSpin.valueChanged.connect(lambda _: self._update_maps())
        self.contourLoPctSpin.valueChanged.connect(
            lambda _: self._update_maps())
        self.contourWinBtn.clicked.connect(self._open_contour_window)
        self.gifBtn.clicked.connect(self._create_contour_gif)
        self.fitWinBtn.clicked.connect(self._open_fit_window)
        for w in (self.sgSweepCheck, self.sgWlCheck, self.sgSweepWin, self.sgSweepOrd, self.sgWlWin, self.sgWlOrd):
            sig = w.toggled if isinstance(
                w, QtWidgets.QCheckBox) else w.valueChanged
            sig.connect(lambda _: self._update_maps())
        self.reverseWlCheck.toggled.connect(self._on_reverse_wl)
        self.fitPeaksBtn.clicked.connect(self._on_fit_peaks_clicked)
        self.peak1GuessEdit.editingFinished.connect(self._refresh_all)
        self.peak2GuessEdit.editingFinished.connect(self._refresh_all)
        self.fitModelCombo.currentTextChanged.connect(
            self._on_fit_model_changed)
        for _w in (self.peak1LoSpin, self.peak1HiSpin, self.peak2LoSpin, self.peak2HiSpin):
            _w.valueChanged.connect(self._on_peak_ranges_changed)
        self.fitDipsBtn.clicked.connect(self._on_fit_dips_clicked)
        self.dipGuessesEdit.editingFinished.connect(self._update_maps)
        self.dipSmoothSpin.valueChanged.connect(
            lambda _: self._on_dip_smooth_changed())
        self.lockAxesCheck.toggled.connect(self._on_lock_axes_toggled)
        self.loadAngleCalibBtn.clicked.connect(self._on_load_angle_calib)
        self.saveGifBtn.clicked.connect(self._on_save_gif)
        # liveView signals connected later in _build_ui() after liveView is created

        return box

    def _make_nonpol_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)

        g.addWidget(QtWidgets.QLabel("Data folder:"), 0, 0)
        self.npDataLbl = QtWidgets.QLabel("(none)")
        self.npDataLbl.setStyleSheet("font-style:italic; color:#555;")
        g.addWidget(self.npDataLbl, 0, 1)
        btn = QtWidgets.QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(lambda: self._browse_data(""))
        g.addWidget(btn, 0, 2)

        g.addWidget(QtWidgets.QLabel("Substrate:"), 1, 0)
        self.npSubLbl = QtWidgets.QLabel("(none)")
        self.npSubLbl.setStyleSheet("font-style:italic; color:#888;")
        g.addWidget(self.npSubLbl, 1, 1)
        bf = QtWidgets.QPushButton("File…")
        bf.setFixedWidth(70)
        bfo = QtWidgets.QPushButton("Folder…")
        bfo.setFixedWidth(80)
        bf.clicked.connect(lambda: self._browse_sub_file(""))
        bfo.clicked.connect(lambda: self._browse_sub_folder(""))
        g.addWidget(bf,  1, 2)
        g.addWidget(bfo, 1, 3)
        g.setColumnStretch(1, 1)
        return page

    def _make_pol_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)

        def _row(label_text: str, key: str, row: int, lbl_attr: str, sub_attr: str) -> None:
            g.addWidget(QtWidgets.QLabel(label_text), row, 0)
            lbl = QtWidgets.QLabel("(none)")
            lbl.setStyleSheet("font-style:italic; color:#555;")
            setattr(self, lbl_attr, lbl)
            g.addWidget(lbl, row, 1)
            bd = QtWidgets.QPushButton("Data…")
            bd.setFixedWidth(70)
            bd.clicked.connect(lambda checked, k=key: self._browse_data(k))
            g.addWidget(bd, row, 2)

            sub_lbl = QtWidgets.QLabel("(none)")
            sub_lbl.setStyleSheet("font-style:italic; color:#888;")
            setattr(self, sub_attr, sub_lbl)
            g.addWidget(QtWidgets.QLabel("Substrate:"), row + 1, 0)
            g.addWidget(sub_lbl, row + 1, 1)
            bf = QtWidgets.QPushButton("File…")
            bf.setFixedWidth(70)
            bfo = QtWidgets.QPushButton("Folder…")
            bfo.setFixedWidth(80)
            bf.clicked.connect(lambda checked, k=key: self._browse_sub_file(k))
            bfo.clicked.connect(
                lambda checked, k=key: self._browse_sub_folder(k))
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
        self.polLHBtn = QtWidgets.QRadioButton("Left Hand")
        self.polLHBtn.setChecked(True)
        self.polRHBtn = QtWidgets.QRadioButton("Right Hand")
        self.polDiffBtn = QtWidgets.QRadioButton("LH − RH")
        for i, rb in enumerate([self.polLHBtn, self.polRHBtn, self.polDiffBtn]):
            self.polDispGroup.addButton(rb, i)
            disp_row.addWidget(rb)
        disp_row.addStretch()
        g.addLayout(disp_row, 6, 0, 1, 5)

        self.polDispGroup.buttonClicked.connect(
            lambda _: (self._clear_all_fixed_overlays(), self._refresh_all()))
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
        try:
            import pandas as pd
            p = Path(path)
            calib_angles = calib_powers_w = None
            for h in (0, 1, None):
                try:
                    df = pd.read_excel(p, header=h) if p.suffix.lower() in (
                        ".xlsx", ".xls") else pd.read_csv(p, header=h)
                    cols = {str(c).lower().strip(): c for c in df.columns}
                    angle_col = next((cols[k]
                                     for k in cols if "angle" in k), None)
                    power_col = next((cols[k]
                                     for k in cols if "power" in k), None)
                    if angle_col is None or power_col is None:
                        continue
                    angles = pd.to_numeric(df[angle_col], errors="coerce")
                    powers = pd.to_numeric(df[power_col], errors="coerce")
                    valid = angles.notna() & powers.notna()
                    if valid.sum() < 2:
                        continue
                    angles = np.asarray(angles[valid], dtype=float)
                    powers = np.asarray(powers[valid], dtype=float)
                    # detect unit from column name suffix
                    pc = str(power_col).lower()
                    if "/mw" in pc or "mw" in pc:
                        scale = 1e-3
                    elif "/nw" in pc or "nw" in pc:
                        scale = 1e-9
                    elif "/w" in pc and "mw" not in pc and "nw" not in pc and "uw" not in pc:
                        scale = 1.0
                    else:
                        scale = 1e-6  # default: µW
                    order = np.argsort(angles)
                    calib_angles = angles[order]
                    calib_powers_w = powers[order] * scale
                    break
                except Exception:
                    continue
            if calib_angles is None:
                raise ValueError(
                    "Could not find 'angle' and 'power' columns in the file.")
            for ch in self._ch.values():
                ch._angle_calib_angles = calib_angles
                ch._angle_calib_powers_w = calib_powers_w
            n = len(calib_angles)
            self.angleCalibStatusLbl.setText(f"{n} pts: {p.name}")
            self._refresh_all()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))

    def _browse_data(self, key: str) -> None:
        ch = self._ch[key]
        start = str(ch.data_dir) if ch.data_dir else ""
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data folder", start)
        if not chosen:
            return
        folder = Path(chosen)
        frames = _load_frames(folder)
        if not frames:
            QtWidgets.QMessageBox.warning(
                self, "No data", f"No .asc files found in:\n{folder}")
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
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select substrate folder", start)
        if not chosen:
            return
        folder = Path(chosen)
        _load_sub_folder(ch, folder)
        if not ch.sub_grid:
            QtWidgets.QMessageBox.warning(
                self, "No data", f"No .asc files found in:\n{folder}")
            return
        self._update_sub_label(key, folder=folder)
        self._refresh_all()

    def _update_data_label(self, key: str) -> None:
        ch = self._ch[key]
        txt = str(ch.data_dir.name) if ch.data_dir else "(none)"
        if key == "":
            self.npDataLbl.setText(txt)
        elif key == "LH":
            self.lhDataLbl.setText(txt)
        elif key == "RH":
            self.rhDataLbl.setText(txt)

    def _update_sub_label(self, key: str, folder: Optional[Path] = None) -> None:
        ch = self._ch[key]
        if ch.sub_single is not None:
            txt = f"File: {ch.sub_single.path.name}"
        elif ch.sub_grid:
            n = len(ch.sub_grid)
            name = folder.name if folder else "?"
            txt = f"Folder: {name} ({n} files)"
        else:
            txt = "(none)"
        if key == "":
            self.npSubLbl.setText(txt)
        elif key == "LH":
            self.lhSubLbl.setText(txt)
        elif key == "RH":
            self.rhSubLbl.setText(txt)

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
        for fw in (self.fitWidthView, self.fitEnergyView):
            _toggle(fw._plot.getViewBox())
            if getattr(fw, "_plot_diff", None) is not None:
                _toggle(fw._plot_diff.getViewBox())
        for vb in self.fitIntensityView.all_viewboxes():
            _toggle(vb)
        for vb in self.fitAucView.all_viewboxes():
            _toggle(vb)
        for vb in self.satDevView.all_viewboxes():
            _toggle(vb)
        for vb in self.fitRatioView.all_viewboxes():
            _toggle(vb)
        _toggle(self.dipPosPlot.getViewBox())

        if not checked:
            self._refresh_all()

    # ── Polarization toggle ───────────────────────────────────────────

    def _on_pol_toggled(self, checked: bool) -> None:
        self.srcStack.setCurrentIndex(1 if checked else 0)
        self._clear_all_fixed_overlays()   # channel data changed
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
        p1, p2 = 727.0, 740.0
        try:
            v = float(self.peak1GuessEdit.text().strip())
            if np.isfinite(v):
                p1 = v
        except Exception:
            pass
        try:
            v = float(self.peak2GuessEdit.text().strip())
            if np.isfinite(v):
                p2 = v
        except Exception:
            pass
        return p1, p2

    def _read_peak_ranges(self) -> Tuple[float, float, float, float]:
        return (float(self.peak1LoSpin.value()), float(self.peak1HiSpin.value()),
                float(self.peak2LoSpin.value()), float(self.peak2HiSpin.value()))

    def _update_range_shading(self) -> None:
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        self.liveView.set_peak_ranges(lo1, hi1, lo2, hi2)

    def _on_peak_ranges_changed(self) -> None:
        self._fit_result = None
        self._update_range_shading()
        self._update_live_linecut_fit_overlay()

    def _on_fit_model_changed(self) -> None:
        self._fit_result = None
        self._update_live_linecut_fit_overlay()

    def _on_fit_peaks_clicked(self) -> None:
        self._active_overlay = "peak"
        self.liveView.clear_dip_overlay()
        if self._last_x is None or self._last_y_wl is None or self._last_map is None:
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView, self.fitAucView, self.satDevView, self.fitRatioView):
                w.clear_data("No map data", x_label=self._last_x_label or "X")
            return
        p1, p2 = self._read_peak_guesses()
        ref_idx = int(self.refIndexSpin.value())
        max_shift = float(self.maxShiftSpin.value())
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        fit_model = self.fitModelCombo.currentText().lower()
        max_width_ratio = float(self.maxWidthRatioSpin.value())
        fit = fit_two_peak_map_sequential(
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map,  dtype=float),
            ref_idx=ref_idx, peak1_guess_nm=p1, peak2_guess_nm=p2, max_shift_nm=max_shift,
            max_width_ratio=max_width_ratio,
            peak1_lo=lo1, peak1_hi=hi1, peak2_lo=lo2, peak2_hi=hi2,
            fit_model=fit_model,
        )
        self._fit_result = fit
        self._fit_map_version = int(self._map_version)
        self._update_fit_trend_tabs(
            np.asarray(self._last_x,    dtype=float),
            str(self._last_x_label or "X"),
            np.asarray(self._last_y_wl, dtype=float),
            np.asarray(self._last_map,  dtype=float),
        )
        self._update_mpl_fit()

    def _compute_all_fixed_fits(self):
        """Two-peak fit for every fixed-axis value → (x_main, [fit dicts]) or None.

        In gate mode the fixed axis is power; in power/angle mode it is gate.
        Cached by (channel, mode, fixed count, fit settings) — toggle the
        checkbox off/on to refresh after changing smoothing or baseline.
        """
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        if not ch.has_data or self._last_x is None:
            return None
        n_fixed = int(ch.powers.size if self._current_mode == self.MODE_GATE
                      else ch.gates.size)
        if n_fixed == 0:
            return None

        p1_g, p2_g = self._read_peak_guesses()
        ref_idx = int(self.refIndexSpin.value())
        max_shift = float(self.maxShiftSpin.value())
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        fit_model = self.fitModelCombo.currentText().lower()
        max_width_ratio = float(self.maxWidthRatioSpin.value())
        key = (ch_key, self._current_mode, n_fixed, p1_g, p2_g, ref_idx,
               max_shift, lo1, hi1, lo2, hi2, fit_model, max_width_ratio)
        cache = getattr(self, "_all_fixed_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]

        saved = self._fixed_idx
        fits: list = []
        canceled = False
        prog = QtWidgets.QProgressDialog(
            "Fitting all fixed-axis values…", "Cancel", 0, n_fixed, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()
        try:
            for k in range(n_fixed):
                prog.setValue(k)
                QtWidgets.QApplication.processEvents()
                if prog.wasCanceled():
                    canceled = True
                    break
                self._fixed_idx = k
                _, wl_nm, map_data = self._build_map(ch)
                m_smooth = self._sg_apply(np.asarray(map_data, float))
                fits.append(fit_two_peak_map_sequential(
                    np.asarray(wl_nm, float), m_smooth,
                    ref_idx=ref_idx,
                    peak1_guess_nm=p1_g, peak2_guess_nm=p2_g,
                    max_shift_nm=max_shift, max_width_ratio=max_width_ratio,
                    peak1_lo=lo1, peak1_hi=hi1, peak2_lo=lo2, peak2_hi=hi2,
                    fit_model=fit_model,
                ))
        finally:
            self._fixed_idx = saved
            prog.close()
        if canceled:
            return None
        x = np.asarray(self._last_x, dtype=float)
        self._all_fixed_cache = (key, x, fits)
        return x, fits

    def _on_show_all_fixed(self, widget, kind: str, on: bool) -> None:
        """Checkbox handler: overlay fit trends for all fixed-axis values."""
        if not on:
            widget.clear_overlays()
            return
        res = self._compute_all_fixed_fits()
        if res is None:
            widget.clear_overlays()
            widget.showAllChk.blockSignals(True)
            widget.showAllChk.setChecked(False)
            widget.showAllChk.blockSignals(False)
            return
        x, fits = res
        n = int(x.size)

        def _pad(a):
            a = np.asarray(a, dtype=float).ravel()
            return np.pad(a, (0, max(0, n - a.size)),
                          constant_values=np.nan)[:n]

        fit_model = self.fitModelCombo.currentText().lower()
        auc_f = np.sqrt(2.0 * np.pi) if fit_model == "gaussian" else np.pi
        series = []
        for fit in fits:
            a1 = _pad(fit.get("amp_1", []))
            a2 = _pad(fit.get("amp_2", []))
            if kind == "energy":
                c1 = _pad(fit.get("center_1_nm", []))
                c2 = _pad(fit.get("center_2_nm", []))
                with np.errstate(divide="ignore", invalid="ignore"):
                    e1 = EV_NM / c1
                    e2 = EV_NM / c2
                e1[~np.isfinite(e1)] = np.nan
                e2[~np.isfinite(e2)] = np.nan
                series.append((e1, e2))
            elif kind == "intensity":
                tot = np.where(
                    np.isfinite(a1) | np.isfinite(a2),
                    np.nansum(np.column_stack([a1, a2]), axis=1).astype(float),
                    np.nan)
                series.append((a1, a2, tot))
            else:   # auc
                gm1 = _pad(fit.get("gamma_1_nm", []))
                gm2 = _pad(fit.get("gamma_2_nm", []))
                auc1 = a1 * gm1 * auc_f
                auc2 = a2 * gm2 * auc_f
                auc1[~(np.isfinite(a1) & np.isfinite(gm1) & (gm1 > 0))] = np.nan
                auc2[~(np.isfinite(a2) & np.isfinite(gm2) & (gm2 > 0))] = np.nan
                tot = np.where(
                    np.isfinite(auc1) | np.isfinite(auc2),
                    np.nansum(np.column_stack([auc1, auc2]), axis=1).astype(float),
                    np.nan)
                series.append((auc1, auc2, tot))
        widget.set_overlays(x, series)

    def _clear_all_fixed_overlays(self) -> None:
        """Clear overlays + uncheck boxes (mode/channel change invalidates x-axis)."""
        for w in (self.fitEnergyView, self.fitIntensityView, self.fitAucView):
            chk = getattr(w, "showAllChk", None)
            if chk is not None and chk.isChecked():
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)
            w.clear_overlays()

    def _on_fit_all_sat_map(self) -> None:
        if self._current_mode != self.MODE_POWER:
            QtWidgets.QMessageBox.warning(self, "Power mode only",
                                          "Saturation maps are only available in power dependence mode.")
            return
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        if not ch.has_data or ch.gates.size == 0 or ch.powers.size == 0:
            self.satMapView.clear("No data")
            self.gateNormMapView.clear("No data")
            return

        # Power axis (calibrated µW) — same for every gate
        if ch.angle_mode and ch._angle_calib_angles is not None:
            powers_uW = np.interp(ch.powers, ch._angle_calib_angles,
                                  ch._angle_calib_powers_w) * 1e6
        else:
            powers_uW = ch.powers * 1e6

        # Fit settings from existing UI controls
        p1_g, p2_g = self._read_peak_guesses()
        ref_idx = int(self.refIndexSpin.value())
        max_shift = float(self.maxShiftSpin.value())
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        fit_model = self.fitModelCombo.currentText().lower()
        max_width_ratio = float(self.maxWidthRatioSpin.value())
        _auc_f = np.sqrt(2.0 * np.pi) if fit_model == "gaussian" else np.pi
        use_auc = self.satMapView.metricCombo.currentText() == "AUC"

        n_gates = int(ch.gates.size)
        n_powers = int(ch.powers.size)
        theta1 = np.full((n_powers, n_gates), np.nan)
        theta2 = np.full((n_powers, n_gates), np.nan)
        theta_t = np.full((n_powers, n_gates), np.nan)
        # raw fitted intensities (amplitude, not AUC) for the Gate Norm tab
        int1 = np.full((n_powers, n_gates), np.nan)
        int2 = np.full((n_powers, n_gates), np.nan)
        int_t = np.full((n_powers, n_gates), np.nan)

        saved_fixed_idx = self._fixed_idx
        prog = QtWidgets.QProgressDialog(
            "Fitting saturation across gates…", "Cancel", 0, n_gates, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()
        try:
            def _hill(x, Vmax, n, K, c):
                return Vmax * x**n / (K**n + x**n) + c

            for i in range(n_gates):
                prog.setValue(i)
                QtWidgets.QApplication.processEvents()
                if prog.wasCanceled():
                    break

                self._fixed_idx = i
                _, wl_nm, map_data = self._build_map(ch)
                m_smooth = self._sg_apply(np.asarray(map_data, float))

                fit = fit_two_peak_map_sequential(
                    np.asarray(wl_nm, float), m_smooth,
                    ref_idx=ref_idx,
                    peak1_guess_nm=p1_g, peak2_guess_nm=p2_g,
                    max_shift_nm=max_shift, max_width_ratio=max_width_ratio,
                    peak1_lo=lo1, peak1_hi=hi1, peak2_lo=lo2, peak2_hi=hi2,
                    fit_model=fit_model,
                )

                def _pad(arr):
                    arr = np.asarray(arr, float).ravel()
                    return np.pad(arr, (0, max(0, n_powers - arr.size)),
                                  constant_values=np.nan)[:n_powers]

                a1 = _pad(fit.get("amp_1", []))
                a2 = _pad(fit.get("amp_2", []))
                int1[:, i] = a1
                int2[:, i] = a2
                int_t[:, i] = np.where(
                    np.isfinite(a1) | np.isfinite(a2),
                    np.nansum(np.column_stack([a1, a2]), axis=1).astype(float),
                    np.nan,
                )
                if use_auc:
                    gm1 = _pad(fit.get("gamma_1_nm", []))
                    gm2 = _pad(fit.get("gamma_2_nm", []))
                    y1 = a1 * gm1 * _auc_f
                    y2 = a2 * gm2 * _auc_f
                    y1[~(np.isfinite(a1) & np.isfinite(gm1) & (gm1 > 0))] = np.nan
                    y2[~(np.isfinite(a2) & np.isfinite(gm2) & (gm2 > 0))] = np.nan
                else:
                    y1 = a1.copy()
                    y2 = a2.copy()

                yt = np.where(
                    np.isfinite(y1) | np.isfinite(y2),
                    np.nansum(np.column_stack([y1, y2]), axis=1).astype(float),
                    np.nan,
                )

                from scipy.optimize import curve_fit as _cf
                for yv, theta_arr in ((y1, theta1), (y2, theta2), (yt, theta_t)):
                    m_ok = np.isfinite(powers_uW) & np.isfinite(
                        yv) & (powers_uW > 0) & (yv > 0)
                    if m_ok.sum() < 4:
                        continue
                    xm, ym = powers_uW[m_ok], yv[m_ok]
                    c0 = float(np.nanmin(ym))
                    Vm0 = max(float(np.nanmax(ym)) - c0, 1e-10)
                    try:
                        popt, _ = _cf(
                            _hill, xm, ym,
                            p0=[Vm0, 0.8, float(np.median(xm)), c0],
                            bounds=([0, 0.3, xm.min() * 0.01, 0.0],
                                    [Vm0 * 20, 1.0, xm.max() * 10, c0 * 2 + 1.0]),
                            maxfev=20000,
                        )
                        Vmax, _, _, c_f = popt
                        theta_arr[:, i] = (yv - c_f) / Vmax
                    except Exception:
                        pass
        finally:
            self._fixed_idx = saved_fixed_idx
            prog.close()

        self.satMapView.set_data(ch.gates, powers_uW, theta1, theta2, theta_t,
                                 status=f"Fitted {n_gates} gates")
        self.gateNormMapView.set_data(ch.gates, powers_uW, int1, int2, int_t,
                                      status=f"Fitted {n_gates} gates")

    def _on_fit_rb_vs_ef(self) -> None:
        """Hill-fit peak intensity vs power at every gate; derive R_b(RP), R_b(AP)
        from each fit's saturation power K and plot both vs E_F."""
        if self._current_mode != self.MODE_POWER:
            QtWidgets.QMessageBox.warning(self, "Power mode only",
                                          "R_b vs E_F requires power dependence mode.")
            return
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        if not ch.has_data or ch.gates.size == 0 or ch.powers.size == 0:
            self.rbEFView.clear("No data")
            return

        if ch.angle_mode and ch._angle_calib_angles is not None:
            powers_uW = np.interp(ch.powers, ch._angle_calib_angles,
                                  ch._angle_calib_powers_w) * 1e6
        else:
            powers_uW = ch.powers * 1e6

        p1_g, p2_g = self._read_peak_guesses()
        ref_idx = int(self.refIndexSpin.value())
        max_shift = float(self.maxShiftSpin.value())
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        fit_model = self.fitModelCombo.currentText().lower()
        max_width_ratio = float(self.maxWidthRatioSpin.value())

        n_gates = int(ch.gates.size)
        ef_mev = _cal_ef_mev(ch.gates)
        rb1 = np.full(n_gates, np.nan)
        rb2 = np.full(n_gates, np.nan)

        saved_fixed_idx = self._fixed_idx
        prog = QtWidgets.QProgressDialog(
            "Fitting R_b across gates…", "Cancel", 0, n_gates, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()
        try:
            def _hill(x, Vmax, n, K, c):
                return Vmax * x**n / (K**n + x**n) + c

            from scipy.optimize import curve_fit as _cf
            for i in range(n_gates):
                prog.setValue(i)
                QtWidgets.QApplication.processEvents()
                if prog.wasCanceled():
                    break

                self._fixed_idx = i
                _, wl_nm, map_data = self._build_map(ch)
                m_smooth = self._sg_apply(np.asarray(map_data, float))

                fit = fit_two_peak_map_sequential(
                    np.asarray(wl_nm, float), m_smooth,
                    ref_idx=ref_idx,
                    peak1_guess_nm=p1_g, peak2_guess_nm=p2_g,
                    max_shift_nm=max_shift, max_width_ratio=max_width_ratio,
                    peak1_lo=lo1, peak1_hi=hi1, peak2_lo=lo2, peak2_hi=hi2,
                    fit_model=fit_model,
                )

                def _pad(arr):
                    arr = np.asarray(arr, float).ravel()
                    return np.pad(arr, (0, max(0, powers_uW.size - arr.size)),
                                  constant_values=np.nan)[:powers_uW.size]

                a1 = _pad(fit.get("amp_1", []))
                a2 = _pad(fit.get("amp_2", []))
                vg = float(ch.gates[i])

                for yv, rb_arr in ((a1, rb1), (a2, rb2)):
                    m_ok = np.isfinite(powers_uW) & np.isfinite(
                        yv) & (powers_uW > 0) & (yv > 0)
                    if m_ok.sum() < 4:
                        continue
                    xm, ym = powers_uW[m_ok], yv[m_ok]
                    c0 = float(np.nanmin(ym))
                    Vm0 = max(float(np.nanmax(ym)) - c0, 1e-10)
                    try:
                        popt, _ = _cf(
                            _hill, xm, ym,
                            p0=[Vm0, 0.8, float(np.median(xm)), c0],
                            bounds=([0, 0.3, xm.min() * 0.01, 0.0],
                                    [Vm0 * 20, 1.0, xm.max() * 10, c0 * 2 + 1.0]),
                            maxfev=20000,
                        )
                        K_f = float(popt[2])
                        rb_arr[i] = _cal_rb_nm(vg, K_f)
                    except Exception:
                        pass
        finally:
            self._fixed_idx = saved_fixed_idx
            prog.close()

        self.rbEFView.set_data(ef_mev, rb1, rb2, status=f"Fitted {n_gates} gates")

    def _update_fit_trend_tabs(self, x, x_label, y_wl, map_data) -> None:
        if map_data.ndim != 2 or y_wl.size != map_data.shape[0]:
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView, self.fitAucView, self.satDevView, self.fitRatioView):
                w.clear_data("No data", x_label=x_label)
            return
        if self._fit_result is None:
            msg = "Click 'Fit Peaks' to run / refresh"
            for w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView, self.fitAucView, self.satDevView, self.fitRatioView):
                w.clear_data(msg, x_label=x_label)
            return

        fit = self._fit_result
        c1 = np.asarray(fit.get("center_1_nm", []), dtype=float)
        w1 = np.asarray(fit.get("fwhm_1_nm",   []), dtype=float)
        a1 = np.asarray(fit.get("amp_1",        []), dtype=float)
        c2 = np.asarray(fit.get("center_2_nm", []), dtype=float)
        w2 = np.asarray(fit.get("fwhm_2_nm",   []), dtype=float)
        a2 = np.asarray(fit.get("amp_2",        []), dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = EV_NM / c1
            e2 = EV_NM / c2
        e1[~np.isfinite(e1)] = np.nan
        e2[~np.isfinite(e2)] = np.nan

        self.fitWidthView.set_data(x, fwhm_nm_to_mev(c1, w1), fwhm_nm_to_mev(c2, w2),
                                   x_label=x_label, label_1="RP", label_2="AP")
        self.fitEnergyView.set_data(x, e1, e2,
                                    x_label=x_label, label_1="RP", label_2="AP")
        _sat = (self._current_mode == self.MODE_POWER)
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        vg_fixed = float(ch.gates[self._fixed_idx]) if (_sat and ch.gates.size) else None
        self.fitIntensityView.set_data(
            x, a1, a2, x_label=x_label, do_saturation_fit=_sat, vg=vg_fixed)

        gm1 = np.asarray(fit.get("gamma_1_nm", []), dtype=float)
        gm2 = np.asarray(fit.get("gamma_2_nm", []), dtype=float)
        n = int(self._last_x.size) if self._last_x is not None else 0
        def _pad_g(arr): return np.pad(
            arr.ravel(), (0, max(0, n - arr.size)), constant_values=np.nan)[:n]
        gm1 = _pad_g(gm1)
        gm2 = _pad_g(gm2)
        fit_model = self.fitModelCombo.currentText().lower()
        _auc_factor = np.sqrt(
            2.0 * np.pi) if fit_model == "gaussian" else np.pi
        auc1 = a1 * gm1 * _auc_factor
        auc2 = a2 * gm2 * _auc_factor
        auc1[~(np.isfinite(a1) & np.isfinite(gm1) & (gm1 > 0))] = np.nan
        auc2[~(np.isfinite(a2) & np.isfinite(gm2) & (gm2 > 0))] = np.nan
        self.fitAucView.set_data(
            x, auc1, auc2, x_label=x_label, do_saturation_fit=_sat)
        self.satDevView.set_data(
            x, a1, a2, auc1, auc2, x_label=x_label, do_sat_dev=_sat)
        self.fitRatioView.set_data(x, a1, a2, auc1, auc2, x_label=x_label)

    def _update_live_linecut_fit_overlay(self) -> None:
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        fr = self._get_frame(ch)
        if fr is None:
            self.liveView.clear_linecut_fit_overlay()
            self.liveView.clear_smooth_linecut()
            return
        img, wl = self._get_corrected(ch, fr)
        h = img.shape[0]
        row_d = self.liveView.linecut_row()
        row_d = h // 2 if row_d is None else max(0, min(int(row_d), h - 1))
        row_raw = h - 1 - row_d
        width = int(self.liveView.linecut_width())
        lc = _linecut_h(img, row_raw, width)
        if lc is None:
            self.liveView.clear_linecut_fit_overlay()
            self.liveView.clear_smooth_linecut()
            return
        wl = np.asarray(wl, dtype=float).ravel()
        lc = np.asarray(lc, dtype=float).ravel()
        if wl.size != lc.size or wl.size < 8:
            self.liveView.clear_linecut_fit_overlay()
            self.liveView.clear_smooth_linecut()
            return

        # SG smooth along wavelength for display
        lc_smooth = None
        if self.sgWlCheck.isChecked() and _savgol is not None:
            win = int(self.sgWlWin.value())
            win = win + 1 if win % 2 == 0 else win
            ord_ = int(self.sgWlOrd.value())
            if win >= 3 and 1 <= ord_ < win and lc.size >= win:
                try:
                    lc_smooth = _savgol(lc, win, ord_, mode="mirror")
                except Exception:
                    lc_smooth = None
        if lc_smooth is not None:
            self.liveView.set_smooth_linecut(wl, lc_smooth)
        else:
            self.liveView.clear_smooth_linecut()

        p1g, p2g = self._read_peak_guesses()
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()
        fit_model = self.fitModelCombo.currentText().lower()
        fit = fit_two_peak_linecut(wl, lc, peak1_guess_nm=p1g, peak2_guess_nm=p2g,
                                   peak1_lo=lo1, peak1_hi=hi1, peak2_lo=lo2, peak2_hi=hi2,
                                   fit_model=fit_model)
        p1 = fit.get("peak_1")
        p2 = fit.get("peak_2")
        if p1 is None and p2 is None:
            self.liveView.clear_linecut_fit_overlay()
            return
        bl = np.full_like(wl, float(fit.get("offset", 0.0)))
        mfunc, _ = _peak_model(fit_model)
        g1 = np.zeros_like(wl)
        g2 = np.zeros_like(wl)
        if p1:
            g1 = mfunc(wl, p1["amplitude"], p1["center_nm"], p1["gamma_nm"])
        if p2:
            g2 = mfunc(wl, p2["amplitude"], p2["center_nm"], p2["gamma_nm"])
        y_total = bl + g1 + g2
        y1 = (bl + g1) if p1 else None
        y2 = (bl + g2) if p2 else None
        self.liveView.set_linecut_fit_overlay(wl, y_total, y1, y2)

    def _on_mode_changed(self, text: str) -> None:
        self._current_mode = text
        self._main_idx = 0
        self._fixed_idx = 0
        self._clear_all_fixed_overlays()   # x-axis meaning changed
        self._reconfigure_sliders()

    def _on_main_changed(self, v: int) -> None:
        self._main_idx = int(v)
        self._refresh_all()

    def _on_fixed_changed(self, v: int) -> None:
        self._fixed_idx = int(v)
        self._refresh_all()

    def _reconfigure_sliders(self) -> None:
        ch = self._primary_ch()
        if not ch.has_data:
            self.mainSlider.setRange(0, 0)
            self.fixedSlider.setRange(0, 0)
            return
        if self._current_mode == self.MODE_GATE:
            n_main = ch.gates.size
            n_fixed = ch.powers.size
        else:
            n_main = ch.powers.size
            n_fixed = ch.gates.size

        for slider, n, attr in [
            (self.mainSlider,  n_main,  "_main_idx"),
            (self.fixedSlider, n_fixed, "_fixed_idx"),
        ]:
            slider.blockSignals(True)
            slider.setRange(0, max(0, n - 1))
            setattr(self, attr, max(0, min(getattr(self, attr), max(0, n - 1))))
            slider.setValue(getattr(self, attr))
            slider.blockSignals(False)

        # auto-select mode if only one is available
        combo_was = self._current_mode
        if ch.can_gate_mode and not ch.can_power_mode:
            self._current_mode = self.MODE_GATE
            self.modeCombo.blockSignals(True)
            self.modeCombo.setCurrentText(self.MODE_GATE)
            self.modeCombo.blockSignals(False)
        elif ch.can_power_mode and not ch.can_gate_mode:
            self._current_mode = self.MODE_POWER
            self.modeCombo.blockSignals(True)
            self.modeCombo.setCurrentText(self.MODE_POWER)
            self.modeCombo.blockSignals(False)

    # ── Per-frame helpers ─────────────────────────────────────────────

    def _get_cropped_raw(self, ch: _Channel, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cropped_image, cropped_wl) — cached, no corrections applied."""
        if fr.path in ch._img_cache:
            return ch._img_cache[fr.path]
        img = _apply_crop(fr.image, self._crop)
        wl = fr.wavelength_nm
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
        do_bl = self.baselineCheck.isChecked()
        do_sub = self.subCheck.isChecked() and ch.has_substrate
        if not do_bl and not do_sub:
            return img, wl
        sub_fr = _ch_substrate_for(ch, fr) if do_sub else None
        sub_img = sub_wl = None
        if sub_fr is not None:
            sub_img_raw = _apply_crop(sub_fr.image, self._crop)
            sub_wl_raw = sub_fr.wavelength_nm
            if sub_wl_raw is None:
                sub_wl_raw = np.arange(sub_fr.image.shape[1], dtype=float)
            sub_wl = _crop_axis(sub_wl_raw, self._crop, sub_fr.image.shape[1])
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
        """Current frame to display, given sliders."""
        if not ch.has_data:
            return None
        if self._current_mode == self.MODE_GATE:
            p_key = _ch_norm_power(ch, float(
                ch.powers[self._fixed_idx])) if ch.powers.size else ch.default_power_key
            vf_key = _ch_norm_gate(
                ch, float(ch.gates[self._main_idx])) if ch.gates.size else ch.default_gate_key
        else:
            p_key = _ch_norm_power(ch, float(
                ch.powers[self._main_idx])) if ch.powers.size else ch.default_power_key
            vf_key = _ch_norm_gate(
                ch, float(ch.gates[self._fixed_idx])) if ch.gates.size else ch.default_gate_key
        return ch.grid.get((p_key, vf_key))

    # ── Refresh ───────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._update_single_frame_view()
        self._update_maps()

    def _update_single_frame_view(self) -> None:
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        fr = self._get_frame(ch)

        if fr is None and not ch.has_data:
            blank = np.zeros((10, 10), dtype=float)
            self.liveView.set_image(blank)
            self.liveView.set_image_title("No data loaded")
            self.mainLbl.setText("--")
            self.fixedLbl.setText("--")
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

        def _f(v): return "--" if v is None or not np.isfinite(float(v)
                                                               ) else f"{float(v):.4g}"
        suffix = " [LH−RH]" if self._is_diff_mode() else (
            f" [{ch_key}]" if ch_key else "")

        if ch.angle_mode:
            angle_val = float(fr.power_w) if fr.power_w is not None else np.nan
            pw = None
            if ch._angle_calib_angles is not None and np.isfinite(angle_val):
                pw = float(np.interp(angle_val, ch._angle_calib_angles,
                           ch._angle_calib_powers_w) * 1e6)
            pwr_str = f"A={_f(angle_val)} °  P={_f(pw)} µW" if pw is not None else f"A={_f(angle_val)} °"
            self.liveView.set_image_title(
                f"{pwr_str}  Vf={_f(fr.v_front)} V  Vb={_f(fr.v_back)} V{suffix}")
            if self._current_mode == self.MODE_GATE:
                self.mainLbl.setText(f"Front gate: {_f(fr.v_front)} V")
                angle_fix = float(
                    ch.powers[self._fixed_idx]) if ch.powers.size else np.nan
                pw_fix = None
                if ch._angle_calib_angles is not None and np.isfinite(angle_fix):
                    pw_fix = float(
                        np.interp(angle_fix, ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6)
                fix_str = (f"Angle: {_f(angle_fix)} °  ({_f(pw_fix)} µW) (fixed)" if pw_fix is not None
                           else f"Angle: {_f(angle_fix)} ° (fixed)")
                self.fixedLbl.setText(fix_str)
            else:
                main_str = (f"Angle: {_f(angle_val)} °  ({_f(pw)} µW)" if pw is not None
                            else f"Angle: {_f(angle_val)} °")
                self.mainLbl.setText(main_str)
                vf_fix = float(ch.gates[self._fixed_idx]
                               ) if ch.gates.size else np.nan
                self.fixedLbl.setText(f"Front gate: {_f(vf_fix)} V (fixed)")
        else:
            p_uW = float(
                fr.power_w * 1e6) if fr.power_w is not None else np.nan
            self.liveView.set_image_title(
                f"P={_f(p_uW)} µW  Vf={_f(fr.v_front)} V  Vb={_f(fr.v_back)} V{suffix}"
            )
            if self._current_mode == self.MODE_GATE:
                self.mainLbl.setText(f"Front gate: {_f(fr.v_front)} V")
                p_fix = float(ch.powers[self._fixed_idx]
                              * 1e6) if ch.powers.size else np.nan
                self.fixedLbl.setText(f"Power: {_f(p_fix)} µW (fixed)")
            else:
                self.mainLbl.setText(f"Power: {_f(p_uW)} µW")
                vf_fix = float(ch.gates[self._fixed_idx]
                               ) if ch.gates.size else np.nan
                self.fixedLbl.setText(f"Front gate: {_f(vf_fix)} V (fixed)")

    def _sg_apply(self, m: np.ndarray) -> np.ndarray:
        ws = int(self.sgSweepWin.value()
                 ) if self.sgSweepCheck.isChecked() else 0
        os = int(self.sgSweepOrd.value()
                 ) if self.sgSweepCheck.isChecked() else 1
        ww = int(self.sgWlWin.value()) if self.sgWlCheck.isChecked() else 0
        ow = int(self.sgWlOrd.value()) if self.sgWlCheck.isChecked() else 1
        return m if (ws < 3 and ww < 3) else _sg_smooth_2d(m, ws, os, ww, ow)

    def _build_map(self, ch: _Channel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, wl_ref, map_data) — shape map: (n_wl, n_x)."""
        if not ch.has_data:
            empty = np.zeros((10, 1), dtype=float)
            return np.array([0.0]), np.arange(10, dtype=float), empty

        row_disp_raw = self.liveView.linecut_row()  # None if no cursor set

        if self._current_mode == self.MODE_GATE:
            x_axis = ch.gates
            fixed_key = _ch_norm_power(ch, float(
                ch.powers[self._fixed_idx])) if ch.powers.size else ch.default_power_key
        else:
            x_axis = ch.powers if ch.angle_mode else ch.powers * 1e6
            fixed_key = _ch_norm_gate(
                ch, float(ch.gates[self._fixed_idx])) if ch.gates.size else ch.default_gate_key

        # Find reference wavelength axis from first available frame
        wl_ref = None
        sweep_vals = ch.gates if self._current_mode == self.MODE_GATE else ch.powers
        for val in sweep_vals:
            if self._current_mode == self.MODE_GATE:
                fr = ch.grid.get((fixed_key, _ch_norm_gate(ch, float(val))))
            else:
                fr = ch.grid.get((_ch_norm_power(ch, float(val)), fixed_key))
            if fr is not None:
                _, wl_ref = self._get_cropped_raw(ch, fr)
                break
        if wl_ref is None:
            wl_ref = ch.blank_wl

        wl_ref = np.asarray(wl_ref, dtype=float)
        width = int(self.liveView.linecut_width())
        cols = []

        for val in sweep_vals:
            if self._current_mode == self.MODE_GATE:
                fr = ch.grid.get((fixed_key, _ch_norm_gate(ch, float(val))))
            else:
                fr = ch.grid.get((_ch_norm_power(ch, float(val)), fixed_key))

            if fr is None:
                cols.append(np.full(wl_ref.shape, np.nan, dtype=float))
                continue
            img, wl = self._get_corrected(ch, fr)
            h = img.shape[0]
            # linecut_row() is in display space (flipud); convert to raw row
            row_d = int(row_disp_raw) if row_disp_raw is not None else h // 2
            row_d = max(0, min(row_d, h - 1))
            row_raw = h - 1 - row_d
            lc = _linecut_h(img, row_raw, width)
            if lc is None:
                cols.append(np.full(wl_ref.shape, np.nan, dtype=float))
                continue
            cols.append(np.asarray(_resample(wl, lc, wl_ref), dtype=float))

        map_data = np.stack(cols, axis=1) if cols else np.full(
            (wl_ref.size, x_axis.size), np.nan)
        return x_axis, wl_ref, map_data

    def _build_fit_map(self) -> Optional[np.ndarray]:
        """Reconstruct the fitted 2D intensity map from stored peak-fit parameters.

        Returns a (n_wl, n_sweep) array in the same wavelength space as
        self._last_y_wl, or None if no valid fit is available.
        """
        if self._fit_result is None or self._last_y_wl is None or self._last_x is None:
            return None
        fit = self._fit_result
        wl = np.asarray(self._last_y_wl, dtype=float)
        n_wl = wl.size
        n_sw = int(self._last_x.size)

        def _arr(key):
            a = np.asarray(fit.get(key, []), dtype=float)
            out = np.full(n_sw, np.nan)
            n = min(len(a), n_sw)
            out[:n] = a[:n]
            return out

        c1 = _arr("center_1_nm")
        gm1 = _arr("gamma_1_nm")
        a1 = _arr("amp_1")
        c2 = _arr("center_2_nm")
        gm2 = _arr("gamma_2_nm")
        a2 = _arr("amp_2")
        off = _arr("offset")

        fit_model = self.fitModelCombo.currentText().lower()
        mfunc, _ = _peak_model(fit_model)

        z_fit = np.zeros((n_wl, n_sw), dtype=float)
        for j in range(n_sw):
            col = np.zeros(n_wl)
            if np.isfinite(a1[j]) and np.isfinite(c1[j]) and np.isfinite(gm1[j]) and gm1[j] > 0:
                col += mfunc(wl, a1[j], c1[j], gm1[j])
            if np.isfinite(a2[j]) and np.isfinite(c2[j]) and np.isfinite(gm2[j]) and gm2[j] > 0:
                col += mfunc(wl, a2[j], c2[j], gm2[j])
            # offset intentionally excluded: baseline-subtracted so all columns share a 0 floor
            z_fit[:, j] = col
        return z_fit

    def _build_sat_norm_map(self):
        """Return (y_ev, sd_sum) where sd_sum[i,j] = Σ_k (V_Pk−c_Pk)·(1−θ_Pk).

        θ_Pk = (V_Pk − c_Pk) / Vmax_Pk  [0=linear, 1=saturated, saturation_fit.py convention]
        So each term (V_Pk−c_Pk)·(1−θ_Pk) is zero in both extremes and peaks at half-saturation.
        V_P1 + V_P2 reconstructs the total fitted map.

        Hill fit is run independently per energy row.  Result cached by _fit_map_version.
        """
        if (self._fit_result is None or self._last_y_wl is None
                or self._last_x is None
                or self._current_mode != self.MODE_POWER):
            return None

        cache = getattr(self, "_sat_norm_cache", None)
        if cache is not None and cache[0] == self._fit_map_version:
            # (version, y_ev, z_p1_ev, z_p2_ev, sd_p1, sd_p2, sd_sum)
            return cache[1], cache[2], cache[3], cache[4], cache[5], cache[6]

        from scipy.optimize import curve_fit as _cf

        fit = self._fit_result
        wl = np.asarray(self._last_y_wl, dtype=float)
        x = np.asarray(self._last_x,    dtype=float)

        if not np.any(x[np.isfinite(x)] > 0):
            return None

        n_wl = wl.size
        n_sw = x.size

        def _arr(key):
            a = np.asarray(fit.get(key, []), dtype=float)
            out = np.full(n_sw, np.nan)
            out[:min(len(a), n_sw)] = a[:min(len(a), n_sw)]
            return out

        c1 = _arr("center_1_nm")
        gm1 = _arr("gamma_1_nm")
        a1 = _arr("amp_1")
        c2 = _arr("center_2_nm")
        gm2 = _arr("gamma_2_nm")
        a2 = _arr("amp_2")

        mfunc, _ = _peak_model(self.fitModelCombo.currentText().lower())

        # Reconstruct per-peak maps in wavelength space
        z_p1 = np.zeros((n_wl, n_sw), dtype=float)
        z_p2 = np.zeros((n_wl, n_sw), dtype=float)
        for j in range(n_sw):
            if np.isfinite(a1[j]) and np.isfinite(c1[j]) and np.isfinite(gm1[j]) and gm1[j] > 0:
                z_p1[:, j] = mfunc(wl, a1[j], c1[j], gm1[j])
            if np.isfinite(a2[j]) and np.isfinite(c2[j]) and np.isfinite(gm2[j]) and gm2[j] > 0:
                z_p2[:, j] = mfunc(wl, a2[j], c2[j], gm2[j])

        y_ev, z_p1_ev = _nm_to_ev(wl, z_p1)
        _,    z_p2_ev = _nm_to_ev(wl, z_p2)

        n_ev = y_ev.size
        x_pos = np.clip(x, 1e-20, None)

        def _hill(xv, Vmax, n, K, c):
            return Vmax * xv**n / (K**n + xv**n) + c

        sd_p1 = np.full((n_ev, n_sw), np.nan)
        sd_p2 = np.full((n_ev, n_sw), np.nan)

        for i in range(n_ev):
            for z_row, sd_row in ((z_p1_ev[i, :], sd_p1[i, :]),
                                  (z_p2_ev[i, :], sd_p2[i, :])):
                m = (np.isfinite(z_row) & np.isfinite(x_pos)
                     & (x_pos > 0) & (z_row > 0))
                if m.sum() < 4:
                    continue
                xm, ym = x_pos[m], z_row[m]
                c0 = float(np.nanmin(ym))
                Vm0 = max(float(np.nanmax(ym)) - c0, 1e-10)
                K0 = float(np.median(xm))
                try:
                    popt, _ = _cf(
                        _hill, xm, ym,
                        p0=[Vm0, 0.8, K0, c0],
                        bounds=([0.0, 0.3, float(xm.min()) * 0.01, 0.0],
                                [Vm0 * 20, 1.0, float(xm.max()) * 10, c0 * 2 + 1.0]),
                        maxfev=10000,
                    )
                    Vmax, n_f, K_f, c_f = popt
                    V_above_c = z_row - c_f                   # S = V − c
                    # θ = S/Vmax  (0=linear, 1=sat)
                    theta = V_above_c / Vmax
                    sd_row[:] = V_above_c * (1.0 - theta)    # (V−c)·(1−θ)
                except Exception:
                    pass

        # Sum per-peak contributions; NaN only where both peaks have no valid fit
        both_nan = ~np.isfinite(sd_p1) & ~np.isfinite(sd_p2)
        sd_sum = (np.where(np.isfinite(sd_p1), sd_p1, 0.0)
                  + np.where(np.isfinite(sd_p2), sd_p2, 0.0))
        sd_sum[both_nan] = np.nan

        self._sat_norm_cache = (
            int(self._fit_map_version), y_ev, z_p1_ev, z_p2_ev, sd_p1, sd_p2, sd_sum)
        return y_ev, z_p1_ev, z_p2_ev, sd_p1, sd_p2, sd_sum

    def _open_fit_window(self) -> None:
        if not _MPL_OK:
            QtWidgets.QMessageBox.warning(self, "matplotlib missing",
                                          "matplotlib is not available.")
            return
        if self._mplFitDialog is None:
            self._mplFitDialog = MatplotlibFitDialog(self)
        self._mplFitDialog.show()
        self._mplFitDialog.raise_()
        self._update_mpl_fit()

    def _compute_fit_quality(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Per-frame RMS(residuals in each peak range) / std(off-peak noise)."""
        if (self._fit_result is None or self._last_y_wl is None
                or self._last_map is None or self._last_x is None):
            return None, None

        fit = self._fit_result
        wl = np.asarray(self._last_y_wl, dtype=float)
        data = np.asarray(self._last_map,  dtype=float)
        n_sw = int(self._last_x.size)

        def _pad(key: str) -> np.ndarray:
            a = np.asarray(fit.get(key, []), dtype=float)
            out = np.full(n_sw, np.nan)
            out[:min(len(a), n_sw)] = a[:min(len(a), n_sw)]
            return out

        c1 = _pad("center_1_nm")
        gm1 = _pad("gamma_1_nm")
        a1 = _pad("amp_1")
        c2 = _pad("center_2_nm")
        gm2 = _pad("gamma_2_nm")
        a2 = _pad("amp_2")
        off = _pad("offset")

        mfunc, _ = _peak_model(self.fitModelCombo.currentText().lower())
        lo1, hi1, lo2, hi2 = self._read_peak_ranges()

        p1_mask = (wl >= lo1) & (wl <= hi1)
        p2_mask = (wl >= lo2) & (wl <= hi2)
        bg_mask = wl < lo1                          # off-peak: below both ranges
        if bg_mask.sum() < 3:
            # fallback: any non-peak region
            bg_mask = ~(p1_mask | p2_mask)

        q1 = np.full(n_sw, np.nan)
        q2 = np.full(n_sw, np.nan)

        for j in range(n_sw):
            model = np.zeros(wl.size)
            if np.isfinite(a1[j]) and np.isfinite(c1[j]) and np.isfinite(gm1[j]) and gm1[j] > 0:
                model += mfunc(wl, a1[j], c1[j], gm1[j])
            if np.isfinite(a2[j]) and np.isfinite(c2[j]) and np.isfinite(gm2[j]) and gm2[j] > 0:
                model += mfunc(wl, a2[j], c2[j], gm2[j])
            if np.isfinite(off[j]):
                model += off[j]

            resid = data[:, j] - model
            bg_col = data[bg_mask, j]
            n_bg = int(np.isfinite(bg_col).sum())
            sigma = float(np.nanstd(bg_col)) if n_bg >= 3 else np.nan

            if np.isfinite(sigma) and sigma > 0:
                r1 = resid[p1_mask]
                if r1.size >= 2:
                    q1[j] = float(np.sqrt(np.nanmean(r1 ** 2))) / sigma
                r2 = resid[p2_mask]
                if r2.size >= 2:
                    q2[j] = float(np.sqrt(np.nanmean(r2 ** 2))) / sigma

        return q1, q2

    def _update_mpl_fit(self) -> None:
        if self._mplFitDialog is None or not self._mplFitDialog.isVisible():
            return
        z_fit_wl = self._build_fit_map()
        if z_fit_wl is None:
            return
        y_ev_fit, z_fit_ev = _nm_to_ev(
            np.asarray(self._last_y_wl, dtype=float), z_fit_wl)
        q1, q2 = self._compute_fit_quality()
        norm_result = self._build_sat_norm_map()
        if norm_result is not None:
            y_norm, z_fp1, z_fp2, z_norm_p1, z_norm_p2, z_norm_sum = norm_result
            # use per-peak grids for both subplots so they stay on the same y axis
            y_ev_fit = y_norm
            z_fit_ev = (np.where(np.isfinite(z_fp1), z_fp1, 0.0)
                        + np.where(np.isfinite(z_fp2), z_fp2, 0.0))
        else:
            z_fp1 = z_fp2 = None
            y_norm = z_norm_p1 = z_norm_p2 = z_norm_sum = None
        self._mplFitDialog.update_plot(
            self._last_x, y_ev_fit, z_fit_ev,
            x_label=str(self._last_x_label or "X"),
            y_label="Energy (eV)",
            n_contours=max(2, int(self.mapContourSpin.value())
                           ) if self.mapContourSpin.value() >= 2 else 15,
            lo_pct=float(self.contourLoPctSpin.value()),
            q1=q1, q2=q2,
            z_fit_p1=z_fp1, z_fit_p2=z_fp2,
            y_norm=y_norm,
            z_norm_p1=z_norm_p1, z_norm_p2=z_norm_p2, z_norm_sum=z_norm_sum,
            x2=self._last_x2, x2_label=self._last_x2_label,
        )

    def _open_contour_window(self) -> None:
        if not _MPL_OK:
            QtWidgets.QMessageBox.warning(self, "matplotlib missing",
                                          "matplotlib is not installed — cannot open contour window.")
            return
        if self._mplDialog is None:
            self._mplDialog = MatplotlibContourDialog(self)
        self._mplDialog.show()
        self._mplDialog.raise_()
        self._update_mpl_contour()

    def _create_contour_gif(self) -> None:
        if not _MPL_OK:
            QtWidgets.QMessageBox.warning(self, "matplotlib missing",
                                          "matplotlib is not available.")
            return
        if not _PIL_OK:
            QtWidgets.QMessageBox.warning(self, "Pillow missing",
                                          "Pillow (PIL) is required to save GIFs.\n"
                                          "Install with:  pip install Pillow")
            return

        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        if not ch.has_data:
            QtWidgets.QMessageBox.information(
                self, "No data", "Load data first.")
            return

        # Determine fixed axis values and their display labels
        if self._current_mode == self.MODE_GATE:
            fixed_arr = ch.powers          # powers are fixed; gates are swept
            if ch.angle_mode and ch._angle_calib_angles is not None:
                fixed_label = "Power (µW)"

                def _fix_str(v):
                    pw = float(
                        np.interp(float(v), ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6)
                    return f"P = {pw:.4g} µW"
            elif ch.angle_mode:
                fixed_label = "Angle (°)"
                def _fix_str(v): return f"Angle = {float(v):.4g} °"
            else:
                fixed_label = "Power (µW)"
                def _fix_str(v): return f"P = {float(v)*1e6:.4g} µW"
        else:
            fixed_arr = ch.gates           # gates are fixed; powers are swept
            fixed_label = "Vf (V)"
            def _fix_str(v): return f"Vf = {float(v):.4g} V"

        n_fixed = int(fixed_arr.size)
        if n_fixed == 0:
            QtWidgets.QMessageBox.information(
                self, "No data", "No fixed-axis values found.")
            return

        # ── Frame selector dialog ─────────────────────────────────────────
        frame_labels = [
            f"{i:3d}   {_fix_str(fixed_arr[i])}" for i in range(n_fixed)]
        _def_y_min = float(self._last_map_ev.min()
                           ) if self._last_map_ev is not None else 1.4
        _def_y_max = float(self._last_map_ev.max()
                           ) if self._last_map_ev is not None else 1.8
        sel_dlg = GifFrameSelectDialog(frame_labels, y_min=_def_y_min, y_max=_def_y_max,
                                       parent=self)
        if sel_dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        # list of original indices in user order
        frame_order = sel_dlg.original_indices()
        if not frame_order:
            QtWidgets.QMessageBox.information(
                self, "No frames", "No frames remaining.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save contour GIF", str(Path.home()), "GIF animation (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"

        # ── Collect one 2-D map per selected fixed-axis value ────────────
        # Use _last_x for the sweep axis: it has the correct post-processed values
        # (e.g. calibrated angle→µW conversion) that _build_map's raw return lacks.
        saved_fixed_idx = self._fixed_idx
        all_z_ev:  list = []
        common_x:  np.ndarray = np.asarray(self._last_x, dtype=float)
        common_y:  Optional[np.ndarray] = None

        n_frames = len(frame_order)
        collect_progress = QtWidgets.QProgressDialog(
            "Building maps…", "Cancel", 0, n_frames, self)
        collect_progress.setWindowModality(QtCore.Qt.WindowModal)
        collect_progress.setMinimumDuration(0)
        collect_progress.setValue(0)
        _build_canceled = False
        try:
            for step, orig_idx in enumerate(frame_order):
                if collect_progress.wasCanceled():
                    _build_canceled = True
                    break
                self._fixed_idx = orig_idx
                _, wl_nm, m_data = self._build_map(ch)
                m_smooth = self._sg_apply(np.asarray(m_data, dtype=float))
                y_ev, z_ev = _nm_to_ev(np.asarray(
                    wl_nm, dtype=float), m_smooth)
                all_z_ev.append(np.asarray(z_ev, dtype=float))
                if common_y is None:
                    common_y = np.asarray(y_ev, dtype=float)
                collect_progress.setValue(step + 1)
                QtWidgets.QApplication.processEvents()
        finally:
            self._fixed_idx = saved_fixed_idx
            collect_progress.close()

        if _build_canceled or not all_z_ev:
            return

        x_label = str(self._last_x_label or "Sweep")

        # Apply user-chosen energy crop
        e_lo, e_hi = sel_dlg.energy_range()
        e_mask = (common_y >= e_lo) & (common_y <= e_hi)
        if e_mask.any():
            common_y = common_y[e_mask]
            all_z_ev = [z[e_mask, :] for z in all_z_ev]

        n_lev = max(2, int(self.mapContourSpin.value())
                    ) if self.mapContourSpin.value() >= 2 else 15
        lo_frac = float(self.contourLoPctSpin.value()) / 100.0
        levels = np.linspace(lo_frac, 1.0, n_lev + 1)

        X_grid, Y_grid = np.meshgrid(common_x, common_y)

        # ── Render frames ─────────────────────────────────────────────────
        progress = QtWidgets.QProgressDialog(
            "Generating GIF frames…", "Cancel", 0, n_frames, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        frames_pil: list = []
        _canceled = False
        _render_error: Optional[str] = None

        try:
            for step, orig_idx in enumerate(frame_order):
                if progress.wasCanceled():
                    _canceled = True
                    break

                try:
                    z_fr = all_z_ev[step]
                    fin = z_fr[np.isfinite(z_fr)]
                    if fin.size == 0:
                        progress.setValue(step + 1)
                        QtWidgets.QApplication.processEvents()
                        continue
                    z_min = float(np.nanmin(fin))
                    z_max = float(np.nanpercentile(fin, 99.0))
                    if not np.isfinite(z_max) or z_max <= z_min:
                        z_max = float(np.nanmax(fin))
                    z_norm = (z_fr - z_min) / (z_max - z_min)

                    fig = Figure(figsize=(7, 4.5), dpi=100)
                    fig.patch.set_facecolor("white")
                    canvas = _FigCanvasAgg(fig)
                    ax = fig.add_subplot(111)
                    ax.set_facecolor("white")

                    ax.contourf(X_grid, Y_grid, z_norm,
                                levels=levels, cmap="viridis", extend="both")
                    ax.set_ylim(e_lo, e_hi)

                    ax.set_xlabel(x_label,       color="black")
                    ax.set_ylabel("Energy (eV)", color="black")
                    ax.tick_params(colors="black")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("black")

                    ax.text(
                        0.02, 0.97,
                        f"{fixed_label}: {_fix_str(fixed_arr[orig_idx])}",
                        transform=ax.transAxes,
                        color="white", fontsize=9, fontweight="bold",
                        verticalalignment="top",
                    )

                    fig.tight_layout()
                    canvas.draw()
                    w, h = canvas.get_width_height()
                    buf = np.frombuffer(
                        canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                    frames_pil.append(_PILImage.fromarray(buf[:, :, :3]))

                except Exception as exc:
                    _render_error = f"Frame {step} (orig {orig_idx}): {exc}"
                    break

                progress.setValue(step + 1)
                QtWidgets.QApplication.processEvents()
        finally:
            progress.close()

        if _render_error:
            QtWidgets.QMessageBox.critical(
                self, "Error rendering GIF frames", _render_error)
            return
        if _canceled or not frames_pil:
            return

        try:
            frames_pil[0].save(
                path, save_all=True, append_images=frames_pil[1:],
                loop=0, duration=500, optimize=False,  # 500 ms = 2 fps
            )
            QtWidgets.QMessageBox.information(
                self, "GIF saved", f"Saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error saving GIF", str(exc))

    def _update_mpl_contour(self) -> None:
        if self._mplDialog is None or not self._mplDialog.isVisible():
            return
        if (self._last_x is None or self._last_map_ev is None or self._last_z_ev is None):
            return
        self._mplDialog.update_plot(
            self._last_x, self._last_map_ev, self._last_z_ev,
            x_label=str(self._last_x_label or "X"),
            y_label="Energy (eV)",
            n_contours=max(2, int(self.mapContourSpin.value())
                           ) if self.mapContourSpin.value() >= 2 else 15,
            lo_pct=float(self.contourLoPctSpin.value()),
            z_deriv=self._last_d_ev,
            x2=self._last_x2, x2_label=self._last_x2_label,
        )

    def _update_maps(self) -> None:
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]

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

        self._map_version += 1
        self._last_y_wl = np.asarray(wl_nm, dtype=float)
        self._last_map = np.asarray(m, dtype=float)

        m_smooth = self._sg_apply(m)

        x2 = None
        x2_label = ""
        if self._current_mode == self.MODE_GATE:
            deriv = _deriv_vs_x(m_smooth, x)
            x_label = "Front gate (V)"
            marker = float(ch.gates[self._main_idx]) if ch.gates.size else 0.0
            x2 = _cal_ef_mev(x)
            x2_label = CAL_EF_LABEL
            self.mapView.set_secondary_axis(x2, label=x2_label)
            self.derivView.set_secondary_axis(x2, label=x2_label)
            self.mapView.setTitle("Linecut vs Front Gate")
            self.derivView.setTitle("dI/dVf")
        elif ch.angle_mode:
            if ch._angle_calib_angles is not None:
                x_pw = np.interp(
                    ch.powers, ch._angle_calib_angles, ch._angle_calib_powers_w)
                x = x_pw * 1e6
                deriv = _deriv_dlogp(m_smooth, x_pw)
                x_label = "Power (µW)"
                marker = float(np.interp(ch.powers[self._main_idx], ch._angle_calib_angles,
                                         ch._angle_calib_powers_w) * 1e6) if ch.powers.size else 0.0
                self.derivView.setTitle("dI/dlog₁₀P")
                vg_fixed = float(ch.gates[self._fixed_idx]) if ch.gates.size else ch.default_gate_key
                x2 = _cal_nex_1e12(vg_fixed, x * 1e-6)
                x2_label = CAL_NEX_LABEL
            else:
                x = ch.powers.copy()
                deriv = _deriv_vs_x(m_smooth, x)
                x_label = "Stage A angle (°)"
                marker = float(ch.powers[self._main_idx]
                               ) if ch.powers.size else 0.0
                self.derivView.setTitle("dI/dAngle")
            self.mapView.set_secondary_axis(x2, label=x2_label)
            self.derivView.set_secondary_axis(x2, label=x2_label)
            self.mapView.setTitle("Linecut vs Angle")
        else:
            deriv = _deriv_dlogp(m_smooth, ch.powers)
            x_label = "Power (µW)"
            marker = float(ch.powers[self._main_idx]
                           * 1e6) if ch.powers.size else 0.0
            vg_fixed = float(ch.gates[self._fixed_idx]) if ch.gates.size else ch.default_gate_key
            x2 = _cal_nex_1e12(vg_fixed, x * 1e-6)
            x2_label = CAL_NEX_LABEL
            self.mapView.set_secondary_axis(x2, label=x2_label)
            self.derivView.set_secondary_axis(x2, label=x2_label)
            self.mapView.setTitle("Linecut vs Power")
            self.derivView.setTitle("dI/dlog₁₀P")

        self._last_x_label = x_label
        self._last_x = np.asarray(x, dtype=float)
        self._last_x2 = None if x2 is None else np.asarray(x2, dtype=float)
        self._last_x2_label = x2_label

        # convert nm → eV for display (uniform energy grid)
        y_ev, m_ev = _nm_to_ev(wl_nm, m_smooth)
        _,    d_ev = _nm_to_ev(wl_nm, deriv)

        # energy-axis derivatives (along y)
        de_ev = _deriv_vs_ev(m_ev, y_ev)
        d2e_ev = _deriv_vs_ev(de_ev, y_ev)

        max_abs = float(self.derivMaxSpin.value())
        self.derivView.set_manual_symmetric_max(
            max_abs if max_abs > 0 else None)
        self.derivEView.set_manual_symmetric_max(
            max_abs if max_abs > 0 else None)
        self.deriv2EView.set_manual_symmetric_max(
            max_abs if max_abs > 0 else None)

        self.derivEView.set_secondary_axis(x2, label=x2_label)
        self.deriv2EView.set_secondary_axis(x2, label=x2_label)

        self._last_map_ev = np.asarray(y_ev, dtype=float)
        self._last_z_ev   = np.asarray(m_ev, dtype=float)
        self._last_d_ev   = np.asarray(d_ev, dtype=float)
        self.mapView.set_map(x, y_ev, m_ev,   x_label=x_label, y_label="Energy (eV)", symmetric=False,
                             n_contours=int(self.mapContourSpin.value()),
                             contour_lo_frac=float(self.contourLoPctSpin.value()) / 100.0)
        self._update_mpl_contour()
        self._update_mpl_fit()
        self.derivView.set_map(
            x, y_ev, d_ev,   x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.derivEView.set_map(
            x, y_ev, de_ev,  x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.deriv2EView.set_map(
            x, y_ev, d2e_ev, x_label=x_label, y_label="Energy (eV)", symmetric=True)
        self.mapView.set_x_marker(marker)
        self.derivView.set_x_marker(marker)
        self.derivEView.set_x_marker(marker)
        self.deriv2EView.set_x_marker(marker)

        self.refIndexSpin.setRange(0, max(0, int(x.size) - 1))
        self._update_range_shading()
        for _w in (self.fitWidthView, self.fitEnergyView, self.fitIntensityView,
                   self.fitAucView, self.satDevView, self.fitRatioView):
            _w.set_secondary_axis(x2, label=x2_label)
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
            ref_idx=int(self.refIndexSpin.value()),
            dip_guesses_nm=np.array(guesses, dtype=float),
            window_nm=float(self.dipWindowSpin.value()),
            max_shift_nm=float(self.dipMaxShiftSpin.value()),
            smooth_sigma_nm=float(self.dipSmoothSpin.value()),
        )
        self._dip_fit_result = fit
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
            _apply_top_axis(self.dipPosPlot.plotItem, None, None, "")
            return
        centers = np.asarray(
            self._dip_fit_result["centers_nm"])  # (n_dips, n_x)
        n_dips = centers.shape[0]
        while len(self._dipPosCurves) < n_dips:
            i = len(self._dipPosCurves)
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
        _apply_top_axis(self.dipPosPlot.plotItem, x, self._last_x2, self._last_x2_label)

    def _update_dip_trend_tabs(self, x, x_label: str) -> None:
        """Populate FitTrend tabs from dip fit result (energy + intensity only)."""
        msg = "Click 'Fit Dips' to run / refresh"
        if self._dip_fit_result is None or \
                int(self._dip_fit_map_version) != int(self._map_version):
            self.fitWidthView.clear_data(msg, x_label=x_label)
            self.fitEnergyView.clear_data(msg, x_label=x_label)
            self.fitIntensityView.clear_data(msg, x_label=x_label)
            self.fitAucView.clear_data(msg, x_label=x_label)
            return

        centers = np.asarray(
            self._dip_fit_result["centers_nm"])  # (n_dips, n_x)
        amps = np.asarray(self._dip_fit_result["amps"])         # (n_dips, n_x)

        def _row(arr, i):
            if i < arr.shape[0]:
                return arr[i, :].copy()
            return np.full(x.size, np.nan)

        c1 = _row(centers, 0)
        c2 = _row(centers, 1)
        a1 = _row(amps,    0)
        a2 = _row(amps,    1)

        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = EV_NM / c1
            e2 = EV_NM / c2
        e1[~np.isfinite(e1)] = np.nan
        e2[~np.isfinite(e2)] = np.nan

        self.fitWidthView.clear_data("N/A in dip mode", x_label=x_label)
        self.fitAucView.clear_data("N/A in dip mode", x_label=x_label)
        self.satDevView.clear_data("N/A in dip mode", x_label=x_label)
        self.fitRatioView.clear_data("N/A in dip mode", x_label=x_label)
        self.fitEnergyView.set_data(x, e1, e2,
                                    x_label=x_label,
                                    label_1="Dip 1", label_2="Dip 2")
        self.fitIntensityView.set_data(x, a1, a2, x_label=x_label)

    def _on_dip_smooth_changed(self) -> None:
        if self._active_overlay == "dip":
            self._update_live_dip_overlay()

    def _update_live_dip_overlay(self) -> None:
        guesses = self._read_dip_guesses()
        if not guesses:
            self.liveView.clear_dip_overlay()
            return
        ch_key = self._current_ch_key()
        ch = self._ch[ch_key]
        fr = self._get_frame(ch)
        if fr is None:
            self.liveView.clear_dip_overlay()
            return
        img, wl = self._get_corrected(ch, fr)
        h = img.shape[0]
        row_d = self.liveView.linecut_row()
        row_d = h // 2 if row_d is None else max(0, min(int(row_d), h - 1))
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
            window_nm=float(self.dipWindowSpin.value()),
            max_shift_nm=float(self.dipMaxShiftSpin.value()),
            smooth_sigma_nm=float(self.dipSmoothSpin.value()),
        )
        positions = [r["center_nm"] if r is not None else None
                     for r in fit["dips"]]
        if not any(p is not None for p in positions):
            self.liveView.clear_dip_overlay()
            return
        self.liveView.set_dip_vlines(positions,
                                     x_smooth=fit.get("x_smooth"),
                                     y_smooth=fit.get("y_smooth"))

    # ── GIF export ────────────────────────────────────────────────────

    def _on_save_gif(self) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency",
                "Pillow is required to save GIFs.\n"
                "Install it with:  pip install Pillow"
            )
            return

        import io as _io

        ch = self._primary_ch()
        if not ch.has_data:
            QtWidgets.QMessageBox.warning(self, "Save GIF", "No data loaded.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save animated GIF", "", "GIF (*.gif)"
        )
        if not path:
            return

        fps, ok = QtWidgets.QInputDialog.getDouble(
            self, "GIF Settings", "Frames per second:", 2.0, 0.1, 30.0, 1
        )
        if not ok:
            return

        if self._current_mode == self.MODE_GATE:
            fixed_values = ch.powers

            def _fmt(v: float) -> str:
                if ch.angle_mode:
                    if ch._angle_calib_angles is not None:
                        pw = float(np.interp(v, ch._angle_calib_angles,
                                             ch._angle_calib_powers_w) * 1e6)
                        return f"P = {pw:.4g} µW"
                    return f"A = {v:.4g}°"
                return f"P = {v * 1e6:.4g} µW"
        else:
            fixed_values = ch.gates

            def _fmt(v: float) -> str:
                return f"Vf = {v:.4g} V"

        n = fixed_values.size
        if n == 0:
            QtWidgets.QMessageBox.warning(
                self, "Save GIF", "No fixed-axis values.")
            return

        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        except Exception:
            font = ImageFont.load_default()

        original_fixed = self._fixed_idx
        progress = QtWidgets.QProgressDialog(
            "Generating GIF…", "Cancel", 0, n, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        frames: List[Image.Image] = []
        try:
            for i in range(n):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()

                self._fixed_idx = i
                self._update_maps()
                QtWidgets.QApplication.processEvents()

                # Grab the full graphics widget then crop to the colormap PlotItem only
                glw = self.mapView._glw
                full_pixmap = glw.grab()
                scene_rect = self.mapView._mapPlot.sceneBoundingRect()
                tl = glw.mapFromScene(scene_rect.topLeft())
                br = glw.mapFromScene(scene_rect.bottomRight())
                crop_rect = QtCore.QRect(
                    max(0, int(tl.x())), max(0, int(tl.y())),
                    max(1, int(br.x() - tl.x())), max(1, int(br.y() - tl.y()))
                )
                pixmap = full_pixmap.copy(crop_rect)

                buf = QtCore.QBuffer()
                buf.open(QtCore.QIODevice.WriteOnly)
                pixmap.save(buf, "PNG")
                buf.close()
                pil_img = Image.open(_io.BytesIO(
                    bytes(buf.data()))).convert("RGB")

                # Text overlay in top-right corner
                label = _fmt(float(fixed_values[i]))
                draw = ImageDraw.Draw(pil_img)
                img_w = pil_img.width
                pad, margin = 3, 8
                if hasattr(draw, "textbbox"):
                    bb = draw.textbbox((0, 0), label, font=font)
                    tw, th = bb[2] - bb[0], bb[3] - bb[1]
                    x0 = img_w - tw - margin - pad * 2
                    y0 = margin
                    draw.rectangle(
                        [x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                        fill="white"
                    )
                else:
                    tw = len(label) * 9
                    x0 = img_w - tw - margin - pad * 2
                    y0 = margin
                    draw.rectangle([x0 - pad, y0 - pad, x0 +
                                   tw + pad, y0 + 20], fill="white")
                draw.text((x0, y0), label, fill="black", font=font)

                frames.append(pil_img)
        finally:
            progress.close()
            self._fixed_idx = original_fixed
            self._update_maps()

        if not frames:
            return

        duration_ms = max(1, int(1000.0 / fps))
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            loop=0, duration=duration_ms, optimize=False
        )
        QtWidgets.QMessageBox.information(
            self, "GIF Saved",
            f"Saved {len(frames)} frame(s) to:\n{path}"
        )


# ── tiny helper ───────────────────────────────────────────────────────

def _bold_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(f"<b>{text}</b>")
    return lbl


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    pg.setConfigOptions(background="w", foreground="k", useOpenGL=False)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Arial", 9))
    win = PLViewerV2()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
