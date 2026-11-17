"""
pl_trpl_dual_gate_viewer.py

PL + TRPL viewer for DUAL-GATE scans  —  one file.

For folders of the PL + TRPL v2 measurement GUI taken in dual-gate mode (PL .asc
files, SPAD23 or SPAD93 TRPL_*.csv files, session.md, DARK_TRPL_*.csv).  Needs
numpy, scipy, pandas, PyQt5, pyqtgraph.

    python pl_trpl_dual_gate_viewer.py [--folder SCAN] [--calib gtd1_calibration.json]
                                       [--ndmap dual_gate_speccube_gtd1.npz | NDMAP_FOLDER]

Tabs
  1. n-D map — gate calibration (r, u_n0, u_D0, d_F, ε) and an n-D map (the gate
     grids viewer's dual_gate_speccube_gtd1.npz cache, or a folder of .asc frames).
     The scan's gate points are drawn on the map as a linecut.  A linecut within
     the tolerance of the u_D axis is a constant-n scan (swept in D), one within
     the tolerance of the u_n axis a constant-D scan (swept in n): the sweep axis
     of the other tabs is then D/ε₀ or n (u_D or u_n without physical units);
     otherwise it is the front gate voltage.
     Peak fit (Gaussian, Lorentzian or Gauss ⊗ 2 exp = Gaussian ⊗ asymmetric exponential):
     clicking a map point fits its spectrum (center, height, FWHM shown under it).  Sub-tabs
     "Spectrum vs n (const D)" / "Spectrum vs D (const n)": linecuts through the map (as the
     gate grids viewer's waterfalls; drawn dashed on the map, slider or click to move) with
     "Fit linecut" → fitted centers on the spectrum map, intensity / center / width / local
     slope dE/dx versus n or D, a straight-line fit over a dragged range, and the dipole
     moment p = −ε_⊥·dE/d(D/ε₀) when the sweep axis is D.
  2. Point — PL image + spectrum, TRPL count map at the time-scrollbar bin with
     the 1σ / 2σ contours of its 2D Gaussian fit, decay of the selected pixel
     (left click on the map; right click = exclude the pixel from fits).
     TRPL view "Spectral" (SPAD spectral mode, as the SPAD93 / SPAD23 GUIs; picked from the
     scan header's spad_view_mode): the centre line (SPAD93 columns at y = 92 µm, SPAD23
     rows at x = 46 µm) on a wavelength axis from the λ-calibration (WLCAL_*.json in the
     folder, else the header's; "Load λ-cal…"): time-resolved spectrum image, spectrum at
     the time cursor (maximum + Gaussian centre / FWHM), decays of the selected lines, and a
     sequential 1D Gaussian fit of the active (last selected) line's pixels along y (SPAD93) / x
     (SPAD23): profile map vs time with ±σ lines and σ²(t) with the D line (tab 3's settings);
     tab 4's decay map then follows the active line.
  3. Spatial fit — radial profile of the 2D Gaussian at the time-scrollbar bin and
     σ_eq²(t) of this point ("Fit this point" / "Fit all"; never automatic).
  4. Dependence — versus gate (calibrated axis) or power: PL spectrum map, decay
     map of the selected pixel, and σ_eq²(t) map (filled by "Fit all" only).
  5. PL fit (scan) — the same linecut fit for the loaded scan's PL spectra along tab 4's
     sweep line (tab 2's CCD row / width / crop / baseline).

The PL and TRPL processing (crop, linecut, time-zero alignment, background, the
sequential 2D Gaussian fit) is copied from pl_trpl_power_gate_dep_viewer_v2.py;
the n-D map (cache builder, coordinates, map quantities) from the gate grids
viewer (pl_ndmap_viewer_gtd1.py, build_gtd1_cache.py, dual_gate_asc_io.py).
"""

import json
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None
try:
    from scipy.special import erfc, erfcx
except ImportError:
    erfc = erfcx = None
try:
    from scipy.interpolate import LinearNDInterpolator as _LinearND
except ImportError:
    _LinearND = None

DEFAULT_CROP  = (50, 50, 200, 200)
EV_NM         = 1239.841984
GRID_DECIMALS = 9
_DIP_COLORS   = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════
# PL frames  (copied from pl_trpl_power_gate_dep_viewer_v2.py)
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
# SingleFrameView  (pyqtgraph single-frame viewer with linecut)
# ══════════════════════════════════════════════════════════════════════

def export_matrix_csv(path, x, y, z) -> None:
    """Origin-ready matrix CSV: first row = x values (top-left cell empty), first column = y values,
    z[k, j] at (x[j], y[k]); numbers only (NaN cells left empty), no header."""
    x, y, z = np.asarray(x, float).ravel(), np.asarray(y, float).ravel(), np.asarray(z, float)
    if z.shape != (y.size, x.size):
        raise ValueError(f"matrix shape {z.shape} does not match y × x = {y.size} × {x.size}")

    def num(v):
        return "" if not np.isfinite(v) else f"{v:.10g}"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("," + ",".join(num(v) for v in x) + "\n")
        for k in range(y.size):
            fh.write(num(y[k]) + "," + ",".join(num(v) for v in z[k]) + "\n")


def _export_csv_dialog(parent, default_name: str, write) -> None:
    """Ask for a file (start in the last export folder) and call write(path)."""
    st = _settings()
    start = str(st.value("export/csv_dir") or "")
    name = re.sub(r"[^\w.+-]+", "_", default_name).strip("_")[:120] or "map"
    fn, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Export colour map (Origin matrix CSV)",
                                                  str(Path(start) / f"{name}.csv"), "CSV (*.csv)")
    if not fn:
        return
    st.setValue("export/csv_dir", str(Path(fn).parent))
    try:
        write(fn)
    except (OSError, ValueError) as exc:
        QtWidgets.QMessageBox.warning(parent, "Export CSV", str(exc))


def _export_button(tip: str) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton("Export CSV…")
    b.setToolTip(tip + "\nOrigin-ready matrix: first row = x values, first column = y values, numbers only.")
    b.setEnabled(False)
    return b


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
        self.exportBtn = _export_button("Export the PL image: x = wavelength (nm; pixel without a wavelength axis),\n"
                                        "y = spatial pixel as displayed (0 = top row)")
        self.exportBtn.clicked.connect(self._on_export)
        ctrl.addWidget(self.exportBtn)
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
        self.exportBtn.setEnabled(bool(np.isfinite(self._img_data).any()))

    def set_image_title(self, text: str) -> None:
        self._titleLbl.setText(str(text))

    def export_csv(self, fn) -> None:
        x = self._wl if self._wl is not None and self._wl.size == self._w else np.arange(self._w, dtype=float)
        export_matrix_csv(fn, x, np.arange(self._h, dtype=float), np.flipud(self._img_data))

    def _on_export(self) -> None:
        if self._img_data is None:
            return
        name = self._titleLbl.text().split("·")[-1].strip() or "PL_image"
        _export_csv_dialog(self, Path(name).stem + "_image", self.export_csv)

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
# Data loading  (pandas C-engine read: ~10x faster than np.loadtxt)
# ══════════════════════════════════════════════════════════════════════

def _read_asc(path: Path) -> Tuple[dict, Optional[np.ndarray], np.ndarray]:
    """Same result as _load_andor_ascii: (meta, wavelength or None, image [rows, wl])."""
    meta = _parse_header(path)
    try:
        import pandas as pd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c").to_numpy(dtype=float)
    except Exception:
        data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    wl = None
    cols = str(meta.get("columns", "")).lower()
    if data.shape[1] > 1 and "wavelength" in cols:
        wl    = data[:, 0]
        image = data[:, 1:].T
    else:
        image = data
    return meta, wl, image


def _load_frames(data_dir: Path, paths: Optional[List[Path]] = None, progress=None, cancelled=None) -> List[PLFrame]:
    """Every .asc in ``data_dir``, or only ``paths`` when given."""
    frames: List[PLFrame] = []
    paths = sorted(Path(data_dir).glob("*.asc")) if paths is None else list(paths)
    for k, path in enumerate(paths):
        if cancelled is not None and cancelled():
            break
        try:
            meta, wl, image = _read_asc(path)
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
            vf_named   = vf_n is not None,
            vb_named   = vb_n is not None,
        ))
        if progress is not None:
            progress(k + 1, len(paths))
    return frames


# ══════════════════════════════════════════════════════════════════════
# n-D map cache  (copied from the gate grids viewer: dual_gate_asc_io.py + build_gtd1_cache.py)
# ══════════════════════════════════════════════════════════════════════
#   cube (nvf, nvb, nwin, nwl) float32 (NaN = not measured), wl (nm), vf_vals, vb_vals,
#   tstamp (nvf, nvb), spat_rows (absolute CCD rows of the window), spot_center.

ND_CACHE_NAME  = "dual_gate_speccube_gtd1.npz"
ND_SPAT_HALF   = 35
ND_DEFAULT_ROW = 131
ND_BAND_HALF   = 6
ND_EDGE_GUARD  = 12
ND_ROW_PROBE_N = 30


def _nd_header_timestamp(path, meta):
    from datetime import datetime
    raw = meta.get("timestamp")
    if raw:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(raw, fmt).timestamp()
            except Exception:
                pass
    return float(path.stat().st_mtime)


def _nd_has_wl_column(meta):
    """True if the frame was saved with a wavelength column ("Columns: wavelength_nm, ...")."""
    return "wavelength" in meta.get("columns", "").lower()


def _nd_read_frame(path, wl_axis=None):
    """Whole data block -> (n_wl, 1 + n_spatial); a RAW image (no wavelength column) gets wl_axis prepended."""
    try:
        import pandas as pd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c")
        data = df.to_numpy(dtype=float)
    except Exception:
        data = np.loadtxt(path, comments="#")
    if wl_axis is not None and not _nd_has_wl_column(_parse_header(path)):
        if data.ndim != 2 or data.shape[1] != wl_axis.size:
            raise ValueError(f"{path.name}: raw frame width {data.shape} != wavelength axis "
                             f"({wl_axis.size} px)")
        data = np.column_stack([wl_axis, data.T])
    return data


def _nd_first_data_ncols(path):
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                return len(line.split())
    return 0


def nd_find_reference_wavelengths(data_dir, ref_file=None):
    """Wavelength axis for a folder of RAW-image frames, borrowed from a nearby .asc with the same
    grating, center wavelength and pixel count (``ref_file`` if given).  Raises LookupError."""
    data_dir = Path(data_dir)
    first = next(iter(sorted(data_dir.glob("*.asc"))), None)
    if first is None:
        raise FileNotFoundError(f"no .asc files in {data_dir}")
    meta = _parse_header(first)
    npix = _nd_first_data_ncols(first)
    want = (meta.get("grating"), meta.get("center_wl_nm"))

    def _match(p):
        try:
            m = _parse_header(p)
        except Exception:
            return None
        if not _nd_has_wl_column(m) or (m.get("grating"), m.get("center_wl_nm")) != want:
            return None
        wl = _nd_read_frame(p)[:, 0]
        return wl if wl.size == npix and np.all(np.isfinite(wl)) else None

    if ref_file is not None:
        wl = _match(Path(ref_file))
        if wl is None:
            raise LookupError(f"{Path(ref_file).name} has no wavelength column matching grating "
                              f"{want[0]}, center {want[1]} nm, {npix} px")
        return wl, Path(ref_file)

    def _subdirs(d):
        try:
            return sorted(s for s in d.iterdir() if s.is_dir())
        except OSError:
            return []

    seen = {data_dir.resolve()}
    for depth, root in enumerate((data_dir.parent, data_dir.parent.parent)):
        dirs = [root] + _subdirs(root)
        if depth == 1:
            dirs += [s for d in dirs[1:] for s in _subdirs(d)]
        for d in dirs:
            if d.resolve() in seen:
                continue
            seen.add(d.resolve())
            p = next(iter(sorted(d.glob("*.asc"))), None)
            wl = _match(p) if p is not None else None
            if wl is not None:
                return wl, p
    raise LookupError(f"frames in {data_dir.name} have no wavelength column and no reference "
                      f".asc (grating {want[0]}, center {want[1]} nm, {npix} px) was found nearby")


def _nd_find_humps(prof):
    """Emission humps well inside the detector: the dominant one, plus a strong, well separated second."""
    p = np.asarray(prof, dtype=float)
    n = p.size
    interior = np.zeros(n, bool)
    interior[ND_EDGE_GUARD:n - ND_EDGE_GUARD] = True
    med = np.median(p[interior])
    mad = np.median(np.abs(p[interior] - med)) + 1e-9
    thresh = med + 5.0 * 1.4826 * mad
    strong = (p > thresh) & interior
    if not strong.any():
        return [int(ND_EDGE_GUARD + np.argmax(p[interior]))]
    humps, i = [], 0
    while i < n:
        if strong[i]:
            j = i
            while j < n and strong[j]:
                j += 1
            seg = np.arange(i, j)
            c = int(round((seg * p[seg]).sum() / p[seg].sum()))
            humps.append((float(p[i:j].max()), c))
            i = j
        else:
            i += 1
    humps.sort(reverse=True)
    keep = [humps[0][1]]
    for pk, c in humps[1:]:
        if pk >= 0.40 * humps[0][0] and abs(c - humps[0][1]) >= 20:
            keep.append(c)
    return keep


def _nd_detect_emission_bands(files, force_row, wl_axis=None):
    """Average PL-only spatial profile over a probe subset; return (profile, bands, wl)."""
    n = len(files)
    idx = np.unique(np.linspace(0, n - 1, min(ND_ROW_PROBE_N, n)).astype(int))
    frames = []
    for i in idx:
        try:
            data = _nd_read_frame(files[i], wl_axis)
        except Exception:
            continue
        if data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 1:
            frames.append(data)
    if not frames:
        raise ValueError("no readable .asc frames to probe")
    shapes = [f.shape for f in frames]
    full = max(set(shapes), key=lambda s: (shapes.count(s), s[0]))
    frames = [f for f in frames if f.shape == full]
    wl = frames[0][:, 0].copy()
    prof = None
    for data in frames:
        img = data[:, 1:]
        pl = img - np.median(img, axis=0, keepdims=True)
        s = np.clip(pl, 0, None).sum(axis=0)
        prof = s if prof is None else prof + s
    prof = prof / len(frames)
    peaks = [int(force_row)] if force_row is not None else _nd_find_humps(prof)
    if not peaks:
        peaks = [ND_DEFAULT_ROW]
    bands = [(p, max(0, p - ND_BAND_HALF), min(len(prof) - 1, p + ND_BAND_HALF)) for p in peaks]
    return prof, bands, wl


def _nd_axis_values(vals):
    """Sorted lattice of the distinct gate values on one axis (near-duplicates within 1/4 step merged)."""
    v = np.array(sorted(float(x) for x in vals), float)
    if v.size == 0:
        return v
    uniq = np.unique(np.round(v, 6))
    tol = 0.25 * float(np.median(np.diff(uniq))) if uniq.size > 1 else 1e-3
    reps, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            reps.append(float(np.mean(cur)))
            cur = [x]
    reps.append(float(np.mean(cur)))
    return np.array(reps, float)


def nd_build_cube(data_dir, cache_path=None, spat_half=ND_SPAT_HALF, progress=None, prev=None, wl_ref=None):
    """Spectral cube of one .asc dual-gate sweep folder (as build_gtd1_cache.build_cube).

    ``prev``: the previous cache of the SAME folder: frames it holds that are not newer than
    prev["built"] are copied, only new ones are read.  ``progress(done, total)`` may raise to abort.
    """
    import time as _time
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.asc"))
    if not files:
        raise FileNotFoundError(f"no .asc files in {data_dir}")
    t_build = _time.time()
    raw = not _nd_has_wl_column(_parse_header(files[0]))

    if prev is not None:
        wl = np.asarray(prev["wl"], float)
        wl_source = str(prev["wl_source"]) if "wl_source" in prev else "frames"
        rows = np.asarray(prev["spat_rows"], int)
        center = int(prev["spot_center"])
        lo, hi = int(rows[0]), int(rows[-1])
        p_cube, p_ts = prev["cube"], np.asarray(prev["tstamp"], float)
        p_vf, p_vb = np.asarray(prev["vf_vals"], float), np.asarray(prev["vb_vals"], float)
        p_built = float(prev["built"]) if "built" in prev else 0.0
    else:
        wl_source = "frames"
        wl = None
        if raw:
            wl, ref = nd_find_reference_wavelengths(data_dir, wl_ref)
            wl_source = str(ref)
        prof, bands, wl = _nd_detect_emission_bands(files, None, wl if raw else None)
        center = int(bands[0][0])
        lo = max(0, center - spat_half)
        hi = center + spat_half
        rows = np.arange(lo, hi + 1)

    parsed, vfs, vbs = [], [], []
    for f in files:
        _, vf, vb = _parse_from_name(f.name)
        if vf is None or vb is None:
            continue
        parsed.append((f, float(vf), float(vb)))
        vfs.append(float(vf))
        vbs.append(float(vb))
    if not parsed:
        raise ValueError(f"no Vf/Vb-tagged .asc files in {data_dir}")

    vf_vals = _nd_axis_values(vfs)
    vb_vals = _nd_axis_values(vbs)
    nvf, nvb, nwin, nwl = vf_vals.size, vb_vals.size, rows.size, wl.size
    cube = np.full((nvf, nvb, nwin, nwl), np.nan, np.float32)
    tstamp = np.full((nvf, nvb), np.nan)
    if prev is not None:
        def _tol(v):
            return 0.25 * float(np.min(np.diff(v))) if v.size > 1 else 1e-3
        tol_f, tol_b = _tol(p_vf), _tol(p_vb)
    total = len(parsed)
    for k, (f, vf, vb) in enumerate(parsed):
        i = int(np.argmin(np.abs(vf_vals - vf)))
        j = int(np.argmin(np.abs(vb_vals - vb)))
        if prev is not None:
            pi = int(np.argmin(np.abs(p_vf - vf)))
            pj = int(np.argmin(np.abs(p_vb - vb)))
            if (abs(p_vf[pi] - vf) <= tol_f and abs(p_vb[pj] - vb) <= tol_b
                    and np.isfinite(p_ts[pi, pj]) and f.stat().st_mtime < p_built):
                cube[i, j] = p_cube[pi, pj]
                tstamp[i, j] = p_ts[pi, pj]
                if progress is not None:
                    progress(k + 1, total)
                continue
        try:
            img = _nd_read_frame(f, wl if raw else None)[:, 1:]
            hi2 = min(rows[-1], img.shape[1] - 1)
            sT = img[:, lo:hi2 + 1].T
            cube[i, j, :sT.shape[0], :] = sT
            tstamp[i, j] = _nd_header_timestamp(f, _parse_header(f))
        except Exception:
            pass
        if progress is not None:
            progress(k + 1, total)

    data = dict(cube=cube, wl=wl, vf_vals=vf_vals, vb_vals=vb_vals, tstamp=tstamp,
                spat_rows=rows, spot_center=center, wl_source=wl_source, n_files=len(files),
                source_dir=str(data_dir), built=float(t_build))
    if cache_path is not None:
        np.savez_compressed(cache_path, **data)
    return data


def _nd_wl_ok(wl):
    """A real wavelength axis: finite, positive and strictly monotonic."""
    wl = np.asarray(wl, float)
    if wl.size < 2 or not np.all(np.isfinite(wl)) or np.any(wl <= 0):
        return False
    d = np.diff(wl)
    return bool(np.all(d > 0) or np.all(d < 0))


def _nd_cache_is_stale(cache, files):
    """True if a frame was written after the cache was built, or the cache's wavelength axis is bad."""
    if not files:
        return False
    try:
        with np.load(cache, allow_pickle=True) as z:
            if not _nd_wl_ok(z["wl"]):
                return True
            built = float(z["built"]) if "built" in z.files else cache.stat().st_mtime
            if "n_files" in z.files and int(z["n_files"]) != len(files):
                return True
    except Exception:
        return True
    return max(f.stat().st_mtime for f in files) >= built


def _nd_load_cache_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def nd_load(path, rebuild=False, update=False, wl_ref=None, progress=None, cancelled=None):
    """(data dict, source) from an .npz cache file or a data folder (cache loaded, updated or built).

    Like the gate grids viewer's load_or_build_cube, without dialogs: a missing wavelength
    reference raises LookupError, cancelling raises RuntimeError("cancelled").
    """
    p = Path(path)
    if p.is_file():
        return _nd_load_cache_dict(p), p
    folder = p
    cache = folder / ND_CACHE_NAME
    files = sorted(folder.glob("*.asc"))
    prev = None
    if cache.exists():
        if not files or not (rebuild or update or _nd_cache_is_stale(cache, files)):
            return _nd_load_cache_dict(cache), cache
        if not rebuild:
            try:
                prev = _nd_load_cache_dict(cache)
            except Exception:
                prev = None
            if prev is not None and not _nd_wl_ok(prev["wl"]):
                prev = None
    if not files:
        subs = [d for d in sorted(folder.iterdir())
                if d.is_dir() and (next(d.glob("*.asc"), None) is not None or (d / ND_CACHE_NAME).exists())]
        if len(subs) == 1:
            return nd_load(subs[0], rebuild, update, wl_ref, progress, cancelled)
        if len(subs) > 1:
            names = "\n".join("   • " + s.name for s in subs)
            raise FileNotFoundError(f"No .asc files directly in:\n{folder}\n\n"
                                    f"It has several data subfolders — pick one:\n{names}")
        raise FileNotFoundError(f"no cache and no .asc files in:\n{folder}")

    def cb(done, total):
        if cancelled is not None and cancelled():
            raise RuntimeError("cancelled")
        if progress is not None:
            progress(done, total)

    data = nd_build_cube(folder, cache_path=cache, progress=cb, prev=prev, wl_ref=wl_ref)
    return data, cache


def nd_spatial_profile(cube: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """PL weight per slit row (dark floor removed), averaged over the measured gate points (in chunks)."""
    nwin = cube.shape[2]
    tot, cnt = np.zeros(nwin), np.zeros(nwin)
    pts = np.argwhere(measured)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(0, len(pts), 64):
            sl = pts[k:k + 64]
            m = cube[sl[:, 0], sl[:, 1]]                                   # (k, nwin, nwl)
            pl = m - np.nanmedian(m, axis=2, keepdims=True)
            s = np.clip(pl, 0, None).sum(axis=2)                          # NaN rows stay NaN
            ok = np.isfinite(s)
            tot += np.where(ok, s, 0.0).sum(axis=0)
            cnt += ok.sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)


def nd_prepare(data: dict) -> dict:
    """Arrays the n-D map tab works with (and the spatial profile, which is slow on big cubes)."""
    cube = data["cube"]
    if cube.dtype != np.float32:
        cube = cube.astype(np.float32)
    wl = np.asarray(data["wl"], float)
    if not _nd_wl_ok(wl):
        raise ValueError("the n-D map has no valid wavelength axis (rebuild its cache)")
    vf = np.asarray(data["vf_vals"], float)
    vb = np.asarray(data["vb_vals"], float)
    measured = np.isfinite(np.asarray(data["tstamp"], float))
    order = np.argsort(EV_NM / wl)
    return dict(cube=cube, wl=wl, vf=vf, vb=vb, measured=measured,
                spat_rows=np.asarray(data["spat_rows"], int), spot_center=int(data["spot_center"]),
                dvf=float(np.median(np.diff(vf))) if vf.size > 1 else 0.2,
                dvb=float(np.median(np.diff(vb))) if vb.size > 1 else 0.2,
                order=order, e=(EV_NM / wl)[order], prof=nd_spatial_profile(cube, measured),
                source_dir=str(data["source_dir"]) if "source_dir" in data else "")


# ══════════════════════════════════════════════════════════════════════
# Spectral peak fits: one peak, Gaussian / Lorentzian / Gaussian ⊗ asymmetric exponential
# ══════════════════════════════════════════════════════════════════════
#   Gaussian      A·exp(−(E−E0)²/2σ²)                 height A, FWHM 2√(2 ln2)·σ, area √(2π)·A·σ
#   Lorentzian    A·γ²/((E−E0)² + γ²)   (γ = HWHM)     height A, FWHM 2γ,          area π·A·γ
#   Gauss ⊗ 2 exp gauss_2exp below: each erfcx term is an exponentially modified Gaussian of area
#                 A·τ, so area = A·(τ_lo + τ_hi); peak maximum, height and FWHM are found numerically.

SHAPE_G2E, SHAPE_GAUSS, SHAPE_LOR = "Gauss ⊗ 2 exp", "Gaussian", "Lorentzian"
PEAK_SHAPES = (SHAPE_G2E, SHAPE_GAUSS, SHAPE_LOR)
E_NM_DEBYE = 48.0321            # 1 e·nm in debye
_SQRT2 = np.sqrt(2.0)


def gauss_2exp(E, A, E0, sigma, tau_lo, tau_hi):
    """Gaussian (sigma) ⊗ asymmetric exponential: low-E tail tau_lo, high-E tail tau_hi.
    erfcx form stays numerically stable when tau << sigma."""
    u = (E - E0) / sigma
    g = np.exp(-0.5 * u**2)
    return 0.5 * A * g * (erfcx((u + sigma/tau_lo) / np.sqrt(2))
                          + erfcx((-u + sigma/tau_hi) / np.sqrt(2)))

# p0 for this spectrum (eV): [1.5, 1.357, 0.0115, 0.021, 0.010]


def _emg_term(u: np.ndarray, k: float) -> np.ndarray:
    """exp(−u²/2)·erfcx((u + k)/√2) without the 0·inf far out in the tail (k = σ/τ).  For a negative
    erfcx argument it equals exp(k·u + k²/2)·erfc(z), which cannot overflow there."""
    z = (u + k) / _SQRT2
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.exp(-0.5 * u[pos] ** 2) * erfcx(z[pos])
    neg = ~pos
    out[neg] = np.exp(k * u[neg] + 0.5 * k * k) * erfc(z[neg])
    return out


def _g2e(E, A, E0, sigma, tau_lo, tau_hi):
    """gauss_2exp, evaluated without overflow far from the peak (same values elsewhere)."""
    u = (np.atleast_1d(np.asarray(E, float)) - E0) / sigma
    return 0.5 * A * (_emg_term(u, sigma / tau_lo) + _emg_term(-u, sigma / tau_hi))


def _gauss_peak(E, A, E0, sigma):
    return A * np.exp(-0.5 * ((np.asarray(E, float) - E0) / sigma) ** 2)


def _lor_peak(E, A, E0, gamma):
    return A * gamma * gamma / ((np.asarray(E, float) - E0) ** 2 + gamma * gamma)


_PEAK_FUNCS = {SHAPE_G2E: (_g2e, 5), SHAPE_GAUSS: (_gauss_peak, 3), SHAPE_LOR: (_lor_peak, 3)}
_PEAK_PARAMS = ("A", "E0", "width", "tau_lo", "tau_hi")


class PeakFitOpts:
    """Lineshape, fit window (eV) and start values (eV) of a single-peak fit."""
    __slots__ = ("shape", "lo", "hi", "width0", "tau_lo0", "tau_hi0", "offset")

    def __init__(self, shape=SHAPE_G2E, lo=1.28, hi=1.52, width0=0.0115, tau_lo0=0.021, tau_hi0=0.010,
                 offset=False):
        self.shape = shape if shape in _PEAK_FUNCS else SHAPE_G2E
        self.lo, self.hi = float(min(lo, hi)), float(max(lo, hi))
        self.width0, self.tau_lo0, self.tau_hi0 = float(width0), float(tau_lo0), float(tau_hi0)
        self.offset = bool(offset)

    def key(self):
        return (self.shape, round(self.lo, 6), round(self.hi, 6), round(self.width0, 7),
                round(self.tau_lo0, 7), round(self.tau_hi0, 7), self.offset)

    def describe(self) -> str:
        s = f"{self.shape}, window {self.lo:.3f}–{self.hi:.3f} eV"
        return s + (", constant offset fitted" if self.offset else "")


def _unit_peak(width: float, tau_lo: float, tau_hi: float) -> Tuple[float, float]:
    """(position of the maximum − E0, maximum) of gauss_2exp with A = 1."""
    x = np.linspace(-8 * width - 15 * tau_lo, 8 * width + 15 * tau_hi, 6001)
    y = _g2e(x, 1.0, 0.0, width, tau_lo, tau_hi)
    k = int(np.argmax(y))
    return float(x[k]), float(y[k])


def _peak_metrics(shape: str, p) -> Tuple[float, float, float, float]:
    """(E_peak, height, FWHM, area) of the fitted peak (offset excluded)."""
    A, E0, w = (float(v) for v in p[:3])
    if shape == SHAPE_GAUSS:
        return E0, A, 2.0 * np.sqrt(2.0 * np.log(2.0)) * w, A * w * np.sqrt(2.0 * np.pi)
    if shape == SHAPE_LOR:
        return E0, A, 2.0 * w, np.pi * A * w
    tl, th = float(p[3]), float(p[4])
    x = np.linspace(E0 - 8 * w - 15 * tl, E0 + 8 * w + 15 * th, 20001)
    y = _g2e(x, A, E0, w, tl, th)
    k = int(np.argmax(y))
    ep, h = float(x[k]), float(y[k])
    dx = float(x[1] - x[0])
    if 0 < k < x.size - 1:                                   # parabolic refinement of the maximum
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if den < 0:
            d = 0.5 * (y[k - 1] - y[k + 1]) / den
            ep, h = ep + d * dx, float(y[k] - 0.25 * (y[k - 1] - y[k + 1]) * d)
    half = 0.5 * h
    lo_i = np.flatnonzero(y[:k] < half)
    hi_i = np.flatnonzero(y[k:] < half)
    if not (lo_i.size and hi_i.size):
        fwhm = float("nan")
    else:
        i, j = int(lo_i[-1]), k + int(hi_i[0])
        xl = x[i] + (half - y[i]) / (y[i + 1] - y[i]) * dx
        xr = x[j - 1] + (half - y[j - 1]) / (y[j] - y[j - 1]) * dx
        fwhm = float(xr - xl)
    return ep, h, fwhm, A * (tl + th)


def peak_curve(res: dict, e) -> np.ndarray:
    f, npk = _PEAK_FUNCS[res["shape"]]
    return f(e, *res["popt"][:npk]) + res["offset"]


def fit_peak(e, y, opts: PeakFitOpts, p0=None) -> Optional[dict]:
    """Fit one peak of ``opts.shape`` to y(e) inside [opts.lo, opts.hi] eV.  ``p0`` = a previous
    result's popt (same opts) to start from; else start values from the data and opts.
    Returns a dict (same keys for every shape; τ = NaN unless Gauss ⊗ 2 exp) or None."""
    if _curve_fit is None or (opts.shape == SHAPE_G2E and erfcx is None):
        return None
    e = np.asarray(e, float)
    y = np.asarray(y, float)
    m = np.isfinite(e) & np.isfinite(y) & (e >= opts.lo) & (e <= opts.hi)
    f, npk = _PEAK_FUNCS[opts.shape]
    g2e = opts.shape == SHAPE_G2E
    n_par = npk + int(opts.offset)
    if m.sum() < n_par + 4:
        return None
    x, yw = e[m], y[m]
    span = max(opts.hi - opts.lo, 1e-3)
    off0 = float(np.percentile(yw, 5)) if opts.offset else 0.0
    ys = np.convolve(yw, np.ones(3) / 3.0, mode="same") if yw.size >= 5 else yw
    amp = float(np.max(ys)) - off0
    if p0 is None or len(p0) != n_par:
        if not amp > 0:
            return None
        k = int(np.argmax(ys))
        w0 = min(max(opts.width0, 2e-4), span)
        if g2e:
            tl0, th0 = min(max(opts.tau_lo0, 2e-4), span), min(max(opts.tau_hi0, 2e-4), span)
            d, pk = _unit_peak(w0, tl0, th0)
            p0 = [amp / max(pk, 1e-300), float(x[k]) - d, w0, tl0, th0]
        else:
            p0 = [amp, float(x[k]), w0]
        if opts.offset:
            p0.append(off0)
    lower = [0.0, opts.lo, 1e-4] + ([1e-4, 1e-4] if g2e else [])
    upper = [np.inf, opts.hi, span] + ([span, span] if g2e else [])
    if opts.offset:
        lower.append(-np.inf)
        upper.append(np.inf)
        model = lambda xx, *p: f(xx, *p[:-1]) + p[-1]           # noqa: E731
    else:
        model = f
    p0 = _clip_into(p0, lower, upper)
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            popt, pcov = _curve_fit(model, x, yw, p0=p0, bounds=(lower, upper), method="trf",
                                    x_scale="jac", maxfev=4000)
            yfit = model(x, *popt)
    except Exception:
        return None
    if not np.all(np.isfinite(yfit)):
        return None
    ss_res = float(np.sum((yw - yfit) ** 2))
    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(n_par, np.nan)
    perr = np.where(np.isfinite(perr), perr, np.nan)
    ep, h, fwhm, area = _peak_metrics(opts.shape, popt[:npk])
    names = ["A", "E0", "σ" if opts.shape != SHAPE_LOR else "γ", "τ_lo", "τ_hi"][:npk]
    at_bound = [nm for nm, v, lo, hi in zip(names, popt, lower, upper)
                if nm != "A" and np.isfinite(lo) and np.isfinite(hi) and min(v - lo, hi - v) <= 1e-3 * (hi - lo)]
    res = dict(shape=opts.shape, popt=[float(v) for v in popt], win=(opts.lo, opts.hi), n=int(m.sum()),
               offset_fit=opts.offset, offset=float(popt[-1]) if opts.offset else 0.0,
               offset_err=float(perr[-1]) if opts.offset else float("nan"),
               E_peak=ep, height=h, fwhm=fwhm, area=area, r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
               at_bound=at_bound)
    for i, nm in enumerate(_PEAK_PARAMS):
        res[nm] = float(popt[i]) if i < npk else float("nan")
        res[nm + "_err"] = float(perr[i]) if i < npk else float("nan")
    return res


def fit_spectra_line(e, Z, opts: PeakFitOpts, progress=None, cancelled=None) -> List[Optional[dict]]:
    """Fit every column of Z (n_e, n_x): the brightest column first from the start values, then
    outwards in both directions, each fit starting from its neighbour's result (and again from the
    start values when that fails or fits poorly — the better one is kept)."""
    e = np.asarray(e, float)
    Z = np.asarray(Z, float)
    nx = Z.shape[1]
    out: List[Optional[dict]] = [None] * nx
    m = (e >= opts.lo) & (e <= opts.hi)
    W = Z[m]
    valid = np.isfinite(W).sum(axis=0) >= 8
    if not valid.any():
        return out
    score = np.where(valid, np.nansum(np.clip(W, 0, None), axis=0), -np.inf)
    k0 = int(np.argmax(score))
    out[k0] = fit_peak(e, Z[:, k0], opts)
    done = 1
    for step in (1, -1):
        prev = out[k0]
        k = k0 + step
        while 0 <= k < nx:
            if cancelled is not None and cancelled():
                raise RuntimeError("cancelled")
            if valid[k]:
                r = fit_peak(e, Z[:, k], opts, p0=prev["popt"] if prev is not None else None)
                if r is None or not r["r2"] >= 0.8:
                    r2 = fit_peak(e, Z[:, k], opts) if prev is not None else None
                    if r2 is not None and (r is None or r2["r2"] > r["r2"]):
                        r = r2
                out[k] = r
                if r is not None and r["r2"] >= 0.5:
                    prev = r
            done += 1
            if progress is not None:
                progress(done, nx)
            k += step
    return out


def peak_text(res: Optional[dict], sep: str = "\n") -> str:
    """Readout of a peak fit: center, height, width first."""
    if res is None:
        return "fit failed"
    wn = "γ" if res["shape"] == SHAPE_LOR else "σ"
    l1 = (f"Center (peak max) {res['E_peak']:.4f} eV   ·   Height {_fmt(res['height'], '.4g')}   ·   "
          f"FWHM {res['fwhm'] * 1e3:.2f} meV")
    l2 = (f"E0 {res['E0']:.4f} ± {_fmt(res['E0_err'] * 1e3, '.2g')} meV   ·   {wn} {res['width'] * 1e3:.2f} meV   ·   "
          f"area {_fmt(res['area'], '.4g')}")
    if res["shape"] == SHAPE_G2E:
        l2 += f"   ·   τ_lo {res['tau_lo'] * 1e3:.2f}   τ_hi {res['tau_hi'] * 1e3:.2f} meV"
    l3 = f"{res['shape']}   ·   R² {_fmt(res['r2'], '.4f')}"
    if res.get("offset_fit"):
        l3 +=f"   ·   offset {_fmt(res['offset'], '.4g')}"
    if res["at_bound"]:
        l3 += "   ·   at bound: " + ", ".join(res["at_bound"])
    return sep.join((l1, l2, l3))


def _peak_html(res: Optional[dict]) -> str:
    lines = peak_text(res).split("\n")
    return f"<b>{lines[0]}</b>" + "".join(f"<br>{ln}" for ln in lines[1:])


def _linfit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares straight line: (slope, intercept, 1σ error of the slope from the scatter)."""
    xm, ym = float(x.mean()), float(y.mean())
    sxx = float(np.sum((x - xm) ** 2))
    if not sxx > 0:
        return float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym))) / sxx
    a = ym - b * xm
    dof = x.size - 2
    err = float(np.sqrt(np.sum((y - a - b * x) ** 2) / dof / sxx)) if dof > 0 else float("nan")
    return b, a, err


def local_slope(x, y, n: int, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """dy/dx at every point from a polynomial (order 1 = straight line, 3 = cubic; Savitzky–Golay)
    fitted to the n nearest valid points (one-sided windows at the ends) and
    differentiated at the point itself; 1σ error from the scatter about the fit.  NaN points are
    skipped and get no slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s, se = np.full(x.size, np.nan), np.full(x.size, np.nan)
    idx = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    order = int(order)
    n = max(int(n), order + 1)
    if idx.size < order + 1:
        return s, se
    n = min(n, idx.size)
    h = n // 2
    for pos, k in enumerate(idx):
        a = max(0, min(pos - h, idx.size - n))
        sel = idx[a:a + n]
        dx = x[sel] - x[k]
        X = np.vander(dx, order + 1, increasing=True)          # 1, dx, dx²
        try:
            coef, _res, rank, _sv = np.linalg.lstsq(X, y[sel], rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < order + 1:
            continue
        s[k] = coef[1]
        dof = sel.size - (order + 1)
        if dof > 0:
            s2 = float(np.sum((y[sel] - X @ coef) ** 2)) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se[k] = float(np.sqrt(max(cov[1, 1], 0.0)))
    return s, se


SPLINE_GCV_GAMMA = 1.4      # modified GCV (Kim & Gu): guards against GCV's occasional under-smoothing


def spline_slope(x, y, yerr=None, factor: float = 1.0):
    """dy/dx from a weighted cubic smoothing spline through the valid points (natural spline,
    penalty λ·∫f''²; Green & Silverman's Reinsch form).  λ is chosen by generalized cross-validation
    and multiplied by ``factor`` (< 1 = more local, > 1 = smoother).  The spline is linear in y, so the
    slope's ±1σ comes from exact error propagation: the y errors (``yerr``, scaled up when the scatter
    about the spline is larger than they say), or the scatter alone when there are none.
    Returns (slope, slope error, smoothed y, info dict with lam, dof = effective number of parameters)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out_s, out_se, out_f = np.full(x.size, np.nan), np.full(x.size, np.nan), np.full(x.size, np.nan)
    info = {"lam": float("nan"), "dof": float("nan")}
    idx = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    n = idx.size
    if n < 5:
        return out_s, out_se, out_f, info
    xs, ys = x[idx], y[idx]
    if np.any(np.diff(xs) <= 0):
        return out_s, out_se, out_f, info
    err = None
    if yerr is not None:
        e = np.asarray(yerr, float)[idx]
        good = np.isfinite(e) & (e > 0)
        if good.sum() >= 3:
            med = float(np.median(e[good]))
            err = np.maximum(np.where(good, e, med), 0.2 * med)        # no point dominates on a tiny error
    w = 1.0 / err ** 2 if err is not None else np.ones(n)
    w = w / w.mean()
    L = float(xs[-1] - xs[0])
    t = (xs - xs[0]) / L                                               # [0, 1]: λ grid independent of units
    h = np.diff(t)
    Q = np.zeros((n, n - 2))
    R = np.zeros((n - 2, n - 2))
    for j in range(1, n - 1):
        c = j - 1
        Q[j - 1, c], Q[j, c], Q[j + 1, c] = 1.0 / h[j - 1], -1.0 / h[j - 1] - 1.0 / h[j], 1.0 / h[j]
        R[c, c] = (h[j - 1] + h[j]) / 3.0
        if j < n - 2:
            R[c, c + 1] = R[c + 1, c] = h[j] / 6.0
    RiQt = np.linalg.solve(R, Q.T)                                     # (n−2, n)
    K = Q @ RiQt
    W = np.diag(w)

    def smoother(lam):
        return np.linalg.solve(W + lam * K, W)                         # f = S·y

    best, best_g = None, np.inf
    for lam in np.logspace(-12, 3, 76):
        S = smoother(lam)
        r = ys - S @ ys
        den = (1.0 - SPLINE_GCV_GAMMA * np.trace(S) / n) ** 2
        if 1.0 - SPLINE_GCV_GAMMA * np.trace(S) / n <= 0:
            continue
        g = float(np.sum(w * r * r)) / n / den if den > 1e-12 else np.inf
        if g < best_g:
            best, best_g = lam, g
    if best is None:
        return out_s, out_se, out_f, info
    lam = best * max(float(factor), 1e-6)
    A = np.linalg.inv(W + lam * K)
    S = A @ W
    G = np.zeros((n, n))
    G[1:-1] = RiQt                                                     # f'' at the knots from f (natural ends: 0)
    M = np.zeros((n, n))                                               # f' at the knots from f
    eye = np.eye(n)
    for i in range(n - 1):
        M[i] = (eye[i + 1] - eye[i]) / h[i] - h[i] * (2.0 * G[i] + G[i + 1]) / 6.0
    M[n - 1] = (eye[n - 1] - eye[n - 2]) / h[n - 2] + h[n - 2] * (G[n - 2] + 2.0 * G[n - 1]) / 6.0
    M /= L
    f = S @ ys
    r = ys - f
    tr = float(np.trace(S))
    dof_res = max(n - tr, 1.0)
    # unit variance of the normalized weights: var_i = s2 / w_i
    if err is not None:
        chi2 = float(np.sum((r / err) ** 2)) / dof_res
        s2 = max(chi2, 1.0) / float(np.mean(1.0 / err ** 2))
    else:
        s2 = float(np.sum(r * r)) / dof_res
    out_s[idx] = M @ f
    # Bayesian (Wahba) posterior covariance of f, s2·(W + λK)⁻¹: unlike the sampling covariance it also
    # covers the smoothing bias on average, so the ±1σ band is closer to honest
    out_se[idx] = np.sqrt(np.maximum(s2 * np.einsum("ij,jk,ik->i", M, A, M), 0.0))
    out_f[idx] = f
    info.update(lam=lam, lam_gcv=best, dof=tr)
    return out_s, out_se, out_f, info


def _uniform_rows(e: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of Z (n_e, n_x) on an evenly spaced copy of the ascending axis e (for an ImageItem)."""
    e = np.asarray(e, float)
    if e.size < 2:
        return e, Z
    eu = np.linspace(e[0], e[-1], e.size)
    out = np.full((eu.size, Z.shape[1]), np.nan)
    for j in range(Z.shape[1]):
        c = Z[:, j]
        ok = np.isfinite(c)
        if ok.sum() >= 2:
            out[:, j] = np.interp(eu, e[ok], c[ok], left=np.nan, right=np.nan)
    return eu, out


# ══════════════════════════════════════════════════════════════════════
# Gate calibration and the scan's linecut through the gate–gate plane
# ══════════════════════════════════════════════════════════════════════
#   u_n = V_f + r·V_b − u_n0 (density),  u_D = V_f − r·V_b − u_D0 (displacement), r = c_b/c_t.
#   Dual-gate capacitor with c_F = ε0·ε_hBN/d_F, u in V, d_F in nm (as in the gate grids viewer):
#   n [10¹² cm⁻²] = K_N·ε·u_n/d_F,   D/ε₀ [V/nm] = ε·u_D/(2 d_F).

K_N       = 5.5264
R_DEFAULT = 0.461


class GateCalib:
    __slots__ = ("r", "un0", "uD0", "dF", "eps", "calibrated", "physical")

    def __init__(self, r=R_DEFAULT, un0=0.0, uD0=0.0, dF=30.0, eps=3.4, calibrated=True, physical=False):
        self.r, self.un0, self.uD0 = float(r), float(un0), float(uD0)
        self.dF, self.eps = float(dF), float(eps)
        self.calibrated, self.physical = bool(calibrated), bool(physical) and bool(calibrated)

    @property
    def mode(self) -> str:
        """'raw' (V_f, V_b) | 'gate' (u_n, u_D in V) | 'phys' (n, D/ε₀)."""
        if not self.calibrated:
            return "raw"
        return "phys" if self.physical else "gate"

    def u(self, vf, vb):
        vf, vb = np.asarray(vf, float), np.asarray(vb, float)
        return vf + self.r * vb - self.un0, vf - self.r * vb - self.uD0

    def scale(self) -> Tuple[float, float]:
        """(s_n, s_D): u_n [V] -> n [10¹² cm⁻²], u_D [V] -> D/ε₀ [V/nm]."""
        dF = max(self.dF, 1e-6)
        return K_N * self.eps / dF, self.eps / (2.0 * dF)

    def display_xy(self, vf, vb):
        """Map coordinates of gate points in the current mode."""
        if self.mode == "raw":
            return np.asarray(vf, float), np.asarray(vb, float)
        un, uD = self.u(vf, vb)
        if self.mode == "phys":
            sN, sD = self.scale()
            return sN * un, sD * uD
        return un, uD

    def axis_labels(self) -> Tuple[str, str]:
        if self.mode == "raw":
            return "V_front (V)", "V_back (V)"
        if self.mode == "phys":
            return "n (10¹² cm⁻²)", "D/ε₀ (V/nm)"
        return "u_n = V_f + r·V_b (V)", "u_D = V_f − r·V_b (V)"

    def key(self):
        return (round(self.r, 9), round(self.un0, 9), round(self.uD0, 9), round(self.dF, 9),
                round(self.eps, 9), self.calibrated, self.physical)


class Linecut:
    """How the scan's gate points lie on the gate–gate plane, and the sweep axis that follows."""
    __slots__ = ("kind", "x", "label", "short", "info", "angle_deg")

    def __init__(self, kind, x, label, short, info, angle_deg=float("nan")):
        self.kind, self.x, self.label, self.short = kind, np.asarray(x, float), label, short
        self.info, self.angle_deg = info, float(angle_deg)


def classify_linecut(vf, vb, calib: GateCalib, tol_deg: float = 5.0) -> Linecut:
    """Constant-n cut (along u_D), constant-D cut (along u_n), or anything else (x = front gate).

    The direction is the principal axis of the points in (u_n, u_D) volts; the cut is "along an
    axis" when it is within ``tol_deg`` of it.  Points far from a line (a grid) count as "grid".
    """
    vf, vb = np.asarray(vf, float), np.asarray(vb, float)
    n = vf.size

    def _raw(kind, info, ang=float("nan")):
        if n and np.ptp(vf) <= 1e-9 and np.ptp(vb) > 1e-9:
            return Linecut(kind, vb, "Back gate V_b (V)", "V_b", info, ang)
        return Linecut(kind, vf, "Front gate V_f (V)", "V_f", info, ang)

    if n == 0:
        return Linecut("none", vf, "Front gate V_f (V)", "V_f", "no scan loaded")
    if n == 1:
        return _raw("single", "a single gate point")
    un, uD = calib.u(vf, vb)
    P = np.column_stack([un, uD])
    P0 = P - P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P0, full_matrices=False)
    if S[0] <= 1e-9:
        return _raw("single", "all scan points at the same gates")
    d = Vt[0]
    ang_D = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]))))      # angle from the u_D axis
    ang_n = 90.0 - ang_D
    if S.size > 1 and S[1] > 0.1 * S[0]:
        return _raw("grid", f"scan points span an area of the gate plane (not a line): x = gate voltage")
    sN, sD = calib.scale()
    if ang_D <= tol_deg:
        kind = "D"
        const = f"constant n: u_n = {un.mean():+.3g} ± {np.ptp(un) / 2:.2g} V  (n = {sN * un.mean():+.3g}×10¹² cm⁻²)"
        what = f"D-field scan, {ang_D:.2f}° from the u_D axis"
    elif ang_n <= tol_deg:
        kind = "n"
        const = f"constant D: u_D = {uD.mean():+.3g} ± {np.ptp(uD) / 2:.2g} V  (D/ε₀ = {sD * uD.mean():+.3g} V/nm)"
        what = f"density scan, {ang_n:.2f}° from the u_n axis"
    else:
        return _raw("oblique", f"oblique cut ({ang_D:.1f}° from u_D, {ang_n:.1f}° from u_n; tolerance "
                               f"{tol_deg:g}°): x = gate voltage", ang_D)
    info = f"{what}; {const}"
    if not calib.calibrated:
        return _raw(kind, info + "  —  calibrated axes off: x = gate voltage", ang_D)
    if kind == "D":
        if calib.physical:
            return Linecut(kind, sD * uD, "D/ε₀ (V/nm)", "D/ε₀", info, ang_D)
        return Linecut(kind, uD, "u_D = V_f − r·V_b (V)", "u_D", info, ang_D)
    if calib.physical:
        return Linecut(kind, sN * un, "n (10¹² cm⁻²)", "n", info, ang_D)
    return Linecut(kind, un, "u_n = V_f + r·V_b (V)", "u_n", info, ang_D)


# ══════════════════════════════════════════════════════════════════════
# The scan: PL frames and TRPL files on a (power × gate point) grid
# ══════════════════════════════════════════════════════════════════════

class ScanModel:
    """Points = distinct (V_f, V_b) pairs; powers = distinct powers (W) or stage angles (°)."""

    def __init__(self, scan, state: str, frames: List[PLFrame]):
        self.scan, self.state, self.folder = scan, state, scan.folder
        self.session = scan.session
        self.frames = frames
        self.has_pl = bool(scan.asc.get(state))
        self.meta = frames[0].meta if frames else {}
        # a gate never named in the file names is fixed: collapse its noisy read-back
        front_named = any(fr.vf_named for fr in frames)
        back_named = any(fr.vb_named for fr in frames)

        def _collapsed(vals) -> float:
            v = [float(x) for x in vals if x is not None and np.isfinite(x)]
            return 0.0 if not v else float(round(float(np.median(v)), 3))

        fixed_f = None if front_named else _collapsed([fr.v_front for fr in frames])
        fixed_b = None if back_named else _collapsed([fr.v_back for fr in frames])

        def _v(val, fixed):
            if fixed is not None:
                return fixed
            return 0.0 if val is None or not np.isfinite(val) else float(val)

        pts: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
        fr_pt = []
        for fr in frames:
            vf, vb = _v(fr.v_front, fixed_f), _v(fr.v_back, fixed_b)
            k = (round(vf, 3) + 0.0, round(vb, 3) + 0.0)
            pts.setdefault(k, []).append((vf, vb))
            fr_pt.append(k)
        keys = sorted(pts)
        self.vf = np.array([np.mean([a for a, _ in pts[k]]) for k in keys], float)
        self.vb = np.array([np.mean([b for _, b in pts[k]]) for k in keys], float)
        index = {k: i for i, k in enumerate(keys)}
        pkeys = sorted({_norm_key(fr.power_w) if _norm_key(fr.power_w) is not None else 0.0 for fr in frames})
        self.powers = np.array(pkeys if pkeys else [0.0], float)
        pindex = {p: i for i, p in enumerate(self.powers.tolist())}
        self.angle_mode = any(fr.angle_deg is not None for fr in frames)
        self.has_power = any(fr.power_w is not None for fr in frames)
        self.grid: Dict[Tuple[int, int], PLFrame] = {}
        for fr, k in zip(frames, fr_pt):
            p = _norm_key(fr.power_w)
            key = (pindex[0.0 if p is None else p], index[k])
            old = self.grid.get(key)
            if old is None or fr.mtime > old.mtime:
                self.grid[key] = fr
        self.trpl: Dict[Tuple[int, int], Path] = {}
        for key, fr in self.grid.items():
            path = _trpl_path_for(scan, state, fr)
            if path is not None:
                self.trpl[key] = path

    @property
    def n_points(self) -> int:
        return int(self.vf.size)

    @property
    def n_powers(self) -> int:
        return int(self.powers.size)

    def frame(self, pi: int, b: int) -> Optional[PLFrame]:
        return self.grid.get((int(pi), int(b)))

    def trpl_path(self, pi: int, b: int) -> Optional[Path]:
        return self.trpl.get((int(pi), int(b)))

    def describe(self) -> str:
        n_tr = len(self.trpl)
        text = f"{self.n_points} gate points × {self.n_powers} power(s), {len(self.grid)} PL"
        if not self.has_pl:
            text = f"{self.n_points} gate points × {self.n_powers} power(s), no PL"
        return text + f", {n_tr} TRPL"


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#   TRPL  (SPAD23 or SPAD93)  —  copied from pl_trpl_power_gate_dep_viewer_v2.py up to
#   PixelMapPlot; changed: fit_point keeps every frame's fit parameters
#   (PointDiffusion.popt; .groups is filled by fit_point_x), clips the reference into the time range and widens the automatic pre-t0
#   window by the smoothing kernel's reach; ViewSettings.smooth
#   (optional time smoothing of every pixel trace, applied in point_view after the rebin).
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

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


# ── spectral mode: the SPAD's centre line on a wavelength axis (as the SPAD93 / SPAD23 v2 GUIs) ──
#   SPAD93: a spectral line = one of the 17 columns (same x), sampled at y = 92 µm (spad93_geometry.row_weights)
#   SPAD23: a spectral line = one of the 5 rows (same y), sampled at x = 46 µm (spad23_geometry.column_weights)
#   A pixel exactly at that height is used alone, otherwise the two straddling pixels are interpolated
#   linearly (0.5 / 0.5 on the half-shifted lines); one of them excluded → the other alone; none → NaN.
#   λ-calibration (WLCAL_*.json): λ = intercept_nm + slope_nm_per_um · (line position in µm: column x / row y).
SPECTRAL_CENTER_UM = {"SPAD93": 92.0, "SPAD23": 46.0}
SPECTRAL_CROSS_PITCH_UM = 23.0
_LINE_W_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def spectral_axes(geom: SpadGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """(along-line position, cross-line position) of every pixel: SPAD93 lines are columns, SPAD23 rows."""
    return (geom.x_um, geom.y_um) if geom.name == "SPAD93" else (geom.y_um, geom.x_um)


def line_name(geom: SpadGeometry) -> str:
    return "col" if geom.name == "SPAD93" else "row"


def center_line_weights(geom: SpadGeometry, excluded=()) -> Tuple[np.ndarray, np.ndarray]:
    """(W [n_lines, n_pixels], line positions µm): ``counts @ W.T`` = the centre-line spectrum per time bin."""
    ex = frozenset(int(p) for p in excluded)
    key = (geom.name, ex)
    hit = _LINE_W_CACHE.get(key)
    if hit is not None:
        return hit
    pos_px, cross = spectral_axes(geom)
    c0 = SPECTRAL_CENTER_UM[geom.name]
    pos = np.unique(np.round(pos_px, 6))
    w = np.zeros((pos.size, geom.num_pixels))
    tol = 1e-6
    for k, p in enumerate(pos):
        ids = np.flatnonzero(np.abs(pos_px - p) < tol)
        cs = cross[ids]
        exact = ids[np.abs(cs - c0) < tol]
        if exact.size:
            q = int(exact[0])
            if q in ex:
                w[k] = np.nan
            else:
                w[k, q] = 1.0
            continue
        below, above = ids[cs < c0], ids[cs > c0]
        if not below.size or not above.size:
            w[k] = np.nan
            continue
        lo = int(below[np.argmax(cross[below])])
        hi = int(above[np.argmin(cross[above])])
        c_lo, c_hi = cross[lo], cross[hi]
        if c_hi - c_lo > SPECTRAL_CROSS_PITCH_UM + tol:
            w[k] = np.nan
            continue
        frac = (c0 - c_lo) / (c_hi - c_lo)
        lo_ok, hi_ok = lo not in ex, hi not in ex
        if lo_ok and hi_ok:
            w[k, lo], w[k, hi] = 1.0 - frac, frac
        elif lo_ok:
            w[k, lo] = 1.0
        elif hi_ok:
            w[k, hi] = 1.0
        else:
            w[k] = np.nan
    if len(_LINE_W_CACHE) > 256:
        _LINE_W_CACHE.clear()
    _LINE_W_CACHE[key] = (w, pos)
    return w, pos


def line_of_pixel(geom: SpadGeometry, p: int) -> int:
    """Spectral line (column / row index) a pixel belongs to."""
    pos_px, _ = spectral_axes(geom)
    _w, pos = center_line_weights(geom)
    return int(np.argmin(np.abs(pos - pos_px[int(p)])))


def spectral_matrix(counts: np.ndarray, bgv: Optional[np.ndarray], geom: SpadGeometry, excluded=()) -> np.ndarray:
    """Centre-line spectrum of every time bin: [bins, pixels] → [bins, lines] (background subtracted first)."""
    w, _pos = center_line_weights(geom, excluded)
    c = np.asarray(counts, dtype=np.float64)
    if bgv is not None:
        c = c - np.asarray(bgv, dtype=np.float64)[None, :]
    return c @ w.T


class WavelengthCal:
    """Linear λ-calibration of the spectral lines (WLCAL_*.json of the SPAD GUIs)."""

    def __init__(self, slope_nm_per_um: float, intercept_nm: float, device: str = "", source: str = "",
                 created: str = ""):
        self.slope, self.intercept = float(slope_nm_per_um), float(intercept_nm)
        self.device, self.source, self.created = str(device or ""), str(source), str(created or "")

    def wavelength(self, pos_um) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(pos_um, dtype=np.float64)

    def problem(self, geom: SpadGeometry) -> str:
        """Why this calibration does not fit the detector ('' = fine)."""
        if self.device and self.device.upper() != geom.name.upper():
            return f"calibration is for {self.device}, the data are {geom.name}"
        return ""

    def describe(self, geom: Optional[SpadGeometry] = None) -> str:
        text = f"{self.source}: λ = {self.intercept:.2f} nm + {self.slope:.4g} nm/µm · x"
        if geom is not None:
            lam = self.wavelength(center_line_weights(geom)[1])
            text += f"  ({lam.min():.1f}–{lam.max():.1f} nm, {abs(self.slope) * 19.92:.2f} nm per line)"
        return text + (f", created {self.created}" if self.created else "")


def wlcal_from_dict(d: dict, source: str) -> WavelengthCal:
    try:
        slope, icpt = float(d["slope_nm_per_um"]), float(d["intercept_nm"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("no slope_nm_per_um / intercept_nm") from None
    if not (np.isfinite(slope) and np.isfinite(icpt)) or slope == 0:
        raise ValueError("invalid slope / intercept")
    return WavelengthCal(slope, icpt, d.get("device", ""), source, d.get("created", ""))


def load_wlcal_json(path) -> WavelengthCal:
    with open(path, "r", encoding="utf-8") as fh:
        return wlcal_from_dict(json.load(fh), Path(path).name)


def wlcal_from_header(meta: dict) -> Optional[WavelengthCal]:
    """The calibration the measuring GUI wrote into the .asc header (a Python dict), or None."""
    import ast
    raw = (meta or {}).get("spad_wavelength_calibration")
    if not raw:
        return None
    try:
        d = ast.literal_eval(str(raw))
        return wlcal_from_dict(d, "scan header") if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None

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
    __slots__ = ("align", "t0_ref", "rebin", "smooth")

    def __init__(self, align: bool, t0_ref: Optional[float], rebin: int, smooth=None):
        self.align, self.t0_ref, self.rebin = bool(align), t0_ref, max(1, int(rebin))
        self.smooth = None if smooth is None else tuple(smooth)    # (method, p1, p2, t_ref): see smooth_traces

    def key(self):
        return (self.align, self.t0_ref if self.align else None, self.rebin, self.smooth)


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
    if vs.smooth is not None:
        c = smooth_traces(t, c, vs.smooth)
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
    __slots__ = ("view", "t_range", "ref_ns", "dark", "pre_window", "fix_offset", "excluded", "dark_ref", "dark_id")

    def __init__(self, view: ViewSettings, t_range, ref_ns, dark, pre_window, fix_offset, excluded,
                 dark_ref: Optional[np.ndarray] = None, dark_id: str = ""):
        """``dark`` = subtract the pre-t0 offset; ``dark_ref`` = a dark reference per pixel (counts per
        view bin), subtracted instead; ``dark_id`` names it (file, scale) for the cache key."""
        self.view, self.t_range, self.ref_ns = view, t_range, float(ref_ns)
        self.dark, self.pre_window, self.fix_offset = bool(dark), pre_window, bool(fix_offset)
        self.excluded = tuple(sorted(excluded))
        self.dark_ref, self.dark_id = dark_ref, (dark_id if dark_ref is not None else "")

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
    __slots__ = ("t_ns", "sigma2", "flagged", "ref_ns", "failures", "note", "popt", "groups")

    def __init__(self, t_ns, sigma2, flagged, ref_ns, failures=0, note="", popt=None, groups=None):
        self.t_ns, self.sigma2, self.flagged = t_ns, sigma2, flagged   # flagged: at a bound or low signal
        self.ref_ns, self.failures, self.note = ref_ns, failures, note
        self.popt = popt               # (frames, 6): A, x0, y0, sx, sy, offset of every fitted frame
        self.groups = groups           # (frames, 2) first / last view bin of each fitted frame; None = one bin each


def fit_point(t: np.ndarray, counts: np.ndarray, s: FitSettings, cancelled=None) -> PointDiffusion:
    """Fit All of one point: reference frame, then every later frame of the range (raises ValueError).

    The detector (pixel layout, fit pitches) follows the number of pixel columns of ``counts``.
    """
    geom = geometry_for(counts.shape[1])
    r0, r1 = range_bins(t, s.t_range)
    ref = int(np.clip(np.argmin(np.abs(t - s.ref_ns)), r0, r1))     # as the measuring GUI
    if max(r0, ref) > r1:
        raise ValueError("the time range has no frames after the reference")
    mask = s.mask(geom.num_pixels)
    if int(mask.sum()) < MIN_FIT_PIXELS_2D:
        raise ValueError(f"not enough enabled pixels (need ≥ {MIN_FIT_PIXELS_2D})")
    frames, note = np.asarray(counts[: r1 + 1], dtype=np.float64), ""
    if s.dark_ref is not None:
        if np.size(s.dark_ref) != counts.shape[1]:
            raise ValueError(f"the dark reference has {np.size(s.dark_ref)} pixels, the data {counts.shape[1]}")
        frames = frames - np.asarray(s.dark_ref, dtype=np.float64)
    elif s.dark:
        win = s.pre_window if s.pre_window is not None else pre_pulse_window(
            t, counts.sum(axis=1), margin_ns=_pre_margin_ns(s.view.smooth, t))
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
                          float(t[ref]), failures, note, np.array([f.popt for f in fits], dtype=np.float64))


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
                  progress=None, cancelled=None):
    """Counts of every point on a common time axis, and/or (``fit``) its Fit All.

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
                    pdf = fit_point(t, c, fit, cancelled=stop)
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
# ══════════════════════════════════════════════════════════════════════
#   Dual-gate viewer  —  everything below is new in this file
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

def _settings() -> QtCore.QSettings:
    return QtCore.QSettings("Automation", "PLTRPLDualGateViewer")


PLOT_TICK_PT  = 11          # axis tick numbers
PLOT_LABEL_PT = 13          # axis labels
PLOT_TITLE_PT = 12          # plot titles
PLOT_LEGEND_PT = 11


def _big_fonts(root: QtWidgets.QWidget) -> None:
    """Larger tick numbers, axis labels, titles and legends on every plot under ``root`` (also in
    exported images).  pyqtgraph keeps the label / title style, so later setLabel / setTitle calls
    without style arguments stay large."""
    tick_font = QtGui.QFont()
    tick_font.setPointSize(PLOT_TICK_PT)
    label_css = {"font-size": f"{PLOT_LABEL_PT}pt", "color": "#000000"}
    for view in root.findChildren(pg.GraphicsView):
        for it in view.scene().items():
            if isinstance(it, pg.AxisItem):
                it.setTickFont(tick_font)
                it.setLabel(it.labelText or None, it.labelUnits or None, it.labelUnitPrefix or None, **label_css)
            elif isinstance(it, pg.LabelItem):
                it.opts["size"] = f"{PLOT_TITLE_PT}pt"
                if it.text:
                    it.setText(it.text)
            elif isinstance(it, pg.LegendItem):
                it.setLabelTextSize(f"{PLOT_LEGEND_PT}pt")


def _fmt(v, spec: str = ".4g") -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return format(v, spec) if np.isfinite(v) else "--"


def _dspin(lo, hi, val, dec, step, suffix="", width=92) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(dec)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setFixedWidth(width)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


# ── optional smoothing ────────────────────────────────────────────────
#   Traces: along time only, every pixel on its own (a frame's spatial profile is never blurred
#   across pixels).  Spec = (method, p1, p2, t_ref):
#     "sg"    Savitzky–Golay, window p1 bins (odd), order p2
#     "ma"    moving average over p1 bins
#     "gauss" Gaussian, σ = p1 ns
#     "adapt" Gaussian with σ_k(t) = max(p1 ns, p2·(t − t_ref))
#   σ²(t): Gaussian-weighted local-linear regression (exact on a straight line, no edge bias).

SMOOTH_METHODS = (("Off", None), ("Savitzky–Golay", "sg"), ("Moving average", "ma"),
                  ("Gaussian", "gauss"), ("Adaptive Gaussian", "adapt"))


def _gauss_kernel(sigma_bins: float) -> np.ndarray:
    """Normalised Gaussian kernel as scipy.ndimage.gaussian_filter1d builds it (truncate 4)."""
    r = int(4.0 * sigma_bins + 0.5)
    if sigma_bins <= 0 or r == 0:
        return np.ones(1)
    k = np.arange(-r, r + 1, dtype=float)
    w = np.exp(-0.5 * (k / sigma_bins) ** 2)
    return w / w.sum()


def _sg_window(p1, p2, n: int) -> Tuple[int, int]:
    order = max(0, int(round(p2)))
    win = max(int(round(p1)) | 1, (order + 2) | 1)
    return min(win, n if n % 2 else n - 1), order


def smooth_traces(t: np.ndarray, counts: np.ndarray, spec) -> np.ndarray:
    """Counts [bins, pixels] smoothed along time (float64); see the spec above."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.shape[0]
    if spec is None or n < 3:
        return c
    method, p1, p2 = spec[0], float(spec[1]), float(spec[2])
    dt = float(np.median(np.diff(t)))
    if method == "sg":
        from scipy.signal import savgol_filter
        win, order = _sg_window(p1, p2, n)
        if win <= order or win < 3:
            return c
        return savgol_filter(c, win, order, axis=0, mode="interp")
    if method == "ma":
        from scipy.ndimage import uniform_filter1d
        size = max(1, int(round(p1)))
        return c if size == 1 else uniform_filter1d(c, size, axis=0, mode="nearest")
    if method == "gauss":
        from scipy.ndimage import gaussian_filter1d
        return c if p1 <= 0 else gaussian_filter1d(c, p1 / dt, axis=0, mode="nearest", truncate=4.0)
    if method == "adapt":
        sig = np.maximum(p1, p2 * (np.asarray(t, float) - float(spec[3]))) / dt
        out = np.empty_like(c)
        for i in range(n):
            w = _gauss_kernel(float(sig[i]))
            r = w.size // 2
            out[i] = w @ c[np.clip(np.arange(i - r, i + r + 1), 0, n - 1)]
        return out
    return c


def _pre_margin_ns(spec, t: np.ndarray) -> float:
    """Gap (ns) the automatic pre-t0 window keeps before the pulse peak: pre_pulse_window's 1 ns plus,
    with trace smoothing, twice the kernel's reach (the smoothed peak moves later by up to one reach,
    and the kernel smears signal one reach earlier).  Otherwise the smoothing smears the bright early
    spot into the "background", which is then subtracted from every frame and inflates the late σ²."""
    if spec is None or t.size < 3:
        return 1.0
    method, p1, p2 = spec[0], float(spec[1]), float(spec[2])
    dt = float(np.median(np.diff(t)))
    if method == "sg":
        reach = (_sg_window(p1, p2, t.size)[0] // 2) * dt
    elif method == "ma":
        reach = (int(round(p1)) // 2 + 1) * dt
    else:                       # Gaussian, or the adaptive one (σ_min before the reference)
        reach = 4.0 * p1 + dt
    return 1.0 + 2.0 * reach


def smooth_corr_bins(spec, t: np.ndarray, t_mid: Optional[float] = None) -> float:
    """Correlation length 1/Σc² (in bins) of the trace smoothing kernel c: how many consecutive
    frames share their noise (1 = independent).  The adaptive kernel is taken at ``t_mid``."""
    if spec is None or t.size < 3:
        return 1.0
    method, p1, p2 = spec[0], float(spec[1]), float(spec[2])
    dt = float(np.median(np.diff(t)))
    if method == "sg":
        from scipy.signal import savgol_coeffs
        win, order = _sg_window(p1, p2, t.size)
        if win <= order or win < 3:
            return 1.0
        c = savgol_coeffs(win, order)
    elif method == "ma":
        return float(max(1, int(round(p1))))
    elif method == "gauss":
        c = _gauss_kernel(p1 / dt)
    else:
        tm = float(t_mid) if t_mid is not None else float(t[t.size // 2])
        c = _gauss_kernel(max(p1, p2 * (tm - float(spec[3]))) / dt)
    return float(max(1.0, 1.0 / np.sum(np.asarray(c, float) ** 2)))


def _sg_sigma_window(window, order, n: int) -> Tuple[int, int]:
    """(odd window ≤ n, order), or window 0 when S-G is not possible."""
    order = max(0, int(order))
    win = int(window) | 1
    win = min(win, n if n % 2 else n - 1)
    return (win, order) if win >= max(3, order + 2) else (0, order)


def sg_sigma2(t: np.ndarray, s2: np.ndarray, use: np.ndarray, window: int, order: int, t_eval: np.ndarray):
    """Savitzky–Golay σ²(t): polynomial least squares (``order``) over the ``window`` frames of ``use``
    nearest each ``t_eval``, on their real times (so frames dropped as flagged are just missing); at the
    ends the window is the first / last ``window`` frames, as savgol_filter(mode="interp").
    Returns (values at t_eval, ok): ok = inside the used frames and no gap in the window larger than 3×
    the window's median spacing (frames whose spacing grows smoothly, as late groups, are fine)."""
    t_eval = np.asarray(t_eval, float)
    out, ok = np.full(t_eval.size, np.nan), np.zeros(t_eval.size, bool)
    m = np.asarray(use, bool) & np.isfinite(s2)
    tu, yu = np.asarray(t, float)[m], np.asarray(s2, float)[m]
    n = tu.size
    N, order = _sg_sigma_window(window, order, n)
    if N == 0 or t_eval.size == 0:
        return out, ok
    half = N // 2
    start = np.clip(np.searchsorted(tu, t_eval - 1e-9) - half, 0, n - N)
    idx = start[:, None] + np.arange(N)[None, :]
    x = tu[idx] - t_eval[:, None]
    span = tu[idx[:, -1]] - tu[idx[:, 0]]
    xs = x / np.where(span > 0, span, 1.0)[:, None]
    V = xs[..., None] ** np.arange(order + 1)                       # (m, N, order + 1)
    A = np.einsum("mni,mnj->mij", V, V)
    b = np.einsum("mni,mn->mi", V, yu[idx])
    out = np.linalg.solve(A, b[..., None])[..., 0][:, 0]            # value of the polynomial at t_eval
    gaps = np.diff(tu[idx], axis=1)
    ok = ((t_eval >= tu[0] - 1e-9) & (t_eval <= tu[-1] + 1e-9)
          & (gaps.max(axis=1) <= 3.0 * np.median(gaps, axis=1) + 1e-9))
    return out, ok


def sg_corr_frames(window: int, order: int, n: int) -> float:
    """Correlation length 1/Σc² (frames) of the S-G σ² smoothing."""
    from scipy.signal import savgol_coeffs
    N, order = _sg_sigma_window(window, order, n)
    if N == 0:
        return 0.0
    return float(1.0 / np.sum(savgol_coeffs(N, order) ** 2))


def fit_line_fixed(pd_: PointDiffusion, window, use_flagged: bool, s0: float):
    """fit_line with the intercept held at ``s0`` = σ²(t_ref): slope only (same dict, s0_err = 0)."""
    tr = pd_.t_ns - pd_.ref_ns
    ok = np.isfinite(pd_.sigma2) & (tr >= window[0] - 1e-9)
    if window[1] is not None:
        ok &= tr <= window[1] + 1e-9
    if not use_flagged:
        ok &= ~pd_.flagged
    n = int(ok.sum())
    if n < 2:
        return None
    x, y = tr[ok], pd_.sigma2[ok] - s0
    sxx = float((x * x).sum())
    if sxx <= 0:
        return None
    slope = float((x * y).sum() / sxx)
    s2 = float(((y - slope * x) ** 2).sum()) / (n - 1)
    k = UM2_NS_TO_CM2_S / 2.0
    return {"D": slope * k, "D_err": float(np.sqrt(s2 / sxx)) * k, "s0": float(s0), "s0_err": 0.0,
            "slope": slope, "n": n, "fixed_s0": True}


def d_line(f: PointDiffusion, window, use_flagged: bool, sg, corr_trace: float, s0_fixed: Optional[float] = None):
    """fit_line on σ²(t), or on its Savitzky–Golay curve (``sg`` = (window frames, order)); with
    ``s0_fixed`` the line goes through σ²(t_ref) = s0_fixed (slope only).  The errors are scaled by √L
    for the frames the smoothing correlates, L = √(L_trace² + L_σ²) (L = 1/Σc² per kernel).
    Returns (line dict or None, error factor, smoothed σ² at the frames or None)."""
    use = np.isfinite(f.sigma2) & (True if use_flagged else ~f.flagged)
    smoothed, L_s = None, 0.0
    n_use = int(use.sum())
    line = (lambda p: fit_line(p, window, use_flagged)) if s0_fixed is None else \
        (lambda p: fit_line_fixed(p, window, use_flagged, s0_fixed))
    if sg is not None and _sg_sigma_window(sg[0], sg[1], n_use)[0]:
        smoothed, _ok = sg_sigma2(f.t_ns, f.sigma2, use, sg[0], sg[1], f.t_ns)
        pd2 = PointDiffusion(f.t_ns, np.where(use, smoothed, np.nan), f.flagged, f.ref_ns)
        ln = line(pd2)
        L_s = sg_corr_frames(sg[0], sg[1], n_use)
    else:
        ln = line(f)
    factor = float(np.sqrt(max(1.0, np.hypot(max(corr_trace, 1.0), L_s))))
    if ln is not None and factor > 1.0:
        ln = dict(ln, D_err=ln["D_err"] * factor, s0_err=ln["s0_err"] * factor)
    return ln, factor, smoothed


# ── σ² fits with a common reference width and late-time grouping ──────

class FitSettingsX(FitSettings):
    """FitSettings plus ``ref_sigma`` = (σx, σy) µm to hold in the reference frame's fit (None = free) and
    ``late`` = (mode "fixed" | "counts", fraction of the peak signal, N) to group the late frames (None = off)."""
    __slots__ = ("ref_sigma", "late")

    def __init__(self, *args, ref_sigma=None, late=None, **kw):
        super().__init__(*args, **kw)
        self.ref_sigma = None if ref_sigma is None else (round(float(ref_sigma[0]), 6), round(float(ref_sigma[1]), 6))
        self.late = None if late is None else (str(late[0]), round(float(late[1]), 6), int(late[2]))

    def key(self):
        return super().key() + (self.ref_sigma, self.late)


def _with_ref_sigma(s: FitSettingsX, ref_sigma) -> FitSettingsX:
    return FitSettingsX(s.view, s.t_range, s.ref_ns, s.dark, s.pre_window, s.fix_offset, s.excluded, s.dark_ref,
                        s.dark_id, ref_sigma=ref_sigma, late=s.late)


def fit_gaussian_2d_fixed_sigma(x, y, z, sx: float, sy: float, pitch_x: float, pitch_y: float,
                                fix_offset: Optional[float] = None) -> Optional[FitResult]:
    """2D Gaussian with σx, σy held (the common reference width): A, x0, y0 (and the offset) fitted."""
    if _curve_fit is None or z.size < MIN_FIT_PIXELS_2D or np.allclose(z, z[0]):
        return None
    sx, sy = float(sx), float(sy)
    bg = float(np.percentile(z, 10)) if fix_offset is None else float(fix_offset)
    w = np.clip(z - bg, 0.0, None)
    tot = float(w.sum())
    x0, y0 = ((float((w * x).sum() / tot), float((w * y).sum() / tot)) if tot > 0
              else (float(np.mean(x)), float(np.mean(y))))
    zs = max(float(np.max(np.abs(z))), 1e-300)
    lo3 = [0.0, float(np.min(x) - pitch_x), float(np.min(y) - pitch_y)]
    hi3 = [np.inf, float(np.max(x) + pitch_x), float(np.max(y) + pitch_y)]
    p0 = [max(float(np.max(z) - bg), 1e-6) / zs, x0, y0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fix_offset is None:
                popt, pcov = _curve_fit(lambda xy, A, a, b, off: gauss2d(xy, A, a, b, sx, sy, off), (x, y), z / zs,
                                        p0=_clip_into(p0 + [bg / zs], lo3 + [-np.inf], hi3 + [np.inf]),
                                        bounds=(lo3 + [-np.inf], hi3 + [np.inf]), maxfev=20000)
                A, a, b, off = popt
            else:
                off = float(fix_offset) / zs
                popt, pcov = _curve_fit(lambda xy, A, a, b: gauss2d(xy, A, a, b, sx, sy, off), (x, y), z / zs,
                                        p0=_clip_into(p0, lo3, hi3), bounds=(lo3, hi3), maxfev=20000)
                A, a, b = popt
    except Exception:
        return None
    full = np.array([A * zs, a, b, sx, sy, off * zs], dtype=np.float64)
    n_free = 4 if fix_offset is None else 3
    var = float(pcov[0, 0]) * zs ** 2 if np.size(pcov) else float("nan")
    dof = int(z.size) - n_free
    snr = float("inf") if dof <= 0 else (0.0 if not np.isfinite(var) or var <= 0 else float(full[0] / np.sqrt(var)))
    at_bound = _hit_bound(full[:3], lo3, hi3)
    return FitResult(full, float(np.sqrt((sx ** 2 + sy ** 2) / 2.0)), snr, at_bound)


def late_groups(S: np.ndarray, i_start: int, i_end: int, ref: int, late) -> Tuple[List[Tuple[int, int]], int]:
    """Frames i_start…i_end as (first, last) bin ranges: one bin each until the total signal ``S`` (per bin,
    background-subtracted, enabled pixels) falls below frac × its peak (at or after ``ref``), then grouped:
    "fixed" = N bins per frame; "counts" = each frame collects at least frac × peak counts (≤ 400 bins).
    Returns (groups, threshold bin)."""
    from scipy.ndimage import uniform_filter1d
    mode, frac, N = late
    S = np.asarray(S, dtype=np.float64)
    Ssm = uniform_filter1d(S, 5, mode="nearest")
    ip = ref + int(np.argmax(Ssm[ref:i_end + 1]))
    peak = float(Ssm[ip])
    below = np.flatnonzero(Ssm[ip:i_end + 1] < frac * peak)
    ith = ip + int(below[0]) if below.size else i_end + 1
    groups = [(i, i) for i in range(i_start, min(ith, i_end + 1))]
    i = max(ith, i_start)
    n_single = len(groups)
    if mode == "fixed":
        N = max(1, int(N))
        while i <= i_end:
            j = min(i + N - 1, i_end)
            groups.append((i, j))
            i = j + 1
        if len(groups) - n_single > 1 and groups[-1][1] - groups[-1][0] + 1 < N:
            last = groups.pop()
            groups[-1] = (groups[-1][0], last[1])
    else:
        target = frac * peak
        acc = 0.0
        while i <= i_end:
            j, acc = i, S[i]
            while acc < target and j < i_end and j - i + 1 < 400:
                j += 1
                acc += S[j]
            groups.append((i, j))
            i = j + 1
        if len(groups) - n_single > 1 and acc < target:
            last = groups.pop()
            groups[-1] = (groups[-1][0], last[1])
    return groups, ith


def fit_point_x(t: np.ndarray, counts: np.ndarray, s, cancelled=None) -> PointDiffusion:
    """fit_point, plus FitSettingsX.ref_sigma (reference frame fitted with σx, σy held) and FitSettingsX.late
    (late frames grouped, each group fitted on its mean frame at its signal-weighted mean time).
    Without either it IS fit_point (identical results)."""
    ref_sigma, late = getattr(s, "ref_sigma", None), getattr(s, "late", None)
    if ref_sigma is None and late is None:
        return fit_point(t, counts, s, cancelled)
    geom = geometry_for(counts.shape[1])
    r0, r1 = range_bins(t, s.t_range)
    ref = int(np.clip(np.argmin(np.abs(t - s.ref_ns)), r0, r1))
    if max(r0, ref) > r1:
        raise ValueError("the time range has no frames after the reference")
    mask = s.mask(geom.num_pixels)
    if int(mask.sum()) < MIN_FIT_PIXELS_2D:
        raise ValueError(f"not enough enabled pixels (need ≥ {MIN_FIT_PIXELS_2D})")
    if s.dark_ref is not None and np.size(s.dark_ref) != counts.shape[1]:
        raise ValueError(f"the dark reference has {np.size(s.dark_ref)} pixels, the data {counts.shape[1]}")
    bg, bg_note = _bg_vector(t, counts, s)
    notes = ["pre-t0 window empty: no offset subtracted"] if bg_note.startswith("pre-t0 window empty") else []
    frames = np.asarray(counts[: r1 + 1], dtype=np.float64) - bg
    fix = 0.0 if s.fix_offset else None
    xm, ym = geom.x_um[mask], geom.y_um[mask]
    px, py = geom.pitch_x, geom.pitch_y
    if ref_sigma is not None:
        prev = fit_gaussian_2d_fixed_sigma(xm, ym, frames[ref][mask], ref_sigma[0], ref_sigma[1], px, py, fix)
    else:
        prev = fit_gaussian_2d(xm, ym, frames[ref][mask], px, py, fix_offset=fix)
    if prev is None:
        raise ValueError("could not fit the reference frame")
    S = frames[:, mask].sum(axis=1)
    i_start = max(r0, ref) + 1
    if late is not None and i_start <= r1:
        groups, ith = late_groups(S, i_start, r1, ref, late)
        n_late = sum(1 for g in groups if g[0] >= ith)
        if ith <= r1:
            notes.append(f"frames after {t[ith]:.1f} ns grouped ({'×' + str(late[2]) if late[0] == 'fixed' else 'equal counts'}"
                         f", {100 * late[1]:.3g} % of the peak): {n_late} groups")
        else:
            notes.append(f"the signal never drops below {100 * late[1]:.3g} % of its peak: no grouping")
    else:
        groups = [(i, i) for i in range(i_start, r1 + 1)]
    times, fits, spans, failures = [float(t[ref])], [prev], [(ref, ref)], 0
    for i0, i1 in groups:
        if cancelled is not None and cancelled():
            break
        z = frames[i0:i1 + 1][:, mask].mean(axis=0)
        fit = fit_gaussian_2d_next(xm, ym, z, prev.popt, 1.0, px, py, fix)
        if fit is None:
            fit = fit_gaussian_2d_next(xm, ym, z, prev.popt, 1.8, px, py, fix)
        if fit is None:
            failures += 1
            continue
        prev = fit
        tt, w = t[i0:i1 + 1], np.clip(S[i0:i1 + 1], 0.0, None)
        times.append(float((w * tt).sum() / w.sum()) if w.sum() > 0 else float(tt.mean()))
        fits.append(fit)
        spans.append((i0, i1))
    if ref_sigma is not None:
        notes.insert(0, f"reference width held at σx = {ref_sigma[0]:.3g}, σy = {ref_sigma[1]:.3g} µm")
    return PointDiffusion(np.array(times), np.array([f.sigma_eq ** 2 for f in fits]),
                          np.array([bool(f.at_bound) or not f.quality_ok for f in fits]), float(t[ref]), failures,
                          "; ".join(notes), np.array([f.popt for f in fits], dtype=np.float64),
                          np.array(spans, dtype=int))


# ── spectral view: 1D Gaussian width along one spectral line (as spad93 fit_all_column, method 1) ──
#   SPAD93: the pixels of one column (same x = same λ) against y; SPAD23: one row against x.  Sites are
#   23 µm apart.  σ² of a 1D cut of an isotropic spot grows as σ₀² + 2D·t, like σ_eq² of the 2D fits.
LINE_FIT_MIN_PX = 3
LINE_FIT_FREE_OFFSET_MIN_PX = 7          # fewer pixels: offset held at 0 (spad93 fit_column_profile rule)


def line_pixels(geom: SpadGeometry, line: int, excluded=()) -> Tuple[np.ndarray, np.ndarray]:
    """(pixel ids sorted by their cross position, cross positions µm) of spectral line ``line``."""
    pos_px, cross = spectral_axes(geom)
    _w, pos = center_line_weights(geom)
    ex = {int(p) for p in excluded}
    ids = np.array([p for p in np.flatnonzero(np.abs(pos_px - pos[int(line)]) < 1e-6) if int(p) not in ex], int)
    ids = ids[np.argsort(cross[ids])] if ids.size else ids
    return ids, cross[ids]


def gauss1d(x, A, x0, s, off):
    return A * np.exp(-((x - x0) ** 2) / (2.0 * s ** 2)) + off


def _fit_gauss1d(x, z, p0, lower, upper, fix_offset: Optional[float]) -> Optional[FitResult]:
    """Bounded 1D Gaussian (z scaled by max|z| for the fit); popt = A, x0, σ, offset; sigma_eq = |σ|."""
    if _curve_fit is None or z.size < LINE_FIT_MIN_PX or np.allclose(z, z[0]):
        return None
    zs = max(float(np.max(np.abs(z))), 1e-300)
    lo, hi = list(lower), list(upper)
    p = [float(p0[0]) / zs, float(p0[1]), float(p0[2]), float(p0[3]) / zs]
    lo[0], hi[0], lo[3], hi[3] = lo[0] / zs, hi[0] / zs, lo[3] / zs, hi[3] / zs
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fix_offset is None:
                popt, pcov = _curve_fit(gauss1d, x, z / zs, p0=_clip_into(p, lo, hi), bounds=(lo, hi), maxfev=20000)
                full = np.asarray(popt, float)
                n_free = 4
            else:
                off = float(fix_offset) / zs
                popt, pcov = _curve_fit(lambda xx, A, c, s: gauss1d(xx, A, c, s, off), x, z / zs,
                                        p0=_clip_into(p[:3], lo[:3], hi[:3]), bounds=(lo[:3], hi[:3]), maxfev=20000)
                full = np.append(np.asarray(popt, float), off)
                n_free = 3
    except Exception:
        return None
    at_bound = _hit_bound(full[:n_free], lo[:n_free], hi[:n_free])
    full[[0, 3]] *= zs
    dof = int(z.size) - n_free
    var = float(pcov[0, 0]) * zs ** 2 if np.size(pcov) else float("nan")
    snr = (float("nan") if dof <= 0 else                     # no degrees of freedom → flagged
           0.0 if not np.isfinite(var) or var <= 0 else float(full[0] / np.sqrt(var)))
    full[2] = abs(full[2])
    return FitResult(full, float(full[2]), snr, at_bound)


def fit_gaussian_1d_free(x, z, pitch: float, fix_offset: Optional[float]) -> Optional[FitResult]:
    """Free fit from moment start values (the reference frame); σ within 0.25–8 pitches (as spad93)."""
    bg = float(np.percentile(z, 10)) if fix_offset is None else float(fix_offset)
    w = np.clip(z - bg, 0.0, None)
    tot = float(w.sum())
    c0 = float((w * x).sum() / tot) if tot > 0 else float(x[int(np.argmax(z))])
    s0 = float(np.sqrt((w * (x - c0) ** 2).sum() / tot)) if tot > 0 else float(np.ptp(x)) / 4.0
    lower = [0.0, float(x.min()) - pitch, 0.25 * pitch, -np.inf]
    upper = [np.inf, float(x.max()) + pitch, 8.0 * pitch, np.inf]
    guess = [max(float(np.max(z)) - bg, 1e-9), c0, max(s0, 0.5 * pitch), bg]
    return _fit_gauss1d(x, z, guess, lower, upper, fix_offset)


def fit_gaussian_1d_next(x, z, prev: np.ndarray, tightness: float, pitch: float,
                         fix_offset: Optional[float]) -> Optional[FitResult]:
    """Next frame from the previous fit, small bounds around it (spad93 fit_gaussian_1d_constrained_next)."""
    A, c, s, off = (float(v) for v in prev)
    s = max(s, 1e-3)
    dA, dc = max(abs(A) * 0.18 * tightness, 1e-3), max(pitch * 0.18 * tightness, 0.8)
    ds, doff = max(s * 0.18 * tightness, 0.5), max(abs(off) * 0.25 * tightness, 0.5)
    lower = [max(0.0, A - dA), c - dc, max(1e-3, s - ds), off - doff]
    upper = [A + dA, c + dc, s + ds, off + doff]
    return _fit_gauss1d(x, z, [A, c, s, off], lower, upper, fix_offset)


def fit_line_x(t: np.ndarray, counts: np.ndarray, s, line: int, free_offset: bool = False,
               cancelled=None) -> PointDiffusion:
    """fit_point_x for one spectral line: sequential 1D Gaussian of the line's pixels against their cross
    position (reference frame free, later frames stepped from the previous one; late grouping as
    FitSettingsX.late).  The offset is 0 (after the background subtraction) unless ``free_offset`` and the
    line has ≥ 7 enabled pixels (as spad93's column fits; FitSettings.fix_offset is the 2D fits' and is
    not used).  PointDiffusion.popt = (frames, 4): A, centre (µm), σ (µm), offset; sigma2 = σ²."""
    geom = geometry_for(counts.shape[1])
    r0, r1 = range_bins(t, s.t_range)
    ref = int(np.clip(np.argmin(np.abs(t - s.ref_ns)), r0, r1))
    if max(r0, ref) > r1:
        raise ValueError("the time range has no frames after the reference")
    ids, x = line_pixels(geom, line, s.excluded)
    if ids.size < LINE_FIT_MIN_PX:
        raise ValueError(f"{line_name(geom)} {line} has {ids.size} enabled pixels (need ≥ {LINE_FIT_MIN_PX})")
    if s.dark_ref is not None and np.size(s.dark_ref) != counts.shape[1]:
        raise ValueError(f"the dark reference has {np.size(s.dark_ref)} pixels, the data {counts.shape[1]}")
    bg, bg_note = _bg_vector(t, counts, s)
    notes = ["pre-t0 window empty: no offset subtracted"] if bg_note.startswith("pre-t0 window empty") else []
    frames = np.asarray(counts[: r1 + 1][:, ids], dtype=np.float64) - bg[ids]
    free_off = bool(free_offset) and ids.size >= LINE_FIT_FREE_OFFSET_MIN_PX
    if free_offset and not free_off:
        notes.append(f"offset held at 0 ({ids.size} pixels < {LINE_FIT_FREE_OFFSET_MIN_PX})")
    fix = None if free_off else 0.0
    pitch = SPECTRAL_CROSS_PITCH_UM
    prev = fit_gaussian_1d_free(x, frames[ref], pitch, fix)
    if prev is None:
        raise ValueError("could not fit the reference frame")
    S = frames.sum(axis=1)
    i_start = max(r0, ref) + 1
    late = getattr(s, "late", None)
    if late is not None and i_start <= r1:
        groups, ith = late_groups(S, i_start, r1, ref, late)
        if ith <= r1:
            notes.append(f"frames after {t[ith]:.1f} ns grouped ({sum(1 for g in groups if g[0] >= ith)} groups)")
    else:
        groups = [(i, i) for i in range(i_start, r1 + 1)]
    times, fits, spans, failures = [float(t[ref])], [prev], [(ref, ref)], 0
    for i0, i1 in groups:
        if cancelled is not None and cancelled():
            break
        z = frames[i0:i1 + 1].mean(axis=0)
        fit = fit_gaussian_1d_next(x, z, prev.popt, 1.0, pitch, fix)
        if fit is None:
            fit = fit_gaussian_1d_next(x, z, prev.popt, 1.8, pitch, fix)
        if fit is None:
            failures += 1
            continue
        prev = fit
        tt, w = t[i0:i1 + 1], np.clip(S[i0:i1 + 1], 0.0, None)
        times.append(float((w * tt).sum() / w.sum()) if w.sum() > 0 else float(tt.mean()))
        fits.append(fit)
        spans.append((i0, i1))
    return PointDiffusion(np.array(times), np.array([f.sigma_eq ** 2 for f in fits]),
                          np.array([bool(f.at_bound) or not f.quality_ok for f in fits]), float(t[ref]), failures,
                          "; ".join(notes), np.array([f.popt for f in fits], dtype=np.float64),
                          np.array(spans, dtype=int))


def ref_width_median(store: TrplStore, paths, s: FitSettings, progress=None, cancelled=None) -> dict:
    """Free fits of every point's reference frame → median σx, σy over the unflagged ones (all if none is)."""
    rows = []
    for k, path in enumerate(paths):
        if cancelled is not None and cancelled():
            break
        try:
            t, c, _tz, _n = point_view(store, store.load(path), s.view)
            geom = geometry_for(c.shape[1])
            r0, r1 = range_bins(t, s.t_range)
            ref = int(np.clip(np.argmin(np.abs(t - s.ref_ns)), r0, r1))
            mask = s.mask(geom.num_pixels)
            bg, _ = _bg_vector(t, c, s)
            fr = fit_gaussian_2d(geom.x_um[mask], geom.y_um[mask], (c[ref].astype(np.float64) - bg)[mask],
                                 geom.pitch_x, geom.pitch_y, fix_offset=0.0 if s.fix_offset else None)
            if fr is not None:
                rows.append((abs(fr.popt[3]), abs(fr.popt[4]), bool(fr.at_bound) or not fr.quality_ok))
        except Exception:
            pass
        if progress is not None:
            progress(k + 1, len(paths))
    if not rows:
        raise ValueError("no reference frame of this sweep line could be fitted")
    a = np.array(rows, dtype=float)
    good = a[:, 2] < 0.5
    use = a[good] if good.any() else a
    q = np.percentile(use[:, :2], [25, 75], axis=0)
    return dict(sx=float(np.median(use[:, 0])), sy=float(np.median(use[:, 1])), n=len(rows), n_used=int(use.shape[0]),
                iqr_x=float(q[1, 0] - q[0, 0]), iqr_y=float(q[1, 1] - q[0, 1]))


def _bg_vector(t: np.ndarray, counts: np.ndarray, s: FitSettings) -> Tuple[np.ndarray, str]:
    """Per-pixel background the 2D fits subtract (same choice as fit_point): dark reference,
    pre-t0 level, or none."""
    n_pix = counts.shape[1]
    if s.dark_ref is not None and np.size(s.dark_ref) == n_pix:
        return np.asarray(s.dark_ref, dtype=np.float64), "dark reference"
    if s.dark and t.size > 3:
        win = s.pre_window if s.pre_window is not None else pre_pulse_window(
            t, counts.sum(axis=1), margin_ns=_pre_margin_ns(s.view.smooth, t))
        sel = np.zeros(t.size, bool) if win is None else (t >= win[0]) & (t <= win[1])
        if sel.any():
            return counts[sel].mean(axis=0).astype(np.float64), f"pre-t0 level ({win[0]:.1f}–{win[1]:.1f} ns)"
        return np.zeros(n_pix), "pre-t0 window empty: no background subtracted"
    return np.zeros(n_pix), "no background subtracted"


def _column_image(x: np.ndarray, z: np.ndarray, min_cols: int = 400):
    """Columns of ``z`` (ny, nx) drawn at their true, possibly non-uniform x: each column spans the
    midpoints to its neighbours.  Returns (image (ny, ncols), x0, span) on a fine uniform grid."""
    x = np.asarray(x, float)
    n = x.size
    if n == 1:
        half = max(abs(x[0]) * 0.05, 0.5)
        edges = np.array([x[0] - half, x[0] + half])
    else:
        mid = 0.5 * (x[1:] + x[:-1])
        edges = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
    span = float(edges[-1] - edges[0])
    w_min = float(np.min(np.diff(edges)))
    ncols = int(np.clip(np.ceil(span / max(w_min, 1e-12)) * 4, max(min_cols, n), 6000))
    g = edges[0] + (np.arange(ncols) + 0.5) * span / ncols
    idx = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, n - 1)
    return z[:, idx], float(edges[0]), span


# ══════════════════════════════════════════════════════════════════════
# Plot widgets
# ══════════════════════════════════════════════════════════════════════

class XMapPlot(QtWidgets.QWidget):
    """Colour map x (sweep, any spacing) × y (uniform: energy or time); a marker at the current x.
    Clicking a column emits its index."""
    column_clicked = QtCore.pyqtSignal(int)

    def __init__(self, cmap: str = "viridis", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.addStretch(1)
        self.exportBtn = _export_button("Export this colour map with its true x (sweep) and y values")
        self.exportBtn.setStyleSheet("padding: 1px 8px")
        self.exportBtn.clicked.connect(self._on_export)
        bar.addWidget(self.exportBtn)
        lay.addLayout(bar)
        self._export = None
        self._glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self._glw)
        self.plot = self._glw.addPlot(row=0, col=0)
        _style(self.plot)
        self.plot.showGrid(x=False, y=False)
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.gradient.setColorMap(_pg_cmap(cmap))
        self._glw.addItem(self.hist, row=0, col=1)
        self._glw.ci.layout.setColumnMaximumWidth(1, 90)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#06b6d4", width=1.5))
        self.marker.setZValue(10)
        self.plot.addItem(self.marker)
        self.msg = pg.TextItem("", color="#6b7280", anchor=(0.5, 0.5))
        self.plot.addItem(self.msg, ignoreBounds=True)
        self._x = None
        self._has_img = False
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    def show_message(self, text: str, title: str = "") -> None:
        self.img.clear()
        self._has_img = False
        self._x = None
        self._export = None
        self.exportBtn.setEnabled(False)
        self.plot.setTitle(title)
        self.msg.setText(text)
        vr = self.plot.getViewBox().viewRange()
        self.msg.setPos(0.5 * (vr[0][0] + vr[0][1]), 0.5 * (vr[1][0] + vr[1][1]))

    def set_data(self, x, y, z, *, x_label: str, y_label: str, title: str = "", levels=None,
                 y_range=None) -> None:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)
        if x.size == 0 or y.size == 0 or z.shape != (y.size, x.size) or not np.isfinite(z).any():
            self.show_message("no data", title)
            return
        self._export = (x.copy(), y.copy(), z.copy(), title or "map")
        self.exportBtn.setEnabled(True)
        if x.size > 1 and np.any(np.diff(x) <= 0):     # not strictly increasing: draw by index
            x_label = f"point # (x not monotonic: {x_label})"
            x = np.arange(x.size, dtype=float)
        self.msg.setText("")
        self._x = x
        imgz, x0, span = _column_image(x, z)
        dy = float(np.median(np.diff(y))) if y.size > 1 else 1.0
        self.img.setImage(imgz.T, autoLevels=False)
        self.img.setRect(QtCore.QRectF(x0, float(y[0]) - 0.5 * dy, span, dy * y.size))
        if levels is None:
            fin = z[np.isfinite(z)]
            lo, hi = float(np.percentile(fin, 1)), float(np.percentile(fin, 99))
            if not hi > lo:
                lo, hi = float(fin.min()), float(fin.max())
            if not hi > lo:
                hi = lo + 1.0
            levels = (lo, hi)
        self.img.setLevels(levels)
        self.hist.setLevels(*levels)
        self._has_img = True
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.setTitle(title)
        self.plot.setXRange(x0, x0 + span, padding=0.01)
        if y_range is not None:
            self.plot.setYRange(*y_range, padding=0.0)
        else:
            self.plot.setYRange(float(y[0]) - 0.5 * dy, float(y[-1]) + 0.5 * dy, padding=0.0)

    def export_csv(self, fn) -> None:
        """The map's data (not the display resampling): x = the sweep values, y = energy / time / ..."""
        x, y, z, _t = self._export
        export_matrix_csv(fn, x, y, z)

    def _on_export(self) -> None:
        if self._export is not None:
            _export_csv_dialog(self, self._export[3], self.export_csv)

    def set_marker(self, x: Optional[float]) -> None:
        self.marker.setVisible(bool(x is not None and np.isfinite(x)))     # Qt wants a Python bool
        if x is not None and np.isfinite(x):
            self.marker.setValue(float(x))

    def _on_click(self, ev) -> None:
        if self._x is None or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        xv = vb.mapSceneToView(ev.scenePos()).x()
        self.column_clicked.emit(int(np.argmin(np.abs(self._x - xv))))


class ContourPixelMap(PixelMapPlot):
    """PixelMapPlot plus the 1σ and 2σ ellipses of a fitted 2D Gaussian and its centre."""

    def __init__(self, parent=None):
        super().__init__(parent)
        pi = self.getPlotItem()
        self.c1 = pg.PlotDataItem(pen=pg.mkPen("#00e5ff", width=2.2))
        self.c2 = pg.PlotDataItem(pen=pg.mkPen("#00e5ff", width=1.8, style=QtCore.Qt.DashLine))
        self.cx = pg.ScatterPlotItem(symbol="+", size=16, pen=pg.mkPen("#00e5ff", width=2.5), brush=None)
        for it in (self.c1, self.c2, self.cx):
            it.setZValue(20)
            pi.addItem(it)

    def set_contours(self, popt) -> None:
        if popt is None:
            for it in (self.c1, self.c2):
                it.setData([], [])
            self.cx.setData([], [])
            return
        a = np.linspace(0, 2 * np.pi, 181)
        x0, y0, sx, sy = float(popt[1]), float(popt[2]), abs(float(popt[3])), abs(float(popt[4]))
        self.c1.setData(x0 + sx * np.cos(a), y0 + sy * np.sin(a))
        self.c2.setData(x0 + 2 * sx * np.cos(a), y0 + 2 * sy * np.sin(a))
        self.cx.setData([x0], [y0])


# ══════════════════════════════════════════════════════════════════════
# Peak-fit widgets: fit settings row, and a spectrum-vs-sweep map with its fitted
# intensity / center / width / local slope
# ══════════════════════════════════════════════════════════════════════

class PeakFitControls(QtWidgets.QWidget):
    """Lineshape + start values (+ optionally the fit window); saved in QSettings under ``prefix``."""
    changed = QtCore.pyqtSignal()

    def __init__(self, prefix: str, window: bool = False, offset: bool = False, parent=None):
        super().__init__(parent)
        self._prefix = prefix
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.shapeCombo = QtWidgets.QComboBox()
        self.shapeCombo.addItems(PEAK_SHAPES)
        self.shapeCombo.setToolTip(
            "Single-peak lineshape:\n"
            "• Gauss ⊗ 2 exp: Gaussian (σ) convolved with an asymmetric exponential — low-energy tail τ_lo,\n"
            "  high-energy tail τ_hi (gauss_2exp, erfcx form).  The peak maximum is not at E0 when τ_lo ≠ τ_hi.\n"
            "• Gaussian: A·exp(−(E−E0)²/2σ²)\n"
            "• Lorentzian: A·γ²/((E−E0)² + γ²), γ = half width at half maximum")
        self.lo = _dspin(0.5, 3.5, 1.28, 3, 0.005, " eV") if window else None
        self.hi = _dspin(0.5, 3.5, 1.52, 3, 0.005, " eV") if window else None
        self.w0 = _dspin(0.0002, 0.5, 0.0115, 4, 0.0005, " eV", width=100)
        self.tlo = _dspin(0.0002, 0.5, 0.021, 4, 0.001, " eV", width=100)
        self.thi = _dspin(0.0002, 0.5, 0.010, 4, 0.001, " eV", width=100)
        self.w0.setToolTip("Start value of the width: σ (Gaussian, Gauss ⊗ 2 exp) or γ (Lorentzian).\n"
                           "Start values are used for the first (brightest) spectrum of a linecut; the others\n"
                           "start from their neighbour's fit.")
        self.tlo.setToolTip("Start value of the low-energy exponential tail τ_lo (Gauss ⊗ 2 exp only)")
        self.thi.setToolTip("Start value of the high-energy exponential tail τ_hi (Gauss ⊗ 2 exp only)")
        self.offsetCheck = QtWidgets.QCheckBox("Fit offset")
        self.offsetCheck.setChecked(offset)
        self.offsetCheck.setToolTip("Also fit a constant background (needed unless the spectra are background-\n"
                                    "subtracted: the n-D map's are; the scan's PL only with tab 2's Baseline on)")
        self.wLbl = QtWidgets.QLabel("σ₀")
        self.tloLbl, self.thiLbl = QtWidgets.QLabel("τ_lo₀"), QtWidgets.QLabel("τ_hi₀")
        lay.addWidget(QtWidgets.QLabel("Lineshape"))
        lay.addWidget(self.shapeCombo)
        if window:
            for lab, w in (("Fit window", self.lo), ("to", self.hi)):
                lay.addWidget(QtWidgets.QLabel(lab))
                lay.addWidget(w)
        for lab, w in ((self.wLbl, self.w0), (self.tloLbl, self.tlo), (self.thiLbl, self.thi)):
            lay.addWidget(lab)
            lay.addWidget(w)
        lay.addWidget(self.offsetCheck)
        self._restore()
        self._update_enabled()
        self.shapeCombo.currentIndexChanged.connect(lambda _i: self._on_change())
        for w in self._spins():
            w.valueChanged.connect(lambda _v: self._on_change())
        self.offsetCheck.toggled.connect(lambda _c: self._on_change())

    def _spins(self):
        return [s for s in (self.lo, self.hi, self.w0, self.tlo, self.thi) if s is not None]

    _KEYS = (("lo", "lo"), ("hi", "hi"), ("width0", "w0"), ("tau_lo0", "tlo"), ("tau_hi0", "thi"))

    def _restore(self) -> None:
        st = _settings()
        for key, attr in self._KEYS:
            w, v = getattr(self, attr), st.value(f"{self._prefix}/{key}")
            if w is not None and v is not None:
                try:
                    w.setValue(float(v))
                except (TypeError, ValueError):
                    pass
        sh = st.value(f"{self._prefix}/shape")
        if sh in PEAK_SHAPES:
            self.shapeCombo.setCurrentText(sh)
        off = st.value(f"{self._prefix}/offset")
        if off is not None:
            self.offsetCheck.setChecked(str(off).lower() in ("true", "1"))

    def save(self) -> None:
        st = _settings()
        for key, attr in self._KEYS:
            w = getattr(self, attr)
            if w is not None:
                st.setValue(f"{self._prefix}/{key}", float(w.value()))
        st.setValue(f"{self._prefix}/shape", self.shapeCombo.currentText())
        st.setValue(f"{self._prefix}/offset", self.offsetCheck.isChecked())

    def _update_enabled(self) -> None:
        sh = self.shapeCombo.currentText()
        self.wLbl.setText("γ₀" if sh == SHAPE_LOR else "σ₀")
        for w in (self.tlo, self.thi, self.tloLbl, self.thiLbl):
            w.setEnabled(sh == SHAPE_G2E)

    def _on_change(self) -> None:
        self._update_enabled()
        self.save()
        self.changed.emit()

    def opts(self, lo: Optional[float] = None, hi: Optional[float] = None) -> PeakFitOpts:
        if lo is None:
            lo = self.lo.value() if self.lo is not None else 1.28
        if hi is None:
            hi = self.hi.value() if self.hi is not None else 1.52
        return PeakFitOpts(self.shapeCombo.currentText(), lo, hi, self.w0.value(), self.tlo.value(),
                           self.thi.value(), self.offsetCheck.isChecked())


SLOPE_METHODS = ("Smoothing spline", "Local cubic", "Local line")
# (a local parabola gives the same slope as a line at the centre of a symmetric window, so the local
#  polynomial option is cubic — Savitzky–Golay order 3 — which removes the line's curvature bias)


def _axis_unit(label: str) -> str:
    """Unit of an axis label 'name (unit)', bracketed when it has a slash (for 'meV/<unit>')."""
    m = re.search(r"\(([^()]*)\)\s*$", label or "")
    u = m.group(1) if m else "unit"
    return f"({u})" if "/" in u else u


class LinecutFitView(QtWidgets.QWidget):
    """Spectra along a sweep (x = n, D, V or power; y = photon energy) with a single-peak fit of every
    column ("Fit linecut"): the fitted centers drawn on the map, and intensity, center, width and the
    local slope dE/dx versus x below it (x-linked).  Right: the selected column's spectrum + fit.

    When x is (proportional to) D/ε₀, the slope is shown as the out-of-plane dipole moment
    p = −ε_⊥·dE/d(D/ε₀)  (E = E0 − p·D/(ε₀ε_⊥); eV per V/nm = e·nm)."""
    column_clicked = QtCore.pyqtSignal(int)
    slider_changed = QtCore.pyqtSignal(int)

    def __init__(self, run, opts_fn, *, slider: bool = False, color: str = "#06b6d4", key: str = "linecut",
                 parent=None):
        super().__init__(parent)
        self._run, self._opts_fn, self._key = run, opts_fn, key
        self._x = self._xd = self._e = self._Z = None
        self._xlabel, self._title, self._dip = "", "", None
        self._fits: Optional[List[Optional[dict]]] = None
        self._fits_key = None
        self._job = None
        self._token = 0
        self._sel: Optional[int] = None
        self._single: Optional[Tuple[int, tuple, Optional[dict]]] = None
        self._q = None
        self._build(slider, color)

    # ── UI ────────────────────────────────────────────────────────────
    def _build(self, slider: bool, color: str) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(3)
        self.slider = None
        if slider:
            r0 = QtWidgets.QHBoxLayout()
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, ND_CUT_N - 1)
            self.slider.setValue(ND_CUT_N // 2)
            self.sliderLbl = QtWidgets.QLabel("—")
            self.sliderLbl.setStyleSheet(f"color:{color}; font-weight:bold; font-size:11pt")
            self.sliderLbl.setMinimumWidth(330)
            r0.addWidget(QtWidgets.QLabel("Cut at"))
            r0.addWidget(self.slider, 1)
            r0.addWidget(self.sliderLbl)
            v.addLayout(r0)
            self.slider.valueChanged.connect(self.slider_changed.emit)
        r1 = QtWidgets.QHBoxLayout()
        self.fitBtn = QtWidgets.QPushButton("Fit linecut")
        self.fitBtn.setToolTip("Fit every spectrum of this linecut (lineshape, window and start values above)")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.cancelBtn.setEnabled(False)
        self.saveBtn = QtWidgets.QPushButton("Save CSV…")
        self.saveBtn.setEnabled(False)
        self.intCombo = QtWidgets.QComboBox()
        self.intCombo.addItems(["Area", "Peak height"])
        self.cenCombo = QtWidgets.QComboBox()
        self.cenCombo.addItems(["Peak max", "E0"])
        self.cenCombo.setToolTip("Peak max = energy of the fitted curve's maximum;  E0 = the lineshape's E0 parameter\n"
                                 "(the same for Gaussian / Lorentzian; for Gauss ⊗ 2 exp they differ when τ_lo ≠ τ_hi)")
        self.widCombo = QtWidgets.QComboBox()
        self.widCombo.addItems(["FWHM", "σ / γ"])
        self.slopeMethod = QtWidgets.QComboBox()
        self.slopeMethod.addItems(SLOPE_METHODS)
        self.slopeMethod.setToolTip(
            "How the slope dE/dx of the center is taken:\n"
            "• Smoothing spline: cubic spline through the centers (weighted by their fit errors), as smooth\n"
            "  as cross-validation (GCV) says the noise needs, differentiated analytically; ±1σ by exact\n"
            "  error propagation.  'smoothing ×' < 1 = more local (noisier), > 1 = smoother.\n"
            "  A real jump of the center (another state) shows up as a slope spike.\n"
            "• Local cubic: cubic over the N nearest points (Savitzky–Golay order 3), slope at the point;\n"
            "  less curvature bias than the line for the same N, but noisier (N ≥ 5).\n"
            "• Local line: straight line over the N nearest points.\n"
            "On the n-D map linecuts the points are interpolated between measured gate points, so nothing is\n"
            "more local than the measured grid spacing along the cut.")
        self.slopeParLbl = QtWidgets.QLabel("")
        self.slopeSpin = QtWidgets.QSpinBox()
        self.slopeSpin.setRange(3, 101)
        self.slopeSpin.setValue(5)
        self.slopeSpin.setFixedWidth(60)
        self.slopeSpin.setKeyboardTracking(False)
        self.slopeSpin.setToolTip("Local fits: number of neighbouring fitted points")
        self.smoothFac = _dspin(0.001, 1000.0, 1.0, 3, 0.1, width=80)
        self.smoothFac.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.smoothFac.setToolTip("Smoothing strength relative to the cross-validated one (1 = GCV choice;\n"
                                  "< 1 = more local / noisier, > 1 = smoother)")
        self.epsSpin = _dspin(0.1, 100.0, 1.0, 2, 0.1, width=70)
        self.epsSpin.setToolTip("Out-of-plane dielectric constant ε_⊥ of the layer, for p = −ε_⊥·dE/d(D/ε₀)\n"
                                "(ε_⊥ = 1: p/ε_⊥ is shown).  Only used when the sweep axis is D.")
        for w in (self.fitBtn, self.cancelBtn, self.saveBtn):
            r1.addWidget(w)
        r1.addSpacing(10)
        for lab, w in (("Intensity", self.intCombo), ("Center", self.cenCombo), ("Width", self.widCombo),
                       ("Slope", self.slopeMethod)):
            r1.addWidget(QtWidgets.QLabel(lab))
            r1.addWidget(w)
        for w in (self.slopeParLbl, self.slopeSpin, self.smoothFac, QtWidgets.QLabel("ε_⊥"), self.epsSpin):
            r1.addWidget(w)
        r1.addStretch(1)
        v.addLayout(r1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#555")
        v.addWidget(self.status)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.map = XMapPlot("magma")
        self.map.plot.getAxis("left").setWidth(70)
        self._cenLine = pg.PlotDataItem(pen=pg.mkPen("#22d3ee", width=1.5))
        self._cenPts = pg.ScatterPlotItem(size=6, pen=pg.mkPen("k", width=0.8), brush=pg.mkBrush("#22d3ee"))
        self._winLines = [pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("w", width=1, style=QtCore.Qt.DashLine))
                          for _ in range(2)]
        for it in (self._cenLine, self._cenPts, *self._winLines):
            it.setZValue(12)
            self.map.plot.addItem(it)
        left.addWidget(self.map)
        self.glw = pg.GraphicsLayoutWidget()
        self.pInt = self.glw.addPlot(row=0, col=0)
        self.pCen = self.glw.addPlot(row=1, col=0)
        self.pWid = self.glw.addPlot(row=2, col=0)
        self.pSlope = self.glw.addPlot(row=3, col=0)
        self._curves, self._vlines = [], []
        colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
        for p, c in zip((self.pInt, self.pCen, self.pWid, self.pSlope), colors):
            _style(p)
            p.getAxis("left").setWidth(70)
            p.setXLink(self.map.plot)
            if p is not self.pSlope:
                p.getAxis("bottom").setStyle(showValues=False)
            cv = p.plot(pen=pg.mkPen(c, width=1.5), symbol="o", symbolSize=5, symbolBrush=c, symbolPen=None)
            self._curves.append(cv)
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#06b6d4", width=1.2))
            vl.setVisible(False)
            p.addItem(vl, ignoreBounds=True)
            self._vlines.append(vl)
        self._slopeErr = pg.ErrorBarItem(pen=pg.mkPen("#9467bd", width=1))
        self.pSlope.addItem(self._slopeErr)
        self.pSlope.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("#999", width=1)), ignoreBounds=True)
        self._region = pg.LinearRegionItem(brush=pg.mkBrush(255, 179, 0, 45))
        self._region.setZValue(-5)
        self.pCen.addItem(self._region, ignoreBounds=True)
        self._regionFit = pg.PlotDataItem(pen=pg.mkPen("#f59e0b", width=2.2, style=QtCore.Qt.DashLine))
        self.pCen.addItem(self._regionFit)
        self._splineCurve = pg.PlotDataItem(pen=pg.mkPen("#7c3aed", width=1.6))
        self.pCen.addItem(self._splineCurve)
        self._region_set = False
        left.addWidget(self.glw)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 4)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.specPlot = pg.PlotWidget()
        _style(self.specPlot.getPlotItem(), "PL (counts)", "Photon energy (eV)")
        self.specPlot.addLegend(offset=(-10, 10))
        self._cData = self.specPlot.plot(pen=pg.mkPen("#1f77b4", width=1.6), name="data")
        self._cFit = self.specPlot.plot(pen=pg.mkPen("#e08a00", width=2.0, style=QtCore.Qt.DashLine), name="fit")
        self._specWin = pg.LinearRegionItem(brush=pg.mkBrush(255, 179, 0, 40), movable=False)
        self._specWin.setZValue(-10)
        self.specPlot.addItem(self._specWin)
        rv.addWidget(self.specPlot, 1)
        self.readout = QtWidgets.QLabel("Click a column of the map (or a point of the plots below it).")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.readout.setStyleSheet("font-size:10pt")
        rv.addWidget(self.readout)
        self.slopeLbl = QtWidgets.QLabel("")
        self.slopeLbl.setWordWrap(True)
        self.slopeLbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.slopeLbl.setStyleSheet("color:#b45309; font-size:10pt; font-weight:bold")
        rv.addWidget(self.slopeLbl)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split, 1)

        self.fitBtn.clicked.connect(self._on_fit)
        self.cancelBtn.clicked.connect(self._on_cancel)
        self.saveBtn.clicked.connect(self._on_save)
        for c in (self.intCombo, self.cenCombo, self.widCombo):
            c.currentIndexChanged.connect(lambda _i: self._draw_fits())
        st = _settings()
        m = st.value(f"{self._key}/slope_method")
        if m in SLOPE_METHODS:
            self.slopeMethod.setCurrentText(m)
        for key, w in (("slope_window", self.slopeSpin), ("slope_smooth", self.smoothFac)):
            val = st.value(f"{self._key}/{key}")
            if val is not None:
                try:
                    w.setValue(type(w.value())(float(val)))
                except (TypeError, ValueError):
                    pass
        self._update_slope_controls()
        self.slopeMethod.currentIndexChanged.connect(lambda _i: self._on_slope_setting())
        self.slopeSpin.valueChanged.connect(lambda _v: self._on_slope_setting())
        self.smoothFac.valueChanged.connect(lambda _v: self._on_slope_setting())
        self.epsSpin.valueChanged.connect(lambda _v: self._draw_fits())
        self._region.sigRegionChangeFinished.connect(lambda _r: self._update_region_fit())
        self.map.column_clicked.connect(self._on_column)
        self.glw.scene().sigMouseClicked.connect(self._on_plots_click)

    # ── public ────────────────────────────────────────────────────────
    def set_slider(self, idx: int, text: str) -> None:
        if self.slider is None:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(int(idx))
        self.slider.blockSignals(False)
        self.sliderLbl.setText(text)

    def has_data(self) -> bool:
        return self._Z is not None

    def show_message(self, text: str, title: str = "") -> None:
        self._token += 1
        self._x = self._xd = self._e = self._Z = None
        self._fits, self._sel, self._single = None, None, None
        self.map.show_message(text, title)
        self._clear_fit_items()
        self._cData.setData([], [])
        self._cFit.setData([], [])
        self.readout.setText(text)
        self.saveBtn.setEnabled(False)

    def set_spectra(self, x, e, Z, *, x_label: str, title: str, dip_scale: Optional[float] = None) -> None:
        """Columns Z[:, k] (counts on the ascending energy axis e, eV) at sweep positions x[k].
        Fits are kept when the data are the same as before."""
        x, e, Z = np.asarray(x, float), np.asarray(e, float), np.asarray(Z, float)
        same = (self._Z is not None and Z.shape == self._Z.shape and np.array_equal(x, self._x)
                and np.array_equal(e, self._e) and np.array_equal(Z, self._Z, equal_nan=True))
        if not same:
            self._token += 1
            self._fits, self._single = None, None
            self._x, self._e, self._Z = x, e, Z
            if self._sel is not None and self._sel >= x.size:
                self._sel = None
            self._region_set = False
        self._xlabel, self._title, self._dip = x_label, title, dip_scale
        self.refresh()

    def refresh(self) -> None:
        """Redraw everything; fits made with other settings are dropped."""
        if self._Z is None:
            return
        if self._fits is not None and self._fits_key != self._opts_fn().key():
            self._fits = None
            self.status.setText("Fit settings changed: press Fit linecut again.")
        self._draw_map()
        self._draw_fits()
        if self._sel is not None:
            self.select_column(self._sel)

    def select_column(self, k: Optional[int]) -> None:
        """Mark column k and show its spectrum + fit (no signal)."""
        if self._Z is None or k is None or not (0 <= k < self._Z.shape[1]):
            return
        self._sel = int(k)
        xd = self._xd
        xv = float(xd[k]) if xd is not None else None
        self.map.set_marker(xv)
        for vl in self._vlines:
            vl.setVisible(xv is not None)
            if xv is not None:
                vl.setValue(xv)
        opts = self._opts_fn()
        y = self._Z[:, k]
        self._cData.setData(self._e, np.where(np.isfinite(y), y, np.nan), connect="finite")
        self._specWin.setRegion((opts.lo, opts.hi))
        res, src = None, ""
        if self._fits is not None:
            res, src = self._fits[k], "linecut fit"
        else:
            key = (k, opts.key(), self._token)
            if self._single is None or self._single[:2] != key[:2] or self._single[3] != self._token:
                self._single = (k, opts.key(), fit_peak(self._e, y, opts), self._token)
            res, src = self._single[2], "single fit (not from Fit linecut)"
        if res is not None:
            ef = np.linspace(opts.lo, opts.hi, 600)
            self._cFit.setData(ef, peak_curve(res, ef))
        else:
            self._cFit.setData([], [])
        span = opts.hi - opts.lo
        self.specPlot.setXRange(opts.lo - 0.15 * span, opts.hi + 0.15 * span, padding=0)
        win = (self._e >= opts.lo - 0.15 * span) & (self._e <= opts.hi + 0.15 * span) & np.isfinite(y)
        if win.any():
            lo_y, hi_y = float(np.min(y[win])), float(np.max(y[win]))
            pad = 0.06 * (hi_y - lo_y or 1.0)
            self.specPlot.setYRange(lo_y - pad, hi_y + pad, padding=0)
        self.specPlot.setTitle(f"{self._xlabel.split(' (')[0]} = {_fmt(self._x[k], '.4g')}")
        head = f"<b>{self._xlabel.split(' (')[0]} = {_fmt(self._x[k], '.4g')}</b>  ({src})<br>"
        self.readout.setText(head + _peak_html(res))

    # ── drawing ───────────────────────────────────────────────────────
    def _draw_map(self) -> None:
        opts = self._opts_fn()
        eu, Zu = _uniform_rows(self._e, self._Z)
        if not np.isfinite(Zu).any():
            self.map.show_message("no data", self._title)
            self._xd = None
            return
        win = (eu >= opts.lo) & (eu <= opts.hi)
        fin = Zu[win][np.isfinite(Zu[win])] if win.any() else Zu[np.isfinite(Zu)]
        levels = None
        if fin.size:
            lo, hi = float(np.percentile(fin, 1)), float(np.percentile(fin, 99.5))
            levels = (lo, hi if hi > lo else lo + 1.0)
        span = opts.hi - opts.lo
        yr = (max(float(eu[0]), opts.lo - 0.25 * span), min(float(eu[-1]), opts.hi + 0.25 * span))
        self.map.set_data(self._x, eu, Zu, x_label=self._xlabel, y_label="Photon energy (eV)",
                          title=self._title, levels=levels, y_range=yr if yr[1] > yr[0] else None)
        self._xd = self.map._x
        for ln, val in zip(self._winLines, (opts.lo, opts.hi)):
            ln.setValue(val)
        self.pSlope.setLabel("bottom", self.map.plot.getAxis("bottom").labelText)

    def _clear_fit_items(self) -> None:
        for cv in self._curves:
            cv.setData([], [])
        self._cenLine.setData([], [])
        self._cenPts.setData([], [])
        self._slopeErr.setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
        self._regionFit.setData([], [])
        self._splineCurve.setData([], [])
        self._region.setVisible(False)
        self.slopeLbl.setText("")
        self._q = None

    def _quantities(self):
        """(intensity, center (eV), width (meV), center error (eV; E0's, also used for the peak max))."""
        f = self._fits

        def col(key):
            return np.array([r[key] if r is not None else np.nan for r in f], float)
        inten = col("area") if self.intCombo.currentIndex() == 0 else col("height")
        cen = col("E_peak") if self.cenCombo.currentIndex() == 0 else col("E0")
        wid = (col("fwhm") if self.widCombo.currentIndex() == 0 else col("width")) * 1e3
        return inten, cen, wid, col("E0_err")

    def _update_slope_controls(self) -> None:
        spline = self.slopeMethod.currentIndex() == 0
        self.slopeParLbl.setText("smoothing ×" if spline else "points")
        self.smoothFac.setVisible(spline)
        self.slopeSpin.setVisible(not spline)

    def _on_slope_setting(self) -> None:
        self._update_slope_controls()
        st = _settings()
        st.setValue(f"{self._key}/slope_method", self.slopeMethod.currentText())
        st.setValue(f"{self._key}/slope_window", int(self.slopeSpin.value()))
        st.setValue(f"{self._key}/slope_smooth", float(self.smoothFac.value()))
        self._draw_fits()

    def _slope(self, xd, cen, cen_err):
        """(dE/dx, its 1σ, smoothed center or None, description) with the chosen method."""
        m = self.slopeMethod.currentIndex()
        if m == 0:
            s, se, f, info = spline_slope(xd, cen, cen_err, self.smoothFac.value())
            fac = self.smoothFac.value()
            desc = (f"smoothing spline (GCV{'' if fac == 1 else f' ×{fac:g}'}, "
                    f"{_fmt(info['dof'], '.1f')} eff. parameters)")
            return s, se, f, desc
        order = 3 if m == 1 else 1
        n = max(self.slopeSpin.value(), order + 2)
        s, se = local_slope(xd, cen, n, order)
        return s, se, None, f"local {'cubic' if order == 3 else 'line'}, {n} points"

    def _slope_units(self):
        """(factor applied to dE/dx [eV per x unit], y label, short) for the slope plot."""
        unit = _axis_unit(self._xlabel)
        by_index = self._xd is not None and self._x is not None and not np.array_equal(self._xd, self._x)
        if self._dip and not by_index:
            eps = float(self.epsSpin.value())
            return -eps / self._dip, f"p (e·nm)  [= −ε_⊥·dE/d(D/ε₀), ε_⊥ = {eps:g}]", "p"
        return 1e3, f"dE/dx (meV / {unit if not by_index else 'point'})", "dE/dx"

    def _draw_fits(self) -> None:
        self._clear_fit_items()
        if self._fits is None or self._xd is None:
            self.saveBtn.setEnabled(False)
            for p in (self.pInt, self.pCen, self.pWid, self.pSlope):
                p.setTitle("")
            return
        self.saveBtn.setEnabled(True)
        xd = self._xd
        inten, cen, wid, cen_err = self._quantities()
        ok = np.isfinite(cen)
        self._cenLine.setData(xd[ok], cen[ok])
        self._cenPts.setData(xd[ok], cen[ok])
        s, se, smooth, sdesc = self._slope(xd, cen, cen_err)
        if smooth is not None:
            m = np.isfinite(smooth)
            self._splineCurve.setData(xd[m], smooth[m])
        fac, slab, _short = self._slope_units()
        sv, sev = s * fac, se * abs(fac)
        for cv, yv in zip(self._curves, (inten, cen, wid, sv)):
            m = np.isfinite(yv)
            cv.setData(xd[m], yv[m])
        m = np.isfinite(sv) & np.isfinite(sev)
        self._slopeErr.setData(x=xd[m], y=sv[m], height=2 * sev[m])
        fin = sv[np.isfinite(sv)]
        if fin.size:                              # scale to the slopes, not to the (wide) end error bars
            lo, hi = (float(v) for v in np.percentile(fin, [2, 98]))
            pad = 0.15 * (hi - lo) or 0.1 * abs(hi) or 1.0
            self.pSlope.setYRange(lo - pad, hi + pad, padding=0)
        for p, t in ((self.pInt, "Intensity: " + ("area (counts·eV)" if self.intCombo.currentIndex() == 0
                                                  else "peak height (counts)")),
                     (self.pCen, "Center: " + ("peak max" if self.cenCombo.currentIndex() == 0 else "E0") + " (eV)"),
                     (self.pWid, "Width: " + ("FWHM" if self.widCombo.currentIndex() == 0 else "σ / γ") + " (meV)"),
                     (self.pSlope, f"Slope, {sdesc}: {slab}  (±1σ)")):
            p.setTitle(t, size="10pt")
        self._q = (xd, cen, s, se)
        if ok.sum() >= 2:
            self._region.setVisible(True)
            lo, hi = float(np.nanmin(xd[ok])), float(np.nanmax(xd[ok]))
            a, b = self._region.getRegion()
            if not self._region_set or b < lo or a > hi:
                self._region.blockSignals(True)
                self._region.setRegion((lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)))
                self._region.blockSignals(False)
                self._region_set = True
            self._update_region_fit()

    def _update_region_fit(self) -> None:
        if self._q is None:
            return
        xd, cen, _s, _se = self._q
        a, b = self._region.getRegion()
        sel = np.isfinite(cen) & (xd >= a) & (xd <= b)
        if sel.sum() < 2:
            self._regionFit.setData([], [])
            self.slopeLbl.setText("Linear fit: fewer than 2 fitted points in the shaded range.")
            return
        slope, icpt, err = _linfit(xd[sel], cen[sel])
        self._regionFit.setData([a, b], [icpt + slope * a, icpt + slope * b])
        unit = _axis_unit(self._xlabel)
        txt = (f"Linear fit of the center over {_fmt(a, '.3g')} … {_fmt(b, '.3g')} ({int(sel.sum())} points): "
               f"dE/dx = {_fmt(slope * 1e3, '.4g')} ± {_fmt(err * 1e3, '.2g')} meV/{unit}")
        fac, _l, short = self._slope_units()
        if short == "p":
            eps = float(self.epsSpin.value())
            p, pe = slope * fac, err * abs(fac)
            txt += (f"<br>→ dipole p = −ε_⊥·dE/d(D/ε₀) = {_fmt(p, '.4g')} ± {_fmt(pe, '.2g')} e·nm "
                    f"= {_fmt(p * E_NM_DEBYE, '.4g')} ± {_fmt(pe * E_NM_DEBYE, '.2g')} D   (ε_⊥ = {eps:g})")
        self.slopeLbl.setText(txt)

    # ── interaction ───────────────────────────────────────────────────
    def _on_column(self, k: int) -> None:
        self.select_column(k)
        self.column_clicked.emit(int(k))

    def _on_plots_click(self, ev) -> None:
        if self._xd is None or ev.button() != QtCore.Qt.LeftButton:
            return
        for p in (self.pInt, self.pCen, self.pWid, self.pSlope):
            vb = p.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                xv = vb.mapSceneToView(ev.scenePos()).x()
                self._on_column(int(np.argmin(np.abs(self._xd - xv))))
                return

    def _on_fit(self) -> None:
        if self._Z is None or self._job is not None:
            return
        if _curve_fit is None:
            self.status.setText("scipy is needed for the fits")
            return
        opts = self._opts_fn()
        e, Z, token = self._e.copy(), self._Z.copy(), self._token
        t0 = time.monotonic()

        def work(progress, cancelled):
            return fit_spectra_line(e, Z, opts, progress, cancelled)

        def done(res, err):
            self._job = None
            self.fitBtn.setEnabled(True)
            self.cancelBtn.setEnabled(False)
            if err:
                self.status.setText(f"Fit {'cancelled' if err == 'cancelled' else 'failed: ' + err}")
                return
            if token != self._token or opts.key() != self._opts_fn().key():
                self.status.setText("The data or fit settings changed during the fit: press Fit linecut again.")
                return
            self._fits, self._fits_key = res, opts.key()
            n_ok = sum(r is not None for r in res)
            self.status.setText(f"{n_ok}/{len(res)} spectra fitted ({opts.describe()}) in "
                                f"{time.monotonic() - t0:.1f} s.  Drag the shaded range on the center plot "
                                "for a straight-line fit.")
            self._draw_fits()
            if self._sel is not None:
                self.select_column(self._sel)

        self.fitBtn.setEnabled(False)
        self.cancelBtn.setEnabled(True)
        self.status.setText("Fitting …")
        self._job = self._run(work, done, lambda d, n: self.status.setText(f"Fitting {d}/{n} …"))

    def _on_cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()

    def _on_save(self) -> None:
        if self._fits is None:
            return
        st = _settings()
        start = str(st.value(f"{self._key}/csv_dir") or "")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save linecut fit", str(Path(start) / "linecut_fit.csv"),
                                                      "CSV (*.csv)")
        if not fn:
            return
        st.setValue(f"{self._key}/csv_dir", str(Path(fn).parent))
        try:
            self.save_csv(fn)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save linecut fit", str(exc))
            return
        self.status.setText(f"Saved {fn}")

    def save_csv(self, fn) -> None:
        f = self._fits
        keys = ("E_peak", "E0", "E0_err", "height", "area", "fwhm", "width", "width_err", "tau_lo", "tau_lo_err",
                "tau_hi", "tau_hi_err", "A", "A_err", "offset", "r2")
        rows = np.array([[r[k] if r is not None else np.nan for k in keys] for r in f], float)
        _i, cen, _w, cen_err = self._quantities()
        s, se, _f, sdesc = self._slope(self._xd, cen, cen_err)
        fac, _l, short = self._slope_units()
        cols = [self._x[:, None], rows, (s * 1e3)[:, None], (se * 1e3)[:, None]]
        names = ["x"] + list(keys) + ["dEdx_meV_per_x", "dEdx_err"]
        if short == "p":
            cols += [(s * fac)[:, None], (se * abs(fac))[:, None]]
            names += ["dipole_e_nm", "dipole_err_e_nm"]
        opts = self._opts_fn()
        head = (f"{self._title}\nx = {self._xlabel}\nfit: {opts.describe()}; energies in eV (width, fwhm, tau too); "
                f"center for the slope = {self.cenCombo.currentText()}; slope: {sdesc}"
                + (f"; dipole with eps_perp = {self.epsSpin.value():g}" if short == "p" else "")
                + "\n" + ",".join(names))
        np.savetxt(fn, np.hstack(cols), delimiter=",", header=head, comments="# ", fmt="%.8g", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# Tab 1: n-D map + gate calibration
# ══════════════════════════════════════════════════════════════════════

ND_QUANTITIES = ["Integrated intensity (window)", "Peak height (max)", "Peak energy (argmax)",
                 "Centroid energy", "Linewidth (2nd moment)"]
_ND_ENERGY_Q = {"Peak energy (argmax)", "Centroid energy"}
ND_CUT_N = 81                 # positions of a linecut, and of its slider (as WF_NGRID in the gate grids viewer)
_ND_CAL_DIR = HERE / "gate grids viewer" / "calibration"


class NDMapTab(QtWidgets.QWidget):
    calib_changed = QtCore.pyqtSignal()
    point_clicked = QtCore.pyqtSignal(int)        # base index of a scan point
    status = QtCore.pyqtSignal(str)

    def __init__(self, run, parent=None):
        super().__init__(parent)
        self._run = run
        self.nd: Optional[dict] = None
        self.nd_source: Optional[str] = None
        self._nd_arg: Optional[str] = None
        self.spec_bs = None
        self.sel = (0, 0)
        self._sync = False
        self._load_job = None
        self.cal_path: Optional[str] = None
        self._scan = (np.zeros(0), np.zeros(0), np.zeros(0, int), None)
        self._overlay_fit = False
        self._interp = None                       # (LinearNDInterpolator of spec_bs over (u_n, u_D), ranges)
        self._cut_idx = {"n": ND_CUT_N // 2, "D": ND_CUT_N // 2}
        self._cut_s: Dict[str, Optional[np.ndarray]] = {"n": None, "D": None}    # swept u of the drawn columns
        self._cut_ends: Dict[str, Optional[np.ndarray]] = {"n": None, "D": None}
        self._build_ui()
        self._restore_settings()
        self._wire()
        self._update_phys_info()
        self._apply_axes()
        self._refresh_map()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        box = QtWidgets.QGroupBox("Gate calibration and n-D map")
        g = QtWidgets.QVBoxLayout(box)
        g.setSpacing(3)

        r1 = QtWidgets.QHBoxLayout()
        self.loadCalBtn = QtWidgets.QPushButton("Load calibration…")
        self.loadCalBtn.setToolTip("A gtd1_calibration.json (gate grids viewer): sets r")
        self.calInfo = QtWidgets.QLabel(f"r = {R_DEFAULT:.3f}  (default; no calibration file loaded)")
        self.calInfo.setStyleSheet("color:#1a5aa0")
        r1.addWidget(self.loadCalBtn)
        r1.addWidget(self.calInfo, 1)
        self.loadNdFileBtn = QtWidgets.QPushButton("Load n-D map (.npz)…")
        self.loadNdFileBtn.setToolTip("A dual_gate_speccube_gtd1.npz cache (gate grids viewer)")
        self.loadNdDirBtn = QtWidgets.QPushButton("Load n-D map folder…")
        self.loadNdDirBtn.setToolTip("A folder of PL_manual_Vf<>_Vb<>.asc frames: its cache is loaded, "
                                     "updated or built (and saved next to the data)")
        self.reloadNdBtn = QtWidgets.QPushButton("Reload")
        self.reloadNdBtn.setToolTip("Re-read the n-D map (a folder: only new frames are read; "
                                    "Shift+click = rebuild the cache)")
        for w in (self.loadNdFileBtn, self.loadNdDirBtn, self.reloadNdBtn):
            r1.addWidget(w)
        g.addLayout(r1)
        self.ndInfo = QtWidgets.QLabel("No n-D map loaded.")
        self.ndInfo.setStyleSheet("color:#555")
        g.addWidget(self.ndInfo)

        r3 = QtWidgets.QHBoxLayout()
        self.rSpin = _dspin(0.05, 5.0, R_DEFAULT, 4, 0.005)
        self.nOrig = _dspin(-20, 20, 0.0, 3, 0.05, " V")
        self.dOrig = _dspin(-20, 20, 0.0, 3, 0.05, " V")
        self.dF = _dspin(0.5, 500.0, 30.0, 1, 1.0, " nm")
        self.dF.setToolTip("Front (thin) hBN thickness d_F.  Back hBN d_B = d_F / r.")
        self.eps = _dspin(1.0, 10.0, 3.4, 2, 0.1)
        self.eps.setToolTip("Out-of-plane hBN dielectric constant (≈ 3.0–3.4)")
        for lab, w in (("r = c_b/c_t", self.rSpin), ("u_n0", self.nOrig), ("u_D0", self.dOrig),
                       ("d_F", self.dF), ("ε_hBN", self.eps)):
            r3.addWidget(QtWidgets.QLabel(lab))
            r3.addWidget(w)
        self.physInfo = QtWidgets.QLabel("")
        self.physInfo.setStyleSheet("color:#1a7a3a")
        r3.addWidget(self.physInfo, 1)
        g.addLayout(r3)

        r4 = QtWidgets.QHBoxLayout()
        self.calChk = QtWidgets.QCheckBox("Calibrate axes (u_n, u_D)")
        self.calChk.setChecked(True)
        self.calChk.setToolTip("Off: raw gate voltages (V_front, V_back) on the map, and the sweep axis is "
                               "the front gate voltage")
        self.physChk = QtWidgets.QCheckBox("Physical units (n, D/ε₀)")
        self.physChk.setToolTip("Axes in n [10¹² cm⁻²] and D/ε₀ [V/nm] from d_F and ε_hBN "
                                "(turns on calibrated axes)")
        self.tolSpin = _dspin(0.1, 45.0, 5.0, 1, 0.5, " °", width=80)
        self.tolSpin.setToolTip("A linecut within this angle of the u_D (u_n) axis is a constant-n "
                                "(constant-D) scan; its sweep axis is then D (n)")
        r4.addWidget(self.calChk)
        r4.addWidget(self.physChk)
        r4.addSpacing(12)
        r4.addWidget(QtWidgets.QLabel("Linecut tolerance"))
        r4.addWidget(self.tolSpin)
        r4.addStretch(1)
        g.addLayout(r4)
        self.cutInfo = QtWidgets.QLabel("No scan loaded.")
        self.cutInfo.setStyleSheet("color:#9d174d; font-weight:bold")
        self.cutInfo.setWordWrap(True)
        g.addWidget(self.cutInfo)

        r5 = QtWidgets.QHBoxLayout()
        self.spatCenter = QtWidgets.QSpinBox()
        self.spatCenter.setRange(0, 4095)
        self.spatCenter.setValue(121)
        self.spatCenter.setFixedWidth(80)
        self.spatWidth = QtWidgets.QSpinBox()
        self.spatWidth.setRange(1, 4096)
        self.spatWidth.setValue(13)
        self.spatWidth.setFixedWidth(70)
        for w in (self.spatCenter, self.spatWidth):
            w.setKeyboardTracking(False)
        self.bgLo = _dspin(200, 2000, 1050.0, 1, 5, " nm")
        self.bgHi = _dspin(200, 2000, 1150.0, 1, 5, " nm")
        self.eLo = _dspin(0.8, 2.5, 1.28, 3, 0.01, " eV")
        self.eHi = _dspin(0.8, 2.5, 1.52, 3, 0.01, " eV")
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems(ND_QUANTITIES)
        for lab, w in (("Spatial px", self.spatCenter), ("width", self.spatWidth), ("Bkg ref", self.bgLo),
                       ("to", self.bgHi), ("Window", self.eLo), ("to", self.eHi), ("Colour by", self.quantity)):
            r5.addWidget(QtWidgets.QLabel(lab))
            r5.addWidget(w)
        r5.addStretch(1)
        g.addLayout(r5)
        r6 = QtWidgets.QHBoxLayout()
        self.fitCtl = PeakFitControls("ndmap/fit")
        fl = QtWidgets.QLabel("Peak fit (in the Window):")
        fl.setStyleSheet("font-weight:bold")
        r6.addWidget(fl)
        r6.addWidget(self.fitCtl)
        r6.addStretch(1)
        g.addLayout(r6)
        root.addWidget(box)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._glw = pg.GraphicsLayoutWidget()
        self._mapPlot = self._glw.addPlot(row=0, col=0)
        _style(self._mapPlot)
        self._mapPlot.showGrid(x=False, y=False)
        self._img = pg.ImageItem()
        self._mapPlot.addItem(self._img)
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._img)
        self._glw.addItem(self._hist, row=0, col=1)
        self._glw.ci.layout.setColumnMaximumWidth(1, 90)
        self._selMark = pg.ScatterPlotItem(size=14, pen=pg.mkPen("k", width=2), brush=None, symbol="x")
        self._cutLine = pg.PlotDataItem(pen=pg.mkPen("#e11d48", width=2))
        self._cutPts = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#e11d48", width=1.5), brush=pg.mkBrush("w"))
        self._cutCur = pg.ScatterPlotItem(size=20, pen=pg.mkPen("#e11d48", width=3), brush=None, symbol="o")
        for z, it in enumerate((self._selMark, self._cutLine, self._cutPts, self._cutCur)):
            it.setZValue(10 + z)
            self._mapPlot.addItem(it)
        self._sliceItems = {kind: pg.PlotDataItem(pen=pg.mkPen(col, width=2.2, style=QtCore.Qt.DashLine))
                            for kind, col in (("n", "#00b3cc"), ("D", "#d43bb0"))}
        for it in self._sliceItems.values():
            it.setZValue(9)
            self._mapPlot.addItem(it)
        self._cutPts.sigClicked.connect(self._on_cut_point_clicked)
        self._mapPlot.scene().sigMouseClicked.connect(self._on_map_click)
        mapBox = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mapBox)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        mbar = QtWidgets.QHBoxLayout()
        mbar.addStretch(1)
        self.exportMapBtn = _export_button(
            "Export the n-D map as shown (the 'Colour by' quantity, unmeasured points empty).\n"
            "Raw axes (calibration off): the exact V_front × V_back grid.\n"
            "u_n/u_D or n/D axes: the gate grid is a parallelogram there, so it is sampled as drawn onto a\n"
            "regular grid (4× finer than the gate grid; cells outside the measured area empty).")
        self.exportMapBtn.setStyleSheet("padding: 1px 8px")
        self.exportMapBtn.clicked.connect(self._on_export_map)
        mbar.addWidget(self.exportMapBtn)
        mv.addLayout(mbar)
        mv.addWidget(self._glw, 1)
        split.addWidget(mapBox)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        specBox = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(specBox)
        sv.setContentsMargins(0, 0, 0, 0)
        self._specPlot = pg.PlotWidget()
        self._specPlot.addLegend(offset=(-10, 10))
        self._cRaw = self._specPlot.plot(pen=pg.mkPen("#8a8aa6", width=1.2), name="raw")
        self._cBs = self._specPlot.plot(pen=pg.mkPen("#1f77b4", width=1.8), name="bkg-sub")
        self._cFit = self._specPlot.plot(pen=pg.mkPen("#e08a00", width=2.0, style=QtCore.Qt.DashLine), name="fit")
        self._winReg = pg.LinearRegionItem(brush=pg.mkBrush(255, 179, 0, 40), movable=False)
        self._winReg.setZValue(-10)
        self._specPlot.addItem(self._winReg)
        _style(self._specPlot.getPlotItem(), "PL (counts, bkg-subtracted)", "Photon energy (eV)")
        sv.addWidget(self._specPlot, 1)
        self.fitLbl = QtWidgets.QLabel("Click a point of the map: its spectrum is fitted.")
        self.fitLbl.setWordWrap(True)
        self.fitLbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.fitLbl.setStyleSheet("font-size:11pt; color:#7c2d12")
        sv.addWidget(self.fitLbl)
        right.addWidget(specBox)
        self._spatPlot = pg.PlotWidget()
        self._cProf = self._spatPlot.plot(pen=pg.mkPen("#1f77b4", width=1.5))
        self._bandReg = pg.LinearRegionItem(brush=pg.mkBrush(255, 140, 0, 55))
        self._spatPlot.addItem(self._bandReg)
        _style(self._spatPlot.getPlotItem(), "PL weight", "spatial pixel (CCD row, along the slit)")
        right.addWidget(self._spatPlot)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        self.rightTabs = QtWidgets.QTabWidget()
        self.rightTabs.addTab(right, "Spectrum")
        self.cutViews: Dict[str, LinecutFitView] = {}
        for kind, title, col in (("n", "Spectrum vs n (const D)", "#00b3cc"),
                                 ("D", "Spectrum vs D (const n)", "#d43bb0")):
            view = LinecutFitView(self._run, self._fit_opts, slider=True, color=col, key=f"ndcut_{kind}")
            view.show_message("Load an n-D map.")
            self.cutViews[kind] = view
            self.rightTabs.addTab(view, title)
        split.addWidget(self.rightTabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        root.addWidget(split, 1)

    def _wire(self) -> None:
        self.loadCalBtn.clicked.connect(self._on_load_calibration)
        self.loadNdFileBtn.clicked.connect(self._on_load_nd_file)
        self.loadNdDirBtn.clicked.connect(self._on_load_nd_folder)
        self.reloadNdBtn.clicked.connect(self._on_reload_nd)
        for w in (self.rSpin, self.nOrig, self.dOrig, self.dF, self.eps, self.tolSpin):
            w.valueChanged.connect(lambda _v: self._on_coords_changed())
        self.calChk.toggled.connect(self._on_display_toggled)
        self.physChk.toggled.connect(self._on_display_toggled)
        for w in (self.spatCenter, self.spatWidth):
            w.valueChanged.connect(lambda _v: self._on_band_changed())
        for w in (self.bgLo, self.bgHi):
            w.valueChanged.connect(lambda _v: self._on_spectra_changed())
        for w in (self.eLo, self.eHi):
            w.valueChanged.connect(lambda _v: self._on_window_changed())
        self.quantity.currentTextChanged.connect(lambda _t: self._refresh_map())
        self._bandReg.sigRegionChangeFinished.connect(self._on_region_changed)
        self.fitCtl.changed.connect(self._on_fit_settings)
        self.rightTabs.currentChanged.connect(lambda _i: self._on_right_tab())
        for kind, view in self.cutViews.items():
            view.slider_changed.connect(lambda v, k=kind: self._on_cut_slider(k, v))

    # ── settings ──────────────────────────────────────────────────────
    _SAVED = (("r", "rSpin"), ("un0", "nOrig"), ("uD0", "dOrig"), ("dF", "dF"), ("eps", "eps"),
              ("tol", "tolSpin"), ("bg_lo", "bgLo"), ("bg_hi", "bgHi"), ("e_lo", "eLo"), ("e_hi", "eHi"))

    def _restore_settings(self) -> None:
        st = _settings()
        for key, attr in self._SAVED:
            v = st.value(f"ndmap/{key}")
            if v is not None:
                try:
                    getattr(self, attr).setValue(float(v))
                except (TypeError, ValueError):
                    pass
        for key, chk in (("calibrated", self.calChk), ("physical", self.physChk)):
            v = st.value(f"ndmap/{key}")
            if v is not None:
                chk.setChecked(str(v).lower() in ("true", "1"))
        q = st.value("ndmap/quantity")
        if q in ND_QUANTITIES:
            self.quantity.setCurrentText(q)
        cal = st.value("ndmap/cal_path")
        if cal:
            self.cal_path = str(cal)
            self.calInfo.setText(f"r = {self.rSpin.value():.4f}  (last session; calibration {Path(cal).name})")

    def save_settings(self) -> None:
        st = _settings()
        for key, attr in self._SAVED:
            st.setValue(f"ndmap/{key}", float(getattr(self, attr).value()))
        st.setValue("ndmap/calibrated", self.calChk.isChecked())
        st.setValue("ndmap/physical", self.physChk.isChecked())
        st.setValue("ndmap/quantity", self.quantity.currentText())
        if self.cal_path:
            st.setValue("ndmap/cal_path", self.cal_path)
        if self._nd_arg:
            st.setValue("ndmap/nd_path", self._nd_arg)

    # ── calibration ───────────────────────────────────────────────────
    def calib(self) -> GateCalib:
        return GateCalib(self.rSpin.value(), self.nOrig.value(), self.dOrig.value(), self.dF.value(),
                         self.eps.value(), self.calChk.isChecked(), self.physChk.isChecked())

    def tolerance(self) -> float:
        return float(self.tolSpin.value())

    def _on_load_calibration(self) -> None:
        start = self.cal_path or (str(_ND_CAL_DIR) if _ND_CAL_DIR.is_dir() else str(HERE))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load calibration file", start,
                                                      "Calibration (*.json);;All files (*)")
        if fn:
            self.load_calibration(fn)

    def load_calibration(self, fn) -> bool:
        try:
            cal = json.loads(Path(fn).read_text(encoding="utf-8"))
            r = float(cal["r"])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load calibration", f"Could not read a calibration r from:\n{fn}\n\n{exc}")
            return False
        self.cal_path = str(fn)
        sig = cal.get("r_sigma")
        cdata = cal.get("data") or {}
        src = cdata.get("source_dir", "") or ""
        if src and ("\\" in src or "/" in src):
            prov = Path(src).name
        else:
            bits = []
            if cdata.get("grid"):
                bits.append(f"{cdata['grid'][0]}x{cdata['grid'][1]}")
            if cdata.get("n_measured"):
                bits.append(f"{cdata['n_measured']} pts")
            prov = ", ".join(bits) or "?"
        self.calChk.blockSignals(True)
        self.calChk.setChecked(True)
        self.calChk.blockSignals(False)
        self.rSpin.setValue(r)
        info = f"r = {r:.4f}" + (f" ± {float(sig):.4f}" if sig is not None else "")
        self.calInfo.setText(info + f"   ·  {Path(fn).name}   ·  fitted on {prov}")
        self._on_coords_changed()
        self.save_settings()
        self.status.emit(f"Calibration loaded: r = {r:.4f} (from {Path(fn).name})")
        return True

    def _update_phys_info(self) -> None:
        sN, sD = self.calib().scale()
        self.physInfo.setText(f"n = {sN:.3f}·u_n ×10¹² cm⁻²  ·  D/ε₀ = {sD:.4f}·u_D V/nm")

    def _on_display_toggled(self, *_a) -> None:
        if self.physChk.isChecked() and not self.calChk.isChecked():      # physical implies calibrated
            if self.sender() is self.physChk:
                self.calChk.blockSignals(True)
                self.calChk.setChecked(True)
                self.calChk.blockSignals(False)
            else:
                self.physChk.blockSignals(True)
                self.physChk.setChecked(False)
                self.physChk.blockSignals(False)
        self._on_coords_changed()

    def _on_coords_changed(self) -> None:
        self._update_phys_info()
        self._apply_axes()
        self._interp = None
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_cuts()
        self._overlay_fit = True
        self.calib_changed.emit()
        self.save_settings()

    def _apply_axes(self) -> None:
        xl, yl = self.calib().axis_labels()
        self._mapPlot.setLabel("bottom", xl)
        self._mapPlot.setLabel("left", yl)

    def set_cut_info(self, text: str) -> None:
        self.cutInfo.setText(text)

    # ── scan overlay ──────────────────────────────────────────────────
    def set_scan(self, vf, vb, order, cur: Optional[int], refit: bool = False) -> None:
        self._scan = (np.asarray(vf, float), np.asarray(vb, float), np.asarray(order, int), cur)
        self._overlay_fit = self._overlay_fit or refit
        self._draw_overlay()

    def set_current(self, cur: Optional[int]) -> None:
        vf, vb, order, _ = self._scan
        self._scan = (vf, vb, order, cur)
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        vf, vb, order, cur = self._scan
        if vf.size == 0:
            self._cutLine.setData([], [])
            self._cutPts.setData([])
            self._cutCur.setData([])
            return
        cal = self.calib()
        x, y = cal.display_xy(vf[order], vb[order])
        self._cutLine.setData(x, y)
        self._cutPts.setData(x=x, y=y, data=list(order))
        if cur is not None and 0 <= cur < vf.size:
            cx, cy = cal.display_xy(vf[cur:cur + 1], vb[cur:cur + 1])
            self._cutCur.setData(x=cx, y=cy)
        else:
            self._cutCur.setData([])
        if self._overlay_fit:
            self._overlay_fit = False
            self._mapPlot.autoRange()

    def _on_cut_point_clicked(self, _item, points) -> None:
        if len(points):
            self.point_clicked.emit(int(points[0].data()))

    # ── n-D map loading ───────────────────────────────────────────────
    def _nd_start_dir(self) -> str:
        last = _settings().value("ndmap/nd_path")
        if last:
            p = Path(str(last))
            return str(p.parent if p.is_file() else p)
        return str(_ND_CAL_DIR) if _ND_CAL_DIR.is_dir() else str(HERE)

    def _on_load_nd_file(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load n-D map cache", self._nd_start_dir(),
                                                      "n-D map cache (*.npz);;All files (*)")
        if fn:
            self.load_ndmap(fn)

    def _on_load_nd_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "n-D map folder (.asc dual-gate grid)",
                                                       self._nd_start_dir())
        if d:
            self.load_ndmap(d)

    def _on_reload_nd(self) -> None:
        if not self._nd_arg:
            return
        rebuild = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        self.load_ndmap(self._nd_arg, reload=True, rebuild=rebuild)

    def load_ndmap(self, path, reload: bool = False, rebuild: bool = False, wl_ref=None) -> None:
        if self._load_job is not None:
            self.status.emit("An n-D map is already loading.")
            return
        path = str(path)
        is_dir = Path(path).is_dir()
        self.ndInfo.setText(f"Loading {Path(path).name} …")
        for w in (self.loadNdFileBtn, self.loadNdDirBtn, self.reloadNdBtn):
            w.setEnabled(False)

        def work(progress, cancelled):
            try:
                data, src = nd_load(path, rebuild=rebuild, update=reload and is_dir, wl_ref=wl_ref,
                                    progress=progress, cancelled=cancelled)
            except LookupError as exc:
                raise RuntimeError(f"WLREF:{exc}") from None
            return nd_prepare(data), str(src)

        def prog(d, n):
            self.ndInfo.setText(f"Loading {Path(path).name}: {d}/{n} frames …")

        self._load_job = self._run(work, lambda res, err: self._on_nd_loaded(path, reload, rebuild, res, err), prog)

    def _on_nd_loaded(self, path, reload, rebuild, res, err: str) -> None:
        self._load_job = None
        for w in (self.loadNdFileBtn, self.loadNdDirBtn, self.reloadNdBtn):
            w.setEnabled(True)
        if err.startswith("WLREF:"):
            QtWidgets.QMessageBox.information(
                self, "Wavelength axis needed",
                f"{err[6:]}\n\nPick an .asc frame that HAS a wavelength column, taken with the same "
                "grating and center wavelength.")
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reference .asc with wavelength column",
                                                          str(Path(path).parent), "Andor ASCII (*.asc);;All files (*)")
            if fn:
                self.load_ndmap(path, reload, rebuild, wl_ref=fn)
            else:
                self.ndInfo.setText("n-D map loading cancelled.")
            return
        if err:
            self.ndInfo.setText(f"Could not load {Path(path).name}: {err}")
            QtWidgets.QMessageBox.warning(self, "Load n-D map", f"Could not load:\n{path}\n\n{err}")
            return
        nd, src = res
        keep_band = (reload and self.nd is not None and np.array_equal(nd["spat_rows"], self.nd["spat_rows"]))
        band = (self.spatCenter.value(), self.spatWidth.value())
        sel_v = None
        if reload and self.nd is not None:
            sel_v = (self.nd["vf"][self.sel[0]], self.nd["vb"][self.sel[1]])
        self.nd, self.nd_source, self._nd_arg = nd, src, path
        r0, r1 = int(nd["spat_rows"][0]), int(nd["spat_rows"][-1])
        nwin = nd["cube"].shape[2]
        self._sync = True
        self.spatCenter.setRange(r0, r1)
        self.spatCenter.setValue(int(band[0]) if keep_band else nd["spot_center"])
        self.spatWidth.setRange(1, nwin)
        self.spatWidth.setValue(int(band[1]) if keep_band else min(13, nwin))
        self._sync = False
        if sel_v is not None:
            self.sel = (int(np.argmin(np.abs(nd["vf"] - sel_v[0]))), int(np.argmin(np.abs(nd["vb"] - sel_v[1]))))
        else:
            self.sel = (nd["vf"].size // 2, nd["vb"].size // 2)
        self._cProf.setData(nd["spat_rows"], nd["prof"])
        nvf, nvb = nd["vf"].size, nd["vb"].size
        self.ndInfo.setText(f"{Path(src).parent.name if Path(src).is_file() else Path(src).name} / "
                            f"{Path(src).name}:  {nvf}×{nvb} grid, {int(nd['measured'].sum())} measured points, "
                            f"V_f {nd['vf'][0]:+.2f}…{nd['vf'][-1]:+.2f} V, V_b {nd['vb'][0]:+.2f}…{nd['vb'][-1]:+.2f} V")
        self.ndInfo.setToolTip(src)
        self._recompute_spectra()
        self._update_band_region()
        self._overlay_fit = True
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_cuts()
        self.save_settings()
        self.status.emit(f"n-D map loaded: {self.ndInfo.text()}")

    # ── spectra and map ───────────────────────────────────────────────
    def _band_slice(self) -> Tuple[int, int]:
        rows = self.nd["spat_rows"]
        c, w = int(self.spatCenter.value()), int(self.spatWidth.value())
        half = w // 2
        lo = max(rows[0], c - half)
        hi = min(rows[-1], c + half)
        i0, i1 = int(lo - rows[0]), int(hi - rows[0])
        return i0, max(i0, i1)

    def _recompute_spectra(self) -> None:
        if self.nd is None:
            self.spec_bs = None
            return
        nd = self.nd
        i0, i1 = self._band_slice()
        band = np.nan_to_num(nd["cube"][:, :, i0:i1 + 1, :]).sum(axis=2)
        lo, hi = float(self.bgLo.value()), float(self.bgHi.value())
        mask = (nd["wl"] >= lo) & (nd["wl"] <= hi)
        if mask.any():
            band = band - np.nanmean(band[:, :, mask], axis=2, keepdims=True)
        self.spec_bs = band[:, :, nd["order"]]
        self._interp = None

    def _raw_map(self, name: str) -> np.ndarray:
        nd = self.nd
        lo, hi = float(self.eLo.value()), float(self.eHi.value())
        m = (nd["e"] >= lo) & (nd["e"] <= hi)
        if not m.any():
            self.status.emit(f"Energy window {lo:.3f}–{hi:.3f} eV is outside the n-D map's spectra "
                             f"({nd['e'].min():.3f}–{nd['e'].max():.3f} eV)")
            return np.full(nd["measured"].shape, np.nan)
        ew = nd["e"][m]
        Y = self.spec_bs[:, :, m]
        if name == "Integrated intensity (window)":
            return Y.sum(-1)
        if name == "Peak height (max)":
            return Y.max(-1)
        if name == "Peak energy (argmax)":
            return ew[np.argmax(Y, axis=-1)]
        Yc = np.clip(Y, 0, None)
        tot = Yc.sum(-1)
        cen = (Yc * ew).sum(-1) / np.where(tot > 0, tot, 1)
        if name == "Centroid energy":
            return cen
        var = (Yc * (ew[None, None, :] - cen[..., None]) ** 2).sum(-1) / np.where(tot > 0, tot, 1)
        return np.sqrt(np.clip(var, 0, None)) * 1000.0

    def _transform(self) -> QtGui.QTransform:
        """ImageItem pixel (i, j) -> map coordinates (sheared grid in u_n, u_D)."""
        cal, nd = self.calib(), self.nd
        vf, vb, dvf, dvb = nd["vf"], nd["vb"], nd["dvf"], nd["dvb"]
        if cal.mode == "raw":
            return QtGui.QTransform(dvf, 0.0, 0.0, dvb, float(vf[0]) - 0.5 * dvf, float(vb[0]) - 0.5 * dvb)
        r = cal.r
        p, q = dvf, r * dvb
        a_n = float(vf[0]) + r * float(vb[0]) - cal.un0
        a_D = float(vf[0]) - r * float(vb[0]) - cal.uD0
        dx, dy = a_n - 0.5 * (p + q), a_D - 0.5 * (p - q)
        if cal.mode == "phys":
            sN, sD = cal.scale()
            return QtGui.QTransform(sN * p, sD * p, sN * q, -sD * q, sN * dx, sD * dy)
        return QtGui.QTransform(p, p, q, -q, dx, dy)

    def _refresh_map(self) -> None:
        if self.nd is None or self.spec_bs is None:
            self._img.clear()
            self._selMark.setData([])
            self._mapPlot.setTitle("no n-D map loaded: only the scan's linecut is drawn")
            self._draw_overlay()
            self.exportMapBtn.setEnabled(False)
            return
        self.exportMapBtn.setEnabled(True)
        name = self.quantity.currentText()
        arr = np.where(self.nd["measured"], np.asarray(self._raw_map(name), float), np.nan)
        self._img.setImage(arr, autoLevels=False)
        self._img.setTransform(self._transform())
        fin = arr[np.isfinite(arr)]
        if fin.size:
            if name in _ND_ENERGY_Q:
                cmap, plo, phi = "plasma", 2, 98
            elif "Linewidth" in name:
                cmap, plo, phi = "magma", 1, 99
            else:
                cmap, plo, phi = "viridis", 1, 99
            self._hist.gradient.setColorMap(_pg_cmap(cmap))
            lv = (float(np.nanpercentile(fin, plo)), float(np.nanpercentile(fin, phi)))
            if not lv[1] > lv[0]:
                lv = (lv[0], lv[0] + 1.0)
            self._img.setLevels(lv)
            self._hist.setLevels(*lv)
        mode = {"raw": "V_front, V_back", "gate": "u_n, u_D", "phys": "n, D/ε₀"}[self.calib().mode]
        self._mapPlot.setTitle(f"{name}   [{mode}]")
        self._update_sel_marker()
        self._draw_overlay()
        self._overlay_fit = False
        self._mapPlot.autoRange()

    def map_matrix(self, oversample: int = 4):
        """(x, y, z (ny, nx)) of the shown n-D map.  Raw axes: the gate grid itself.  Sheared axes: the
        image sampled as drawn on a regular grid (each sample takes the gate cell it falls in)."""
        nd = self.nd
        name = self.quantity.currentText()
        arr = np.where(nd["measured"], np.asarray(self._raw_map(name), float), np.nan)     # (nvf, nvb)
        if self.calib().mode == "raw":
            return nd["vf"].copy(), nd["vb"].copy(), arr.T
        tr = self._transform()
        nvf, nvb = arr.shape
        corners = [tr.map(float(a), float(b)) for a in (0, nvf) for b in (0, nvb)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        n = oversample * max(nvf, nvb)
        dx, dy = (max(xs) - min(xs)) / n, (max(ys) - min(ys)) / n
        x = min(xs) + (np.arange(n) + 0.5) * dx
        y = min(ys) + (np.arange(n) + 0.5) * dy
        inv, ok = tr.inverted()
        if not ok:
            raise ValueError("the map transform is not invertible")
        X, Y = np.meshgrid(x, y)                                       # (ny, nx)
        I = np.floor(inv.m11() * X + inv.m21() * Y + inv.dx()).astype(int)
        J = np.floor(inv.m12() * X + inv.m22() * Y + inv.dy()).astype(int)
        inside = (I >= 0) & (I < nvf) & (J >= 0) & (J < nvb)
        z = np.full(X.shape, np.nan)
        z[inside] = arr[I[inside], J[inside]]
        return x, y, z

    def _on_export_map(self) -> None:
        if self.nd is None or self.spec_bs is None:
            return
        mode = {"raw": "Vf_Vb", "gate": "un_uD", "phys": "n_D"}[self.calib().mode]
        src = Path(self.nd_source).stem if self.nd_source else "ndmap"

        def write(fn):
            export_matrix_csv(fn, *self.map_matrix())
        _export_csv_dialog(self, f"{src}_{self.quantity.currentText()}_{mode}", write)

    def _cell_xy(self, i: int, j: int):
        nd = self.nd
        x, y = self.calib().display_xy(nd["vf"][i:i + 1], nd["vb"][j:j + 1])
        return float(x[0]), float(y[0])

    def _update_sel_marker(self) -> None:
        if self.nd is None:
            self._selMark.setData([])
            return
        x, y = self._cell_xy(*self.sel)
        self._selMark.setData([x], [y])

    def _show_spectrum(self, i: int, j: int) -> None:
        if self.nd is None or self.spec_bs is None:
            self._cRaw.setData([], [])
            self._cBs.setData([], [])
            self._cFit.setData([], [])
            self._specPlot.setTitle("")
            return
        nd = self.nd
        i = int(np.clip(i, 0, nd["vf"].size - 1))
        j = int(np.clip(j, 0, nd["vb"].size - 1))
        self.sel = (i, j)
        i0, i1 = self._band_slice()
        raw = np.nan_to_num(nd["cube"][i, j, i0:i1 + 1, :]).sum(0)[nd["order"]]
        self._cRaw.setData(nd["e"], raw)
        self._cBs.setData(nd["e"], self.spec_bs[i, j])
        self._winReg.setRegion((float(self.eLo.value()), float(self.eHi.value())))
        cal = self.calib()
        un, uD = cal.u(nd["vf"][i], nd["vb"][j])
        sN, sD = cal.scale()
        meas = "" if nd["measured"][i, j] else "   (not measured)"
        self._specPlot.setTitle(f"Vf={nd['vf'][i]:+.3f} Vb={nd['vb'][j]:+.3f} | u_n={float(un):+.3f} "
                                f"u_D={float(uD):+.3f} | n={sN * float(un):+.3g}×10¹² D/ε₀={sD * float(uD):+.3g} V/nm{meas}")
        self._update_sel_marker()
        if not nd["measured"][i, j]:
            self._cFit.setData([], [])
            self.fitLbl.setText("(not measured: no fit)")
            return
        opts = self._fit_opts()
        res = fit_peak(nd["e"], self.spec_bs[i, j], opts)
        if res is not None:
            ef = np.linspace(opts.lo, opts.hi, 600)
            self._cFit.setData(ef, peak_curve(res, ef))
        else:
            self._cFit.setData([], [])
        self.fitLbl.setText(_peak_html(res))

    # ── peak fit + linecuts through the n-D map ───────────────────────
    def _fit_opts(self) -> PeakFitOpts:
        return self.fitCtl.opts(self.eLo.value(), self.eHi.value())

    def _on_fit_settings(self) -> None:
        if self.nd is not None:
            self._show_spectrum(*self.sel)
        kind = self._active_cut()
        if kind is not None:
            self.cutViews[kind].refresh()

    def _active_cut(self) -> Optional[str]:
        return {1: "n", 2: "D"}.get(self.rightTabs.currentIndex())

    def _on_right_tab(self) -> None:
        kind = self._active_cut()
        if kind is not None:
            self._update_cut_view(kind)
        self._draw_slices()

    def _refresh_cuts(self) -> None:
        """Recompute the shown linecut (the hidden one is recomputed when it is shown)."""
        kind = self._active_cut()
        if kind is not None:
            self._update_cut_view(kind)
        else:
            self._draw_slices()

    def _on_cut_slider(self, kind: str, v: int) -> None:
        self._cut_idx[kind] = int(v)
        self._update_cut_view(kind)

    def _cut_interp(self):
        """Linear interpolation of the background-subtracted spectra over the measured (u_n, u_D) points
        (as the gate grids viewer's waterfall maps), with the measured u_n and u_D ranges."""
        if self._interp is not None:
            return self._interp
        nd = self.nd
        if nd is None or self.spec_bs is None or _LinearND is None:
            return None
        VF, VB = np.meshgrid(nd["vf"], nd["vb"], indexing="ij")
        un, uD = self.calib().u(VF, VB)
        meas = nd["measured"]
        if meas.sum() < 3:
            return None
        try:
            ip = _LinearND(np.column_stack([un[meas], uD[meas]]), np.asarray(self.spec_bs[meas], float))
        except Exception:                       # points on a line: no triangulation
            return None
        self._interp = (ip, (float(un[meas].min()), float(un[meas].max())),
                        (float(uD[meas].min()), float(uD[meas].max())))
        return self._interp

    def _update_cut_view(self, kind: str) -> None:
        view = self.cutViews[kind]
        self._cut_s[kind] = self._cut_ends[kind] = None
        if self.nd is None:
            view.show_message("Load an n-D map.")
            self._draw_slices()
            return
        ip = self._cut_interp()
        if ip is None:
            view.show_message("scipy.interpolate is needed for linecuts" if _LinearND is None
                              else "Not enough measured points (spanning an area) for a linecut.")
            self._draw_slices()
            return
        interp, (n_lo, n_hi), (d_lo, d_hi) = ip
        k = int(np.clip(self._cut_idx[kind], 0, ND_CUT_N - 1))
        cal = self.calib()
        sN, sD = cal.scale()
        phys = cal.mode == "phys"
        if kind == "n":                                   # sweep n at constant D
            c = d_lo + (d_hi - d_lo) * k / (ND_CUT_N - 1)
            s = np.linspace(n_lo, n_hi, ND_CUT_N)
            pts = np.column_stack([s, np.full_like(s, c)])
            const = f"const D:  D/ε₀ = {sD * c:+.4g} V/nm   (u_D = {c:+.3f} V)"
            xs, xl = (sN * s, "n (10¹² cm⁻²)") if phys else (s, "u_n = V_f + r·V_b (V)")
            dip = None
        else:                                             # sweep D at constant n
            c = n_lo + (n_hi - n_lo) * k / (ND_CUT_N - 1)
            s = np.linspace(d_lo, d_hi, ND_CUT_N)
            pts = np.column_stack([np.full_like(s, c), s])
            const = f"const n:  n = {sN * c:+.4g}×10¹² cm⁻²   (u_n = {c:+.3f} V)"
            xs, xl = (sD * s, "D/ε₀ (V/nm)") if phys else (s, "u_D = V_f − r·V_b (V)")
            dip = 1.0 if phys else sD                     # d(D/ε₀)/dx
        view.set_slider(k, const)
        with np.errstate(all="ignore"):
            Zc = np.asarray(interp(pts), float)           # (ND_CUT_N, n_e); NaN outside the measured area
        keep = np.isfinite(Zc).any(axis=1)
        if keep.sum() < 2:
            view.show_message(f"This cut misses the measured area ({const}).")
            self._draw_slices()
            return
        self._cut_s[kind] = s[keep]
        self._cut_ends[kind] = pts[keep][[0, -1]]
        view.set_spectra(xs[keep], self.nd["e"], Zc[keep].T, x_label=xl,
                         title=f"n-D map linecut, {const}", dip_scale=dip)
        self._draw_slices()

    def _draw_slices(self) -> None:
        """The shown linecut as a dashed line on the n-D map (any axes mode)."""
        act = self._active_cut()
        cal = self.calib()
        for kind, item in self._sliceItems.items():
            ends = self._cut_ends.get(kind)
            if self.nd is None or kind != act or ends is None:
                item.setData([], [])
                continue
            a, b = ends[:, 0] + cal.un0, ends[:, 1] + cal.uD0
            vf, vb = 0.5 * (a + b), (a - b) / (2.0 * cal.r)
            x, y = cal.display_xy(vf, vb)
            item.setData(x, y)

    def _move_cut_to(self, i: int, j: int) -> None:
        """Put the shown linecut through grid point (i, j) and select the column nearest to it."""
        kind = self._active_cut()
        ip = self._cut_interp() if kind is not None else None
        if ip is None:
            return
        _interp, (n_lo, n_hi), (d_lo, d_hi) = ip
        un, uD = (float(v) for v in self.calib().u(self.nd["vf"][i], self.nd["vb"][j]))
        c, lo, hi, sw = (uD, d_lo, d_hi, un) if kind == "n" else (un, n_lo, n_hi, uD)
        k = int(np.clip(round((c - lo) / (hi - lo) * (ND_CUT_N - 1)), 0, ND_CUT_N - 1)) if hi > lo else ND_CUT_N // 2
        self._cut_idx[kind] = k
        self._update_cut_view(kind)
        s = self._cut_s[kind]
        if s is not None:
            self.cutViews[kind].select_column(int(np.argmin(np.abs(s - sw))))

    def _update_band_region(self) -> None:
        if self.nd is None:
            return
        rows = self.nd["spat_rows"]
        i0, i1 = self._band_slice()
        self._sync = True
        self._bandReg.setRegion((rows[i0] - 0.5, rows[i1] + 0.5))
        self._sync = False

    def _on_region_changed(self) -> None:
        if self._sync or self.nd is None:
            return
        a, b = self._bandReg.getRegion()
        rows = self.nd["spat_rows"]
        c, w = int(round(0.5 * (a + b))), int(round(b - a))
        self._sync = True
        self.spatCenter.setValue(max(int(rows[0]), min(int(rows[-1]), c)))
        self.spatWidth.setValue(max(1, min(rows.size, w)))
        self._sync = False
        self._on_band_changed()

    def _on_band_changed(self) -> None:
        if self._sync:
            return
        self._recompute_spectra()
        self._update_band_region()
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_cuts()

    def _on_spectra_changed(self) -> None:
        self._recompute_spectra()
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_cuts()
        self.save_settings()

    def _on_window_changed(self) -> None:
        self._refresh_map()
        self._on_fit_settings()
        self.save_settings()

    def _on_map_click(self, ev) -> None:
        if self.nd is None or ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.scenePos()
        if not self._mapPlot.sceneBoundingRect().contains(pos):
            return
        pt = self._img.mapFromScene(pos)
        i, j = int(np.floor(pt.x())), int(np.floor(pt.y()))
        if 0 <= i < self.nd["vf"].size and 0 <= j < self.nd["vb"].size:
            self._show_spectrum(i, j)
            self._move_cut_to(i, j)


# ══════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════

BG_ITEMS = (("Subtract dark reference", "dark_ref"), ("Subtract pre-t0 level", "pre_t0"), ("None", "none"))


class DualGateViewer(QtWidgets.QMainWindow):
    MODE_GATE, MODE_POWER = "Gate dependence", "Power dependence"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL + TRPL dual-gate viewer")
        self.store = TrplStore(max_files=64)
        self.model: Optional[ScanModel] = None
        self._frames_cache: Dict[str, List[PLFrame]] = {}
        self._jobs: List[_Job] = []
        self._job_id = 0
        self._close_t0: Optional[float] = None
        self._load_job = None
        self._align_job = None
        self._sweep_t0: Optional[float] = None
        self._t0_ready = False
        self._times_after_t0 = False      # Begin / End / Ref are after time zero (aligned view) or raw times
        self._last_off: Optional[float] = None
        self._pump2 = P2_OFF
        self._angle_calib: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # sweep axes
        self._cut: Optional[Linecut] = None
        self._order = np.zeros(0, int)            # base point indices, sorted by the sweep x
        self._px = np.zeros(0)                    # power axis values
        self._plabel = "Power"
        self._porder = np.zeros(0, int)
        self._b = 0                               # current base point index
        self._pi = 0                              # current power index
        # current TRPL point
        self.geom = SPAD23_GEOM
        self.sel_pixel = SPAD23_GEOM.center_pixel
        self.excluded: set = set()
        self._path: Optional[str] = None
        self._t = np.zeros(0)
        self._c = np.zeros((0, self.geom.num_pixels))
        self._tz = None
        self._bgv = np.zeros(self.geom.num_pixels)
        self._bg_note = ""
        self._t_cursor: Optional[float] = None
        self._dark: Optional[DarkReference] = None
        self._dark_err = ""
        self._acq_s: Optional[float] = None
        self._bg_fallback = False
        # fits
        self._fits: Dict[Tuple[int, int], Tuple[tuple, str, PointDiffusion]] = {}
        self._fit_job = None
        self._fitall_job = None
        self._sigma_snap: Optional[dict] = None
        self._ref_common: Optional[dict] = None       # median reference width of a sweep line (+ its key)
        self._frame_req = None
        self._frame_pending = None
        self._frame_job = None
        self._frame_cache: "OrderedDict" = OrderedDict()
        self._frame_res: Optional[dict] = None
        # dependence maps
        self._traces: Dict[str, np.ndarray] = {}
        self._traces_t: Dict[str, np.ndarray] = {}
        self._traces_vs = None
        self._traces_job = None
        self._dep_dirty = True
        self._pl_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._crop = DEFAULT_CROP
        self._build_ui()
        self.resize(1500, 1000)

    # ══ UI ═════════════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        self.bar = self._make_bar()
        root.addWidget(self.bar)
        self.tabs = QtWidgets.QTabWidget()
        self.ndTab = NDMapTab(self._run)
        self.tabs.addTab(self.ndTab, "1  n-D map")
        self.tabs.addTab(self._make_point_tab(), "2  Point (PL + TRPL)")
        self.tabs.addTab(self._make_fit_tab(), "3  Spatial fit")
        self.tabs.addTab(self._make_dep_tab(), "4  Dependence")
        self.tabs.addTab(self._make_scanfit_tab(), "5  PL fit (scan)")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.ndTab.calib_changed.connect(self._on_calib_changed)
        self.ndTab.point_clicked.connect(self._on_nd_point_clicked)
        self.ndTab.status.connect(lambda t: self.statusBar().showMessage(t, 15000))
        _big_fonts(self)
        self._on_tab_changed(0)

    def _make_bar(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Scan")
        v = QtWidgets.QVBoxLayout(box)
        v.setSpacing(3)
        v.setContentsMargins(6, 4, 6, 4)

        r1 = QtWidgets.QHBoxLayout()
        self.loadBtn = QtWidgets.QPushButton("Load scan…")
        self.reloadBtn = QtWidgets.QPushButton("Reload")
        self.reloadBtn.setToolTip("Re-read the scan folder (new points of a running scan)")
        self.folderLbl = QtWidgets.QLabel("No scan loaded.")
        self.folderLbl.setStyleSheet("color:#1f2937")
        self.pump2Combo = QtWidgets.QComboBox()
        self.pump2Combo.setEnabled(False)
        self.pump2Combo.setToolTip("Pump-2 state to show (dI/dP folders: files tagged _P2off / _P2on)")
        self.bgCombo = QtWidgets.QComboBox()
        for text, key in BG_ITEMS:
            self.bgCombo.addItem(text, key)
        self.bgCombo.setToolTip("Background subtracted before the 2D Gaussian fits (and on the count map)")
        self.darkBtn = QtWidgets.QPushButton("Dark…")
        self.darkBtn.setToolTip("Load a dark-count reference (DARK_TRPL_*.csv). Default: the newest one in the scan folder.")
        self.noDarkBtn = QtWidgets.QPushButton("No dark")
        self.darkLbl = QtWidgets.QLabel("dark: --")
        self.darkLbl.setStyleSheet("color:#555")
        self.powerCalBtn = QtWidgets.QPushButton("Power calib…")
        self.powerCalBtn.setToolTip("Angle → power calibration (xlsx / csv with 'angle' and 'power' columns) "
                                    "for scans whose files carry a stage angle")
        r1.addWidget(self.loadBtn)
        r1.addWidget(self.reloadBtn)
        r1.addWidget(self.folderLbl, 1)
        r1.addWidget(QtWidgets.QLabel("Pump 2:"))
        r1.addWidget(self.pump2Combo)
        r1.addWidget(QtWidgets.QLabel("Background:"))
        r1.addWidget(self.bgCombo)
        r1.addWidget(self.darkBtn)
        r1.addWidget(self.noDarkBtn)
        r1.addWidget(self.darkLbl)
        r1.addWidget(self.powerCalBtn)
        v.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        self.voltBar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.voltLbl = QtWidgets.QLabel("--")
        self.voltLbl.setMinimumWidth(420)
        self.powerBar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.powerLbl = QtWidgets.QLabel("--")
        self.powerLbl.setMinimumWidth(200)
        for w in (self.voltBar, self.powerBar):
            w.setPageStep(1)
            w.setMinimumWidth(160)
        r2.addWidget(QtWidgets.QLabel("Voltage"))
        r2.addWidget(self.voltBar, 3)
        r2.addWidget(self.voltLbl)
        r2.addWidget(QtWidgets.QLabel("Power"))
        r2.addWidget(self.powerBar, 1)
        r2.addWidget(self.powerLbl)
        v.addLayout(r2)

        r3 = QtWidgets.QHBoxLayout()
        self.timeBar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.timeBar.setPageStep(10)
        self.timeBar.setMinimumWidth(160)
        self.timeLbl = QtWidgets.QLabel("t = --")
        self.timeLbl.setMinimumWidth(110)
        val = QtGui.QDoubleValidator()
        val.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.beginEdit, self.endEdit, self.refEdit = (QtWidgets.QLineEdit() for _ in range(3))
        for w, tip in ((self.beginEdit, "Begin of the time range (ns): time axes, time scrollbar, last fitted frame"),
                       (self.endEdit, "End of the time range (ns)"),
                       (self.refEdit, "Reference time (ns): first frame of the σ²(t) fit, t_ref of the D line.\n"
                                      f"Empty: signal peak (t = 0) + {T0_REF_BINS} bins with 'Align t0' on (the begin of the "
                                      "range without it)")):
            w.setValidator(val)
            w.setFixedWidth(70)
            w.setToolTip(tip)
        self.alignCheck = QtWidgets.QCheckBox("Align t0")
        self.alignCheck.setChecked(True)
        self.alignCheck.setToolTip("Time-zero alignment on each pixel's first peak; every point is then moved to "
                                   "the scan's common time zero; t = 0 ns is then the signal peak (brightest\n"
                                   "pixel) of each point.\n"
                                   "Begin / End / Ref are converted when this is switched.")
        self.rebinSpin = QtWidgets.QSpinBox()
        self.rebinSpin.setRange(1, 200)
        self.rebinSpin.setValue(1)
        self.rebinSpin.setPrefix("×")
        self.rebinSpin.setFixedWidth(64)
        self.rebinSpin.setKeyboardTracking(False)
        self.rebinSpin.setToolTip("Sum this many consecutive time bins (after the time-zero alignment) before\n"
                                  "everything else: count map, traces, 2D fits, σ²(t), dependence maps.\n"
                                  "The dark reference is rescaled to the new bin width.")
        self.binLbl = QtWidgets.QLabel("")
        self.binLbl.setStyleSheet("color:#555")
        self.fixOffsetCheck = QtWidgets.QCheckBox("Gaussian offset = 0")
        self.exclEdit = QtWidgets.QLineEdit()
        self.exclEdit.setPlaceholderText("excluded px")
        self.exclEdit.setFixedWidth(110)
        self.exclEdit.setToolTip("Pixels left out of the 2D fits (or right click them on the count map)")
        r3.addWidget(QtWidgets.QLabel("Time"))
        r3.addWidget(self.timeBar, 3)
        r3.addWidget(self.timeLbl)
        for lab, w in (("Begin", self.beginEdit), ("End", self.endEdit), ("Ref", self.refEdit)):
            r3.addWidget(QtWidgets.QLabel(lab))
            r3.addWidget(w)
        r3.addWidget(QtWidgets.QLabel("ns"))
        r3.addSpacing(8)
        r3.addWidget(self.alignCheck)
        r3.addWidget(self.fixOffsetCheck)
        r3.addWidget(QtWidgets.QLabel("Excluded"))
        r3.addWidget(self.exclEdit)
        v.addLayout(r3)

        r4 = QtWidgets.QHBoxLayout()
        self.smoothCombo = QtWidgets.QComboBox()
        for text, key in SMOOTH_METHODS:
            self.smoothCombo.addItem(text, key)
        self.smoothCombo.setToolTip(
            "Smooth every pixel's time trace (along time only, never across pixels). The smoothed counts are\n"
            "used everywhere: count map, traces, 2D fits, σ²(t), decay map.\n"
            "• Savitzky–Golay keeps the rise and peak shape best; it can go negative on weak tails.\n"
            "• Moving average / Gaussian: steady smoothing, counts stay positive.\n"
            "• Adaptive Gaussian: σ_k = max(σ_min, f·(t − t_ref)), heavy only in the weak late tail; it bends\n"
            "  σ²(t) slightly (D bias ≈ 4f²(t − t_ref)/τ for a lifetime τ).\n"
            "Smoothing mixes neighbouring frames: σ² shifts by ≈ −2D·σ_k²/τ (the slope is unbiased to first\n"
            "order); frames within ~2σ_k of the rising edge are blurred, so start the D window a few kernel\n"
            "widths after the reference. The error on D is scaled for the frames it correlates.")
        self.sm1Lbl, self.sm2Lbl = QtWidgets.QLabel(""), QtWidgets.QLabel("")
        self.sm1Spin = _dspin(0.0, 1000.0, 0.0, 2, 1.0, width=86)
        self.sm2Spin = _dspin(0.0, 1000.0, 0.0, 2, 1.0, width=86)
        self.smInfo = QtWidgets.QLabel("")
        self.smInfo.setStyleSheet("color:#555")
        self._smooth_vals = {"sg": [11.0, 2.0], "ma": [5.0, 0.0], "gauss": [1.0, 0.0], "adapt": [0.5, 0.05]}
        self._smooth_method = None
        r4.addWidget(QtWidgets.QLabel("Rebin"))
        r4.addWidget(self.rebinSpin)
        r4.addWidget(self.binLbl)
        r4.addSpacing(16)
        r4.addWidget(QtWidgets.QLabel("Trace smoothing"))
        r4.addWidget(self.smoothCombo)
        for w in (self.sm1Lbl, self.sm1Spin, self.sm2Lbl, self.sm2Spin):
            r4.addWidget(w)
        r4.addWidget(self.smInfo, 1)
        v.addLayout(r4)
        self._configure_smooth_spins()

        self.infoLbl = QtWidgets.QLabel("")
        self.infoLbl.setStyleSheet("color:#555; font-size: 8pt")
        self.infoLbl.setWordWrap(True)
        v.addWidget(self.infoLbl)

        self.loadBtn.clicked.connect(self._on_load_clicked)
        self.reloadBtn.clicked.connect(self._on_reload_clicked)
        self.pump2Combo.currentIndexChanged.connect(self._on_pump2_changed)
        self.bgCombo.currentIndexChanged.connect(self._on_bg_changed)
        self.darkBtn.clicked.connect(self._on_load_dark)
        self.noDarkBtn.clicked.connect(self._on_clear_dark)
        self.powerCalBtn.clicked.connect(self._on_power_calib)
        self.voltBar.valueChanged.connect(self._on_volt_bar)
        self.powerBar.valueChanged.connect(self._on_power_bar)
        self.timeBar.valueChanged.connect(self._on_time_bar)
        for w in (self.beginEdit, self.endEdit, self.refEdit):
            w.editingFinished.connect(self._on_times_edited)
        self.alignCheck.toggled.connect(self._on_align_toggled)
        self.rebinSpin.valueChanged.connect(lambda _v: self._on_fit_setting_changed(reload_view=True))
        self.smoothCombo.currentIndexChanged.connect(lambda _i: self._on_smooth_method())
        for s in (self.sm1Spin, self.sm2Spin):
            s.valueChanged.connect(lambda _v: self._on_smooth_param())
        self.fixOffsetCheck.toggled.connect(lambda _c: self._on_fit_setting_changed())
        self.exclEdit.editingFinished.connect(self._on_excl_edited)
        return box

    def _make_point_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        ctrl = QtWidgets.QHBoxLayout()
        self.cropSpins = []
        ctrl.addWidget(QtWidgets.QLabel("Crop T/B/L/R"))
        for _ in range(4):
            s = QtWidgets.QSpinBox()
            s.setRange(0, 4000)
            s.setFixedWidth(60)
            ctrl.addWidget(s)
            self.cropSpins.append(s)
        self.cropApplyBtn = QtWidgets.QPushButton("Apply")
        ctrl.addWidget(self.cropApplyBtn)
        ctrl.addSpacing(12)
        self.baselineCheck = QtWidgets.QCheckBox("Baseline (subtract per-row mean of)")
        self.blLo = _dspin(200, 2000, 1050.0, 1, 5, " nm")
        self.blHi = _dspin(200, 2000, 1100.0, 1, 5, " nm")
        ctrl.addWidget(self.baselineCheck)
        ctrl.addWidget(self.blLo)
        ctrl.addWidget(QtWidgets.QLabel("to"))
        ctrl.addWidget(self.blHi)
        ctrl.addStretch(1)
        tv.addLayout(ctrl)
        self.frameView = SingleFrameView("PL")
        tv.addWidget(self.frameView, 1)
        split.addWidget(top)

        mid = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(2)
        vrow = QtWidgets.QHBoxLayout()
        self.viewCombo = QtWidgets.QComboBox()
        self.viewCombo.addItems(["2D", "Spectral"])
        self.viewCombo.setToolTip("2D: the SPAD as an image (2D Gaussian fits).\n"
                                  "Spectral: the SPAD's centre line on a wavelength axis (SPAD93: the 17 columns "
                                  "at y = 92 µm; SPAD23: the 5 rows at x = 46 µm), as the SPAD GUIs' spectral mode.\n"
                                  "Set from the scan's header (spad_view_mode) when a scan is loaded.")
        self.wlcalLbl = QtWidgets.QLabel("λ-cal: none")
        self.wlcalLbl.setStyleSheet("color:#555")
        self.wlcalBtn = QtWidgets.QPushButton("Load λ-cal…")
        self.wlcalBtn.setToolTip("Wavelength calibration (WLCAL_*.json of the SPAD GUIs). Default: the newest one in "
                                 "the scan folder, else the one written into the scan's header.")
        self.wlcalClearBtn = QtWidgets.QPushButton("Clear")
        self.wlcalClearBtn.setToolTip("No calibration: the spectral axis is the line position (µm)")
        self.specLogCheck = QtWidgets.QCheckBox("Log image")
        self.specLogCheck.setToolTip("Spectral image as log10(1 + counts − background)")
        self.specLogCheck.setChecked(str(_settings().value("spectral/log_image", "false")).lower() == "true")
        vrow.addWidget(QtWidgets.QLabel("TRPL view"))
        vrow.addWidget(self.viewCombo)
        vrow.addSpacing(10)
        vrow.addWidget(self.wlcalBtn)
        vrow.addWidget(self.wlcalClearBtn)
        vrow.addWidget(self.specLogCheck)
        vrow.addWidget(self.wlcalLbl, 1)
        mv.addLayout(vrow)
        mh = QtWidgets.QHBoxLayout()
        mh.setContentsMargins(0, 0, 0, 0)
        self.pixMap = ContourPixelMap()
        self.pixMap.setMinimumWidth(320)
        mh.addWidget(self.pixMap, 3)
        self.frameFitLbl = QtWidgets.QLabel("")
        self.frameFitLbl.setWordWrap(True)
        self.frameFitLbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.frameFitLbl.setTextFormat(QtCore.Qt.RichText)
        # spectral view: time-resolved spectrum image + spectrum at the time cursor
        specW = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.specMap = XMapPlot("inferno")
        self.specMap.marker.setVisible(False)
        self._specCursor = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#06b6d4", width=1.3))
        self._specCursor.setZValue(12)
        self.specMap.plot.addItem(self._specCursor)
        self._specLineItems: list = []
        specW.addWidget(self.specMap)
        sr = QtWidgets.QWidget()
        srv = QtWidgets.QVBoxLayout(sr)
        srv.setContentsMargins(0, 0, 0, 0)
        self.specPlot = pg.PlotWidget()
        _style(self.specPlot.getPlotItem(), "Counts − background", "Wavelength (nm)")
        self._specPts = pg.PlotDataItem(pen=pg.mkPen("#1f77b4", width=1.6), symbol="o", symbolSize=6,
                                        symbolBrush="#1f77b4", symbolPen=None)
        self._specFit = pg.PlotDataItem(pen=pg.mkPen("#dc2626", width=1.6, style=QtCore.Qt.DashLine))
        self.specPlot.addItem(self._specPts)
        self.specPlot.addItem(self._specFit)
        self._specMarks: list = []
        srv.addWidget(self.specPlot, 1)
        self.specLbl = QtWidgets.QLabel("")
        self.specLbl.setWordWrap(True)
        self.specLbl.setTextFormat(QtCore.Qt.RichText)
        srv.addWidget(self.specLbl)
        specW.addWidget(sr)
        specW.setStretchFactor(0, 3)
        specW.setStretchFactor(1, 2)
        self.midStack = QtWidgets.QStackedWidget()
        self.midStack.addWidget(self.frameFitLbl)
        self.midStack.addWidget(specW)
        mh.addWidget(self.midStack, 2)
        self._midLay = mh
        mv.addLayout(mh, 1)
        split.addWidget(mid)

        bot = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bot)
        bv.setContentsMargins(0, 0, 0, 0)
        brow = QtWidgets.QHBoxLayout()
        self.traceLbl = QtWidgets.QLabel("Decay of the selected pixel (left click on the count map)")
        self.logCheck = QtWidgets.QCheckBox("Log y")
        self.logCheck.setChecked(True)
        self.rawCheck = QtWidgets.QCheckBox("Show raw")
        self.rawCheck.setChecked(True)
        self.rawCheck.setToolTip("With trace smoothing on: the unsmoothed trace in light grey behind the smoothed one")
        brow.addWidget(self.traceLbl, 1)
        brow.addWidget(self.rawCheck)
        brow.addWidget(self.logCheck)
        bv.addLayout(brow)
        self.tracePlot = pg.PlotWidget()
        _style(self.tracePlot.getPlotItem(), "Counts", "Time (ns)")
        self._traceRaw = self.tracePlot.plot(pen=pg.mkPen("#c7cbd1", width=1.0))
        self._traceCurve = self.tracePlot.plot(pen=pg.mkPen("#1f77b4", width=1.4))
        self._traceBg = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#9ca3af", width=1, style=QtCore.Qt.DashLine))
        self._traceRegion = pg.LinearRegionItem(brush=pg.mkBrush(250, 204, 21, 35), movable=False)
        self._traceRegion.setZValue(-10)
        self._traceRef = pg.InfiniteLine(angle=90, movable=False,
                                         pen=pg.mkPen("#16a34a", width=1.5, style=QtCore.Qt.DashDotLine))
        self._traceCursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#f97316", width=1.5))
        for it in (self._traceRegion, self._traceRef, self._traceCursor):
            self.tracePlot.addItem(it)
        self.tracePlot.addItem(self._traceBg)
        botSplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        botSplit.addWidget(self.tracePlot)
        # spectral view: 1D Gaussian width along the active line → profile map + σ²(t)
        self.lineFitW = QtWidgets.QWidget()
        lw = QtWidgets.QVBoxLayout(self.lineFitW)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setSpacing(2)
        self.lineCtlW = QtWidgets.QWidget()           # in the row above the decay plot (spectral view only)
        lrow = QtWidgets.QHBoxLayout(self.lineCtlW)
        lrow.setContentsMargins(0, 0, 0, 0)
        self.lineProfCombo = QtWidgets.QComboBox()
        self.lineProfCombo.addItems(["Fitted Gaussian", "Measured counts"])
        self.lineProfCombo.setToolTip("Fitted: A·exp(−d²/2σ²) of each frame's 1D fit (offset left out).\n"
                                      "Measured: the line's background-subtracted counts against the distance "
                                      "from that frame's fitted centre (linear between pixels).")
        self.lineProfNorm = QtWidgets.QCheckBox("Normalize each time")
        self.lineProfNorm.setToolTip("Fitted: exp(−d²/2σ²); measured: (counts − offset) / A, so the fitted peak = 1")
        self.lineProfRSpin = _dspin(0.0, 100000.0, 0.0, 1, 10.0, " µm", width=90)
        self.lineProfRSpin.setSpecialValueText("auto")
        self.lineProfRSpin.setToolTip("Fix the profile map's x axis to −max … +max (µm). 0 = auto.")
        st = _settings()
        try:
            self.lineProfCombo.setCurrentIndex(int(st.value("spectral/prof_mode", 1)))
            self.lineProfNorm.setChecked(str(st.value("spectral/prof_norm", "true")).lower() == "true")
            self.lineProfRSpin.setValue(float(st.value("spectral/prof_r_max", 0.0)))
        except (TypeError, ValueError):
            pass
        lrow.addWidget(QtWidgets.QLabel("Profile"))
        lrow.addWidget(self.lineProfCombo)
        lrow.addWidget(self.lineProfNorm)
        lrow.addWidget(QtWidgets.QLabel("x max ±"))
        lrow.addWidget(self.lineProfRSpin)
        self.lineFreeOffCheck = QtWidgets.QCheckBox("Free offset (≥ 7 px)")
        self.lineFreeOffCheck.setToolTip(
            "1D fit of the active line: offset fitted (only on lines with ≥ 7 enabled pixels) instead of held at 0\n"
            "after the background subtraction (as spad93's column fits). Steps of the sequential fit limit the\n"
            "offset to ±25 % per frame, so most frames end up flagged with it free.")
        self.lineFreeOffCheck.setChecked(str(st.value("spectral/free_offset", "false")).lower() == "true")
        lrow.addWidget(self.lineFreeOffCheck)
        lrow.addSpacing(16)
        brow.insertWidget(1, self.lineCtlW)
        self.lineCtlW.setVisible(False)
        lsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.lineProfMap = XMapPlot("inferno")
        self.lineProfMap.marker.setVisible(False)
        self._lineProfPlus = pg.PlotDataItem(pen=pg.mkPen("w", width=1.8), connect="finite")
        self._lineProfMinus = pg.PlotDataItem(pen=pg.mkPen("w", width=1.8), connect="finite")
        self._lineProfCursor = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#06b6d4", width=1.2))
        for it in (self._lineProfPlus, self._lineProfMinus, self._lineProfCursor):
            it.setZValue(15)
            self.lineProfMap.plot.addItem(it)
        lsplit.addWidget(self.lineProfMap)
        self.lineSigPlot = pg.PlotWidget()
        _style(self.lineSigPlot.getPlotItem(), "σ² (µm²)", "Time (ns)")
        self._lineSigGood = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#1f77b4"), brush=pg.mkBrush("#1f77b4"))
        self._lineSigBad = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#1f77b4"), brush=None)
        self._lineSigLine = pg.PlotDataItem(pen=pg.mkPen("#dc2626", width=2, style=QtCore.Qt.DashLine))
        self._lineSigSmooth = pg.PlotDataItem(pen=pg.mkPen("#f59e0b", width=2.2))
        self._lineSigCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f97316", width=1.2))
        self._lineSigRef = pg.InfiniteLine(angle=90, movable=False,
                                           pen=pg.mkPen("#16a34a", width=1.2, style=QtCore.Qt.DashDotLine))
        for it in (self._lineSigGood, self._lineSigBad, self._lineSigSmooth, self._lineSigLine,
                   self._lineSigCursor, self._lineSigRef):
            self.lineSigPlot.addItem(it)
        lsplit.addWidget(self.lineSigPlot)
        lw.addWidget(lsplit, 1)
        self.lineFitLbl = QtWidgets.QLabel("")
        self.lineFitLbl.setWordWrap(True)
        self.lineFitLbl.setTextFormat(QtCore.Qt.RichText)
        lw.addWidget(self.lineFitLbl)
        botSplit.addWidget(self.lineFitW)
        botSplit.setStretchFactor(0, 1)
        botSplit.setStretchFactor(1, 2)
        self.lineFitW.setVisible(False)
        bv.addWidget(botSplit, 1)
        split.addWidget(bot)
        split.setSizes([400, 320, 320])
        self._pointSplit = split
        self._view_was_spec = False
        v.addWidget(split)

        self.cropApplyBtn.clicked.connect(self._on_apply_crop)
        self.baselineCheck.toggled.connect(lambda _c: self._on_pl_settings())
        for s in (self.blLo, self.blHi):
            s.valueChanged.connect(lambda _v: self._on_pl_settings())
        self.frameView.linecut_changed.connect(lambda _r: self._mark_dep_dirty())
        self.frameView.linecutWidthSpin.valueChanged.connect(lambda _v: self._mark_dep_dirty())
        self.pixMap.pixel_clicked.connect(self._on_pixel_clicked)
        self.logCheck.toggled.connect(lambda _c: self._draw_trace())
        self.rawCheck.toggled.connect(lambda _c: self._draw_trace())
        self._traceCursor.sigPositionChangeFinished.connect(
            lambda ln: self._set_cursor_time(float(ln.value())))
        self._specTraceCurves: list = []
        self.tracePlot.getPlotItem().addLegend(offset=(-8, 8), brush=pg.mkBrush(255, 255, 255, 210),
                                               pen=pg.mkPen("#9ca3af"))
        self._wlcal: Optional[WavelengthCal] = None
        self._wlcal_note = ""
        self._spec_lines: List[int] = []         # selected spectral lines (the last one = active)
        self.viewCombo.currentIndexChanged.connect(lambda _i: self._on_view_mode())
        self.wlcalBtn.clicked.connect(self._on_load_wlcal)
        self.wlcalClearBtn.clicked.connect(self._on_clear_wlcal)
        self.specLogCheck.toggled.connect(self._on_spec_log)
        self.specMap.plot.scene().sigMouseClicked.connect(self._on_spec_map_click)
        self._line_cache: "OrderedDict" = OrderedDict()    # (path, fit key, line) → (PointDiffusion | None, error)
        self._line_req = None
        self._line_text = ""
        self._line_job = None
        self._line_pending = None
        self.lineProfCombo.currentIndexChanged.connect(self._on_line_prof_setting)
        self.lineProfNorm.toggled.connect(self._on_line_prof_setting)
        self.lineProfRSpin.valueChanged.connect(self._on_line_prof_setting)
        self.lineFreeOffCheck.toggled.connect(self._on_line_free_offset)
        return w

    def _make_fit_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        self.fitPointBtn = QtWidgets.QPushButton("Fit this point")
        self.fitPointBtn.setToolTip("Sequential 2D Gaussian fit of every frame from the reference to the end → σ_eq²(t)")
        self.fitAllBtn = QtWidgets.QPushButton("Fit all")
        self.fitAllBtn.setToolTip("Fit every point of the current sweep line (tab 4: gate or power dependence); "
                                  "fills the σ_eq²(t) map of tab 4")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.cancelBtn.setEnabled(False)
        self.fitProg = QtWidgets.QProgressBar()
        self.fitProg.setFixedWidth(180)
        self.fitProg.setVisible(False)
        self.d0Spin = _dspin(0.0, 1e5, 0.0, 2, 0.5, " ns", width=90)
        self.d1Spin = _dspin(0.0, 1e5, 0.0, 2, 0.5, " ns", width=90)
        self.d1Spin.setSpecialValueText("end")
        self.flaggedCheck = QtWidgets.QCheckBox("Include flagged")
        self.flaggedCheck.setToolTip("Use flagged frames (fit at a bound, or amplitude/error < 3) in the D line "
                                     "and the σ_eq² map")
        for x in (self.fitPointBtn, self.fitAllBtn, self.cancelBtn, self.fitProg):
            row.addWidget(x)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("D line from t_ref +"))
        row.addWidget(self.d0Spin)
        row.addWidget(QtWidgets.QLabel("to +"))
        row.addWidget(self.d1Spin)
        row.addWidget(self.flaggedCheck)
        tip = ("Savitzky–Golay smoothing of σ_eq²(t): a polynomial of this order fitted over this many frames\n"
               "around each time (on their real times, so frames left out as flagged are simply missing; flagged\n"
               "frames are used only with 'Include flagged'). The D line is fitted to the smoothed curve, with its\n"
               "error scaled for the frames the smoothing correlates. The tab-4 σ² map shows the smoothed curve\n"
               "and fills gaps between good frames.")
        self.s2winSpin = QtWidgets.QSpinBox()
        self.s2winSpin.setRange(1, 301)
        self.s2winSpin.setSingleStep(2)
        self.s2winSpin.setValue(1)
        self.s2winSpin.setSpecialValueText("off")
        self.s2winSpin.setSuffix(" frames")
        self.s2winSpin.setFixedWidth(96)
        self.s2ordSpin = QtWidgets.QSpinBox()
        self.s2ordSpin.setRange(1, 5)
        self.s2ordSpin.setValue(2)
        self.s2ordSpin.setFixedWidth(48)
        for s in (self.s2winSpin, self.s2ordSpin):
            s.setKeyboardTracking(False)
            s.setToolTip(tip)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("σ² smoothing (S-G) window"))
        row.addWidget(self.s2winSpin)
        row.addWidget(QtWidgets.QLabel("order"))
        row.addWidget(self.s2ordSpin)
        row.addStretch(1)
        v.addLayout(row)

        rowR = QtWidgets.QHBoxLayout()
        self.refModeCombo = QtWidgets.QComboBox()
        self.refModeCombo.addItems(["Free", "Common (median of the sweep line)", "Fixed"])
        self.refModeCombo.setToolTip(
            "Width of the 2D Gaussian in the REFERENCE frame (later frames are then fitted from it as usual):\n"
            "• Free: fitted at every point (as before).\n"
            "• Common: every point's reference frame is fitted freely first, then all points are fitted with the\n"
            "  median σx, σy of the sweep line held (for spots that start at the same size).\n"
            "• Fixed: the σx, σy typed here.")
        self.refSxSpin = _dspin(0.5, 1000.0, 40.0, 2, 1.0, " µm", width=92)
        self.refSySpin = _dspin(0.5, 1000.0, 40.0, 2, 1.0, " µm", width=92)
        self.s0FixCheck = QtWidgets.QCheckBox("D line through σ²(t_ref)")
        self.s0FixCheck.setToolTip("Hold the line's intercept at the reference frame's σ_eq² (the common / fixed width,\n"
                                   "or this point's own reference fit) and fit only the slope.")
        self.refInfo = QtWidgets.QLabel("")
        self.refInfo.setStyleSheet("color:#555")
        rowR.addWidget(QtWidgets.QLabel("Reference width"))
        rowR.addWidget(self.refModeCombo)
        rowR.addWidget(QtWidgets.QLabel("σx"))
        rowR.addWidget(self.refSxSpin)
        rowR.addWidget(QtWidgets.QLabel("σy"))
        rowR.addWidget(self.refSySpin)
        rowR.addWidget(self.s0FixCheck)
        rowR.addWidget(self.refInfo, 1)
        v.addLayout(rowR)

        rowL = QtWidgets.QHBoxLayout()
        self.lateCombo = QtWidgets.QComboBox()
        self.lateCombo.addItems(["Off", "×N after the threshold", "Equal counts"])
        self.lateCombo.setToolTip(
            "Group the late, weak frames for the σ² fits only (count map, traces and decay map keep every bin).\n"
            "Once the total signal (background-subtracted, enabled pixels) falls below the threshold × its peak:\n"
            "• ×N: every N bins become one frame;\n"
            "• Equal counts: each frame collects at least as many counts as one bin at the threshold (frames widen\n"
            "  as the signal decays, ~constant signal-to-noise).\n"
            "Each group is fitted on its mean frame, placed at its signal-weighted mean time.")
        self.lateFracSpin = _dspin(0.01, 0.95, 0.368, 3, 0.05, width=80)
        self.lateFracSpin.setToolTip("Fraction of the peak signal where grouping starts (0.368 = 1/e)")
        self.lateNSpin = QtWidgets.QSpinBox()
        self.lateNSpin.setRange(2, 400)
        self.lateNSpin.setValue(4)
        self.lateNSpin.setFixedWidth(64)
        self.lateNSpin.setKeyboardTracking(False)
        rowL.addWidget(QtWidgets.QLabel("Late frames"))
        rowL.addWidget(self.lateCombo)
        rowL.addWidget(QtWidgets.QLabel("threshold"))
        rowL.addWidget(self.lateFracSpin)
        rowL.addWidget(QtWidgets.QLabel("× peak   N"))
        rowL.addWidget(self.lateNSpin)
        rowL.addStretch(1)
        v.addLayout(rowL)
        row2 = QtWidgets.QHBoxLayout()
        self.profCombo = QtWidgets.QComboBox()
        self.profCombo.addItems(["Fitted Gaussian", "Measured counts"])
        self.profCombo.setToolTip("Fitted: A·exp(−r²/2σ_eq²) of each frame's fit (offset left out).\n"
                                  "Measured: each frame's background-subtracted counts against the distance from "
                                  "that frame's fitted centre.")
        self.profNormCheck = QtWidgets.QCheckBox("Normalize each time to [0, 1]")
        self.profNormCheck.setToolTip("Fitted: exp(−r²/2σ_eq²); measured: (counts − offset) / A, so the fitted peak = 1")
        row2.addWidget(QtWidgets.QLabel("Profile map (right of σ_eq²(t)):"))
        row2.addWidget(self.profCombo)
        row2.addWidget(self.profNormCheck)
        self.profRSpin = _dspin(0.0, 100000.0, 0.0, 1, 10.0, " µm", width=100)
        self.profRSpin.setSpecialValueText("auto")
        self.profRSpin.setToolTip("Fix the map's x axis to −max … +max (µm). 0 = auto (fit the data).")
        try:
            self.profRSpin.setValue(float(_settings().value("profile/r_max", 0.0)))
        except (TypeError, ValueError):
            pass
        row2.addWidget(QtWidgets.QLabel("x max ±"))
        row2.addWidget(self.profRSpin)
        row2.addWidget(QtWidgets.QLabel("— white lines: ±σ_eq(t) of the σ² used for the D line"))
        row2.addStretch(1)
        v.addLayout(row2)
        self.fitLbl = QtWidgets.QLabel("Not fitted.")
        self.fitLbl.setWordWrap(True)
        self.fitLbl.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(self.fitLbl)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.radialPlot = pg.PlotWidget()
        _style(self.radialPlot.getPlotItem(), "Counts (background subtracted)", "Distance from the fitted centre (µm)")
        self._radPts = pg.ScatterPlotItem(size=7, pen=pg.mkPen("#1f2937", width=1), brush=pg.mkBrush("#60a5fa"))
        self._radCurve = pg.PlotDataItem(pen=pg.mkPen("#dc2626", width=2))
        self.radialPlot.addItem(self._radPts)
        self.radialPlot.addItem(self._radCurve)
        split.addWidget(self.radialPlot)
        self.sigPlot = pg.PlotWidget()
        _style(self.sigPlot.getPlotItem(), "σ_eq² (µm²)", "Time (ns)")
        self._sigGood = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#1f77b4"), brush=pg.mkBrush("#1f77b4"))
        self._sigBad = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#1f77b4"), brush=None)
        self._sigLine = pg.PlotDataItem(pen=pg.mkPen("#dc2626", width=2, style=QtCore.Qt.DashLine))
        self._sigSmooth = pg.PlotDataItem(pen=pg.mkPen("#f59e0b", width=2.2))
        self._sigCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f97316", width=1.2))
        self._sigRef = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen("#16a34a", width=1.2, style=QtCore.Qt.DashDotLine))
        for it in (self._sigGood, self._sigBad, self._sigSmooth, self._sigLine, self._sigCursor, self._sigRef):
            self.sigPlot.addItem(it)
        low = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        low.addWidget(self.sigPlot)
        self.profMap = XMapPlot("inferno")
        self.profMap.marker.setVisible(False)
        self._profPlus = pg.PlotDataItem(pen=pg.mkPen("w", width=1.8), connect="finite")
        self._profMinus = pg.PlotDataItem(pen=pg.mkPen("w", width=1.8), connect="finite")
        self._profCursor = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#06b6d4", width=1.2))
        for it in (self._profPlus, self._profMinus, self._profCursor):
            it.setZValue(15)
            self.profMap.plot.addItem(it)
        low.addWidget(self.profMap)
        low.setStretchFactor(0, 3)
        low.setStretchFactor(1, 2)
        split.addWidget(low)
        v.addWidget(split, 1)
        self.fitPointBtn.clicked.connect(self._on_fit_point)
        self.fitAllBtn.clicked.connect(self._on_fit_all)
        self.cancelBtn.clicked.connect(self._on_cancel_fit)
        for s in (self.d0Spin, self.d1Spin):
            s.valueChanged.connect(lambda _v: (self._draw_sigma_plot(), self._update_smooth_info()))
        for s in (self.s2winSpin, self.s2ordSpin):
            s.valueChanged.connect(lambda _v: (self._draw_sigma_plot(), self._mark_dep_dirty()))
        self.refModeCombo.currentIndexChanged.connect(lambda _i: self._on_ref_setting())
        for s in (self.refSxSpin, self.refSySpin):
            s.valueChanged.connect(lambda _v: self._on_ref_setting() if self.refModeCombo.currentIndex() == 2 else None)
        self.s0FixCheck.toggled.connect(lambda _c: (self._draw_sigma_plot(), self._mark_dep_dirty()))
        self.lateCombo.currentIndexChanged.connect(lambda _i: self._on_ref_setting())
        self.lateFracSpin.valueChanged.connect(lambda _v: self._on_ref_setting())
        self.lateNSpin.valueChanged.connect(lambda _v: self._on_ref_setting())
        self._update_ref_info()
        self.profCombo.currentIndexChanged.connect(lambda _i: self._draw_profile_map())
        self.profNormCheck.toggled.connect(lambda _c: self._draw_profile_map())
        self.profRSpin.valueChanged.connect(self._on_prof_range)
        self.flaggedCheck.toggled.connect(lambda _c: (self._draw_sigma_plot(), self._mark_dep_dirty()))
        return w

    def _make_dep_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        self.depCombo = QtWidgets.QComboBox()
        self.depCombo.addItems([self.MODE_GATE, self.MODE_POWER])
        self.normCheck = QtWidgets.QCheckBox("Normalize each column")
        self.normCheck.setChecked(True)
        self.logColCheck = QtWidgets.QCheckBox("Log colour")
        self.logColCheck.setChecked(True)
        self.depInfo = QtWidgets.QLabel("")
        self.depInfo.setStyleSheet("color:#555")
        row.addWidget(QtWidgets.QLabel("Dependence"))
        row.addWidget(self.depCombo)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Decay map:"))
        row.addWidget(self.normCheck)
        row.addWidget(self.logColCheck)
        row.addSpacing(12)
        self.pickCheck = QtWidgets.QCheckBox("Click picks linecuts")
        self.pickCheck.setChecked(True)
        self.pickCheck.setToolTip("On: clicking a column of any map adds it to the linecut plots on the right "
                                  "(click it again to remove it); Shift+click moves the voltage / power cursor.\n"
                                  "Off: clicking moves the cursor.")
        self.clearPicksBtn = QtWidgets.QPushButton("Clear linecuts")
        row.addWidget(self.pickCheck)
        row.addWidget(self.clearPicksBtn)
        row.addWidget(self.depInfo, 1)
        v.addLayout(row)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plMap = XMapPlot("viridis")
        self.decayMap = XMapPlot("inferno")
        self.sigmaMap = XMapPlot("viridis")
        for m in (self.plMap, self.decayMap, self.sigmaMap):
            split.addWidget(m)
            m.column_clicked.connect(self._on_dep_map_click)
        # second column: the picked columns as curves
        self._picks: List[int] = []
        self._picks_key = None
        self._pick_lines: Dict[int, list] = {}
        self._dep_pl = self._dep_decay = self._dep_sig = None
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.pickPlPlot = pg.PlotWidget()
        _style(self.pickPlPlot.getPlotItem(), "PL counts", "Photon energy (eV)")
        self.pickDecayPlot = pg.PlotWidget()
        _style(self.pickDecayPlot.getPlotItem(), "Counts", "Time (ns)")
        self.pickSigPlot = pg.PlotWidget()
        _style(self.pickSigPlot.getPlotItem(), "σ_eq² (µm²)", "Time (ns)")
        for p in (self.pickPlPlot, self.pickDecayPlot, self.pickSigPlot):
            p.getPlotItem().addLegend(offset=(-8, 8), brush=pg.mkBrush(255, 255, 255, 210),
                                      pen=pg.mkPen("#9ca3af"))
            right.addWidget(p)
        outer = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split)
        outer.addWidget(right)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 2)
        v.addWidget(outer, 1)
        self.depCombo.currentIndexChanged.connect(lambda _i: self._on_dep_mode_changed())
        self.normCheck.toggled.connect(lambda _c: self._draw_decay_map())
        self.logColCheck.toggled.connect(lambda _c: self._draw_decay_map())
        self.clearPicksBtn.clicked.connect(self._clear_picks)
        return w

    def _make_scanfit_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        fl = QtWidgets.QLabel("Peak fit:")
        fl.setStyleSheet("font-weight:bold")
        self.scanFitCtl = PeakFitControls("scanfit", window=True, offset=True)
        self.scanFitInfo = QtWidgets.QLabel("")
        self.scanFitInfo.setStyleSheet("color:#555")
        self.scanFitInfo.setWordWrap(True)
        row.addWidget(fl)
        row.addWidget(self.scanFitCtl)
        row.addWidget(self.scanFitInfo, 1)
        v.addLayout(row)
        self.scanFitView = LinecutFitView(self._run, self.scanFitCtl.opts, key="scanfit")
        self.scanFitView.show_message("Load a scan.")
        v.addWidget(self.scanFitView, 1)
        self._scanfit_dirty = True
        self.scanFitCtl.changed.connect(self.scanFitView.refresh)
        self.scanFitView.column_clicked.connect(self._on_dep_column_clicked)
        return w

    # ══ jobs ═══════════════════════════════════════════════════════════
    def _run(self, fn, on_done, progress=None) -> _Job:
        self._job_id += 1
        job = _Job(self._job_id, fn, self)
        job.done.connect(lambda _id, res, err: on_done(res, err))
        if progress is not None:
            job.progress.connect(progress)
        job.finished.connect(lambda job=job: self._on_job_finished(job))
        self._jobs.append(job)
        job.start()
        return job

    def _on_job_finished(self, job: _Job) -> None:
        job.wait(1000)
        if job in self._jobs:
            self._jobs.remove(job)
        job.deleteLater()

    # ══ loading ════════════════════════════════════════════════════════
    def _on_load_clicked(self) -> None:
        start = str(self.model.folder.parent) if self.model else str(_settings().value("scan/last", "") or "")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a PL + TRPL dual-gate scan folder", start)
        if d:
            self.load_folder(d)

    def _on_reload_clicked(self) -> None:
        if self.model is not None:
            self.load_folder(self.model.folder, reload=True)

    def load_folder(self, folder, reload: bool = False) -> None:
        if self._load_job is not None:
            return
        scan = scan_folder(Path(folder))
        if not scan.states():
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc or TRPL_*.csv files in:\n{folder}")
            return
        states = scan.states()
        state = self._pump2 if self._pump2 in states else states[0]
        keep = None
        if reload and self.model is not None:
            keep = (float(self.model.vf[self._b]), float(self.model.vb[self._b]),
                    float(self.model.powers[self._pi]) if self.model.n_powers else 0.0)
        self.folderLbl.setText(f"Loading {scan.folder.name} …")
        self.loadBtn.setEnabled(False)
        self.reloadBtn.setEnabled(False)
        paths = scan.asc.get(state, [])

        def work(progress, cancelled):
            if paths:
                frames = _load_frames(scan.folder, paths, progress, cancelled)
            else:
                frames = _load_state_frames(scan, state)
            return ScanModel(scan, state, frames)

        self._load_job = self._run(
            work, lambda res, err: self._on_folder_loaded(scan, res, err, keep),
            lambda d, n: self.folderLbl.setText(f"Loading {scan.folder.name}: {d}/{n} PL files …"))

    def _on_folder_loaded(self, scan, model, err: str, keep) -> None:
        self._load_job = None
        self.loadBtn.setEnabled(True)
        self.reloadBtn.setEnabled(True)
        if err or model is None:
            self.folderLbl.setText(f"Could not load {scan.folder.name}: {err}")
            return
        same_folder = self.model is not None and self.model.folder == model.folder
        self.model = model
        self._pump2 = model.state
        _settings().setValue("scan/last", str(model.folder))
        if not same_folder or keep is None:
            self.store.clear()
            self._sweep_t0, self._t0_ready = None, False
        self._fits.clear()
        self._sigma_snap = None
        self._traces.clear()
        self._traces_t.clear()
        self._traces_vs = None
        self._frame_cache.clear()
        self._pl_cache.clear()
        self.folderLbl.setText(f"<b>{model.folder.name}</b> — {model.describe()}")
        self.folderLbl.setToolTip(str(model.folder))
        self._sync_pump2_combo(scan)
        if not same_folder or keep is None:
            self._apply_session(model)
        self._b, self._pi = 0, 0
        if keep is not None:
            d = (model.vf - keep[0]) ** 2 + (model.vb - keep[1]) ** 2
            self._b = int(np.argmin(d)) if d.size else 0
            self._pi = int(np.argmin(np.abs(model.powers - keep[2]))) if model.n_powers else 0
        self._update_power_axis()
        self._update_cut(refit=True)
        # dependence mode: gate if the scan has several gate points
        self.depCombo.blockSignals(True)
        self.depCombo.setCurrentText(self.MODE_GATE if model.n_points > 1 or model.n_powers < 2 else self.MODE_POWER)
        self.depCombo.blockSignals(False)
        self._sync_bars()
        self._start_alignment()
        self._on_point_changed()
        self._mark_dep_dirty()
        self.statusBar().showMessage(f"Loaded {model.folder.name}: {model.describe()}", 15000)

    def _apply_session(self, model: ScanModel) -> None:
        ses = dict(model.session)
        # no session.md (scan still running / aborted): the same values from the .asc header
        hm = model.meta or {}
        h_acq, h0, h1, href = (_parse_float(hm, k) for k in ("tcspc_acq_s", "hist_start_ns", "hist_end_ns",
                                                             "hist_ref_ns"))
        if "acq_s" not in ses and h_acq is not None and h_acq > 0:
            ses["acq_s"] = h_acq
        if "hist_ns" not in ses and h0 is not None and h1 is not None and h1 > h0:
            ses["hist_ns"] = (h0, h1)
        if "ref_ns" not in ses and href is not None:
            ses["ref_ns"] = href
        if "hist_after_t0" not in ses and "hist_times_after_t0" in hm:
            ses["hist_after_t0"] = str(hm.get("hist_times_after_t0")).strip().lower() in ("true", "1", "yes")
        if "ref_auto" not in ses and "hist_ref_auto" in hm:
            ses["ref_auto"] = str(hm.get("hist_ref_auto")).strip().lower() in ("true", "1", "yes")
        if "hist_ns" in ses:
            self.beginEdit.setText(f"{ses['hist_ns'][0]:g}")
            self.endEdit.setText(f"{ses['hist_ns'][1]:g}")
        else:
            self.beginEdit.setText("")
            self.endEdit.setText("")
        align = bool(ses.get("align", True))
        # reference: empty = time zero + T0_REF_BINS bins (aligned); the GUI's value only if it was set by hand
        # (older GUIs without the flag: only when the view is not aligned, as before)
        manual = ses.get("ref_auto") is False or ("ref_auto" not in ses and not align)
        self.refEdit.setText(f"{ses['ref_ns']:g}" if "ref_ns" in ses and manual else "")
        self._times_after_t0 = bool(ses.get("hist_after_t0", False))   # older GUIs: raw times
        self.alignCheck.blockSignals(True)
        self.alignCheck.setChecked(align)
        self.alignCheck.blockSignals(False)
        self._acq_s = ses.get("acq_s")
        # dark reference: the newest DARK_TRPL_*.csv the measuring GUI copied into the folder
        self._dark, self._dark_err = None, ""
        darks = sorted(model.folder.glob("DARK_TRPL_*.csv"), key=lambda p: p.stat().st_mtime)
        if darks:
            try:
                self._dark = load_dark_reference(darks[-1])
            except Exception as exc:
                self._dark_err = f"{darks[-1].name}: {exc}"
        want = ses.get("background") or ("dark_ref" if self._dark is not None else "pre_t0")
        self.bgCombo.blockSignals(True)
        self.bgCombo.setCurrentIndex(max(0, self.bgCombo.findData(want)))
        self.bgCombo.blockSignals(False)
        self._bg_fallback = False
        self._update_dark_label()
        # SPAD spectral mode: view from the header, λ-cal = newest WLCAL_*.json in the folder, else the header's
        meta = model.meta or {}
        self._wlcal, self._wlcal_note = None, ""
        cals = sorted(model.folder.glob("WLCAL_*.json"), key=lambda p: p.stat().st_mtime)
        if cals:
            try:
                self._wlcal = load_wlcal_json(cals[-1])
            except Exception as exc:
                self._wlcal_note = f"{cals[-1].name}: {exc}"
        if self._wlcal is None:
            self._wlcal = wlcal_from_header(meta)
        self._spec_lines = []
        self.viewCombo.blockSignals(True)
        self.viewCombo.setCurrentIndex(1 if str(meta.get("spad_view_mode", "")).strip().lower() == "spectral" else 0)
        self.viewCombo.blockSignals(False)
        self._apply_view_layout()
        self._update_wlcal_label()
        # PL crop and linecut as in the measuring GUI
        meta = model.meta or {}
        crop = tuple(_parse_int(meta, k, d) for k, d in zip(("crop_top", "crop_bottom", "crop_left", "crop_right"),
                                                             DEFAULT_CROP))
        self._crop = crop
        for s, val in zip(self.cropSpins, crop):
            s.setValue(int(val))
        self._pl_row_init = _parse_int_opt(meta, "linecut_row")
        w = _parse_int_opt(meta, "linecut_width")
        if w:
            self.frameView.linecutWidthSpin.blockSignals(True)
            self.frameView.linecutWidthSpin.setValue(int(w))
            self.frameView.linecutWidthSpin.blockSignals(False)

    def _sync_pump2_combo(self, scan) -> None:
        states = scan.states()
        self.pump2Combo.blockSignals(True)
        self.pump2Combo.clear()
        for s in states:
            self.pump2Combo.addItem(P2_LABELS[s], s)
        self.pump2Combo.setCurrentIndex(max(0, self.pump2Combo.findData(self._pump2)))
        self.pump2Combo.blockSignals(False)
        self.pump2Combo.setEnabled(len(states) > 1)

    def _on_pump2_changed(self, _i: int) -> None:
        st = self.pump2Combo.currentData()
        if st is None or self.model is None or st == self._pump2:
            return
        self._pump2 = str(st)
        self.load_folder(self.model.folder, reload=True)

    # ── dark reference ────────────────────────────────────────────────
    def _on_load_dark(self) -> None:
        start = str(self.model.folder) if self.model else ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Dark reference", start, "Dark TRPL (DARK_TRPL_*.csv *.csv)")
        if not fn:
            return
        try:
            self._dark, self._dark_err = load_dark_reference(fn), ""
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Dark reference", f"{Path(fn).name}: {exc}")
            return
        self.bgCombo.setCurrentIndex(self.bgCombo.findData("dark_ref"))
        self._update_dark_label()
        self._on_fit_setting_changed()

    def _on_clear_dark(self) -> None:
        self._dark, self._dark_err = None, "no dark reference (cleared)"
        self._update_dark_label()
        self._on_fit_setting_changed()

    def _dark_reason(self) -> str:
        if self._dark is None:
            return self._dark_err or "no DARK_TRPL_*.csv in this folder"
        if self._acq_s is None:
            return "session.md has no 'Acquisition (s)': the dark cannot be scaled"
        if self._dark.num_pixels != self.geom.num_pixels:
            return f"the dark reference has {self._dark.num_pixels} pixels, the data {self.geom.num_pixels}"
        return ""

    def _dark_scaled(self) -> Tuple[Optional[np.ndarray], str]:
        if self._dark_reason() or self._t.size < 2:
            return None, ""
        bin_ps = float(np.median(np.diff(self._t))) * 1000.0
        return self._dark.per_bin(self._acq_s, bin_ps), f"{Path(self._dark.path).name}|{self._acq_s:g}|{bin_ps:.6g}"

    def _update_dark_label(self) -> None:
        why = self._dark_reason()
        if self._dark is not None:
            self.darkLbl.setText(f"dark: {Path(self._dark.path).name}")
            self.darkLbl.setToolTip(self._dark.describe() + (f"\nNot usable: {why}" if why else ""))
        else:
            self.darkLbl.setText("dark: none")
            self.darkLbl.setToolTip(why)
        i = self.bgCombo.findData("dark_ref")
        item = self.bgCombo.model().item(i)
        item.setEnabled(not why)
        item.setToolTip(why)
        self.bgCombo.blockSignals(True)
        if why and self.bgCombo.currentIndex() == i:
            self.bgCombo.setCurrentIndex(self.bgCombo.findData("pre_t0"))
            self._bg_fallback = True
        elif not why and self._bg_fallback:
            self.bgCombo.setCurrentIndex(i)
            self._bg_fallback = False
        self.bgCombo.blockSignals(False)

    def _on_bg_changed(self, _i: int) -> None:
        self._bg_fallback = False
        self._on_fit_setting_changed()

    # ── power calibration ─────────────────────────────────────────────
    def _on_power_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load angle→power calibration", "",
                                                        "Excel / CSV (*.xlsx *.xls *.csv);;All files (*)")
        if not path:
            return
        p = Path(path)
        try:
            angles, powers_w, extra_a, extra_p = _parse_angle_power_calib(p)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Calibration load error", f"{p.name}\n\n{exc}")
            return
        if extra_a.size:
            ans = QtWidgets.QMessageBox.question(
                self, "Extra calibration points",
                f"{p.name} has {extra_a.size} numeric pair(s) outside the labelled 'angle'/'power' columns.\n\n"
                "Include them in the calibration?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
            if ans == QtWidgets.QMessageBox.Yes:
                angles = np.concatenate([angles, extra_a])
                powers_w = np.concatenate([powers_w, extra_p])
        angles, powers_w = _finalise_calib(angles, powers_w)
        if angles.size < 2:
            QtWidgets.QMessageBox.critical(self, "Calibration load error", f"{p.name}: need at least 2 distinct angles.")
            return
        self._angle_calib = (angles, powers_w)
        self.powerCalBtn.setToolTip(f"{p}\n{angles.size} points, {angles[0]:g}° – {angles[-1]:g}°")
        if self.model is not None:
            self._update_power_axis()
            self._sync_bars()
            self._update_labels()
            self._mark_dep_dirty()

    # ══ sweep axes ═════════════════════════════════════════════════════
    def _update_power_axis(self) -> None:
        m = self.model
        p = m.powers
        if m.angle_mode and self._angle_calib is not None:
            self._px = np.interp(p, *self._angle_calib) * 1e6
            self._plabel = "Power (µW)"
        elif m.angle_mode:
            self._px, self._plabel = p.astype(float), "Stage angle (°)"
        elif m.has_power:
            self._px, self._plabel = p * 1e6, "Power (µW)"
        else:
            self._px, self._plabel = np.zeros(p.size), "Power (not recorded)"
        self._porder = np.argsort(self._px, kind="stable")
        self.powerCalBtn.setEnabled(m.angle_mode)

    def _update_cut(self, refit: bool = False) -> None:
        m = self.model
        cal = self.ndTab.calib()
        self._cut = classify_linecut(m.vf, m.vb, cal, self.ndTab.tolerance())
        x = self._cut.x
        self._order = np.lexsort((m.vb, m.vf, x)) if x.size else np.zeros(0, int)
        self.ndTab.set_cut_info(f"Scan: {self._cut.info}.   Sweep axis: {self._cut.label}")
        self.ndTab.set_scan(m.vf, m.vb, self._order, self._b, refit=refit)

    def _sync_bars(self) -> None:
        m = self.model
        for bar, n, pos in ((self.voltBar, m.n_points, int(np.flatnonzero(self._order == self._b)[0]) if m.n_points else 0),
                            (self.powerBar, m.n_powers, int(np.flatnonzero(self._porder == self._pi)[0]) if m.n_powers else 0)):
            bar.blockSignals(True)
            bar.setRange(0, max(n - 1, 0))
            bar.setValue(pos)
            bar.setEnabled(n > 1)
            bar.blockSignals(False)

    def _on_calib_changed(self) -> None:
        if self.model is None:
            return
        self._update_cut()
        self._sync_bars()
        self._update_labels()
        self._mark_dep_dirty()

    def _on_nd_point_clicked(self, b: int) -> None:
        if self.model is not None and 0 <= b < self.model.n_points:
            self.voltBar.setValue(int(np.flatnonzero(self._order == b)[0]))

    def _on_volt_bar(self, pos: int) -> None:
        if self.model is None or not self._order.size:
            return
        self._b = int(self._order[int(np.clip(pos, 0, self._order.size - 1))])
        self._on_point_changed()

    def _on_power_bar(self, pos: int) -> None:
        if self.model is None or not self._porder.size:
            return
        self._pi = int(self._porder[int(np.clip(pos, 0, self._porder.size - 1))])
        self._on_point_changed()

    def _point_text(self, b: int) -> str:
        m = self.model
        x = self._cut.x[b] if self._cut is not None and self._cut.x.size else np.nan
        text = f"{self._cut.short} = {_fmt(x)}" if self._cut is not None else ""
        return text + f"   (V_f {m.vf[b]:+.4g} V, V_b {m.vb[b]:+.4g} V)"

    def _update_labels(self) -> None:
        m = self.model
        if m is None:
            return
        pos = int(np.flatnonzero(self._order == self._b)[0]) if m.n_points else 0
        self.voltLbl.setText(f"{self._point_text(self._b)}   [{pos + 1}/{m.n_points}]")
        ppos = int(np.flatnonzero(self._porder == self._pi)[0])
        self.powerLbl.setText(f"{self._plabel} = {_fmt(self._px[self._pi])}   [{ppos + 1}/{m.n_powers}]")

    # ══ current point ══════════════════════════════════════════════════
    def _on_point_changed(self) -> None:
        if self.model is None:
            return
        self._update_labels()
        self.ndTab.set_current(self._b)
        self._show_pl()
        self._load_point()
        line = tuple(self._line()[0])
        if line != getattr(self, "_drawn_line", None):      # another sweep line: redraw the maps
            self._mark_dep_dirty()
        else:
            self._update_dep_markers()

    def _view_settings(self) -> ViewSettings:
        return ViewSettings(self.alignCheck.isChecked(), self._sweep_t0 if self.alignCheck.isChecked() else None,
                            self.rebinSpin.value(), self._smooth_spec())

    # ── trace smoothing ───────────────────────────────────────────────
    _SMOOTH_SPINS = {   # method: ((label, decimals, lo, hi, step, suffix), second spin or None)
        "sg": (("window", 0, 3, 401, 2, " bins"), ("order", 0, 1, 6, 1, "")),
        "ma": (("N", 0, 1, 400, 1, " bins"), None),
        "gauss": (("σ", 2, 0.05, 200.0, 0.25, " ns"), None),
        "adapt": (("σ_min", 2, 0.05, 100.0, 0.25, " ns"), ("f", 3, 0.0, 1.0, 0.01, "")),
    }

    def _configure_smooth_spins(self) -> None:
        m = self.smoothCombo.currentData()
        cfg = self._SMOOTH_SPINS.get(m, (None, None))
        vals = self._smooth_vals.get(m, [0.0, 0.0])
        for k, (lbl, spin) in enumerate(((self.sm1Lbl, self.sm1Spin), (self.sm2Lbl, self.sm2Spin))):
            c = cfg[k]
            lbl.setVisible(c is not None)
            spin.setVisible(c is not None)
            if c is None:
                continue
            name, dec, lo, hi, step, suf = c
            spin.blockSignals(True)
            spin.setDecimals(dec)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setSuffix(suf)
            spin.setValue(vals[k])
            spin.blockSignals(False)
            lbl.setText(name)
        self._smooth_method = m
        self._update_smooth_info()

    def _smooth_spec(self):
        m = self.smoothCombo.currentData()
        if m is None:
            return None
        p1, p2 = float(self.sm1Spin.value()), float(self.sm2Spin.value())
        if m in ("sg", "ma"):
            p1 = float(int(round(p1)) | 1) if m == "sg" else float(int(round(p1)))
            p2 = float(int(round(p2))) if m == "sg" else 0.0
        if m in ("ma", "gauss"):
            p2 = 0.0
        t_ref = round(self._ref_ns(), 6) if m == "adapt" else 0.0     # only the adaptive kernel needs it
        return (m, p1, p2, t_ref)

    def _trace_corr(self) -> float:
        """Correlation length (bins) of the trace smoothing, the adaptive kernel at the middle of the D window."""
        spec = self._smooth_spec()
        if spec is None or self._t.size < 3:
            return 1.0
        w0, w1 = self._line_window()
        r0, r1 = self._range_idx()
        t_end = float(self._t[r1]) if w1 is None else self._ref_ns() + w1
        return smooth_corr_bins(spec, self._t, 0.5 * (self._ref_ns() + w0 + t_end))

    def _update_smooth_info(self) -> None:
        spec = self._smooth_spec()
        if spec is None or self._t.size < 3:
            self.smInfo.setText("")
            return
        dt = float(np.median(np.diff(self._t)))
        L = self._trace_corr()
        m = spec[0]
        if m == "sg":
            span = f"window {spec[1] * dt:.3g} ns"
        elif m == "ma":
            span = f"window {spec[1] * dt:.3g} ns"
        elif m == "gauss":
            span = f"FWHM {2.3548 * spec[1]:.3g} ns"
        else:
            span = f"σ_k = {spec[1]:.3g} ns at t_ref, {max(spec[1], spec[2] * 100):.3g} ns 100 ns later"
        self.smInfo.setText(f"{span};  ≈ {L:.1f} bins share their noise (D error × {np.sqrt(L):.2f})")

    def _on_smooth_method(self) -> None:
        old = self._smooth_method
        if old is not None:
            self._smooth_vals[old] = [float(self.sm1Spin.value()), float(self.sm2Spin.value())]
        self._configure_smooth_spins()
        self._on_fit_setting_changed(reload_view=True)

    def _on_smooth_param(self) -> None:
        m = self.smoothCombo.currentData()
        if m is not None:
            self._smooth_vals[m] = [float(self.sm1Spin.value()), float(self.sm2Spin.value())]
        self._update_smooth_info()
        self._on_fit_setting_changed(reload_view=True)

    def _num(self, edit: QtWidgets.QLineEdit) -> Optional[float]:
        try:
            v = float(edit.text())
        except ValueError:
            return None
        return v if np.isfinite(v) else None

    def _t_range(self) -> Optional[Tuple[float, float]]:
        b, e = self._num(self.beginEdit), self._num(self.endEdit)
        if b is None and e is None:
            return None
        if b is None:
            b = float(self._t[0]) if self._t.size else -1e12
        if e is None:
            e = float(self._t[-1]) if self._t.size else 1e12
        return tuple(sorted((b, e)))

    def _ref_ns(self) -> float:
        r = self._num(self.refEdit)
        if r is not None:
            return r
        if self.alignCheck.isChecked() and self._times_after_t0 and self._t.size:
            return float(self._t[t0_ref_index(self._t)])      # time zero + T0_REF_BINS bins
        tr = self._t_range()
        if tr is not None:
            return tr[0]
        return float(self._t[0]) if self._t.size else 0.0

    def _fit_settings_base(self) -> FitSettings:
        """Everything the reference-frame fit depends on (the common reference width is computed with these)."""
        key = self.bgCombo.currentData()
        dark, dark_id = (None, "")
        if key == "dark_ref":
            dark, dark_id = self._dark_scaled()
        return FitSettings(self._view_settings(), self._t_range(), self._ref_ns(), key == "pre_t0", None,
                           self.fixOffsetCheck.isChecked(), self.excluded, dark, dark_id)

    def _fit_settings(self, ref_sigma="current") -> FitSettingsX:
        b = self._fit_settings_base()
        rs = self._ref_sigma_now() if ref_sigma == "current" else ref_sigma
        return FitSettingsX(b.view, b.t_range, b.ref_ns, b.dark, b.pre_window, b.fix_offset, b.excluded,
                            b.dark_ref, b.dark_id, ref_sigma=rs, late=self._late_spec())

    # ── common reference width / late grouping settings ───────────────
    def _common_key(self):
        paths = tuple(str(self.model.trpl_path(*pt)) for pt in self._line()[0]
                      if self.model is not None and self.model.trpl_path(*pt) is not None)
        return paths, self._fit_settings_base().key()

    def _ref_sigma_now(self):
        """(σx, σy) to hold in the reference fit: None (free), the typed values, or the cached median of the
        sweep line (None while it has not been computed for the current line and settings)."""
        mode = self.refModeCombo.currentIndex()
        if mode == 2:
            return (float(self.refSxSpin.value()), float(self.refSySpin.value()))
        if mode == 1:
            c = self._ref_common
            if c is not None and c["key"] == self._common_key():
                return (c["sx"], c["sy"])
        return None

    def _late_spec(self):
        m = self.lateCombo.currentIndex()
        if m == 0:
            return None
        return ("fixed" if m == 1 else "counts", float(self.lateFracSpin.value()), int(self.lateNSpin.value()))

    def _update_ref_info(self) -> None:
        mode = self.refModeCombo.currentIndex()
        for s in (self.refSxSpin, self.refSySpin):
            s.setReadOnly(mode != 2)
            s.setEnabled(mode != 0)
        self.lateNSpin.setEnabled(self.lateCombo.currentIndex() == 1)
        self.lateFracSpin.setEnabled(self.lateCombo.currentIndex() != 0)
        if mode == 1:
            c = self._ref_common
            if c is not None and self.model is not None and c["key"] == self._common_key():
                for s, v in ((self.refSxSpin, c["sx"]), (self.refSySpin, c["sy"])):
                    s.blockSignals(True)
                    s.setValue(v)
                    s.blockSignals(False)
                self.refInfo.setText(f"median of {c['n_used']}/{c['n']} reference fits: σx = {c['sx']:.3g}, "
                                     f"σy = {c['sy']:.3g} µm (IQR {c['iqr_x']:.2g}, {c['iqr_y']:.2g})")
            else:
                self.refInfo.setText("computed at the next 'Fit this point' / 'Fit all' (median over the sweep line)")
        elif mode == 2:
            self.refInfo.setText(f"σ_eq at the reference = {np.sqrt((self.refSxSpin.value() ** 2 + self.refSySpin.value() ** 2) / 2):.3g} µm")
        else:
            self.refInfo.setText("")

    def _on_ref_setting(self) -> None:
        self._update_ref_info()
        self._on_fit_setting_changed()

    def _set_geometry(self, geom: SpadGeometry) -> None:
        if geom is self.geom:
            return
        self.geom = geom
        self.sel_pixel = geom.center_pixel
        self.excluded = set()
        self.exclEdit.setText("")
        self.pixMap.set_geometry(geom)
        self._spec_lines = []
        self._update_dark_label()
        self._update_wlcal_label()

    def _load_point(self) -> None:
        m = self.model
        path = m.trpl_path(self._pi, self._b)
        vs = self._view_settings()
        self._path = None if path is None else str(path)
        self._tz = None
        note = ""
        self._c_raw = None                   # unsmoothed counts (drawn faintly under the smoothed trace)
        if path is None:
            self._t, self._c = np.zeros(0), np.zeros((0, self.geom.num_pixels))
            self.infoLbl.setText("No TRPL file at this point.")
        else:
            try:
                data = self.store.load(path)
                self._set_geometry(data.geom)
                self._t, self._c, self._tz, note = point_view(self.store, data, vs)
                off = view_offset(data.time_ns, self.store.zero_of(data, self._tz, vs.t0_ref)) if vs.align else None
                # aligned: converted once the sweep's time zero is known (a point's own differs by a few ns)
                if not vs.align or (off is not None and self._t0_ready):
                    self._sync_time_frame(off)
                if vs.smooth is not None:
                    self._c_raw = point_view(self.store, data, ViewSettings(vs.align, vs.t0_ref, vs.rebin))[1]
            except Exception as exc:
                self._t, self._c = np.zeros(0), np.zeros((0, self.geom.num_pixels))
                self.infoLbl.setText(f"Could not load {Path(path).name}: {exc}")
                path = None
        if path is not None:
            parts = [f"{Path(path).name} [{self.geom.label}]"]
            if note:
                parts.append(note)
            if self.alignCheck.isChecked():
                parts.append(self._tz.message if self._tz is not None else
                             ("waiting for the alignment of all TRPL files …" if not self._t0_ready else ""))
            self.infoLbl.setText("   ".join(p for p in parts if p))
        self.binLbl.setText(f"= {np.median(np.diff(self._t)):.3g} ns" if self._t.size > 1 else "")
        self._update_smooth_info()
        self._update_dark_label()
        self._setup_time_bar()
        self._update_bg()
        self.pixMap.set_selected(self._sel_pixels(), self.excluded)
        self._draw_trace()
        self._draw_spectral()
        self._update_cursor()
        self._draw_sigma_plot()
        self._request_frame_fit()

    def _sync_time_frame(self, off: Optional[float]) -> None:
        """The view switches between raw times (off None) and times after time zero: move Begin / End / Ref
        by the offset so they stay on the same data."""
        after = off is not None
        if after:
            self._last_off = off
        self.refEdit.setPlaceholderText(f"t0+{T0_REF_BINS}" if after else "begin")
        if after == self._times_after_t0:
            return
        d = -float(off) if after else float(self._last_off or 0.0)
        for e in (self.beginEdit, self.endEdit, self.refEdit):
            v = self._num(e)
            if v is not None:
                e.setText(f"{round(v + d, 6):g}")
        self._times_after_t0 = after

    def _update_bg(self) -> None:
        if not self._t.size:
            self._bgv, self._bg_note = np.zeros(self.geom.num_pixels), ""
            return
        self._bgv, self._bg_note = _bg_vector(self._t, self._c, self._fit_settings())

    # ── time ──────────────────────────────────────────────────────────
    def _range_idx(self) -> Tuple[int, int]:
        return range_bins(self._t, self._t_range())

    def _setup_time_bar(self) -> None:
        n = self._t.size
        self.timeBar.blockSignals(True)
        if not n:
            self.timeBar.setRange(0, 0)
            self.timeBar.blockSignals(False)
            self.timeLbl.setText("t = --")
            return
        r0, r1 = self._range_idx()
        self.timeBar.setRange(0, r1 - r0)
        if self._t_cursor is None or not (self._t[r0] - 1e-9 <= self._t_cursor <= self._t[r1] + 1e-9):
            tot = self._c[r0:r1 + 1].sum(axis=1)
            i = r0 + int(np.argmax(tot)) if tot.size else r0
        else:
            i = int(np.argmin(np.abs(self._t - self._t_cursor)))
        i = int(np.clip(i, r0, r1))
        self.timeBar.setValue(i - r0)
        self._t_cursor = float(self._t[i])
        self.timeBar.blockSignals(False)

    def _ti(self) -> Optional[int]:
        if not self._t.size:
            return None
        r0, r1 = self._range_idx()
        return int(np.clip(r0 + self.timeBar.value(), r0, r1))

    def _on_time_bar(self, _v: int) -> None:
        i = self._ti()
        if i is None:
            return
        self._t_cursor = float(self._t[i])
        self._update_cursor()
        self._request_frame_fit()

    def _set_cursor_time(self, t: float) -> None:
        if not self._t.size:
            return
        r0, r1 = self._range_idx()
        i = int(np.clip(np.argmin(np.abs(self._t - t)), r0, r1))
        self.timeBar.setValue(i - r0)

    def _update_cursor(self) -> None:
        i = self._ti()
        if i is None:
            self.pixMap.set_values(np.full(self.geom.num_pixels, np.nan), self.excluded)
            self.timeLbl.setText("t = --")
            return
        t = float(self._t[i])
        self.timeLbl.setText(f"t = {t:.3f} ns")
        self._traceCursor.setValue(t)
        self._sigCursor.setValue(t)
        self._profCursor.setValue(t)
        self._lineSigCursor.setValue(t)
        self._lineProfCursor.setValue(t)
        self._update_line_fit_text()
        self.pixMap.set_values(self._c[i].astype(float) - self._bgv, self.excluded)
        self.pixMap.getPlotItem().setTitle(f"t = {t:.2f} ns, counts − background")
        self._draw_spec_at_t()

    def _on_times_edited(self) -> None:
        if self.model is None:
            return
        self._setup_time_bar()
        self._draw_trace()
        self._draw_spectral()
        self._update_cursor()
        # the adaptive smoothing kernel depends on the reference time: the counts change too
        self._on_fit_setting_changed(reload_view=self.smoothCombo.currentData() == "adapt")

    def _on_align_toggled(self, _c: bool) -> None:
        if self.model is None:
            return
        self._start_alignment()
        self._on_fit_setting_changed(reload_view=True)

    def _on_fit_setting_changed(self, reload_view: bool = False) -> None:
        """Anything the fits depend on changed: background, offset, exclusions, times, alignment."""
        if self.model is None:
            return
        if reload_view:
            self._load_point()
        else:
            self._update_dark_label()
            self._update_bg()
            self._update_cursor()
            self._draw_trace()
            self._draw_spectral()
            self._draw_sigma_plot()
            self._request_frame_fit()
        self._mark_dep_dirty()

    # ── time-zero alignment of all TRPL files (background) ─────────────
    def _start_alignment(self) -> None:
        if self._align_job is not None:
            self._align_job.cancel()
            self._align_job = None
        paths = self.model.scan.trpl_paths() if self.model is not None else []
        if not self.alignCheck.isChecked() or not paths:
            return
        if all(self.store.has_alignment(p) for p in paths):
            self._sweep_t0, self._t0_ready = self.store.common_t0(paths), True
            return
        self._t0_ready = False
        store = self.store

        def work(progress, cancelled):
            for k, p in enumerate(paths):
                if cancelled():
                    return "cancelled"
                try:
                    store.alignment(p)
                except Exception:
                    pass
                progress(k + 1, len(paths))
            return store.common_t0(paths)

        job = self._run(work, lambda res, err: self._on_alignment_done(job, res, err),
                        lambda d, n: self.statusBar().showMessage(f"Time-zero alignment: {d}/{n} TRPL files"))
        self._align_job = job

    def _on_alignment_done(self, job, t0, err: str) -> None:
        if job is not self._align_job or t0 == "cancelled":
            return
        self._align_job = None
        self._sweep_t0 = None if err else t0
        self._t0_ready = True
        self.statusBar().showMessage(f"Time-zero alignment failed: {err}" if err else
                                     f"Time-zero alignment done: scan time zero {_fmt(t0)} ns", 15000)
        if self.alignCheck.isChecked() and self.model is not None:
            self._load_point()
            self._mark_dep_dirty()

    # ── PL ────────────────────────────────────────────────────────────
    def _pl_image(self, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        key = str(fr.path)
        if key in self._pl_cache:
            return self._pl_cache[key]
        img = _apply_crop(fr.image, self._crop)
        wl = fr.wavelength_nm if fr.wavelength_nm is not None else np.arange(fr.image.shape[1], dtype=float)
        wl_c = _crop_axis(wl, self._crop, fr.image.shape[1])
        if self.baselineCheck.isChecked() and fr.wavelength_nm is not None:
            img = _baseline_correct(img, wl_c, float(self.blLo.value()), float(self.blHi.value()))
        else:
            img = np.asarray(img, dtype=float)
        if len(self._pl_cache) > 600:
            self._pl_cache.clear()
        self._pl_cache[key] = (img, wl_c)
        return img, wl_c

    def _show_pl(self) -> None:
        m = self.model
        fr = m.frame(self._pi, self._b)
        if fr is None or not m.has_pl:
            self.frameView.set_image(np.zeros((10, 10)))
            self.frameView.set_image_title("(no PL at this point)" if m.has_pl else "(no PL in this scan)")
            return
        img, wl = self._pl_image(fr)
        self.frameView.set_image(img, wl)
        row0 = getattr(self, "_pl_row_init", None)
        if row0 is not None:
            self._pl_row_init = None
            self.frameView._hline.setValue(int(np.clip(row0, 0, img.shape[0] - 1)))
            self.frameView._refresh_linecut()
        self.frameView.set_image_title(
            f"{self._point_text(self._b)}   ·   {self._plabel.split(' (')[0]} {_fmt(self._px[self._pi])}   ·   {fr.path.name}")

    def _on_apply_crop(self) -> None:
        self._crop = tuple(int(s.value()) for s in self.cropSpins)
        self._on_pl_settings()

    def _on_pl_settings(self) -> None:
        self._pl_cache.clear()
        if self.model is not None:
            self._show_pl()
            self._mark_dep_dirty()

    # ── TRPL point: count map, trace ──────────────────────────────────
    def _on_pixel_clicked(self, p: int, button: int) -> None:
        if button == 2:
            self._set_excluded(self.excluded ^ {p})
            return
        self.sel_pixel = p
        if self._spectral():                     # toggle the pixel's spectral line
            k = line_of_pixel(self.geom, p)
            lines = list(self._active_lines())
            if k in lines and len(lines) > 1:
                lines.remove(k)
            elif k not in lines:
                lines.append(k)
            self._spec_lines = lines
            self._spec_lines_changed()
            return
        self.pixMap.set_selected({p}, self.excluded)
        self._draw_trace()
        self._draw_decay_map()

    def _set_excluded(self, ex: set) -> None:
        self.excluded = set(ex)
        self.exclEdit.setText(", ".join(str(p) for p in sorted(self.excluded)))
        self.pixMap.set_selected(self._sel_pixels(), self.excluded)
        self._on_fit_setting_changed()

    def _on_excl_edited(self) -> None:
        ex = set()
        for tok in self.exclEdit.text().replace(";", ",").replace(" ", ",").split(","):
            if tok.strip().isdigit() and int(tok) < self.geom.num_pixels:
                ex.add(int(tok))
        if ex != self.excluded:
            self._set_excluded(ex)

    def _draw_trace(self) -> None:
        pi = self.tracePlot.getPlotItem()
        log = self.logCheck.isChecked()
        pi.setLogMode(y=log)
        if self._spectral():
            self._traceCurve.setData([], [])
            self._traceRaw.setData([], [])
            self._traceBg.setVisible(False)
            self._draw_spec_traces()
            return
        self._draw_spec_traces()                 # removes the spectral curves
        if not self._t.size:
            self._traceCurve.setData([], [])
            self._traceRaw.setData([], [])
            self._traceBg.setVisible(False)
            return
        p = int(self.sel_pixel)
        y = self._c[:, p].astype(float)
        if log:
            y = np.clip(y, 0.5, None)
        self._traceCurve.setData(self._t, y)
        raw = getattr(self, "_c_raw", None)
        if raw is not None and self.rawCheck.isChecked():
            yr = raw[:, p].astype(float)
            self._traceRaw.setData(self._t, np.clip(yr, 0.5, None) if log else yr)
        else:
            self._traceRaw.setData([], [])
        r0, r1 = self._range_idx()
        t0, t1 = float(self._t[r0]), float(self._t[r1])
        self._traceRegion.setRegion((t0, t1))
        self._traceRef.setValue(self._ref_ns())
        bg = float(self._bgv[p]) if p < self._bgv.size else 0.0
        self._traceBg.setVisible(bg > 0)
        if bg > 0:
            self._traceBg.setValue(np.log10(bg) if log else bg)
        pad = 0.02 * max(t1 - t0, 1e-9)
        pi.setXRange(t0 - pad, t1 + pad, padding=0)
        seg = y[r0:r1 + 1]
        if seg.size:
            lo, hi = float(np.min(seg)), float(np.max(seg))
            if log:
                pi.setYRange(np.log10(max(lo, 0.5)), np.log10(max(hi, 1.0)) + 0.1, padding=0)
            else:
                pi.setYRange(lo, hi + 0.05 * (hi - lo + 1), padding=0)
        excl = " (excluded from fits)" if p in self.excluded else ""
        sm = f", {self.smoothCombo.currentText().lower()} smoothed" if raw is not None else ""
        self.traceLbl.setText(f"Pixel {p}{excl}: aligned counts{sm} (not background-subtracted; dashed = "
                              f"background {bg:.3g}/bin). Shaded = begin–end, green = reference.")

    # ── spectral view (the SPAD's centre line on a wavelength axis) ───
    def _spectral(self) -> bool:
        return self.viewCombo.currentIndex() == 1

    def _wlcal_ok(self) -> Optional[WavelengthCal]:
        cal = self._wlcal
        return cal if cal is not None and not cal.problem(self.geom) else None

    def _spec_x(self) -> Tuple[np.ndarray, str, str]:
        """(x of every spectral line, axis label, unit): λ (nm) with a calibration, else the line position (µm)."""
        _w, pos = center_line_weights(self.geom)
        cal = self._wlcal_ok()
        if cal is not None:
            return cal.wavelength(pos), "Wavelength (nm)", "nm"
        return pos, f"Line position ({'column x' if self.geom.name == 'SPAD93' else 'row y'}, µm)", "µm"

    def _active_lines(self) -> List[int]:
        n = center_line_weights(self.geom)[1].size
        lines = [k for k in self._spec_lines if 0 <= k < n]
        return lines if lines else [line_of_pixel(self.geom, self.geom.center_pixel)]

    def _active_line(self) -> int:
        return self._active_lines()[-1]

    def _line_label(self, k: int) -> str:
        x, _xl, unit = self._spec_x()
        return f"{line_name(self.geom)} {k} ({x[k]:.1f} {unit})"

    def _line_color(self, i: int) -> str:
        return self.PICK_COLORS[i % len(self.PICK_COLORS)]

    def _sel_pixels(self) -> set:
        """Pixels ringed on the count map: the selected pixel, or the pixels of the selected spectral lines."""
        if not self._spectral():
            return {self.sel_pixel}
        w, _pos = center_line_weights(self.geom, self.excluded)
        return {int(p) for k in self._active_lines() for p in np.flatnonzero(np.nan_to_num(w[k]) != 0)}

    def _apply_view_layout(self) -> None:
        spec = self._spectral()
        self.midStack.setCurrentIndex(1 if spec else 0)
        self._midLay.setStretch(0, 2 if spec else 3)
        self._midLay.setStretch(1, 5 if spec else 2)
        for w in (self.wlcalBtn, self.wlcalClearBtn, self.specLogCheck, self.wlcalLbl):
            w.setEnabled(spec)
        self.lineFitW.setVisible(spec)
        self.lineCtlW.setVisible(spec)
        self.traceLbl.setWordWrap(spec)
        if spec != self._view_was_spec:          # spectral: give the bottom row (decays, profile, σ²) more room
            self._view_was_spec = spec
            sp = self._pointSplit
            tot = sum(sp.sizes()) or 1000
            sp.setSizes([int(tot * f) for f in ((0.24, 0.36, 0.40) if spec else (0.42, 0.31, 0.27))])

    def _update_wlcal_label(self) -> None:
        cal = self._wlcal
        if cal is None:
            self.wlcalLbl.setText("λ-cal: none (axis = line position in µm)" +
                                  (f" — {self._wlcal_note}" if self._wlcal_note else ""))
            self.wlcalLbl.setStyleSheet("color:#555")
            return
        why = cal.problem(self.geom)
        if why:
            self.wlcalLbl.setText(f"λ-cal not used: {why}")
            self.wlcalLbl.setStyleSheet("color:#b91c1c")
        else:
            self.wlcalLbl.setText("λ-cal " + cal.describe(self.geom))
            self.wlcalLbl.setStyleSheet("color:#555")

    def _on_view_mode(self) -> None:
        self._apply_view_layout()
        self.pixMap.set_selected(self._sel_pixels(), self.excluded)
        self._draw_frame_fit()
        self._draw_spectral()
        self._draw_trace()
        self._mark_dep_dirty()

    def _on_load_wlcal(self) -> None:
        start = str(self.model.folder) if self.model else ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wavelength calibration", start,
                                                      "λ-calibration (WLCAL_*.json *.json)")
        if not fn:
            return
        try:
            cal = load_wlcal_json(fn)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "λ-calibration", f"{Path(fn).name}: {exc}")
            return
        why = cal.problem(self.geom)
        if why:
            QtWidgets.QMessageBox.warning(self, "λ-calibration", f"{Path(fn).name}: {why}")
            return
        self.set_wlcal(cal)

    def _on_clear_wlcal(self) -> None:
        self._wlcal_note = "cleared"
        self.set_wlcal(None)

    def set_wlcal(self, cal: Optional[WavelengthCal]) -> None:
        self._wlcal = cal
        self._update_wlcal_label()
        self._draw_spectral()
        self._draw_trace()
        self._mark_dep_dirty()

    def _on_spec_log(self, on: bool) -> None:
        _settings().setValue("spectral/log_image", bool(on))
        self._draw_spectral()

    def _spec_lines_changed(self) -> None:
        self.pixMap.set_selected(self._sel_pixels(), self.excluded)
        self._draw_spec_markers()
        self._draw_spec_at_t()
        self._draw_trace()
        self._request_line_fit()
        self._draw_decay_map()
        if self.tabs.currentIndex() == 3:
            self._draw_picks()

    def _on_spec_click(self, k: int, t: Optional[float] = None, remove: bool = False) -> None:
        """Click on the spectral image column k (sorted order): that line becomes the active one (``remove``
        = Ctrl+click: drop it) and the time cursor moves to ``t``."""
        order = getattr(self, "_spec_order", None)
        if order is None or not (0 <= k < order.size):
            return
        line = int(order[k])
        lines = list(self._active_lines())
        if remove:
            if line in lines and len(lines) > 1:
                lines.remove(line)
        else:
            if line in lines:
                lines.remove(line)
            lines.append(line)
        self._spec_lines = lines
        if t is not None:
            self._set_cursor_time(t)
        self._spec_lines_changed()

    def _on_spec_map_click(self, ev) -> None:
        mp = self.specMap
        vb = mp.plot.getViewBox()
        if mp._x is None or ev.button() != QtCore.Qt.LeftButton or not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = vb.mapSceneToView(ev.scenePos())
        k = int(np.argmin(np.abs(mp._x - pt.x())))
        ctrl = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier)
        self._on_spec_click(k, float(pt.y()), remove=ctrl)

    def _draw_spectral(self) -> None:
        """Time-resolved spectrum image (x = λ or line position, y = time over Begin–End)."""
        if not self._spectral():
            return
        mp = self.specMap
        self._spec_order = None
        self._request_line_fit()
        if not self._t.size:
            mp.show_message("No TRPL data at this point.", "Time-resolved spectrum")
            self._draw_spec_markers()
            self._draw_spec_at_t()
            return
        r0, r1 = self._range_idx()
        S = spectral_matrix(self._c[r0:r1 + 1], self._bgv, self.geom, self.excluded)
        x, xl, _unit = self._spec_x()
        order = np.argsort(x)
        self._spec_order = order
        z = S[:, order]
        log = self.specLogCheck.isChecked()
        if log:
            with np.errstate(invalid="ignore"):
                z = np.log10(1.0 + np.clip(z, 0, None))
        what = "log10(1 + counts − background)" if log else "counts − background"
        mp.set_data(x[order], self._t[r0:r1 + 1], z, x_label=xl, y_label="Time (ns)",
                    title=f"Time-resolved spectrum ({what}; {self.geom.name} centre line)")
        self._draw_spec_markers()
        self._draw_spec_at_t()

    def _draw_spec_markers(self) -> None:
        for it in self._specLineItems:
            self.specMap.plot.removeItem(it)
        for it in self._specMarks:
            self.specPlot.removeItem(it)
        self._specLineItems, self._specMarks = [], []
        if not self._spectral() or not self._t.size:
            return
        x, _xl, _u = self._spec_x()
        for i, k in enumerate(self._active_lines()):
            pen = pg.mkPen(self._line_color(i), width=2, style=QtCore.Qt.DashLine)
            if self.specMap._x is not None:
                ln = pg.InfiniteLine(pos=float(x[k]), angle=90, movable=False, pen=pen)
                ln.setZValue(11)
                self.specMap.plot.addItem(ln)
                self._specLineItems.append(ln)
            mk = pg.InfiniteLine(pos=float(x[k]), angle=90, movable=False, pen=pen)
            self.specPlot.addItem(mk)
            self._specMarks.append(mk)

    def _draw_spec_at_t(self) -> None:
        """Spectrum at the time cursor + its maximum and a Gaussian fit (centre, FWHM)."""
        if not self._spectral():
            return
        i = self._ti()
        pi = self.specPlot.getPlotItem()
        if i is None:
            self._specPts.setData([], [])
            self._specFit.setData([], [])
            self.specLbl.setText("")
            pi.setTitle("")
            return
        t = float(self._t[i])
        self._specCursor.setValue(t)
        s = spectral_matrix(self._c[i:i + 1], self._bgv, self.geom, self.excluded)[0]
        x, xl, unit = self._spec_x()
        order = np.argsort(x)
        xs, ys = x[order], s[order]
        self._specPts.setData(xs, ys, connect="finite")
        pi.setLabel("bottom", xl)
        pi.setTitle(f"Spectrum at t = {t:.2f} ns")
        fin = np.isfinite(ys)
        if not fin.any():
            self._specFit.setData([], [])
            self.specLbl.setText("no usable line at this time")
            return
        km = int(np.nanargmax(np.where(fin, ys, -np.inf)))
        text = f"Maximum {ys[km]:.4g} at <b>{xs[km]:.1f} {unit}</b> ({line_name(self.geom)} {int(order[km])})"
        dx = float(np.median(np.abs(np.diff(xs)))) if xs.size > 1 else 1.0
        res = fit_peak(xs, ys, PeakFitOpts(SHAPE_GAUSS, float(xs.min()), float(xs.max()), width0=1.5 * dx))
        if res is not None and res["at_bound"] == [] and np.isfinite(res["fwhm"]):
            xx = np.linspace(xs.min(), xs.max(), 400)
            self._specFit.setData(xx, peak_curve(res, xx))
            text += (f"<br>Gaussian: centre <b>{res['E0']:.1f} ± {_fmt(res['E0_err'], '.2g')} {unit}</b>, "
                     f"FWHM {res['fwhm']:.1f} {unit}, R² {_fmt(res['r2'], '.3f')}")
        else:
            self._specFit.setData([], [])
            text += "<br>Gaussian fit: none (peak at the edge, too few lines or no signal)"
        text += f"<br><span style='color:#555'>Background: {self._bg_note}</span>"
        self.specLbl.setText(text)

    def _draw_spec_traces(self) -> None:
        """Bottom plot in the spectral view: the decay of every selected line (background subtracted)."""
        pi = self.tracePlot.getPlotItem()
        for it in self._specTraceCurves:
            pi.removeItem(it)
        self._specTraceCurves = []
        if not self._spectral() or not self._t.size:
            return
        log = self.logCheck.isChecked()
        S = spectral_matrix(self._c, self._bgv, self.geom, self.excluded)
        r0, r1 = self._range_idx()
        lo, hi = np.inf, -np.inf
        for i, k in enumerate(self._active_lines()):
            y = S[:, k]
            if log:
                y = np.clip(y, 0.5, None)
            it = pi.plot(self._t, y, pen=pg.mkPen(self._line_color(i), width=1.4), name=self._line_label(k),
                         connect="finite")
            self._specTraceCurves.append(it)
            seg = y[r0:r1 + 1]
            if np.isfinite(seg).any():
                lo, hi = min(lo, float(np.nanmin(seg))), max(hi, float(np.nanmax(seg)))
        t0, t1 = float(self._t[r0]), float(self._t[r1])
        self._traceRegion.setRegion((t0, t1))
        self._traceRef.setValue(self._ref_ns())
        pad = 0.02 * max(t1 - t0, 1e-9)
        pi.setXRange(t0 - pad, t1 + pad, padding=0)
        if np.isfinite(lo) and np.isfinite(hi):
            if log:
                pi.setYRange(np.log10(max(lo, 0.5)), np.log10(max(hi, 1.0)) + 0.1, padding=0)
            else:
                pi.setYRange(lo, hi + 0.05 * (hi - lo + 1), padding=0)
        sm = f", {self.smoothCombo.currentText().lower()} smoothed" if getattr(self, "_c_raw", None) is not None else ""
        self.traceLbl.setText(f"Spectral lines (centre line{sm}; background: {self._bg_note}). Click a pixel / the "
                              f"image adds a line, Ctrl+click removes; the last one is fitted (right).")

    # ── spectral view: 1D Gaussian width along the active line (profile map + σ²(t)) ──
    def _cross_name(self) -> str:
        return "y" if self.geom.name == "SPAD93" else "x"

    def _line_fit_req(self):
        if not self._spectral() or self._path is None or not self._t.size:
            return None
        return (self._path, self._fit_settings().key(), int(self._active_line()), self.lineFreeOffCheck.isChecked())

    def _on_line_free_offset(self, on: bool) -> None:
        _settings().setValue("spectral/free_offset", bool(on))
        self._request_line_fit()

    def _request_line_fit(self) -> None:
        """Fit the active line (background job, cached per point / settings / line) and redraw."""
        if not hasattr(self, "_line_cache"):
            return
        req = self._line_fit_req()
        self._line_req = req
        if req is None or req in self._line_cache:
            self._line_pending = None
            self._draw_line_fit()
            return
        self._line_pending = (req, self._t, self._c, self._fit_settings(), int(req[2]), bool(req[3]))
        if self._line_job is not None:
            self._line_job.cancel()                 # its done handler starts the pending fit
        else:
            self._start_line_job()
        self._draw_line_fit()

    def _start_line_job(self) -> None:
        if self._line_job is not None or self._line_pending is None:
            return
        req, t, c, s, line, free_off = self._line_pending
        self._line_pending = None

        def work(_progress, cancelled):
            res = fit_line_x(t, c, s, line, free_off, cancelled)
            return res, bool(cancelled())

        self._line_job = self._run(work, lambda res, err: self._on_line_fit_done(req, res, err))

    def _on_line_fit_done(self, req, res, err: str) -> None:
        self._line_job = None
        if err:
            self._line_cache[req] = (None, err)
        elif res is not None and not res[1]:
            self._line_cache[req] = (res[0], "")
        while len(self._line_cache) > 200:
            self._line_cache.popitem(last=False)
        if req == self._line_req:
            self._draw_line_fit()
        self._start_line_job()

    def _line_fit(self) -> Optional[PointDiffusion]:
        req = getattr(self, "_line_req", None)
        e = self._line_cache.get(req) if req is not None else None
        return None if e is None else e[0]

    def _on_line_prof_setting(self, *_a) -> None:
        st = _settings()
        st.setValue("spectral/prof_mode", int(self.lineProfCombo.currentIndex()))
        st.setValue("spectral/prof_norm", bool(self.lineProfNorm.isChecked()))
        st.setValue("spectral/prof_r_max", float(self.lineProfRSpin.value()))
        self._draw_line_fit()

    def _draw_line_fit(self) -> None:
        """Spectral view, bottom right: σ²(t) of the active line's 1D fits (+ the D line with tab 3's settings)
        and the profile along the line vs time."""
        if not hasattr(self, "_line_cache") or not self._spectral():
            return
        mp, pi = self.lineProfMap, self.lineSigPlot.getPlotItem()
        for it in (self._lineSigGood, self._lineSigBad):
            it.setData([])
        for it in (self._lineSigLine, self._lineSigSmooth, self._lineProfPlus, self._lineProfMinus):
            it.setData([], [])
        self._line_text = ""
        req = self._line_req
        if req is None:
            mp.show_message("No TRPL data at this point.", "Profile along the line")
            pi.setTitle("σ²(t)")
            self.lineFitLbl.setText("")
            return
        name, ax = self._line_label(req[2]), self._cross_name()
        e = self._line_cache.get(req)
        f = None if e is None else e[0]
        if f is None:
            msg = f"Fit of {name} failed: {e[1]}" if e is not None else f"Fitting {name} …"
            mp.show_message(msg, f"Profile along {name}")
            pi.setTitle(f"σ²(t) of {name}")
            self.lineFitLbl.setText(msg)
            return
        # σ²(t)
        good = ~f.flagged
        self._lineSigGood.setData(f.t_ns[good], f.sigma2[good])
        self._lineSigBad.setData(f.t_ns[~good], f.sigma2[~good])
        self._lineSigRef.setValue(f.ref_ns)
        r0, r1 = self._range_idx()
        pad = 0.02 * max(float(self._t[r1] - self._t[r0]), 1e-9)
        pi.setXRange(float(self._t[r0]) - pad, float(self._t[r1]) + pad, padding=0)
        sg = self._sg_spec()
        s0_fixed = float(f.sigma2[0]) if self.s0FixCheck.isChecked() and f.sigma2.size else None
        ln, factor, _sm = d_line(f, self._line_window(), self.flaggedCheck.isChecked(), sg, self._corr_for(f),
                                 s0_fixed)
        s2c = self._contour_s2(f)
        if sg is not None:
            self._lineSigSmooth.setData(f.t_ns, s2c, connect="finite")
        good_s2 = f.sigma2[good] if good.any() else f.sigma2
        if good_s2.size:
            lo, hi = np.percentile(good_s2, [1, 99])
            if hi > lo:
                pi.setYRange(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), padding=0)
        pi.setTitle(f"σ²(t) along {ax}, {name}")
        n_px = line_pixels(self.geom, req[2], self.excluded)[0].size
        text = (f"<b>{name}</b>: 1D Gaussian along {ax} ({n_px} pixels), {f.t_ns.size} frames fitted from "
                f"{f.ref_ns:.2f} ns ({int(f.flagged.sum())} flagged = hollow), {f.failures} failed.  ")
        if ln is None:
            text += "Fewer than 3 usable frames in the D line window."
        else:
            w0, w1 = self._line_window()
            rel = f.t_ns - f.ref_ns
            hi_t = float(rel.max()) if w1 is None else min(w1, float(rel.max()))
            xx = np.array([w0, hi_t])
            self._lineSigLine.setData(xx + f.ref_ns, ln["s0"] + ln["slope"] * xx)
            corr = f", errors × {factor:.2f}" if factor > 1.0 + 1e-9 else ""
            text += (f"<b>D = {ln['D']:.3g} ± {ln['D_err']:.2g} cm²/s</b> (σ² = σ₀² + 2D(t − t_ref){corr}; "
                     f"σ₀² = {ln['s0']:.3g} µm², {ln['n']} frames; D window / σ² smoothing / flagged as in tab 3)")
        if f.note:
            text += f"  [{f.note}]"
        self._line_text = text
        self._draw_line_profile(f, s2c, req[2], name)
        self._update_line_fit_text()

    def _draw_line_profile(self, f: PointDiffusion, s2c: np.ndarray, line: int, name: str) -> None:
        mp = self.lineProfMap
        r0, r1 = self._range_idx()
        t = self._t[r0:r1 + 1]
        spans = self._fit_spans(f)
        inside = (spans[:, 1] >= r0) & (spans[:, 0] <= r1) & (spans[:, 1] < self._t.size)
        use = np.isfinite(f.sigma2) & (True if self.flaggedCheck.isChecked() else ~f.flagged) & inside
        if not use.any():
            mp.show_message("No usable fitted frames in the time range.", f"Profile along {name}")
            return
        ids, xc = line_pixels(self.geom, line, self.excluded)
        P = f.popt
        d = xc[None, :] - P[:, 1:2]                      # signed distance from each frame's fitted centre
        R_fix = float(self.lineProfRSpin.value())
        R = R_fix if R_fix > 0 else float(max(np.abs(d[use]).max(), 3.0 * np.sqrt(P[use, 2].max() ** 2)))
        x = np.linspace(-R, R, 241)
        z = np.full((t.size, x.size), np.nan)
        measured = self.lineProfCombo.currentIndex() == 1
        norm = self.lineProfNorm.isChecked()
        for k in np.flatnonzero(use):
            A, s, off = float(P[k, 0]), float(P[k, 2]), float(P[k, 3])
            i0, i1 = int(spans[k, 0]), int(spans[k, 1])
            if measured:
                zz = self._c[i0:i1 + 1][:, ids].astype(np.float64).mean(axis=0) - self._bgv[ids]
                row = np.interp(x, d[k], zz, left=np.nan, right=np.nan)
                if norm:
                    row = (row - off) / A if A > 0 else row * np.nan
            else:
                gauss = np.exp(-x ** 2 / (2.0 * s ** 2))
                row = gauss if norm else A * gauss
            z[max(i0, r0) - r0:min(i1, r1) - r0 + 1] = row
        what = "measured counts" if measured else "fitted Gaussian"
        mp.set_data(x, t, z, x_label=f"{self._cross_name()} − fitted centre (µm)", y_label="Time (ns)",
                    levels=(0.0, 1.0) if norm else None,
                    title=f"Profile along {name} ({what}{', normalized' if norm else ''})")
        if R_fix > 0:
            mp.plot.setXRange(-R_fix, R_fix, padding=0.0)
        sc = np.where(inside, np.sqrt(np.clip(s2c, 0, None)), np.nan)
        self._lineProfPlus.setData(sc, f.t_ns, connect="finite")
        self._lineProfMinus.setData(-sc, f.t_ns, connect="finite")
        if self._t_cursor is not None:
            self._lineProfCursor.setValue(self._t_cursor)

    def _update_line_fit_text(self) -> None:
        """The line fit's summary + its fit at the time cursor."""
        if not hasattr(self, "_line_cache") or not self._spectral() or self._line_req is None:
            return
        f = self._line_fit()
        if f is None:
            return
        i = self._ti()
        extra = ""
        if i is not None:
            spans = self._fit_spans(f)
            k = np.flatnonzero((spans[:, 0] <= i) & (i <= spans[:, 1]))
            if k.size:
                A, c, s, off = (float(v) for v in f.popt[k[0]])
                flag = " <span style='color:#b45309'>(flagged)</span>" if f.flagged[k[0]] else ""
                extra = (f"At t = {self._t[i]:.2f} ns: {self._cross_name()}₀ = {c:.1f} µm, <b>σ = {s:.2f} µm</b> "
                         f"(σ² = {s * s:.4g} µm²), A = {A:.4g}, offset {off:.3g}{flag}")
            else:
                extra = f"At t = {self._t[i]:.2f} ns: no fitted frame (before the reference or a failed fit)"
        self.lineFitLbl.setText(self._line_text + ("<br>" + extra if extra else ""))

    # ── 2D fit of the frame at the time cursor (contours + radial) ────
    def _valid_fit(self, pt) -> Optional[PointDiffusion]:
        e = self._fits.get(pt)
        if e is None or self.model is None:
            return None
        key, path, res = e
        cur = self.model.trpl_path(*pt)
        if cur is None or str(cur) != path or key != self._fit_settings().key():
            return None
        return res

    def _request_frame_fit(self) -> None:
        self._frame_res = None
        self._frame_req = None
        i = self._ti()
        if i is None or self._path is None:
            self._draw_frame_fit()
            return
        pdf = self._valid_fit((self._pi, self._b))
        if pdf is not None and getattr(pdf, "popt", None) is not None:
            spans = self._fit_spans(pdf)
            k = np.flatnonzero((spans[:, 0] <= i) & (i <= spans[:, 1]))
            if k.size:
                pop = pdf.popt[k[0]]
                i0, i1 = int(spans[k[0], 0]), int(spans[k[0], 1])
                src = "sequential fit (Fit this point / Fit all)"
                if i1 > i0:
                    src = (f"sequential fit of the group {self._t[i0]:.1f}–{self._t[i1]:.1f} ns "
                           f"({i1 - i0 + 1} bins, mean frame)")
                self._frame_res = dict(popt=pop, sigma_eq=float(np.sqrt((pop[3] ** 2 + pop[4] ** 2) / 2.0)),
                                       flagged=bool(pdf.flagged[k[0]]), snr=None, source=src, bins=(i0, i1))
                self._draw_frame_fit()
                return
        s = self._fit_settings()
        mask = s.mask(self.geom.num_pixels)
        if int(mask.sum()) < MIN_FIT_PIXELS_2D:
            self._draw_frame_fit(f"Not enough enabled pixels (need ≥ {MIN_FIT_PIXELS_2D}).")
            return
        req = (self._path, s.key(), i)
        self._frame_req = req
        if req in self._frame_cache:
            self._frame_res = self._frame_cache[req]
            self._draw_frame_fit()
            return
        z = (self._c[i].astype(np.float64) - self._bgv)[mask]
        g = self.geom
        xm, ym = g.x_um[mask], g.y_um[mask]
        fix = 0.0 if s.fix_offset else None
        self._frame_pending = (req, xm, ym, z, g.pitch_x, g.pitch_y, fix)
        self._draw_frame_fit("Fitting this frame …")
        self._start_frame_job()

    def _start_frame_job(self) -> None:
        if self._frame_job is not None or self._frame_pending is None:
            return
        req, xm, ym, z, px, py, fix = self._frame_pending
        self._frame_pending = None
        self._frame_job = self._run(lambda _p, _c: fit_gaussian_2d(xm, ym, z, px, py, fix_offset=fix),
                                    lambda res, err: self._on_frame_fit_done(req, res, err))

    def _on_frame_fit_done(self, req, res, err: str) -> None:
        self._frame_job = None
        out = None
        if res is not None and not err:
            out = dict(popt=res.popt, sigma_eq=res.sigma_eq, flagged=bool(res.at_bound) or not res.quality_ok,
                       snr=res.snr, source="free 2D fit of this frame")
        self._frame_cache[req] = out
        while len(self._frame_cache) > 2000:
            self._frame_cache.popitem(last=False)
        if req == self._frame_req:
            self._frame_res = out
            self._draw_frame_fit(None if out is not None else f"2D fit failed{': ' + err if err else ''}.")
        self._start_frame_job()

    def _draw_frame_fit(self, msg: Optional[str] = None) -> None:
        res = self._frame_res
        i = self._ti()
        self.pixMap.set_contours(None if res is None or self._spectral() else res["popt"])
        if res is None:
            self.frameFitLbl.setText(msg or ("" if i is not None else "No TRPL data at this point."))
            self._radPts.setData([])
            self._radCurve.setData([], [])
            self.radialPlot.getPlotItem().setTitle(msg or "")
            return
        A, x0, y0, sx, sy, off = (float(v) for v in res["popt"])
        t = float(self._t[i])
        flag = " — <span style='color:#b45309'>flagged (at a bound or low signal)</span>" if res["flagged"] else ""
        snr = f", A/σ_A = {res['snr']:.3g}" if res.get("snr") is not None and np.isfinite(res["snr"]) else ""
        self.frameFitLbl.setText(
            f"<b>2D Gaussian at t = {t:.2f} ns</b> ({res['source']}){flag}<br>"
            f"centre ({x0:.2f}, {y0:.2f}) µm<br>σx = {sx:.2f} µm, σy = {sy:.2f} µm<br>"
            f"<b>σ_eq = {res['sigma_eq']:.2f} µm</b> (σ_eq² = {res['sigma_eq'] ** 2:.3g} µm²)<br>"
            f"A = {A:.4g}, offset = {off:.3g}{snr}<br>"
            f"Background: {self._bg_note}<br>"
            f"Contours: 1σ (solid) and 2σ (dashed) of the fit.  Left click = select pixel, right click = "
            f"exclude from fits.")
        g = self.geom
        mask = self._fit_settings().mask(g.num_pixels)
        i0, i1 = res.get("bins") or (i, i)
        z = (self._c[i0:i1 + 1].astype(np.float64).mean(axis=0) - self._bgv)[mask]
        r = np.hypot(g.x_um[mask] - x0, g.y_um[mask] - y0)
        self._radPts.setData(np.concatenate([r, -r]), np.concatenate([z, z]))
        rmax = max(float(r.max()) * 1.05, 3.0 * res["sigma_eq"]) if r.size else 3.0 * res["sigma_eq"]
        rr = np.linspace(-rmax, rmax, 400)
        self._radCurve.setData(rr, A * np.exp(-rr ** 2 / (2.0 * res["sigma_eq"] ** 2)) + off)
        self.radialPlot.getPlotItem().setTitle(
            f"t = {t:.2f} ns:  σ_eq = {res['sigma_eq']:.2f} µm  ({res['source']})")

    def _fit_spans(self, f: PointDiffusion) -> np.ndarray:
        """(frames, 2) first / last bin of the current view that each fitted frame covers."""
        g = getattr(f, "groups", None)
        if g is not None and len(g) == f.t_ns.size:
            return np.asarray(g, dtype=int)
        i = np.searchsorted(self._t, f.t_ns - 1e-6) if self._t.size else np.zeros(f.t_ns.size, int)
        return np.column_stack([i, i]).astype(int)

    # ══ σ_eq²(t) fits ══════════════════════════════════════════════════
    def _line_window(self):
        d1 = float(self.d1Spin.value())
        return float(self.d0Spin.value()), (None if d1 <= 0 else d1)

    def _set_fit_busy(self, busy: bool) -> None:
        self.fitPointBtn.setEnabled(not busy)
        self.fitAllBtn.setEnabled(not busy)
        self.cancelBtn.setEnabled(busy)
        self.fitProg.setVisible(busy)

    def _on_fit_point(self) -> None:
        if self.model is None or not self._t.size or self._fit_job is not None or self._fitall_job is not None:
            return
        s = self._fit_settings()
        t, c, pt, path = self._t, self._c, (self._pi, self._b), self._path
        common = self._common_job_args()
        self.fitLbl.setText("Fitting this point …" if common is None else
                            "Median reference width of the sweep line, then this point …")
        self._set_fit_busy(True)
        self.fitProg.setRange(0, 0)

        def work(progress, cancelled):
            com, s_fit = None, s
            if common is not None:
                com = ref_width_median(self.store, common[0], common[1], None, cancelled)
                s_fit = _with_ref_sigma(s, (com["sx"], com["sy"]))
            return com, s_fit, fit_point_x(t, c, s_fit, cancelled)

        def done(res, err):
            self._fit_job = None
            self._set_fit_busy(False)
            if err:
                self.fitLbl.setText(f"Fit failed: {err}")
                return
            com, s_fit, r = res
            if com is not None:
                self._ref_common = dict(com, key=common[2])
                self._update_ref_info()
            self._fits[pt] = (s_fit.key(), path, r)
            self._draw_sigma_plot()
            self._request_frame_fit()

        self._fit_job = self._run(work, done)

    def _common_job_args(self):
        """(paths, base settings, cache key) when "Common" needs the median of this sweep line first, else None."""
        if self.refModeCombo.currentIndex() != 1 or self._ref_sigma_now() is not None:
            return None
        key = self._common_key()
        return list(key[0]), self._fit_settings_base(), key

    def _on_fit_all(self) -> None:
        if self.model is None or self._fit_job is not None or self._fitall_job is not None:
            return
        pts, _x, _xl, ctx = self._line()
        if not pts:
            return
        s = self._fit_settings()
        m, store = self.model, self.store
        todo = [(pt, str(m.trpl_path(*pt))) for pt in pts if m.trpl_path(*pt) is not None]
        common = self._common_job_args()
        cached = {} if common is not None else {pt: self._valid_fit(pt) for pt, _ in todo}
        n0 = len(todo) if common is not None else 0
        self._set_fit_busy(True)
        self.fitProg.setRange(0, n0 + len(todo))
        self.fitProg.setValue(0)
        self.fitLbl.setText(f"Fit all: {len(todo)} points of the {ctx} …"
                            + (" (median reference width first)" if common is not None else ""))

        def work(progress, cancelled):
            out, errors, com, s_fit = {}, [], None, s
            if common is not None:
                com = ref_width_median(store, common[0], common[1], progress, cancelled)
                s_fit = _with_ref_sigma(s, (com["sx"], com["sy"]))
            for k, (pt, path) in enumerate(todo):
                if cancelled():
                    break
                if cached.get(pt) is not None:
                    out[pt] = (path, cached[pt])
                else:
                    try:
                        t, c, _tz, _note = point_view(store, store.load(path), s_fit.view)
                        res = fit_point_x(t, c, s_fit, cancelled)
                        if not cancelled():
                            out[pt] = (path, res)
                    except Exception as exc:
                        errors.append(f"{Path(path).name}: {exc}")
                progress(n0 + k + 1, n0 + len(todo))
            return out, errors, bool(cancelled()), com, s_fit

        def done(res, err):
            self._fitall_job = None
            self._set_fit_busy(False)
            if err:
                self.fitLbl.setText(f"Fit all failed: {err}")
                return
            out, errors, cancelled, com, s_fit = res
            key = s_fit.key()
            if com is not None:
                self._ref_common = dict(com, key=common[2])
                self._update_ref_info()
            for pt, (path, r) in out.items():
                self._fits[pt] = (key, path, r)
            self._sigma_snap = dict(key=key, fits={pt: r for pt, (_p, r) in out.items()}, ctx=ctx,
                                    cancelled=cancelled, n=len(todo))
            msg = f"Fit all: {len(out)}/{len(todo)} points fitted" + (" (cancelled)" if cancelled else "")
            if errors:
                msg += f"; {len(errors)} failed: " + "; ".join(errors[:3]) + (" …" if len(errors) > 3 else "")
            self.statusBar().showMessage(msg, 20000)
            self._draw_sigma_plot()
            self._request_frame_fit()
            self._dep_dirty = True
            self._draw_sigma_map()

        self._fitall_job = self._run(work, done, lambda d, n: self.fitProg.setValue(d))

    def _on_cancel_fit(self) -> None:
        for j in (self._fit_job, self._fitall_job):
            if j is not None:
                j.cancel()

    def _draw_sigma_plot(self) -> None:
        pi = self.sigPlot.getPlotItem()
        f = self._valid_fit((self._pi, self._b)) if self.model is not None else None
        if self._t.size:
            r0, r1 = self._range_idx()
            pad = 0.02 * max(float(self._t[r1] - self._t[r0]), 1e-9)
            pi.setXRange(float(self._t[r0]) - pad, float(self._t[r1]) + pad, padding=0)
        self._sigRef.setValue(self._ref_ns())
        if f is None:
            for it in (self._sigGood, self._sigBad):
                it.setData([])
            self._sigLine.setData([], [])
            self._sigSmooth.setData([], [])
            stale = self.model is not None and (self._pi, self._b) in self._fits
            self.fitLbl.setText(("Settings or data changed since this point was fitted: fit again. " if stale else
                                 "Not fitted. ") + "Click 'Fit this point', or 'Fit all' for the whole sweep line.")
            pi.setTitle("σ_eq²(t)")
            self._draw_profile_map()
            self._draw_line_fit()
            return
        good = ~f.flagged
        self._sigGood.setData(f.t_ns[good], f.sigma2[good])
        self._sigBad.setData(f.t_ns[~good], f.sigma2[~good])
        use_flagged = self.flaggedCheck.isChecked()
        sg = self._sg_spec()
        s0_fixed = float(f.sigma2[0]) if self.s0FixCheck.isChecked() and f.sigma2.size else None
        ln, factor, _sm = d_line(f, self._line_window(), use_flagged, sg, self._corr_for(f), s0_fixed)
        s2c = self._contour_s2(f)
        if sg is not None:
            self._sigSmooth.setData(f.t_ns, s2c, connect="finite")
        else:
            self._sigSmooth.setData([], [])
        rel = f.t_ns - f.ref_ns
        text = (f"{f.t_ns.size} frames fitted from the reference {f.ref_ns:.2f} ns ({int(f.flagged.sum())} flagged = "
                f"hollow: at a bound or low signal), {f.failures} failed.  ")
        if ln is None:
            self._sigLine.setData([], [])
            text += "Fewer than 3 usable frames in the D line window."
        else:
            w0, w1 = self._line_window()
            hi = float(rel.max()) if w1 is None else min(w1, float(rel.max()))
            xx = np.array([w0, hi])
            self._sigLine.setData(xx + f.ref_ns, ln["s0"] + ln["slope"] * xx)
            src = (f"fitted to the S-G smoothed σ² ({sg[0]} frames, order {sg[1]})" if sg is not None
                   else "fitted to the σ² points")
            corr = f"; errors × {factor:.2f} for frames correlated by the smoothing" if factor > 1.0 + 1e-9 else ""
            s0txt = (f"σ₀² = {ln['s0']:.3g} µm² (fixed: σ²(t_ref))" if ln.get("fixed_s0") else
                     f"σ₀² = {ln['s0']:.3g} ± {ln['s0_err']:.2g} µm²")
            text += (f"<b>D = {ln['D']:.3g} ± {ln['D_err']:.2g} cm²/s</b> (σ² = σ₀² + 2D(t − t_ref), {src}{corr}), "
                     f"{s0txt} ({ln['n']} frames)")
        if f.note:
            text += f"  [{f.note}]"
        self.fitLbl.setText(text)
        good_s2 = f.sigma2[good] if good.any() else f.sigma2
        if good_s2.size:
            lo, hi = np.percentile(good_s2, [1, 99])
            if hi > lo:
                pi.setYRange(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), padding=0)
        pi.setTitle(f"σ_eq²(t) at {self._point_text(self._b)}")
        self._draw_profile_map()
        self._draw_line_fit()

    def _corr_for(self, f: PointDiffusion) -> float:
        """Trace-smoothing correlation length in fitted frames: bins ÷ the mean group size in the D window."""
        L = self._trace_corr()
        g = getattr(f, "groups", None)
        if g is None or L <= 1.0:
            return L
        w0, w1 = self._line_window()
        rel = f.t_ns - f.ref_ns
        sel = (rel >= w0) & ((rel <= w1) if w1 is not None else True)
        sizes = (np.asarray(g)[:, 1] - np.asarray(g)[:, 0] + 1)[sel]
        return max(1.0, L / float(sizes.mean())) if sizes.size else L

    def _sg_spec(self) -> Optional[Tuple[int, int]]:
        """σ² Savitzky–Golay (window frames, order), or None when off / too short for the order."""
        win, order = int(self.s2winSpin.value()) | 1, int(self.s2ordSpin.value())
        return (win, order) if win >= max(3, order + 2) else None

    def _contour_s2(self, f: PointDiffusion) -> np.ndarray:
        """σ² per fitted frame as the D line uses it: S-G smoothed when on, else the frame's own; NaN for
        frames not used (flagged unless 'Include flagged', or a smoothing window across a large gap)."""
        use = np.isfinite(f.sigma2) & (True if self.flaggedCheck.isChecked() else ~f.flagged)
        sg = self._sg_spec()
        if sg is None:
            return np.where(use, f.sigma2, np.nan)
        s2s, ok = sg_sigma2(f.t_ns, f.sigma2, use, sg[0], sg[1], f.t_ns)
        return np.where(ok, s2s, np.nan)

    def _on_prof_range(self, v: float) -> None:
        _settings().setValue("profile/r_max", float(v))
        self._draw_profile_map()

    def _draw_profile_map(self) -> None:
        """Tab 3, right: radial profile vs time from the sequential fit of this point, ±σ_eq(t) lines."""
        mp = self.profMap
        f = self._valid_fit((self._pi, self._b)) if self.model is not None else None
        self._profPlus.setData([], [])
        self._profMinus.setData([], [])
        if f is None or getattr(f, "popt", None) is None or not self._t.size:
            mp.show_message("Fit this point to see its radial profile vs time.", "Radial profile")
            return
        r0, r1 = self._range_idx()
        t = self._t[r0:r1 + 1]
        spans = self._fit_spans(f)
        inside = (spans[:, 1] >= r0) & (spans[:, 0] <= r1) & (spans[:, 1] < self._t.size)
        use = np.isfinite(f.sigma2) & (True if self.flaggedCheck.isChecked() else ~f.flagged) & inside
        if not use.any():
            mp.show_message("No usable fitted frames in the time range.", "Radial profile")
            return
        g = self.geom
        mask = self._fit_settings().mask(g.num_pixels)
        P = f.popt
        sig2 = (P[:, 3] ** 2 + P[:, 4] ** 2) / 2.0
        dist = np.hypot(g.x_um[mask][None, :] - P[:, 1:2], g.y_um[mask][None, :] - P[:, 2:3])
        R_fix = float(self.profRSpin.value())
        R = R_fix if R_fix > 0 else float(max(dist[use].max(), 3.0 * np.sqrt(sig2[use].max())))
        x = np.linspace(-R, R, 241)
        z = np.full((t.size, x.size), np.nan)
        measured = self.profCombo.currentIndex() == 1
        norm = self.profNormCheck.isChecked()
        for k in np.flatnonzero(use):
            A, off = float(P[k, 0]), float(P[k, 5])
            i0, i1 = int(spans[k, 0]), int(spans[k, 1])
            if measured:
                zz = (self._c[i0:i1 + 1].astype(np.float64).mean(axis=0) - self._bgv)[mask]
                order = np.argsort(dist[k])
                rr, zz = dist[k][order], zz[order]
                row = np.interp(np.abs(x), rr, zz, left=zz[0], right=np.nan)
                if norm:
                    row = (row - off) / A if A > 0 else row * np.nan
            else:
                gauss = np.exp(-x ** 2 / (2.0 * sig2[k]))
                row = gauss if norm else A * gauss
            z[max(i0, r0) - r0:min(i1, r1) - r0 + 1] = row         # a group fills every bin it covers
        what = "measured counts" if measured else "fitted Gaussian"
        mp.set_data(x, t, z, x_label="Distance from the fitted centre (µm)", y_label="Time (ns)",
                    levels=(0.0, 1.0) if norm else None,
                    title=f"Radial profile ({what}{', normalized each time' if norm else ''})")
        if R_fix > 0:
            mp.plot.setXRange(-R_fix, R_fix, padding=0.0)
        s2c = self._contour_s2(f)
        sc = np.where(inside, np.sqrt(np.clip(s2c, 0, None)), np.nan)
        self._profPlus.setData(sc, f.t_ns, connect="finite")
        self._profMinus.setData(-sc, f.t_ns, connect="finite")
        if self._t_cursor is not None:
            self._profCursor.setValue(self._t_cursor)

    # ══ tab 4: dependence ══════════════════════════════════════════════
    def _dep_mode(self) -> str:
        return self.depCombo.currentText()

    def _line(self):
        """(points [(power idx, base idx)], x, x label, context) of the current sweep line, x ascending."""
        m = self.model
        if m is None:
            return [], np.zeros(0), "", ""
        if self._dep_mode() == self.MODE_GATE or m.n_powers < 2:
            pts = [(self._pi, int(b)) for b in self._order]
            x = self._cut.x[self._order]
            ctx = f"gate sweep at {self._plabel.split(' (')[0].lower()} {_fmt(self._px[self._pi])}"
            return pts, x, self._cut.label, ctx
        pts = [(int(p), self._b) for p in self._porder]
        return pts, self._px[self._porder], self._plabel, f"power sweep at {self._point_text(self._b).strip()}"

    def _cur_x(self) -> Optional[float]:
        if self.model is None:
            return None
        if self._dep_mode() == self.MODE_GATE or self.model.n_powers < 2:
            return float(self._cut.x[self._b])
        return float(self._px[self._pi])

    def _update_dep_markers(self) -> None:
        x = self._cur_x()
        _p, xs, _l, _c = self._line()
        if x is not None and xs.size > 1 and np.any(np.diff(xs) <= 0):     # drawn by index
            pts = self._line()[0]
            cur = (self._pi, self._b)
            x = float(pts.index(cur)) if cur in pts else None
        for mp in (self.plMap, self.decayMap, self.sigmaMap):
            mp.set_marker(x)
        if not self._scanfit_dirty and self.scanFitView.has_data():
            pts = self._line()[0]
            cur = (self._pi, self._b)
            if cur in pts:
                self.scanFitView.select_column(pts.index(cur))

    def _on_dep_column_clicked(self, k: int) -> None:
        pts = self._line()[0]
        if not (0 <= k < len(pts)):
            return
        pi, b = pts[k]
        if self._dep_mode() == self.MODE_GATE or self.model.n_powers < 2:
            self.voltBar.setValue(int(np.flatnonzero(self._order == b)[0]))
        else:
            self.powerBar.setValue(int(np.flatnonzero(self._porder == pi)[0]))

    # ── tab 4, second column: picked linecuts ──────────────────────────
    PICK_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
                   "#17becf", "#bcbd22", "#7f7f7f")

    def _on_dep_map_click(self, k: int) -> None:
        shift = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        if not self.pickCheck.isChecked() or shift:
            self._on_dep_column_clicked(k)
            return
        if k in self._picks:
            self._picks.remove(k)
        else:
            self._picks.append(k)
        self._draw_picks()

    def _decay_source(self) -> str:
        """What tab 4's decays show: the selected pixel, or (spectral view) the active spectral line."""
        if self._spectral():
            return f"spectral {self._line_label(self._active_line())}"
        return f"pixel {int(self.sel_pixel)}"

    def _clear_picks(self) -> None:
        self._picks = []
        self._draw_picks()

    def _draw_picks(self) -> None:
        """Curves of the picked sweep columns next to the tab-4 maps + coloured lines on the maps."""
        if self.model is not None:
            _pts, x, xl, _ctx = self._line()
        else:
            x, xl = np.zeros(0), ""
        self._picks = [k for k in self._picks if k < x.size]
        mu = re.search(r"\(([^()]*)\)\s*$", xl)
        unit = f" {mu.group(1)}" if mu else ""
        name = xl[:mu.start()].strip() if mu else xl
        labels = {k: f"{_fmt(x[k])}{unit}" for k in self._picks}
        colors = {k: self.PICK_COLORS[i % len(self.PICK_COLORS)] for i, k in enumerate(self._picks)}
        # lines on the maps
        for mp in (self.plMap, self.decayMap, self.sigmaMap):
            for ln in self._pick_lines.pop(id(mp), []):
                mp.plot.removeItem(ln)
            lines = []
            if mp._x is not None:
                for k in self._picks:
                    if k < mp._x.size:
                        ln = pg.InfiniteLine(pos=float(mp._x[k]), angle=90, movable=False,
                                             pen=pg.mkPen(colors[k], width=2, style=QtCore.Qt.DashLine))
                        ln.setZValue(12)
                        mp.plot.addItem(ln)
                        lines.append(ln)
            self._pick_lines[id(mp)] = lines
        hint = "click map columns to add linecuts" if not self._picks else name

        def reset(pw, title):
            pi = pw.getPlotItem()
            pi.clear()
            if pi.legend is not None:
                pi.legend.clear()
            pi.setTitle(title)
            return pi

        # PL spectra
        pi = reset(self.pickPlPlot, f"PL spectrum — {hint}")
        if self._dep_pl is not None:
            y, z, yl = self._dep_pl
            pi.setLabel("bottom", yl)
            for k in self._picks:
                if k < z.shape[1]:
                    pi.plot(y, z[:, k], pen=pg.mkPen(colors[k], width=2), name=labels[k], connect="finite")
        elif self._picks:
            pi.setTitle("PL spectrum — no PL map")
        # decay at the selected pixel
        norm, log = self.normCheck.isChecked(), self.logColCheck.isChecked()
        pi = reset(self.pickDecayPlot, f"Decay of {self._decay_source()} — {hint}")
        pi.setLogMode(x=False, y=log)
        pi.setLabel("left", ("Normalized counts" if norm else "Counts"))
        if self._dep_decay is not None:
            t, z = self._dep_decay
            for k in self._picks:
                if k >= z.shape[1]:
                    continue
                col = z[:, k].astype(float)
                with np.errstate(all="ignore"):
                    if norm:
                        mx = np.nanmax(col) if np.isfinite(col).any() else np.nan
                        col = col / mx if mx > 0 else col * np.nan
                    if log:
                        col = np.clip(col, 1e-3 if norm else 0.5, None)
                fin = np.isfinite(col)
                pi.plot(t[fin], col[fin], pen=None, symbol="o", symbolSize=3, symbolPen=None,
                        symbolBrush=colors[k], name=labels[k])
        elif self._picks:
            pi.setTitle("Decay — TRPL not loaded yet")
        # σ_eq²(t)
        pi = reset(self.pickSigPlot, f"σ_eq²(t) — {hint}")
        if self._dep_sig is not None:
            for k in self._picks:
                if k < len(self._dep_sig) and self._dep_sig[k] is not None and self._dep_sig[k][0].size:
                    tp, s2p = self._dep_sig[k]
                    pi.plot(tp, s2p, pen=None, symbol="o", symbolSize=5, symbolPen=None,
                            symbolBrush=colors[k], name=labels[k])
        elif self._picks:
            pi.setTitle("σ_eq²(t) — run 'Fit all' in tab 3")

    def _on_dep_mode_changed(self) -> None:
        self._picks = []
        self._mark_dep_dirty()

    def _on_tab_changed(self, idx: int) -> None:
        self.bar.setVisible(idx != 0)
        if idx == 3 and self._dep_dirty:
            self._redraw_dep()
        if idx == 4 and self._scanfit_dirty:
            self._redraw_scanfit()

    def _mark_dep_dirty(self) -> None:
        self._dep_dirty = True
        self._scanfit_dirty = True
        if self.tabs.currentIndex() == 3:
            self._redraw_dep()
        elif self.tabs.currentIndex() == 4:
            self._redraw_scanfit()

    def _redraw_dep(self) -> None:
        self._dep_dirty = False
        if self.model is None:
            for mp in (self.plMap, self.decayMap, self.sigmaMap):
                mp.show_message("Load a scan.")
            return
        pts, x, xl, ctx = self._line()
        self._drawn_line = tuple(pts)
        key = (str(self.model.folder), self._dep_mode(), len(pts))
        if key != self._picks_key:            # another scan / sweep kind: the picked columns mean nothing now
            self._picks, self._picks_key = [], key
        self.depInfo.setText(f"{len(pts)} points: {ctx}")
        self._draw_pl_map()
        self._draw_decay_map()
        self._draw_sigma_map()
        self._update_dep_markers()
        self._draw_picks()

    def _line_pl_columns(self):
        """PL linecuts of the current sweep line, resampled onto the first frame's wavelength axis:
        (pts, x, x label, context, wl, columns (n_wl, n_pts), has wavelength, CCD row, width), or a message."""
        m = self.model
        pts, x, xl, ctx = self._line()
        if not m.has_pl:
            return "No PL in this scan."
        frames = [m.frame(*pt) for pt in pts]
        ref = next((f for f in frames if f is not None), None)
        if ref is None:
            return "No PL on this sweep line."
        _img, wl_ref = self._pl_image(ref)
        wl_ref = np.asarray(wl_ref, float)
        row_d = self.frameView.linecut_row()
        width = self.frameView.linecut_width()
        cols = []
        for fr in frames:
            if fr is None:
                cols.append(np.full(wl_ref.shape, np.nan))
                continue
            img, wl = self._pl_image(fr)
            h = img.shape[0]
            rd = int(np.clip(row_d if row_d is not None else h // 2, 0, h - 1))
            lc = _linecut_h(img, h - 1 - rd, width)
            cols.append(np.full(wl_ref.shape, np.nan) if lc is None else np.asarray(_resample(wl, lc, wl_ref), float))
        mp = np.stack(cols, axis=1)
        return pts, x, xl, ctx, wl_ref, mp, ref.wavelength_nm is not None, row_d, width

    def _draw_pl_map(self) -> None:
        r = self._line_pl_columns()
        if isinstance(r, str):
            self.plMap.show_message(r, "PL spectrum")
            self._dep_pl = None
            return
        _pts, x, xl, ctx, wl_ref, mp, has_wl, row_d, width = r
        if has_wl:
            y, z, yl = *_nm_to_ev(wl_ref, mp), "Photon energy (eV)"
        else:
            y, z, yl = wl_ref, mp, "Pixel"
        self._dep_pl = (np.asarray(y, float), np.asarray(z, float), yl)
        self.plMap.set_data(x, y, z, x_label=xl, y_label=yl,
                            title=f"PL linecut (CCD row {row_d}, width {width}) — {ctx}")

    # ── tab 5: peak fits of the scan's PL along the sweep line ─────────
    def _scan_dip_scale(self) -> Optional[float]:
        """d(D/ε₀)/dx of the sweep axis when it is D (for the dipole moment), else None."""
        if self._dep_mode() != self.MODE_GATE and self.model.n_powers >= 2:
            return None
        c = self._cut
        if c is None or c.kind != "D":
            return None
        if c.short == "D/ε₀":
            return 1.0
        if c.short == "u_D":
            return self.ndTab.calib().scale()[1]
        return None

    def _redraw_scanfit(self) -> None:
        self._scanfit_dirty = False
        view = self.scanFitView
        if self.model is None:
            view.show_message("Load a scan.")
            self.scanFitInfo.setText("")
            return
        r = self._line_pl_columns()
        if isinstance(r, str):
            view.show_message(r)
            self.scanFitInfo.setText("")
            return
        pts, x, xl, ctx, wl_ref, mp, has_wl, row_d, width = r
        if not has_wl:
            view.show_message("This scan's PL frames have no wavelength axis: no energy fits.")
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            ev = EV_NM / np.asarray(wl_ref, float)
        ok = np.isfinite(ev)
        order = np.argsort(ev[ok])
        e, Z = ev[ok][order], mp[ok][order]
        dip = self._scan_dip_scale()
        view.set_spectra(x, e, Z, x_label=xl, title=f"PL linecut (CCD row {row_d}, width {width}) — {ctx}",
                         dip_scale=dip)
        self.scanFitInfo.setText(f"{len(pts)} points: {ctx}.  Spectra = tab 2's PL linecut (row, width, crop, "
                                 "baseline); sweep line as in tab 4 (Gate / Power dependence)."
                                 + ("  Sweep axis is D: the slope gives the dipole moment." if dip else ""))
        cur = (self._pi, self._b)
        if cur in pts:
            view.select_column(pts.index(cur))

    def _draw_decay_map(self) -> None:
        if self.model is None:
            return
        if self.tabs.currentIndex() != 3:
            self._dep_dirty = True
            return
        m = self.model
        pts, x, xl, ctx = self._line()
        paths = [m.trpl_path(*pt) for pt in pts]
        vs = self._view_settings()
        if self._traces_vs != vs.key():
            self._traces.clear()
            self._traces_t.clear()
            self._traces_vs = vs.key()
        missing = [str(p) for p in paths if p is not None and str(p) not in self._traces]
        if missing:
            if self._traces_job is None:
                self._start_traces_job(missing, vs)
            self.decayMap.show_message(f"Loading {len(missing)} TRPL files …", "Decay at the selected pixel")
            self._dep_decay = None
            return
        p = int(self.sel_pixel)
        line_w = None                       # spectral view: the active line's centre-line weights
        if self._spectral():
            line_w = center_line_weights(self.geom, self.excluded)[0][self._active_line()]
        avail = [str(pp) for pp in paths if pp is not None]
        if not avail:
            self.decayMap.show_message("No TRPL on this sweep line.", "Decay at the selected pixel")
            self._dep_decay = None
            return
        t_ref = self._traces_t[avail[0]]
        tr = self._t_range()
        r0, r1 = range_bins(t_ref, tr)
        t = t_ref[r0:r1 + 1]
        z = np.full((t.size, len(paths)), np.nan)
        for k, pp in enumerate(paths):
            if pp is None:
                continue
            c = self._traces[str(pp)]
            tt = self._traces_t[str(pp)]
            if line_w is not None:
                if c.shape[1] != line_w.size:
                    continue
                col = c.astype(np.float64) @ line_w
            elif p >= c.shape[1]:
                continue
            else:
                col = c[:, p].astype(float)
            if tt.size == t_ref.size and np.allclose(tt, t_ref):
                z[:, k] = col[r0:r1 + 1]
            else:
                z[:, k] = np.interp(t, tt, col, left=np.nan, right=np.nan)
        self._dep_decay = (t.copy(), z.copy())
        norm, log = self.normCheck.isChecked(), self.logColCheck.isChecked()
        levels = None
        with np.errstate(all="ignore"):
            if norm:
                mx = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=0)
                z = z / np.where(mx > 0, mx, np.nan)[None, :]
            if log:
                z = np.log10(np.clip(z, 1e-3 if norm else 0.5, None))
                if norm:
                    levels = (-3.0, 0.0)
        what = ("normalized, " if norm else "") + ("log₁₀ " if log else "") + "counts"
        self.decayMap.set_data(x, t, z, x_label=xl, y_label="Time (ns)", levels=levels,
                               title=f"Decay of {self._decay_source()} ({what}) — {ctx}")
        self._update_dep_markers()
        self._draw_picks()

    def _start_traces_job(self, paths: List[str], vs: ViewSettings) -> None:
        store = self.store
        key = vs.key()

        def work(progress, cancelled):
            out = {}
            for k, path in enumerate(paths):
                if cancelled():
                    break
                try:
                    t, c, _tz, _n = point_view(store, store.load(path), vs)
                    out[path] = (t, c.astype(np.float32))
                except Exception:
                    out[path] = (np.zeros(0), np.zeros((0, 0), np.float32))
                progress(k + 1, len(paths))
            return out

        def done(res, err):
            self._traces_job = None
            if err or res is None or self._traces_vs != key:
                if self._traces_vs != key and self.tabs.currentIndex() == 3:
                    self._draw_decay_map()
                return
            for path, (t, c) in res.items():
                if t.size:
                    self._traces[path], self._traces_t[path] = c, t
                else:
                    self._traces[path], self._traces_t[path] = (np.zeros((1, self.geom.num_pixels), np.float32),
                                                                np.zeros(1))
            if len(self._traces) > 400:
                keep = {str(p) for p in (self.model.trpl_path(*pt) for pt in self._line()[0]) if p is not None}
                for k in [k for k in self._traces if k not in keep]:
                    self._traces.pop(k, None)
                    self._traces_t.pop(k, None)
            self._draw_decay_map()

        self._traces_job = self._run(
            work, done, lambda d, n: self.decayMap.show_message(f"Loading TRPL files: {d}/{n} …",
                                                                "Decay at the selected pixel"))

    def _draw_sigma_map(self) -> None:
        if self.model is None:
            return
        if self.tabs.currentIndex() != 3:
            self._dep_dirty = True
            return
        snap = self._sigma_snap
        pts, x, xl, ctx = self._line()
        title = "σ_eq²(t) (µm²)"
        self._dep_sig = None
        if snap is None:
            self.sigmaMap.show_message("Run 'Fit all' in tab 3 to fill this map.", title)
            return
        fits = [snap["fits"].get(pt) for pt in pts]
        if all(f is None for f in fits):
            self.sigmaMap.show_message(f"'Fit all' was run on another sweep line ({snap['ctx']}): run it for this one.",
                                       title)
            return
        t_ax = self._t if self._t.size else next(f for f in fits if f is not None).t_ns
        r0, r1 = range_bins(t_ax, self._t_range())
        t = t_ax[r0:r1 + 1]
        z = np.full((t.size, len(pts)), np.nan)
        use_flagged = self.flaggedCheck.isChecked()
        sg = self._sg_spec()
        pts_k = [None] * len(pts)          # one point per fitted frame / late group, for the picked-linecut plot
        for k, f in enumerate(fits):
            if f is None:
                continue
            ok = np.isfinite(f.sigma2) if use_flagged else (np.isfinite(f.sigma2) & ~f.flagged)
            sel = ok & (f.t_ns >= t[0] - 1e-6) & (f.t_ns <= t[-1] + 1e-6)
            if sel.any():
                tp = np.asarray(f.t_ns[sel], float)
                if sg is not None:
                    s2p, fp = sg_sigma2(f.t_ns, f.sigma2, ok, sg[0], sg[1], tp)
                    pts_k[k] = (tp[fp], np.asarray(s2p, float)[fp])
                else:
                    pts_k[k] = (tp, np.asarray(f.sigma2[sel], float))
            if sg is not None:   # S-G curve at every bin with good frames around it (fills the gaps between them)
                s2s, fill = sg_sigma2(f.t_ns, f.sigma2, ok, sg[0], sg[1], t)
                z[fill, k] = s2s[fill]
                continue
            if getattr(f, "groups", None) is not None and self._t.size and t is not None:
                for (i0, i1), v in zip(np.asarray(f.groups)[ok], f.sigma2[ok]):    # a group fills its bins
                    a, b = max(int(i0), r0) - r0, min(int(i1), r1) - r0
                    if b >= a:
                        z[a:b + 1, k] = v
                continue
            idx = np.searchsorted(t, f.t_ns[ok] - 1e-6)
            good = (idx < t.size)
            idx, vals, tt = idx[good], f.sigma2[ok][good], f.t_ns[ok][good]
            hit = np.abs(t[idx] - tt) < 1e-6
            z[idx[hit], k] = vals[hit]
        stale = snap["key"] != self._fit_settings().key()
        note = "  [settings changed since Fit all]" if stale else ""
        if snap.get("cancelled"):
            note += "  [Fit all was cancelled]"
        if not np.isfinite(z).any():
            self.sigmaMap.show_message("No usable σ_eq² in the time range" + (" (flagged frames hidden)" if not use_flagged else ""),
                                       title + note)
            return
        sm = f", S-G smoothed ({sg[0]} frames, order {sg[1]})" if sg is not None else ""
        self._dep_sig = pts_k
        self.sigmaMap.set_data(x, t, z, x_label=xl, y_label="Time (ns)",
                               title=f"{title} — {ctx}" + (", flagged hidden" if not use_flagged else "") + sm + note)
        self._update_dep_markers()
        self._draw_picks()

    # ══ closing ════════════════════════════════════════════════════════
    def closeEvent(self, event) -> None:
        self.ndTab.save_settings()
        if not self._jobs:
            event.accept()
            return
        if self._close_t0 is None:
            self._close_t0 = time.monotonic()
            for j in self._jobs:
                j.cancel()
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
    ap = argparse.ArgumentParser(description="PL + TRPL dual-gate viewer")
    ap.add_argument("--folder", default="", help="PL + TRPL scan folder to open at startup")
    ap.add_argument("--calib", default="", help="gate calibration json (gtd1_calibration.json)")
    ap.add_argument("--ndmap", default="", help="n-D map: dual_gate_speccube_gtd1.npz or a folder of .asc frames")
    args = ap.parse_args()
    pg.setConfigOptions(background="w", foreground="k")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = DualGateViewer()
    win.show()
    if args.calib:
        QtCore.QTimer.singleShot(0, lambda: win.ndTab.load_calibration(args.calib))
    if args.ndmap:
        QtCore.QTimer.singleShot(0, lambda: win.ndTab.load_ndmap(args.ndmap))
    if args.folder:
        QtCore.QTimer.singleShot(0, lambda: win.load_folder(args.folder))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
