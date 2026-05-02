"""
measurements/ta_gate_power_viewer.py

Viewer for gate- and power-dependent transient absorption data
produced by the pump-probe gate-dep GUI (pump_probe_gate_dep/).

Each *para.txt + *sum.txt pair is one scan at one (gate voltage, ND angle).
An optional XLSX maps ND-wheel angles to power in µW.

Usage:
    python -m measurements.ta_gate_power_viewer [folder]
    or via run_ta_gate_power_viewer.ps1
"""
import sys
import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")


# ─────────────────────────────────────────────────────────────────────────────
#  Colour maps
# ─────────────────────────────────────────────────────────────────────────────

def _rdbu_lut() -> np.ndarray:
    """RdBu_r – red=positive, blue=negative (standard TA convention)."""
    pts = np.linspace(0.0, 1.0, 11)
    rgba = np.array([
        [103,   0,  31, 255],
        [178,  24,  43, 255],
        [214,  96,  77, 255],
        [244, 165, 130, 255],
        [253, 219, 199, 255],
        [247, 247, 247, 255],
        [209, 229, 240, 255],
        [146, 197, 222, 255],
        [ 67, 147, 195, 255],
        [ 33, 102, 172, 255],
        [  5,  48,  97, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(pts, rgba).getLookupTable(nPts=256)


def _viridis_lut() -> np.ndarray:
    return pg.colormap.get("viridis").getLookupTable(nPts=256)


LUTS = {"RdBu_r": None, "viridis": None, "gray": None}


def _get_lut(name: str) -> Optional[np.ndarray]:
    if name == "RdBu_r":
        if LUTS["RdBu_r"] is None:
            LUTS["RdBu_r"] = _rdbu_lut()
        return LUTS["RdBu_r"]
    if name == "viridis":
        if LUTS["viridis"] is None:
            LUTS["viridis"] = _viridis_lut()
        return LUTS["viridis"]
    return None  # gray → no LUT (pyqtgraph default grayscale)


# ─────────────────────────────────────────────────────────────────────────────
#  Filename / para.txt parsers
# ─────────────────────────────────────────────────────────────────────────────

_VF_RE  = re.compile(r"Vf([pn])(\d+)d(\d+)V", re.I)
_VB_RE  = re.compile(r"Vb([pn])(\d+)d(\d+)V", re.I)
_ND_RE  = re.compile(r"ND(n?)(\d+)(?:p(\d+))?deg", re.I)
_HEL_RE = re.compile(r"(?<![A-Za-z])(LH|RH)(?![A-Za-z])", re.I)


def _volt(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    return sign * float(f"{m.group(2)}.{m.group(3)}")


def _ang(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    dec = m.group(3) or "0"
    return sign * float(f"{m.group(2)}.{dec}")


def _parse_fname(stem: str):
    """Return (v_front, v_back, angle, helicity) from filename; NaN/'' if absent."""
    vf = vb = angle = np.nan
    helicity = ""
    m = _VF_RE.search(stem)
    if m:
        vf = _volt(m)
    m = _VB_RE.search(stem)
    if m:
        vb = _volt(m)
    m = _ND_RE.search(stem)
    if m:
        angle = _ang(m)
    m = _HEL_RE.search(stem)
    if m:
        helicity = m.group(1).upper()
    return vf, vb, angle, helicity


def _parse_para(path: Path) -> dict:
    """
    Parse a *para.txt produced by _save_gate_step / _save_angle.

    Line layout (0-indexed, comment lines excluded):
      0  : tab-separated delays in ps
      1  : avg counts per delay
      2  : 0
      3  : scale factor (1.0)
      4  : pixel_size (µm/pixel)
      5  : crop_w  – spectral pixels  (≈ 480)  ← N2 in TAM notation
      6  : pixel_size (repeated)
      7  : crop_h  – spatial pixels   (≈ 128)  ← N1 in TAM notation
      8  : 0
      9  : t0_ps
      10-14: 0s
      15 : 0
      16 : angle (degrees)
      17 : 1
      18 : 0
      19 : n_frames
      20 : 1
    Comment lines: # FrontGateV=…  # BackGateV=…  # Helicity=LH/RH
    """
    data_lines: List[str] = []
    comments: dict = {}
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("#"):
                if "=" in s:
                    k, v = s[1:].split("=", 1)
                    comments[k.strip()] = v.strip()
            else:
                data_lines.append(s)

    def _f(idx: int, default: float = 0.0) -> float:
        try:
            return float(data_lines[idx])
        except Exception:
            return default

    def _i(idx: int, default: int = 128) -> int:
        try:
            return int(float(data_lines[idx]))
        except Exception:
            return default

    delays = np.array(
        [float(x) for x in data_lines[0].split("\t") if x.strip()],
        dtype=np.float64,
    ) if data_lines else np.array([], dtype=np.float64)

    hel = (comments.get("Helicity", "") or comments.get("helicity", "")).strip().upper()

    return dict(
        delays_ps  = delays,
        pixel_size = _f(4, 0.07),   # µm/pixel
        crop_w     = _i(5, 480),    # spectral width  (N2)
        crop_h     = _i(7, 128),    # spatial height  (N1)
        t0_ps      = _f(9, 0.0),
        angle      = _f(16, np.nan),
        v_front    = float(comments["FrontGateV"]) if "FrontGateV" in comments else np.nan,
        v_back     = float(comments["BackGateV"])  if "BackGateV"  in comments else np.nan,
        helicity   = hel,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ScanRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanRecord:
    path_para:  Path
    path_sum:   Path
    delays_ps:  np.ndarray
    crop_h:     int        # spatial pixels  (N1, ≈128)
    crop_w:     int        # spectral pixels (N2, ≈480)
    t0_ps:      float
    pixel_size: float      # µm/pixel (spatial)
    v_front:    float      # NaN if unknown
    v_back:     float      # NaN if unknown
    angle:      float      # rotation stage degrees; NaN if absent
    power_uw:   float      # µW from wheel map; NaN if unmapped
    helicity:   str        # "LH", "RH", or ""
    _stack: Optional[np.ndarray] = field(default=None, repr=False)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def n_delays(self) -> int:
        return len(self.delays_ps)

    @property
    def label(self) -> str:
        parts = []
        if self.helicity:
            parts.append(self.helicity)
        if np.isfinite(self.v_front):
            parts.append(f"Vf={self.v_front:+.3f} V")
        if np.isfinite(self.v_back):
            parts.append(f"Vb={self.v_back:+.3f} V")
        if np.isfinite(self.power_uw):
            parts.append(f"{self.power_uw:.3g} µW")
        elif np.isfinite(self.angle):
            parts.append(f"ND {self.angle:.1f}°")
        return "  ".join(parts) if parts else self.path_para.stem[:50]

    # ── stack loading ─────────────────────────────────────────────────────────

    def load_stack(self) -> Optional[np.ndarray]:
        """
        Load full (T, crop_h, crop_w) stack from sum.txt.

        sum.txt layout (written by _save_gate_step / _save_angle):
          For each time step t:  crop_w rows × crop_h columns
          (= transposed log-ratio image, crop_w ≈ 480 rows, crop_h ≈ 128 cols)
        Reading back: raw[t*N2 : (t+1)*N2, :] → rot90 → (N1, N2) = (crop_h, crop_w)
        """
        if self._stack is not None:
            return self._stack
        try:
            raw = (
                pd.read_csv(
                    self.path_sum,
                    delimiter="\t",
                    header=None,
                    na_values=["--", "---"],
                )
                .fillna(0.0)
                .values.astype(np.float32)
            )
            N1, N2, Nt = self.crop_h, self.crop_w, self.n_delays
            stack = np.zeros((Nt, N1, N2), dtype=np.float32)
            for i in range(Nt):
                s = i * N2
                if s + N2 > raw.shape[0]:
                    break
                stack[i] = np.rot90(raw[s : s + N2, :N1])
            self._stack = stack
            return stack
        except Exception as exc:
            print(f"load_stack {self.path_sum.name}: {exc}")
            return None

    def value_at(self, t_idx: int, iy: int, ix: int) -> float:
        """
        Signal at one pixel, loading only the required frame if the full
        stack is not yet cached.
        """
        if self._stack is not None:
            stk = self._stack
            t = int(np.clip(t_idx, 0, stk.shape[0] - 1))
            y = int(np.clip(iy,    0, stk.shape[1] - 1))
            x = int(np.clip(ix,    0, stk.shape[2] - 1))
            return float(stk[t, y, x])
        # Load only the needed frame rows (fast for large files)
        N1, N2 = self.crop_h, self.crop_w
        t = int(np.clip(t_idx, 0, self.n_delays - 1))
        y = int(np.clip(iy, 0, N1 - 1))
        x = int(np.clip(ix, 0, N2 - 1))
        try:
            chunk = (
                pd.read_csv(
                    self.path_sum,
                    delimiter="\t",
                    header=None,
                    skiprows=t * N2,
                    nrows=N2,
                    na_values=["--", "---"],
                )
                .fillna(0.0)
                .values.astype(np.float32)
            )
            frame = np.rot90(chunk[:N2, :N1])  # (N1, N2)
            return float(frame[y, x])
        except Exception:
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Folder / XLSX helpers
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(folder: str, wheel_map: dict) -> List[ScanRecord]:
    """Discover all *para.txt + *sum.txt pairs, build ScanRecord list."""
    records = []
    for para_path in sorted(Path(folder).glob("*para.txt")):
        sum_path = para_path.with_name(para_path.name.replace("para.txt", "sum.txt"))
        if not sum_path.exists():
            continue
        try:
            meta = _parse_para(para_path)
        except Exception as exc:
            print(f"Skip {para_path.name}: {exc}")
            continue

        fn_vf, fn_vb, fn_ang, fn_hel = _parse_fname(para_path.stem)
        # Prefer comment-line metadata; fall back to filename
        vf  = meta["v_front"]  if np.isfinite(meta["v_front"]) else fn_vf
        vb  = meta["v_back"]   if np.isfinite(meta["v_back"])  else fn_vb
        ang = meta["angle"]    if np.isfinite(meta["angle"])   else fn_ang
        hel = meta["helicity"] or fn_hel

        # Map angle → power via wheel_map (nearest within 0.5°)
        puw = np.nan
        if np.isfinite(ang) and wheel_map:
            best = min(wheel_map, key=lambda k: abs(k - ang))
            if abs(best - ang) <= 0.5:
                puw = wheel_map[best]

        records.append(ScanRecord(
            path_para  = para_path,
            path_sum   = sum_path,
            delays_ps  = meta["delays_ps"],
            crop_h     = meta["crop_h"],
            crop_w     = meta["crop_w"],
            t0_ps      = meta["t0_ps"],
            pixel_size = meta["pixel_size"],
            v_front    = vf,
            v_back     = vb,
            angle      = ang,
            power_uw   = puw,
            helicity   = hel,
        ))
    return records


def load_wheel_map(xlsx_path: str) -> dict:
    """
    Read angle→power_µW from an XLSX file.
    Expected layout (matching the screenshot):
      Row N   : column headers containing 'angle' and 'power'
      Row N+1…: numeric data
    Returns {angle_float: power_uw_float}.
    """
    try:
        df = pd.read_excel(xlsx_path, header=None)
        # Find the header row (first row that contains 'angle' as a cell value)
        hdr_row = None
        for i, row in df.iterrows():
            if any("angle" in str(v).lower() for v in row):
                hdr_row = i
                break
        if hdr_row is None:
            return {}
        header = [str(v).lower().strip() for v in df.iloc[hdr_row]]
        data   = df.iloc[hdr_row + 1 :].reset_index(drop=True)
        data.columns = header
        angle_col = next((c for c in header if "angle" in c), None)
        power_col = next((c for c in header if "power" in c), None)
        if not angle_col or not power_col:
            return {}
        result: dict = {}
        for _, row in data.iterrows():
            try:
                a = float(row[angle_col])
                p = float(row[power_col])
                if np.isfinite(a) and np.isfinite(p):
                    result[a] = p
            except (ValueError, TypeError):
                pass
        return result
    except Exception as exc:
        print(f"XLSX load failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class TAGatePowerViewer(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TA Gate/Power Viewer")
        self.resize(1650, 980)

        self.records:      List[ScanRecord]    = []
        self.current:      Optional[ScanRecord] = None
        self.wheel_map:    dict                = {}
        self.corx:         int                 = 0   # spectral pixel
        self.cory:         int                 = 0   # spatial pixel
        self.wl_m:         float               = 1.0
        self.wl_b:         float               = 0.0
        self.smooth_sigma: float               = 0.0
        self._unique_vf:   list                = []
        self._unique_pwr:  list                = []
        self._unique_hel:  list                = []
        self._hel_mode:    str                 = "single"  # "single" | "LH-RH" | "norm"
        self._lh_rec:      Optional[ScanRecord] = None
        self._rh_rec:      Optional[ScanRecord] = None
        self._derived_stack: Optional[np.ndarray] = None

        self._build_ui()
        self._connect_signals()

        # Optional command-line folder
        if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
            self._load_folder(sys.argv[1])

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        root.addWidget(self._build_left_panel())

        cr_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        cr_split.addWidget(self._build_center_panel())
        cr_split.addWidget(self._build_right_panel())
        cr_split.setSizes([980, 420])
        root.addWidget(cr_split, stretch=1)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(260)
        w.setMaximumWidth(320)
        layout = QtWidgets.QVBoxLayout(w)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 4, 0)

        # ── Folder ──
        fg = QtWidgets.QGroupBox("Data Folder")
        fl = QtWidgets.QGridLayout(fg)
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder…")
        fl.addWidget(self.folder_edit, 0, 0)
        self.folder_btn = QtWidgets.QPushButton("Browse")
        self.folder_btn.setFixedWidth(58)
        fl.addWidget(self.folder_btn, 0, 1)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setFixedWidth(45)
        fl.addWidget(self.load_btn, 0, 2)
        layout.addWidget(fg)

        # ── XLSX ──
        xg = QtWidgets.QGroupBox("ND Wheel → Power (XLSX)")
        xl = QtWidgets.QGridLayout(xg)
        self.xlsx_edit = QtWidgets.QLineEdit()
        self.xlsx_edit.setPlaceholderText("Optional wheel-power map…")
        xl.addWidget(self.xlsx_edit, 0, 0)
        self.xlsx_btn = QtWidgets.QPushButton("Browse")
        self.xlsx_btn.setFixedWidth(58)
        xl.addWidget(self.xlsx_btn, 0, 1)
        layout.addWidget(xg)

        # ── Scan table ──
        sg = QtWidgets.QGroupBox("Scans")
        sl = QtWidgets.QVBoxLayout(sg)
        self.scan_table = QtWidgets.QTableWidget(0, 6)
        self.scan_table.setHorizontalHeaderLabels(
            ["Vf (V)", "Vb (V)", "ND (°)", "Power (µW)", "Hel", "#t"]
        )
        hh = self.scan_table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.scan_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.scan_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.scan_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.scan_table.setAlternatingRowColors(True)
        self.scan_table.verticalHeader().setDefaultSectionSize(22)
        sl.addWidget(self.scan_table)
        layout.addWidget(sg, stretch=1)

        # ── Wavelength calibration ──
        cg = QtWidgets.QGroupBox("Wavelength Cal  (two-point linear)")
        cl = QtWidgets.QGridLayout(cg)
        cl.addWidget(QtWidgets.QLabel("px₁:"), 0, 0)
        self.wl_px1_edit = QtWidgets.QLineEdit("")
        self.wl_px1_edit.setPlaceholderText("pixel 1")
        self.wl_px1_edit.setFixedWidth(68)
        cl.addWidget(self.wl_px1_edit, 0, 1)
        cl.addWidget(QtWidgets.QLabel("nm₁:"), 0, 2)
        self.wl_wl1_edit = QtWidgets.QLineEdit("")
        self.wl_wl1_edit.setPlaceholderText("λ₁ (nm)")
        self.wl_wl1_edit.setFixedWidth(68)
        cl.addWidget(self.wl_wl1_edit, 0, 3)
        cl.addWidget(QtWidgets.QLabel("px₂:"), 1, 0)
        self.wl_px2_edit = QtWidgets.QLineEdit("")
        self.wl_px2_edit.setPlaceholderText("pixel 2")
        self.wl_px2_edit.setFixedWidth(68)
        cl.addWidget(self.wl_px2_edit, 1, 1)
        cl.addWidget(QtWidgets.QLabel("nm₂:"), 1, 2)
        self.wl_wl2_edit = QtWidgets.QLineEdit("")
        self.wl_wl2_edit.setPlaceholderText("λ₂ (nm)")
        self.wl_wl2_edit.setFixedWidth(68)
        cl.addWidget(self.wl_wl2_edit, 1, 3)
        self.wl_apply_btn = QtWidgets.QPushButton("Apply")
        self.wl_apply_btn.setFixedWidth(48)
        cl.addWidget(self.wl_apply_btn, 1, 4)
        layout.addWidget(cg)

        # ── Gaussian smoothing ──
        smg = QtWidgets.QGroupBox("Gaussian Smoothing")
        sml = QtWidgets.QHBoxLayout(smg)
        sml.addWidget(QtWidgets.QLabel("σ:"))
        self.smooth_edit = QtWidgets.QLineEdit("0")
        self.smooth_edit.setFixedWidth(50)
        self.smooth_edit.setToolTip(
            "Gaussian sigma in pixels (0 = off).\n"
            "Applied to image and lineouts; dynamics use raw data."
        )
        sml.addWidget(self.smooth_edit)
        sml.addWidget(QtWidgets.QLabel("px"))
        self.smooth_apply_btn = QtWidgets.QPushButton("Apply")
        self.smooth_apply_btn.setFixedWidth(48)
        sml.addWidget(self.smooth_apply_btn)
        layout.addWidget(smg)

        # ── Colormap + axis lock ──
        cmg = QtWidgets.QGroupBox("Display")
        cml = QtWidgets.QGridLayout(cmg)
        cml.addWidget(QtWidgets.QLabel("Colormap:"), 0, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["RdBu_r", "viridis", "gray"])
        cml.addWidget(self.cmap_combo, 0, 1)
        self.lock_axes_cb = QtWidgets.QCheckBox("Lock axes")
        self.lock_axes_cb.setToolTip(
            "Freeze x/y ranges on all plots so you can compare across\n"
            "conditions (LH/RH, gate, power) without auto-rescaling."
        )
        cml.addWidget(self.lock_axes_cb, 1, 0)
        self.reset_axes_btn = QtWidgets.QPushButton("Reset all scales")
        self.reset_axes_btn.setToolTip("Unlock and auto-scale all plots.")
        cml.addWidget(self.reset_axes_btn, 1, 1)
        layout.addWidget(cmg)

        return w

    # ── Center panel ──────────────────────────────────────────────────────────

    def _build_center_panel(self) -> QtWidgets.QSplitter:
        center = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Control rows: time slider (row 1) + Vf/Power/Hel scan sliders (row 2)
        ctrl_w = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(ctrl_w)
        cv.setContentsMargins(2, 2, 2, 2)
        cv.setSpacing(2)

        row1 = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(row1)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(QtWidgets.QLabel("t:"))
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        cl.addWidget(self.time_slider, stretch=1)
        self.time_label = QtWidgets.QLabel("-- ps  (step 0)")
        cl.addWidget(self.time_label)
        cl.addSpacing(12)
        cl.addWidget(QtWidgets.QLabel("Min:"))
        self.cmin_edit = QtWidgets.QLineEdit("-3")
        self.cmin_edit.setFixedWidth(55)
        cl.addWidget(self.cmin_edit)
        cl.addWidget(QtWidgets.QLabel("Max:"))
        self.cmax_edit = QtWidgets.QLineEdit("3")
        self.cmax_edit.setFixedWidth(55)
        cl.addWidget(self.cmax_edit)
        self.clevel_btn = QtWidgets.QPushButton("Apply")
        self.clevel_btn.setFixedWidth(48)
        cl.addWidget(self.clevel_btn)
        cl.addSpacing(12)
        self.cursor_label = QtWidgets.QLabel("Cursor: --")
        cl.addWidget(self.cursor_label)
        cv.addWidget(row1)

        row2 = QtWidgets.QWidget()
        cl2 = QtWidgets.QHBoxLayout(row2)
        cl2.setContentsMargins(0, 0, 0, 0)
        cl2.addWidget(QtWidgets.QLabel("Hel:"))
        self.hel_combo = QtWidgets.QComboBox()
        self.hel_combo.setFixedWidth(55)
        self.hel_combo.setEnabled(False)
        self.hel_combo.setToolTip("Switch between LH and RH helicity (keeps Vf and Power fixed).")
        cl2.addWidget(self.hel_combo)
        cl2.addSpacing(8)
        cl2.addWidget(QtWidgets.QLabel("Vf:"))
        self.vf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vf_slider.setEnabled(False)
        cl2.addWidget(self.vf_slider, stretch=1)
        self.vf_label = QtWidgets.QLabel("--")
        self.vf_label.setMinimumWidth(85)
        cl2.addWidget(self.vf_label)
        cl2.addSpacing(12)
        cl2.addWidget(QtWidgets.QLabel("P:"))
        self.pwr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pwr_slider.setEnabled(False)
        cl2.addWidget(self.pwr_slider, stretch=1)
        self.pwr_label = QtWidgets.QLabel("--")
        self.pwr_label.setMinimumWidth(85)
        cl2.addWidget(self.pwr_label)
        cv.addWidget(row2)

        center.addWidget(ctrl_w)

        # TAM image
        self.img_gw = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_gw.addPlot()
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(_get_lut("RdBu_r"))
        self.img_plot.addItem(self.img_item)
        self.img_plot.setLabel("bottom", "Spectral pixel")
        self.img_plot.setLabel("left", "Position (µm)")
        self.img_plot.setTitle("TA map (select a scan to load)")
        dash = QtCore.Qt.DashLine
        self.ch_h = pg.InfiniteLine(angle=0,  pen=pg.mkPen("k", width=1, style=dash))
        self.ch_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=dash))
        self.img_plot.addItem(self.ch_h)
        self.img_plot.addItem(self.ch_v)
        center.addWidget(self.img_gw)

        # Lineout row: spectrum | spatial profile
        lo_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.spec_pw = pg.PlotWidget(title="Spectrum (x-lineout at cursor row)")
        self.spec_pw.setLabel("bottom", "Spectral pixel")
        self.spec_pw.setLabel("left", "ΔT/T (mOD)")
        self.spec_curve = self.spec_pw.plot(pen=pg.mkPen((0, 130, 0), width=1.5))
        self.spec_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spec_pw.addItem(self.spec_vline)
        lo_split.addWidget(self.spec_pw)

        self.spat_pw = pg.PlotWidget(title="Spatial profile (y-lineout at cursor column)")
        self.spat_pw.setLabel("bottom", "Position (µm)")
        self.spat_pw.setLabel("left", "ΔT/T (mOD)")
        self.spat_curve = self.spat_pw.plot(
            pen=None, symbol="o", symbolSize=4,
            symbolPen=(180, 50, 50), symbolBrush=(180, 50, 50),
        )
        self.spat_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=dash))
        self.spat_pw.addItem(self.spat_vline)
        lo_split.addWidget(self.spat_pw)
        center.addWidget(lo_split)

        # Dynamics
        self.dyn_pw = pg.PlotWidget(title="Dynamics at cursor")
        self.dyn_pw.setLabel("bottom", "Time delay (ps)")
        self.dyn_pw.setLabel("left", "ΔT/T (mOD)")
        self.dyn_pw.showGrid(x=True, y=True, alpha=0.25)
        self.dyn_curve = self.dyn_pw.plot(
            pen=pg.mkPen((0, 90, 200), width=1.8),
            symbol="o", symbolSize=5,
            symbolPen=(0, 90, 200), symbolBrush=(0, 90, 200),
        )
        self.dyn_lh_curve = self.dyn_pw.plot(
            pen=pg.mkPen((0, 160, 0, 140), width=1.2, style=dash),
            name="LH",
        )
        self.dyn_rh_curve = self.dyn_pw.plot(
            pen=pg.mkPen((200, 100, 0, 140), width=1.2, style=dash),
            name="RH",
        )
        self.dyn_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", style=dash))
        self.dyn_pw.addItem(self.dyn_vline)
        center.addWidget(self.dyn_pw)

        center.setSizes([34, 380, 210, 240])
        return center

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QtWidgets.QSplitter:
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Gate dependence
        self.gate_pw = pg.PlotWidget(title="Gate dependence at cursor")
        self.gate_pw.setLabel("bottom", "V_front (V)")
        self.gate_pw.setLabel("left", "ΔT/T (mOD)")
        self.gate_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.gate_pw)

        # Power dependence
        self.pwr_pw = pg.PlotWidget(title="Power dependence at cursor")
        self.pwr_pw.setLabel("bottom", "Power (µW)")
        self.pwr_pw.setLabel("left", "ΔT/T (mOD)")
        self.pwr_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.pwr_pw)

        # Update button (dependency plots can require loading many files)
        btn_w = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btn_w)
        bl.setContentsMargins(4, 2, 4, 2)
        self.dep_btn = QtWidgets.QPushButton("Update Gate/Power Plots")
        self.dep_btn.setToolTip(
            "Extracts signal at the current cursor position and time from ALL "
            "loaded scans to build the gate- and power-dependence plots.\n"
            "May load remaining stacks on first use."
        )
        bl.addWidget(self.dep_btn)
        right.addWidget(btn_w)

        right.setSizes([420, 420, 40])
        return right

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.folder_btn.clicked.connect(self._browse_folder)
        self.load_btn.clicked.connect(self._do_load)
        self.folder_edit.returnPressed.connect(self._do_load)
        self.xlsx_btn.clicked.connect(self._browse_xlsx)
        self.xlsx_edit.returnPressed.connect(self._load_xlsx_from_edit)
        self.scan_table.itemSelectionChanged.connect(self._on_scan_selected)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.clevel_btn.clicked.connect(self._apply_levels)
        self.wl_apply_btn.clicked.connect(self._apply_wl_cal)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.dep_btn.clicked.connect(self._update_dep_plots)
        self.vf_slider.valueChanged.connect(self._on_vf_slider_changed)
        self.pwr_slider.valueChanged.connect(self._on_pwr_slider_changed)
        self.hel_combo.currentTextChanged.connect(self._on_hel_changed)
        self.smooth_apply_btn.clicked.connect(self._apply_smoothing)
        self.lock_axes_cb.stateChanged.connect(self._on_lock_axes_changed)
        self.reset_axes_btn.clicked.connect(self._reset_all_scales)
        self.img_gw.scene().sigMouseClicked.connect(self._on_image_click)

    # ── Axis helpers ──────────────────────────────────────────────────────────

    def _x_axis(self, rec: ScanRecord) -> np.ndarray:
        """Spectral axis: wavelength (nm) if calibrated, else pixel index."""
        px = np.arange(rec.crop_w, dtype=float)
        if self.wl_m != 1.0 or self.wl_b != 0.0:
            return self.wl_m * px + self.wl_b
        return px

    def _y_axis(self, rec: ScanRecord) -> np.ndarray:
        """Spatial axis in µm, centred at 0."""
        px = np.arange(rec.crop_h, dtype=float)
        return (px - rec.crop_h / 2.0) * rec.pixel_size

    def _levels(self):
        try:
            vmin = float(self.cmin_edit.text())
        except ValueError:
            vmin = -3.0
        try:
            vmax = float(self.cmax_edit.text())
        except ValueError:
            vmax = 3.0
        return vmin, vmax

    def _t_idx(self) -> int:
        return self.time_slider.value()

    def _x_label(self) -> str:
        return "Wavelength (nm)" if (self.wl_m != 1.0 or self.wl_b != 0.0) else "Spectral pixel"

    # ── Axis locking ──────────────────────────────────────────────────────────

    def _plot_viewboxes(self) -> list:
        """All ViewBox instances managed by Lock axes / Reset scales."""
        vbs = [self.img_plot.getViewBox()]
        for pw in [self.spec_pw, self.spat_pw, self.dyn_pw, self.gate_pw, self.pwr_pw]:
            vbs.append(pw.getViewBox())
        return vbs

    def _on_lock_axes_changed(self, state: int) -> None:
        locked = bool(state)
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(not locked)

    def _reset_all_scales(self) -> None:
        self.lock_axes_cb.blockSignals(True)
        self.lock_axes_cb.setChecked(False)
        self.lock_axes_cb.blockSignals(False)
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(True)

    # ── Folder / XLSX loading ─────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data folder", self.folder_edit.text() or ""
        )
        if d:
            self.folder_edit.setText(d)
            self._do_load()

    def _browse_xlsx(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ND-wheel power map",
            os.path.dirname(self.xlsx_edit.text() or ""),
            "Excel (*.xlsx *.xls);;All (*.*)",
        )
        if p:
            self.xlsx_edit.setText(p)
            self._load_xlsx_from_edit()

    def _load_xlsx_from_edit(self) -> None:
        p = self.xlsx_edit.text().strip()
        if p and os.path.isfile(p):
            self.wheel_map = load_wheel_map(p)
            self.statusBar().showMessage(
                f"Wheel map: {len(self.wheel_map)} angle→power entries loaded from {Path(p).name}"
            )
        else:
            self.wheel_map = {}
        # If scans were already loaded, update their power values in-place
        # (avoids re-scanning the folder and discarding cached stacks)
        if self.records:
            self._apply_wheel_map()

    def _apply_wheel_map(self) -> None:
        """Re-apply the current wheel_map to all loaded records and refresh the table."""
        for r in self.records:
            r.power_uw = np.nan
            if np.isfinite(r.angle) and self.wheel_map:
                best = min(self.wheel_map, key=lambda k: abs(k - r.angle))
                if abs(best - r.angle) <= 0.5:
                    r.power_uw = self.wheel_map[best]
        self._populate_table()
        self._rebuild_sliders()

    def _do_load(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.statusBar().showMessage("Invalid folder path.")
            return
        self._load_xlsx_from_edit()
        self.records = scan_folder(folder, self.wheel_map)
        self._populate_table()
        self._rebuild_sliders()
        n = len(self.records)
        self.statusBar().showMessage(
            f"Found {n} scan{'s' if n != 1 else ''} in {folder}"
        )
        if n > 0:
            self.scan_table.selectRow(0)

    def _load_folder(self, folder: str) -> None:
        self.folder_edit.setText(folder)
        self._do_load()

    # ── Scan table ────────────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        self.scan_table.setRowCount(len(self.records))
        for i, r in enumerate(self.records):
            def _cell(txt: str):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                return item
            self.scan_table.setItem(i, 0, _cell(f"{r.v_front:+.3f}" if np.isfinite(r.v_front) else "--"))
            self.scan_table.setItem(i, 1, _cell(f"{r.v_back:+.3f}"  if np.isfinite(r.v_back)  else "--"))
            self.scan_table.setItem(i, 2, _cell(f"{r.angle:.1f}"    if np.isfinite(r.angle)    else "--"))
            self.scan_table.setItem(i, 3, _cell(f"{r.power_uw:.3g}" if np.isfinite(r.power_uw) else "--"))
            self.scan_table.setItem(i, 4, _cell(r.helicity or "--"))
            self.scan_table.setItem(i, 5, _cell(str(r.n_delays)))

    # ── Scan selection ────────────────────────────────────────────────────────

    def _on_scan_selected(self) -> None:
        idx = self.scan_table.currentRow()
        if idx < 0 or idx >= len(self.records):
            return
        rec = self.records[idx]
        self.statusBar().showMessage(f"Loading {rec.path_para.name} …")
        QtWidgets.QApplication.processEvents()

        stk = rec.load_stack()
        if stk is None:
            self.statusBar().showMessage(f"Failed to load {rec.path_para.name}")
            return

        self.current = rec
        self._sync_sliders_to_current()
        # Reset crosshair to centre
        self.corx = rec.crop_w // 2
        self.cory = rec.crop_h // 2

        # Move slider to t0 (delay closest to t0_ps, default 0 ps)
        t0_idx = int(np.argmin(np.abs(rec.delays_ps - rec.t0_ps))) if rec.n_delays else 0
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, rec.n_delays - 1))
        self.time_slider.setValue(t0_idx)
        self.time_slider.blockSignals(False)

        self._update_display(t0_idx)
        self.statusBar().showMessage(
            f"{rec.label}  |  {rec.n_delays} delays  |  "
            f"{rec.crop_h}×{rec.crop_w} px  |  pixel={rec.pixel_size:.3f} µm"
        )

    def _on_time_changed(self, t_idx: int) -> None:
        self._update_display(t_idx)

    # ── Main display update ───────────────────────────────────────────────────

    def _update_display(self, t_idx: int) -> None:
        rec = self.current
        if rec is None:
            return
        if self._derived_stack is not None:
            stk = self._derived_stack
        else:
            stk = rec.load_stack()
            if stk is None:
                return

        t_idx = int(np.clip(t_idx, 0, stk.shape[0] - 1))
        frame = self._smooth_frame(stk[t_idx])   # smoothed for image + lineouts

        xax = self._x_axis(rec)
        yax = self._y_axis(rec)
        vmin, vmax = self._levels()

        # ── Image ──
        self.img_item.setImage(frame, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])
        # Map to real-unit axes
        self.img_item.setRect(QtCore.QRectF(
            float(xax[0]), float(yax[0]),
            float(xax[-1] - xax[0]),
            float(yax[-1] - yax[0]),
        ))
        self.img_plot.setLabel("bottom", self._x_label())
        t_ps = rec.delays_ps[t_idx]
        mode_sfx = f"  [{self._hel_mode}]" if self._hel_mode != "single" else ""
        self.img_plot.setTitle(
            f"{rec.label}{mode_sfx}  |  t = {t_ps:.2f} ps  (step {t_idx})"
        )

        # Crosshair
        ix = int(np.clip(self.corx, 0, rec.crop_w - 1))
        iy = int(np.clip(self.cory, 0, rec.crop_h - 1))
        self.ch_v.setValue(xax[ix])
        self.ch_h.setValue(yax[iy])

        self.time_label.setText(f"{t_ps:.2f} ps  (step {t_idx})")

        # ── Spectrum (x-lineout at row iy) ──
        self.spec_curve.setData(xax, frame[iy, :])
        self.spec_vline.setValue(xax[ix])
        self.spec_pw.setLabel("bottom", self._x_label())
        self.spec_pw.setTitle(f"Spectrum at y = {yax[iy]:.2f} µm")

        # ── Spatial profile (y-lineout at column ix) ──
        self.spat_curve.setData(yax, frame[:, ix])
        self.spat_vline.setValue(yax[iy])
        xlbl = f"λ={xax[ix]:.1f} nm" if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}"
        self.spat_pw.setTitle(f"Spatial profile at {xlbl}")

        # ── Dynamics (all t at pixel ix, iy) ──
        delays = self._current_delays()
        self.dyn_curve.setData(delays, stk[:, iy, ix])
        self.dyn_vline.setValue(t_ps)
        mode_title = f"  [{self._hel_mode}]" if self._hel_mode != "single" else ""
        self.dyn_pw.setTitle(
            f"Dynamics at {xlbl}, y = {yax[iy]:.2f} µm{mode_title}"
        )
        # Ghost LH/RH component traces in derived modes
        if (self._hel_mode != "single"
                and self._lh_rec is not None and self._rh_rec is not None
                and self._lh_rec._stack is not None and self._rh_rec._stack is not None):
            Nt = stk.shape[0]
            self.dyn_lh_curve.setData(
                self._lh_rec.delays_ps[:Nt], self._lh_rec._stack[:Nt, iy, ix]
            )
            self.dyn_rh_curve.setData(
                self._rh_rec.delays_ps[:Nt], self._rh_rec._stack[:Nt, iy, ix]
            )
        else:
            self.dyn_lh_curve.setData([], [])
            self.dyn_rh_curve.setData([], [])

        # Cursor info
        val = frame[iy, ix]
        self.cursor_label.setText(
            f"({'λ=' if (self.wl_m!=1 or self.wl_b!=0) else 'px'}"
            f"{xax[ix]:.1f}, {yax[iy]:.2f} µm) = {val:.3g} mOD"
        )

    # ── Image click ───────────────────────────────────────────────────────────

    def _on_image_click(self, ev) -> None:
        if self.current is None:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if not self.img_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = self.img_plot.vb.mapSceneToView(ev.scenePos())
        xax = self._x_axis(self.current)
        yax = self._y_axis(self.current)
        self.corx = int(np.clip(
            np.argmin(np.abs(xax - pos.x())), 0, self.current.crop_w - 1
        ))
        self.cory = int(np.clip(
            np.argmin(np.abs(yax - pos.y())), 0, self.current.crop_h - 1
        ))
        self._update_display(self._t_idx())

    # ── Levels / calibration / colormap ──────────────────────────────────────

    def _apply_levels(self) -> None:
        if self.current is None:
            return
        vmin, vmax = self._levels()
        self.img_item.setLevels([vmin, vmax])

    def _apply_wl_cal(self) -> None:
        try:
            px1 = float(self.wl_px1_edit.text())
            wl1 = float(self.wl_wl1_edit.text())
            px2 = float(self.wl_px2_edit.text())
            wl2 = float(self.wl_wl2_edit.text())
        except ValueError:
            self.statusBar().showMessage("Wavelength cal: enter all four values (px1, nm1, px2, nm2).")
            return
        if abs(px2 - px1) < 1e-9:
            self.statusBar().showMessage("Wavelength cal: px1 and px2 must be different.")
            return
        self.wl_m = (wl2 - wl1) / (px2 - px1)
        self.wl_b = wl1 - self.wl_m * px1
        self.statusBar().showMessage(
            f"Wavelength cal applied: m={self.wl_m:.4f} nm/px, b={self.wl_b:.2f} nm"
        )
        self._update_display(self._t_idx())

    def _on_cmap_changed(self, name: str) -> None:
        lut = _get_lut(name)
        self.img_item.setLookupTable(lut)
        if self.current is not None:
            vmin, vmax = self._levels()
            self.img_item.setLevels([vmin, vmax])

    # ── Gaussian smoothing ────────────────────────────────────────────────────

    def _smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_sigma <= 0:
            return frame
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(frame.astype(np.float32), sigma=self.smooth_sigma)
        except ImportError:
            self.statusBar().showMessage(
                "Smoothing requires scipy — install with: pip install scipy"
            )
            return frame

    def _apply_smoothing(self) -> None:
        try:
            sigma = max(0.0, float(self.smooth_edit.text()))
        except ValueError:
            return
        self.smooth_sigma = sigma
        self.statusBar().showMessage(
            f"Smoothing σ = {sigma} px" if sigma > 0 else "Smoothing off"
        )
        self._update_display(self._t_idx())

    # ── Vf / Power / Hel scan sliders ────────────────────────────────────────

    def _rebuild_sliders(self) -> None:
        """Populate Vf, Power, and Helicity controls from the currently loaded records."""
        self._unique_vf  = sorted({r.v_front  for r in self.records if np.isfinite(r.v_front)})
        self._unique_pwr = sorted({r.power_uw for r in self.records if np.isfinite(r.power_uw)})
        self._unique_hel = sorted({r.helicity for r in self.records if r.helicity})

        for slider, vals in (
            (self.vf_slider,  self._unique_vf),
            (self.pwr_slider, self._unique_pwr),
        ):
            slider.blockSignals(True)
            if len(vals) > 1:
                slider.setMinimum(0)
                slider.setMaximum(len(vals) - 1)
                slider.setEnabled(True)
            else:
                slider.setEnabled(False)
            slider.blockSignals(False)

        # Helicity combo — add LH-RH and norm entries when both polarizations present
        self.hel_combo.blockSignals(True)
        self.hel_combo.clear()
        has_both = "LH" in self._unique_hel and "RH" in self._unique_hel
        if len(self._unique_hel) >= 1:
            self.hel_combo.addItems(self._unique_hel)
            if has_both:
                self.hel_combo.addItem("LH-RH")
                self.hel_combo.addItem("norm")
            self.hel_combo.setEnabled(len(self._unique_hel) > 1 or has_both)
        else:
            self.hel_combo.setEnabled(False)
        self.hel_combo.blockSignals(False)

        self._sync_sliders_to_current()

    def _sync_sliders_to_current(self) -> None:
        """Move slider handles to match the currently selected scan."""
        rec = self.current

        if self._unique_vf and rec is not None and np.isfinite(rec.v_front):
            idx = min(range(len(self._unique_vf)),
                      key=lambda i: abs(self._unique_vf[i] - rec.v_front))
            self.vf_slider.blockSignals(True)
            self.vf_slider.setValue(idx)
            self.vf_slider.blockSignals(False)
            self.vf_label.setText(f"{rec.v_front:+.3f} V")
        else:
            self.vf_label.setText("--" if not self._unique_vf else
                                  f"{self._unique_vf[self.vf_slider.value()]:+.3f} V")

        if self._unique_pwr and rec is not None and np.isfinite(rec.power_uw):
            idx = min(range(len(self._unique_pwr)),
                      key=lambda i: abs(self._unique_pwr[i] - rec.power_uw))
            self.pwr_slider.blockSignals(True)
            self.pwr_slider.setValue(idx)
            self.pwr_slider.blockSignals(False)
            self.pwr_label.setText(f"{rec.power_uw:.3g} µW")
        else:
            self.pwr_label.setText("--" if not self._unique_pwr else
                                   f"{self._unique_pwr[self.pwr_slider.value()]:.3g} µW")

        self.hel_combo.blockSignals(True)
        if self._hel_mode != "single":
            self.hel_combo.setCurrentText(self._hel_mode)
        elif self._unique_hel and rec is not None and rec.helicity in self._unique_hel:
            self.hel_combo.setCurrentText(rec.helicity)
        self.hel_combo.blockSignals(False)

    def _find_record_row(
        self, target_vf: float, target_pwr: float, target_hel: str = ""
    ) -> int:
        """Return the index of the record that best matches (target_vf, target_pwr, target_hel)."""
        best_row, best_dist = -1, float("inf")
        for i, r in enumerate(self.records):
            d = 0.0
            # Helicity mismatch is a hard penalty
            if target_hel and r.helicity and r.helicity != target_hel:
                d += 1e8
            if np.isfinite(target_vf) and np.isfinite(r.v_front):
                d += (r.v_front - target_vf) ** 2
            elif np.isfinite(target_vf) != np.isfinite(r.v_front):
                d += 1e6
            if np.isfinite(target_pwr) and np.isfinite(r.power_uw):
                ref = max(abs(target_pwr), 1e-9)
                d += ((r.power_uw - target_pwr) / ref) ** 2
            elif np.isfinite(target_pwr) != np.isfinite(r.power_uw):
                d += 1e6
            if d < best_dist:
                best_dist, best_row = d, i
        return best_row

    def _switch_to_scan(self, row: int) -> None:
        """
        Switch to records[row] while preserving:
          - current time in ps (snaps to nearest step in the new scan)
          - cursor pixel position (corx, cory unchanged)
        Updates the time slider range and redraws.
        """
        # Capture current time in ps before switching
        old_rec = self.current
        old_t_ps: Optional[float] = None
        if old_rec is not None and old_rec.n_delays > 0:
            old_t_ps = old_rec.delays_ps[int(np.clip(
                self._t_idx(), 0, old_rec.n_delays - 1
            ))]

        rec = self.records[row]
        self.scan_table.blockSignals(True)
        self.scan_table.selectRow(row)
        self.scan_table.blockSignals(False)
        self.current = rec
        self._sync_sliders_to_current()

        stk = rec.load_stack()
        if stk is None:
            return

        # Find nearest time step in new scan
        if old_t_ps is not None and rec.n_delays > 0:
            new_t_idx = int(np.argmin(np.abs(rec.delays_ps - old_t_ps)))
        else:
            new_t_idx = 0

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, rec.n_delays - 1))
        self.time_slider.setValue(new_t_idx)
        self.time_slider.blockSignals(False)

        self._update_display(new_t_idx)

    def _on_vf_slider_changed(self, idx: int) -> None:
        if not self._unique_vf:
            return
        target_vf  = self._unique_vf[idx]
        target_pwr = self.current.power_uw if self.current is not None else np.nan
        self.vf_label.setText(f"{target_vf:+.3f} V")
        if self._hel_mode != "single":
            # In derived mode: navigate via LH scan, then recompute derived
            lh_row = self._find_record_row(target_vf, target_pwr, "LH")
            if lh_row >= 0:
                self.current = self.records[lh_row]
                self._sync_sliders_to_current()
                self._enter_derived_mode(self._hel_mode)
            return
        target_hel = self.hel_combo.currentText() if self.hel_combo.isEnabled() else ""
        if target_hel not in self._unique_hel:
            target_hel = ""
        row = self._find_record_row(target_vf, target_pwr, target_hel)
        if row >= 0:
            self._switch_to_scan(row)

    def _on_pwr_slider_changed(self, idx: int) -> None:
        if not self._unique_pwr:
            return
        target_pwr = self._unique_pwr[idx]
        target_vf  = self.current.v_front  if self.current is not None else np.nan
        self.pwr_label.setText(f"{target_pwr:.3g} µW")
        if self._hel_mode != "single":
            lh_row = self._find_record_row(target_vf, target_pwr, "LH")
            if lh_row >= 0:
                self.current = self.records[lh_row]
                self._sync_sliders_to_current()
                self._enter_derived_mode(self._hel_mode)
            return
        target_hel = self.hel_combo.currentText() if self.hel_combo.isEnabled() else ""
        if target_hel not in self._unique_hel:
            target_hel = ""
        row = self._find_record_row(target_vf, target_pwr, target_hel)
        if row >= 0:
            self._switch_to_scan(row)

    def _on_hel_changed(self, hel: str) -> None:
        if not hel:
            return
        if hel in ("LH-RH", "norm"):
            self._enter_derived_mode(hel)
        elif hel in self._unique_hel:
            self._hel_mode = "single"
            self._derived_stack = None
            target_vf  = self.current.v_front  if self.current is not None else np.nan
            target_pwr = self.current.power_uw if self.current is not None else np.nan
            row = self._find_record_row(target_vf, target_pwr, hel)
            if row >= 0:
                self._switch_to_scan(row)

    def _enter_derived_mode(self, mode: str) -> None:
        """Compute and display the LH-RH or norm derived stack for current Vf/Power."""
        target_vf  = self.current.v_front  if self.current is not None else np.nan
        target_pwr = self.current.power_uw if self.current is not None else np.nan

        lh_row = self._find_record_row(target_vf, target_pwr, "LH")
        rh_row = self._find_record_row(target_vf, target_pwr, "RH")
        if lh_row < 0 or rh_row < 0:
            self.statusBar().showMessage("Could not find matching LH and RH scans.")
            return

        self._lh_rec = self.records[lh_row]
        self._rh_rec = self.records[rh_row]
        lh_stk = self._lh_rec.load_stack()
        rh_stk = self._rh_rec.load_stack()
        if lh_stk is None or rh_stk is None:
            self.statusBar().showMessage("Failed to load LH/RH stacks for derived mode.")
            return

        Nt = min(lh_stk.shape[0], rh_stk.shape[0])
        lh = lh_stk[:Nt].astype(np.float32)
        rh = rh_stk[:Nt].astype(np.float32)

        if mode == "LH-RH":
            self._derived_stack = lh - rh
        else:  # norm: (LH-RH)/(LH+RH)  clipped to [-1, 1]
            denom = lh + rh
            self._derived_stack = np.clip(
                (lh - rh) / (np.abs(denom) + 1e-9), -1.0, 1.0
            ).astype(np.float32)

        self._hel_mode = mode
        # Use LH record as the metadata source (axes, pixel_size, etc.)
        old_t_ps: Optional[float] = None
        if self.current is not None and self.current.n_delays > 0:
            old_t_ps = self.current.delays_ps[int(np.clip(
                self._t_idx(), 0, self.current.n_delays - 1
            ))]
        self.current = self._lh_rec

        if old_t_ps is not None and Nt > 0:
            new_t_idx = int(np.argmin(np.abs(self._lh_rec.delays_ps[:Nt] - old_t_ps)))
        else:
            new_t_idx = 0

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, Nt - 1))
        self.time_slider.setValue(new_t_idx)
        self.time_slider.blockSignals(False)

        self._sync_sliders_to_current()
        self._update_display(new_t_idx)
        self.statusBar().showMessage(
            f"Derived [{mode}]  LH: {self._lh_rec.label}  |  RH: {self._rh_rec.label}"
        )

    def _current_delays(self) -> np.ndarray:
        """Return the delay axis for the currently displayed stack."""
        rec = self.current
        if rec is None or rec.n_delays == 0:
            return np.array([])
        if self._derived_stack is not None:
            return rec.delays_ps[:self._derived_stack.shape[0]]
        return rec.delays_ps

    # ── Gate / power dependence plots ────────────────────────────────────────

    def _update_dep_plots(self) -> None:
        """
        Collect the signal at the cursor pixel and current time from every
        scan record, then plot:
          - Gate dep: signal vs V_front (grouped by angle/power)
          - Power dep: signal vs power_µW (grouped by V_front)
        The currently selected scan is highlighted in red.
        """
        if not self.records:
            return
        rec = self.current
        t_idx = self._t_idx()
        ix = self.corx
        iy = self.cory
        n = len(self.records)
        self.statusBar().showMessage(f"Computing dependence plots ({n} scans)…")
        QtWidgets.QApplication.processEvents()

        # Common reference values from the active scan
        ref_puw = rec.power_uw  if rec is not None else np.nan
        ref_ang = rec.angle     if rec is not None else np.nan
        ref_vf  = rec.v_front   if rec is not None else np.nan
        ref_hel = rec.helicity  if rec is not None else ""

        # ── Gate dependence: group by similar angle / power / helicity ──
        gate_vf:  List[float] = []
        gate_sig: List[float] = []
        gate_col: List[tuple] = []

        for r in self.records:
            if not np.isfinite(r.v_front):
                continue
            # Filter to same helicity when applicable
            if ref_hel and r.helicity and r.helicity != ref_hel:
                continue
            # Include scans at the same power / angle as the active scan
            same_pwr = (
                np.isfinite(ref_puw) and np.isfinite(r.power_uw)
                and abs(r.power_uw - ref_puw) / max(abs(ref_puw), 1e-9) < 0.06
            )
            same_ang = (
                np.isfinite(ref_ang) and np.isfinite(r.angle)
                and abs(r.angle - ref_ang) < 0.6
            )
            no_pwr_info = (
                not np.isfinite(ref_puw) and not np.isfinite(r.power_uw)
                and not np.isfinite(ref_ang) and not np.isfinite(r.angle)
            )
            if not (same_pwr or same_ang or no_pwr_info):
                continue
            val = r.value_at(t_idx, iy, ix)
            if np.isfinite(val):
                gate_vf.append(r.v_front)
                gate_sig.append(val)
                gate_col.append((200, 60, 60) if r is rec else (80, 80, 200))

        # Fallback: if power/angle filter left < 2 pts, show all finite-Vf records
        if len(gate_vf) < 2:
            gate_vf.clear(); gate_sig.clear(); gate_col.clear()
            for r in self.records:
                if not np.isfinite(r.v_front):
                    continue
                if ref_hel and r.helicity and r.helicity != ref_hel:
                    continue
                val = r.value_at(t_idx, iy, ix)
                if np.isfinite(val):
                    gate_vf.append(r.v_front)
                    gate_sig.append(val)
                    gate_col.append((200, 60, 60) if r is rec else (80, 80, 200))

        self.gate_pw.clear()
        if gate_vf:
            order   = np.argsort(gate_vf)
            vf_s    = np.array(gate_vf)[order]
            sig_s   = np.array(gate_sig)[order]
            clrs    = [gate_col[i] for i in order]
            self.gate_pw.plot(vf_s, sig_s, pen=pg.mkPen((100, 100, 100), width=1))
            self.gate_pw.addItem(pg.ScatterPlotItem(
                x=vf_s, y=sig_s, size=11,
                brush=[pg.mkBrush(c) for c in clrs],
                pen=pg.mkPen("k", width=1),
            ))

        if rec is not None:
            xax = self._x_axis(rec)
            yax = self._y_axis(rec)
            t_ps = rec.delays_ps[t_idx] if rec.n_delays else 0.0
            xlbl = f"λ={xax[ix]:.1f} nm" if (self.wl_m != 1.0 or self.wl_b != 0.0) else f"px {ix}"
            hel_sfx = f"  [{ref_hel}]" if ref_hel else ""
            self.gate_pw.setTitle(
                f"Gate dep  {xlbl}, y={yax[iy]:.1f} µm, t={t_ps:.1f} ps{hel_sfx}"
            )

        # ── Power dependence: group by similar gate voltage / helicity ──
        pwr_p:   List[float] = []
        pwr_sig: List[float] = []
        pwr_col: List[tuple] = []

        for r in self.records:
            if not np.isfinite(r.power_uw):
                continue
            if ref_hel and r.helicity and r.helicity != ref_hel:
                continue
            same_vf = (
                np.isfinite(ref_vf) and np.isfinite(r.v_front)
                and abs(r.v_front - ref_vf) < 0.006   # within 6 mV
            )
            no_gate = not np.isfinite(ref_vf) and not np.isfinite(r.v_front)
            if not (same_vf or no_gate):
                continue
            val = r.value_at(t_idx, iy, ix)
            if np.isfinite(val):
                pwr_p.append(r.power_uw)
                pwr_sig.append(val)
                pwr_col.append((200, 60, 60) if r is rec else (60, 160, 60))

        # Fallback: if Vf filter left < 2 pts, show all finite-power records
        if len(pwr_p) < 2:
            pwr_p.clear(); pwr_sig.clear(); pwr_col.clear()
            for r in self.records:
                if not np.isfinite(r.power_uw):
                    continue
                if ref_hel and r.helicity and r.helicity != ref_hel:
                    continue
                val = r.value_at(t_idx, iy, ix)
                if np.isfinite(val):
                    pwr_p.append(r.power_uw)
                    pwr_sig.append(val)
                    pwr_col.append((200, 60, 60) if r is rec else (60, 160, 60))

        self.pwr_pw.clear()
        if pwr_p:
            order  = np.argsort(pwr_p)
            pw_s   = np.array(pwr_p)[order]
            sg_s   = np.array(pwr_sig)[order]
            clrs   = [pwr_col[i] for i in order]
            self.pwr_pw.plot(pw_s, sg_s, pen=pg.mkPen((100, 100, 100), width=1))
            self.pwr_pw.addItem(pg.ScatterPlotItem(
                x=pw_s, y=sg_s, size=11,
                brush=[pg.mkBrush(c) for c in clrs],
                pen=pg.mkPen("k", width=1),
            ))

        if rec is not None:
            self.pwr_pw.setTitle(
                f"Power dep  {xlbl}, y={yax[iy]:.1f} µm, t={t_ps:.1f} ps{hel_sfx}"
            )

        self.statusBar().showMessage(
            f"Dep plots updated — {len(gate_vf)} gate pts, {len(pwr_p)} power pts  "
            f"(red = current scan)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TAGatePowerViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
