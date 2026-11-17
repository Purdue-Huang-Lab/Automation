"""
measurements/ta_viewer.py

File-by-file viewer for transient-absorption (pump-probe) data: the
*para.txt + *sum.txt pairs written by pump_probe_gate_dep/,
pump_probe_gate_dep_montana/ and the LabVIEW code.

Pick a folder: every dataset in it is listed by name (nothing is read yet).
Click a name to read that one dataset.  Analysis (as ta_gate_power_viewer.py):
  • TA map at a chosen delay: colour map and limits, Gaussian smoothing,
    two-point wavelength calibration; click the map to move the cursor;
  • spectrum and spatial line-outs through the cursor, dynamics at the cursor;
  • LH − RH and (LH − RH)/(LH + RH) with the matching opposite-helicity file;
  • gate and power dependence at the cursor and delay, over the other files
    of the folder (read on demand; click a point to open that file).
Gate voltages, ND angle, helicity and power come from the file names; an
optional angle → power table (.xlsx / .csv) overrides the power.
File loading is shared with ta_gate_power_viewer.py (measurements/ta_data.py).

Usage:
    python -m measurements.ta_viewer [folder]
    or .\\run_ta_viewer.ps1 [-DataFolder folder]
"""
import html
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets   # PyQt5 must come before pyqtgraph
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")

from measurements import ta_data as td  # noqa: E402  (after pyqtgraph config)

# ─────────────────────────────────────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────────────────────────────────────

BTN_W = 64                 # small buttons
STACK_CACHE = 6            # datasets kept in memory (≈12 MB each for 480×128×50)
SELECT_DEBOUNCE_MS = 120   # wait for the list selection to settle before reading
DEFAULT_LEVELS = (-3.0, 3.0)
DASH = QtCore.Qt.DashLine

MODES = [("single", "Single file"), ("LH-RH", "LH − RH"), ("norm", "(LH − RH)/(LH + RH)")]
MODE_NAMES = dict(MODES)

RED = (200, 60, 60)
ERROR_BRUSH = None   # QBrush, made once a QApplication exists


def _error_brush() -> QtGui.QBrush:
    global ERROR_BRUSH
    if ERROR_BRUSH is None:
        ERROR_BRUSH = QtGui.QBrush(QtGui.QColor(190, 0, 0))
    return ERROR_BRUSH


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _power_unit(max_uw: float) -> Tuple[float, str]:
    """(µW per display unit, unit) suited to ``max_uw``."""
    if not _finite(max_uw) or max_uw <= 0:
        return 1.0, "µW"
    if max_uw >= 1e3:
        return 1e3, "mW"
    if max_uw >= 1.0:
        return 1.0, "µW"
    if max_uw >= 1e-3:
        return 1e-3, "nW"
    return 1e-6, "pW"


def _fmt_power(uw: float) -> str:
    if not _finite(uw):
        return "--"
    scale, unit = _power_unit(abs(uw))
    return f"{uw / scale:.4g} {unit}"


def _grid(g: QtWidgets.QGridLayout) -> QtWidgets.QGridLayout:
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return g


def _black_axes(pw: pg.PlotWidget) -> None:
    for ax in ("bottom", "left"):
        pw.getAxis(ax).setPen(pg.mkPen("k"))
        pw.getAxis(ax).setTextPen(pg.mkPen("k"))


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset list entry (file name only until it is read)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Dataset:
    para: Path
    sum: Path
    name: str
    mtime: float
    fn_vf: float
    fn_vb: float
    fn_angle: float
    fn_hel: str
    power_tag_uw: float
    rec: Optional[td.ScanRecord] = None   # para file, once read
    error: str = ""

    @property
    def key(self) -> str:
        return str(self.para)

    @property
    def v_front(self) -> float:
        return self.rec.v_front if self.rec is not None else self.fn_vf

    @property
    def v_back(self) -> float:
        return self.rec.v_back if self.rec is not None else self.fn_vb

    @property
    def angle(self) -> float:
        if self.rec is not None and np.isfinite(self.rec.angle):
            return self.rec.angle
        return self.fn_angle

    @property
    def helicity(self) -> str:
        return self.rec.helicity if self.rec is not None else self.fn_hel


def list_datasets(folder: str) -> List[Dataset]:
    out = []
    for para, summ in td.find_datasets(folder):
        vf, vb, ang, hel = td.parse_fname(para.stem)
        try:
            mtime = summ.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append(Dataset(para, summ, para.name[: -len("para.txt")], mtime,
                           vf, vb, ang, hel, td.parse_power_tag(para.stem)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Workers (file reading off the GUI thread)
# ─────────────────────────────────────────────────────────────────────────────

class LoadWorker(QtCore.QThread):
    """Read the para + sum files of a few datasets."""
    loaded = QtCore.pyqtSignal(object)   # [(Dataset, ScanRecord|None, stack|None, error)]

    def __init__(self, datasets: List[Dataset]) -> None:
        super().__init__()
        self._datasets = list(datasets)

    def run(self) -> None:
        out = []
        for ds in self._datasets:
            try:
                rec = ds.rec if ds.rec is not None else td.record_from_para(ds.para, ds.sum)
                if rec.n_delays == 0:
                    raise ValueError("no delays in the para file")
                stack, n = td.read_sum_stack(ds.sum, rec.crop_h, rec.crop_w, rec.n_delays)
                if n == 0:
                    raise ValueError("the sum file holds no complete delay frame")
                out.append((ds, rec, stack[:n], ""))
            except Exception as exc:
                out.append((ds, None, None, f"{type(exc).__name__}: {exc}"))
        self.loaded.emit(out)


class DepWorker(QtCore.QThread):
    """Signal at one pixel and delay (nearest in ps) from many datasets."""
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(object)     # [(Dataset, ScanRecord|None, value)]

    def __init__(self, datasets: List[Dataset], cached: dict, t_ps: float, iy: int, ix: int) -> None:
        super().__init__()
        self._datasets = list(datasets)
        self._cached = cached            # key -> stack (read only)
        self._t_ps, self._iy, self._ix = t_ps, iy, ix
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        out = []
        n = len(self._datasets)
        for i, ds in enumerate(self._datasets):
            if self._stop:
                break
            rec, val = ds.rec, np.nan
            try:
                if rec is None:
                    rec = td.record_from_para(ds.para, ds.sum)
                stk = self._cached.get(ds.key)
                n_ok = stk.shape[0] if stk is not None else rec.n_delays
                if n_ok > 0:
                    t = rec.t_index(self._t_ps, n_ok)
                    y = int(np.clip(self._iy, 0, rec.crop_h - 1))
                    x = int(np.clip(self._ix, 0, rec.crop_w - 1))
                    if stk is not None:
                        val = float(stk[t, y, x])
                    else:
                        frame = td.read_sum_frame(ds.sum, t, rec.crop_h, rec.crop_w)
                        if frame is not None:
                            val = float(frame[y, x])
            except Exception:
                pass
            out.append((ds, rec, val))
            self.progress.emit(i + 1, n)
        self.done.emit(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class TAViewer(QtWidgets.QMainWindow):

    def __init__(self, folder: str = "") -> None:
        super().__init__()
        self.setWindowTitle("TA Viewer")
        self.resize(1650, 980)
        self._settings = QtCore.QSettings("HuangLab", "TAViewer")

        self.folder = ""
        self.datasets: List[Dataset] = []
        self._by_key: dict = {}
        self.wheel_map: dict = {}
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._threads: set = set()
        self._loading = False
        self._dep_worker: Optional[DepWorker] = None
        self._dep_sets: Tuple[list, list] = ([], [])
        self._dep_ref = None                      # (dataset, t_ps, ix, iy) of the last request
        self._close_t0: Optional[float] = None

        self._target: Optional[Dataset] = None    # dataset the user asked for
        self.cur: Optional[Dataset] = None        # dataset on display
        self._rec: Optional[td.ScanRecord] = None  # axes / metadata source of the display
        self._stack: Optional[np.ndarray] = None  # displayed stack (raw or LH/RH derived)
        self._delays = np.array([])
        self._ghost = None                        # (delays, lh, rh) in derived modes
        self.mode = "single"
        self._shown_mode = "single"
        self.corx: Optional[int] = None           # cursor, spectral pixel
        self.cory: Optional[int] = None           # cursor, spatial pixel
        self.wl_m, self.wl_b = 1.0, 0.0
        self.smooth_sigma = 0.0

        self._select_timer = QtCore.QTimer(self)
        self._select_timer.setSingleShot(True)
        self._select_timer.setInterval(SELECT_DEBOUNCE_MS)
        self._select_timer.timeout.connect(self._on_select_timeout)

        self._build_ui()
        self._connect_signals()

        if not folder:
            self.folder_edit.setText(str(self._settings.value("folder", "") or ""))
        if folder:
            QtCore.QTimer.singleShot(0, lambda: self.load_folder(folder))

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(self._build_left_panel())
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_center_panel())
        split.addWidget(self._build_right_panel())
        split.setSizes([980, 420])
        root.addWidget(split, stretch=1)
        self.statusBar().showMessage("Select a folder with *para.txt + *sum.txt files.")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(290)
        w.setMaximumWidth(380)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 4, 0)

        # ── Folder ──
        fg = QtWidgets.QGroupBox("Data folder")
        fl = _grid(QtWidgets.QGridLayout(fg))
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Folder with *para.txt + *sum.txt…")
        fl.addWidget(self.folder_edit, 0, 0, 1, 3)
        self.folder_btn = QtWidgets.QPushButton("Browse")
        self.load_btn = QtWidgets.QPushButton("Load")
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("List the folder again (new files of a running sweep);\n"
                                    "files that did not change stay in memory.")
        for i, b in enumerate((self.folder_btn, self.load_btn, self.refresh_btn)):
            fl.addWidget(b, 1, i)
        lay.addWidget(fg)

        # ── File list ──
        lg = QtWidgets.QGroupBox("TA datasets")
        ll = _grid(QtWidgets.QGridLayout(lg))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter, e.g.  LH Vfp1")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip("Show names containing all the words (case-insensitive).")
        ll.addWidget(self.filter_edit, 0, 0)
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItems(["Name", "Time"])
        self.sort_combo.setToolTip("Sort by name, or by file time (acquisition order).")
        ll.addWidget(self.sort_combo, 0, 1)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setUniformItemSizes(True)
        self.file_list.setTextElideMode(QtCore.Qt.ElideNone)
        self.file_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        ll.addWidget(self.file_list, 1, 0, 1, 2)
        self.file_list.setMinimumHeight(220)
        self.count_label = QtWidgets.QLabel("No folder loaded")
        ll.addWidget(self.count_label, 2, 0, 1, 2)
        lay.addWidget(lg, stretch=1)          # the list takes all spare height

        # ── Info ──
        ig = QtWidgets.QGroupBox("Selected dataset")
        il = QtWidgets.QVBoxLayout(ig)
        il.setContentsMargins(8, 8, 8, 8)
        self.info_label = QtWidgets.QLabel("—")
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        il.addWidget(self.info_label)
        ig.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        lay.addWidget(ig, stretch=0)

        # ── Power map ──
        pg_box = QtWidgets.QGroupBox("ND wheel → power (optional)")
        pl = _grid(QtWidgets.QGridLayout(pg_box))
        self.power_edit = QtWidgets.QLineEdit()
        self.power_edit.setPlaceholderText("Default: power in file name")
        self.power_edit.setToolTip(
            "Optional angle → power table (.xlsx or .csv, header with 'angle' and\n"
            "'power (nW)' / 'power (uW)'; no unit = µW).  Overrides the _P…nW power\n"
            "tag of the file names (Montana GUI) for the angles it covers."
        )
        pl.addWidget(self.power_edit, 0, 0)
        self.power_btn = QtWidgets.QPushButton("Browse")
        self.power_btn.setFixedWidth(BTN_W)
        pl.addWidget(self.power_btn, 0, 1)
        self.power_clear_btn = QtWidgets.QPushButton("Clear")
        self.power_clear_btn.setFixedWidth(BTN_W)
        pl.addWidget(self.power_clear_btn, 0, 2)
        lay.addWidget(pg_box)

        # ── Wavelength calibration ──
        cg = QtWidgets.QGroupBox("Wavelength cal  (two-point linear)")
        cl = _grid(QtWidgets.QGridLayout(cg))
        self.wl_px1_edit, self.wl_wl1_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.wl_px2_edit, self.wl_wl2_edit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for r, (lbl_px, e_px, lbl_nm, e_nm) in enumerate((
            ("px₁:", self.wl_px1_edit, "nm₁:", self.wl_wl1_edit),
            ("px₂:", self.wl_px2_edit, "nm₂:", self.wl_wl2_edit),
        )):
            cl.addWidget(QtWidgets.QLabel(lbl_px), r, 0)
            e_px.setPlaceholderText("pixel")
            e_px.setFixedWidth(62)
            cl.addWidget(e_px, r, 1)
            cl.addWidget(QtWidgets.QLabel(lbl_nm), r, 2)
            e_nm.setPlaceholderText("λ (nm)")
            e_nm.setFixedWidth(62)
            cl.addWidget(e_nm, r, 3)
        self.wl_apply_btn = QtWidgets.QPushButton("Apply")
        self.wl_apply_btn.setFixedWidth(52)
        cl.addWidget(self.wl_apply_btn, 0, 4)
        self.wl_reset_btn = QtWidgets.QPushButton("Pixels")
        self.wl_reset_btn.setFixedWidth(52)
        self.wl_reset_btn.setToolTip("Back to the spectral pixel axis.")
        cl.addWidget(self.wl_reset_btn, 1, 4)
        lay.addWidget(cg)

        # ── Smoothing + display ──
        dg = QtWidgets.QGroupBox("Display")
        dl = _grid(QtWidgets.QGridLayout(dg))
        dl.addWidget(QtWidgets.QLabel("Smoothing σ:"), 0, 0)
        self.smooth_edit = QtWidgets.QLineEdit("0")
        self.smooth_edit.setFixedWidth(50)
        self.smooth_edit.setToolTip("Gaussian sigma in pixels (0 = off).\n"
                                    "Applied to the image and line-outs; dynamics and\n"
                                    "dependence plots use the raw data.")
        dl.addWidget(self.smooth_edit, 0, 1)
        self.smooth_apply_btn = QtWidgets.QPushButton("Apply")
        self.smooth_apply_btn.setFixedWidth(52)
        dl.addWidget(self.smooth_apply_btn, 0, 2)
        dl.addWidget(QtWidgets.QLabel("Colormap:"), 1, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["RdBu_r", "viridis", "gray"])
        dl.addWidget(self.cmap_combo, 1, 1, 1, 2)
        self.lock_axes_cb = QtWidgets.QCheckBox("Lock axes")
        self.lock_axes_cb.setToolTip("Freeze x/y ranges on all plots so files can be\n"
                                     "compared without auto-rescaling.")
        dl.addWidget(self.lock_axes_cb, 2, 0)
        self.reset_axes_btn = QtWidgets.QPushButton("Reset all scales")
        dl.addWidget(self.reset_axes_btn, 2, 1, 1, 2)
        lay.addWidget(dg)
        return w

    def _build_center_panel(self) -> QtWidgets.QSplitter:
        center = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        ctrl_w = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(ctrl_w)
        cv.setContentsMargins(2, 2, 2, 2)
        cv.setSpacing(2)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("t:"))
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        row1.addWidget(self.time_slider, stretch=1)
        self.time_label = QtWidgets.QLabel("-- ps  (step 0)")
        self.time_label.setMinimumWidth(120)
        row1.addWidget(self.time_label)
        row1.addSpacing(12)
        row1.addWidget(QtWidgets.QLabel("Min:"))
        self.cmin_edit = QtWidgets.QLineEdit(f"{DEFAULT_LEVELS[0]:g}")
        self.cmin_edit.setFixedWidth(55)
        row1.addWidget(self.cmin_edit)
        row1.addWidget(QtWidgets.QLabel("Max:"))
        self.cmax_edit = QtWidgets.QLineEdit(f"{DEFAULT_LEVELS[1]:g}")
        self.cmax_edit.setFixedWidth(55)
        row1.addWidget(self.cmax_edit)
        self.clevel_btn = QtWidgets.QPushButton("Apply")
        self.clevel_btn.setFixedWidth(52)
        row1.addWidget(self.clevel_btn)
        self.cauto_btn = QtWidgets.QPushButton("Auto")
        self.cauto_btn.setFixedWidth(52)
        self.cauto_btn.setToolTip("Colour limits from the shown frame (99.5th percentile;\n"
                                  "symmetric for RdBu_r).")
        row1.addWidget(self.cauto_btn)
        cv.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Helicity:"))
        self.mode_combo = QtWidgets.QComboBox()
        for key, text in MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.setToolTip("Single file, or LH − RH / (LH − RH)/(LH + RH) with the\n"
                                   "opposite-helicity file of the same gate / power.")
        row2.addWidget(self.mode_combo)
        self.partner_label = QtWidgets.QLabel("")
        self.partner_label.setStyleSheet("color: #555;")
        row2.addWidget(self.partner_label, stretch=1)
        self.cursor_label = QtWidgets.QLabel("Cursor: --")
        row2.addWidget(self.cursor_label)
        cv.addLayout(row2)
        center.addWidget(ctrl_w)

        # TA map
        self.img_gw = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_gw.addPlot()
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(td.get_lut("RdBu_r"))
        self.img_plot.addItem(self.img_item)
        self.img_plot.setLabel("bottom", "Spectral pixel")
        self.img_plot.setLabel("left", "Position (µm)")
        self.img_plot.setTitle("TA map (select a file)")
        for ax in ("bottom", "left"):
            self.img_plot.getAxis(ax).setPen(pg.mkPen("k"))
            self.img_plot.getAxis(ax).setTextPen(pg.mkPen("k"))
        self.ch_h = pg.InfiniteLine(angle=0,  pen=pg.mkPen("k", width=1, style=DASH))
        self.ch_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=DASH))
        self.img_plot.addItem(self.ch_h)
        self.img_plot.addItem(self.ch_v)
        center.addWidget(self.img_gw)

        # Line-outs
        lo = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.spec_pw = pg.PlotWidget(title="Spectrum (x-lineout at cursor row)")
        self.spec_pw.setLabel("bottom", "Spectral pixel")
        self.spec_pw.setLabel("left", "ΔT/T (mOD)")
        self.spec_curve = self.spec_pw.plot(pen=pg.mkPen((0, 130, 0), width=1.5))
        self.spec_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=DASH))
        self.spec_pw.addItem(self.spec_vline)
        lo.addWidget(self.spec_pw)
        self.spat_pw = pg.PlotWidget(title="Spatial profile (y-lineout at cursor column)")
        self.spat_pw.setLabel("bottom", "Position (µm)")
        self.spat_pw.setLabel("left", "ΔT/T (mOD)")
        self.spat_curve = self.spat_pw.plot(pen=None, symbol="o", symbolSize=4,
                                            symbolPen=(180, 50, 50), symbolBrush=(180, 50, 50))
        self.spat_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", style=DASH))
        self.spat_pw.addItem(self.spat_vline)
        lo.addWidget(self.spat_pw)
        center.addWidget(lo)

        # Dynamics
        self.dyn_pw = pg.PlotWidget(title="Dynamics at cursor")
        self.dyn_pw.setLabel("bottom", "Time delay (ps)")
        self.dyn_pw.setLabel("left", "ΔT/T (mOD)")
        self.dyn_pw.showGrid(x=True, y=True, alpha=0.25)
        self.dyn_curve = self.dyn_pw.plot(pen=pg.mkPen((0, 90, 200), width=1.8), symbol="o",
                                          symbolSize=5, symbolPen=(0, 90, 200),
                                          symbolBrush=(0, 90, 200))
        self.dyn_lh_curve = self.dyn_pw.plot(pen=pg.mkPen((0, 160, 0, 140), width=1.2, style=DASH))
        self.dyn_rh_curve = self.dyn_pw.plot(pen=pg.mkPen((200, 100, 0, 140), width=1.2, style=DASH))
        self.dyn_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", style=DASH))
        self.dyn_pw.addItem(self.dyn_vline)
        center.addWidget(self.dyn_pw)

        for pw in (self.spec_pw, self.spat_pw, self.dyn_pw):
            _black_axes(pw)
        center.setSizes([60, 380, 210, 240])
        return center

    def _build_right_panel(self) -> QtWidgets.QSplitter:
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.gate_pw = pg.PlotWidget(title="Gate dependence at cursor")
        self.gate_pw.setLabel("bottom", "V_front (V)")
        self.gate_pw.setLabel("left", "ΔT/T (mOD)")
        self.gate_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.gate_pw)
        self.pwr_pw = pg.PlotWidget(title="Power dependence at cursor")
        self.pwr_pw.setLabel("bottom", "Power (µW)")
        self.pwr_pw.setLabel("left", "ΔT/T (mOD)")
        self.pwr_pw.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.pwr_pw)
        for pw in (self.gate_pw, self.pwr_pw):
            _black_axes(pw)

        btn_w = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btn_w)
        bl.setContentsMargins(4, 2, 4, 2)
        self.dep_btn = QtWidgets.QPushButton("Update Gate/Power Plots")
        self.dep_btn.setToolTip(
            "Reads the signal at the cursor pixel and delay (nearest in ps) from the\n"
            "other files of the folder: same power / ND angle for the gate plot, same\n"
            "front gate for the power plot, same helicity for both.\n"
            "Red = the file on display; click a point to open its file."
        )
        bl.addWidget(self.dep_btn)
        right.addWidget(btn_w)
        right.setSizes([420, 420, 40])
        return right

    def _connect_signals(self) -> None:
        self.folder_btn.clicked.connect(self._browse_folder)
        self.load_btn.clicked.connect(lambda: self.load_folder(self.folder_edit.text()))
        self.folder_edit.returnPressed.connect(lambda: self.load_folder(self.folder_edit.text()))
        self.refresh_btn.clicked.connect(self._refresh_folder)
        self.filter_edit.textChanged.connect(self._fill_list)
        self.sort_combo.currentIndexChanged.connect(self._fill_list)
        self.file_list.currentItemChanged.connect(lambda *_: self._select_timer.start())
        self.power_btn.clicked.connect(self._browse_power_map)
        self.power_edit.returnPressed.connect(self._load_power_map)
        self.power_clear_btn.clicked.connect(self._clear_power_map)
        self.wl_apply_btn.clicked.connect(self._apply_wl_cal)
        self.wl_reset_btn.clicked.connect(self._reset_wl_cal)
        self.smooth_apply_btn.clicked.connect(self._apply_smoothing)
        self.smooth_edit.returnPressed.connect(self._apply_smoothing)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.lock_axes_cb.stateChanged.connect(self._on_lock_axes_changed)
        self.reset_axes_btn.clicked.connect(self._reset_all_scales)
        self.time_slider.valueChanged.connect(self._update_display)
        self.clevel_btn.clicked.connect(self._apply_levels)
        self.cmin_edit.returnPressed.connect(self._apply_levels)
        self.cmax_edit.returnPressed.connect(self._apply_levels)
        self.cauto_btn.clicked.connect(self._auto_levels)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.img_gw.scene().sigMouseClicked.connect(self._on_image_click)
        self.dep_btn.clicked.connect(self._on_dep_btn)

    # ── Threads ───────────────────────────────────────────────────────────────

    def _start_thread(self, th: QtCore.QThread) -> None:
        self._threads.add(th)
        th.finished.connect(lambda th=th: self._thread_finished(th))
        th.start()

    def _thread_finished(self, th: QtCore.QThread) -> None:
        th.wait()            # already finished: returns at once
        self._threads.discard(th)

    # ── Stack cache ───────────────────────────────────────────────────────────

    def _cache_put(self, key: str, stack: np.ndarray) -> None:
        self._cache[key] = stack
        self._cache.move_to_end(key)
        while len(self._cache) > STACK_CACHE:
            self._cache.popitem(last=False)

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        stk = self._cache.get(key)
        if stk is not None:
            self._cache.move_to_end(key)
        return stk

    # ── Folder / list ─────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select TA data folder", self.folder_edit.text().strip() or "")
        if d:
            self.load_folder(d)

    def load_folder(self, folder: str) -> None:
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            self.statusBar().showMessage(f"Not a folder: {folder}")
            return
        self.folder = folder
        self.folder_edit.setText(folder)
        self._settings.setValue("folder", folder)
        self.datasets = list_datasets(folder)
        self._by_key = {d.key: d for d in self.datasets}
        self._cache.clear()
        self._target = None
        self._clear_display("TA map (select a file)")
        self.gate_pw.clear()
        self.pwr_pw.clear()
        self.info_label.setText("—")
        self._fill_list()
        n = len(self.datasets)
        self.statusBar().showMessage(f"{n} TA dataset{'s' if n != 1 else ''} in {folder}")

    def _refresh_folder(self) -> None:
        if not self.folder or not os.path.isdir(self.folder):
            self.load_folder(self.folder_edit.text())
            return
        old = self._by_key
        self.datasets = list_datasets(self.folder)
        for d in self.datasets:
            o = old.get(d.key)
            if o is not None and o.mtime == d.mtime:
                d.rec, d.error = o.rec, o.error
            else:
                self._cache.pop(d.key, None)
        self._by_key = {d.key: d for d in self.datasets}
        self._apply_power_map_to_recs()
        tgt = self._target.key if self._target is not None else None
        self._target = self._by_key.get(tgt) if tgt else None
        if self.cur is not None:
            self.cur = self._by_key.get(self.cur.key, self.cur)
        self._fill_list()
        self.statusBar().showMessage(f"{len(self.datasets)} TA datasets in {self.folder}")
        if self._target is not None:
            self._want(self._target)

    def _visible_datasets(self) -> List[Dataset]:
        terms = self.filter_edit.text().lower().split()
        items = [d for d in self.datasets if all(t in d.name.lower() for t in terms)]
        if self.sort_combo.currentText() == "Time":
            items.sort(key=lambda d: (d.mtime, d.name))
        return items

    def _fill_list(self) -> None:
        items = self._visible_datasets()
        keep = self._target.key if self._target is not None else None
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for d in items:
            it = QtWidgets.QListWidgetItem(d.name.rstrip("_") or d.para.name)
            it.setData(QtCore.Qt.UserRole, d.key)
            it.setToolTip(str(d.para)[: -len("para.txt")] + "{para,sum}.txt")
            if d.error:
                it.setForeground(_error_brush())
            self.file_list.addItem(it)
            if d.key == keep:
                self.file_list.setCurrentItem(it)
        self.file_list.blockSignals(False)
        self.count_label.setText(f"{len(items)} of {len(self.datasets)} datasets shown"
                                 if len(items) != len(self.datasets)
                                 else f"{len(items)} datasets")

    def _list_item(self, key: str) -> Optional[QtWidgets.QListWidgetItem]:
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.data(QtCore.Qt.UserRole) == key:
                return it
        return None

    def _select_key(self, key: str) -> None:
        it = self._list_item(key)
        if it is None and key in self._by_key:
            self.filter_edit.clear()            # refills the list
            it = self._list_item(key)
        if it is not None:
            self.file_list.setCurrentItem(it)
            self.file_list.scrollToItem(it)

    def _on_select_timeout(self) -> None:
        it = self.file_list.currentItem()
        if it is None:
            return
        ds = self._by_key.get(it.data(QtCore.Qt.UserRole))
        if ds is not None:
            self._want(ds)

    # ── Partner (opposite helicity) ───────────────────────────────────────────

    def _partner(self, ds: Dataset) -> Optional[Dataset]:
        hel = ds.helicity
        if hel not in ("LH", "RH"):
            return None
        other = "RH" if hel == "LH" else "LH"
        # 1) the same name with the helicity token swapped (sweeps share one timestamp)
        m = td._HEL_RE.search(ds.name)
        if m:
            name = ds.name[: m.start()] + other + ds.name[m.end():]
            for d in self.datasets:
                if d.name == name:
                    return d

        # 2) same gates / angle / power, opposite helicity, nearest in time
        def same(a, b, tol):
            return (not _finite(a) and not _finite(b)) or (_finite(a) and _finite(b) and abs(a - b) <= tol)

        p_ref = self._power(ds)
        p_tol = 1e-3 * abs(p_ref) if _finite(p_ref) else 0.0
        cands = [d for d in self.datasets if d.helicity == other
                 and same(d.v_front, ds.v_front, 1e-4) and same(d.v_back, ds.v_back, 1e-4)
                 and same(d.angle, ds.angle, 0.05) and same(self._power(d), p_ref, p_tol)]
        if not cands:
            return None
        return min(cands, key=lambda d: abs(d.mtime - ds.mtime))

    # ── Loading / display ─────────────────────────────────────────────────────

    def _needed(self, ds: Dataset) -> List[Dataset]:
        need = [ds]
        if self.mode != "single":
            p = self._partner(ds)
            if p is not None:
                need.append(p)
        return need

    def _have(self, d: Dataset) -> bool:
        return bool(d.error) or (d.rec is not None and d.key in self._cache)

    def _want(self, ds: Dataset) -> None:
        """Show ``ds`` (and its partner in LH/RH modes), reading what is not in memory."""
        self._target = ds
        self._update_info(ds)
        self._update_partner_label(ds)
        need = [d for d in self._needed(ds) if not self._have(d)]
        if not need:
            self._show(ds)
            return
        self.statusBar().showMessage(f"Reading {ds.name} …")
        if self._loading:          # re-checked when the running read ends
            return
        self._loading = True
        w = LoadWorker(need)
        w.loaded.connect(self._on_loaded)
        self._start_thread(w)

    def _on_loaded(self, results: list) -> None:
        self._loading = False
        for ds, rec, stack, err in results:
            if err:
                ds.error = err
                it = self._list_item(ds.key)
                if it is not None:
                    it.setForeground(_error_brush())
                continue
            ds.rec, ds.error = rec, ""
            p = td.power_from_map(rec.angle, self.wheel_map)
            rec.power_uw = p if np.isfinite(p) else rec.power_tag_uw
            self._cache_put(ds.key, stack)
        if self._target is not None:
            self._want(self._target)

    def _current_t_ps(self) -> Optional[float]:
        if self._stack is None or not len(self._delays):
            return None
        return float(self._delays[int(np.clip(self.time_slider.value(), 0, len(self._delays) - 1))])

    def _show(self, ds: Dataset) -> None:
        if ds.error:
            self._clear_display(f"Could not read {ds.name}")
            self.statusBar().showMessage(f"Could not read {ds.name}: {ds.error}")
            return
        rec, stack = ds.rec, self._cache_get(ds.key)
        if rec is None or stack is None:
            return
        old_t_ps = self._current_t_ps()
        first = self._stack is None

        disp, delays, src, self._ghost, shown = stack, rec.delays_ps[: len(stack)], rec, None, "single"
        note = ""
        if self.mode != "single":
            partner = self._partner(ds)
            p_stack = self._cache_get(partner.key) if partner is not None else None
            if partner is None:
                note = "no opposite-helicity file: showing the single file"
            elif partner.error or p_stack is None:
                note = f"could not read {partner.name}: showing the single file"
            elif p_stack.shape[1:] != stack.shape[1:]:
                note = "LH and RH images differ in size: showing the single file"
            else:
                lh_ds, lh, rh = ((ds, stack, p_stack) if ds.helicity == "LH"
                                 else (partner, p_stack, stack))
                nt = min(len(lh), len(rh))
                lh = lh[:nt].astype(np.float32)
                rh = rh[:nt].astype(np.float32)
                if self.mode == "LH-RH":
                    disp = lh - rh
                else:
                    disp = np.clip((lh - rh) / (np.abs(lh + rh) + 1e-9), -1.0, 1.0).astype(np.float32)
                src = lh_ds.rec
                delays = src.delays_ps[:nt]
                self._ghost = (delays, lh, rh)
                shown = self.mode

        self.cur, self._rec, self._stack, self._delays, self._shown_mode = ds, src, disp, delays, shown
        n1, n2 = disp.shape[1], disp.shape[2]
        if first or self.corx is None or self.cory is None:
            self.corx, self.cory = n2 // 2, n1 // 2
        self.corx = int(np.clip(self.corx, 0, n2 - 1))
        self.cory = int(np.clip(self.cory, 0, n1 - 1))
        t_idx = int(np.argmin(np.abs(delays - (0.0 if old_t_ps is None else old_t_ps))))

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, len(delays) - 1))
        self.time_slider.setValue(t_idx)
        self.time_slider.blockSignals(False)
        self._update_display(t_idx)

        partial = (f"  |  partial: {len(stack)}/{rec.n_delays} delays"
                   if len(stack) < rec.n_delays else "")
        self.statusBar().showMessage(
            f"{ds.name}  |  {len(delays)} delays  |  {n1}×{n2} px  |  "
            f"pixel={src.pixel_size:.3f} µm{partial}" + (f"  |  {note}" if note else ""))
        self._update_info(ds)
        self._update_partner_label(ds, note)

    def _clear_display(self, title: str) -> None:
        self.cur = self._rec = self._stack = self._ghost = None
        self._delays = np.array([])
        self.img_item.clear()
        self.img_plot.setTitle(title)
        for c in (self.spec_curve, self.spat_curve, self.dyn_curve,
                  self.dyn_lh_curve, self.dyn_rh_curve):
            c.setData([], [])
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, 0)
        self.time_slider.blockSignals(False)
        self.time_label.setText("-- ps  (step 0)")
        self.cursor_label.setText("Cursor: --")

    # ── Axes ──────────────────────────────────────────────────────────────────

    def _wl_on(self) -> bool:
        return self.wl_m != 1.0 or self.wl_b != 0.0

    def _x_axis(self, n2: int) -> np.ndarray:
        """Spectral axis: wavelength (nm) if calibrated, else pixel index."""
        px = np.arange(n2, dtype=float)
        return self.wl_m * px + self.wl_b if self._wl_on() else px

    def _y_axis(self, n1: int) -> np.ndarray:
        """Spatial axis in µm, centred at 0."""
        px = np.arange(n1, dtype=float)
        ps = self._rec.pixel_size if self._rec is not None else 1.0
        return (px - n1 / 2.0) * ps

    def _x_label(self) -> str:
        return "Wavelength (nm)" if self._wl_on() else "Spectral pixel"

    @staticmethod
    def _image_rect(xax: np.ndarray, yax: np.ndarray) -> QtCore.QRectF:
        """Pixel i centred on axis value i (edges half a pixel outside)."""
        dx = (xax[-1] - xax[0]) / (len(xax) - 1) if len(xax) > 1 else 1.0
        dy = (yax[-1] - yax[0]) / (len(yax) - 1) if len(yax) > 1 else 1.0
        return QtCore.QRectF(float(xax[0] - dx / 2), float(yax[0] - dy / 2),
                             float(dx * len(xax)), float(dy * len(yax)))

    def _levels(self) -> Tuple[float, float]:
        try:
            vmin = float(self.cmin_edit.text())
        except ValueError:
            vmin = DEFAULT_LEVELS[0]
        try:
            vmax = float(self.cmax_edit.text())
        except ValueError:
            vmax = DEFAULT_LEVELS[1]
        return vmin, vmax

    # ── Main display update ───────────────────────────────────────────────────

    def _update_display(self, t_idx: Optional[int] = None) -> None:
        stk, rec = self._stack, self._rec
        if stk is None or rec is None or not len(stk):
            return
        if t_idx is None:
            t_idx = self.time_slider.value()
        t_idx = int(np.clip(t_idx, 0, stk.shape[0] - 1))
        frame = self._smooth_frame(stk[t_idx])     # smoothed for image + line-outs
        n1, n2 = frame.shape
        xax, yax = self._x_axis(n2), self._y_axis(n1)
        vmin, vmax = self._levels()

        self.img_item.setImage(frame, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])
        self.img_item.setRect(self._image_rect(xax, yax))
        self.img_plot.setLabel("bottom", self._x_label())
        t_ps = float(self._delays[t_idx])
        mode_sfx = f"  [{MODE_NAMES[self._shown_mode]}]" if self._shown_mode != "single" else ""
        self.img_plot.setTitle(f"{rec.label}{mode_sfx}  |  t = {t_ps:.2f} ps  (step {t_idx})")
        self.time_label.setText(f"{t_ps:.2f} ps  (step {t_idx})")

        ix = int(np.clip(self.corx, 0, n2 - 1))
        iy = int(np.clip(self.cory, 0, n1 - 1))
        self.ch_v.setValue(xax[ix])
        self.ch_h.setValue(yax[iy])

        self.spec_curve.setData(xax, frame[iy, :])
        self.spec_vline.setValue(xax[ix])
        self.spec_pw.setLabel("bottom", self._x_label())
        self.spec_pw.setTitle(f"Spectrum at y = {yax[iy]:.2f} µm")

        self.spat_curve.setData(yax, frame[:, ix])
        self.spat_vline.setValue(yax[iy])
        xlbl = f"λ={xax[ix]:.1f} nm" if self._wl_on() else f"px {ix}"
        self.spat_pw.setTitle(f"Spatial profile at {xlbl}")

        self.dyn_curve.setData(self._delays, stk[:, iy, ix])
        self.dyn_vline.setValue(t_ps)
        self.dyn_pw.setTitle(f"Dynamics at {xlbl}, y = {yax[iy]:.2f} µm{mode_sfx}")
        if self._ghost is not None:
            d, lh, rh = self._ghost
            self.dyn_lh_curve.setData(d, lh[:, iy, ix])
            self.dyn_rh_curve.setData(d, rh[:, iy, ix])
        else:
            self.dyn_lh_curve.setData([], [])
            self.dyn_rh_curve.setData([], [])

        self.cursor_label.setText(
            f"Cursor ({'λ=' if self._wl_on() else 'px'}{xax[ix]:.1f}, {yax[iy]:.2f} µm)"
            f" = {frame[iy, ix]:.3g}" + ("" if self._shown_mode == "norm" else " mOD"))

    def _on_image_click(self, ev) -> None:
        if self._stack is None or ev.button() != QtCore.Qt.LeftButton:
            return
        if not self.img_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = self.img_plot.vb.mapSceneToView(ev.scenePos())
        n1, n2 = self._stack.shape[1], self._stack.shape[2]
        self.corx = int(np.argmin(np.abs(self._x_axis(n2) - pos.x())))
        self.cory = int(np.argmin(np.abs(self._y_axis(n1) - pos.y())))
        self._update_display()

    # ── Levels / calibration / colormap / smoothing ───────────────────────────

    def _apply_levels(self) -> None:
        if self._stack is not None:
            self.img_item.setLevels(list(self._levels()))

    def _auto_levels(self) -> None:
        if self._stack is None:
            return
        t = int(np.clip(self.time_slider.value(), 0, len(self._stack) - 1))
        frame = self._smooth_frame(self._stack[t])
        vals = frame[np.isfinite(frame)]          # LabVIEW files can hold Inf frames
        if not vals.size:
            self.statusBar().showMessage("Auto levels: no finite values in this frame")
            return
        if self.cmap_combo.currentText() == "RdBu_r":
            v = float(np.percentile(np.abs(vals), 99.5)) or 1.0
            vmin, vmax = -v, v
        else:
            vmin, vmax = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
            if vmax <= vmin:
                vmax = vmin + 1.0
        self.cmin_edit.setText(f"{vmin:.3g}")
        self.cmax_edit.setText(f"{vmax:.3g}")
        self.img_item.setLevels([vmin, vmax])

    def _apply_wl_cal(self) -> None:
        try:
            px1, wl1 = float(self.wl_px1_edit.text()), float(self.wl_wl1_edit.text())
            px2, wl2 = float(self.wl_px2_edit.text()), float(self.wl_wl2_edit.text())
        except ValueError:
            self.statusBar().showMessage("Wavelength cal: enter all four values (px1, nm1, px2, nm2).")
            return
        if abs(px2 - px1) < 1e-9:
            self.statusBar().showMessage("Wavelength cal: px1 and px2 must be different.")
            return
        self.wl_m = (wl2 - wl1) / (px2 - px1)
        self.wl_b = wl1 - self.wl_m * px1
        self.statusBar().showMessage(f"Wavelength cal applied: m={self.wl_m:.4f} nm/px, b={self.wl_b:.2f} nm")
        self._update_display()

    def _reset_wl_cal(self) -> None:
        self.wl_m, self.wl_b = 1.0, 0.0
        self.statusBar().showMessage("Spectral axis in pixels")
        self._update_display()

    def _on_cmap_changed(self, name: str) -> None:
        self.img_item.setLookupTable(td.get_lut(name))
        if self._stack is not None:
            self.img_item.setLevels(list(self._levels()))

    def _smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_sigma <= 0:
            return frame
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(frame.astype(np.float32), sigma=self.smooth_sigma)
        except ImportError:
            self.statusBar().showMessage("Smoothing requires scipy")
            return frame

    def _apply_smoothing(self) -> None:
        try:
            sigma = max(0.0, float(self.smooth_edit.text()))
        except ValueError:
            return
        self.smooth_sigma = sigma
        self.statusBar().showMessage(f"Smoothing σ = {sigma} px" if sigma > 0 else "Smoothing off")
        self._update_display()

    # ── Axis locking ──────────────────────────────────────────────────────────

    def _plot_viewboxes(self) -> list:
        vbs = [self.img_plot.getViewBox()]
        for pw in (self.spec_pw, self.spat_pw, self.dyn_pw, self.gate_pw, self.pwr_pw):
            vbs.append(pw.getViewBox())
        return vbs

    def _on_lock_axes_changed(self, state: int) -> None:
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(not bool(state))

    def _reset_all_scales(self) -> None:
        self.lock_axes_cb.blockSignals(True)
        self.lock_axes_cb.setChecked(False)
        self.lock_axes_cb.blockSignals(False)
        for vb in self._plot_viewboxes():
            vb.enableAutoRange(True)

    # ── Helicity mode ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, idx: int) -> None:
        self.mode = self.mode_combo.itemData(idx) or "single"
        if self._target is not None:
            self._want(self._target)

    def _update_partner_label(self, ds: Dataset, note: str = "") -> None:
        p = self._partner(ds)
        if p is None:
            txt = "no opposite-helicity file" if ds.helicity else "no LH/RH tag"
        else:
            txt = f"{p.helicity} partner: {p.name.rstrip('_')}"
        self.partner_label.setText(f"{txt}  —  {note}" if note else txt)
        self.partner_label.setToolTip(p.name if p is not None else "")

    # ── Power ─────────────────────────────────────────────────────────────────

    def _power(self, ds: Dataset) -> float:
        """µW: power map for the dataset's angle, else the file-name tag."""
        if ds.rec is not None:
            return ds.rec.power_uw
        p = td.power_from_map(ds.angle, self.wheel_map)
        return p if np.isfinite(p) else ds.power_tag_uw

    def _power_source(self, ds: Dataset) -> str:
        if np.isfinite(td.power_from_map(ds.angle, self.wheel_map)):
            return "power map"
        return "file name" if np.isfinite(ds.power_tag_uw) else ""

    def _apply_power_map_to_recs(self) -> None:
        td.apply_power_map([d.rec for d in self.datasets if d.rec is not None], self.wheel_map)

    def _browse_power_map(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ND-wheel → power table",
            os.path.dirname(self.power_edit.text().strip()) or self.folder,
            "Power table (*.xlsx *.xls *.csv);;All (*.*)")
        if p:
            self.power_edit.setText(p)
            self._load_power_map()

    def _load_power_map(self) -> None:
        p = self.power_edit.text().strip()
        if not p:
            self._clear_power_map()
            return
        if not os.path.isfile(p):
            self.statusBar().showMessage(f"No such file: {p}")
            return
        self.wheel_map = td.load_wheel_map(p)
        self._apply_power_map_to_recs()
        self.statusBar().showMessage(
            f"Power map: {len(self.wheel_map)} angle → power entries from {Path(p).name}"
            if self.wheel_map else f"No angle / power columns found in {Path(p).name}")
        self._after_power_change()

    def _clear_power_map(self) -> None:
        self.power_edit.clear()
        self.wheel_map = {}
        self._apply_power_map_to_recs()
        self.statusBar().showMessage("Power from the file names")
        self._after_power_change()

    def _after_power_change(self) -> None:
        if self._target is not None:
            self._update_info(self._target)
        self._update_display()

    # ── Info box ──────────────────────────────────────────────────────────────

    def _update_info(self, ds: Dataset) -> None:
        def row(k, v):
            return f"<tr><td style='color:#555;padding-right:8px'>{k}</td><td>{v}</td></tr>"

        rows = []
        if _finite(ds.v_front) or _finite(ds.v_back):
            rows.append(row("Gate", "  ".join(
                [f"Vf = {ds.v_front:+.3f} V" if _finite(ds.v_front) else "",
                 f"Vb = {ds.v_back:+.3f} V" if _finite(ds.v_back) else ""]).strip()))
        if _finite(ds.angle):
            rows.append(row("ND angle", f"{ds.angle:.2f}°"))
        p = self._power(ds)
        if _finite(p):
            rows.append(row("Power", f"{_fmt_power(p)}  <span style='color:#777'>"
                                     f"({self._power_source(ds)})</span>"))
        if ds.helicity:
            rows.append(row("Helicity", ds.helicity))
        r = ds.rec
        if r is not None:
            d = r.delays_ps
            stk = self._cache.get(ds.key)
            n_ok = len(stk) if stk is not None else r.n_delays
            dl = f"{r.n_delays}" + (f" ({d.min():.2f} … {d.max():.2f} ps)" if len(d) else "")
            if n_ok < r.n_delays:
                dl += f"<br><span style='color:#b00'>partial: {n_ok} complete</span>"
            rows.append(row("Delays", dl))
            rows.append(row("Image", f"{r.crop_h} × {r.crop_w} px, {r.pixel_size:.3f} µm/px"))
            if r.n_frames:
                avg = f", avg {int(r.avg.min())}–{int(r.avg.max())}" if len(r.avg) and r.avg.min() != r.avg.max() \
                    else (f", avg {int(r.avg[0])}" if len(r.avg) else "")
                rows.append(row("Frames", f"{r.n_frames}{avg}"))
            rows.append(row("t0 (stage)", f"{r.t0_ps:.2f} ps"))
        elif not ds.error:
            rows.append(row("", "<i>not read yet</i>"))
        if ds.error:
            rows.append(row("Error", f"<span style='color:#b00'>{html.escape(ds.error)}</span>"))
        try:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ds.mtime)) if ds.mtime else ""
        except (OverflowError, ValueError, OSError):
            when = ""
        if when:
            rows.append(row("Saved", when))
        name = html.escape(ds.name.rstrip("_")).replace("_", "_&#8203;")   # allow wrapping
        self.info_label.setText(f"<b>{name}</b><table>{''.join(rows)}</table>")

    # ── Gate / power dependence ───────────────────────────────────────────────

    def _on_dep_btn(self) -> None:
        if self._dep_worker is not None:
            self._dep_worker.stop()
            return
        self._update_dep_plots()

    def _dep_candidates(self, ref: Dataset) -> Tuple[List[Dataset], List[Dataset]]:
        ref_p, ref_ang, ref_vf, ref_hel = self._power(ref), ref.angle, ref.v_front, ref.helicity

        def hel_ok(d):
            return not (ref_hel and d.helicity and d.helicity != ref_hel)

        gate = []
        for d in self.datasets:
            if not _finite(d.v_front) or not hel_ok(d):
                continue
            p = self._power(d)
            same_pwr = _finite(ref_p) and _finite(p) and abs(p - ref_p) / max(abs(ref_p), 1e-9) < 0.06
            same_ang = _finite(ref_ang) and _finite(d.angle) and abs(d.angle - ref_ang) < 0.6
            no_info = not (_finite(ref_p) or _finite(p) or _finite(ref_ang) or _finite(d.angle))
            if same_pwr or same_ang or no_info:
                gate.append(d)
        if len(gate) < 2:
            gate = [d for d in self.datasets if _finite(d.v_front) and hel_ok(d)]

        pwr = []
        for d in self.datasets:
            if not _finite(self._power(d)) or not hel_ok(d):
                continue
            same_vf = _finite(ref_vf) and _finite(d.v_front) and abs(d.v_front - ref_vf) < 0.006
            no_gate = not _finite(ref_vf) and not _finite(d.v_front)
            if same_vf or no_gate:
                pwr.append(d)
        if len(pwr) < 2:
            pwr = [d for d in self.datasets if _finite(self._power(d)) and hel_ok(d)]
        return gate, pwr

    def _update_dep_plots(self) -> None:
        if self.cur is None or self._stack is None:
            self.statusBar().showMessage("Open a file first.")
            return
        gate, pwr = self._dep_candidates(self.cur)
        jobs = list({id(d): d for d in gate + pwr}.values())
        if not jobs:
            self.statusBar().showMessage("No other files with gate or power information.")
            return
        self._dep_sets = (gate, pwr)
        self._dep_ref = (self.cur, float(self._delays[self.time_slider.value()]), self.corx, self.cory)
        w = DepWorker(jobs, dict(self._cache), self._dep_ref[1], self.cory, self.corx)
        w.progress.connect(lambda i, n: self.statusBar().showMessage(f"Reading dependence: {i}/{n} files…"))
        w.done.connect(self._on_dep_done)
        self._dep_worker = w
        self.dep_btn.setText("Stop")
        self._start_thread(w)

    def _on_dep_done(self, out: list) -> None:
        self._dep_worker = None
        self.dep_btn.setText("Update Gate/Power Plots")
        vals = {}
        for ds, rec, val in out:
            if rec is not None and ds.rec is None:
                ds.rec = rec
                p = td.power_from_map(rec.angle, self.wheel_map)
                rec.power_uw = p if np.isfinite(p) else rec.power_tag_uw
            vals[ds.key] = val
        ref, t_ps, ix, iy = self._dep_ref
        gate, pwr = self._dep_sets

        g_pts = [(d.v_front, vals[d.key], d) for d in gate
                 if d.key in vals and _finite(vals[d.key]) and _finite(d.v_front)]
        p_pts = [(self._power(d), vals[d.key], d) for d in pwr
                 if d.key in vals and _finite(vals[d.key]) and _finite(self._power(d))]
        self._plot_dep(self.gate_pw, g_pts, (80, 80, 200), ref)
        scale, unit = _power_unit(max((abs(p[0]) for p in p_pts), default=np.nan))
        p_pts = [(x / scale, y, d) for x, y, d in p_pts]
        self._plot_dep(self.pwr_pw, p_pts, (60, 160, 60), ref)
        self.pwr_pw.setLabel("bottom", f"Power ({unit})")

        n2 = ref.rec.crop_w if ref.rec is not None else 1
        n1 = ref.rec.crop_h if ref.rec is not None else 1
        xax = self._x_axis(n2)
        ps = ref.rec.pixel_size if ref.rec is not None else 1.0
        y_um = (iy - n1 / 2.0) * ps
        xlbl = f"λ={xax[min(ix, n2 - 1)]:.1f} nm" if self._wl_on() else f"px {ix}"
        hel = f"  [{ref.helicity}]" if ref.helicity else ""
        where = f"{xlbl}, y={y_um:.1f} µm, t={t_ps:.1f} ps{hel}"
        self.gate_pw.setTitle(f"Gate dep  {where}")
        self.pwr_pw.setTitle(f"Power dep  {where}")
        stopped = "  (stopped)" if len(out) < len({id(d) for d in gate + pwr}) else ""
        self.statusBar().showMessage(
            f"Dependence plots: {len(g_pts)} gate pts, {len(p_pts)} power pts  "
            f"(red = file on display; click a point to open its file){stopped}")
        if self._target is not None:
            self._update_info(self._target)

    def _plot_dep(self, pw: pg.PlotWidget, pts: list, color: tuple, ref: Dataset) -> None:
        pw.clear()
        if not pts:
            return
        pts = sorted(pts, key=lambda p: p[0])
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        pw.plot(x, y, pen=pg.mkPen((100, 100, 100), width=1))
        sc = pg.ScatterPlotItem(
            x=x, y=y, size=11, pen=pg.mkPen("k", width=1),
            brush=[pg.mkBrush(RED if p[2] is ref else color) for p in pts],
            data=[p[2].key for p in pts], hoverable=True,
            tip=lambda x, y, data: f"{Path(data).name[:-len('para.txt')]}\n{y:.4g}",
        )
        sc.sigClicked.connect(self._on_dep_point_clicked)
        pw.addItem(sc)

    def _on_dep_point_clicked(self, _item, points, *_) -> None:
        if len(points):
            self._select_key(points[0].data())

    # ── Closing ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        # let running reads stop, so no QThread is destroyed while running
        if not any(t.isRunning() for t in self._threads):
            event.accept()
            return
        if self._close_t0 is None:
            self._close_t0 = time.monotonic()
            if self._dep_worker is not None:
                self._dep_worker.stop()
        if time.monotonic() - self._close_t0 > 15.0:
            event.accept()
            return
        event.ignore()
        QtCore.QTimer.singleShot(200, self.close)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    folder = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else ""
    win = TAViewer(folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
