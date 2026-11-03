"""Two-pump differential PL / TRPL viewer: (pump 1 + pump 2) − (pump 1) → dI/dP.

Data: one folder from the PL + TRPL v2 GUI's dI/dP mode (``*_P2off`` / ``*_P2on``
files of every point, paired by name), or two separate sweep folders (legacy).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from measurements.pl_power_gate_dep_viewer_v2 import (
    PLFrame, SingleFrameView, _Channel, _apply_crop, _build_channel, _ch_norm_back,
    _ch_norm_gate, _ch_norm_power, _ch_substrate_for, _corrected_image, _crop_axis,
    _finalise_calib, _frame_at, _linecut_h, _load_andor_ascii, _load_frames,
    _load_sub_folder, _parse_angle_power_calib, _parse_from_name, _resample, _sg_smooth_2d,
)

from . import analysis as an
from . import data as dd
from .config import (
    BTN_W, DATA_DIR, DEFAULT_BG_HI_NM, DEFAULT_BG_LO_NM, DEFAULT_CROP, DEFAULT_P2_UW,
    DEFAULT_REL_FLOOR, DIV_CMAP, SEQ_CMAP, STATUS_W,
)
from .plots import DiffMapWidget, RawCurvesPanel
from .trpl_panel import TrplColumn, TrplDiffPanel
from .ui import px

SWEEP_POWER, SWEEP_FRONT, SWEEP_BACK = "power", "front", "back"
_SWEEP_LABELS = {SWEEP_POWER: "Power", SWEEP_FRONT: "Front gate", SWEEP_BACK: "Back gate"}
MATCH_PAIR, MATCH_NEAREST, MATCH_ORDER = "Same point (file pairs)", "Nearest value", "Sweep order"
VIEW_SET1, VIEW_SET2, VIEW_DIFF = "Pump 2 off", "Pump 2 on", "On − off"


def _f(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.4g}" if np.isfinite(v) else "--"


def _nearest(arr: np.ndarray, v: float, *, log: bool = False) -> float:
    a = np.asarray(arr, dtype=float)
    if not a.size:
        return float(v)
    if log and v > 0 and np.all(a > 0):
        d = np.abs(np.log(a) - np.log(v))
    else:
        d = np.abs(a - v)
    return float(a[int(np.argmin(d))])


class PLdIdPWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ch1 = _Channel()          # PL, pump 1 only (pump 2 off)
        self._ch2 = _Channel()          # PL, pump 1 + pump 2 (pump 2 on)
        self._tch1 = _Channel()         # TRPL stub frames, pump 2 off
        self._tch2 = _Channel()         # TRPL stub frames, pump 2 on
        self._on_by_stem: Dict[str, PLFrame] = {}     # PL pump-2-on frames by pair stem
        self._toff_by_stem: Dict[str, PLFrame] = {}   # TRPL by pair stem
        self._ton_by_stem: Dict[str, PLFrame] = {}
        self._folder: Optional[Path] = None           # dI/dP folder (None in legacy mode)
        self._plan: Optional[dict] = None             # sweep columns shared by the PL and TRPL tabs
        self._crop = DEFAULT_CROP
        self._sweep = SWEEP_POWER
        self._p_idx = self._f_idx = self._b_idx = 0
        self._img_cache: Dict[Tuple[int, Path], Tuple[np.ndarray, np.ndarray]] = {}
        self._calib_path: Optional[Path] = None
        self._maps: Optional[dict] = None      # nm-space maps for the current sweep
        self._q: Dict[str, np.ndarray] = {}    # quantities on the nm grid
        self._disp_y = self._disp_m1 = self._disp_m2 = None   # display-grid copies for the raw panel
        self._row_initialised = False
        self._build_ui()
        self._update_enabled()

    # ══════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._make_data_box(), 3)
        top.addWidget(self._make_processing_box(), 3)
        top.addWidget(self._make_didp_box(), 4)
        root.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        frame_box = QtWidgets.QGroupBox("CCD frame (sets the spatial linecut row)")
        fl = QtWidgets.QVBoxLayout(frame_box)
        fl.setContentsMargins(6, 6, 6, 6)
        rb_row = QtWidgets.QHBoxLayout()
        rb_row.addWidget(QtWidgets.QLabel("Show:"))
        self.viewGroup = QtWidgets.QButtonGroup(self)
        for i, name in enumerate((VIEW_SET1, VIEW_SET2, VIEW_DIFF)):
            rb = QtWidgets.QRadioButton(name)
            rb.setChecked(i == 0)
            self.viewGroup.addButton(rb, i)
            rb_row.addWidget(rb)
        rb_row.addStretch()
        fl.addLayout(rb_row)
        self.liveView = SingleFrameView(title="No data loaded", default_linecut_width=5)
        # SingleFrameView (pl_power_gate_dep_viewer_v2) sizes these for the 8 pt default font
        self.liveView.linecutWidthSpin.setFixedWidth(px(60))
        self.liveView._glw.ci.layout.setColumnMaximumWidth(1, px(65))
        fl.addWidget(self.liveView, 1)
        split.addWidget(frame_box)

        self.mapView = DiffMapWidget("dI/dP map")
        split.addWidget(self.mapView)
        self.rawPanel = RawCurvesPanel()
        split.addWidget(self.rawPanel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)
        self.tabs.addTab(split, "PL")
        self.trplPanel = TrplDiffPanel()
        self.tabs.addTab(self.trplPanel, "TRPL")
        root.addWidget(self.tabs, 1)

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizeGripEnabled(False)
        self.matchLbl = QtWidgets.QLabel("")
        self.matchLbl.setMinimumWidth(px(STATUS_W))
        self.statusBar.addPermanentWidget(self.matchLbl)
        root.addWidget(self.statusBar)

        # signals
        self.viewGroup.buttonClicked.connect(lambda _: self._update_live_view())
        self.liveView.linecut_changed.connect(lambda _: self._rebuild_maps())
        self.liveView.linecutWidthSpin.valueChanged.connect(lambda _: self._rebuild_maps())
        self.mapView.cursor_changed.connect(self._on_map_cursor)
        self.trplPanel.column_clicked.connect(lambda ix: self._on_map_cursor(ix, -1))

    @staticmethod
    def _grid(box: QtWidgets.QGroupBox) -> QtWidgets.QGridLayout:
        g = QtWidgets.QGridLayout(box)
        g.setContentsMargins(8, 8, 8, 8)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        return g

    @staticmethod
    def _path_label() -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel("(none)")
        lbl.setStyleSheet("font-style: italic; color: #555;")
        lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        return lbl

    def _make_data_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Data")
        g = self._grid(box)

        g.addWidget(QtWidgets.QLabel("dI/dP folder:"), 0, 0)
        self.folderLbl = self._path_label()
        self.folderLbl.setToolTip("One sweep folder with *_P2off / *_P2on files (PL + TRPL v2 GUI, dI/dP mode)")
        g.addWidget(self.folderLbl, 0, 1, 1, 2)
        bf = QtWidgets.QPushButton("Browse…")
        bf.setFixedWidth(px(BTN_W))
        g.addWidget(bf, 0, 3)

        g.addWidget(QtWidgets.QLabel("Legacy set 1  (pump 1):"), 1, 0)
        self.set1Lbl = self._path_label()
        g.addWidget(self.set1Lbl, 1, 1, 1, 2)
        b1 = QtWidgets.QPushButton("Browse…")
        b1.setFixedWidth(px(BTN_W))
        g.addWidget(b1, 1, 3)

        g.addWidget(QtWidgets.QLabel("Legacy set 2  (pump 1 + 2):"), 2, 0)
        self.set2Lbl = self._path_label()
        g.addWidget(self.set2Lbl, 2, 1, 1, 2)
        b2 = QtWidgets.QPushButton("Browse…")
        b2.setFixedWidth(px(BTN_W))
        g.addWidget(b2, 2, 3)
        for w in (self.set1Lbl, self.set2Lbl):
            w.setToolTip("Two separate sweep folders (older data): set 1 = pump 1 only, set 2 = pump 1 + 2")

        g.addWidget(QtWidgets.QLabel("Substrate (both):"), 3, 0)
        self.subLbl = self._path_label()
        g.addWidget(self.subLbl, 3, 1)
        sub_btns = QtWidgets.QHBoxLayout()
        bsf = QtWidgets.QPushButton("File…")
        bsd = QtWidgets.QPushButton("Folder…")
        bsc = QtWidgets.QPushButton("Clear")
        for b in (bsf, bsd, bsc):
            b.setFixedWidth(px(60))
            sub_btns.addWidget(b)
        g.addLayout(sub_btns, 3, 2, 1, 2)

        g.addWidget(QtWidgets.QLabel("Power calib:"), 4, 0)
        self.calibLbl = self._path_label()
        self.calibLbl.setToolTip("Angle→power table for data named by stage angle (A…deg)")
        g.addWidget(self.calibLbl, 4, 1, 1, 2)
        bc = QtWidgets.QPushButton("Load…")
        bc.setFixedWidth(px(BTN_W))
        g.addWidget(bc, 4, 3)

        g.addWidget(QtWidgets.QLabel("Match pump 2 on by:"), 5, 0)
        self.matchCombo = QtWidgets.QComboBox()
        self.matchCombo.addItems([MATCH_PAIR, MATCH_NEAREST, MATCH_ORDER])
        self.matchCombo.setToolTip(
            "How each pump-2-off frame finds its pump-2-on partner.\n"
            "Same point: the file with the same name apart from the _P2off / _P2on tag (exact).\n"
            "Nearest value: same power / gate key (closest if not identical).\n"
            "Sweep order: i-th sorted value of set 1 ↔ i-th of set 2 along the swept axis.")
        g.addWidget(self.matchCombo, 5, 1)
        g.setColumnStretch(1, 1)

        bf.clicked.connect(self._browse_folder)
        b1.clicked.connect(lambda: self._browse_set(1))
        b2.clicked.connect(lambda: self._browse_set(2))
        bsf.clicked.connect(self._browse_sub_file)
        bsd.clicked.connect(self._browse_sub_folder)
        bsc.clicked.connect(self._clear_sub)
        bc.clicked.connect(self._load_calib)
        self.matchCombo.currentTextChanged.connect(lambda _: self._rebuild_all())
        return box

    def _make_processing_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PL processing (applied identically to pump 2 off and on)")
        g = self._grid(box)

        g.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 0, 0)
        crop_row = QtWidgets.QHBoxLayout()
        self.cropSpins = []
        for v in DEFAULT_CROP:
            s = QtWidgets.QSpinBox()
            s.setRange(0, 10000)
            s.setValue(v)
            s.setFixedWidth(px(58))
            self.cropSpins.append(s)
            crop_row.addWidget(s)
        self.cropBtn = QtWidgets.QPushButton("Apply")
        self.cropBtn.setFixedWidth(px(60))
        crop_row.addWidget(self.cropBtn)
        crop_row.addStretch()
        g.addLayout(crop_row, 0, 1, 1, 3)

        self.baselineCheck = QtWidgets.QCheckBox("Baseline")
        self.baselineCheck.setToolTip("Subtract, per spatial row, the mean over this wavelength range")
        g.addWidget(self.baselineCheck, 1, 0)
        bl_row = QtWidgets.QHBoxLayout()
        self.bgLoSpin = QtWidgets.QDoubleSpinBox()
        self.bgHiSpin = QtWidgets.QDoubleSpinBox()
        for s, v in ((self.bgLoSpin, DEFAULT_BG_LO_NM), (self.bgHiSpin, DEFAULT_BG_HI_NM)):
            s.setDecimals(1)
            s.setRange(200.0, 2000.0)
            s.setValue(v)
            s.setSuffix(" nm")
            s.setFixedWidth(px(85))
        bl_row.addWidget(self.bgLoSpin)
        bl_row.addWidget(QtWidgets.QLabel("to"))
        bl_row.addWidget(self.bgHiSpin)
        bl_row.addStretch()
        g.addLayout(bl_row, 1, 1, 1, 3)

        self.subCheck = QtWidgets.QCheckBox("Subtract substrate")
        self.subCheck.setToolTip("Cancels in ΔI; matters for I₁, I₁₊₂, ΔI/I and dlnI/dlnP")
        g.addWidget(self.subCheck, 2, 0, 1, 2)
        self.reverseWlCheck = QtWidgets.QCheckBox("Reverse λ")
        g.addWidget(self.reverseWlCheck, 2, 2)

        g.addWidget(QtWidgets.QLabel("SG smooth:"), 3, 0)
        sg_row = QtWidgets.QHBoxLayout()
        self.sgSweepCheck = QtWidgets.QCheckBox("Sweep")
        self.sgSweepWin = QtWidgets.QSpinBox()
        self.sgSweepOrd = QtWidgets.QSpinBox()
        self.sgWlCheck = QtWidgets.QCheckBox("λ")
        self.sgWlWin = QtWidgets.QSpinBox()
        self.sgWlOrd = QtWidgets.QSpinBox()
        for win, order in ((self.sgSweepWin, self.sgSweepOrd), (self.sgWlWin, self.sgWlOrd)):
            win.setRange(3, 501)
            win.setSingleStep(2)
            win.setValue(5)
            win.setFixedWidth(px(52))
            win.setToolTip("Window (odd)")
            order.setRange(1, 10)
            order.setValue(2)
            order.setFixedWidth(px(42))
            order.setToolTip("Polynomial order")
        for w in (self.sgSweepCheck, self.sgSweepWin, self.sgSweepOrd,
                  self.sgWlCheck, self.sgWlWin, self.sgWlOrd):
            sg_row.addWidget(w)
        sg_row.addStretch()
        g.addLayout(sg_row, 3, 1, 1, 3)

        g.addWidget(QtWidgets.QLabel("Energy axis:"), 4, 0)
        self.yUnitCombo = QtWidgets.QComboBox()
        self.yUnitCombo.addItems(["Energy (eV)", "Wavelength (nm)"])
        g.addWidget(self.yUnitCombo, 4, 1, 1, 2)
        g.setColumnStretch(3, 1)

        self.cropBtn.clicked.connect(self._on_apply_crop)
        self.baselineCheck.toggled.connect(lambda _: self._rebuild_all())
        self.bgLoSpin.editingFinished.connect(self._on_baseline_range)
        self.bgHiSpin.editingFinished.connect(self._on_baseline_range)
        self.subCheck.toggled.connect(lambda _: self._rebuild_all())
        self.reverseWlCheck.toggled.connect(lambda _: self._clear_cache_and_rebuild())
        for w in (self.sgSweepCheck, self.sgWlCheck):
            w.toggled.connect(lambda _: self._rebuild_maps())
        for w in (self.sgSweepWin, self.sgSweepOrd, self.sgWlWin, self.sgWlOrd):
            w.valueChanged.connect(lambda _: self._rebuild_maps())
        self.yUnitCombo.currentTextChanged.connect(lambda _: self._redisplay())
        return box

    def _make_didp_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Sweep and dI/dP")
        g = self._grid(box)

        g.addWidget(QtWidgets.QLabel("Map x-axis:"), 0, 0)
        self.sweepCombo = QtWidgets.QComboBox()
        for key in (SWEEP_POWER, SWEEP_FRONT, SWEEP_BACK):
            self.sweepCombo.addItem(_SWEEP_LABELS[key], key)
        g.addWidget(self.sweepCombo, 0, 1)
        self.logXCheck = QtWidgets.QCheckBox("Log power axis")
        self.logXCheck.setChecked(True)
        g.addWidget(self.logXCheck, 0, 2, 1, 2)

        self.powerSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frontSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.backSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.powerLbl = QtWidgets.QLabel("--")
        self.frontLbl = QtWidgets.QLabel("--")
        self.backLbl = QtWidgets.QLabel("--")
        for r, (name, s, lbl) in enumerate((("P₁:", self.powerSlider, self.powerLbl),
                                            ("Front gate:", self.frontSlider, self.frontLbl),
                                            ("Back gate:", self.backSlider, self.backLbl)), start=1):
            g.addWidget(QtWidgets.QLabel(name), r, 0)
            g.addWidget(s, r, 1, 1, 2)
            lbl.setMinimumWidth(px(190))
            g.addWidget(lbl, r, 3)

        g.addWidget(QtWidgets.QLabel("P₂ (µW):"), 4, 0)
        p2_row = QtWidgets.QHBoxLayout()
        self.p2Spin = QtWidgets.QDoubleSpinBox()
        self.p2Spin.setDecimals(6)
        self.p2Spin.setRange(0.0, 1e6)
        self.p2Spin.setValue(DEFAULT_P2_UW)
        self.p2Spin.setFixedWidth(px(110))
        self.p2Spin.setToolTip(
            "Pump-2 power in pump-1-equivalent µW.\n"
            "If pump 2 differs in wavelength / spot / polarization, its absorbed\n"
            "power differs from its incident power — use 'Estimate P₂,eff'.")
        p2_row.addWidget(self.p2Spin)
        self.estimateBtn = QtWidgets.QPushButton("Estimate P₂,eff")
        self.estimateBtn.setToolTip(
            "Least-squares P₂ that makes ΔI/P₂ match the numerical dI/dP of set 1\n"
            "(evaluated at P₁ + P₂/2) over the whole map.")
        p2_row.addWidget(self.estimateBtn)
        self.midpointCheck = QtWidgets.QCheckBox("Evaluate at P₁ + P₂/2")
        self.midpointCheck.setChecked(True)
        self.midpointCheck.setToolTip(
            "ΔI/P₂ is a forward difference; it equals dI/dP at the midpoint P₁ + P₂/2\n"
            "to second order. Also uses Ī = (I₁ + I₁₊₂)/2 for dlnI/dlnP.")
        p2_row.addWidget(self.midpointCheck)
        p2_row.addStretch()
        g.addLayout(p2_row, 4, 1, 1, 3)

        g.addWidget(QtWidgets.QLabel("Show:"), 5, 0)
        self.quantityCombo = QtWidgets.QComboBox()
        self.quantityCombo.addItems(an.QUANTITIES)
        g.addWidget(self.quantityCombo, 5, 1)
        q_row = QtWidgets.QHBoxLayout()
        q_row.addWidget(QtWidgets.QLabel("Colour ±max:"))
        self.cmaxSpin = QtWidgets.QDoubleSpinBox()
        self.cmaxSpin.setDecimals(4)
        self.cmaxSpin.setRange(0.0, 1e12)
        self.cmaxSpin.setSpecialValueText("Auto")
        self.cmaxSpin.setFixedWidth(px(95))
        self.cmaxSpin.setToolTip("Half-range of signed colour scales (0 = auto, 98th percentile)")
        q_row.addWidget(self.cmaxSpin)
        q_row.addWidget(QtWidgets.QLabel("I floor:"))
        self.floorSpin = QtWidgets.QDoubleSpinBox()
        self.floorSpin.setDecimals(3)
        self.floorSpin.setRange(0.0, 1.0)
        self.floorSpin.setSingleStep(0.01)
        self.floorSpin.setValue(DEFAULT_REL_FLOOR)
        self.floorSpin.setFixedWidth(px(65))
        self.floorSpin.setToolTip("ΔI/I and dlnI/dlnP are blanked where I < floor × max|I|")
        q_row.addWidget(self.floorSpin)
        q_row.addStretch()
        g.addLayout(q_row, 5, 2, 1, 2)
        g.setColumnStretch(1, 1)

        self.sweepCombo.currentIndexChanged.connect(self._on_sweep_changed)
        self.logXCheck.toggled.connect(lambda _: self._redisplay())
        self.logXCheck.toggled.connect(lambda _: self._refresh_trpl())
        for w in (self.p2Spin, self.floorSpin):
            w.valueChanged.connect(lambda _: self._refresh_trpl())
        self.powerSlider.valueChanged.connect(lambda v: self._on_slider(SWEEP_POWER, v))
        self.frontSlider.valueChanged.connect(lambda v: self._on_slider(SWEEP_FRONT, v))
        self.backSlider.valueChanged.connect(lambda v: self._on_slider(SWEEP_BACK, v))
        self.p2Spin.valueChanged.connect(lambda _: self._recompute())
        self.estimateBtn.clicked.connect(self._on_estimate_p2)
        self.midpointCheck.toggled.connect(lambda _: self._recompute())
        self.quantityCombo.currentTextChanged.connect(lambda _: self._redisplay())
        self.cmaxSpin.valueChanged.connect(lambda _: self._redisplay())
        self.floorSpin.valueChanged.connect(lambda _: self._recompute())
        return box

    # ══════════════════════════════════════════════════════════════════
    # Loading
    # ══════════════════════════════════════════════════════════════════

    def _start_dir(self) -> str:
        if self._folder is not None:
            return str(self._folder.parent)
        for ch in (self._ch1, self._ch2, self._tch1, self._tch2):
            if ch.data_dir is not None:
                return str(ch.data_dir.parent)
        return str(DATA_DIR) if Path(DATA_DIR).exists() else ""

    def _axis_ch(self) -> _Channel:
        """Channel the sliders and the sweep plan come from: PL if loaded, else TRPL."""
        return self._ch1 if self._ch1.has_data else self._tch1

    def _new_channel(self, frames, folder: Path, like: _Channel) -> _Channel:
        """A channel built from ``frames`` that keeps the calibration / substrate of ``like``."""
        new = _Channel()
        new.data_dir = folder
        new._angle_calib_angles, new._angle_calib_powers_w = like._angle_calib_angles, like._angle_calib_powers_w
        new.sub_single, new.sub_grid = like.sub_single, like.sub_grid
        if frames:
            _build_channel(new, frames)
        return new

    def _browse_folder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a dI/dP sweep folder (*_P2off / *_P2on files)", self._start_dir())
        if chosen:
            self.load_folder(Path(chosen))

    def load_folder(self, folder: Path) -> bool:
        """Load one dI/dP folder: PL and TRPL, pump 2 off and on, paired by file name."""
        folder = Path(folder)
        off, on, n_untagged = dd.load_didp_folder(folder)
        pl_by_stem = dd.by_stem(on)
        pl_by_stem.update(dd.by_stem(off))
        toff, err1 = dd.load_trpl_points(folder, False, pl_by_stem)
        ton, err2 = dd.load_trpl_points(folder, True, pl_by_stem)
        if not (off or on or toff or ton):
            extra = (f"\n\n{n_untagged} untagged .asc file(s) found: load them as Legacy set 1 / set 2."
                     if n_untagged else "")
            QtWidgets.QMessageBox.warning(
                self, "No dI/dP data", f"No *_P2off / *_P2on files found in:\n{folder}{extra}")
            return False
        self._folder = folder
        self._ch1 = self._new_channel(off, folder, self._ch1)
        self._ch2 = self._new_channel(on, folder, self._ch2)
        self._tch1 = self._new_channel(toff, folder, self._ch1)
        self._tch2 = self._new_channel(ton, folder, self._ch1)
        self._on_by_stem = dd.by_stem(on)
        self._toff_by_stem = dd.by_stem(toff)
        self._ton_by_stem = dd.by_stem(ton)
        self._img_cache.clear()
        self.trplPanel.clear_data(folder)

        session = dd.read_session(folder)
        if "hist_ns" in session:
            self.trplPanel.set_window(*session["hist_ns"])
        n_pl = len(set(dd.by_stem(off)) & set(self._on_by_stem))
        n_tr = len(set(self._toff_by_stem) & set(self._ton_by_stem))
        text = f"{folder.name}  ({n_pl} PL pairs, {n_tr} TRPL pairs)"
        tip = [str(folder)]
        if session.get("measure"):
            tip.append(f"Measure: {session['measure']}")
        if session.get("didp"):
            tip.append(f"dI/dP: {session['didp']}")
        if n_untagged:
            tip.append(f"{n_untagged} untagged .asc file(s) ignored")
        errors = err1 + err2
        if errors:
            tip.append(f"{len(errors)} TRPL file(s) could not be read:")
            tip.extend(errors[:10])
            text += f"  ·  {len(errors)} unreadable"
        self.folderLbl.setText(text)
        self.folderLbl.setToolTip("\n".join(tip))
        for lbl in (self.set1Lbl, self.set2Lbl):
            lbl.setText("(none)")
            lbl.setToolTip("")
        self.matchCombo.blockSignals(True)
        self.matchCombo.setCurrentText(MATCH_PAIR)
        self.matchCombo.blockSignals(False)

        self._configure_sliders()
        if self._ch1.has_data and not self._row_initialised:
            self._jump_to_brightest_row()
        if not self._ch1.has_data and self._tch1.has_data:
            self.tabs.setCurrentWidget(self.trplPanel)
        self._update_enabled()
        self._rebuild_all()
        return True

    def _browse_set(self, which: int) -> None:
        title = "Select set 1 folder (pump 1 only)" if which == 1 else "Select set 2 folder (pump 1 + pump 2)"
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, title, self._start_dir())
        if chosen:
            self.load_set(which, Path(chosen))

    def load_set(self, which: int, folder: Path) -> bool:
        """Legacy: one of two separate sweep folders (untagged PL .asc and TRPL_*.csv)."""
        folder = Path(folder)
        frames = _load_frames(folder)
        trpl, errors = dd.load_trpl_points(folder, None, dd.by_stem(frames))
        if not frames and not trpl:
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc or TRPL_*.csv files found in:\n{folder}")
            return False
        if self._folder is not None:
            # Leaving single-folder mode: start legacy loading from empty sets
            self._folder = None
            self._ch1 = self._new_channel([], None, self._ch1)
            self._ch2 = self._new_channel([], None, self._ch2)
            self._tch1, self._tch2 = _Channel(), _Channel()
            self._on_by_stem, self._toff_by_stem, self._ton_by_stem = {}, {}, {}
            self.folderLbl.setText("(none)")
            self.folderLbl.setToolTip("")
            self._img_cache.clear()
        ch = self._ch1 if which == 1 else self._ch2
        new = self._new_channel(frames, folder, ch)
        tnew = self._new_channel(trpl, folder, ch)
        text = f"{folder.name}  ({len(frames)} PL, {len(trpl)} TRPL files)"
        if errors:
            text += f"  ·  {len(errors)} unreadable"
        if which == 1:
            self._ch1, self._tch1 = new, tnew
            self._toff_by_stem = dd.by_stem(trpl)
            self.set1Lbl.setText(text)
            self.set1Lbl.setToolTip("\n".join([str(folder)] + errors[:10]))
        else:
            self._ch2, self._tch2 = new, tnew
            self._on_by_stem = dd.by_stem(frames)
            self._ton_by_stem = dd.by_stem(trpl)
            self.set2Lbl.setText(text)
            self.set2Lbl.setToolTip("\n".join([str(folder)] + errors[:10]))
        if self.matchCombo.currentText() == MATCH_PAIR:
            self.matchCombo.blockSignals(True)
            self.matchCombo.setCurrentText(MATCH_NEAREST)
            self.matchCombo.blockSignals(False)
        self._img_cache = {k: v for k, v in self._img_cache.items() if k[0] != which}
        self.trplPanel.clear_data(folder)
        session = dd.read_session(folder)
        if which == 1 and "hist_ns" in session:
            self.trplPanel.set_window(*session["hist_ns"])
        if which == 1:
            self._configure_sliders()
            if self._ch1.has_data and not self._row_initialised:
                self._jump_to_brightest_row()
        self._update_enabled()
        self._rebuild_all()
        return True

    def _jump_to_brightest_row(self) -> None:
        """Put the spatial linecut on the brightest row of the highest-P₁ set-1 frame."""
        ch = self._ch1
        p, vf, vb = self._cur_keys()
        fr = _frame_at(ch, _ch_norm_power(ch, float(ch.powers[-1])), vf, vb) if ch.powers.size else None
        if fr is None:
            return
        img, _ = self._cropped(1, fr)
        prof = np.nansum(img - np.nanmedian(img), axis=1)
        if not np.isfinite(prof).any():
            return
        raw_row = int(np.nanargmax(prof))
        self.liveView.set_image(img)                      # so the view knows the frame height
        self.liveView._hline.setValue(img.shape[0] - 1 - raw_row)   # display row (0 = top)
        self._row_initialised = True

    def _browse_sub_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select substrate file", self._start_dir(), "Andor ASCII (*.asc);;All files (*)")
        if not path:
            return
        p = Path(path)
        try:
            meta, wl, image = _load_andor_ascii(p)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Substrate", str(exc))
            return
        pw, vf, vb = _parse_from_name(p.name)
        fr = PLFrame(path=p, power_w=pw, v_front=vf, v_back=vb, wavelength_nm=wl,
                     image=image, meta=meta, mtime=float(p.stat().st_mtime))
        for ch in (self._ch1, self._ch2):
            ch.sub_single, ch.sub_grid = fr, {}
        self.subLbl.setText(f"File: {p.name}")
        self.subCheck.setChecked(True)
        self._rebuild_all()

    def _browse_sub_folder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select substrate folder", self._start_dir())
        if not chosen:
            return
        folder = Path(chosen)
        _load_sub_folder(self._ch1, folder)
        if not self._ch1.sub_grid:
            QtWidgets.QMessageBox.warning(self, "No data", f"No .asc files found in:\n{folder}")
            return
        self._ch2.sub_single, self._ch2.sub_grid = None, dict(self._ch1.sub_grid)
        self.subLbl.setText(f"Folder: {folder.name} ({len(self._ch1.sub_grid)} files)")
        self.subCheck.setChecked(True)
        self._rebuild_all()

    def _clear_sub(self) -> None:
        for ch in (self._ch1, self._ch2):
            ch.sub_single, ch.sub_grid = None, {}
        self.subLbl.setText("(none)")
        self._rebuild_all()

    def _load_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load angle→power calibration", self._start_dir(),
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
                f"{p.name} has {extra_a.size} numeric pair(s) outside the labelled "
                "'angle'/'power' columns. Include them?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
            if ans == QtWidgets.QMessageBox.Yes:
                angles = np.concatenate([angles, extra_a])
                powers_w = np.concatenate([powers_w, extra_p])
        angles, powers_w = _finalise_calib(angles, powers_w)
        if angles.size < 2:
            QtWidgets.QMessageBox.critical(self, "Calibration load error",
                                           f"{p.name}\n\nNeed at least 2 distinct angles.")
            return
        for ch in (self._ch1, self._ch2, self._tch1, self._tch2):
            ch._angle_calib_angles, ch._angle_calib_powers_w = angles, powers_w
        self._calib_path = p
        self.calibLbl.setText(f"{p.name}  ({angles.size} pts)")
        axis = self._axis_ch()
        if axis.has_data and not axis.angle_mode:
            self.statusBar.showMessage(
                "Calibration loaded, but the file names carry powers, not stage angles — not applied.", 8000)
        self._configure_sliders()
        self._rebuild_all()

    # ══════════════════════════════════════════════════════════════════
    # Sliders / sweep axis
    # ══════════════════════════════════════════════════════════════════

    def _update_enabled(self) -> None:
        ch = self._axis_ch()
        model = self.sweepCombo.model()
        avail = {
            SWEEP_POWER: ch.has_data and ch.powers.size > 1,
            SWEEP_FRONT: ch.has_data and ch.gates.size > 1,
            SWEEP_BACK: ch.has_data and ch.is_grid and ch.backs.size > 1,
        }
        for i in range(self.sweepCombo.count()):
            model.item(i).setEnabled(bool(avail[self.sweepCombo.itemData(i)]) or not ch.has_data)
        if ch.has_data and not avail.get(self._sweep, False):
            for key in (SWEEP_POWER, SWEEP_FRONT, SWEEP_BACK):
                if avail[key]:
                    self._set_sweep(key)
                    break
        self.estimateBtn.setEnabled(self._ch1.has_data and self._ch2.has_data and self._sweep == SWEEP_POWER)
        self._update_slider_labels()

    def _set_sweep(self, key: str) -> None:
        self._sweep = key
        self.sweepCombo.blockSignals(True)
        self.sweepCombo.setCurrentIndex(self.sweepCombo.findData(key))
        self.sweepCombo.blockSignals(False)

    def _on_sweep_changed(self, _i: int) -> None:
        self._sweep = self.sweepCombo.currentData()
        self._update_enabled()
        self._update_slider_labels()
        self._rebuild_all()

    def _configure_sliders(self) -> None:
        ch = self._axis_ch()
        sliders = ((self.powerSlider, ch.powers, "_p_idx"), (self.frontSlider, ch.gates, "_f_idx"),
                   (self.backSlider, ch.backs, "_b_idx"))
        for s, arr, attr in sliders:
            n = int(arr.size) if ch.has_data else 0
            idx = max(0, min(getattr(self, attr), max(0, n - 1)))
            setattr(self, attr, idx)
            s.blockSignals(True)
            s.setRange(0, max(0, n - 1))
            s.setValue(idx)
            s.setEnabled(n > 1)
            s.blockSignals(False)
        # Non-grid data: the back gate follows the front gate
        self.backSlider.setEnabled(ch.is_grid and ch.backs.size > 1)
        if ch.has_data and not ch.is_grid:
            self._sync_back_to_front()
        self._update_slider_labels()

    def _sync_back_to_front(self) -> None:
        ch = self._axis_ch()
        if not ch.gates.size or not ch.backs.size:
            return
        fi = max(0, min(self._f_idx, ch.gates.size - 1))
        vb = ch.backs_for_gate[fi] if fi < ch.backs_for_gate.size else np.nan
        if np.isfinite(vb):
            self._b_idx = int(np.argmin(np.abs(ch.backs - float(vb))))
            self.backSlider.blockSignals(True)
            self.backSlider.setValue(self._b_idx)
            self.backSlider.blockSignals(False)

    def _on_slider(self, axis: str, v: int) -> None:
        setattr(self, {SWEEP_POWER: "_p_idx", SWEEP_FRONT: "_f_idx", SWEEP_BACK: "_b_idx"}[axis], int(v))
        if axis == SWEEP_FRONT and not self._axis_ch().is_grid:
            self._sync_back_to_front()
        self._update_slider_labels()
        if axis == self._sweep and self._plan is not None:
            # Only the cursor moves along the maps
            col = self._col_for_slider(int(v))
            if col is not None and self._maps is not None:
                self.mapView.set_cursor(ix=col)
                self._update_raw_panel()
            self._update_live_view()
            self._refresh_trpl()
        else:
            self._rebuild_maps()
            self._update_live_view()

    def _cur_keys(self) -> Tuple[float, float, float]:
        ch = self._axis_ch()
        p = _ch_norm_power(ch, float(ch.powers[self._p_idx])) if ch.powers.size else ch.default_power_key
        vf = _ch_norm_gate(ch, float(ch.gates[self._f_idx])) if ch.gates.size else ch.default_gate_key
        vb = _ch_norm_back(ch, float(ch.backs[self._b_idx])) if ch.backs.size else ch.default_back_key
        return p, vf, vb

    def _p1_uw_of_key(self, p_key: float) -> Optional[float]:
        """Pump-1 power (µW) for a set-1 power key; None for uncalibrated angles."""
        ch = self._axis_ch()
        if ch.angle_mode:
            if ch._angle_calib_angles is None:
                return None
            return float(np.interp(p_key, ch._angle_calib_angles, ch._angle_calib_powers_w) * 1e6)
        return float(p_key) * 1e6

    def _power_text(self, p_key: float) -> str:
        if self._axis_ch().angle_mode:
            pw = self._p1_uw_of_key(p_key)
            return f"A = {_f(p_key)}°" + (f"  ({_f(pw)} µW)" if pw is not None else "")
        return f"{_f(p_key * 1e6)} µW"

    def _update_slider_labels(self) -> None:
        if not self._axis_ch().has_data:
            for lbl in (self.powerLbl, self.frontLbl, self.backLbl):
                lbl.setText("--")
            return
        p, vf, vb = self._cur_keys()
        tag = {k: ("  ← swept" if k == self._sweep else "") for k in (SWEEP_POWER, SWEEP_FRONT, SWEEP_BACK)}
        self.powerLbl.setText(self._power_text(p) + tag[SWEEP_POWER])
        self.frontLbl.setText(f"{_f(vf)} V" + tag[SWEEP_FRONT])
        self.backLbl.setText(f"{_f(vb)} V" + tag[SWEEP_BACK])

    def _swept_slider_value(self) -> int:
        return {SWEEP_POWER: self._p_idx, SWEEP_FRONT: self._f_idx, SWEEP_BACK: self._b_idx}[self._sweep]

    def _col_for_slider(self, s: int) -> Optional[int]:
        if self._plan is None:
            return None
        hits = np.flatnonzero(self._plan["slider_idx"] == int(s))
        return int(hits[0]) if hits.size else None

    def _on_map_cursor(self, ix: int, _iy: int) -> None:
        """A map column was picked (PL map, or the TRPL tab with ``_iy`` = -1)."""
        if self._plan is None or not 0 <= ix < self._plan["slider_idx"].size:
            return
        s = int(self._plan["slider_idx"][ix])
        slider = {SWEEP_POWER: self.powerSlider, SWEEP_FRONT: self.frontSlider,
                  SWEEP_BACK: self.backSlider}[self._sweep]
        if s != self._swept_slider_value():
            setattr(self, {SWEEP_POWER: "_p_idx", SWEEP_FRONT: "_f_idx", SWEEP_BACK: "_b_idx"}[self._sweep], s)
            slider.blockSignals(True)
            slider.setValue(s)
            slider.blockSignals(False)
            if self._sweep == SWEEP_FRONT and not self._axis_ch().is_grid:
                self._sync_back_to_front()
            self._update_slider_labels()
            self._update_live_view()
            self._refresh_trpl()
        if self._maps is not None:
            if _iy < 0:
                self.mapView.set_cursor(ix=ix)
            self._update_raw_panel()

    # ══════════════════════════════════════════════════════════════════
    # Frames
    # ══════════════════════════════════════════════════════════════════

    def _on_apply_crop(self) -> None:
        self._crop = tuple(int(s.value()) for s in self.cropSpins)
        self._clear_cache_and_rebuild()

    def _on_baseline_range(self) -> None:
        if self.baselineCheck.isChecked():
            self._rebuild_all()

    def _clear_cache_and_rebuild(self) -> None:
        self._img_cache.clear()
        self._rebuild_all()

    def _cropped(self, which: int, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        key = (which, fr.path)
        hit = self._img_cache.get(key)
        if hit is not None:
            return hit
        out = self._crop_frame(fr)
        self._img_cache[key] = out
        return out

    def _crop_frame(self, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        img = _apply_crop(fr.image, self._crop)
        wl = fr.wavelength_nm if fr.wavelength_nm is not None else np.arange(fr.image.shape[1], dtype=float)
        wl = _crop_axis(wl, self._crop, fr.image.shape[1])
        if self.reverseWlCheck.isChecked():
            img = img[:, ::-1]
        return img, wl

    def _corrected(self, which: int, fr: PLFrame) -> Tuple[np.ndarray, np.ndarray]:
        ch = self._ch1 if which == 1 else self._ch2
        img, wl = self._cropped(which, fr)
        do_bl = self.baselineCheck.isChecked()
        do_sub = self.subCheck.isChecked() and ch.has_substrate
        if not (do_bl or do_sub):
            return img, wl
        sub_img = sub_wl = None
        sub_fr = _ch_substrate_for(ch, fr) if do_sub else None
        if sub_fr is not None:
            sub_img, sub_wl = self._crop_frame(sub_fr)
        out = _corrected_image(img, wl, sub_img, sub_wl, use_baseline=do_bl,
                               lo=float(self.bgLoSpin.value()), hi=float(self.bgHiSpin.value()),
                               use_sub=sub_img is not None)
        return out, wl

    def _match_set2(self, key: Tuple[float, float, float], slider_idx: int, fr1: Optional[PLFrame] = None):
        """Pump-2-on PL frame paired with the pump-2-off frame at `key`; returns (frame, its key)."""
        if fr1 is None and self._ch1.has_data:
            fr1 = _frame_at(self._ch1, *key)
        return self._match_in(self._ch1, self._ch2, self._on_by_stem, fr1, key, slider_idx)

    def _match_in(self, ch1: _Channel, ch2: _Channel, on_by_stem: Dict[str, PLFrame],
                  fr1: Optional[PLFrame], key: Tuple[float, float, float], slider_idx: int):
        """Partner of ``fr1`` (at ``key``) in ``ch2`` by the chosen matching rule."""
        if not ch2.has_data:
            return None, None
        if self.matchCombo.currentText() == MATCH_PAIR:
            fr2 = on_by_stem.get(dd.pair_stem(fr1.path.name)) if fr1 is not None else None
            return fr2, (key if fr2 is not None else None)
        p, vf, vb = key
        p2 = _nearest(ch2.powers, p, log=not ch1.angle_mode)
        vf2 = _nearest(ch2.gates, vf)
        vb2 = _nearest(ch2.backs, vb)
        if self.matchCombo.currentText() == MATCH_ORDER:
            arr = {SWEEP_POWER: ch2.powers, SWEEP_FRONT: ch2.gates, SWEEP_BACK: ch2.backs}[self._sweep]
            if slider_idx >= arr.size:
                return None, None
            v = float(arr[slider_idx])
            if self._sweep == SWEEP_POWER:
                p2 = v
            elif self._sweep == SWEEP_FRONT:
                vf2 = v
            else:
                vb2 = v
        return _frame_at(ch2, p2, vf2, vb2), (p2, vf2, vb2)

    def _linecut(self, which: int, fr: PLFrame, wl_ref: np.ndarray) -> np.ndarray:
        img, wl = self._corrected(which, fr)
        h = img.shape[0]
        row_d = self.liveView.linecut_row()
        row_d = h // 2 if row_d is None else max(0, min(int(row_d), h - 1))
        lc = _linecut_h(img, h - 1 - row_d, int(self.liveView.linecut_width()))
        if lc is None:
            return np.full(wl_ref.shape, np.nan)
        return np.asarray(_resample(wl, lc, wl_ref), dtype=float)

    # ══════════════════════════════════════════════════════════════════
    # Map pipeline:  _rebuild_maps → _recompute → _redisplay
    # ══════════════════════════════════════════════════════════════════

    def _rebuild_all(self) -> None:
        self._update_live_view()
        self._rebuild_maps()

    def _sweep_plan(self):
        """Pump-2-off keys for every map column, their x values and labels."""
        ch = self._axis_ch()
        p, vf, vb = self._cur_keys()
        if self._sweep == SWEEP_POWER:
            keys = [(_ch_norm_power(ch, float(v)), vf, vb) for v in ch.powers]
            p1 = [self._p1_uw_of_key(k[0]) for k in keys]
            if all(v is not None for v in p1):
                p1_uw = np.array(p1, dtype=float)
                return keys, p1_uw, "P₁ (µW)", p1_uw, True
            return keys, ch.powers.astype(float), "Stage angle (°)", None, True
        if self._sweep == SWEEP_FRONT:
            keys = [(p, _ch_norm_gate(ch, float(v)), vb) for v in ch.gates]
            x, label = ch.gates.astype(float), "Front gate (V)"
        else:
            keys = [(p, vf, _ch_norm_back(ch, float(v))) for v in ch.backs]
            x, label = ch.backs.astype(float), "Back gate (V)"
        p_uw = self._p1_uw_of_key(p)
        p1_uw = None if p_uw is None else np.full(x.size, p_uw)
        return keys, x, label, p1_uw, False

    def _rebuild_maps(self) -> None:
        if not self._axis_ch().has_data:
            self._plan = None
            self._maps = None
            self.mapView.clear("Load a dI/dP folder (or legacy set 1 and set 2)")
            self.rawPanel.clear()
            self._refresh_trpl()
            return
        keys, x, x_label, p1_uw, power_sweep = self._sweep_plan()
        # Columns ordered by x (calibrated powers need not follow stage angle)
        order = np.argsort(x, kind="stable")
        self._plan = dict(keys=keys, order=order, slider_idx=np.arange(len(keys))[order], x=x[order],
                          x_label=x_label, power_sweep=power_sweep)
        self._rebuild_pl_maps(keys, x, x_label, p1_uw, power_sweep, order)
        self._refresh_trpl()

    def _rebuild_pl_maps(self, keys, x, x_label, p1_uw, power_sweep, order) -> None:
        ch1 = self._ch1
        if not ch1.has_data:
            self._maps = None
            self.mapView.clear("No PL (.asc) files loaded")
            self.rawPanel.clear()
            self.matchLbl.setText("")
            return
        frames1 = [_frame_at(ch1, *k) for k in keys]
        pairs = [self._match_set2(k, j, fr) for j, (k, fr) in enumerate(zip(keys, frames1))]

        wl_ref = None
        for fr in frames1:
            if fr is not None:
                wl_ref = np.asarray(self._cropped(1, fr)[1], dtype=float)
                break
        if wl_ref is None:
            wl_ref = np.asarray(ch1.blank_wl, dtype=float)

        n = len(keys)
        m1 = np.full((wl_ref.size, n), np.nan)
        m2 = np.full((wl_ref.size, n), np.nan)
        for j, (fr1, (fr2, _)) in enumerate(zip(frames1, pairs)):
            if fr1 is not None:
                m1[:, j] = self._linecut(1, fr1, wl_ref)
            if fr2 is not None:
                m2[:, j] = self._linecut(2, fr2, wl_ref)

        slider_idx = np.arange(n)[order]
        x = x[order]
        m1, m2 = m1[:, order], m2[:, order]
        if p1_uw is not None:
            p1_uw = p1_uw[order]
        m1, m2 = self._sg(m1), self._sg(m2)

        self._maps = dict(x=x, x_label=x_label, wl=wl_ref, m1=m1, m2=m2, p1_uw=p1_uw,
                          power_sweep=power_sweep, slider_idx=slider_idx)
        self._report_matching(keys, pairs, frames1)
        self._recompute()

    def _sg(self, m: np.ndarray) -> np.ndarray:
        ws = int(self.sgSweepWin.value()) if self.sgSweepCheck.isChecked() else 0
        os_ = int(self.sgSweepOrd.value()) if self.sgSweepCheck.isChecked() else 1
        ww = int(self.sgWlWin.value()) if self.sgWlCheck.isChecked() else 0
        ow = int(self.sgWlOrd.value()) if self.sgWlCheck.isChecked() else 1
        return m if (ws < 3 and ww < 3) else _sg_smooth_2d(m, ws, os_, ww, ow)

    def _report_matching(self, keys, pairs, frames1) -> None:
        if not self._ch2.has_data:
            self.matchLbl.setText("No pump-2-on PL files")
            return
        n = len(keys)
        n_ok = sum(1 for fr1, (fr2, _) in zip(frames1, pairs) if fr1 is not None and fr2 is not None)
        if self.matchCombo.currentText() == MATCH_PAIR:
            self.matchLbl.setText(f"PL: matched {n_ok}/{n} columns (same-point file pairs)")
            self.matchLbl.setStyleSheet("color: #c00000;" if n_ok < n else "")
            self.matchLbl.setToolTip("Some points have no _P2off or no _P2on file." if n_ok < n else "")
            return
        axis = {SWEEP_POWER: 0, SWEEP_FRONT: 1, SWEEP_BACK: 2}[self._sweep]
        worst = 0.0
        for k1, (fr2, k2) in zip(keys, pairs):
            if fr2 is None:
                continue
            a, b = float(k1[axis]), float(k2[axis])
            if axis == 0 and not self._ch1.angle_mode and a > 0:
                worst = max(worst, abs(b / a - 1.0) * 100.0)
            else:
                worst = max(worst, abs(b - a))
        unit = "%" if (axis == 0 and not self._ch1.angle_mode) else ("°" if axis == 0 else " V")
        text = f"Matched {n_ok}/{n} columns · worst swept-value mismatch {worst:.3g}{unit}"
        self.matchLbl.setText(text)
        bad = (unit == "%" and worst > 1.0) or n_ok < n
        self.matchLbl.setStyleSheet("color: #c00000;" if bad else "")
        self.matchLbl.setToolTip(
            "Set-1 and set-2 sweep values differ — check that both sets are named by\n"
            "pump-1 power, or switch 'Match pump 2 on by' to 'Sweep order'." if bad else "")

    def _recompute(self) -> None:
        mp = self._maps
        if mp is None:
            return
        self._q = an.compute_quantities(
            mp["m1"], mp["m2"], p1_uw=mp["p1_uw"], p2_uw=float(self.p2Spin.value()),
            midpoint=self.midpointCheck.isChecked(), rel_floor=float(self.floorSpin.value()),
            power_sweep=mp["power_sweep"])
        self._redisplay()

    def _redisplay(self) -> None:
        mp = self._maps
        if mp is None or not self._q:
            return
        unit = "eV" if self.yUnitCombo.currentIndex() == 0 else "nm"
        y_label = "Energy (eV)" if unit == "eV" else "Wavelength (nm)"
        q = self.quantityCombo.currentText()
        ref_name = an.REFERENCE_OF.get(q)

        y, z = an.to_display_y(mp["wl"], self._q[q], unit)
        ref = an.to_display_y(mp["wl"], self._q[ref_name], unit)[1] if ref_name else None
        _, self._disp_m1 = an.to_display_y(mp["wl"], mp["m1"], unit)
        _, self._disp_m2 = an.to_display_y(mp["wl"], mp["m2"], unit)
        self._disp_y = y

        p2 = float(self.p2Spin.value())
        x_eval = None
        if (q in an.MIDPOINT_QUANTITIES and self.midpointCheck.isChecked()
                and mp["power_sweep"] and mp["p1_uw"] is not None and p2 > 0):
            x_eval = mp["p1_uw"] + 0.5 * p2

        symmetric, center = an.SCALE_OF[q]
        units = an.UNITS_OF[q]
        z_label = q.split("≈")[0].split("  ")[0].strip() + (f" ({units})" if units else "")
        cmax = float(self.cmaxSpin.value())
        self.mapView.set_manual_max(cmax if cmax > 0 else None)
        self.mapView.setTitle(self._map_title(q, p2))
        self.mapView.set_data(
            mp["x"], y, z, x_label=mp["x_label"], y_label=y_label, z_label=z_label,
            log_x=self.logXCheck.isChecked() and mp["power_sweep"],
            symmetric=symmetric, center=center,
            cmap=DIV_CMAP if symmetric else SEQ_CMAP, x_eval=x_eval, ref=ref)
        col = self._col_for_slider(self._swept_slider_value())
        if col is not None:
            self.mapView.set_cursor(ix=col)
        self._update_raw_panel()
        self._explain_blank(q, z)

    def _map_title(self, q: str, p2: float) -> str:
        mp = self._maps
        row = self.liveView.linecut_row()
        where = f"row {row if row is not None else '--'} ± {self.liveView.linecut_width() // 2}"
        extra = f"   ·   P₂ = {_f(p2)} µW" if q in (an.Q_DIDP, an.Q_EXP) else ""
        if not mp["power_sweep"]:
            p, _, _ = self._cur_keys()
            extra += f"   ·   P₁ = {self._power_text(p)}"
        return f"{q}   ·   {where}{extra}"

    def _explain_blank(self, q: str, z: np.ndarray) -> None:
        if np.isfinite(z).any():
            return
        mp = self._maps
        if not self._ch2.has_data and q not in (an.Q_I1, an.Q_DIDP_NUM, an.Q_EXP_NUM):
            msg = "No pump-2-on PL files: load them to see this quantity."
        elif q in (an.Q_DIDP, an.Q_EXP) and float(self.p2Spin.value()) <= 0:
            msg = "Set P₂ > 0."
        elif mp["p1_uw"] is None and q in (an.Q_EXP, an.Q_DIDP_NUM, an.Q_EXP_NUM):
            msg = "Pump-1 power unknown — load the angle→power calibration."
        elif not mp["power_sweep"] and q in (an.Q_DIDP_NUM, an.Q_EXP_NUM):
            msg = "Numerical dI/dP needs the map x-axis to be Power."
        else:
            msg = "No finite values (check matching / I floor)."
        self.statusBar.showMessage(msg, 8000)

    def _update_raw_panel(self) -> None:
        mp = self._maps
        if mp is None or self._disp_m1 is None:
            return
        ix, iy = self.mapView.selected()
        if ix is None or iy is None:
            return
        unit_lbl = "Energy (eV)" if self.yUnitCombo.currentIndex() == 0 else "Wavelength (nm)"
        self.rawPanel.update_cursor(
            x=mp["x"], y=self._disp_y, m1=self._disp_m1, m2=self._disp_m2, ix=ix, iy=iy,
            x_label=mp["x_label"], y_label=unit_lbl,
            log_xy=self.logXCheck.isChecked() and mp["power_sweep"])

    # ══════════════════════════════════════════════════════════════════
    # TRPL tab
    # ══════════════════════════════════════════════════════════════════

    def _trpl_pair_at(self, key: Tuple[float, float, float], slider_idx: int):
        """(pump 2 off, pump 2 on) TRPL stub frames of the sweep point at ``key``.

        With PL loaded the TRPL files are found through the PL file names (same
        stem), so both tabs always show the same points.
        """
        fr1 = _frame_at(self._ch1, *key) if self._ch1.has_data else None
        if fr1 is not None:
            off = self._toff_by_stem.get(dd.pair_stem(fr1.path.name))
        else:
            off = _frame_at(self._tch1, *key) if self._tch1.has_data else None
        if self.matchCombo.currentText() == MATCH_PAIR:
            on = self._ton_by_stem.get(dd.pair_stem(off.path.name)) if off is not None else None
        elif fr1 is not None:
            fr2, _ = self._match_set2(key, slider_idx, fr1)
            on = self._ton_by_stem.get(dd.pair_stem(fr2.path.name)) if fr2 is not None else None
        else:
            on, _ = self._match_in(self._tch1, self._tch2, self._ton_by_stem, off, key, slider_idx)
        return off, on

    def _refresh_trpl(self) -> None:
        plan = self._plan
        p2, floor = float(self.p2Spin.value()), float(self.floorSpin.value())
        if plan is None or not (self._tch1.has_data or self._tch2.has_data):
            self.trplPanel.refresh([], 0, x_label="", log_x=False, p2_uw=p2, rel_floor=floor)
            return
        cols: List[TrplColumn] = []
        for i, j in enumerate(plan["order"]):
            off, on = self._trpl_pair_at(plan["keys"][j], int(j))
            ref = off if off is not None else on
            cols.append(TrplColumn(x=float(plan["x"][i]), off=off, on=on,
                                   label=dd.pair_stem(ref.path.name) if ref is not None else ""))
        cur = self._col_for_slider(self._swept_slider_value())
        p, vf, vb = self._cur_keys()
        cond = f"P₁ = {self._power_text(p)}   Vf = {_f(vf)} V   Vb = {_f(vb)} V"
        self.trplPanel.refresh(cols, 0 if cur is None else cur, x_label=plan["x_label"],
                               log_x=self.logXCheck.isChecked() and plan["power_sweep"],
                               p2_uw=p2, rel_floor=floor, cond=cond)

    # ══════════════════════════════════════════════════════════════════
    # P₂ estimate
    # ══════════════════════════════════════════════════════════════════

    def _on_estimate_p2(self) -> None:
        mp = self._maps
        if mp is None or not self._ch2.has_data:
            return
        if not mp["power_sweep"] or mp["p1_uw"] is None:
            QtWidgets.QMessageBox.information(
                self, "Estimate P₂,eff", "Needs a power sweep with known pump-1 powers (µW).")
            return
        p2, n = an.estimate_p2_eff(mp["m1"], mp["m2"], mp["p1_uw"], float(self.p2Spin.value()))
        if not np.isfinite(p2) or p2 <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Estimate P₂,eff",
                f"Fit gave P₂ = {_f(p2)} µW from {n} map cells.\n\n"
                "A non-positive value means ΔI does not track the set-1 slope — usually\n"
                "drift between the two sweeps, mismatched frames, or a pump-2 background.")
            return
        self.p2Spin.setValue(p2)
        self.statusBar.showMessage(f"P₂,eff = {_f(p2)} µW  (least squares over {n} map cells)", 10000)

    # ══════════════════════════════════════════════════════════════════
    # CCD frame view
    # ══════════════════════════════════════════════════════════════════

    def _update_live_view(self) -> None:
        ch1 = self._ch1
        if not ch1.has_data:
            self.liveView.set_image(np.zeros((10, 10)))
            self.liveView.set_image_title("No PL (.asc) files loaded" if self._tch1.has_data else "No data loaded")
            return
        key = self._cur_keys()
        fr1 = _frame_at(ch1, *key)
        fr2, _ = self._match_set2(key, self._swept_slider_value(), fr1)
        view = self.viewGroup.checkedButton().text()

        p, vf, vb = key
        cond = f"P₁ = {self._power_text(p)}   Vf = {_f(vf)} V   Vb = {_f(vb)} V"
        if view == VIEW_SET1:
            fr, which, label = fr1, 1, "pump 2 off"
        elif view == VIEW_SET2:
            fr, which, label = fr2, 2, "pump 2 on"
        else:
            fr, which, label = None, 0, "pump 2 on − off"

        if view == VIEW_DIFF:
            if fr1 is None or fr2 is None:
                self._show_missing(cond, "missing the pump-2-off or pump-2-on frame")
                return
            a, wl = self._corrected(1, fr1)
            b, _ = self._corrected(2, fr2)
            if a.shape != b.shape:
                self._show_missing(cond, f"frame shapes differ {a.shape} vs {b.shape}")
                return
            self.liveView.set_image(b - a, wl)
            self.liveView.set_image_title(f"{label}   ·   {cond}")
            return
        if fr is None:
            self._show_missing(cond, f"no {label} frame")
            return
        img, wl = self._corrected(which, fr)
        self.liveView.set_image(img, wl)
        self.liveView.set_image_title(f"{label}: {fr.path.name}   ·   {cond}")

    def _show_missing(self, cond: str, why: str) -> None:
        blank = self._ch1.blank_image
        img, wl = (np.zeros((10, 10)), None) if blank is None else (
            _apply_crop(blank, self._crop), _crop_axis(self._ch1.blank_wl, self._crop, blank.shape[1]))
        self.liveView.set_image(img, wl)
        self.liveView.set_image_title(f"({why})   ·   {cond}")
