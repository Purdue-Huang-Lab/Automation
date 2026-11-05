"""Wheel (ND) + HWP power calibration v2 widget: sweeps, analysis and the power planner.

Every device call runs off the GUI thread: the stages use ``StageDevice`` (connect,
position poll) plus a ``CallThread`` per move, the power meter lives in a ``PMThread``
and the sweeps run in the v1 ``StageSweepWorker``.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from measurements.dual_wheel_power_calibration import load_hwp_nd_calibration
from measurements.pl_trpl_power_voltage_v2.devices import CallThread, StageDevice, wait_threads
from PM100A.pm100a_wrapper import PM100A
from rot.rot_wrapper import MotionController, RotationStage, list_kinesis_serials

from . import calib
from .calib import fmt_power
from .config import (
    BAD_ROW_BG,
    BTN_W,
    DATA_DIR,
    DEFAULT_HWP_FIXED_ANGLE,
    DEFAULT_HWP_SERIAL,
    DEFAULT_HWP_START,
    DEFAULT_HWP_STEP,
    DEFAULT_HWP_STOP,
    DEFAULT_N_READINGS,
    DEFAULT_ND_ANGLE,
    DEFAULT_ND_SERIAL,
    DEFAULT_ND_START,
    DEFAULT_ND_STEP,
    DEFAULT_ND_STOP,
    DEFAULT_PM_AVERAGING,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_STAGE_SCALE,
    DEFAULT_TARGET_UNIT,
    DEFAULT_TOL_PCT,
    DEFAULT_WAVELENGTH_NM,
    STAGE_POLL_S,
    STATUS_W,
    TARGET_UNITS,
)
from .plots import CoveragePlot, CurvePlot
from .tables import ArrayTableModel
from .workers import PMThread, StageSweepWorker

STAGES = ("nd", "hwp")
STAGE_LABEL = {"nd": "ND wheel", "hwp": "HWP"}
MODES = ("ND wheel + HWP", "ND wheel only")  # ND only: no HWP stage, saved as angle | power (nW) .xlsx
TAB_HWP, TAB_ND, TAB_ANALYSIS, TAB_PLANNER = range(4)


def _grid(box: QtWidgets.QWidget) -> QtWidgets.QGridLayout:
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return g


def _dspin(val, lo, hi, dec=1, step=1.0, suffix="") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


def _button(text: str, width: int = BTN_W) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.setFixedWidth(width)
    return b


def _table_view(model: QtCore.QAbstractTableModel) -> QtWidgets.QTableView:
    v = QtWidgets.QTableView()
    v.setModel(model)
    v.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    v.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    v.verticalHeader().setDefaultSectionSize(20)
    hh = v.horizontalHeader()
    hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    hh.setStretchLastSection(True)
    return v


def _angle(x) -> str:
    return f"{x:.2f}"


def _pct(x) -> str:
    return f"{x * 100:+.3f}"


class WheelHWPCalibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._devs: Dict[str, StageDevice] = {
            name: StageDevice(STAGE_LABEL[name], lambda s: RotationStage(s, scale=DEFAULT_STAGE_SCALE),
                              STAGE_POLL_S, self)
            for name in STAGES
        }
        self._pos: Dict[str, Optional[float]] = {name: None for name in STAGES}
        self._move_threads: Dict[str, CallThread] = {}
        self._move_ctrl: Dict[str, MotionController] = {}
        self._threads = set()  # every CallThread / worker still running (waited for in shutdown)
        self._stage_scan_busy = False
        self._pm_scan_busy = False

        self._pm_thread: Optional[PMThread] = None
        self._pm_ok = False
        self._pm_error = ""
        self._after_pm_close = None
        self._zeroing = False

        self._worker: Optional[StageSweepWorker] = None
        self._running = False
        self._stopping = False
        self._sweep_type: Optional[str] = None  # "hwp" or "nd"
        self._queue_nd = False  # chain the ND sweep after the HWP sweep ("Run both")

        self._hwp_angles: List[float] = []
        self._hwp_powers: List[float] = []
        self._hwp_stds: List[float] = []
        self._nd_angles: List[float] = []
        self._nd_powers: List[float] = []
        self._nd_stds: List[float] = []
        self._nd_fixed: Optional[float] = None  # ND angle during the HWP sweep
        self._hwp_fixed: Optional[float] = None  # HWP angle during the ND sweep
        self._calib_path = ""  # file the current data came from / was saved to

        self._grid_data: Optional[calib.Grid] = None
        self._plan_rows: List[calib.PlanRow] = []
        self._plan_targets: List[float] = []
        self._regrid_timer = QtCore.QTimer(self)
        self._regrid_timer.setSingleShot(True)
        self._regrid_timer.setInterval(200)
        self._regrid_timer.timeout.connect(self._rebuild_grid)

        self._build_ui()
        self._apply_mode()
        for name, dev in self._devs.items():
            dev.state_changed.connect(lambda _s, n=name: self._on_stage_state(n))
            dev.connected.connect(lambda info, n=name: self._on_stage_connected(n, info))
            dev.readback.connect(lambda d, n=name: self._on_stage_readback(n, d))
            dev.message.connect(self._status.showMessage)
        self._sync_controls()

    # ================================================================== UI
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 4)
        outer.setSpacing(8)

        # 1. top bar
        top = QtWidgets.QHBoxLayout()
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems(MODES)
        self.modeCombo.setToolTip("ND wheel only: no HWP; one ND sweep, saved as an angle | power (nW) .xlsx\n"
                                  "that PL+TRPL v2 loads as a single-wheel calibration")
        top.addWidget(QtWidgets.QLabel("Mode:"))
        top.addWidget(self.modeCombo)
        top.addSpacing(16)
        self.btn_run_hwp = _button("Run HWP sweep", 110)
        self.btn_run_nd = _button("Run ND sweep", 110)
        self.btn_run_both = _button("Run both")
        self.btn_stop = _button("Stop")
        self.btn_load = _button("Load CSV")
        self.btn_save = _button("Save CSV")
        for b in (self.btn_run_hwp, self.btn_run_nd, self.btn_run_both, self.btn_stop):
            top.addWidget(b)
        top.addSpacing(16)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_save)
        top.addStretch()
        self.stateLbl = QtWidgets.QLabel("Idle")
        self.stateLbl.setFixedWidth(STATUS_W)
        self.stateLbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.stateLbl)
        outer.addLayout(top)

        # 2. settings panels (a splitter above the tabs: drag up / collapse to give the planner room)
        settings = QtWidgets.QWidget()
        middle = QtWidgets.QHBoxLayout(settings)
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(8)
        middle.addWidget(self._build_devices_box(), 3)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        col.addWidget(self._build_stage_box())
        col.addWidget(self._build_pm_box())
        middle.addLayout(col, 2)
        middle.addWidget(self._build_sweep_box(), 2)

        # 3. display
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_curve_tab("hwp"), "HWP calibration")
        self.tabs.addTab(self._build_curve_tab("nd"), "ND calibration")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_planner_tab(), "Power planner")
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(settings)
        split.addWidget(self.tabs)
        split.setCollapsible(1, False)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)

        # 4. status bar
        self._status = QtWidgets.QStatusBar()
        self._status.setSizeGripEnabled(False)
        outer.addWidget(self._status)

        self.btn_run_hwp.clicked.connect(self._on_start_hwp)
        self.btn_run_nd.clicked.connect(self._on_start_nd)
        self.btn_run_both.clicked.connect(self._on_start_both)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_save.clicked.connect(self._on_save)
        self.modeCombo.currentIndexChanged.connect(self._apply_mode)

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status

    # ------------------------------------------------------------------ devices
    def _build_devices_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Devices")
        g = _grid(box)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.serialCombo: Dict[str, QtWidgets.QComboBox] = {}
        self.btn_scan: Dict[str, QtWidgets.QPushButton] = {}
        self.btn_conn: Dict[str, QtWidgets.QPushButton] = {}
        self.btn_disc: Dict[str, QtWidgets.QPushButton] = {}
        self.devStatusLbl: Dict[str, QtWidgets.QLabel] = {}
        self.posLbl: Dict[str, QtWidgets.QLabel] = {}
        self._hwp_widgets: List[QtWidgets.QWidget] = []  # hidden in ND-only mode
        g.addWidget(QtWidgets.QLabel("Position"), 0, 6)
        for row, (name, default) in enumerate((("nd", DEFAULT_ND_SERIAL), ("hwp", DEFAULT_HWP_SERIAL)), start=1):
            cb = QtWidgets.QComboBox()
            cb.setEditable(True)
            cb.setMinimumWidth(110)
            cb.addItem(default)
            self.serialCombo[name] = cb
            self.btn_scan[name] = _button("Scan", 60)
            self.btn_conn[name] = _button("Connect")
            self.btn_disc[name] = _button("Disconnect")
            self.devStatusLbl[name] = QtWidgets.QLabel("Disconnected")
            self.posLbl[name] = QtWidgets.QLabel("---")
            self.posLbl[name].setFont(mono)
            lbl = QtWidgets.QLabel(f"{STAGE_LABEL[name]}:")
            g.addWidget(lbl, row, 0)
            g.addWidget(cb, row, 1)
            g.addWidget(self.btn_scan[name], row, 2)
            g.addWidget(self.btn_conn[name], row, 3)
            g.addWidget(self.btn_disc[name], row, 4)
            g.addWidget(self.devStatusLbl[name], row, 5)
            g.addWidget(self.posLbl[name], row, 6)
            if name == "hwp":
                self._hwp_widgets += [lbl, cb, self.btn_scan[name], self.btn_conn[name], self.btn_disc[name],
                                      self.devStatusLbl[name], self.posLbl[name]]
            self.btn_scan[name].clicked.connect(self._scan_stages)
            self.btn_conn[name].clicked.connect(lambda _=False, n=name: self._connect_stage(n))
            self.btn_disc[name].clicked.connect(lambda _=False, n=name: self._devs[n].disconnect_device())

        self.pmResourceEdit = QtWidgets.QLineEdit(PM100A.DEFAULT_RESOURCE)
        self.pmResourceEdit.setMinimumWidth(110)
        self.btn_pm_scan = _button("Scan", 60)
        self.btn_pm_conn = _button("Connect")
        self.btn_pm_disc = _button("Disconnect")
        self.pmStatusLbl = QtWidgets.QLabel("Disconnected")
        g.addWidget(QtWidgets.QLabel("Power meter:"), 3, 0)
        g.addWidget(self.pmResourceEdit, 3, 1)
        g.addWidget(self.btn_pm_scan, 3, 2)
        g.addWidget(self.btn_pm_conn, 3, 3)
        g.addWidget(self.btn_pm_disc, 3, 4)
        g.addWidget(self.pmStatusLbl, 3, 5, 1, 2)
        g.setColumnStretch(5, 1)
        g.setRowStretch(4, 1)
        self.btn_pm_scan.clicked.connect(self._scan_pm)
        self.btn_pm_conn.clicked.connect(self._connect_pm)
        self.btn_pm_disc.clicked.connect(self._disconnect_pm)
        return box

    def _build_stage_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Stage control")
        g = _grid(box)
        self.targetSpin = {"nd": _dspin(DEFAULT_ND_ANGLE, 0, 360, 2, 1.0, " °"),
                           "hwp": _dspin(0.0, 0, 360, 2, 1.0, " °")}
        self.btn_move: Dict[str, QtWidgets.QPushButton] = {}
        for row, name in enumerate(STAGES):
            self.btn_move[name] = _button("Move to")
            lbl = QtWidgets.QLabel(f"{STAGE_LABEL[name]}:")
            g.addWidget(lbl, row, 0)
            g.addWidget(self.targetSpin[name], row, 1)
            g.addWidget(self.btn_move[name], row, 2)
            if name == "hwp":
                self._hwp_widgets += [lbl, self.targetSpin[name], self.btn_move[name]]
            self.btn_move[name].clicked.connect(lambda _=False, n=name: self._move(n))
        g.setColumnStretch(1, 1)
        return box

    def _build_pm_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Power meter settings")
        g = _grid(box)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.pmWaveSpin = _dspin(DEFAULT_WAVELENGTH_NM, 300, 1100, 1, 1.0, " nm")
        self.pmAvgSpin = QtWidgets.QSpinBox()
        self.pmAvgSpin.setRange(1, 5000)
        self.pmAvgSpin.setValue(DEFAULT_PM_AVERAGING)
        self.btn_pm_wave = _button("Set")
        self.btn_pm_avg = _button("Set")
        self.btn_pm_read = _button("Read power")
        self.btn_pm_zero = _button("Zero")
        self.pmReadLbl = QtWidgets.QLabel("---")
        self.pmReadLbl.setFont(mono)
        self.pmZeroLbl = QtWidgets.QLabel("not zeroed")
        g.addWidget(QtWidgets.QLabel("Wavelength:"), 0, 0)
        g.addWidget(self.pmWaveSpin, 0, 1)
        g.addWidget(self.btn_pm_wave, 0, 2)
        g.addWidget(QtWidgets.QLabel("Averaging:"), 1, 0)
        g.addWidget(self.pmAvgSpin, 1, 1)
        g.addWidget(self.btn_pm_avg, 1, 2)
        g.addWidget(self.btn_pm_read, 2, 0)
        g.addWidget(self.pmReadLbl, 2, 1, 1, 2)
        g.addWidget(self.btn_pm_zero, 3, 0)
        g.addWidget(self.pmZeroLbl, 3, 1, 1, 2)
        g.setColumnStretch(1, 1)
        self.btn_pm_wave.clicked.connect(lambda: self._pm_submit("wavelength", self.pmWaveSpin.value()))
        self.btn_pm_avg.clicked.connect(lambda: self._pm_submit("averaging", self.pmAvgSpin.value()))
        self.btn_pm_read.clicked.connect(lambda: self._pm_submit("read"))
        self.btn_pm_zero.clicked.connect(self._pm_zero)
        return box

    def _build_sweep_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Sweep parameters")
        vl = QtWidgets.QVBoxLayout(box)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)
        self.sweepTabs = sweep_tabs = QtWidgets.QTabWidget()
        sweep_tabs.setDocumentMode(True)
        vl.addWidget(sweep_tabs)

        self.ndFixedSpin = _dspin(DEFAULT_ND_ANGLE, 0, 360, 1, 1.0, " °")
        self.hwpStartSpin = _dspin(DEFAULT_HWP_START, 0, 360, 1, 1.0, " °")
        self.hwpStopSpin = _dspin(DEFAULT_HWP_STOP, 0, 360, 1, 1.0, " °")
        self.hwpStepSpin = _dspin(DEFAULT_HWP_STEP, 0.1, 90, 2, 0.1, " °")
        self.hwpTotalLbl = QtWidgets.QLabel("")
        self.hwpFixedSpin = _dspin(DEFAULT_HWP_FIXED_ANGLE, 0, 360, 1, 1.0, " °")
        self.ndStartSpin = _dspin(DEFAULT_ND_START, 0, 360, 1, 1.0, " °")
        self.ndStopSpin = _dspin(DEFAULT_ND_STOP, 0, 360, 1, 1.0, " °")
        self.ndStepSpin = _dspin(DEFAULT_ND_STEP, 0.1, 90, 2, 0.5, " °")
        self.ndTotalLbl = QtWidgets.QLabel("")
        for title, rows, total in (
            ("HWP sweep", [("ND fixed angle:", self.ndFixedSpin), ("HWP start:", self.hwpStartSpin),
                           ("HWP stop:", self.hwpStopSpin), ("HWP step:", self.hwpStepSpin)], self.hwpTotalLbl),
            ("ND sweep", [("HWP fixed angle:", self.hwpFixedSpin), ("ND start:", self.ndStartSpin),
                          ("ND stop:", self.ndStopSpin), ("ND step:", self.ndStepSpin)], self.ndTotalLbl),
        ):
            w = QtWidgets.QWidget()
            g = _grid(w)
            for r, (lbl, spin) in enumerate(rows):
                lbl_w = QtWidgets.QLabel(lbl)
                g.addWidget(lbl_w, r, 0)
                g.addWidget(spin, r, 1)
                if spin is self.hwpFixedSpin:
                    self._hwp_widgets += [lbl_w, spin]
            g.addWidget(total, len(rows), 0, 1, 2)
            g.setColumnStretch(1, 1)
            sweep_tabs.addTab(w, title)

        shared = QtWidgets.QGridLayout()
        shared.setHorizontalSpacing(8)
        shared.setVerticalSpacing(6)
        self.nReadingsSpin = QtWidgets.QSpinBox()
        self.nReadingsSpin.setRange(1, 10000)
        self.nReadingsSpin.setValue(DEFAULT_N_READINGS)
        self.rampSpin = _dspin(DEFAULT_RAMP_STEP_DEG, 0.1, 20, 1, 0.5, " °")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setValue(0)
        shared.addWidget(QtWidgets.QLabel("Readings/step:"), 0, 0)
        shared.addWidget(self.nReadingsSpin, 0, 1)
        shared.addWidget(QtWidgets.QLabel("Ramp step:"), 1, 0)
        shared.addWidget(self.rampSpin, 1, 1)
        shared.addWidget(self.progressBar, 2, 0, 1, 2)
        shared.setColumnStretch(1, 1)
        vl.addLayout(shared)

        for spin in (self.hwpStartSpin, self.hwpStopSpin, self.hwpStepSpin,
                     self.ndStartSpin, self.ndStopSpin, self.ndStepSpin):
            spin.valueChanged.connect(self._update_totals)
        self._update_totals()
        return box

    def _build_curve_tab(self, which: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        label = STAGE_LABEL[which]
        plot = CurvePlot(f"{label} calibration", f"{label} angle (°)")
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["#", f"{label} angle (°)", "Mean power"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(20)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(plot)
        split.addWidget(table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        vl.addWidget(split)
        if which == "hwp":
            self.hwpPlot, self.hwpTable = plot, table
        else:
            self.ndPlot, self.ndTable = plot, table
        return w

    def _build_analysis_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("ND wheel angle:"))
        self.ndSelCombo = QtWidgets.QComboBox()
        self.ndSelCombo.setMinimumWidth(100)
        row.addWidget(self.ndSelCombo)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Max power at selection:"))
        self.analysisMaxLbl = QtWidgets.QLabel("---")
        self.analysisMaxLbl.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        row.addWidget(self.analysisMaxLbl)
        row.addStretch()
        vl.addLayout(row)
        self.analysisPlot = CurvePlot("Expected power vs HWP angle", "HWP angle (°)")
        vl.addWidget(self.analysisPlot, 1)
        self.ndSelCombo.currentIndexChanged.connect(self._update_analysis)
        return w

    # ------------------------------------------------------------------ planner tab
    def _build_planner_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        self.hwpLoSpin = _dspin(0, 0, 360, 1, 1.0, " °")
        self.hwpHiSpin = _dspin(360, 0, 360, 1, 1.0, " °")
        self.ndLoSpin = _dspin(0, 0, 360, 1, 1.0, " °")
        self.ndHiSpin = _dspin(360, 0, 360, 1, 1.0, " °")
        self.tolSpin = _dspin(DEFAULT_TOL_PCT, 0.01, 100, 2, 0.1, " %")
        self.tolSpin.setToolTip("Pairs within this of the target are candidates")
        self.angErrSpin = _dspin(0.1, 0.0, 5.0, 2, 0.05, " °")
        self.angErrSpin.setToolTip(
            "Stage repeatability used to rank the candidates:\n"
            "expected error = sqrt(power error² + (sensitivity × angle error)²).\n"
            "0 = strictly the closest power; larger = prefer flat parts of the curves.")
        for lbl, lo, hi in (("HWP limits:", self.hwpLoSpin, self.hwpHiSpin),
                            ("ND limits:", self.ndLoSpin, self.ndHiSpin)):
            row.addWidget(QtWidgets.QLabel(lbl))
            row.addWidget(lo)
            row.addWidget(QtWidgets.QLabel("–"))
            row.addWidget(hi)
            row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Tolerance:"))
        row.addWidget(self.tolSpin)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Angle error:"))
        row.addWidget(self.angErrSpin)
        row.addStretch()
        vl.addLayout(row)
        self.coverageLbl = QtWidgets.QLabel("Load a calibration file or run both sweeps.")
        self.coverageLbl.setWordWrap(True)
        font = self.coverageLbl.font()
        font.setBold(True)
        self.coverageLbl.setFont(font)
        vl.addWidget(self.coverageLbl)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vl.addWidget(vsplit, 1)

        # top: coverage plot | reachable power list
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.coveragePlot = CoveragePlot()
        top.addWidget(self.coveragePlot)
        list_w = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(list_w)
        lv.setContentsMargins(0, 0, 0, 0)
        rb_row = QtWidgets.QHBoxLayout()
        self.rbRanges = QtWidgets.QRadioButton("Range per ND angle")
        self.rbAll = QtWidgets.QRadioButton("All combinations (sorted by power)")
        self.rbRanges.setChecked(True)
        rb_row.addWidget(self.rbRanges)
        rb_row.addWidget(self.rbAll)
        rb_row.addStretch()
        lv.addLayout(rb_row)
        self.listStack = QtWidgets.QStackedWidget()
        self.rangeModel = ArrayTableModel(
            ["ND (°)", "P min", "HWP @ min (°)", "P max", "HWP @ max (°)"],
            [_angle, fmt_power, _angle, fmt_power, _angle], self)
        self.allModel = ArrayTableModel(
            ["Power", "−1σ", "+1σ", "ND (°)", "HWP (°)", "Sens (%/°)"],
            [fmt_power, fmt_power, fmt_power, _angle, _angle, _angle], self)
        self.listStack.addWidget(_table_view(self.rangeModel))
        self.listStack.addWidget(_table_view(self.allModel))
        lv.addWidget(self.listStack)
        top.addWidget(list_w)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        vsplit.addWidget(top)

        # bottom: target list | solved combinations
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        tbox = QtWidgets.QGroupBox("Target powers")
        g = _grid(tbox)
        self.targetEdit = QtWidgets.QPlainTextEdit()
        self.targetEdit.setMinimumHeight(40)
        self.targetEdit.setPlaceholderText("e.g.  1, 2, 5, 10, 20, 50\nor  300 nW, 1.5 uW, 2 µW")
        self.targetUnitCombo = QtWidgets.QComboBox()
        self.targetUnitCombo.addItems(TARGET_UNITS)
        self.targetUnitCombo.setCurrentText(DEFAULT_TARGET_UNIT)
        self.targetUnitCombo.setToolTip("Unit of numbers typed without a unit")
        self.genStartSpin = _dspin(1.0, 1e-4, 1e9, 4, 1.0)
        self.genStopSpin = _dspin(1000.0, 1e-4, 1e9, 4, 1.0)
        self.genNSpin = QtWidgets.QSpinBox()
        self.genNSpin.setRange(1, 1000)
        self.genNSpin.setValue(10)
        self.genLogChk = QtWidgets.QCheckBox("log")
        self.genLogChk.setChecked(True)
        self.btn_gen = _button("Generate")
        self.btn_solve = _button("Solve")
        self.btn_export = _button("Export for PL+TRPL…", 150)
        g.addWidget(self.targetEdit, 0, 0, 1, 6)
        g.addWidget(QtWidgets.QLabel("Unit:"), 1, 0)
        g.addWidget(self.targetUnitCombo, 1, 1)
        g.addWidget(self.btn_solve, 1, 4)
        g.addWidget(self.btn_export, 1, 5)
        g.addWidget(QtWidgets.QLabel("From:"), 2, 0)
        g.addWidget(self.genStartSpin, 2, 1)
        g.addWidget(QtWidgets.QLabel("to"), 2, 2)
        g.addWidget(self.genStopSpin, 2, 3)
        gen_row = QtWidgets.QHBoxLayout()
        gen_row.addWidget(QtWidgets.QLabel("N:"))
        gen_row.addWidget(self.genNSpin)
        gen_row.addWidget(self.genLogChk)
        g.addLayout(gen_row, 2, 4)
        g.addWidget(self.btn_gen, 2, 5)
        g.setRowStretch(0, 1)
        bottom.addWidget(tbox)

        rbox = QtWidgets.QGroupBox("Combinations to use (in list order)")
        rl = QtWidgets.QVBoxLayout(rbox)
        rl.setContentsMargins(8, 8, 8, 8)
        self.planModel = ArrayTableModel(
            ["Target", "Predicted", "Error (%)", "±1σ", "Expected (%)", "ND (°)", "HWP (°)", "Sens (%/°)", "Note"],
            [fmt_power, fmt_power, _pct, fmt_power, lambda x: f"{x * 100:.3f}", _angle, _angle, _angle, str],
            self)
        self.planView = _table_view(self.planModel)
        self.planSummaryLbl = QtWidgets.QLabel("")
        rl.addWidget(self.planView, 1)
        rl.addWidget(self.planSummaryLbl)
        bottom.addWidget(rbox)
        bottom.setStretchFactor(0, 2)
        bottom.setStretchFactor(1, 5)
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 1)

        self.rbRanges.toggled.connect(lambda on: self.listStack.setCurrentIndex(0 if on else 1))
        for spin in (self.hwpLoSpin, self.hwpHiSpin, self.ndLoSpin, self.ndHiSpin):
            spin.valueChanged.connect(lambda _v: self._regrid_timer.start())
        self.tolSpin.valueChanged.connect(self._resolve)
        self.angErrSpin.valueChanged.connect(self._resolve)
        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_solve.clicked.connect(self._on_solve)
        self.btn_export.clicked.connect(self._on_export)
        return w

    # ================================================================== stages
    def _keep(self, th: QtCore.QThread) -> QtCore.QThread:
        self._threads.add(th)
        th.finished.connect(lambda th=th: self._threads.discard(th))
        return th

    def _scan_stages(self) -> None:
        th = self._keep(CallThread(list_kinesis_serials, self))
        th.done.connect(self._on_scan_done)
        self._stage_scan_busy = True
        self._status.showMessage("Scanning Kinesis devices…")
        th.start()
        self._sync_controls()

    def _on_scan_done(self, serials, err: str) -> None:
        self._stage_scan_busy = False
        self._sync_controls()
        if err:
            QtWidgets.QMessageBox.warning(self, "Scan", f"Scan failed: {err}")
            return
        if not serials:
            QtWidgets.QMessageBox.information(self, "Scan", "No Kinesis devices found.")
            return
        for combo in self.serialCombo.values():
            current = combo.currentText()
            combo.clear()
            combo.addItems(serials)
            combo.setCurrentText(current if current in serials else combo.currentText())
        self._status.showMessage(f"Found {len(serials)} Kinesis device(s)")

    def _connect_stage(self, name: str) -> None:
        serial = self.serialCombo[name].currentText().strip()
        if serial:
            self._devs[name].connect_device(serial=serial)

    def _on_stage_state(self, name: str) -> None:
        state = self._devs[name].state
        if state != "connected":
            self.devStatusLbl[name].setText(state.capitalize())
            self._pos[name] = None
            self.posLbl[name].setText("---")
        self._sync_controls()

    def _on_stage_connected(self, name: str, info) -> None:
        self.devStatusLbl[name].setText(f"OK ({info.get('serial', '')})")

    def _on_stage_readback(self, name: str, d: dict) -> None:
        if "error" in d:
            self.posLbl[name].setText("err")
            return
        pos = d.get("pos")
        self._pos[name] = pos
        moving = " (moving)" if d.get("moving") else ""
        self.posLbl[name].setText("err" if pos is None else f"{pos:7.2f}°{moving}")

    def _move(self, name: str) -> None:
        dev = self._devs[name]
        if not dev.is_connected or name in self._move_threads:
            return
        self._move_to(name, self.targetSpin[name].value())

    def _move_to(self, name: str, target: float) -> None:
        """Move one stage in a worker thread (the position poll keeps running: RotationStage locks per call)."""
        stage = self._devs[name].stage
        ramp = self.rampSpin.value()
        ctrl = MotionController()
        th = self._keep(CallThread(lambda: stage.move_to(target, step_deg=ramp, controller=ctrl), self))
        th.done.connect(lambda _res, err, n=name: self._on_move_done(n, err))
        self._move_threads[name] = th
        self._move_ctrl[name] = ctrl
        self._status.showMessage(f"{STAGE_LABEL[name]}: moving to {target:.2f}°…")
        th.start()
        self._sync_controls()

    def _on_move_done(self, name: str, err: str) -> None:
        self._move_threads.pop(name, None)
        self._move_ctrl.pop(name, None)
        if err:
            QtWidgets.QMessageBox.critical(self, f"{STAGE_LABEL[name]} move failed", err)
        else:
            self._status.showMessage(f"{STAGE_LABEL[name]}: move done")
        self._sync_controls()

    # ================================================================== power meter
    def _scan_pm(self) -> None:
        th = self._keep(CallThread(PM100A.list_resources, self))
        th.done.connect(self._on_pm_scan_done)
        self._pm_scan_busy = True
        th.start()
        self._sync_controls()

    def _on_pm_scan_done(self, resources, err: str) -> None:
        self._pm_scan_busy = False
        self._sync_controls()
        if err or not resources:
            QtWidgets.QMessageBox.information(self, "PM scan", err or "No power meters found.")
            return
        if len(resources) == 1:
            self.pmResourceEdit.setText(resources[0])
            return
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Select power meter", "Resource:", resources, 0, False)
        if ok and choice:
            self.pmResourceEdit.setText(choice)

    def _connect_pm(self) -> None:
        resource = self.pmResourceEdit.text().strip()
        if not resource or self._pm_thread is not None:
            return
        self._pm_error = ""
        th = PMThread(resource, self)
        th.result.connect(self._on_pm_result)
        th.finished.connect(self._on_pm_finished)
        self._pm_thread = th
        self.pmStatusLbl.setText("Connecting…")
        th.start()
        self._sync_controls()

    def _disconnect_pm(self) -> None:
        if self._pm_thread is not None:
            self.pmStatusLbl.setText("Disconnecting…")
            self._pm_thread.stop()
            self._sync_controls()

    def _close_pm(self, then) -> None:
        """Close the PM session (the sweep worker opens its own), then call ``then``."""
        if self._pm_thread is None:
            then()
            return
        self._after_pm_close = then
        self._disconnect_pm()

    def _pm_submit(self, cmd: str, arg=None) -> None:
        if self._pm_thread is not None and self._pm_ok:
            self._pm_thread.submit(cmd, arg)

    def _pm_zero(self) -> None:
        if not self._pm_ok or self._zeroing:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Zero power meter", "Block all light to the sensor, then press OK to zero.",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if reply != QtWidgets.QMessageBox.Ok:
            return
        self._zeroing = True
        self.pmZeroLbl.setText("zeroing…")
        self.stateLbl.setText("Zeroing power meter…")
        self._pm_submit("zero")
        self._sync_controls()

    def _on_pm_result(self, cmd: str, val, err: str) -> None:
        if cmd == "open":
            if err:
                self._pm_error = err
                QtWidgets.QMessageBox.critical(self, "PM connect failed", err)
            else:
                self._pm_ok = True
                self.pmStatusLbl.setText("OK")
        elif cmd == "read":
            self.pmReadLbl.setText("err" if err else fmt_power(val))
        elif cmd == "zero":
            self._zeroing = False
            if err:
                self.pmZeroLbl.setText("zero failed")
                QtWidgets.QMessageBox.critical(self, "PM zero", err)
            else:
                self.pmZeroLbl.setText(f"zeroed {datetime.now():%H:%M:%S}")
                self._status.showMessage("Power meter zeroed — unblock the beam.")
            self.stateLbl.setText("Idle")
        elif cmd in ("wavelength", "averaging"):
            self._status.showMessage(f"PM {cmd}: {err}" if err else f"PM {cmd} set to {val}")
        self._sync_controls()

    def _on_pm_finished(self) -> None:
        self._pm_thread = None
        self._pm_ok = False
        self._zeroing = False
        self.pmStatusLbl.setText("Error" if self._pm_error else "Disconnected")
        cb, self._after_pm_close = self._after_pm_close, None
        self._sync_controls()
        if cb is not None:
            cb()

    # ================================================================== mode
    @property
    def _nd_only(self) -> bool:
        return self.modeCombo.currentIndex() == 1

    def _apply_mode(self) -> None:
        """Show / hide the HWP parts; ND only = ND sweep, ND tab and .xlsx save."""
        nd_only = self._nd_only
        for w in self._hwp_widgets:
            w.setVisible(not nd_only)
        for b in (self.btn_run_hwp, self.btn_run_both):
            b.setVisible(not nd_only)
        self.sweepTabs.setTabVisible(0, not nd_only)
        for i in (TAB_HWP, TAB_ANALYSIS, TAB_PLANNER):
            self.tabs.setTabVisible(i, not nd_only)
        if nd_only:
            self.sweepTabs.setCurrentIndex(1)
            self.tabs.setCurrentIndex(TAB_ND)
        self.btn_load.setText("Load XLSX" if nd_only else "Load CSV")
        self.btn_save.setText("Save XLSX" if nd_only else "Save CSV")
        self._sync_controls()

    def _set_mode(self, nd_only: bool) -> None:
        if self._nd_only != nd_only:
            self.modeCombo.setCurrentIndex(1 if nd_only else 0)  # -> _apply_mode

    # ================================================================== state
    def _sync_controls(self) -> None:
        busy = self._running or self._zeroing
        moving = bool(self._move_threads)
        for name, dev in self._devs.items():
            st = dev.state
            self.btn_conn[name].setEnabled(st == "disconnected" and not busy)
            self.btn_disc[name].setEnabled(st == "connected" and not busy and name not in self._move_threads)
            self.btn_scan[name].setEnabled(not busy and not self._stage_scan_busy)
            self.serialCombo[name].setEnabled(st == "disconnected" and not busy)
            self.btn_move[name].setEnabled(st == "connected" and not busy and name not in self._move_threads)

        pm_idle = self._pm_thread is None
        pm_cmd = self._pm_ok and not busy
        self.btn_pm_conn.setEnabled(pm_idle and not busy)
        self.btn_pm_disc.setEnabled(self._pm_ok and not busy)
        self.btn_pm_scan.setEnabled(pm_idle and not busy and not self._pm_scan_busy)
        self.pmResourceEdit.setEnabled(pm_idle and not busy)
        for b in (self.btn_pm_wave, self.btn_pm_avg, self.btn_pm_read, self.btn_pm_zero):
            b.setEnabled(pm_cmd)

        nd_only = self._nd_only
        needed = ("nd",) if nd_only else STAGES
        ready = all(self._devs[n].is_connected for n in needed) and self._pm_ok
        for b in (self.btn_run_hwp, self.btn_run_nd, self.btn_run_both):
            b.setEnabled(ready and not busy and not moving)
        self.btn_stop.setEnabled((self._running and not self._stopping) or moving)

        has_data = bool(self._nd_angles) if nd_only else bool(self._hwp_angles or self._nd_angles)
        self.btn_save.setEnabled(has_data and not self._running)
        self.btn_load.setEnabled(not self._running)
        self.modeCombo.setEnabled(not self._running)
        for spin in (self.ndFixedSpin, self.hwpStartSpin, self.hwpStopSpin, self.hwpStepSpin,
                     self.hwpFixedSpin, self.ndStartSpin, self.ndStopSpin, self.ndStepSpin,
                     self.nReadingsSpin, self.rampSpin, self.pmWaveSpin, self.pmAvgSpin):
            spin.setEnabled(not busy)

        has_grid = self._grid_data is not None
        self.btn_solve.setEnabled(has_grid)
        self.btn_export.setEnabled(bool(self._plan_rows))

    # ================================================================== sweeps
    @staticmethod
    def _angles(start: float, stop: float, step: float) -> np.ndarray:
        if step <= 0 or stop <= start:
            return np.array([])
        return np.arange(start, stop + step * 0.5, step)

    def _hwp_sweep_angles(self) -> np.ndarray:
        return self._angles(self.hwpStartSpin.value(), self.hwpStopSpin.value(), self.hwpStepSpin.value())

    def _nd_sweep_angles(self) -> np.ndarray:
        return self._angles(self.ndStartSpin.value(), self.ndStopSpin.value(), self.ndStepSpin.value())

    def _update_totals(self) -> None:
        self.hwpTotalLbl.setText(f"{len(self._hwp_sweep_angles())} steps")
        self.ndTotalLbl.setText(f"{len(self._nd_sweep_angles())} steps")

    def _on_start_hwp(self) -> None:
        angles = self._hwp_sweep_angles()
        if len(angles) == 0:
            QtWidgets.QMessageBox.warning(self, "Bad sweep", "No HWP steps — check start/stop/step.")
            self._queue_nd = False
            return
        self._hwp_angles, self._hwp_powers, self._hwp_stds = [], [], []
        self._nd_fixed = self.ndFixedSpin.value()
        self.hwpTable.setRowCount(0)
        self.hwpPlot.clear()
        self.tabs.setCurrentIndex(TAB_HWP)
        self._start_sweep("hwp", fixed="nd", sweep="hwp", fixed_angle=self._nd_fixed, angles=angles)

    def _on_start_nd(self) -> None:
        angles = self._nd_sweep_angles()
        if len(angles) == 0:
            QtWidgets.QMessageBox.warning(self, "Bad sweep", "No ND steps — check start/stop/step.")
            self._finish_sweep()
            return
        self._nd_angles, self._nd_powers, self._nd_stds = [], [], []
        self.ndTable.setRowCount(0)
        self.ndPlot.clear()
        self.tabs.setCurrentIndex(TAB_ND)
        if self._nd_only:  # no HWP: the "fixed" move just brings the ND wheel to the start angle
            self._hwp_fixed = None
            self._start_sweep("nd", fixed="nd", sweep="nd", fixed_angle=float(angles[0]), angles=angles)
            return
        self._hwp_fixed = self.hwpFixedSpin.value()
        self._start_sweep("nd", fixed="hwp", sweep="nd", fixed_angle=self._hwp_fixed, angles=angles)

    def _on_start_both(self) -> None:
        self._queue_nd = True
        self._on_start_hwp()

    def _start_sweep(self, sweep_type: str, *, fixed: str, sweep: str, fixed_angle: float, angles) -> None:
        self._running = True
        self._stopping = False
        self._sweep_type = sweep_type
        self.progressBar.setRange(0, len(angles))
        self.progressBar.setValue(0)
        self.stateLbl.setText(f"{STAGE_LABEL[sweep]} sweep: starting…")
        self._sync_controls()
        resource = self.pmResourceEdit.text().strip()
        kwargs = dict(
            fixed_angle=fixed_angle, sweep_angles=angles, n_readings=self.nReadingsSpin.value(),
            fixed_label=STAGE_LABEL[fixed], sweep_label=STAGE_LABEL[sweep],
            ramp_step_deg=self.rampSpin.value(), pm_averaging=self.pmAvgSpin.value())

        def launch():
            if self._stopping:  # Stop pressed while the PM session was closing
                self._finish_sweep()
                return
            worker = StageSweepWorker(self._devs[fixed].stage, self._devs[sweep].stage, resource,
                                      parent=self, **kwargs)
            worker.point_done.connect(self._on_point_done)
            worker.status.connect(self._status.showMessage)
            worker.done.connect(self._on_done)
            self._worker = self._keep(worker)
            worker.start()

        self._close_pm(launch)  # TLPM: the worker opens its own session in its thread

    def _on_stop(self) -> None:
        self._queue_nd = False
        for ctrl in self._move_ctrl.values():
            ctrl.abort()
        if self._running:
            self._stopping = True
            self.stateLbl.setText("Stopping…")
            worker = self._worker
            if worker is not None:
                # StageSweepWorker.stop() also sends stage.stop(): keep that off the GUI thread
                self._keep(CallThread(worker.stop, self)).start()
        self._sync_controls()

    def _on_point_done(self, step: int, total: int, angle: float, mean_p: float, std_p: float) -> None:
        self.progressBar.setValue(step + 1)
        self.stateLbl.setText(f"{STAGE_LABEL[self._sweep_type]} sweep: {step + 1}/{total}")
        if self._sweep_type == "hwp":
            self._hwp_angles.append(angle)
            self._hwp_powers.append(mean_p)
            self._hwp_stds.append(std_p)
            self._append_row(self.hwpTable, angle, mean_p, std_p)
            self.hwpPlot.set_data(self._hwp_angles, self._hwp_powers, self._hwp_stds)
        else:
            self._nd_angles.append(angle)
            self._nd_powers.append(mean_p)
            self._nd_stds.append(std_p)
            self._append_row(self.ndTable, angle, mean_p, std_p)
            self.ndPlot.set_data(self._nd_angles, self._nd_powers, self._nd_stds)

    def _on_done(self, result: str, _data) -> None:
        self._worker = None
        label = STAGE_LABEL[self._sweep_type]
        if result == "ok":
            self._calib_path = ""
            self._status.showMessage(f"{label} sweep done.")
            if self._sweep_type == "hwp" and self._queue_nd and not self._stopping:
                self._queue_nd = False
                QtCore.QTimer.singleShot(500, self._start_chained_nd)  # PM stays closed between the two
                return
        elif result == "aborted":
            self._status.showMessage(f"{label} sweep aborted.")
        else:
            QtWidgets.QMessageBox.critical(self, "Sweep error", result)
            self._status.showMessage(result)
        self._finish_sweep()

    def _start_chained_nd(self) -> None:
        if self._stopping:  # Stop pressed between the two sweeps
            self._finish_sweep()
        else:
            self._on_start_nd()

    def _finish_sweep(self) -> None:
        self._running = False
        self._stopping = False
        self._queue_nd = False
        self.stateLbl.setText("Idle")
        self._data_changed()
        self._connect_pm()
        self._sync_controls()

    # ================================================================== tables / analysis
    @staticmethod
    def _append_row(table: QtWidgets.QTableWidget, angle: float, power: float, std: float) -> None:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{angle:.2f}"))
        table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{fmt_power(power)} ± {fmt_power(std, 3)}"))
        table.scrollToBottom()

    def _fill_table(self, table: QtWidgets.QTableWidget, angles, powers, stds) -> None:
        table.setRowCount(0)
        table.setUpdatesEnabled(False)
        for a, p, s in zip(angles, powers, stds):
            self._append_row(table, a, p, s)
        table.setUpdatesEnabled(True)
        table.scrollToTop()

    def _data_changed(self) -> None:
        """New HWP and/or ND data: refresh the analysis combo and the planner."""
        idx = self.ndSelCombo.currentIndex()
        self.ndSelCombo.blockSignals(True)
        self.ndSelCombo.clear()
        self.ndSelCombo.addItems([f"{a:.2f}°" for a in self._nd_angles])
        if self._nd_angles:
            self.ndSelCombo.setCurrentIndex(idx if 0 <= idx < len(self._nd_angles)
                                            else int(np.nanargmax(self._nd_powers)))
        self.ndSelCombo.blockSignals(False)
        self._update_analysis()

        for spins, angles in (((self.hwpLoSpin, self.hwpHiSpin), self._hwp_angles),
                              ((self.ndLoSpin, self.ndHiSpin), self._nd_angles)):
            if angles:
                for s, v in zip(spins, (min(angles), max(angles))):
                    s.blockSignals(True)
                    s.setValue(v)
                    s.blockSignals(False)
        self._rebuild_grid()

    def _update_analysis(self) -> None:
        idx = self.ndSelCombo.currentIndex()
        hwp_p = np.asarray(self._hwp_powers, dtype=float)
        if idx < 0 or hwp_p.size == 0 or not np.any(hwp_p > 0):
            self.analysisMaxLbl.setText("---")
            self.analysisPlot.clear()
            return
        p_max = float(np.nanmax(hwp_p))
        p_sel = self._nd_powers[idx]
        scale = p_sel / p_max
        self.analysisMaxLbl.setText(fmt_power(p_sel))
        self.analysisPlot.set_title(f"Expected power  (ND = {self._nd_angles[idx]:.2f}°)")
        self.analysisPlot.set_data(self._hwp_angles, hwp_p * scale, np.asarray(self._hwp_stds) * scale)

    # ================================================================== planner
    def _rebuild_grid(self) -> None:
        self._grid_data = None
        if not (self._hwp_angles and self._nd_angles):
            self.coverageLbl.setText("Needs both the HWP and the ND sweep: load a calibration file or run both sweeps.")
            self.coveragePlot.clear()
            self.rangeModel.clear()
            self.allModel.clear()
            self._show_plan([])
            self._sync_controls()
            return
        try:
            grid = calib.build_grid(
                self._hwp_angles, self._hwp_powers, self._hwp_stds,
                self._nd_angles, self._nd_powers, self._nd_stds,
                hwp_lim=(self.hwpLoSpin.value(), self.hwpHiSpin.value()),
                nd_lim=(self.ndLoSpin.value(), self.ndHiSpin.value()))
        except ValueError as exc:
            self.coverageLbl.setText(f"No combinations: {exc}")
            self.coveragePlot.clear()
            self.rangeModel.clear()
            self.allModel.clear()
            self._show_plan([])
            self._sync_controls()
            return
        self._grid_data = grid
        ranges = calib.nd_ranges(grid)
        lo, hi, gaps = calib.coverage(ranges)
        if not gaps:
            gap_txt = "no gaps"
        else:  # the widest few (by ratio)
            widest = sorted(gaps, key=lambda g: g[1] / g[0], reverse=True)[:3]
            gap_txt = (f"{len(gaps)} gap{'s' if len(gaps) > 1 else ''}: "
                       + ", ".join(f"{fmt_power(a)} – {fmt_power(b)}" for a, b in sorted(widest))
                       + (", …" if len(gaps) > 3 else ""))
        self.coverageLbl.setText(
            f"Reachable: {fmt_power(lo)} – {fmt_power(hi)}  ({gap_txt})   ·   {grid.size} combinations "
            f"= {grid.nd_deg.size} ND × {grid.hwp_deg.size} HWP angles")
        self.coveragePlot.set_ranges(ranges["nd_deg"], ranges["p_min"], ranges["p_max"])
        self.rangeModel.set_columns([ranges["nd_deg"], ranges["p_min"], ranges["hwp_at_min"],
                                     ranges["p_max"], ranges["hwp_at_max"]])
        f = grid.flat()
        self.allModel.set_columns([f["power_w"], f["power_w"] - f["sigma_w"], f["power_w"] + f["sigma_w"],
                                   f["nd_deg"], f["hwp_deg"], f["sens"]])
        self._resolve()
        self._sync_controls()

    def _on_generate(self) -> None:
        vals = calib.spaced(self.genStartSpin.value(), self.genStopSpin.value(), self.genNSpin.value(),
                            self.genLogChk.isChecked())
        self.targetEdit.setPlainText(", ".join(f"{v:.4g}" for v in vals))

    def _on_solve(self) -> None:
        vals, bad = calib.parse_power_list(self.targetEdit.toPlainText(), self.targetUnitCombo.currentText())
        if bad:
            self._status.showMessage("Ignored: " + " ".join(bad), 8000)
        if not vals:
            QtWidgets.QMessageBox.information(self, "Power planner", "No target powers in the list.")
            return
        self._plan_targets = vals
        self._resolve()

    def _resolve(self) -> None:
        """(Re)solve the current target list on the current grid / tolerance."""
        if self._grid_data is None or not self._plan_targets:
            self._show_plan([])
            return
        rows = calib.solve(self._grid_data, self._plan_targets, self.tolSpin.value() / 100.0,
                           self.angErrSpin.value())
        self._show_plan(rows)

    def _show_plan(self, rows: List[calib.PlanRow]) -> None:
        self._plan_rows = rows
        bad_bg = QtGui.QBrush(QtGui.QColor(BAD_ROW_BG))
        self.planModel.set_columns(
            [[r.target_w for r in rows], [r.power_w for r in rows], [r.err for r in rows],
             [r.sigma_w for r in rows], [r.expected_err for r in rows], [r.nd_deg for r in rows],
             [r.hwp_deg for r in rows], [r.sens for r in rows], [r.note for r in rows]],
            row_bg=[None if r.ok else bad_bg for r in rows])
        self.coveragePlot.set_points([r.nd_deg for r in rows], [r.power_w for r in rows], [r.ok for r in rows])
        if rows:
            n_bad = sum(not r.ok for r in rows)
            worst = max(abs(r.err) for r in rows if r.ok) if n_bad < len(rows) else float("nan")
            txt = f"{len(rows)} targets, {len(rows) - n_bad} within ±{self.tolSpin.value():g} %"
            if n_bad < len(rows):
                txt += f" (worst {worst * 100:.2f} %)"
            if n_bad:
                txt += f";  {n_bad} not reachable within tolerance (red)"
            self.planSummaryLbl.setText(txt)
        else:
            self.planSummaryLbl.setText("")
        self.btn_export.setEnabled(bool(rows))

    def _on_export(self) -> None:
        rows = list(self._plan_rows)
        if not rows:
            return
        n_bad = sum(not r.ok for r in rows)
        if n_bad:
            ans = QtWidgets.QMessageBox.question(
                self, "Export",
                f"{n_bad} target(s) are not reachable within the tolerance.\n\n"
                "Yes: export them anyway (at the nearest reachable power).\nNo: leave them out.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.No)
            if ans == QtWidgets.QMessageBox.Cancel:
                return
            if ans == QtWidgets.QMessageBox.No:
                rows = [r for r in rows if r.ok]
                if not rows:
                    return
        folder = os.path.dirname(self._calib_path) if self._calib_path else DATA_DIR
        default = os.path.join(folder, f"power_plan_{datetime.now():%Y%m%d_%H%M%S}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export power plan (dual-wheel CSV)", default,
                                                        "CSV files (*.csv)")
        if not path:
            return
        try:
            unit = calib.write_dual_wheel_csv(path, rows, self._calib_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._status.showMessage(
            f"Exported {len(rows)} combinations ({unit}) → {path}   "
            "(PL+TRPL: load as calibration; stage A = HWP, stage B = ND)")

    # ================================================================== save / load
    def _on_save(self) -> None:
        if self._nd_only:
            self._save_nd_xlsx()
            return
        default = os.path.join(DATA_DIR, f"wheel_hwp_calib_{datetime.now():%Y%m%d_%H%M%S}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration CSV", default, "CSV files (*.csv)")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = [
            f"Wheel+HWP power calibration  {datetime.now().isoformat()}",
            f"ND serial:      {self.serialCombo['nd'].currentText()}",
            f"HWP serial:     {self.serialCombo['hwp'].currentText()}",
            f"PM averaging:   {self.pmAvgSpin.value()}",
            f"Readings/step:  {self.nReadingsSpin.value()}",
            f"Wavelength (nm):{self.pmWaveSpin.value()}",
            f"PM zero:        {self.pmZeroLbl.text()}",
        ]
        try:
            calib.save_calibration(path, header, (self._hwp_angles, self._hwp_powers, self._hwp_stds),
                                   (self._nd_angles, self._nd_powers, self._nd_stds),
                                   nd_fixed=self._nd_fixed if self._hwp_angles else None,
                                   hwp_fixed=self._hwp_fixed if self._nd_angles else None)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._calib_path = path
        self._status.showMessage(f"Saved → {path}")

    def _save_nd_xlsx(self) -> None:
        folder = os.path.dirname(self._calib_path) if self._calib_path else DATA_DIR
        default = os.path.join(folder, f"power{datetime.now():%m%d}.xlsx")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ND wheel calibration", default,
                                                        "Excel files (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = [
            f"ND wheel power calibration (no HWP)  {datetime.now().isoformat()}",
            f"ND serial:      {self.serialCombo['nd'].currentText()}",
            f"PM averaging:   {self.pmAvgSpin.value()}",
            f"Readings/step:  {self.nReadingsSpin.value()}",
            f"Wavelength (nm):{self.pmWaveSpin.value()}",
            f"PM zero:        {self.pmZeroLbl.text()}",
        ]
        try:
            n = calib.save_nd_xlsx(path, header, (self._nd_angles, self._nd_powers, self._nd_stds))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._calib_path = path
        skipped = len(self._nd_angles) - n
        self._status.showMessage(f"Saved {n} angles → {path}"
                                 + (f"   ({skipped} without a valid power left out)" if skipped else ""))

    def _on_load(self) -> None:
        folder = os.path.dirname(self._calib_path) if self._calib_path else DATA_DIR
        filters = ["CSV files (*.csv)", "Excel files (*.xlsx)"]
        if self._nd_only:
            filters.reverse()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load calibration", folder, ";;".join(filters))
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> bool:
        if path.lower().endswith((".xlsx", ".xlsm")):
            return self._load_nd_xlsx(path)
        try:
            data = load_hwp_nd_calibration(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return False
        nd_fixed, hwp_fixed = calib.read_fixed_angles(path)
        self._hwp_angles, self._hwp_powers, self._hwp_stds = (
            list(data.hwp_angles), list(data.hwp_powers_w), list(data.hwp_stds_w))
        self._nd_fixed = nd_fixed
        self._fill_table(self.hwpTable, self._hwp_angles, self._hwp_powers, self._hwp_stds)
        self.hwpPlot.set_data(self._hwp_angles, self._hwp_powers, self._hwp_stds)
        if data.nd_angles:  # as v1: a file without an ND section keeps the current ND data
            self._nd_angles, self._nd_powers, self._nd_stds = (
                list(data.nd_angles), list(data.nd_powers_w), list(data.nd_stds_w))
            self._hwp_fixed = hwp_fixed
            self._fill_table(self.ndTable, self._nd_angles, self._nd_powers, self._nd_stds)
            self.ndPlot.set_data(self._nd_angles, self._nd_powers, self._nd_stds)
        self._calib_path = path
        self._set_mode(False)
        self._data_changed()
        if self._grid_data is not None:
            self.tabs.setCurrentIndex(TAB_PLANNER)
        self._status.showMessage(f"Loaded ← {os.path.basename(path)}")
        self._sync_controls()
        return True

    def _load_nd_xlsx(self, path: str) -> bool:
        """ND-wheel-only calibration (angle | power): switches to ND-only mode."""
        try:
            angles, powers, stds = calib.load_nd_xlsx(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return False
        self._nd_angles, self._nd_powers, self._nd_stds = angles, powers, stds
        self._hwp_fixed = None
        self._fill_table(self.ndTable, angles, powers, stds)
        self.ndPlot.set_data(angles, powers, stds)
        self._calib_path = path
        self._set_mode(True)
        self._data_changed()
        self.tabs.setCurrentIndex(TAB_ND)
        self._status.showMessage(f"Loaded ← {os.path.basename(path)}")
        self._sync_controls()
        return True

    # ================================================================== shutdown
    def shutdown(self) -> None:
        """Window close: stop the sweep and moves, close the power meter and the stages."""
        self._queue_nd = False
        self._after_pm_close = None
        if self._worker is not None:
            self._worker.stop()
        for ctrl in self._move_ctrl.values():
            ctrl.abort()
        wait_threads(list(self._threads), 10.0)
        if self._pm_thread is not None:
            self._pm_thread.stop()
            wait_threads([self._pm_thread], 5.0)
        for dev in self._devs.values():
            dev.shutdown()
