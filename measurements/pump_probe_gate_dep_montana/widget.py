"""Main widget of the Montana pump-probe gate / power dependence GUI.

Layout (CLAUDE.md v2): top bar (connect / live / scans / stop + state) -> settings tabs
(Hardware, Delay scan, Power, Gates, Analysis & save) -> live display tabs (Main,
Monitoring, Gaussian fit, Gate V/I, Sweep log) -> status bar.

Every device call runs in a worker thread (device controllers in ``devices.py``,
scan threads in ``workers.py``); the widget only reacts to signals.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from keithley.keithley_wrapper import SweepController
from measurements.power_calibration import load_power_calibration
from measurements.pump_probe_gate_dep_montana import config as C
from measurements.pump_probe_gate_dep_montana import ta_core
from measurements.pump_probe_gate_dep_montana.devices import (
    CONNECTED,
    DISCONNECTED,
    CallThread,
    CameraDevice,
    DelayStageDevice,
    GateDevice,
    GateOpener,
    SimulatedCamera,
    StageDevice,
    make_rotation_stage,
    wait_threads,
)
from measurements.pump_probe_gate_dep_montana.plots import GateVIPanel, GaussianPanel, MonitorPanel, TAImagePanel
from measurements.pump_probe_gate_dep_montana.sweep_progress import STATE_PREFIX, SweepProgress
from measurements.pump_probe_gate_dep_montana.workers import (
    DelayScanThread,
    GatePowerSweepThread,
    GateRampThread,
    LiveCaptureThread,
    append_scan_log,
)
from smc100.smc100_wrapper import list_serial_ports

GATES = ("front", "back")


def _grid_box(title: str) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
    box = QtWidgets.QGroupBox(title)
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return box, g


def _dspin(lo, hi, val, dec=3, step=None, width=None, suffix="") -> QtWidgets.QDoubleSpinBox:
    sp = QtWidgets.QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setValue(val)
    if step is not None:
        sp.setSingleStep(step)
    if width:
        sp.setFixedWidth(width)
    if suffix:
        sp.setSuffix(suffix)
    return sp


def _ispin(lo, hi, val, width=None) -> QtWidgets.QSpinBox:
    sp = QtWidgets.QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(val)
    if width:
        sp.setFixedWidth(width)
    return sp


def _btn(text, slot=None, width=C.BTN_W) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if width:
        b.setFixedWidth(width)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def _page(*boxes, stretch=True) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(4, 4, 4, 4)
    for b in boxes:
        h.addWidget(b)
    if stretch:
        h.addStretch(1)
    return w


class PumpProbeMontanaWidget(QtWidgets.QWidget):
    error_reported = QtCore.pyqtSignal(str)  # uncaught exception text (any thread; app.py excepthook)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ---- devices
        self.cameraDev = CameraDevice(self)
        self.delayDev = DelayStageDevice(C.SMC_POLL_S, self)
        self.wheelDev = StageDevice("ND wheel", make_rotation_stage, C.STAGE_POLL_S, self)
        self.polDev = StageDevice("Pol. stage", make_rotation_stage, C.STAGE_POLL_S, self)
        opener = GateOpener()
        self.gateDevs: Dict[str, GateDevice] = {
            w: GateDevice(f"{w.capitalize()} gate", opener, C.GATE_READ_TIMEOUT_MS, C.GATE_POLL_ERROR_LIMIT,
                          C.GATE_POLL_S, C.GATE_RECONNECT_S, C.RAMP_STEP_V, C.RAMP_DWELL_S, self)
            for w in GATES}

        # ---- state
        self.t0_mm = 0.0
        self.delay_avg_map: Dict[float, int] = {}
        self._stage_state: Dict = {}
        self._wheel_pos: Optional[float] = None
        self._pol_pos: Optional[float] = None
        self._gate_vi: Dict[str, Tuple[float, float]] = {}
        self._gate_ramps: Dict[str, Tuple[GateRampThread, SweepController]] = {}
        self._live_thread: Optional[LiveCaptureThread] = None
        self._scan_thread: Optional[DelayScanThread] = None
        self._sweep_thread: Optional[GatePowerSweepThread] = None
        self._aux_threads = set()
        self._last_display = 0.0
        self._current_results = None
        self._calib = None
        self._calib_path = ""
        self._save_dir = ""
        self._session_path = ""
        self._sweep_t0 = 0.0
        self._sweep_done0: Optional[int] = None
        self._progress: Optional[SweepProgress] = None  # last sweep started / resumed (state file)
        self._zombies: List[Tuple[str, QtCore.QThread]] = []  # threads abandoned by Terminate
        self._stall_key = None
        self._stalled = False

        self._build_ui()
        self.error_reported.connect(self._on_error_reported)
        self._watchdog = QtCore.QTimer(self)
        self._watchdog.timeout.connect(self._on_watchdog)
        self._watchdog.start(C.WATCHDOG_TICK_MS)
        self._wire_devices()
        self._refresh_ports()
        self._update_delay_count()
        self._update_power_preview()
        self._update_gate_preview()
        self._apply_display_settings()
        self._update_buttons()

    # =================================================================== UI
    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ---- 1. top bar
        top = QtWidgets.QHBoxLayout()
        self.connectBtn = _btn("Connect", self.on_connect_main)
        self.connectBtn.setToolTip("Connect the camera and the delay stage")
        self.disconnectBtn = _btn("Disconnect", self.on_disconnect_main)
        self.liveBtn = _btn("Live", self.on_live)
        self.scanBtn = _btn("Delay scan", self.on_delay_scan)
        self.sweepBtn = _btn("Start sweep", self.on_sweep)
        self.sweepBtn.setStyleSheet("font-weight: bold;")
        self.stopBtn = _btn("Stop", self.on_stop)
        self.terminateBtn = _btn("Terminate", self.on_terminate)
        self.terminateBtn.setStyleSheet("color: #c62828; font-weight: bold;")
        self.terminateBtn.setToolTip(
            "Hardware stuck (delay stage / camera not responding, Stop does nothing)?\n"
            "Terminate abandons the running live view / scan / sweep at once and frees the GUI,\n"
            "so you can disconnect, reconnect and test the hardware.\n"
            "Sweep: the scan in progress (or, between scans, the last finished one) is treated as bad\n"
            "and measured again by 'Resume sweep'.")
        self.resumeBtn = _btn("Resume sweep", self.on_resume)
        self.resumeBtn.setToolTip(
            "Continue an unfinished sweep (terminated, stopped, error, or the GUI crashed):\n"
            "same plan, folder and file names; the scans already done are skipped.\n"
            "Without a sweep in memory (after a restart) choose its sweep_state_*.json file.")
        for b in (self.connectBtn, self.disconnectBtn, self.liveBtn, self.scanBtn, self.sweepBtn, self.stopBtn,
                  self.terminateBtn, self.resumeBtn):
            top.addWidget(b)
        top.addStretch(1)
        self.progressLbl = QtWidgets.QLabel("")
        top.addWidget(self.progressLbl)
        self.stateLbl = QtWidgets.QLabel("Disconnected")
        self.stateLbl.setMinimumWidth(C.STATUS_W)
        self.stateLbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.stateLbl)
        outer.addLayout(top)

        # ---- 2. settings tabs
        self.settingsTabs = QtWidgets.QTabWidget()
        self.settingsTabs.setMaximumHeight(C.SETTINGS_MAX_H)
        self.hwPage = self._build_hardware_page()
        self.delayPage = self._build_delay_page()
        self.powerPage = self._build_power_page()
        self.gatePage = self._build_gate_page()
        self.analysisPage = self._build_analysis_page()
        for page, name in ((self.hwPage, "Hardware"), (self.delayPage, "Delay scan"), (self.powerPage, "Power"),
                           (self.gatePage, "Gates"), (self.analysisPage, "Analysis && save")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(page)
            self.settingsTabs.addTab(scroll, name)
        outer.addWidget(self.settingsTabs)

        # ---- 3. live display
        self.displayTabs = QtWidgets.QTabWidget()
        self.imagePanel = TAImagePanel()
        self.monitorPanel = MonitorPanel()
        self.gaussPanel = GaussianPanel()
        self.gateVIPanel = GateVIPanel()
        self.logEdit = QtWidgets.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        self.displayTabs.addTab(self.imagePanel, "Main")
        self.displayTabs.addTab(self.monitorPanel, "Monitoring")
        self.displayTabs.addTab(self.gaussPanel, "Gaussian fit")
        self.displayTabs.addTab(self.gateVIPanel, "Gate V/I")
        self.displayTabs.addTab(self.logEdit, "Sweep log")
        outer.addWidget(self.displayTabs, 1)
        self.imagePanel.crosshair_moved.connect(lambda *_: self._update_gaussian())

        # ---- 4. status bar
        self._status = QtWidgets.QStatusBar()
        self._status.setSizeGripEnabled(False)
        outer.addWidget(self._status)

    # ------------------------------------------------------------------- hardware page
    def _build_hardware_page(self) -> QtWidgets.QWidget:
        # camera
        cam, g = _grid_box("Camera (PixeLink)")
        self.camConnectBtn = _btn("Connect", self._connect_camera, C.SMALL_BTN_W + 10)
        self.camDisconnectBtn = _btn("Disconnect", self.cameraDev.disconnect_device, C.SMALL_BTN_W + 10)
        self.camStatusLbl = QtWidgets.QLabel("Not connected")
        self.simulateChk = QtWidgets.QCheckBox("Simulate camera + delay stage")
        self.simulateChk.setToolTip("No hardware: a simulated PixeLink camera and SMC100 (for testing the GUI)")
        g.addWidget(self.camConnectBtn, 0, 0)
        g.addWidget(self.camDisconnectBtn, 0, 1)
        g.addWidget(self.camStatusLbl, 1, 0, 1, 2)
        g.addWidget(QtWidgets.QLabel("Settings from PixeLink Capture\n(trigger, exposure, ROI unchanged)"), 2, 0, 1, 2)
        g.addWidget(self.simulateChk, 3, 0, 1, 2)
        g.setRowStretch(4, 1)

        # delay stage
        st, g = _grid_box("Delay stage (Newport SMC100)")
        self.portCombo = QtWidgets.QComboBox()
        self.portCombo.setEditable(True)
        self.portCombo.setMinimumWidth(170)
        self.portRefreshBtn = _btn("Refresh", self._refresh_ports, C.SMALL_BTN_W)
        self.addrSpin = _ispin(1, 31, C.DEFAULT_SMC_ADDRESS, 50)
        self.stageConnectBtn = _btn("Connect", self._connect_stage, C.SMALL_BTN_W + 10)
        self.stageDisconnectBtn = _btn("Disconnect", self.delayDev.disconnect_device, C.SMALL_BTN_W + 10)
        self.stageHomeBtn = _btn("Home", self._home_stage, C.SMALL_BTN_W)
        self.stageStopBtn = _btn("Stop", self._stop_stage, C.SMALL_BTN_W)
        self.stageStateLbl = QtWidgets.QLabel("Not connected")
        self.stagePosLbl = QtWidgets.QLabel("Position: -- mm")
        self.gotoSpin = _dspin(-100000, 100000, 0.0, 2, 1.0, 100, " ps")
        self.gotoBtn = _btn("Go to delay", self._goto_delay, C.SMALL_BTN_W + 20)
        self.t0Spin = _dspin(-1000, 1000, 0.0, 4, 0.01, 100, " mm")
        self.t0Spin.setToolTip("Stage position of time zero (delays are relative to it)")
        self.t0Spin.valueChanged.connect(self._on_t0_changed)
        self.t0FromPosBtn = _btn("t0 = current", self._set_t0_from_stage, C.SMALL_BTN_W + 20)
        self.stageLimitsLbl = QtWidgets.QLabel("")
        g.addWidget(QtWidgets.QLabel("Port:"), 0, 0)
        g.addWidget(self.portCombo, 0, 1, 1, 2)
        g.addWidget(self.portRefreshBtn, 0, 3)
        g.addWidget(QtWidgets.QLabel("Addr:"), 0, 4)
        g.addWidget(self.addrSpin, 0, 5)
        g.addWidget(self.stageConnectBtn, 1, 0, 1, 2)
        g.addWidget(self.stageDisconnectBtn, 1, 2)
        g.addWidget(self.stageHomeBtn, 1, 3)
        g.addWidget(self.stageStopBtn, 1, 4, 1, 2)
        g.addWidget(self.stageStateLbl, 2, 0, 1, 6)
        g.addWidget(self.stagePosLbl, 3, 0, 1, 6)
        g.addWidget(QtWidgets.QLabel("Delay:"), 4, 0)
        g.addWidget(self.gotoSpin, 4, 1, 1, 2)
        g.addWidget(self.gotoBtn, 4, 3, 1, 2)
        g.addWidget(QtWidgets.QLabel("t0:"), 5, 0)
        g.addWidget(self.t0Spin, 5, 1, 1, 2)
        g.addWidget(self.t0FromPosBtn, 5, 3, 1, 2)
        g.addWidget(self.stageLimitsLbl, 6, 0, 1, 6)

        # ND wheel
        wh, g = _grid_box("ND wheel (power)")
        self.wheelSerialCombo = QtWidgets.QComboBox()
        self.wheelSerialCombo.setEditable(True)
        self.wheelSerialCombo.addItem(C.DEFAULT_WHEEL_SERIAL)
        self.wheelSerialCombo.setMinimumWidth(110)
        self.detectBtn = _btn("Detect", self._detect_kinesis, C.SMALL_BTN_W)
        self.wheelConnectBtn = _btn("Connect", lambda: self._connect_rot(self.wheelDev, self.wheelSerialCombo),
                                    C.SMALL_BTN_W + 10)
        self.wheelDisconnectBtn = _btn("Disconnect", self.wheelDev.disconnect_device, C.SMALL_BTN_W + 10)
        self.wheelHomeBtn = _btn("Home", lambda: self._home_rot(self.wheelDev), C.SMALL_BTN_W)
        self.wheelPosLbl = QtWidgets.QLabel("Angle: --")
        self.wheelGotoSpin = _dspin(0, 360, 0.0, 3, 1.0, 90, " °")
        self.wheelGotoBtn = _btn("Move", lambda: self._move_rot(self.wheelDev, self.wheelGotoSpin.value()),
                                 C.SMALL_BTN_W)
        g.addWidget(QtWidgets.QLabel("Serial:"), 0, 0)
        g.addWidget(self.wheelSerialCombo, 0, 1)
        g.addWidget(self.detectBtn, 0, 2)
        g.addWidget(self.wheelConnectBtn, 1, 0)
        g.addWidget(self.wheelDisconnectBtn, 1, 1)
        g.addWidget(self.wheelHomeBtn, 1, 2)
        g.addWidget(self.wheelPosLbl, 2, 0, 1, 3)
        g.addWidget(self.wheelGotoSpin, 3, 0, 1, 2)
        g.addWidget(self.wheelGotoBtn, 3, 2)
        g.setRowStretch(4, 1)

        # polarization
        po, g = _grid_box("Polarization stage (LH / RH)")
        self.polEnableChk = QtWidgets.QCheckBox("Enable: every step taken at LH and RH")
        self.polSerialCombo = QtWidgets.QComboBox()
        self.polSerialCombo.setEditable(True)
        self.polSerialCombo.addItem(C.DEFAULT_POL_SERIAL)
        self.polConnectBtn = _btn("Connect", lambda: self._connect_rot(self.polDev, self.polSerialCombo),
                                  C.SMALL_BTN_W + 10)
        self.polDisconnectBtn = _btn("Disconnect", self.polDev.disconnect_device, C.SMALL_BTN_W + 10)
        self.polHomeBtn = _btn("Home", lambda: self._home_rot(self.polDev), C.SMALL_BTN_W)
        self.polPosLbl = QtWidgets.QLabel("Angle: --")
        self.lhSpin = _dspin(0, 360, C.DEFAULT_LH_ANGLE, 2, 1.0, 80, " °")
        self.rhLbl = QtWidgets.QLabel("")
        self.lhSpin.valueChanged.connect(self._update_rh_label)
        self.polSettleSpin = _dspin(0, 60, C.DEFAULT_POL_SETTLE_S, 2, 0.1, 80, " s")
        g.addWidget(self.polEnableChk, 0, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("Serial:"), 1, 0)
        g.addWidget(self.polSerialCombo, 1, 1, 1, 2)
        g.addWidget(self.polConnectBtn, 2, 0)
        g.addWidget(self.polDisconnectBtn, 2, 1)
        g.addWidget(self.polHomeBtn, 2, 2)
        g.addWidget(self.polPosLbl, 3, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("LH:"), 4, 0)
        g.addWidget(self.lhSpin, 4, 1)
        g.addWidget(self.rhLbl, 4, 2)
        g.addWidget(QtWidgets.QLabel("Settle:"), 5, 0)
        g.addWidget(self.polSettleSpin, 5, 1)
        self._update_rh_label()
        return _page(cam, st, wh, po)

    # ------------------------------------------------------------------- delay page
    def _build_delay_page(self) -> QtWidgets.QWidget:
        acq, g = _grid_box("Acquisition")
        self.framesSpin = _ispin(2, 100000, C.DEFAULT_N_FRAMES, 90)
        self.avgSpin = _ispin(1, 10000, C.DEFAULT_AVG_TIMES, 90)
        g.addWidget(QtWidgets.QLabel("Frames / capture:"), 0, 0)
        g.addWidget(self.framesSpin, 0, 1)
        g.addWidget(QtWidgets.QLabel("Default avg (captures):"), 1, 0)
        g.addWidget(self.avgSpin, 1, 1)
        g.addWidget(QtWidgets.QLabel("Multi-range rows set their\nown avg per delay."), 2, 0, 1, 2)
        g.setRowStretch(3, 1)

        gen, g = _grid_box("Generate delays")
        self.simpleRb = QtWidgets.QRadioButton("Simple")
        self.multiRb = QtWidgets.QRadioButton("Multi-range")
        self.simpleRb.setChecked(True)
        grp = QtWidgets.QButtonGroup(self)
        grp.addButton(self.simpleRb)
        grp.addButton(self.multiRb)
        self.genStack = QtWidgets.QStackedWidget()
        self.simpleRb.toggled.connect(lambda on: self.genStack.setCurrentIndex(0 if on else 1))

        sp = QtWidgets.QWidget()
        sg = QtWidgets.QGridLayout(sp)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setHorizontalSpacing(8)
        sg.setVerticalSpacing(6)
        s0, n0, i0 = C.DEFAULT_SIMPLE_SCAN
        self.simpleStartSpin = _dspin(-100000, 100000, s0, 2, 1.0, 90, " ps")
        self.simpleStepsSpin = _ispin(1, 100000, n0, 90)
        self.simpleIncSpin = _dspin(-100000, 100000, i0, 3, 0.5, 90, " ps")
        self.simplePreviewLbl = QtWidgets.QLabel("")
        for w in (self.simpleStartSpin, self.simpleStepsSpin, self.simpleIncSpin):
            w.valueChanged.connect(self._update_simple_preview)
        sg.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        sg.addWidget(self.simpleStartSpin, 0, 1)
        sg.addWidget(QtWidgets.QLabel("Steps:"), 0, 2)
        sg.addWidget(self.simpleStepsSpin, 0, 3)
        sg.addWidget(QtWidgets.QLabel("Increment:"), 1, 0)
        sg.addWidget(self.simpleIncSpin, 1, 1)
        sg.addWidget(self.simplePreviewLbl, 1, 2, 1, 2)
        sg.addWidget(_btn("Around t=0", self._scan_around_zero, 90), 2, 0, 1, 2)
        sg.addWidget(_btn("From t=0", self._scan_from_zero, 90), 2, 2, 1, 2)
        sg.setRowStretch(3, 1)
        self.genStack.addWidget(sp)

        mp = QtWidgets.QWidget()
        mg = QtWidgets.QGridLayout(mp)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setHorizontalSpacing(8)
        mg.setVerticalSpacing(6)
        self.rangeTable = QtWidgets.QTableWidget(0, 4)
        self.rangeTable.setHorizontalHeaderLabels(["Start (ps)", "End (ps)", "Step (ps)", "Avg"])
        self.rangeTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.rangeTable.verticalHeader().setDefaultSectionSize(22)
        for r in C.DEFAULT_MULTI_RANGES:
            self._add_range_row(*r)
        mg.addWidget(self.rangeTable, 0, 0, 4, 1)
        mg.addWidget(_btn("+ Range", lambda: self._add_range_row(0, 10, 1, 1), 80), 0, 1)
        mg.addWidget(_btn("- Last", self._remove_range_row, 80), 1, 1)
        mg.addWidget(_btn("+ Point", self._add_point_row, 80), 2, 1)
        self.genStack.addWidget(mp)

        self.t0OffsetSpin = _dspin(-100000, 100000, 0.0, 2, 1.0, 90, " ps")
        self.t0OffsetSpin.setToolTip("Added to every generated delay")
        g.addWidget(self.simpleRb, 0, 0)
        g.addWidget(self.multiRb, 0, 1)
        g.addWidget(QtWidgets.QLabel("t0 offset:"), 0, 2)
        g.addWidget(self.t0OffsetSpin, 0, 3)
        g.addWidget(self.genStack, 1, 0, 1, 4)
        g.addWidget(_btn("Generate", self._generate_delays), 2, 3)
        gen.setMinimumWidth(470)

        lst, g = _grid_box("Time delays (ps) — edit freely")
        self.delayEdit = QtWidgets.QPlainTextEdit(C.DEFAULT_DELAY_TEXT)
        self.delayEdit.setMinimumWidth(320)
        self.delayEdit.textChanged.connect(self._update_delay_count)
        self.delayCountLbl = QtWidgets.QLabel("")
        self.delayCountLbl.setWordWrap(True)
        self.avgMapLbl = QtWidgets.QLabel("")
        g.addWidget(self.delayEdit, 0, 0)
        g.addWidget(self.delayCountLbl, 1, 0)
        g.addWidget(self.avgMapLbl, 2, 0)
        self._update_simple_preview()
        return _page(acq, gen, lst, stretch=False)

    # ------------------------------------------------------------------- power page
    def _build_power_page(self) -> QtWidgets.QWidget:
        sw, g = _grid_box("Power sweep (one ND wheel)")
        self.sweepPowerChk = QtWidgets.QCheckBox("Sweep power")
        self.sweepPowerChk.setChecked(True)
        self.powerFileRb = QtWidgets.QRadioButton("Power list file")
        self.powerTableRb = QtWidgets.QRadioButton("Manual angle table")
        self.powerFileRb.setChecked(True)
        grp = QtWidgets.QButtonGroup(self)
        grp.addButton(self.powerFileRb)
        grp.addButton(self.powerTableRb)
        self.skipSpin = _ispin(0, 100000, 0, 80)
        self.skipSpin.setToolTip("Skip N power points: 0 = all, 1 = every other, ... (power list file only)")
        self.powerSettleSpin = _dspin(0, 600, C.DEFAULT_POWER_SETTLE_S, 2, 0.1, 80, " s")
        self.wheelStepSpin = _dspin(0.1, 360, C.DEFAULT_RAMP_STEP_DEG, 1, 1.0, 80, " °")
        self.wheelStepSpin.setToolTip("Wheel moves are made in steps of at most this (rot_wrapper ramp step)")
        self.powerPreviewLbl = QtWidgets.QLabel("")
        self.powerPreviewLbl.setWordWrap(True)
        self.powerPreviewLbl.setMinimumWidth(260)
        g.addWidget(self.sweepPowerChk, 0, 0, 1, 2)
        g.addWidget(self.powerFileRb, 1, 0)
        g.addWidget(self.powerTableRb, 1, 1)
        g.addWidget(QtWidgets.QLabel("Skip N power points:"), 2, 0)
        g.addWidget(self.skipSpin, 2, 1)
        g.addWidget(QtWidgets.QLabel("Settle after move:"), 3, 0)
        g.addWidget(self.powerSettleSpin, 3, 1)
        g.addWidget(QtWidgets.QLabel("Wheel move step:"), 4, 0)
        g.addWidget(self.wheelStepSpin, 4, 1)
        g.addWidget(self.powerPreviewLbl, 5, 0, 1, 2)

        fb, g = _grid_box("Power list file (ND-only .xlsx from wheel_hwp_power_calib_v2, or angle/power .csv)")
        self.loadCalibBtn = _btn("Load…", self._load_calib)
        self.calibPathLbl = QtWidgets.QLabel("No file loaded")
        self.calibPathLbl.setWordWrap(True)
        self.calibPathLbl.setMinimumWidth(300)
        self.seriesLbl = QtWidgets.QLabel("Series:")
        self.seriesCombo = QtWidgets.QComboBox()
        self.wheelStartSpin = _dspin(0, 360, 0.0, 3, 1.0, 90, " °")
        self.wheelEndSpin = _dspin(0, 360, 360.0, 3, 1.0, 90, " °")
        self.fullRangeBtn = _btn("Full range", self._set_full_wheel_range, 90)
        g.addWidget(self.loadCalibBtn, 0, 0)
        g.addWidget(self.calibPathLbl, 0, 1, 1, 4)
        g.addWidget(QtWidgets.QLabel("Wheel range:"), 1, 0)
        g.addWidget(self.wheelStartSpin, 1, 1)
        g.addWidget(QtWidgets.QLabel("→"), 1, 2)
        g.addWidget(self.wheelEndSpin, 1, 3)
        g.addWidget(self.fullRangeBtn, 1, 4)
        g.addWidget(self.seriesLbl, 2, 0)
        g.addWidget(self.seriesCombo, 2, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("Swept from the highest to the lowest power; the range runs from start\n"
                                     "towards increasing angle (wraps through 360°)."), 3, 0, 1, 5)
        self.seriesLbl.setVisible(False)
        self.seriesCombo.setVisible(False)

        tb, g = _grid_box("Manual angle table (swept in table order)")
        self.angleTable = QtWidgets.QTableWidget(0, 2)
        self.angleTable.setHorizontalHeaderLabels(["Angle (°)", "Power (optional)"])
        self.angleTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.angleTable.verticalHeader().setDefaultSectionSize(22)
        self.angleTable.setMinimumWidth(240)
        for a in C.DEFAULT_MANUAL_ANGLES:
            self._add_angle_row(a)
        self.tableUnitCombo = QtWidgets.QComboBox()
        self.tableUnitCombo.addItems(list(C.POWER_UNITS))
        self.tableUnitCombo.setCurrentText(C.DEFAULT_TABLE_UNIT)
        self.fillStartSpin = _dspin(0, 360, 0.0, 2, 1.0, 75)
        self.fillEndSpin = _dspin(0, 360, 90.0, 2, 1.0, 75)
        self.fillStepSpin = _dspin(0.01, 360, 10.0, 2, 1.0, 75)
        g.addWidget(self.angleTable, 0, 0, 5, 1)
        g.addWidget(_btn("Add row", lambda: self._add_angle_row(0.0), 80), 0, 1)
        g.addWidget(_btn("Delete row", self._delete_angle_row, 80), 0, 2)
        g.addWidget(QtWidgets.QLabel("Power unit:"), 1, 1)
        g.addWidget(self.tableUnitCombo, 1, 2)
        g.addWidget(QtWidgets.QLabel("Start / end / step:"), 2, 1, 1, 2)
        fill = QtWidgets.QHBoxLayout()
        for w in (self.fillStartSpin, self.fillEndSpin, self.fillStepSpin):
            fill.addWidget(w)
        g.addLayout(fill, 3, 1, 1, 2)
        g.addWidget(_btn("Fill from range", self._fill_angle_table, 120), 4, 1, 1, 2)

        for w in (self.sweepPowerChk, self.powerFileRb, self.powerTableRb):
            w.toggled.connect(self._update_power_preview)
        for w in (self.skipSpin, self.wheelStartSpin, self.wheelEndSpin):
            w.valueChanged.connect(self._update_power_preview)
        self.seriesCombo.currentIndexChanged.connect(self._update_power_preview)
        self.angleTable.itemChanged.connect(self._update_power_preview)
        self.tableUnitCombo.currentIndexChanged.connect(self._update_power_preview)
        return _page(sw, fb, tb, stretch=False)

    # ------------------------------------------------------------------- gate page
    def _build_gate_page(self) -> QtWidgets.QWidget:
        self.gateW: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        boxes = []
        for which, res in (("front", C.DEFAULT_FRONT_RESOURCE), ("back", C.DEFAULT_BACK_RESOURCE)):
            box, g = _grid_box(f"{which.capitalize()} gate (Keithley)")
            w = {
                "res": QtWidgets.QLineEdit(res),
                "icomp": _dspin(0.001, 1e6, C.DEFAULT_ICOMP_NA, 3, 1.0, 90, " nA"),
                "connect": _btn("Connect", lambda _=False, x=which: self._connect_gate(x), C.SMALL_BTN_W + 10),
                "disconnect": _btn("Disconnect", lambda _=False, x=which: self._disconnect_gate(x), C.SMALL_BTN_W + 10),
                "vset": _dspin(-C.V_LIMIT, C.V_LIMIT, 0.0, 4, 0.1, 90, " V"),
                "apply": _btn("Apply V", lambda _=False, x=which: self._apply_gate(x), C.SMALL_BTN_W),
                "abort": _btn("Abort", lambda _=False, x=which: self._abort_gate_ramp(x), C.SMALL_BTN_W),
                "vi": QtWidgets.QLabel("V = --   I = --"),
                "status": QtWidgets.QLabel("Not connected"),
            }
            w["res"].setMinimumWidth(170)
            w["status"].setWordWrap(True)
            g.addWidget(QtWidgets.QLabel("VISA:"), 0, 0)
            g.addWidget(w["res"], 0, 1, 1, 3)
            g.addWidget(QtWidgets.QLabel("I comp:"), 1, 0)
            g.addWidget(w["icomp"], 1, 1)
            g.addWidget(w["connect"], 1, 2)
            g.addWidget(w["disconnect"], 1, 3)
            g.addWidget(QtWidgets.QLabel("V set:"), 2, 0)
            g.addWidget(w["vset"], 2, 1)
            g.addWidget(w["apply"], 2, 2)
            g.addWidget(w["abort"], 2, 3)
            g.addWidget(w["vi"], 3, 0, 1, 4)
            g.addWidget(w["status"], 4, 0, 1, 4)
            w["icomp"].valueChanged.connect(lambda v, x=which: self.gateDevs[x].set_compliance_a(v * 1e-9))
            self.gateW[which] = w
            boxes.append(box)

        sw, g = _grid_box("Gate sweep (PL+TRPL v2 style)")
        self.sweepVoltageChk = QtWidgets.QCheckBox("Sweep voltage")
        self.sweepVoltageChk.setChecked(True)
        self.gateModeCombo = QtWidgets.QComboBox()
        self.gateModeCombo.addItems(["Dual gate", "Front gate", "Back gate"])
        self.gridChk = QtWidgets.QCheckBox("Grid (fix A, sweep B)")
        self.gridChk.setToolTip("Dual gate only: hold the front gate (A) at each step while the back gate (B)\n"
                                "runs through its list (snaked), then advance A.  A uses Steps, B uses B steps.")
        self.frontV0Spin = _dspin(-C.V_LIMIT, C.V_LIMIT, C.DEFAULT_SWEEP_V0, 4, 0.1, 90, " V")
        self.frontStepSpin = _dspin(-50, 50, C.DEFAULT_SWEEP_STEP, 4, 0.1, 90, " V")
        self.backV0Spin = _dspin(-C.V_LIMIT, C.V_LIMIT, C.DEFAULT_SWEEP_V0, 4, 0.1, 90, " V")
        self.backStepSpin = _dspin(-50, 50, C.DEFAULT_SWEEP_STEP, 4, 0.1, 90, " V")
        self.stepsSpin = _ispin(1, 100000, C.DEFAULT_SWEEP_STEPS, 70)
        self.bStepsSpin = _ispin(1, 100000, C.DEFAULT_SWEEP_STEPS, 70)
        self.bStepsLbl = QtWidgets.QLabel("B steps:")
        self.frontEndLbl = QtWidgets.QLabel("")
        self.backEndLbl = QtWidgets.QLabel("")
        self.gateSettleSpin = _ispin(0, 600000, C.DEFAULT_GATE_SETTLE_MS, 90)
        self.gateSettleSpin.setSuffix(" ms")
        self.orderPowerOuterRb = QtWidgets.QRadioButton("Voltages per power → next power")
        self.orderVoltageOuterRb = QtWidgets.QRadioButton("Powers per voltage → next voltage")
        self.orderPowerOuterRb.setChecked(True)
        og = QtWidgets.QButtonGroup(self)
        og.addButton(self.orderPowerOuterRb)
        og.addButton(self.orderVoltageOuterRb)
        self.homeGatesChk = QtWidgets.QCheckBox("Ramp swept gates to 0 V after the sweep (not after Stop)")
        self.gatePreviewLbl = QtWidgets.QLabel("")
        self.gatePreviewLbl.setWordWrap(True)
        g.addWidget(self.sweepVoltageChk, 0, 0, 1, 2)
        g.addWidget(QtWidgets.QLabel("Sweep gate:"), 0, 2)
        g.addWidget(self.gateModeCombo, 0, 3)
        g.addWidget(self.gridChk, 0, 4, 1, 2)
        g.addWidget(QtWidgets.QLabel("Front V0:"), 1, 0)
        g.addWidget(self.frontV0Spin, 1, 1)
        g.addWidget(QtWidgets.QLabel("Front step:"), 1, 2)
        g.addWidget(self.frontStepSpin, 1, 3)
        g.addWidget(QtWidgets.QLabel("Steps:"), 1, 4)
        g.addWidget(self.stepsSpin, 1, 5)
        g.addWidget(self.frontEndLbl, 1, 6)
        g.addWidget(QtWidgets.QLabel("Back V0:"), 2, 0)
        g.addWidget(self.backV0Spin, 2, 1)
        g.addWidget(QtWidgets.QLabel("Back step:"), 2, 2)
        g.addWidget(self.backStepSpin, 2, 3)
        g.addWidget(self.bStepsLbl, 2, 4)
        g.addWidget(self.bStepsSpin, 2, 5)
        g.addWidget(self.backEndLbl, 2, 6)
        g.addWidget(QtWidgets.QLabel("Settle after ramp:"), 3, 0)
        g.addWidget(self.gateSettleSpin, 3, 1)
        g.addWidget(QtWidgets.QLabel("Order:"), 4, 0)
        g.addWidget(self.orderPowerOuterRb, 4, 1, 1, 3)
        g.addWidget(self.orderVoltageOuterRb, 4, 4, 1, 3)
        g.addWidget(self.homeGatesChk, 5, 0, 1, 7)
        g.addWidget(self.gatePreviewLbl, 6, 0, 1, 7)
        for w in (self.frontV0Spin, self.frontStepSpin, self.backV0Spin, self.backStepSpin,
                  self.stepsSpin, self.bStepsSpin):
            w.valueChanged.connect(self._update_gate_preview)
        for w in (self.sweepVoltageChk, self.gridChk):
            w.toggled.connect(self._update_gate_preview)
        self.gateModeCombo.currentIndexChanged.connect(self._update_gate_preview)
        return _page(boxes[0], boxes[1], sw, stretch=False)

    # ------------------------------------------------------------------- analysis page
    def _build_analysis_page(self) -> QtWidgets.QWidget:
        ab, g = _grid_box("Reference ROI (pump ON / OFF)  &  save region")
        x, y, w, h = C.DEFAULT_ROI
        self.roiX, self.roiY = _ispin(0, 100000, x, 70), _ispin(0, 100000, y, 70)
        self.roiW, self.roiH = _ispin(1, 100000, w, 70), _ispin(1, 100000, h, 70)
        self.cropChk = QtWidgets.QCheckBox("Crop columns for display && save")
        self.cropChk.setChecked(C.DEFAULT_CROP_ENABLED)
        self.cropX0 = _ispin(0, 100000, C.DEFAULT_CROP_X0, 70)
        self.cropW = _ispin(1, 100000, C.DEFAULT_CROP_W, 70)
        g.addWidget(QtWidgets.QLabel("ROI X:"), 0, 0)
        g.addWidget(self.roiX, 0, 1)
        g.addWidget(QtWidgets.QLabel("Y:"), 0, 2)
        g.addWidget(self.roiY, 0, 3)
        g.addWidget(QtWidgets.QLabel("W:"), 1, 0)
        g.addWidget(self.roiW, 1, 1)
        g.addWidget(QtWidgets.QLabel("H:"), 1, 2)
        g.addWidget(self.roiH, 1, 3)
        g.addWidget(self.cropChk, 2, 0, 1, 4)
        g.addWidget(QtWidgets.QLabel("X start:"), 3, 0)
        g.addWidget(self.cropX0, 3, 1)
        g.addWidget(QtWidgets.QLabel("Width:"), 3, 2)
        g.addWidget(self.cropW, 3, 3)
        g.addWidget(QtWidgets.QLabel("The ROI uses full-frame pixels."), 4, 0, 1, 4)

        db, g = _grid_box("Calibration && display")
        self.pixelSizeSpin = _dspin(1e-6, 1000, C.DEFAULT_PIXEL_SIZE_UM, 4, 0.01, 90, " µm/px")
        self.wlKSpin = _dspin(-1e6, 1e6, C.DEFAULT_WAVELENGTH_K, 5, 0.01, 90)
        self.wlBSpin = _dspin(-1e6, 1e6, C.DEFAULT_WAVELENGTH_B, 3, 1.0, 90)
        self.zAutoChk = QtWidgets.QCheckBox("Auto (symmetric)")
        self.zAutoChk.setChecked(True)
        self.zMinSpin = _dspin(-1e6, 1e6, C.DEFAULT_Z_RANGE[0], 3, 1.0, 80)
        self.zMaxSpin = _dspin(-1e6, 1e6, C.DEFAULT_Z_RANGE[1], 3, 1.0, 80)
        self.monAutoChk = QtWidgets.QCheckBox("Auto")
        self.monAutoChk.setChecked(True)
        self.monMinSpin = _dspin(-1e6, 1e7, C.DEFAULT_MONITOR_Z[0], 0, 100, 80)
        self.monMaxSpin = _dspin(-1e6, 1e7, C.DEFAULT_MONITOR_Z[1], 0, 100, 80)
        g.addWidget(QtWidgets.QLabel("Pixel size (space):"), 0, 0)
        g.addWidget(self.pixelSizeSpin, 0, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("λ = k·px + b:  k"), 1, 0)
        g.addWidget(self.wlKSpin, 1, 1)
        g.addWidget(QtWidgets.QLabel("b"), 1, 2)
        g.addWidget(self.wlBSpin, 1, 3)
        g.addWidget(QtWidgets.QLabel("Signal range:"), 2, 0)
        g.addWidget(self.zAutoChk, 2, 1)
        g.addWidget(self.zMinSpin, 2, 2)
        g.addWidget(self.zMaxSpin, 2, 3)
        g.addWidget(QtWidgets.QLabel("ON/OFF image range:"), 3, 0)
        g.addWidget(self.monAutoChk, 3, 1)
        g.addWidget(self.monMinSpin, 3, 2)
        g.addWidget(self.monMaxSpin, 3, 3)
        for w in (self.pixelSizeSpin, self.wlKSpin, self.wlBSpin, self.zMinSpin, self.zMaxSpin,
                  self.monMinSpin, self.monMaxSpin):
            w.valueChanged.connect(self._apply_display_settings)
        for w in (self.zAutoChk, self.monAutoChk):
            w.toggled.connect(self._apply_display_settings)

        sb, g = _grid_box("Save")
        self.saveDirEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveDirEdit.setMinimumWidth(220)
        self.newDirBtn = _btn("New", lambda: self.saveDirEdit.setText(self._default_save_name()), C.SMALL_BTN_W)
        self.saveBaseEdit = QtWidgets.QLineEdit(C.DEFAULT_SAVE_BASE)
        self.autoSaveChk = QtWidgets.QCheckBox("Auto-save single delay scans")
        self.autoSaveChk.setChecked(True)
        self.openDirBtn = _btn("Open folder", self._open_save_dir, 100)
        dl = QtWidgets.QLabel(f"DATA_DIR: {C.DATA_DIR}")
        dl.setWordWrap(True)
        g.addWidget(QtWidgets.QLabel("Subfolder:"), 0, 0)
        g.addWidget(self.saveDirEdit, 0, 1)
        g.addWidget(self.newDirBtn, 0, 2)
        g.addWidget(QtWidgets.QLabel("File base:"), 1, 0)
        g.addWidget(self.saveBaseEdit, 1, 1, 1, 2)
        g.addWidget(self.autoSaveChk, 2, 0, 1, 2)
        g.addWidget(self.openDirBtn, 2, 2)
        g.addWidget(dl, 3, 0, 1, 3)
        return _page(ab, db, sb)

    # =================================================================== device wiring
    def _wire_devices(self) -> None:
        for dev in (self.cameraDev, self.delayDev, self.wheelDev, self.polDev, *self.gateDevs.values()):
            dev.message.connect(self._status.showMessage)
            dev.state_changed.connect(lambda *_: self._update_buttons())
        self.cameraDev.connected.connect(self._on_camera_connected)
        self.cameraDev.state_changed.connect(self._on_camera_state)
        self.delayDev.connected.connect(self._on_stage_connected)
        self.delayDev.readback.connect(self._on_stage_readback)
        self.delayDev.state_changed.connect(self._on_stage_state)
        self.wheelDev.readback.connect(lambda d: self._on_rot_readback("wheel", d))
        self.polDev.readback.connect(lambda d: self._on_rot_readback("pol", d))
        self.wheelDev.connected.connect(lambda info: self._on_rot_connected(self.wheelDev, info))
        self.polDev.connected.connect(lambda info: self._on_rot_connected(self.polDev, info))
        self.wheelDev.state_changed.connect(lambda s: s == DISCONNECTED and self.wheelPosLbl.setText("Angle: --"))
        self.polDev.state_changed.connect(lambda s: s == DISCONNECTED and self.polPosLbl.setText("Angle: --"))
        for which, dev in self.gateDevs.items():
            dev.connected.connect(lambda info, x=which: self._on_gate_connected(x, info))
            dev.readback.connect(lambda d, x=which: self._on_gate_readback(x, d))
            dev.state_changed.connect(lambda s, x=which: self._on_gate_state(x, s))

    # ------------------------------------------------------------------- camera
    def _connect_camera(self) -> None:
        self.cameraDev.connect_device(simulate=self.simulateChk.isChecked())

    def _on_camera_connected(self, info) -> None:
        sim = " (simulated)" if info.get("simulated") else ""
        self.camStatusLbl.setText(f"Connected{sim}: {info['width']} x {info['height']} {info['dtype']}")
        self.camStatusLbl.setStyleSheet("color: green;")

    def _on_camera_state(self, state: str) -> None:
        if state == DISCONNECTED:
            self.camStatusLbl.setText("Not connected")
            self.camStatusLbl.setStyleSheet("")
        elif state != CONNECTED:
            self.camStatusLbl.setText(state.capitalize() + "...")

    @property
    def camera(self):
        return self.cameraDev.camera

    # ------------------------------------------------------------------- delay stage
    def _refresh_ports(self) -> None:
        cur = self.portCombo.currentText().strip() or C.DEFAULT_SMC_PORT
        self.portCombo.clear()
        for dev, desc in list_serial_ports():
            self.portCombo.addItem(f"{dev} — {desc}" if desc else dev, dev)
        idx = next((i for i in range(self.portCombo.count()) if self.portCombo.itemData(i) == cur), -1)
        if idx >= 0:
            self.portCombo.setCurrentIndex(idx)
        else:
            self.portCombo.setEditText(cur)

    def _port(self) -> str:
        i = self.portCombo.currentIndex()
        txt = self.portCombo.currentText().strip()
        if i >= 0 and self.portCombo.itemText(i) == txt and self.portCombo.itemData(i):
            return str(self.portCombo.itemData(i))
        return txt.split(" ")[0]

    def _connect_stage(self) -> None:
        self.delayDev.connect_device(port=self._port(), address=self.addrSpin.value(),
                                     simulate=self.simulateChk.isChecked())

    def _on_stage_connected(self, info) -> None:
        lo, hi = info.limits_mm
        self.stageLimitsLbl.setText(f"{info.version}  |  limits {lo:g} .. {hi:g} mm  |  v {info.velocity_mm_s:g} mm/s")
        self._stage_state = {"pos": info.position_mm, "homed": info.state.is_referenced,
                             "ready": info.state.is_ready, "state_text": info.state.text()}
        self._show_stage_state()
        if not info.state.is_referenced:
            ans = QtWidgets.QMessageBox.question(
                self, "Delay stage",
                f"The SMC100 is {info.state.name}.\nIt must be homed before any absolute move.\n\nHome it now?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if ans == QtWidgets.QMessageBox.Yes:
                self._home_stage()
        self._update_buttons()

    def _on_stage_state(self, state: str) -> None:
        if state == DISCONNECTED:
            self._stage_state = {}
            self.stageStateLbl.setText("Not connected")
            self.stageStateLbl.setStyleSheet("")
            self.stagePosLbl.setText("Position: -- mm")
            self.stageLimitsLbl.setText("")

    def _on_stage_readback(self, d: Dict) -> None:
        if not d:
            return
        self._stage_state.update({k: v for k, v in d.items() if v is not None})
        self._show_stage_state()
        self._update_buttons()

    def _show_stage_state(self) -> None:
        s = self._stage_state
        txt = s.get("state_text", "")
        if s.get("error"):
            txt = f"read error: {s['error']}"
        self.stageStateLbl.setText(txt)
        color = "green" if s.get("ready") else ("orange" if s.get("moving") else "red")
        self.stageStateLbl.setStyleSheet(f"color: {color};")
        pos = s.get("pos")
        if pos is not None:
            rel_ps = ta_core.mm_to_ps(pos - self.t0_mm, C.DELAY_PASS_FACTOR)
            self.stagePosLbl.setText(f"Position: {pos:.4f} mm   ({rel_ps:+.2f} ps from t0)")

    def _stage_call(self, fn: Callable, done_msg: str) -> None:
        def on_done(_res, err):
            self._status.showMessage(f"Delay stage: {err}" if err else done_msg)
            if err:
                QtWidgets.QMessageBox.warning(self, "Delay stage", err)
            self._update_buttons()
        if self.delayDev.call(fn, on_done):
            self._update_buttons()

    def _home_stage(self) -> None:
        stage = self.delayDev.stage
        if stage is not None:
            self._status.showMessage("Delay stage: homing...")
            self._stage_call(lambda: stage.home(wait=True), "Delay stage homed")

    def _stop_stage(self) -> None:
        stage = self.delayDev.stage
        if stage is not None:
            # not through call(): a move / home call may be running; ST goes in between its polls
            th = CallThread(stage.stop, self)
            self._keep_thread(th)
            th.start()

    def _goto_delay(self) -> None:
        stage = self.delayDev.stage
        if stage is None:
            return
        ps = self.gotoSpin.value()
        mm = self.t0_mm + ta_core.ps_to_mm(ps, C.DELAY_PASS_FACTOR)
        self._stage_call(lambda: stage.move_to(mm, wait=True), f"Moved to {ps:+.2f} ps ({mm:.4f} mm)")

    def _set_t0_from_stage(self) -> None:
        pos = self._stage_state.get("pos")
        if pos is None:
            QtWidgets.QMessageBox.warning(self, "t0", "Stage position not known yet.")
            return
        self.t0Spin.setValue(float(pos))

    def _on_t0_changed(self, v: float) -> None:
        self.t0_mm = float(v)
        self._show_stage_state()
        self._update_delay_count()

    # ------------------------------------------------------------------- rotation stages
    def _keep_thread(self, th: QtCore.QThread) -> None:
        self._aux_threads.add(th)
        th.finished.connect(lambda th=th: self._aux_threads.discard(th))

    def _detect_kinesis(self) -> None:
        def listing():
            from rot.rot_wrapper import list_kinesis_serials
            return list_kinesis_serials()

        def done(res, err):
            self.detectBtn.setEnabled(True)
            if err:
                self._status.showMessage(f"Detect failed: {err}")
                return
            serials = [str(s) for s in (res or [])]
            for combo in (self.wheelSerialCombo, self.polSerialCombo):
                cur = combo.currentText()
                combo.clear()
                combo.addItems(serials or [cur])
                combo.setEditText(cur if cur in serials or not serials else serials[0])
            self._status.showMessage(f"Kinesis devices: {', '.join(serials) or 'none'}")

        self.detectBtn.setEnabled(False)
        th = CallThread(listing, self)
        th.done.connect(done)
        self._keep_thread(th)
        th.start()

    def _connect_rot(self, dev: StageDevice, combo: QtWidgets.QComboBox) -> None:
        serial = combo.currentText().strip()
        if serial:
            dev.connect_device(serial=serial, home=False)

    def _on_rot_connected(self, dev: StageDevice, info) -> None:
        self._update_buttons()

    def _on_rot_readback(self, which: str, d: Dict) -> None:
        if not d:
            return
        pos = d.get("pos")
        homed = d.get("homed")
        txt = f"Angle: {pos:.3f} °" if pos is not None else "Angle: --"
        if homed is False:
            txt += "  (not homed)"
        if d.get("moving"):
            txt += "  moving"
        if which == "wheel":
            self._wheel_pos = pos
            self.wheelPosLbl.setText(txt)
        else:
            self._pol_pos = pos
            self.polPosLbl.setText(txt)

    def _home_rot(self, dev: StageDevice) -> None:
        st = dev.stage
        if st is not None:
            dev.call(lambda: st.home(), lambda _r, e: self._status.showMessage(
                f"{dev.name}: {e}" if e else f"{dev.name} homed"))

    def _move_rot(self, dev: StageDevice, angle: float) -> None:
        st = dev.stage
        if st is not None:
            dev.call(lambda: st.move_to(angle, step_deg=self.wheelStepSpin.value()),
                     lambda _r, e: self._status.showMessage(f"{dev.name}: {e}" if e else
                                                            f"{dev.name} at {angle:.3f} °"))

    def _update_rh_label(self, *_a) -> None:
        self.rhLbl.setText(f"RH = {self.lhSpin.value() + C.POL_RH_OFFSET_DEG:.2f} °")

    # ------------------------------------------------------------------- gates
    def _connect_gate(self, which: str) -> None:
        w = self.gateW[which]
        res = w["res"].text().strip()
        if res:
            self.gateDevs[which].connect_device(resource=res, icomp_a=w["icomp"].value() * 1e-9)

    def _disconnect_gate(self, which: str) -> None:
        dev = self.gateDevs[which]
        dev.leave_output_on = False
        dev.disconnect_device()

    def _on_gate_connected(self, which: str, info) -> None:
        w = self.gateW[which]
        w["status"].setText(("Connected: " + (info.get("idn") or ""))[:70])
        w["status"].setStyleSheet("color: green;")
        if info.get("v_now") is not None:
            w["vset"].setValue(float(info["v_now"]))

    def _on_gate_state(self, which: str, state: str) -> None:
        w = self.gateW[which]
        if state == DISCONNECTED:
            w["status"].setText("Not connected" + (f" ({self.gateDevs[which].last_close_error})"
                                                   if self.gateDevs[which].last_close_error else ""))
            w["status"].setStyleSheet("")
            w["vi"].setText("V = --   I = --")
            self._gate_vi.pop(which, None)
        elif state != CONNECTED:
            w["status"].setText(state.capitalize() + "...")
        self._update_buttons()

    def _on_gate_readback(self, which: str, d: Dict) -> None:
        if not d:
            return
        if "v" in d and "i" in d:
            self._show_gate_vi(which, d["v"], d["i"])
        elif d.get("lost"):
            self.gateW[which]["status"].setText("Link lost: reconnecting..." if not d.get("reconnect_failed")
                                                else f"Reconnect failed: {d['reconnect_failed']}"[:80])
            self.gateW[which]["status"].setStyleSheet("color: red;")
        elif d.get("reconnected"):
            self.gateW[which]["status"].setText("Reconnected")
            self.gateW[which]["status"].setStyleSheet("color: green;")

    def _show_gate_vi(self, which: str, v: float, i: float) -> None:
        self._gate_vi[which] = (v, i)
        self.gateW[which]["vi"].setText(f"V = {v:.4g} V   I = {i * 1e9:.4g} nA")
        self.gateVIPanel.add_point(which, v, i)

    def _apply_gate(self, which: str) -> None:
        dev = self.gateDevs[which]
        if dev.dev is None or which in self._gate_ramps:
            return
        w = self.gateW[which]
        ctrl = SweepController()
        th = GateRampThread(dev.dev, w["vset"].value(), w["icomp"].value() * 1e-9, ctrl, C.RAMP_STEP_V,
                            C.RAMP_DWELL_S, self)
        th.progress.connect(lambda v, i, x=which: self._show_gate_vi(x, v, i))
        th.status.connect(lambda s, x=which: self.gateW[x]["status"].setText(s))
        th.done.connect(lambda s, x=which: self._on_gate_ramp_done(x, s))
        self._gate_ramps[which] = (th, ctrl)
        dev.pause()

        def start():
            if dev.quiet or time.monotonic() - t0 > 5.0:
                th.start()
            else:
                QtCore.QTimer.singleShot(50, start)
        t0 = time.monotonic()
        start()
        self._update_buttons()

    def _abort_gate_ramp(self, which: str) -> None:
        item = self._gate_ramps.get(which)
        if item is not None:
            item[1].abort()

    def _on_gate_ramp_done(self, which: str, status: str) -> None:
        self._gate_ramps.pop(which, None)
        w = self.gateW[which]
        if status == "ok":
            w["status"].setText(f"Set to {w['vset'].value():.4g} V")
            w["status"].setStyleSheet("color: green;")
        else:
            w["status"].setText(f"Ramp {status}"[:90])
            w["status"].setStyleSheet("color: orange;" if status == "aborted" else "color: red;")
        if not self._busy():
            self.gateDevs[which].resume()
        self._update_buttons()

    def gates_at_voltage(self) -> Dict[str, float]:
        out = {}
        for which, dev in self.gateDevs.items():
            if dev.state == DISCONNECTED:
                continue
            vi = self._gate_vi.get(which)
            v = vi[0] if vi else float("nan")
            if not (abs(v) <= 5e-3):
                out[which] = v
        return out

    def leave_gate_outputs_on(self) -> None:
        for dev in self.gateDevs.values():
            dev.leave_output_on = True

    def ramp_down_gates(self, then: Callable[[Dict[str, str]], None]) -> None:
        """Window close: stop, ramp every connected gate to 0 V and switch it off, then ``then(errors)``."""
        self.on_stop()
        self._status.showMessage("Ramping the gates to 0 V before closing...")
        t0 = time.monotonic()

        def start():
            if self._busy() and time.monotonic() - t0 < 30:
                QtCore.QTimer.singleShot(100, start)
                return
            for which in GATES:
                self._disconnect_gate(which)
            wait()

        def wait():
            if any(d.state != DISCONNECTED for d in self.gateDevs.values()):
                QtCore.QTimer.singleShot(100, wait)
                return
            then({w: d.last_close_error for w, d in self.gateDevs.items() if d.last_close_error})

        start()

    # =================================================================== delays
    def _add_range_row(self, s, e, st, av) -> None:
        r = self.rangeTable.rowCount()
        self.rangeTable.insertRow(r)
        for c, v in enumerate((s, e, st, av)):
            self.rangeTable.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v:g}"))

    def _remove_range_row(self) -> None:
        if self.rangeTable.rowCount() > 1:
            self.rangeTable.removeRow(self.rangeTable.rowCount() - 1)

    def _add_point_row(self) -> None:
        t, ok = QtWidgets.QInputDialog.getDouble(self, "Add time point", "Time delay (ps):", 0.0, -10000, 10000, 2)
        if not ok:
            return
        av, ok2 = QtWidgets.QInputDialog.getInt(self, "Averaging", "Captures to average at this point:", 5, 1, 100)
        self._add_range_row(t, t, 1, av if ok2 else 5)

    def _update_simple_preview(self, *_a) -> None:
        start, n, inc = self.simpleStartSpin.value(), self.simpleStepsSpin.value(), self.simpleIncSpin.value()
        end = start + (n - 1) * inc
        self.simplePreviewLbl.setText(f"End {end:.2f} ps (abs {end + self.t0OffsetSpin.value():.2f})")

    def _scan_around_zero(self) -> None:
        rng, ok = QtWidgets.QInputDialog.getDouble(self, "Around t=0", "Total range (ps):", 20.0, 0.1, 100000, 2)
        if not ok:
            return
        inc, ok2 = QtWidgets.QInputDialog.getDouble(self, "Around t=0", "Step (ps):", 2.0, 0.01, 10000, 2)
        if not ok2:
            return
        self.simpleStartSpin.setValue(-(rng / 2))
        self.simpleStepsSpin.setValue(int(rng / inc) + 1)
        self.simpleIncSpin.setValue(inc)

    def _scan_from_zero(self) -> None:
        end, ok = QtWidgets.QInputDialog.getDouble(self, "From t=0", "End time (ps):", 50.0, -100000, 100000, 2)
        if not ok:
            return
        inc, ok2 = QtWidgets.QInputDialog.getDouble(self, "From t=0", "Step (ps):", 5.0, 0.01, 10000, 2)
        if not ok2:
            return
        self.simpleStartSpin.setValue(0.0 if end >= 0 else end)
        self.simpleStepsSpin.setValue(int(abs(end) / inc) + 1)
        self.simpleIncSpin.setValue(inc)

    def _generate_delays(self) -> None:
        off = self.t0OffsetSpin.value()
        try:
            if self.simpleRb.isChecked():
                delays = ta_core.simple_delays(self.simpleStartSpin.value(), self.simpleStepsSpin.value(),
                                               self.simpleIncSpin.value(), off)
                self.delay_avg_map = {}
            else:
                rows = []
                for r in range(self.rangeTable.rowCount()):
                    vals = [self.rangeTable.item(r, c).text() if self.rangeTable.item(r, c) else "" for c in range(4)]
                    if not any(v.strip() for v in vals):
                        continue
                    rows.append((float(vals[0]), float(vals[1]), float(vals[2]), int(float(vals[3]))))
                delays, self.delay_avg_map, info = ta_core.multi_range_delays(rows, off)
                self._status.showMessage(f"Generated {len(delays)} delays: " + "; ".join(i.strip() for i in info))
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Delays", str(exc))
            return
        self.delayEdit.setPlainText(", ".join(f"{p:.2f}" for p in delays))

    def _delays(self) -> List[float]:
        return ta_core.parse_delay_text(self.delayEdit.toPlainText())

    def _update_delay_count(self, *_a) -> None:
        try:
            d = self._delays()
        except ValueError:
            self.delayCountLbl.setText("Invalid number in the delay list")
            self.delayCountLbl.setStyleSheet("color: red;")
            return
        self.delayCountLbl.setStyleSheet("")
        if not d:
            self.delayCountLbl.setText("No delays")
            return
        mm = ta_core.delay_positions_mm(d, self.t0_mm, C.DELAY_PASS_FACTOR)
        n_avg = sum(ta_core.avg_for_delay(x, self.delay_avg_map, self.avgSpin.value()) for x in d)
        self.delayCountLbl.setText(f"{len(d)} delays, {d[0]:.2f} .. {d[-1]:.2f} ps  →  stage "
                                   f"{min(mm):.4f} .. {max(mm):.4f} mm  ({n_avg} captures per delay scan)")
        n_map = sum(1 for x in d if round(x, 2) in self.delay_avg_map)
        self.avgMapLbl.setText(f"Per-delay avg from multi-range: {n_map}/{len(d)} delays" if self.delay_avg_map else "")

    # =================================================================== power plan
    def _load_calib(self) -> None:
        start_dir = os.path.dirname(self._calib_path) if self._calib_path else C.DATA_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load power list", start_dir,
                                                        "Power list (*.xlsx *.xlsm *.csv);;All files (*)")
        if not path:
            return
        try:
            data = load_power_calibration(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load power list", f"{type(exc).__name__}: {exc}")
            return
        if not data.entries:
            QtWidgets.QMessageBox.critical(self, "Load power list",
                                           "No (angle, power) rows found.\nExpected a header like 'angle, power (nW)'.")
            return
        self._calib = data
        self._calib_path = path
        self.calibPathLbl.setText(f"{os.path.basename(path)}  ({len(data.entries)} points, {data.unit})")
        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItems(data.series_names)
        if "base" in data.series_names:
            self.seriesCombo.setCurrentText("base")
        self.seriesCombo.blockSignals(False)
        many = len(data.series_names) > 1
        self.seriesLbl.setVisible(many)
        self.seriesCombo.setVisible(many)
        self.powerFileRb.setChecked(True)
        self._set_full_wheel_range()
        self._status.showMessage(f"Loaded power list {os.path.basename(path)}")

    def _set_full_wheel_range(self) -> None:
        if self._calib is None or not self._calib.entries:
            return
        pos = [e.position_deg for e in self._calib.entries]
        lo, hi = min(pos), max(pos)
        for spin, val in ((self.wheelStartSpin, lo), (self.wheelEndSpin, hi)):
            spin.blockSignals(True)
            spin.setRange(min(0.0, lo), max(360.0, hi))
            spin.setValue(val)
            spin.blockSignals(False)
        self._update_power_preview()

    def _file_entries(self):
        """pl_trpl v2 single-wheel plan: entries in the wheel range, by power, skip-N, highest -> lowest."""
        if self._calib is None:
            return []
        series = self.seriesCombo.currentText().strip() or "base"
        start, end = self.wheelStartSpin.value(), self.wheelEndSpin.value()
        span = (end - start) % 360.0
        if span == 0.0 and end != start:
            span = 360.0
        entries = [e for e in self._calib.entries if e.series == series
                   and (e.position_deg - start + 1e-6) % 360.0 <= span + 2e-6]
        entries = sorted(entries, key=lambda e: (e.power_w, e.position_deg))
        stride = max(1, self.skipSpin.value() + 1)
        entries = entries[::stride]  # stride from the lowest power so it is always kept
        return entries[::-1]  # sweep from the highest to the lowest power

    def _table_entries(self) -> Tuple[List[Tuple[float, float]], str]:
        scale = C.POWER_UNITS.get(self.tableUnitCombo.currentText(), 1e-9)
        out = []
        for r in range(self.angleTable.rowCount()):
            a_item, p_item = self.angleTable.item(r, 0), self.angleTable.item(r, 1)
            a_txt = a_item.text().strip() if a_item else ""
            p_txt = p_item.text().strip() if p_item else ""
            if not a_txt:
                continue
            try:
                a = float(a_txt)
            except ValueError:
                return [], f"Row {r + 1}: bad angle '{a_txt}'"
            p = float("nan")
            if p_txt:
                try:
                    p = float(p_txt) * scale
                except ValueError:
                    return [], f"Row {r + 1}: bad power '{p_txt}'"
            out.append((a, p))
        return out, ""

    def _power_plan(self) -> Tuple[List[Optional[Dict]], str, str]:
        """(plan, error, unit): [{"angle", "power_w", "idx"}], or [None] when power is not swept."""
        if not self.sweepPowerChk.isChecked():
            return [None], "", "nW"
        if self.powerFileRb.isChecked():
            if self._calib is None:
                return [], "Load a power list file first (or use the manual angle table).", "nW"
            entries = self._file_entries()
            if not entries:
                return [], (f"No power points in the wheel range {self.wheelStartSpin.value():g}° → "
                            f"{self.wheelEndSpin.value():g}°."), self._calib.unit
            plan = [{"angle": float(e.position_deg), "power_w": float(e.power_w), "idx": i}
                    for i, e in enumerate(entries)]
            return plan, "", self._calib.unit
        rows, err = self._table_entries()
        if err:
            return [], err, self.tableUnitCombo.currentText()
        if not rows:
            return [], "The angle table is empty.", self.tableUnitCombo.currentText()
        return [{"angle": a, "power_w": p, "idx": i} for i, (a, p) in enumerate(rows)], "", \
            self.tableUnitCombo.currentText()

    def _update_power_preview(self, *_a) -> None:
        file_mode = self.powerFileRb.isChecked()
        self.skipSpin.setEnabled(file_mode and self.sweepPowerChk.isChecked() and not self._busy())
        plan, err, unit = self._power_plan()
        if err:
            self.powerPreviewLbl.setText(err)
            self.powerPreviewLbl.setStyleSheet("color: #c62828;")
            return
        self.powerPreviewLbl.setStyleSheet("")
        if plan == [None]:
            self.powerPreviewLbl.setText("Power not swept: the wheel is not moved.")
            return
        scale = C.POWER_UNITS.get(unit, 1e-9)

        def fmt(e):
            p = e["power_w"]
            return f"{e['angle']:.2f}°" + (f" ({p / scale:.4g} {unit})" if math.isfinite(p) else "")

        head = ", ".join(fmt(e) for e in plan[:6]) + (" …" if len(plan) > 6 else "")
        self.powerPreviewLbl.setText(f"{len(plan)} powers: {head}")

    def _add_angle_row(self, angle: float, power: str = "") -> None:
        r = self.angleTable.rowCount()
        self.angleTable.insertRow(r)
        self.angleTable.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{angle:g}"))
        self.angleTable.setItem(r, 1, QtWidgets.QTableWidgetItem(power))

    def _delete_angle_row(self) -> None:
        rows = sorted({i.row() for i in self.angleTable.selectedIndexes()}, reverse=True)
        if not rows and self.angleTable.rowCount():
            rows = [self.angleTable.rowCount() - 1]
        for r in rows:
            self.angleTable.removeRow(r)
        self._update_power_preview()

    def _fill_angle_table(self) -> None:
        s, e, st = self.fillStartSpin.value(), self.fillEndSpin.value(), self.fillStepSpin.value()
        vals = []
        a = s
        direction = 1.0 if e >= s else -1.0
        while (a - e) * direction <= abs(st) * 1e-6:
            vals.append(round(a, 6))
            a += direction * abs(st)
        self.angleTable.blockSignals(True)
        self.angleTable.setRowCount(0)
        for v in vals:
            self._add_angle_row(v)
        self.angleTable.blockSignals(False)
        self._update_power_preview()

    # =================================================================== gate plan
    def _gate_mode(self) -> str:
        return {0: "dual", 1: "front", 2: "back"}.get(self.gateModeCombo.currentIndex(), "dual")

    def _grid_selected(self) -> bool:
        return self.sweepVoltageChk.isChecked() and self.gridChk.isChecked() and self._gate_mode() == "dual"

    def _gate_lists(self) -> Tuple[List[float], List[float]]:
        n = self.stepsSpin.value()
        nb = self.bStepsSpin.value() if self._grid_selected() else n
        vf = [self.frontV0Spin.value() + k * self.frontStepSpin.value() for k in range(n)]
        vb = [self.backV0Spin.value() + k * self.backStepSpin.value() for k in range(nb)]
        return vf, vb

    def _update_gate_preview(self, *_a) -> None:
        sweep = self.sweepVoltageChk.isChecked()
        mode = self._gate_mode()
        grid = self._grid_selected()
        idle = not self._busy()
        for w in (self.gateModeCombo, self.stepsSpin, self.gateSettleSpin):
            w.setEnabled(sweep and idle)
        self.gridChk.setEnabled(sweep and mode == "dual" and idle)
        self.frontV0Spin.setEnabled(sweep and mode in ("front", "dual") and idle)
        self.frontStepSpin.setEnabled(sweep and mode in ("front", "dual") and idle)
        self.backV0Spin.setEnabled(sweep and mode in ("back", "dual") and idle)
        self.backStepSpin.setEnabled(sweep and mode in ("back", "dual") and idle)
        self.bStepsSpin.setEnabled(grid and idle)
        self.bStepsLbl.setEnabled(grid)
        vf, vb = self._gate_lists()
        self.frontEndLbl.setText(f"End: {vf[-1]:.4g} V")
        self.backEndLbl.setText(f"End: {vb[-1]:.4g} V")
        if not sweep:
            self.gatePreviewLbl.setText("Voltage not swept: the gates stay where they are (V/I still read and saved).")
            return
        bad = [v for v in (vf if mode in ("front", "dual") else []) + (vb if mode in ("back", "dual") else [])
               if abs(v) > C.V_LIMIT]
        txt = ""
        if mode in ("front", "dual"):
            txt += "Front: " + ", ".join(f"{v:.4g}" for v in vf[:8]) + (" …" if len(vf) > 8 else "") + " V\n"
        if mode in ("back", "dual"):
            txt += "Back: " + ", ".join(f"{v:.4g}" for v in vb[:8]) + (" …" if len(vb) > 8 else "") + " V\n"
        n = len(vf) * len(vb) if grid else len(vf)
        txt += f"{n} gate steps" + (" (grid: A outer, B snaked)" if grid else " (paired by step index)")
        if bad:
            txt += f"\n|V| > {C.V_LIMIT:g} V: not allowed"
        self.gatePreviewLbl.setText(txt)
        self.gatePreviewLbl.setStyleSheet("color: #c62828;" if bad else "")

    # =================================================================== display
    def _crop(self) -> Optional[Tuple[int, int]]:
        return (self.cropX0.value(), self.cropW.value()) if self.cropChk.isChecked() else None

    def _roi(self) -> Tuple[int, int, int, int]:
        return self.roiX.value(), self.roiY.value(), self.roiW.value(), self.roiH.value()

    def _apply_display_settings(self, *_a) -> None:
        self.imagePanel.set_levels(self.zAutoChk.isChecked(), self.zMinSpin.value(), self.zMaxSpin.value())
        self.imagePanel.set_calibration(self.wlKSpin.value(), self.wlBSpin.value(), self.pixelSizeSpin.value())
        self.monitorPanel.set_levels(self.monAutoChk.isChecked(), self.monMinSpin.value(), self.monMaxSpin.value())
        for w in (self.zMinSpin, self.zMaxSpin):
            w.setEnabled(not self.zAutoChk.isChecked())
        for w in (self.monMinSpin, self.monMaxSpin):
            w.setEnabled(not self.monAutoChk.isChecked())

    def _display_results(self, results: Dict, roi, info: str = "") -> None:
        self._current_results = results
        crop = self._crop()
        on = ta_core.crop_columns(results.get("avg_pump_on"), crop)
        off = ta_core.crop_columns(results.get("avg_pump_off"), crop)
        self.monitorPanel.update_results(results, roi, crop)
        if on is None or off is None:
            self.imagePanel.infoLbl.setText(f"No pump {'ON' if on is None else 'OFF'} frames "
                                            f"(check the reference ROI)")
            return
        lr = ta_core.log_ratio(on, off)
        self.imagePanel.set_roi(roi, crop[0] if crop else 0)
        self.imagePanel.set_image(lr)
        if info:
            self.imagePanel.setTitle(f"log(Pump ON / Pump OFF)  [-1000 log10]   {info}")
        self._update_gaussian()

    def _update_gaussian(self) -> None:
        prof = self.imagePanel.space_profile()
        if prof is not None and self.displayTabs.currentWidget() is self.gaussPanel:
            unit = "um" if self.pixelSizeSpin.value() != 1.0 else "px"
            self.gaussPanel.update_fit(prof[0], prof[1], unit)

    # =================================================================== live / scans
    def _busy(self) -> bool:
        return any(t is not None and t.isRunning() for t in (self._live_thread, self._scan_thread, self._sweep_thread))

    def on_connect_main(self) -> None:
        if self.cameraDev.state == DISCONNECTED:
            self._connect_camera()
        if self.delayDev.state == DISCONNECTED:
            self._connect_stage()

    def on_disconnect_main(self) -> None:
        if self._busy():
            return
        self.cameraDev.disconnect_device()
        self.delayDev.disconnect_device()

    def on_live(self) -> None:
        if self._busy() or self.camera is None:
            return
        self._live_thread = LiveCaptureThread(self.camera, self.framesSpin.value(), self._roi(), self)
        self._live_thread.result_ready.connect(self._on_live_result)
        self._live_thread.error_occurred.connect(lambda m: QtWidgets.QMessageBox.critical(self, "Capture", m))
        self._live_thread.finished.connect(self._on_thread_finished)
        self._live_count = 0
        self._live_roi = self._roi()
        self._live_thread.start()
        self._status.showMessage("Live capture running")
        self._update_buttons()

    def _on_live_result(self, results: Dict) -> None:
        self._live_count += 1
        now = time.monotonic()
        if (now - self._last_display) * 1000.0 >= C.DISPLAY_REFRESH_MS:
            self._last_display = now
            self._display_results(results, self._live_roi,
                                  f"live #{self._live_count}: ON {results['n_pump_on']} / OFF {results['n_pump_off']}")

    def _check_delay_setup(self) -> Optional[Tuple[List[float], List[float]]]:
        try:
            delays = self._delays()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Delays", "The delay list has an invalid number.")
            return None
        if not delays:
            QtWidgets.QMessageBox.critical(self, "Delays", "No time delays (Delay scan tab).")
            return None
        if self.delayDev.stage is None:
            QtWidgets.QMessageBox.critical(self, "Delay stage", "Connect the delay stage first.")
            return None
        if not self._stage_state.get("homed", False):
            QtWidgets.QMessageBox.critical(self, "Delay stage", "The SMC100 is not homed: click Home first.")
            return None
        mm = ta_core.delay_positions_mm(delays, self.t0_mm, C.DELAY_PASS_FACTOR)
        lo, hi = self.delayDev.stage.limits_mm
        out = [p for p in mm if p < lo - 1e-9 or p > hi + 1e-9]
        if out:
            QtWidgets.QMessageBox.critical(
                self, "Delay stage", f"{len(out)} delay positions are outside the stage limits {lo:g} .. {hi:g} mm\n"
                f"(positions {min(mm):.4f} .. {max(mm):.4f} mm, t0 = {self.t0_mm:.4f} mm).")
            return None
        if self.t0_mm == 0.0:
            ans = QtWidgets.QMessageBox.warning(
                self, "Time zero not set", "Time zero is 0.0 mm: it has probably not been set.\n\n"
                f"Stage positions: {min(mm):.4f} – {max(mm):.4f} mm.\n\nContinue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return None
        return delays, mm

    def _hook_sim_camera(self, thread) -> None:
        if isinstance(self.camera, SimulatedCamera):
            self.camera.delay_ps_fn = lambda th=thread: th.current_delay_ps

    def on_delay_scan(self) -> None:
        if self._busy() or self.camera is None:
            return
        setup = self._check_delay_setup()
        if setup is None:
            return
        delays, mm = setup
        th = DelayScanThread(self.camera, self.delayDev.stage, delays, mm, self.framesSpin.value(), self._roi(),
                             self.delay_avg_map, self.avgSpin.value(), self)
        th.status.connect(self._status.showMessage)
        th.step_result.connect(self._on_step_result)
        th.finished_scan.connect(self._on_delay_scan_finished)
        th.error_occurred.connect(lambda m: QtWidgets.QMessageBox.critical(self, "Delay scan", m))
        th.finished.connect(self._on_thread_finished)
        self._scan_thread = th
        self._hook_sim_camera(th)
        self._pause_polls()
        th.start()
        self._update_buttons()

    def _on_step_result(self, results, roi, info) -> None:
        self.progressLbl.setText(f"delay {info.get('index', 0) + 1}/{info.get('total', 0)}")
        self._display_results(results, roi, f"{info['time_delay_ps']:+.2f} ps (avg {info['avg_times_used']})")

    def _on_delay_scan_finished(self, results: list, completed: bool) -> None:
        self._status.showMessage(f"Delay scan {'complete' if completed else 'stopped'}: {len(results)} delays")
        if completed and results and self.autoSaveChk.isChecked():
            try:
                save_dir = self._resolve_save_dir()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                angle = self._wheel_pos if self.wheelDev.is_connected else None
                prefix = f"{self.saveBaseEdit.text().strip() or C.DEFAULT_SAVE_BASE}{ta_core.angle_tag(angle)}_{ts}"
                para, summ = ta_core.save_scan(
                    save_dir, prefix, results, pixel_size=self.pixelSizeSpin.value(), crop=self._crop(),
                    camera_width=getattr(self.camera, "width", 480) or 480, crop_h=self._crop_h(),
                    t0_ps=ta_core.mm_to_ps(self.t0_mm, C.DELAY_PASS_FACTOR),
                    angle_deg=angle if angle is not None else 0.0, n_frames=self.framesSpin.value())
                self._log(f"Saved {os.path.basename(summ)}")
                self._status.showMessage(f"Saved {os.path.basename(summ)} in {save_dir}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Auto-save", f"{type(exc).__name__}: {exc}")

    def _crop_h(self) -> int:
        cam = self.camera
        return int(cam.height) if cam is not None and getattr(cam, "is_initialized", False) else 128

    # ------------------------------------------------------------------- sweep
    def on_sweep(self) -> None:
        if self._busy() or self.camera is None:
            return
        setup = self._check_delay_setup()
        if setup is None:
            return
        delays, mm = setup

        plan, err, unit = self._power_plan()
        if err:
            QtWidgets.QMessageBox.critical(self, "Power", err)
            return
        sweep_power = plan != [None]
        sweep_v = self.sweepVoltageChk.isChecked()
        mode = self._gate_mode()
        pol_on = self.polEnableChk.isChecked()
        if not self._check_sweep_devices(sweep_power, sweep_v, mode, pol_on):
            return
        if sweep_v:
            vf, vb = self._gate_lists()
            vals = (vf if mode in ("front", "dual") else []) + (vb if mode in ("back", "dual") else [])
            if any(abs(v) > C.V_LIMIT for v in vals):
                QtWidgets.QMessageBox.critical(self, "Gates", f"A gate voltage exceeds ±{C.V_LIMIT:g} V.")
                return

        grid = self._grid_selected()
        n_gate = (self.stepsSpin.value() * self.bStepsSpin.value()) if grid else (self.stepsSpin.value() if sweep_v else 1)
        n_pol = 2 if pol_on else 1
        n_scans = len(plan) * n_gate * n_pol
        caps = sum(ta_core.avg_for_delay(d, self.delay_avg_map, self.avgSpin.value()) for d in delays)
        order = "voltage_outer" if self.orderVoltageOuterRb.isChecked() else "power_outer"
        msg = (f"Start the gate / power dependent pump-probe sweep?\n\n"
               f"Powers: {len(plan) if sweep_power else 'not swept'}\n"
               f"Gate steps: {n_gate if sweep_v else 'not swept'}"
               f"{'  (' + mode + (', grid' if grid else '') + ')' if sweep_v else ''}\n"
               f"Polarization: {'LH + RH' if pol_on else 'off'}\n"
               f"Delays per scan: {len(delays)} ({delays[0]:.2f} .. {delays[-1]:.2f} ps), stage "
               f"{min(mm):.4f} .. {max(mm):.4f} mm (t0 {self.t0_mm:.4f} mm)\n"
               f"Frames / capture: {self.framesSpin.value()}, captures / scan: {caps}\n"
               f"Order: {'powers per voltage' if order == 'voltage_outer' else 'voltages per power'}"
               f"{' (grid)' if grid else ''}\n\n"
               f"Total delay scans: {n_scans}\nTotal captures: {n_scans * caps}")
        if QtWidgets.QMessageBox.question(self, "Confirm sweep", msg,
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return

        save_dir = self._resolve_save_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # every GatePowerSweepThread argument except the devices, JSON-safe: kept in the sweep state file
        params = dict(
            time_delays_ps=list(delays), positions_mm=list(mm), n_frames=self.framesSpin.value(),
            roi=list(self._roi()), avg_map=[[k, v] for k, v in self.delay_avg_map.items()],
            default_avg=self.avgSpin.value(), power_plan=plan,
            wheel_step_deg=self.wheelStepSpin.value(), power_settle_s=self.powerSettleSpin.value(),
            sweep_voltage=sweep_v, sweep_mode=mode, grid=grid,
            front_v0=self.frontV0Spin.value(), front_step=self.frontStepSpin.value(),
            back_v0=self.backV0Spin.value(), back_step=self.backStepSpin.value(),
            nsteps=self.stepsSpin.value(), nsteps_b=self.bStepsSpin.value(),
            ramp_step_v=C.RAMP_STEP_V, ramp_dwell_s=C.RAMP_DWELL_S, long_ramp_steps=C.LONG_RAMP_STEPS,
            gate_settle_s=self.gateSettleSpin.value() / 1000.0, order=order,
            home_gates_after=self.homeGatesChk.isChecked(),
            pol_enabled=pol_on, lh_angle=self.lhSpin.value(),
            rh_offset=C.POL_RH_OFFSET_DEG, pol_settle_s=self.polSettleSpin.value(),
            save_base=self.saveBaseEdit.text().strip() or C.DEFAULT_SAVE_BASE,
            crop=list(self._crop()) if self._crop() else None,
            t0_ps=ta_core.mm_to_ps(self.t0_mm, C.DELAY_PASS_FACTOR),
            pixel_size=self.pixelSizeSpin.value(), crop_h=self._crop_h(),
            camera_width=getattr(self.camera, "width", 480) or 480, power_unit=unit,
            _info={"t0_mm": self.t0_mm, "simulated": self.simulateChk.isChecked()})
        th = self._make_sweep_thread(params, save_dir, timestamp)
        prog = SweepProgress.new(save_dir, timestamp, params, th.scan_labels())
        if prog.last_save_error:
            self._log(f"WARNING: sweep state file not written ({prog.last_save_error}): Resume after a crash "
                      f"will not be possible")
        th.sweep_progress = prog
        self._progress = prog
        self._write_session(th, plan, unit, delays, mm, timestamp)
        self._log(f"=== Sweep {timestamp}: {n_scans} delay scans -> {save_dir}")
        self._log(f"    progress file: {os.path.basename(prog.path)}")
        self._start_sweep_thread(th)

    def _check_sweep_devices(self, sweep_power: bool, sweep_v: bool, mode: str, pol_on: bool) -> bool:
        if sweep_power and self.wheelDev.stage is None:
            QtWidgets.QMessageBox.critical(self, "Power", "Connect the ND wheel first (Hardware tab).")
            return False
        if sweep_v:
            need = {"dual": ("front", "back"), "front": ("front",), "back": ("back",)}[mode]
            missing = [w for w in need if self.gateDevs[w].dev is None]
            if missing:
                QtWidgets.QMessageBox.critical(self, "Gates", f"Connect the {' and '.join(missing)} gate(s) first.")
                return False
        if pol_on and self.polDev.stage is None:
            QtWidgets.QMessageBox.critical(self, "Polarization", "Polarization enabled but the stage is not connected.")
            return False
        return True

    def _make_sweep_thread(self, params: Dict, save_dir: str, timestamp: str) -> GatePowerSweepThread:
        """Sweep thread from the saved ``params`` and the devices connected now.

        The gate current compliances are the values in the Gates tab now (they may have been
        raised after a trip); everything else comes from ``params``.
        """
        p = {k: v for k, v in params.items() if not k.startswith("_")}
        p["roi"] = tuple(p["roi"])
        p["crop"] = tuple(p["crop"]) if p.get("crop") else None
        p["avg_map"] = {float(k): int(v) for k, v in p["avg_map"]}
        sweep_power = p["power_plan"] != [None]
        front_dev = self.gateDevs["front"].dev
        back_dev = self.gateDevs["back"].dev
        return GatePowerSweepThread(
            camera=self.camera, stage=self.delayDev.stage,
            wheel=self.wheelDev.stage if sweep_power else None,
            front_dev=front_dev, back_dev=back_dev,
            front_icomp_a=self.gateW["front"]["icomp"].value() * 1e-9 if front_dev is not None else None,
            back_icomp_a=self.gateW["back"]["icomp"].value() * 1e-9 if back_dev is not None else None,
            pol_stage=self.polDev.stage if p["pol_enabled"] else None,
            save_dir=save_dir, timestamp=timestamp, parent=self, **p)

    def _start_sweep_thread(self, th: GatePowerSweepThread) -> None:
        th.status.connect(self._status.showMessage)
        th.step_result.connect(self._on_step_result)
        th.vi_ready.connect(self._on_sweep_vi)
        th.scan_saved.connect(self._on_scan_saved)
        th.progress.connect(self._on_sweep_progress)
        th.done.connect(self._on_sweep_done)
        th.finished.connect(self._on_thread_finished)
        self._sweep_thread = th
        self._hook_sim_camera(th)
        self._sweep_t0 = time.monotonic()
        self._sweep_done0 = None
        self._pause_polls()
        t0 = time.monotonic()

        def start():  # let in-flight polls finish before the sweep thread drives the devices
            if self._sweep_thread is not th:  # terminated before it started
                return
            if self._devices_quiet() or time.monotonic() - t0 > 5.0:
                th.start()
                self._update_buttons()
            else:
                QtCore.QTimer.singleShot(50, start)
        start()
        self._update_buttons()

    def _on_sweep_vi(self, d: Dict) -> None:
        for which in GATES:
            if f"v_{which}" in d:
                self._show_gate_vi(which, d[f"v_{which}"], d[f"i_{which}"])

    def _on_sweep_progress(self, done: int, total: int) -> None:
        if self.sender() is not self._sweep_thread:  # late signal of a terminated sweep
            return
        if self._sweep_done0 is None:
            self._sweep_done0 = done  # scans already done when this run (re)started
        el = time.monotonic() - self._sweep_t0
        eta = ""
        n_run = done - self._sweep_done0
        if n_run > 0 and total > done:
            rem = el / n_run * (total - done)
            eta = f", ~{rem / 60:.0f} min left"
        self.progressLbl.setText(f"scan {done}/{total}{eta}")

    def _on_scan_saved(self, info: Dict) -> None:
        num = f"#{info['scan_idx'] + 1}/{info['total']} " if "scan_idx" in info else ""
        self._log(f"[{info['status']}] {num}{info['prefix']}  ({info['n_delays']} delays)")

    def _on_sweep_done(self, status: str) -> None:
        if self.sender() is not self._sweep_thread:  # late signal of a terminated sweep
            return
        el = time.monotonic() - self._sweep_t0
        prog = self._progress
        self._log(f"=== Sweep ended: {status} after {el / 60:.1f} min")
        prog_txt = ""
        if prog is not None:
            self._log(f"    progress: {prog.n_done}/{prog.total} scans done"
                      + (f"; unfinished: {prog.label(prog.data.get('redo'))}" if prog.data.get("redo") is not None
                         else ""))
            prog_txt = f"- scans done: {prog.n_done}/{prog.total}\n"
        self._append_session(f"\n## End\n\n- status: {status}\n- ended: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                             f"- duration: {el / 60:.1f} min\n{prog_txt}")
        if status == "ok":
            QtWidgets.QMessageBox.information(self, "Sweep", f"Sweep finished.\nData in:\n{self._save_dir}")
        elif status == "aborted":
            if prog is not None and not prog.finished:
                self._log("    'Resume sweep' continues it (the unfinished scan is measured again)")
        else:
            more = ""
            if prog is not None and not prog.finished:
                more = (f"\n\nProgress is saved: {prog.n_done}/{prog.total} scans done "
                        f"({os.path.basename(prog.path)}).\n"
                        "Check / reconnect the hardware, then click 'Resume sweep'\n"
                        "(the unfinished scan is measured again).")
            QtWidgets.QMessageBox.critical(self, "Sweep", f"Sweep ended: {status}{more}")
        self._status.showMessage(f"Sweep {status}")

    def on_stop(self) -> None:
        for th in (self._live_thread, self._scan_thread, self._sweep_thread):
            if th is not None and th.isRunning():
                th.stop()
        QtCore.QTimer.singleShot(200, self._check_stop)

    # ------------------------------------------------------------------- terminate / resume
    def on_terminate(self) -> None:
        """Abandon the running live view / delay scan / sweep at once (stuck hardware)."""
        running = [(name, th) for name, th in (("live view", self._live_thread), ("delay scan", self._scan_thread),
                                               ("sweep", self._sweep_thread))
                   if th is not None and th.isRunning()]
        if not running:
            return
        sweep = self._sweep_thread if any(n == "sweep" for n, _ in running) else None
        prog = self._progress if sweep is not None else None
        txt = f"Terminate the {' and '.join(n for n, _ in running)} now?\n\n"
        if prog is not None:
            cur = prog.data.get("current")
            bad = cur if cur is not None else prog.data.get("last_done")
            txt += (f"Scans done: {prog.n_done}/{prog.total}.\n"
                    f"Treated as bad and measured again by 'Resume sweep':\n  {prog.label(bad)}\n"
                    f"  ({'in progress' if cur is not None else 'last finished scan'})\n\n")
        txt += ("A hardware call that does not return is left behind; it ends when the device answers\n"
                "or is disconnected.  Then reconnect / test the hardware (Hardware tab, Live)"
                + (" and click 'Resume sweep'." if prog is not None else "."))
        if QtWidgets.QMessageBox.question(self, "Terminate", txt, QtWidgets.QMessageBox.Yes |
                                          QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        running = [(n, th) for n, th in running if th.isRunning()]  # one may have ended during the dialog
        if not running:
            return
        if not any(n == "sweep" for n, _ in running):
            prog = None
        for name, th in running:
            th.blockSignals(True)  # nothing from it reaches the GUI any more
            th.abandon()
            self._zombies.append((name, th))
        self._live_thread = self._scan_thread = self._sweep_thread = None
        self._stalled = False
        self._stall_key = None
        self._log(f"=== TERMINATED ({', '.join(n for n, _ in running)})")
        if prog is not None:
            bad = prog.terminate()
            if bad is not None:
                what = "finished scan discarded" if bad["was_done"] else "unfinished scan"
                self._log(f"    {what}: {bad['label']} -> measured again on Resume")
                for p in bad["renamed"]:
                    self._log(f"    renamed {os.path.basename(p)}")
                try:
                    append_scan_log(prog.save_dir, [prog.label(bad["idx"]), "", datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")] + [""] * 12 + ["discarded (terminated): redo on Resume"])
                except Exception:
                    pass
            self._log(f"    progress: {prog.n_done}/{prog.total} scans done; state in {os.path.basename(prog.path)}")
            self._append_session(f"\n## Terminated\n\n- time: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                                 f"- scans done: {prog.n_done}/{prog.total}\n"
                                 f"- redo: {bad['label'] if bad else '-'}\n")
        self._resume_polls()
        self.progressLbl.setText("")
        self._update_buttons()
        self._status.showMessage("Terminated: reconnect / test the hardware"
                                 + (", then 'Resume sweep'" if prog is not None else ""))

    def _prune_zombies(self) -> None:
        for name, th in list(self._zombies):
            if not th.isRunning():
                self._zombies.remove((name, th))
                self._log(f"(the terminated {name} thread has ended: the stuck hardware call returned)")

    def on_resume(self) -> None:
        if self._busy():
            return
        prog = self._progress if (self._progress is not None and not self._progress.finished) else None
        if prog is not None:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Resume sweep")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText(f"Resume the last sweep ({prog.timestamp}, {prog.n_done}/{prog.total} scans done)?")
            this_btn = box.addButton("This sweep", QtWidgets.QMessageBox.AcceptRole)
            file_btn = box.addButton("Other file…", QtWidgets.QMessageBox.ActionRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is file_btn:
                prog = None
            elif box.clickedButton() is not this_btn:
                return
        if prog is None:
            prog = self._load_progress_file()
            if prog is None:
                return
        self._resume(prog)

    def _load_progress_file(self) -> Optional[SweepProgress]:
        start = self._save_dir if self._save_dir and os.path.isdir(self._save_dir) else C.DATA_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Resume sweep: choose its state file", start,
                                                        f"Sweep state ({STATE_PREFIX}*.json);;All files (*)")
        if not path:
            return None
        try:
            prog = SweepProgress.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Resume sweep", f"Cannot read {os.path.basename(path)}:\n"
                                                                 f"{type(exc).__name__}: {exc}")
            return None
        if prog.finished:
            QtWidgets.QMessageBox.information(self, "Resume sweep", f"All {prog.total} scans of this sweep are done.")
            return None
        if prog.needs_crash_recovery():
            # the GUI died during this sweep: same rule as Terminate
            bad = prog.discard_last("interrupted (GUI closed / crashed during the sweep)")
            self._log(f"=== Sweep {prog.timestamp} was interrupted (GUI crash?)")
            if bad is not None:
                self._log(f"    {'finished scan discarded' if bad['was_done'] else 'unfinished scan'}: "
                          f"{bad['label']} -> measured again")
                for p in bad["renamed"]:
                    self._log(f"    renamed {os.path.basename(p)}")
        return prog

    def _resume(self, prog: SweepProgress) -> None:
        p = prog.params
        if self.camera is None or self.delayDev.stage is None:
            QtWidgets.QMessageBox.critical(self, "Resume sweep", "Connect the camera and the delay stage first.")
            return
        if not self._stage_state.get("homed", False):
            QtWidgets.QMessageBox.critical(self, "Resume sweep", "The SMC100 is not homed: click Home first.")
            return
        if not self._check_sweep_devices(p["power_plan"] != [None], p["sweep_voltage"], p["sweep_mode"],
                                         p["pol_enabled"]):
            return
        mm = p["positions_mm"]
        lo, hi = self.delayDev.stage.limits_mm
        if any(x < lo - 1e-9 or x > hi + 1e-9 for x in mm):
            QtWidgets.QMessageBox.critical(self, "Resume sweep", f"Delay positions {min(mm):.4f} .. {max(mm):.4f} mm "
                                                                 f"are outside the stage limits {lo:g} .. {hi:g} mm.")
            return
        try:
            th = self._make_sweep_thread(p, prog.save_dir, prog.timestamp)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Resume sweep", f"Cannot rebuild the sweep: {type(exc).__name__}: {exc}")
            return
        if th.total_scans() != prog.total:
            QtWidgets.QMessageBox.critical(self, "Resume sweep", f"The rebuilt plan has {th.total_scans()} scans, the "
                                                                 f"state file {prog.total}: not resumed.")
            th.deleteLater()
            return
        remaining = prog.remaining()
        redo = prog.data.get("redo")
        info = p.get("_info", {})
        warn = ""
        cam_w = getattr(self.camera, "width", None)
        if cam_w and int(cam_w) != int(p["camera_width"]):
            warn += f"\n\nWARNING: camera width is {cam_w} px now, {p['camera_width']} px at the start."
        t0_start = info.get("t0_mm")
        if t0_start is not None and abs(float(t0_start) - self.t0_mm) > 1e-9:
            warn += (f"\n\nNote: the delay positions of the sweep are used (t0 = {float(t0_start):.4f} mm at the "
                     f"start); the t0 field now says {self.t0_mm:.4f} mm.")
        icomp = "  ".join(f"{w} {self.gateW[w]['icomp'].value():g} nA" for w in GATES if self.gateDevs[w].dev)
        msg = (f"Resume sweep {prog.timestamp}?\n\nFolder: {prog.save_dir}\nStatus: {prog.status}\n"
               f"Scans done: {prog.n_done}/{prog.total}   remaining: {len(remaining)}\n"
               + (f"Measured again first: {prog.label(redo)}\n" if redo is not None and redo in remaining else "")
               + f"Next: {prog.label(remaining[0])}\n"
               + (f"Gate compliance (Gates tab now): {icomp}\n" if icomp else "")
               + "\nSame delays, powers, gates, file names; finished scans are skipped." + warn)
        if QtWidgets.QMessageBox.question(self, "Resume sweep", msg, QtWidgets.QMessageBox.Yes |
                                          QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            th.deleteLater()
            return
        prog.resumed()
        if prog.last_save_error:
            self._log(f"WARNING: sweep state file not written: {prog.last_save_error}")
        th.sweep_progress = prog
        self._progress = prog
        self._save_dir = prog.save_dir
        self._session_path = os.path.join(prog.save_dir, f"session_{prog.timestamp}.md")
        self._append_session(f"\n## Resumed\n\n- time: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                             f"- scans done: {prog.n_done}/{prog.total}; redo first: {prog.label(redo)}\n"
                             + (f"- gate compliance: {icomp}\n" if icomp else ""))
        self._log(f"=== Sweep {prog.timestamp} resumed: {prog.n_done}/{prog.total} done, {len(remaining)} to go "
                  f"-> {prog.save_dir}")
        self._start_sweep_thread(th)

    # ------------------------------------------------------------------- watchdog / errors
    def _running_thread(self):
        for th in (self._sweep_thread, self._scan_thread, self._live_thread):
            if th is not None and th.isRunning():
                return th
        return None

    def _on_watchdog(self) -> None:
        """Warn when a hardware call takes far longer than it should (nothing is stopped)."""
        if self._zombies:
            self._prune_zombies()
        th = self._running_thread()
        act = getattr(th, "activity", None) if th is not None else None
        if act is not None:
            text, t_start, limit = act
            el = time.monotonic() - t_start
            if el > limit:
                key = (id(th), t_start)
                if key != self._stall_key:
                    self._stall_key = key
                    self._log(f"WARNING: no response for {el:.0f} s: {text}.  If it is stuck: Terminate, reconnect / "
                              f"test the hardware, then Resume sweep.")
                    QtWidgets.QApplication.beep()
                self._stalled = True
                self.stateLbl.setText(f"⚠ No response for {el:.0f} s: {text}")
                self.stateLbl.setStyleSheet("color: #c62828; font-weight: bold;")
                return
        if self._stalled:
            self._stalled = False
            self._update_buttons()

    def _on_error_reported(self, text: str) -> None:
        self._log(f"ERROR (caught, GUI kept running): {text}")
        self._status.showMessage(f"Error: {text.splitlines()[0] if text else ''}")

    def _check_stop(self) -> None:
        if self._busy():
            QtCore.QTimer.singleShot(200, self._check_stop)
            return
        self._update_buttons()

    def _on_thread_finished(self) -> None:
        if not self._busy():
            self._resume_polls()
            self.progressLbl.setText("")
        self._update_buttons()

    # ------------------------------------------------------------------- polls
    def _poll_devices(self):
        return [self.delayDev, self.wheelDev, self.polDev, *self.gateDevs.values()]

    def _pause_polls(self) -> None:
        for d in self._poll_devices():
            d.pause()

    def _resume_polls(self) -> None:
        for which, d in self.gateDevs.items():
            if which not in self._gate_ramps:
                d.resume()
        for d in (self.delayDev, self.wheelDev, self.polDev):
            d.resume()

    def _devices_quiet(self) -> bool:
        return all(d.quiet for d in self._poll_devices())

    # ------------------------------------------------------------------- saving
    @staticmethod
    def _default_save_name() -> str:
        return time.strftime(f"{C.SAVE_DIR_PREFIX}_%Y%m%d_%H%M%S")

    def _resolve_save_dir(self) -> str:
        name = self.saveDirEdit.text().strip()
        if not name:
            name = self._default_save_name()
            self.saveDirEdit.setText(name)
        path = os.path.join(C.DATA_DIR, name)
        os.makedirs(path, exist_ok=True)
        self._save_dir = path
        return path

    def _open_save_dir(self) -> None:
        path = os.path.join(C.DATA_DIR, self.saveDirEdit.text().strip())
        if os.path.isdir(path):
            os.startfile(path)  # noqa: S606 (Windows only)
        else:
            self._status.showMessage(f"Folder does not exist yet: {path}")

    def _log(self, text: str) -> None:
        self.logEdit.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def _write_session(self, th: GatePowerSweepThread, plan, unit, delays, mm, timestamp) -> None:
        path = os.path.join(self._save_dir, f"session_{timestamp}.md")
        self._session_path = path
        scale = C.POWER_UNITS.get(unit, 1e-9)
        calib_copy = ""
        if plan != [None] and self.powerFileRb.isChecked() and self._calib_path:
            try:
                dst = os.path.join(self._save_dir, os.path.basename(self._calib_path))
                if os.path.abspath(dst) != os.path.abspath(self._calib_path):
                    shutil.copy2(self._calib_path, dst)
                calib_copy = os.path.basename(dst)
            except Exception as exc:
                calib_copy = f"(copy failed: {exc})"
        vf, vb = self._gate_lists()
        lines = [
            f"# Pump-probe gate / power sweep (Montana) {timestamp}", "",
            f"- started: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- save dir: {self._save_dir}",
            f"- file base: {th.save_base}",
            "", "## Delay", "",
            f"- SMC100 port: {self._port()} (address {self.addrSpin.value()}), "
            f"{'simulated' if self.simulateChk.isChecked() else 'hardware'}",
            f"- t0: {self.t0_mm:.4f} mm  (pass factor {C.DELAY_PASS_FACTOR:g}; t0 in para = "
            f"{ta_core.mm_to_ps(self.t0_mm, C.DELAY_PASS_FACTOR):.6f} ps)",
            f"- delays (ps): {', '.join(f'{d:.2f}' for d in delays)}",
            f"- stage positions (mm): {min(mm):.4f} .. {max(mm):.4f}",
            f"- per-delay avg map: {self.delay_avg_map or 'none'}; default avg {self.avgSpin.value()}",
            "", "## Camera / analysis", "",
            f"- frames per capture: {self.framesSpin.value()}",
            f"- reference ROI (x, y, w, h): {self._roi()}",
            f"- crop (x0, w): {self._crop()}",
            f"- pixel size: {self.pixelSizeSpin.value()} um/px; wavelength k={self.wlKSpin.value()}, "
            f"b={self.wlBSpin.value()}",
            "", "## Power", "",
        ]
        if plan == [None]:
            lines.append("- not swept")
        else:
            lines += [f"- source: {'power list file ' + self._calib_path if self.powerFileRb.isChecked() else 'manual table'}",
                      f"- copied list: {calib_copy or '-'}",
                      f"- wheel: {self.wheelSerialCombo.currentText()} (rot_wrapper, {C.DEFAULT_STAGE_SCALE}), "
                      f"range {self.wheelStartSpin.value():g} -> {self.wheelEndSpin.value():g} deg, "
                      f"skip N {self.skipSpin.value()}, settle {self.powerSettleSpin.value():g} s",
                      "", "| # | angle (deg) | power (" + unit + ") |", "|---|---|---|"]
            for e in plan:
                p = e["power_w"]
                lines.append(f"| {e['idx']} | {e['angle']:.4f} | {p / scale:.6g} |" if math.isfinite(p)
                             else f"| {e['idx']} | {e['angle']:.4f} | - |")
        lines += ["", "## Gates", ""]
        if not th.sweep_voltage:
            lines.append("- not swept")
        else:
            lines += [f"- mode: {th.sweep_mode}{' (grid: front A outer, back B snaked)' if th.grid else ''}",
                      f"- front: V0 {th.front_v0:g} V, step {th.front_step:g} V; back: V0 {th.back_v0:g} V, "
                      f"step {th.back_step:g} V; steps {th.nsteps}{', B steps ' + str(th.nsteps_b) if th.grid else ''}",
                      f"- front list: {', '.join(f'{v:.4g}' for v in vf)}",
                      f"- back list: {', '.join(f'{v:.4g}' for v in vb)}",
                      f"- I comp: front {self.gateW['front']['icomp'].value():g} nA, back "
                      f"{self.gateW['back']['icomp'].value():g} nA; ramp {C.RAMP_STEP_V * 1000:g} mV / "
                      f"{C.RAMP_DWELL_S:g} s; settle {th.gate_settle_s:g} s",
                      f"- order: {th.order}; home after: {th.home_gates_after}"]
        lines += ["", "## Polarization", "",
                  f"- {'LH ' + format(th.lh_angle, 'g') + ' deg / RH ' + format(th.lh_angle + th.rh_offset, 'g') + ' deg' if th.pol_enabled else 'off'}",
                  "", f"Per-scan details: {os.path.basename(os.path.join(self._save_dir, 'scan_log.csv'))}", ""]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as exc:
            self._status.showMessage(f"session file not written: {exc}")

    def _append_session(self, text: str) -> None:
        if not self._session_path:
            return
        try:
            with open(self._session_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    # =================================================================== buttons
    def _update_buttons(self, *_a) -> None:
        busy = self._busy()
        cam = self.cameraDev.is_connected
        stage = self.delayDev.is_connected
        stage_ok = stage and bool(self._stage_state.get("homed"))
        self.connectBtn.setEnabled(not busy and (self.cameraDev.state == DISCONNECTED
                                                 or self.delayDev.state == DISCONNECTED))
        self.disconnectBtn.setEnabled(not busy and (cam or stage))
        self.liveBtn.setEnabled(cam and not busy)
        self.scanBtn.setEnabled(cam and stage_ok and not busy)
        self.sweepBtn.setEnabled(cam and stage_ok and not busy)
        self.stopBtn.setEnabled(busy)
        self.terminateBtn.setEnabled(busy)
        self.resumeBtn.setEnabled(cam and stage_ok and not busy)

        idle = not busy
        self.camConnectBtn.setEnabled(idle and self.cameraDev.state == DISCONNECTED)
        self.camDisconnectBtn.setEnabled(idle and cam)
        self.simulateChk.setEnabled(self.cameraDev.state == DISCONNECTED and self.delayDev.state == DISCONNECTED)
        call = self.delayDev.call_running
        self.stageConnectBtn.setEnabled(idle and self.delayDev.state == DISCONNECTED)
        self.stageDisconnectBtn.setEnabled(idle and stage and not call)
        self.stageHomeBtn.setEnabled(idle and stage and not call)
        self.gotoBtn.setEnabled(idle and stage_ok and not call)
        self.t0FromPosBtn.setEnabled(stage)
        self.stageStopBtn.setEnabled(stage)
        for dev, cbtn, dbtn, hbtn in ((self.wheelDev, self.wheelConnectBtn, self.wheelDisconnectBtn, self.wheelHomeBtn),
                                      (self.polDev, self.polConnectBtn, self.polDisconnectBtn, self.polHomeBtn)):
            cbtn.setEnabled(idle and dev.state == DISCONNECTED)
            dbtn.setEnabled(idle and dev.is_connected)
            hbtn.setEnabled(idle and dev.is_connected)
        self.wheelGotoBtn.setEnabled(idle and self.wheelDev.is_connected)
        for which, dev in self.gateDevs.items():
            w = self.gateW[which]
            ramping = which in self._gate_ramps
            w["connect"].setEnabled(idle and dev.state == DISCONNECTED)
            w["disconnect"].setEnabled(idle and dev.is_connected and not ramping)
            w["apply"].setEnabled(idle and dev.is_connected and not ramping)
            w["abort"].setEnabled(ramping)
            w["res"].setEnabled(dev.state == DISCONNECTED)

        for page in (self.delayPage, self.powerPage):
            page.setEnabled(idle)
        for w in (self.sweepVoltageChk, self.orderPowerOuterRb, self.orderVoltageOuterRb, self.homeGatesChk,
                  self.polEnableChk, self.lhSpin, self.polSettleSpin, self.roiX, self.roiY, self.roiW, self.roiH,
                  self.cropChk, self.cropX0, self.cropW, self.saveDirEdit, self.saveBaseEdit, self.newDirBtn,
                  self.pixelSizeSpin):
            w.setEnabled(idle)
        self._update_gate_preview()
        self._update_power_preview()

        if busy:
            what = "Sweep running" if (self._sweep_thread and self._sweep_thread.isRunning()) else \
                ("Delay scan running" if (self._scan_thread and self._scan_thread.isRunning()) else "Live")
        else:
            parts = []
            parts.append("Camera ✓" if cam else "Camera –")
            parts.append(("Stage ✓" if stage_ok else "Stage (not homed)") if stage else "Stage –")
            parts.append("Wheel ✓" if self.wheelDev.is_connected else "Wheel –")
            parts.append("Gates " + "".join("F" if self.gateDevs["front"].is_connected else "-")
                         + "".join("B" if self.gateDevs["back"].is_connected else "-"))
            what = "   ".join(parts)
            if self._zombies:
                what += f"   ({len(self._zombies)} terminated call(s) still pending)"
        if not self._stalled:  # the watchdog warning stays until the stuck call moves on
            self.stateLbl.setText(what)
            self.stateLbl.setStyleSheet("")

    # =================================================================== shutdown
    def shutdown(self) -> None:
        self._watchdog.stop()
        for th in (self._live_thread, self._scan_thread, self._sweep_thread):
            if th is not None and th.isRunning():
                th.stop()
        # terminated threads stuck in a driver call: park them (a running QThread must not be destroyed)
        wait_threads([th for _n, th in self._zombies], 1.0)
        for th, ctrl in list(self._gate_ramps.values()):
            ctrl.abort()
        wait_threads([t for t in (self._live_thread, self._scan_thread, self._sweep_thread) if t is not None]
                     + [t for t, _c in self._gate_ramps.values()] + list(self._aux_threads), 15.0)
        for dev in (self.cameraDev, self.delayDev, self.wheelDev, self.polDev, *self.gateDevs.values()):
            try:
                dev.shutdown()
            except Exception:
                pass
