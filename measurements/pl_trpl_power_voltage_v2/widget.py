"""PL + TRPL power/voltage dependence widget, v2 (pyqtgraph).

Built on the current PL power/gate GUI (measurements/pl_power_voltage_widget.py):
every PL setting, calibration mode (single / dual / HWP+ND / angle list), sweep
mode (power-outer, voltage-outer, grid) and Andor option is the same.  Added:

* pyqtgraph plotting (AndorLiveViewWidgetV2(match_v1=True), linecut / grid maps);
* SPAD TRPL over the pSPAD server, SPAD23 or SPAD93 ("Detector": Auto detects it on
  Connect from the number of pixel lines; detectors.py): count map, Histogram and
  Lifetime tabs (that detector's gui_v2 widgets and analysis fit, t0 / pre-t0 /
  fix-offset options);
* Measure: PL only, TRPL only or PL + TRPL (acquired in parallel at each point);
* Connect / Disconnect for every device; every device call runs in a worker
  thread (devices.py), so the GUI never blocks on hardware.
"""

import csv
import dataclasses
import os
import shutil
import re
import threading
import time
import types as _types
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import measurements.pl_trpl_power_voltage_v2  # noqa: F401  (pins PyQt5 for pyqtgraph, sets pg options)

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from andor.andor_wrapper import AndorSystem
from andor.gui.workers import LiveAcqThread, SingleAcqThread
from andor.gui_v2.live_view_widget_v2 import AndorLiveViewWidgetV2
from rot.rot_wrapper import MotionController, RotationStage, list_kinesis_serials
from rot.gui.workers import HomeThread, MoveThread

from keithley.keithley_wrapper import KEITHLEY_V_LIMIT, KeithleySMU, SweepController, make_resource_manager

from measurements.pl_trpl_power_voltage_v2.config import (
    DEFAULT_ACQ_NUMBER,
    DEFAULT_BACK_RESOURCE,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_FRONT_RESOURCE,
    DEFAULT_GATE_ICOMP_NA,
    DEFAULT_GATE_VSET,
    DEFAULT_GRATING,
    DEFAULT_LINECUT_ROW,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_POWER_CALIB_PATH,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_READOUT_RATE,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_SPEC_INDEX,
    DEFAULT_STAGE_ACCEL,
    DEFAULT_STAGE_SCALE,
    DEFAULT_STAGE_B_SERIAL,
    DEFAULT_STAGE_SERIAL,
    DEFAULT_SWEEP_ICOMP_NA,
    DEFAULT_SWEEP_STEP,
    DEFAULT_SWEEP_STEPS,
    DEFAULT_SWEEP_V0,
    POLL_ERROR_LIMIT,
    POLL_TIMEOUT_MS,
    RAMP_DWELL_S,
    RAMP_STEP_V,
    DATA_DIR,
    SWEEP_SETTLE_MS,
    ZERO_V_EPS,
    ZERO_V_EXTRA_SETTLE_MS,
    VISA_DLL,
    DEFAULT_TCSPC_HOST,
    DEFAULT_TCSPC_PORT,
    DEFAULT_TCSPC_ACQ_S,
    DEFAULT_TCSPC_BIN_PS,
    DEFAULT_HIST_START_NS,
    DEFAULT_HIST_END_NS,
    DEFAULT_HIST_REF_NS,
    SPAD_LIVE_INTEGRATION_MS,
    SPAD_SWEEP_COUNT_MS,
    SPAD_DETECTORS,
    DETECTOR_AUTO,
    DETECTOR_SETTINGS,
    DETECTOR_SETTINGS_KEY,
    SPAD_VIEW_KEY,
    WLCAL_PATH_KEY,
    TRPL_ALIGN_MS,
    CAM_POLL_S,
    SPEC_POLL_S,
    STAGE_POLL_S,
    GATE_POLL_S,
    GATE_READ_TIMEOUT_MS,
    GATE_RECONNECT_S,
    LONG_RAMP_STEPS,
    QUIET_WAIT_MAX_S,
    MEASURE_PL,
    MEASURE_TRPL,
    MEASURE_BOTH,
    MEASURE_MODES,
    DEFAULT_FLIPPER_SERIAL,
    FLIPPER_SETTLE_MS,
    FLIPPER_POLL_S,
    FLIPPER_SETTINGS,
    PUMP2_ON_DEFAULT,
    PUMP2_TAGS,
)
from measurements.dual_wheel_power_calibration import (
    DualWheelPowerEntry,
    DualWheelPowerData,
    HWPNDCalibData,
    is_dual_wheel_calibration,
    is_hwp_nd_calibration,
    load_dual_wheel_calibration,
    load_hwp_nd_calibration,
    compute_hwp_nd_entries,
)
from measurements.wheel_hwp_power_calib_v2 import calib as hwp_planner
from measurements.power_calibration import (
    PowerCalibEntry,
    entry_priority,
    load_power_calibration,
    power_key,
)
from measurements.pl_trpl_power_voltage_v2.devices import (
    CONNECTED,
    CONNECTING,
    DISCONNECTED,
    CallThread,
    CameraDevice,
    FlipperDevice,
    GateDevice,
    LatestJobThread,
    SpectrographDevice,
    StageDevice,
    wait_threads,
)
from measurements.pl_trpl_power_voltage_v2.plots import GateVIPlot, LifetimeMapPlot, LinecutMapPlot
from measurements.pl_trpl_power_voltage_v2.scan_time import (
    CSV_HEADER as SCAN_TIMING_CSV_HEADER,
    Schedule,
    ScanSpec,
    ScanTimeWidget,
    ScanTracker,
    TimingModel,
    fmt_duration,
    simulate,
)
from measurements.pl_trpl_power_voltage_v2.workers import GateRampThread, PowerVoltageSweepThread
from measurements.pl_trpl_power_voltage_v2 import trpl_dynamic as tdyn
from measurements.pl_trpl_power_voltage_v2 import detectors as spad_det
from measurements.pl_trpl_power_voltage_v2.detectors import DetectWorker, is_count_snapshot, is_trpl_snapshot
from measurements.pl_trpl_power_voltage_v2 import spectral as spad_spec
from measurements.pl_trpl_power_voltage_v2.spectral import (
    MODE_2D,
    MODE_SPECTRAL,
    VIEW_MODES,
    SpectralContext,
    SpectralHistPage,
    SpectralLivePanel,
)
# geometry-free analysis and plots (same in the spad23 and spad93 packages); the pixel maps, the 2D fit
# and the clients come from the active detector's profile (self._det)
from spad23.spad23_analysis import T0_REF_BINS, align_first_peaks, peak_display_offset, pre_pulse_window, t0_ref_index
from spad23.gui_v2.plots import HistogramPlot, RadialFitPlot, TracePlot, trace_color
from spad23.gui_v2.workers import FitAllWorker

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None


class PLTRPLPowerVoltageWidgetV2(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)
        self.stage: Optional[RotationStage] = None
        self.stage_b: Optional[RotationStage] = None

        self._live_thread: Optional[LiveAcqThread] = None
        self._accum_thread: Optional[SingleAcqThread] = None
        self._sweep_thread: Optional[PowerVoltageSweepThread] = None
        # per-stage motion (v1 shared one busy flag, so stage B never homed after A)
        self._stage_threads: Dict[str, Optional[QtCore.QThread]] = {"A": None, "B": None}
        self._stage_controllers: Dict[str, Optional[MotionController]] = {"A": None, "B": None}
        self._stage_busy_by: Dict[str, bool] = {"A": False, "B": False}
        self._busy = False
        self._aborting = False
        self._live_active = False

        self._calib_data = None
        self._dual_calib_data: Optional[DualWheelPowerData] = None
        self._hwp_nd_calib_data: Optional[HWPNDCalibData] = None
        self._series_entries: Dict[str, List[PowerCalibEntry]] = {}
        self._dual_entries: List[DualWheelPowerEntry] = []
        self._power_display: Dict[float, dict] = {}
        self._power_list_keys: List[float] = []
        self._power_view_keys: List[float] = []
        self._pv_images: Dict[float, Dict[float, dict]] = {}
        self._voltage_keys: List[float] = []
        self._voltage_labels: List[str] = []
        self._voltage_pairs: Dict[float, Tuple[Optional[float], Optional[float]]] = {}
        # Grid (fix-A, sweep-B) sweep state
        self._grid_active = False
        self._a_keys: List[float] = []
        self._a_labels: List[str] = []
        self._b_keys: List[float] = []
        self._b_labels: List[str] = []
        # power -> a_key -> b_key -> {"rank", "raw"}
        self._grid_images: Dict[float, Dict[float, Dict[float, dict]]] = {}
        self._taken_powers = set()
        self._taken_entries = set()
        self._power_done_counts: Dict[float, int] = {}
        self._power_target_counts: Dict[float, int] = {}
        self._total_entries = 0
        self._angle_list_mode = False
        self._power_plan: List[hwp_planner.PlanRow] = []  # power list (HWP+ND): solved rows, list order
        self._power_list_bad: List[str] = []  # tokens of the power list that could not be read
        self._session_saved = False
        self._sweep_dyn = None  # DynamicSettings of the running sweep (None = fixed TRPL time)
        self._dyn_results: List[dict] = []  # dynamic TRPL: one dict per acquisition (trpl_done "dyn")
        self._file_counters: Dict[str, int] = {}
        self._save_dir = None
        self._calib_unit = "nW"
        self._calib_scale = 1e-9
        self._calib_mode = "single"

        self._crop_top = int(DEFAULT_CROP_TOP)
        self._crop_bottom = int(DEFAULT_CROP_BOTTOM)
        self._crop_left = int(DEFAULT_CROP_LEFT)
        self._crop_right = int(DEFAULT_CROP_RIGHT)

        self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        self._preamp_gain_label = str(DEFAULT_PREAMP_GAIN)
        self._output_amp_label = str(DEFAULT_OUTPUT_AMP)

        # VISA resource manager: created lazily in a gate worker thread
        self.rm = None
        self._rm_lock = threading.Lock()

        self.front_dev: Optional[KeithleySMU] = None
        self.back_dev: Optional[KeithleySMU] = None
        self._front_output_on = False
        self._back_output_on = False
        self._front_ramp_thread: Optional[GateRampThread] = None
        self._back_ramp_thread: Optional[GateRampThread] = None
        self._front_ramp_controller: Optional[SweepController] = None
        self._back_ramp_controller: Optional[SweepController] = None

        # SPAD (SPAD23 or SPAD93) / TRPL
        self._det_settings = QtCore.QSettings(DETECTOR_SETTINGS[0], DETECTOR_SETTINGS[1])
        self._det_choice = str(self._det_settings.value(DETECTOR_SETTINGS_KEY, DETECTOR_AUTO, type=str))
        if self._det_choice not in (DETECTOR_AUTO,) + tuple(SPAD_DETECTORS):
            self._det_choice = DETECTOR_AUTO
        last = str(self._det_settings.value(DETECTOR_SETTINGS_KEY + "_last", SPAD_DETECTORS[0], type=str))
        self._det = spad_det.profile(last if last in SPAD_DETECTORS else SPAD_DETECTORS[0])
        self._det_detected = False  # the active detector was confirmed by the server on Connect
        self._detect_worker: Optional[DetectWorker] = None
        self._detect_cancelled = False
        self._spad_live_enabled = True
        self._spad_last_count_snapshot = None  # CountSnapshot of the active detector
        self._spad_count_times: List[float] = []
        self._spad_latest_hist_snapshot = None  # TrplSnapshot of the active detector
        self._spad_selected_pixel_live = self._det.center_pixel
        self._spad_selected_hist_pixels: set = {self._det.center_pixel}
        self._spad_excluded_fit_pixels: set = set()
        # (power key, voltage key, gate-A key or None) -> {"time_ns", "counts", "filepath"}
        self._spad_trpl_data: Dict[Tuple[float, float, Optional[float]], dict] = {}
        self._hist_fit_result = None  # FitResult of the detector's analysis module
        self._spad_fit_x_um, self._spad_fit_y_um = self._det.pixel_x_um.copy(), self._det.pixel_y_um.copy()
        self._spad_map_rates = np.zeros(self._det.num_pixels, dtype=np.float64)
        self._hist_t0_cache: dict = {}
        # time-zero aligned view: t = raw - offset (time zero at 0 ns).  The Start / End / Reference / pre-t0
        # times are in the coordinates of the view; they are converted when it switches between raw times and
        # times after time zero (_hist_rel), so they stay on the same data.
        self._hist_rel = False
        self._hist_last_off: Optional[float] = None
        self._hist_view_cache = None  # (snapshot, align, t_ns, counts)
        # live TRPL updates (sweep or Take Histogram, every 0.4 s): only the newest is drawn, at most once per
        # max(LIVE_HIST_MIN_S, LIVE_HIST_COST_X x the last redraw time), so a slow redraw (93 px x 13k bins with
        # time-zero alignment ~1 s) can never queue up snapshots (that froze the GUI at 6.7 GB on 2026-11-11)
        self._live_hist_timer = QtCore.QTimer(self)
        self._live_hist_timer.setSingleShot(True)
        self._live_hist_timer.timeout.connect(self._flush_live_hist)
        self._live_hist_next = 0.0  # time.monotonic() before which no live redraw starts
        self._pre_t0_auto = True
        # dark reference (beam blocked): an alternative to the pre-t0 offset for the 2D fit
        self._dark_ref = None  # DarkReference of the active detector's analysis module
        self._dark_pending = False  # the running Take-Histogram acquisition is a dark reference
        # sigma^2(t) of the histogram (on request only, never during a scan): (key, snapshot, FitAllResult, t_ns)
        self._sigma_result = None
        self._sigma_worker: Optional[FitAllWorker] = None
        self._sigma_job = 0
        self._sweep_dark_file: Optional[str] = None  # copy of the dark reference in the current sweep folder
        # spectral mode (the SPAD's center line = wavelength axis, spectral.py): view and fits only
        mode = str(self._det_settings.value(SPAD_VIEW_KEY, MODE_2D, type=str))
        self._spad_view_mode = mode if mode in (MODE_2D, MODE_SPECTRAL) else MODE_2D
        self._spec_geom = spad_spec.geometry(self._det.key)
        self._wl_calib = None  # WavelengthCalibration of the active detector (loaded JSON)
        self._wl_calib_path = ""
        self._spec_lines: set = {self._spec_geom.center_line}  # lines selected in the Histogram tab
        self._spec_active_line = self._spec_geom.center_line  # live linecut, spectrum marker, Lifetime tab
        self._sweep_wlcal_file: Optional[str] = None  # copy of the calibration JSON in the current sweep folder
        self._spad_popup_pending = False
        self._spad_align_after_connect = False

        # Device controllers: all hardware calls run in their worker threads
        self._andor_lock = threading.RLock()  # AndorSystem has no lock: camera + spectrograph share one
        self.camDev = CameraDevice(self.cam, self._camera_setup_io, CAM_POLL_S, io_lock=self._andor_lock, parent=self)
        self.specDev = SpectrographDevice(self.cam, int(DEFAULT_SPEC_INDEX), DEFAULT_SLIT_ID, SPEC_POLL_S,
                                          io_lock=self._andor_lock, parent=self)
        self.stageDevs = {
            "A": StageDevice("Stage A", self._make_stage, STAGE_POLL_S, parent=self),
            "B": StageDevice("Stage B", self._make_stage, STAGE_POLL_S, parent=self),
        }
        self.gateDevs = {
            which: GateDevice(
                f"{which.capitalize()} gate",
                lambda res, icomp, which=which: self._open_gate_io(which, res, icomp),
                max(POLL_TIMEOUT_MS, GATE_READ_TIMEOUT_MS), POLL_ERROR_LIMIT, GATE_POLL_S, GATE_RECONNECT_S,
                ramp_step_v=float(RAMP_STEP_V), ramp_dwell_s=float(RAMP_DWELL_S), parent=self,
            )
            for which in ("front", "back")
        }
        self.spadDev = self._det.device_controller(DEFAULT_TCSPC_HOST, int(DEFAULT_TCSPC_PORT), self)
        self.spadDev.set_integration_ms(SPAD_LIVE_INTEGRATION_MS)
        # pump-2 flipper (dI/dP); "pump 2 passes at" is shared with the flipper GUI (QSettings)
        self.flipperDev = FlipperDevice(FLIPPER_POLL_S, parent=self)
        self._flipper_settings = QtCore.QSettings(FLIPPER_SETTINGS[0], FLIPPER_SETTINGS[1])
        self._flipper_position: Optional[int] = None
        self._bg_threads = set()
        self._pending_after_cam: List[Callable[[], None]] = []
        self._pending_after_spec_off: List[Callable[[], None]] = []
        self._pending_after_stage_a: List[Callable[[], None]] = []
        self._quiet_waits = 0
        # 2D Gaussian fit of the Histogram tab: off the GUI thread, newest request wins
        self._fit_worker = LatestJobThread(self)
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.start()
        self._fit_job = 0
        self._fit_ctx: Optional[dict] = None
        self._hist_dirty = False
        self._lifetime_dirty = False

        # scan time estimate (status bar, bottom right)
        self._timing_model = TimingModel.load()
        self._gate_last_v: Dict[str, Optional[float]] = {"front": None, "back": None}
        # gate status lines: last ramp / poll message and last V, I reading
        self._gate_state: Dict[str, str] = {"front": "Disconnected", "back": "Disconnected"}
        self._gate_vi: Dict[str, Optional[Tuple[float, float]]] = {"front": None, "back": None}
        self._stage_velocity: Dict[str, Optional[Tuple[float, float]]] = {"A": None, "B": None}
        self._cam_readout_s: Dict[str, float] = {}  # Andor SDK readout time per readout-rate label
        # wavelength axis: (grating, center nm) it was read at, a fetch in flight, retry time, warned once
        self._wl_read_at: Optional[Tuple[object, float]] = None
        self._wl_fetch_pending = False
        self._wl_retry_at = 0.0
        self._wl_warned = False
        self._scan_sched_cache: Optional[Tuple[tuple, Schedule]] = None
        self._scan_tracker: Optional[ScanTracker] = None
        self._scan_csv_path: Optional[str] = None

        self._build_ui()
        self._wire_devices()

    @property
    def _stage_busy(self) -> bool:
        return any(self._stage_busy_by.values())

    def _threads_running(self) -> bool:
        live = self._live_thread is not None and self._live_thread.isRunning()
        accum = self._accum_thread is not None and self._accum_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return live or accum or sweep or self.spadDev.trpl_running

    def _gate_threads_running(self) -> bool:
        front = self._front_ramp_thread is not None and self._front_ramp_thread.isRunning()
        back = self._back_ramp_thread is not None and self._back_ramp_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return front or back or sweep

    def _is_busy(self) -> bool:
        return self._busy or self._stage_busy or self._threads_running() or self._gate_threads_running()

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self._build_dashboard(vbox)
        self._build_plots(vbox)

        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")
        self.scanTimeWidget = ScanTimeWidget(self)
        self._status_bar.addPermanentWidget(self.scanTimeWidget)

        self._wire_signals()
        self._update_gate_sweep_controls()
        self._wire_dyn_controls()
        self._wire_scan_estimate()

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    def _build_dashboard(self, parent_layout):
        box = QtWidgets.QGroupBox("Dashboard")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # Consolas 9 pt matches the 9 pt Segoe UI text (the system fixed font can be smaller)
        if "Consolas" in QtGui.QFontDatabase().families():
            mono = QtGui.QFont("Consolas", 9)
        else:
            mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            mono.setPointSize(9)

        # one Connect / Disconnect button per device
        self.camConnBtn = self._make_conn_btn()
        self.specConnBtn = self._make_conn_btn()
        self.specConnBtn.setToolTip("Needs the Andor camera (pixel calibration of the wavelength axis).")
        self.stageConnBtn = self._make_conn_btn()
        self.stageBConnBtn = self._make_conn_btn()
        self.frontConnBtn = self._make_conn_btn()
        self.backConnBtn = self._make_conn_btn()
        self.tcspcConnBtn = self._make_conn_btn()

        # Andor line
        self.camDot = self._make_dot()
        self.camStatusLbl = QtWidgets.QLabel("Andor: disconnected")
        self.camStatusLbl.setFont(mono)

        self.exposureSpin = QtWidgets.QDoubleSpinBox()
        self.exposureSpin.setDecimals(3)
        self.exposureSpin.setRange(0.001, 1.0e7)
        self.exposureSpin.setValue(DEFAULT_EXPOSURE_MS)
        self.exposureSpin.setFixedWidth(90)

        self.accumSpin = QtWidgets.QSpinBox()
        self.accumSpin.setRange(1, 100000)
        self.accumSpin.setValue(DEFAULT_ACQ_NUMBER)
        self.accumSpin.setFixedWidth(70)

        self.readoutRateCombo = QtWidgets.QComboBox()
        self.readoutRateCombo.setEditable(False)
        self.readoutRateCombo.addItem(str(DEFAULT_READOUT_RATE))
        self.readoutRateCombo.setFixedWidth(150)

        self.preampCombo = QtWidgets.QComboBox()
        self.preampCombo.setEditable(False)
        self.preampCombo.addItem(str(DEFAULT_PREAMP_GAIN))
        self.preampCombo.setFixedWidth(70)

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.camDot)
        line1.addWidget(QtWidgets.QLabel("Andor:"))
        line1.addWidget(self.camStatusLbl, 1)
        line1.addWidget(self.camConnBtn)
        line1.addStretch()
        line1.addWidget(QtWidgets.QLabel("Exp (ms):"))
        line1.addWidget(self.exposureSpin)
        line1.addWidget(QtWidgets.QLabel("Acq N:"))
        line1.addWidget(self.accumSpin)
        line1.addWidget(QtWidgets.QLabel("Rate:"))
        line1.addWidget(self.readoutRateCombo)
        line1.addWidget(QtWidgets.QLabel("Preamp:"))
        line1.addWidget(self.preampCombo)

        grid.addLayout(line1, 0, 0, 1, 1)

        # Spectrograph line
        self.specDot = self._make_dot()
        self.specStatusLbl = QtWidgets.QLabel("Spectrograph: disconnected")
        self.specStatusLbl.setFont(mono)

        self.gratingSpin = QtWidgets.QSpinBox()
        self.gratingSpin.setRange(0, 10)
        self.gratingSpin.setValue(int(DEFAULT_GRATING))
        self.gratingSpin.setFixedWidth(60)

        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setDecimals(3)
        self.centerSpin.setRange(0.0, 20000.0)
        self.centerSpin.setValue(float(DEFAULT_CENTER_WL_NM))
        self.centerSpin.setFixedWidth(90)

        self.slitSpin = QtWidgets.QDoubleSpinBox()
        self.slitSpin.setDecimals(2)
        self.slitSpin.setRange(0.0, 5000.0)
        self.slitSpin.setValue(float(DEFAULT_SLIT_UM))
        self.slitSpin.setFixedWidth(80)

        self.applySpecBtn = QtWidgets.QPushButton("Apply Spec")
        self.applySpecBtn.setFixedWidth(120)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.specDot)
        line2.addWidget(QtWidgets.QLabel("Spectrograph:"))
        line2.addWidget(self.specStatusLbl, 1)
        line2.addWidget(self.specConnBtn)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Grating:"))
        line2.addWidget(self.gratingSpin)
        line2.addWidget(QtWidgets.QLabel("Center (nm):"))
        line2.addWidget(self.centerSpin)
        line2.addWidget(QtWidgets.QLabel("Slit (um):"))
        line2.addWidget(self.slitSpin)
        line2.addWidget(self.applySpecBtn)

        grid.addLayout(line2, 1, 0, 1, 1)

        # Stage A line
        self.stageDot = self._make_dot()
        self.stageStatusLbl = QtWidgets.QLabel("Stage A: disconnected")
        self.stageStatusLbl.setFont(mono)

        self.stageSerialCombo = QtWidgets.QComboBox()
        self.stageSerialCombo.setEditable(True)
        self.stageSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_SERIAL:
            self.stageSerialCombo.addItem(DEFAULT_STAGE_SERIAL)

        self.stageHomeBtn = QtWidgets.QPushButton("Home A")
        self.stageHomeBtn.setFixedWidth(90)
        self.stageMoveBtn = QtWidgets.QPushButton("Move A")
        self.stageMoveBtn.setFixedWidth(90)
        self.stageDetectBtn = QtWidgets.QPushButton("Search")
        self.stageDetectBtn.setFixedWidth(80)

        self.stageTargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageTargetSpin.setDecimals(3)
        self.stageTargetSpin.setRange(0.0, 360.0)
        self.stageTargetSpin.setValue(0.0)
        self.stageTargetSpin.setFixedWidth(90)

        line3 = QtWidgets.QHBoxLayout()
        line3.addWidget(self.stageDot)
        line3.addWidget(QtWidgets.QLabel("Stage A:"))
        line3.addWidget(self.stageStatusLbl, 1)
        line3.addWidget(self.stageConnBtn)
        line3.addStretch()
        line3.addWidget(QtWidgets.QLabel("Serial:"))
        line3.addWidget(self.stageSerialCombo)
        line3.addWidget(self.stageDetectBtn)
        line3.addWidget(self.stageHomeBtn)
        line3.addWidget(QtWidgets.QLabel("Target (deg):"))
        line3.addWidget(self.stageTargetSpin)
        line3.addWidget(self.stageMoveBtn)

        grid.addLayout(line3, 2, 0, 1, 1)

        # Stage B line
        self.stageBDot = self._make_dot()
        self.stageBStatusLbl = QtWidgets.QLabel("Stage B: disconnected")
        self.stageBStatusLbl.setFont(mono)

        self.stageBSerialCombo = QtWidgets.QComboBox()
        self.stageBSerialCombo.setEditable(True)
        self.stageBSerialCombo.setFixedWidth(140)
        if DEFAULT_STAGE_B_SERIAL:
            self.stageBSerialCombo.addItem(DEFAULT_STAGE_B_SERIAL)

        self.stageBHomeBtn = QtWidgets.QPushButton("Home B")
        self.stageBHomeBtn.setFixedWidth(90)
        self.stageBMoveBtn = QtWidgets.QPushButton("Move B")
        self.stageBMoveBtn.setFixedWidth(90)
        self.stageBDetectBtn = QtWidgets.QPushButton("Search")
        self.stageBDetectBtn.setFixedWidth(80)

        self.stageBTargetSpin = QtWidgets.QDoubleSpinBox()
        self.stageBTargetSpin.setDecimals(3)
        self.stageBTargetSpin.setRange(0.0, 360.0)
        self.stageBTargetSpin.setValue(0.0)
        self.stageBTargetSpin.setFixedWidth(90)

        line4 = QtWidgets.QHBoxLayout()
        line4.addWidget(self.stageBDot)
        line4.addWidget(QtWidgets.QLabel("Stage B:"))
        line4.addWidget(self.stageBStatusLbl, 1)
        line4.addWidget(self.stageBConnBtn)
        line4.addStretch()
        line4.addWidget(QtWidgets.QLabel("Serial:"))
        line4.addWidget(self.stageBSerialCombo)
        line4.addWidget(self.stageBDetectBtn)
        line4.addWidget(self.stageBHomeBtn)
        line4.addWidget(QtWidgets.QLabel("Target (deg):"))
        line4.addWidget(self.stageBTargetSpin)
        line4.addWidget(self.stageBMoveBtn)

        grid.addLayout(line4, 3, 0, 1, 1)

        # Front gate line
        self.frontGateDot = self._make_dot()
        self.frontGateStatusLbl = QtWidgets.QLabel("Front gate: disconnected")
        self.frontGateStatusLbl.setFont(mono)

        self.frontResourceEdit = QtWidgets.QLineEdit(DEFAULT_FRONT_RESOURCE)
        self.frontResourceEdit.setMinimumWidth(340)

        line5 = QtWidgets.QHBoxLayout()
        line5.addWidget(self.frontGateDot)
        line5.addWidget(QtWidgets.QLabel("Front gate:"))
        line5.addWidget(self.frontGateStatusLbl, 1)
        line5.addWidget(self.frontConnBtn)
        line5.addSpacing(12)
        line5.addWidget(QtWidgets.QLabel("Resource:"))
        line5.addWidget(self.frontResourceEdit)

        self.frontVsetSpin = QtWidgets.QDoubleSpinBox()
        self.frontVsetSpin.setDecimals(6)
        self.frontVsetSpin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.frontVsetSpin.setValue(DEFAULT_GATE_VSET)
        self.frontVsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.frontVsetSpin.setFixedWidth(110)

        self.frontIcompSpin = QtWidgets.QDoubleSpinBox()
        self.frontIcompSpin.setDecimals(3)
        self.frontIcompSpin.setRange(0.0, 1e9)
        self.frontIcompSpin.setValue(DEFAULT_GATE_ICOMP_NA)
        self.frontIcompSpin.setFixedWidth(110)

        self.frontApplyBtn = QtWidgets.QPushButton("Apply Front")
        self.frontApplyBtn.setFixedWidth(120)

        line6 = QtWidgets.QHBoxLayout()
        line6.addStretch()
        line6.addWidget(QtWidgets.QLabel("V set (V):"))
        line6.addWidget(self.frontVsetSpin)
        line6.addWidget(QtWidgets.QLabel("I comp (nA):"))
        line6.addWidget(self.frontIcompSpin)
        line6.addWidget(self.frontApplyBtn)

        grid.addLayout(line5, 4, 0, 1, 1)
        grid.addLayout(line6, 5, 0, 1, 1)

        # Back gate line
        self.backGateDot = self._make_dot()
        self.backGateStatusLbl = QtWidgets.QLabel("Back gate: disconnected")
        self.backGateStatusLbl.setFont(mono)

        self.backResourceEdit = QtWidgets.QLineEdit(DEFAULT_BACK_RESOURCE)
        self.backResourceEdit.setMinimumWidth(340)

        line7 = QtWidgets.QHBoxLayout()
        line7.addWidget(self.backGateDot)
        line7.addWidget(QtWidgets.QLabel("Back gate:"))
        line7.addWidget(self.backGateStatusLbl, 1)
        line7.addWidget(self.backConnBtn)
        line7.addSpacing(12)
        line7.addWidget(QtWidgets.QLabel("Resource:"))
        line7.addWidget(self.backResourceEdit)

        self.backVsetSpin = QtWidgets.QDoubleSpinBox()
        self.backVsetSpin.setDecimals(6)
        self.backVsetSpin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.backVsetSpin.setValue(DEFAULT_GATE_VSET)
        self.backVsetSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.backVsetSpin.setFixedWidth(110)

        self.backIcompSpin = QtWidgets.QDoubleSpinBox()
        self.backIcompSpin.setDecimals(3)
        self.backIcompSpin.setRange(0.0, 1e9)
        self.backIcompSpin.setValue(DEFAULT_GATE_ICOMP_NA)
        self.backIcompSpin.setFixedWidth(110)

        self.backApplyBtn = QtWidgets.QPushButton("Apply Back")
        self.backApplyBtn.setFixedWidth(120)

        line8 = QtWidgets.QHBoxLayout()
        line8.addStretch()
        line8.addWidget(QtWidgets.QLabel("V set (V):"))
        line8.addWidget(self.backVsetSpin)
        line8.addWidget(QtWidgets.QLabel("I comp (nA):"))
        line8.addWidget(self.backIcompSpin)
        line8.addWidget(self.backApplyBtn)

        grid.addLayout(line7, 6, 0, 1, 1)
        grid.addLayout(line8, 7, 0, 1, 1)

        # TCSPC detector line (SPAD23 or SPAD93 over the pSPAD server)
        self.detectorCombo = QtWidgets.QComboBox()
        self.detectorCombo.addItems([DETECTOR_AUTO] + list(SPAD_DETECTORS))
        self.detectorCombo.setCurrentText(self._det_choice)
        self.detectorCombo.setFixedWidth(90)
        self.detectorCombo.setToolTip(
            "SPAD detector behind the pSPAD server.\n"
            "Auto: on Connect, one short count frame is read and its pixel lines counted (23 = SPAD23,\n"
            "93 = SPAD93); the count maps and the 2D fit are rebuilt for that detector.\n"
            "SPAD23 / SPAD93: use that one (Connect still checks the server and asks if it disagrees).\n"
            "Locked while connected or acquiring.")
        self.tcspcDot = self._make_dot()
        self.tcspcStatusLbl = QtWidgets.QLabel("TCSPC detector: disconnected")
        self.tcspcStatusLbl.setFont(mono)
        self.tcspcAcqSpin = QtWidgets.QDoubleSpinBox()
        self.tcspcAcqSpin.setDecimals(2)
        self.tcspcAcqSpin.setRange(0.1, 3600.0)
        self.tcspcAcqSpin.setValue(float(DEFAULT_TCSPC_ACQ_S))
        self.tcspcAcqSpin.setFixedWidth(90)
        self.tcspcBinSpin = QtWidgets.QSpinBox()
        self.tcspcBinSpin.setRange(1, 50000)
        self.tcspcBinSpin.setValue(int(DEFAULT_TCSPC_BIN_PS))
        self.tcspcBinSpin.setFixedWidth(90)
        self.tcspcCalibAlignBtn = QtWidgets.QPushButton("Calibrate + Align")
        self.tcspcCalibAlignBtn.setFixedWidth(150)
        self.stopCountsBtn = QtWidgets.QPushButton("Pause Live Counts")
        self.stopCountsBtn.setFixedWidth(150)

        line9 = QtWidgets.QHBoxLayout()
        line9.addWidget(self.tcspcDot)
        line9.addWidget(QtWidgets.QLabel("TCSPC detector:"))
        line9.addWidget(self.detectorCombo)
        line9.addWidget(self.tcspcStatusLbl, 1)
        line9.addWidget(self.tcspcConnBtn)
        line9.addStretch()
        line9.addWidget(QtWidgets.QLabel("Acq (s):"))
        line9.addWidget(self.tcspcAcqSpin)
        line9.addWidget(QtWidgets.QLabel("Bin (ps):"))
        line9.addWidget(self.tcspcBinSpin)
        line9.addWidget(self.tcspcCalibAlignBtn)
        line9.addWidget(self.stopCountsBtn)
        self.tcspcAcqSpin.setToolTip("TRPL acquisition time: the Histogram tab's Take histogram, and every sweep point\n"
                                     "when the sweep TRPL time is Fixed.")

        # Sweep TRPL time: Fixed (Acq above) or Dynamic (chunks until the brightest pixel's peak is high enough)
        self.trplTimeCombo = QtWidgets.QComboBox()
        self.trplTimeCombo.addItems(["Fixed", "Dynamic"])
        self.trplTimeCombo.setFixedWidth(90)
        self.trplTimeCombo.setToolTip(
            "Fixed: every sweep point takes 'Acq (s)'.\n"
            "Dynamic: TRPL is taken in chunks (first chunk, then steps) until the brightest pixel's peak,\n"
            "background subtracted, reaches the target; at the max time the point is flagged and the sweep\n"
            "moves on. Each spot's chunks are summed bin-for-bin into its TRPL_*.csv, each chunk is also saved\n"
            "in trpl_chunks/, and every spot's time / peak / flag goes to dynamic_trpl_log.csv.\n"
            "Background: the dark reference (scaled to the time so far), else each pixel's pre-t0 level.\n"
            "dI/dP: pump 2 on repeats pump 2 off's chunks (same time).")
        self.dynTargetSpin = QtWidgets.QSpinBox()
        self.dynTargetSpin.setRange(1, 100_000_000)
        self.dynTargetSpin.setFixedWidth(80)
        self.dynTargetSpin.setToolTip("Stop when max over pixels and bins of (counts - background) reaches this")
        self.dynFirstSpin = QtWidgets.QDoubleSpinBox()
        self.dynFirstSpin.setDecimals(1)
        self.dynFirstSpin.setRange(0.1, 36000.0)
        self.dynFirstSpin.setFixedWidth(80)
        self.dynFirstPlChk = QtWidgets.QCheckBox("= PL time")
        self.dynFirstPlChk.setToolTip("First chunk = exposure × accumulations (when the sweep measures PL)")
        self.dynStepSpin = QtWidgets.QDoubleSpinBox()
        self.dynStepSpin.setDecimals(1)
        self.dynStepSpin.setRange(0.1, 36000.0)
        self.dynStepSpin.setFixedWidth(80)
        self.dynStepSpin.setToolTip("Length of every further chunk")
        self.dynMaxSpin = QtWidgets.QDoubleSpinBox()
        self.dynMaxSpin.setDecimals(1)
        self.dynMaxSpin.setRange(0.1, 360000.0)
        self.dynMaxSpin.setFixedWidth(90)
        self.dynMaxSpin.setToolTip("Give up at this total (the last chunk is shortened to end exactly here);\n"
                                   "the point is flagged 'below target'")
        ds = self._det_settings
        self.trplTimeCombo.setCurrentIndex(1 if str(ds.value("trpl_time_mode", "fixed", type=str)) == "dynamic" else 0)
        self.dynTargetSpin.setValue(int(ds.value("trpl_dyn_target", 500, type=int)))
        self.dynFirstSpin.setValue(float(ds.value("trpl_dyn_first_s", 60.0, type=float)))
        self.dynFirstPlChk.setChecked(bool(ds.value("trpl_dyn_first_pl", True, type=bool)))
        self.dynStepSpin.setValue(float(ds.value("trpl_dyn_step_s", 60.0, type=float)))
        self.dynMaxSpin.setValue(float(ds.value("trpl_dyn_max_s", 600.0, type=float)))
        self.dynCtrlWidget = QtWidgets.QWidget()
        dyn_row = QtWidgets.QHBoxLayout(self.dynCtrlWidget)
        dyn_row.setContentsMargins(0, 0, 0, 0)
        for w in (QtWidgets.QLabel("Target peak (counts):"), self.dynTargetSpin, QtWidgets.QLabel("First (s):"),
                  self.dynFirstSpin, self.dynFirstPlChk, QtWidgets.QLabel("Step (s):"), self.dynStepSpin,
                  QtWidgets.QLabel("Max (s):"), self.dynMaxSpin):
            dyn_row.addWidget(w)
        line9b = QtWidgets.QHBoxLayout()
        line9b.addWidget(QtWidgets.QLabel("Sweep TRPL time:"))
        line9b.addWidget(self.trplTimeCombo)
        line9b.addWidget(self.dynCtrlWidget)
        line9b.addStretch()
        tcspc_rows = QtWidgets.QVBoxLayout()
        tcspc_rows.setSpacing(6)
        tcspc_rows.addLayout(line9)
        tcspc_rows.addLayout(line9b)
        grid.addLayout(tcspc_rows, 8, 0, 1, 1)

        # Pump-2 flipper line (dI/dP mode)
        self.flipperDot = self._make_dot()
        self.flipperStatusLbl = QtWidgets.QLabel("Flipper: disconnected")
        self.flipperStatusLbl.setFont(mono)
        self.flipperConnBtn = self._make_conn_btn()
        self.flipperSerialEdit = QtWidgets.QLineEdit(str(DEFAULT_FLIPPER_SERIAL))
        self.flipperSerialEdit.setFixedWidth(110)
        self.pump2PosCombo = QtWidgets.QComboBox()
        self.pump2PosCombo.addItems(["Position 1", "Position 2"])
        self.pump2PosCombo.setToolTip("Which flipper position lets pump 2 through (shared with the flipper GUI)")
        on_pos = self._flipper_settings.value(FLIPPER_SETTINGS[2], PUMP2_ON_DEFAULT, type=int)
        self.pump2PosCombo.setCurrentIndex(1 if int(on_pos) == 2 else 0)
        self.pump2PosCombo.setFixedWidth(110)
        self.pump2OffBtn = QtWidgets.QPushButton("Pump 2 off")
        self.pump2OffBtn.setFixedWidth(100)
        self.pump2OnBtn = QtWidgets.QPushButton("Pump 2 on")
        self.pump2OnBtn.setFixedWidth(100)

        line10 = QtWidgets.QHBoxLayout()
        line10.addWidget(self.flipperDot)
        line10.addWidget(QtWidgets.QLabel("Pump-2 flipper:"))
        line10.addWidget(self.flipperStatusLbl, 1)
        line10.addWidget(self.flipperConnBtn)
        line10.addStretch()
        line10.addWidget(QtWidgets.QLabel("Serial:"))
        line10.addWidget(self.flipperSerialEdit)
        line10.addWidget(QtWidgets.QLabel("Pump 2 passes at:"))
        line10.addWidget(self.pump2PosCombo)
        line10.addWidget(self.pump2OffBtn)
        line10.addWidget(self.pump2OnBtn)
        grid.addLayout(line10, 9, 0, 1, 1)

        # Action line
        self.initBtn = QtWidgets.QPushButton("Connect All")
        self.initBtn.setFixedWidth(120)
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect All")
        self.disconnectBtn.setFixedWidth(120)
        self.homeOnConnectChk = QtWidgets.QCheckBox("Home stages on connect")
        self.homeOnConnectChk.setChecked(True)
        self.takeHistogramBtn = QtWidgets.QPushButton("Take Histogram")
        self.takeHistogramBtn.setFixedWidth(140)
        self.takeCurrentBtn = QtWidgets.QPushButton("Take Current Image")
        self.takeCurrentBtn.setFixedWidth(160)
        self.stopLiveBtn = QtWidgets.QPushButton("Stop Live")
        self.stopLiveBtn.setFixedWidth(120)
        self.sweepBtn = QtWidgets.QPushButton("Start Sweep")
        self.sweepBtn.setFixedWidth(140)
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.abortBtn.setFixedWidth(100)
        self.homeBothBtn = QtWidgets.QPushButton("Home Both Gates")
        self.homeBothBtn.setFixedWidth(150)
        self.homeBothBtn.setEnabled(False)

        line5 = QtWidgets.QHBoxLayout()
        line5.addWidget(self.homeOnConnectChk)
        line5.addStretch()
        line5.addWidget(self.initBtn)
        line5.addWidget(self.disconnectBtn)
        line5.addSpacing(10)
        line5.addWidget(self.homeBothBtn)
        line5.addSpacing(10)
        line5.addWidget(self.takeCurrentBtn)
        line5.addWidget(self.stopLiveBtn)
        line5.addWidget(self.takeHistogramBtn)
        line5.addWidget(self.sweepBtn)
        line5.addWidget(self.abortBtn)

        grid.addLayout(line5, 10, 0, 1, 1)

        parent_layout.addWidget(box)

    def _build_session_controls(self, parent_layout):
        box = QtWidgets.QGroupBox("Session / Calibration")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.saveNameEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveNameEdit.setPlaceholderText("subfolder name under the data folder")
        self.saveNameNowBtn = QtWidgets.QPushButton("Refresh Time")
        self.saveNameNowBtn.setFixedWidth(110)
        self.saveNameNowBtn.setToolTip("Update the date/time in the subfolder name to now")

        self.calibPathEdit = QtWidgets.QLineEdit(DEFAULT_POWER_CALIB_PATH)
        self.calibBrowseBtn = QtWidgets.QPushButton("Browse")
        self.calibBrowseBtn.setFixedWidth(90)
        self.calibLoadBtn = QtWidgets.QPushButton("Load/Reload")
        self.calibLoadBtn.setFixedWidth(110)

        self.seriesCombo = QtWidgets.QComboBox()
        self.seriesInfoLbl = QtWidgets.QLabel("Series: --")
        self.calibModeLbl = QtWidgets.QLabel("Mode: --")

        self.ndWheelAngleLbl = QtWidgets.QLabel("ND wheel angle:")
        self.ndWheelAngleCombo = QtWidgets.QComboBox()
        self.ndWheelAngleCombo.setMinimumWidth(100)

        self.hwpAngleLbl = QtWidgets.QLabel("HWP range:")
        self.hwpMinSpin = QtWidgets.QDoubleSpinBox()
        self.hwpMinSpin.setRange(0, 360); self.hwpMinSpin.setDecimals(2)
        self.hwpMinSpin.setSingleStep(0.5); self.hwpMinSpin.setSuffix(" °")
        self.hwpMinSpin.setFixedWidth(90)
        self.hwpMaxSpin = QtWidgets.QDoubleSpinBox()
        self.hwpMaxSpin.setRange(0, 360); self.hwpMaxSpin.setDecimals(2)
        self.hwpMaxSpin.setSingleStep(0.5); self.hwpMaxSpin.setSuffix(" °")
        self.hwpMaxSpin.setFixedWidth(90)
        self.hwpDashLbl = QtWidgets.QLabel("–")

        # Single wheel: only the calibration points between these wheel angles are swept
        self.wheelRangeLbl = QtWidgets.QLabel("ND wheel range:")
        self.wheelStartSpin = QtWidgets.QDoubleSpinBox()
        self.wheelEndSpin = QtWidgets.QDoubleSpinBox()
        for spin in (self.wheelStartSpin, self.wheelEndSpin):
            spin.setRange(0, 360); spin.setDecimals(2)
            spin.setSingleStep(1.0); spin.setSuffix(" °")
            spin.setFixedWidth(90)
            spin.setToolTip("Sweep only the calibration points from start to end (inclusive), going towards\n"
                            "increasing angle and wrapping through 360° = 0°: 100 → 20 is 100–360° plus 0–20°")
        self.wheelRangeBox = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(self.wheelRangeBox)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(QtWidgets.QLabel("start"))
        hl.addWidget(self.wheelStartSpin)
        hl.addWidget(QtWidgets.QLabel("→ end"))
        hl.addWidget(self.wheelEndSpin)
        self.wheelFullBtn = QtWidgets.QPushButton("Full range")
        self.wheelFullBtn.setFixedWidth(90)
        hl.addWidget(self.wheelFullBtn)
        hl.addStretch()

        self.cropTopSpin = QtWidgets.QSpinBox()
        self.cropBottomSpin = QtWidgets.QSpinBox()
        self.cropLeftSpin = QtWidgets.QSpinBox()
        self.cropRightSpin = QtWidgets.QSpinBox()
        for spin in (self.cropTopSpin, self.cropBottomSpin, self.cropLeftSpin, self.cropRightSpin):
            spin.setRange(0, 10000)
            spin.setFixedWidth(70)
        self.cropTopSpin.setValue(self._crop_top)
        self.cropBottomSpin.setValue(self._crop_bottom)
        self.cropLeftSpin.setValue(self._crop_left)
        self.cropRightSpin.setValue(self._crop_right)
        self.cropApplyBtn = QtWidgets.QPushButton("Apply Crop")
        self.cropApplyBtn.setFixedWidth(110)

        grid.addWidget(QtWidgets.QLabel("Save subfolder:"), 0, 0)
        grid.addWidget(self.saveNameEdit, 0, 1, 1, 2)
        grid.addWidget(self.saveNameNowBtn, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Power calib file:"), 1, 0)
        grid.addWidget(self.calibPathEdit, 1, 1)
        grid.addWidget(self.calibBrowseBtn, 1, 2)
        grid.addWidget(self.calibLoadBtn, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Calibration mode:"), 2, 0)
        grid.addWidget(self.calibModeLbl, 2, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Current series:"), 3, 0)
        grid.addWidget(self.seriesCombo, 3, 1)
        grid.addWidget(self.seriesInfoLbl, 3, 2, 1, 2)

        grid.addWidget(self.ndWheelAngleLbl,   4, 0)
        grid.addWidget(self.ndWheelAngleCombo, 4, 1)
        grid.addWidget(self.hwpAngleLbl,       4, 2)
        grid.addWidget(self.hwpMinSpin,        4, 3)
        grid.addWidget(self.hwpDashLbl,        4, 4)
        grid.addWidget(self.hwpMaxSpin,        4, 5)
        # same row, shown instead of the HWP+ND controls in single-wheel mode
        grid.addWidget(self.wheelRangeLbl,     4, 0)
        grid.addWidget(self.wheelRangeBox,     4, 1, 1, 5)

        grid.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"), 5, 0)
        grid.addWidget(self.cropTopSpin, 5, 1)
        grid.addWidget(self.cropBottomSpin, 5, 2)
        grid.addWidget(self.cropLeftSpin, 5, 3)
        grid.addWidget(self.cropRightSpin, 5, 4)
        grid.addWidget(self.cropApplyBtn, 5, 5)

        parent_layout.addWidget(box)

    def _build_gate_controls(self, parent_layout):
        box = QtWidgets.QGroupBox("Sweep Options")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.sweepPowerChk = QtWidgets.QCheckBox("Sweep power")
        self.sweepPowerChk.setChecked(True)
        self.sweepVoltageChk = QtWidgets.QCheckBox("Sweep voltage")
        self.sweepVoltageChk.setChecked(False)

        self.sweepGateCombo = QtWidgets.QComboBox()
        self.sweepGateCombo.addItems(["Dual gate", "Front gate", "Back gate"])
        self.sweepGateCombo.setEnabled(False)

        self.frontV0Spin = QtWidgets.QDoubleSpinBox()
        self.frontV0Spin.setDecimals(6)
        self.frontV0Spin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.frontV0Spin.setValue(DEFAULT_SWEEP_V0)
        self.frontV0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.frontStepSpin = QtWidgets.QDoubleSpinBox()
        self.frontStepSpin.setDecimals(6)
        self.frontStepSpin.setRange(-50.0, 50.0)
        self.frontStepSpin.setValue(DEFAULT_SWEEP_STEP)
        self.frontStepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.backV0Spin = QtWidgets.QDoubleSpinBox()
        self.backV0Spin.setDecimals(6)
        self.backV0Spin.setRange(-KEITHLEY_V_LIMIT, KEITHLEY_V_LIMIT)
        self.backV0Spin.setValue(DEFAULT_SWEEP_V0)
        self.backV0Spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.backStepSpin = QtWidgets.QDoubleSpinBox()
        self.backStepSpin.setDecimals(6)
        self.backStepSpin.setRange(-50.0, 50.0)
        self.backStepSpin.setValue(DEFAULT_SWEEP_STEP)
        self.backStepSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.sweepStepsSpin = QtWidgets.QSpinBox()
        self.sweepStepsSpin.setRange(1, 100000)
        self.sweepStepsSpin.setValue(int(DEFAULT_SWEEP_STEPS))
        self.sweepStepsSpin.setFixedWidth(90)

        self.backStepsSpin = QtWidgets.QSpinBox()
        self.backStepsSpin.setRange(1, 100000)
        self.backStepsSpin.setValue(int(DEFAULT_SWEEP_STEPS))
        self.backStepsSpin.setFixedWidth(90)

        self.gridSweepChk = QtWidgets.QCheckBox("Grid (fix A, sweep B)")
        self.gridSweepChk.setChecked(False)
        self.gridSweepChk.setToolTip(
            "Dual gate only: hold gate A at each step while sweeping gate B "
            "through its full list, then advance A. Uses separate step counts "
            "for A (Steps) and B (B steps)."
        )

        self.powerSkipSpin = QtWidgets.QSpinBox()
        self.powerSkipSpin.setRange(0, 100000)
        self.powerSkipSpin.setValue(0)
        self.powerSkipSpin.setFixedWidth(90)

        grid.addWidget(self.sweepPowerChk, 0, 0)
        grid.addWidget(self.sweepVoltageChk, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Sweep gate:"), 0, 2)
        grid.addWidget(self.sweepGateCombo, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Skip N power points:"), 0, 4)
        grid.addWidget(self.powerSkipSpin, 0, 5)
        grid.addWidget(self.gridSweepChk, 0, 6)

        # Angle list row (stage A, no calibration file needed)
        self.angleListChk = QtWidgets.QCheckBox("Use angle list (stage A)")
        self.angleListEdit = QtWidgets.QLineEdit()
        self.angleListEdit.setPlaceholderText("e.g.  0, 15.5, 30  (comma-separated °)")
        self.angleListPreviewLbl = QtWidgets.QLabel("(0 angles)")
        self.angleListPreviewLbl.setStyleSheet("color: #555;")

        # Power list row (HWP+ND calibration: target powers -> solved HWP / ND pairs)
        self.powerListChk = QtWidgets.QCheckBox("Use power list (HWP+ND calib)")
        self.powerListChk.setToolTip(
            "Needs a wheel+HWP calibration file.  Each target power is reached with the measured\n"
            "(HWP, ND) pair within the tolerance that has the smallest expected error (power error\n"
            "plus a 0.1° stage error), HWP limited to the HWP range above.  Swept in list order;\n"
            "targets that cannot be reached within the tolerance are skipped.")
        self.powerListEdit = QtWidgets.QLineEdit()
        self.powerListEdit.setPlaceholderText("e.g.  1, 2, 5, 10, 20 nW, 0.5 uW")
        self.powerListUnitCombo = QtWidgets.QComboBox()
        self.powerListUnitCombo.addItems(["pW", "nW", "µW", "mW", "W"])
        self.powerListUnitCombo.setCurrentText("nW")
        self.powerListUnitCombo.setToolTip("Unit of numbers typed without a unit")
        self.powerListTolSpin = QtWidgets.QDoubleSpinBox()
        self.powerListTolSpin.setRange(0.01, 100.0)
        self.powerListTolSpin.setDecimals(2)
        self.powerListTolSpin.setSingleStep(0.1)
        self.powerListTolSpin.setValue(1.0)
        self.powerListTolSpin.setSuffix(" %")
        self.powerListTolSpin.setToolTip("Tolerance on the predicted power")
        self.powerListPreviewLbl = QtWidgets.QLabel("(0 powers)")
        self.powerListPreviewLbl.setStyleSheet("color: #555;")
        self.powerListDetailsBtn = QtWidgets.QPushButton("Details…")
        self.powerListDetailsBtn.setFixedWidth(90)
        self._power_list_timer = QtCore.QTimer(self)
        self._power_list_timer.setSingleShot(True)
        self._power_list_timer.setInterval(300)

        lists = QtWidgets.QGridLayout()
        lists.setContentsMargins(0, 0, 0, 0)
        lists.setHorizontalSpacing(8)
        lists.setVerticalSpacing(6)
        lists.addWidget(self.angleListChk,        0, 0)
        lists.addWidget(self.angleListEdit,        0, 1, 1, 3)
        lists.addWidget(self.angleListPreviewLbl,  0, 4, 1, 2)
        lists.addWidget(self.powerListChk,        1, 0)
        lists.addWidget(self.powerListEdit,        1, 1)
        lists.addWidget(self.powerListUnitCombo,   1, 2)
        lists.addWidget(self.powerListTolSpin,     1, 3)
        lists.addWidget(self.powerListDetailsBtn,  1, 4)
        lists.addWidget(self.powerListPreviewLbl,  1, 5)
        lists.setColumnStretch(1, 1)
        lists.setColumnStretch(5, 1)
        grid.addLayout(lists, 1, 0, 1, 7)

        self.frontVendLbl = QtWidgets.QLabel("End: 0.000000 V")
        self.backVendLbl = QtWidgets.QLabel("End: 0.000000 V")

        grid.addWidget(QtWidgets.QLabel("Front V0 (V):"), 2, 0)
        grid.addWidget(self.frontV0Spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Front step (V):"), 2, 2)
        grid.addWidget(self.frontStepSpin, 2, 3)
        self.stepsLbl = QtWidgets.QLabel("Steps:")
        grid.addWidget(self.stepsLbl, 2, 4)
        grid.addWidget(self.sweepStepsSpin, 2, 5)
        grid.addWidget(self.frontVendLbl, 2, 6)

        grid.addWidget(QtWidgets.QLabel("Back V0 (V):"), 3, 0)
        grid.addWidget(self.backV0Spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Back step (V):"), 3, 2)
        grid.addWidget(self.backStepSpin, 3, 3)
        self.backStepsLbl = QtWidgets.QLabel("B steps:")
        grid.addWidget(self.backStepsLbl, 3, 4)
        grid.addWidget(self.backStepsSpin, 3, 5)
        grid.addWidget(self.backVendLbl, 3, 6)

        # Sweep order row (only meaningful when both power and voltage are swept)
        self.sweepOrderPowerOuterRb = QtWidgets.QRadioButton("Sweep voltages per power  →  next power")
        self.sweepOrderVoltageOuterRb = QtWidgets.QRadioButton("Sweep powers per voltage  →  next voltage")
        self.sweepOrderPowerOuterRb.setChecked(True)
        self._sweep_order_grp = QtWidgets.QButtonGroup(self)
        self._sweep_order_grp.addButton(self.sweepOrderPowerOuterRb)
        self._sweep_order_grp.addButton(self.sweepOrderVoltageOuterRb)

        grid.addWidget(QtWidgets.QLabel("Sweep order:"), 4, 0)
        grid.addWidget(self.sweepOrderPowerOuterRb, 4, 1, 1, 3)
        grid.addWidget(self.sweepOrderVoltageOuterRb, 4, 4, 1, 3)

        # What to acquire at every sweep point
        self.measureCombo = QtWidgets.QComboBox()
        self.measureCombo.addItems(list(MEASURE_MODES))
        self.measureCombo.setCurrentText(MEASURE_PL)
        self.measureCombo.setToolTip(
            "PL: Andor spectrum.  TRPL: SPAD23 / SPAD93 histogram (pSPAD).  PL + TRPL: both, acquired at the\n"
            "same time at every point (the point takes the longer of the two).")
        grid.addWidget(QtWidgets.QLabel("Measure:"), 5, 0)
        grid.addWidget(self.measureCombo, 5, 1)
        self.didpChk = QtWidgets.QCheckBox("dI/dP: pump 2 off / on at every point")
        self.didpChk.setToolTip(
            "Acquire every power / gate point twice with the filter flipper, pump 2 off (pump 1 only)\n"
            "then pump 2 on (pump 1 + 2), before moving on. Files are tagged _P2off / _P2on in the same\n"
            "folder; the maps and TRPL tabs show the pump-2-off data.")
        self.flipperSettleSpin = QtWidgets.QSpinBox()
        self.flipperSettleSpin.setRange(0, 60000)
        self.flipperSettleSpin.setSuffix(" ms")
        self.flipperSettleSpin.setValue(int(FLIPPER_SETTLE_MS))
        self.flipperSettleSpin.setFixedWidth(90)
        self.flipperSettleSpin.setToolTip("Wait after each flip before acquiring")
        grid.addWidget(self.didpChk, 5, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Flipper settle:"), 5, 4)
        grid.addWidget(self.flipperSettleSpin, 5, 5)
        self.rampReadChk = QtWidgets.QCheckBox("Read V/I during gate ramps")
        self.rampReadChk.setChecked(False)
        self.rampReadChk.setToolTip(
            "Read V and I after every 10 mV ramp microstep during a sweep (live V/I plot only; the compliance\n"
            "check is done at every point either way). On a 2450 each read is two measurements, about 0.4 s,\n"
            "so a 0.4 V gate step takes ~20 s longer. Off: one reading per ramp is plotted.")
        grid.addWidget(self.rampReadChk, 5, 6)
        self.homeAfterScanChk = QtWidgets.QCheckBox("Home both gates after scan")
        self.homeAfterScanChk.setChecked(False)
        self.homeAfterScanChk.setToolTip(
            "When the sweep ends (complete, or stopped by a trip / error), ramp every connected gate to 0 V,\n"
            "the same as the Home Both Gates button. Not after Abort: the gates stay where they are.")
        grid.addWidget(self.homeAfterScanChk, 6, 0, 1, 3)

        parent_layout.addWidget(box)

    def _build_histogram_settings(self, parent_layout):
        box = QtWidgets.QGroupBox("Histogram Settings (visualization only)")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.histStartNsSpin = QtWidgets.QDoubleSpinBox()
        self.histStartNsSpin.setDecimals(4)
        self.histStartNsSpin.setRange(-1e9, 1e9)
        self.histStartNsSpin.setValue(float(DEFAULT_HIST_START_NS))
        self.histStartNsSpin.setFixedWidth(110)
        self.histEndNsSpin = QtWidgets.QDoubleSpinBox()
        self.histEndNsSpin.setDecimals(4)
        self.histEndNsSpin.setRange(-1e9, 1e9)
        self.histEndNsSpin.setValue(float(DEFAULT_HIST_END_NS))
        self.histEndNsSpin.setFixedWidth(110)
        self.histRefNsSpin = QtWidgets.QDoubleSpinBox()
        self.histRefNsSpin.setDecimals(4)
        self.histRefNsSpin.setRange(-1e9, 1e9)
        self.histRefNsSpin.setValue(float(DEFAULT_HIST_REF_NS))
        self.histRefNsSpin.setFixedWidth(110)

        grid.addWidget(QtWidgets.QLabel("Start time (ns)"), 0, 0)
        grid.addWidget(self.histStartNsSpin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("End time (ns)"), 0, 2)
        grid.addWidget(self.histEndNsSpin, 0, 3)
        self.chk_hist_ref_auto = QtWidgets.QCheckBox(f"t0 + {T0_REF_BINS} bins")
        self.chk_hist_ref_auto.setChecked(True)
        self.chk_hist_ref_auto.setToolTip(
            f"With 'Align time zero' on, t = 0 ns is the signal peak (maximum of the brightest pixel, at every\n"
            f"histogram of a scan) and the reference follows t0 + {T0_REF_BINS} bins.\n"
            "Typing a reference (or clicking a time in the spectral view) switches this off.")
        grid.addWidget(QtWidgets.QLabel("Reference time (ns)"), 0, 4)
        grid.addWidget(self.histRefNsSpin, 0, 5)
        grid.addWidget(self.chk_hist_ref_auto, 0, 6)
        grid.addWidget(QtWidgets.QLabel("All bins are saved; settings only affect visualization. With time-zero "
                                        "alignment on, t = 0 is the signal peak (brightest pixel)."), 1, 0, 1, 7)

        parent_layout.addWidget(box)


    def _build_plots(self, parent_layout):
        self.tabs = QtWidgets.QTabWidget()

        # Options tab (default)
        options_tab = QtWidgets.QWidget()
        options_layout = QtWidgets.QVBoxLayout(options_tab)
        options_layout.setContentsMargins(8, 8, 8, 8)
        options_layout.setSpacing(8)
        self._build_session_controls(options_layout)
        self._build_gate_controls(options_layout)
        self._build_histogram_settings(options_layout)
        options_layout.addStretch(1)
        self.tabs.addTab(options_tab, "Session/Calibration/Sweep")

        # Live tab
        live_tab = QtWidgets.QWidget()
        live_layout = QtWidgets.QHBoxLayout(live_tab)
        live_layout.setContentsMargins(8, 8, 8, 8)
        live_layout.setSpacing(8)
        self.liveView = AndorLiveViewWidgetV2(self.cam, title="PL Image", match_v1=True)
        self._apply_crop_to_liveview()
        self.liveView.set_linecut_row(int(DEFAULT_LINECUT_ROW))
        self.liveView.linecut_changed.connect(lambda _: self._refresh_linecut_maps())

        self.spad_live_box = spad_live_box = QtWidgets.QGroupBox(f"{self._det.key} Count Rate Map")
        self._spad_live_layout = spad_live_layout = QtWidgets.QVBoxLayout(spad_live_box)
        spad_live_layout.setContentsMargins(6, 6, 6, 6)
        live_view_row = QtWidgets.QHBoxLayout()
        self.cmb_spad_view_live = self._make_view_combo()
        live_view_row.addWidget(QtWidgets.QLabel("View:"))
        live_view_row.addWidget(self.cmb_spad_view_live)
        live_view_row.addStretch(1)
        spad_live_layout.addLayout(live_view_row)
        self.spadLiveMap = self._det.PixelMapPlot(f"{self._det.key} Count-Rate Map")
        spad_live_layout.addWidget(self.spadLiveMap, 2)
        # spectral mode: center-line spectrum and the active line's linecut under the map
        self.spectralLive = SpectralLivePanel(self._spec_geom)
        spad_live_layout.addWidget(self.spectralLive, 3)
        self.lbl_spad_live_info = QtWidgets.QLabel(f"Pixel {self._spad_selected_pixel_live} selected")
        spad_live_layout.addWidget(self.lbl_spad_live_info)

        gate_plot_box = QtWidgets.QWidget()
        gate_plot_layout = QtWidgets.QVBoxLayout(gate_plot_box)
        gate_plot_layout.setContentsMargins(0, 0, 0, 0)
        gate_plot_layout.setSpacing(6)
        front_plot_box = QtWidgets.QGroupBox("Front gate V/I")
        front_plot_layout = QtWidgets.QVBoxLayout(front_plot_box)
        front_plot_layout.setContentsMargins(6, 6, 6, 6)
        self.frontPlot = GateVIPlot(front_plot_box)
        front_plot_layout.addWidget(self.frontPlot)
        back_plot_box = QtWidgets.QGroupBox("Back gate V/I")
        back_plot_layout = QtWidgets.QVBoxLayout(back_plot_box)
        back_plot_layout.setContentsMargins(6, 6, 6, 6)
        self.backPlot = GateVIPlot(back_plot_box)
        back_plot_layout.addWidget(self.backPlot)
        gate_plot_layout.addWidget(front_plot_box, 1)
        gate_plot_layout.addWidget(back_plot_box, 1)
        live_layout.addWidget(self.liveView, 4)
        live_layout.addWidget(spad_live_box, 2)
        live_layout.addWidget(gate_plot_box, 2)
        self.tabs.addTab(live_tab, "Live")

        # Power/Voltage tab
        power_tab = QtWidgets.QWidget()
        power_layout = QtWidgets.QHBoxLayout(power_tab)
        power_layout.setContentsMargins(8, 8, 8, 8)
        power_layout.setSpacing(8)
        self.powerMap = LinecutMapPlot("Linecut vs Power", "Power", "Wavelength (nm)")
        self.voltageSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.voltageSlider.setRange(0, 0)
        self.voltageSlider.setEnabled(False)
        self.voltageSelLbl = QtWidgets.QLabel("Voltage: --")
        power_plot_box = QtWidgets.QWidget()
        power_plot_layout = QtWidgets.QVBoxLayout(power_plot_box)
        power_plot_layout.setContentsMargins(0, 0, 0, 0)
        power_plot_layout.setSpacing(4)
        power_plot_layout.addWidget(self.powerMap)
        power_slider_row = QtWidgets.QHBoxLayout()
        power_slider_row.addWidget(QtWidgets.QLabel("Select V:"))
        power_slider_row.addWidget(self.voltageSlider, 1)
        power_slider_row.addWidget(self.voltageSelLbl)
        power_plot_layout.addLayout(power_slider_row)
        power_clim_row = QtWidgets.QHBoxLayout()
        power_clim_row.addWidget(QtWidgets.QLabel("Color min:"))
        self.mapCminEdit = QtWidgets.QLineEdit()
        self.mapCminEdit.setPlaceholderText("auto")
        self.mapCminEdit.setFixedWidth(90)
        power_clim_row.addWidget(self.mapCminEdit)
        power_clim_row.addWidget(QtWidgets.QLabel("max:"))
        self.mapCmaxEdit = QtWidgets.QLineEdit()
        self.mapCmaxEdit.setPlaceholderText("auto")
        self.mapCmaxEdit.setFixedWidth(90)
        power_clim_row.addWidget(self.mapCmaxEdit)
        power_clim_row.addStretch()
        power_plot_layout.addLayout(power_clim_row)
        self.voltageMap = LinecutMapPlot("Linecut vs Voltage", "Voltage (V)", "Wavelength (nm)")
        self.powerSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.powerSlider.setRange(0, 0)
        self.powerSlider.setEnabled(False)
        self.powerSelLbl = QtWidgets.QLabel("Power: --")

        self.aSelSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aSelSlider.setRange(0, 0)
        self.aSelSlider.setEnabled(False)
        self.aSelLbl = QtWidgets.QLabel("Gate A: --")

        voltage_plot_box = QtWidgets.QWidget()
        voltage_plot_layout = QtWidgets.QVBoxLayout(voltage_plot_box)
        voltage_plot_layout.setContentsMargins(0, 0, 0, 0)
        voltage_plot_layout.setSpacing(4)
        voltage_plot_layout.addWidget(self.voltageMap)
        voltage_slider_row = QtWidgets.QHBoxLayout()
        voltage_slider_row.addWidget(QtWidgets.QLabel("Select power:"))
        voltage_slider_row.addWidget(self.powerSlider, 1)
        voltage_slider_row.addWidget(self.powerSelLbl)
        voltage_plot_layout.addLayout(voltage_slider_row)
        a_slider_row = QtWidgets.QHBoxLayout()
        a_slider_row.addWidget(QtWidgets.QLabel("Select A (V):"))
        a_slider_row.addWidget(self.aSelSlider, 1)
        a_slider_row.addWidget(self.aSelLbl)
        voltage_plot_layout.addLayout(a_slider_row)
        voltage_clim_row = QtWidgets.QHBoxLayout()
        voltage_clim_row.addWidget(QtWidgets.QLabel("Color min:"))
        self.vmapCminEdit = QtWidgets.QLineEdit()
        self.vmapCminEdit.setPlaceholderText("auto")
        self.vmapCminEdit.setFixedWidth(90)
        voltage_clim_row.addWidget(self.vmapCminEdit)
        voltage_clim_row.addWidget(QtWidgets.QLabel("max:"))
        self.vmapCmaxEdit = QtWidgets.QLineEdit()
        self.vmapCmaxEdit.setPlaceholderText("auto")
        self.vmapCmaxEdit.setFixedWidth(90)
        voltage_clim_row.addWidget(self.vmapCmaxEdit)
        voltage_clim_row.addStretch()
        voltage_plot_layout.addLayout(voltage_clim_row)
        self.powerTable = QtWidgets.QTableWidget(0, 2)
        self.powerTable.setHorizontalHeaderLabels(["Power", "Taken"])
        self.powerTable.horizontalHeader().setStretchLastSection(True)
        self.powerTable.verticalHeader().setVisible(False)
        self.powerTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.powerTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.powerTable.setMinimumWidth(150)
        power_layout.addWidget(power_plot_box, 2)
        power_layout.addWidget(voltage_plot_box, 2)
        power_layout.addWidget(self.powerTable, 1)
        self.tabs.addTab(power_tab, "Power/Voltage")

        # Histogram tab
        hist_tab = QtWidgets.QWidget()
        hist_layout = QtWidgets.QVBoxLayout(hist_tab)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        hist_layout.setSpacing(6)
        hist_ctrl = QtWidgets.QHBoxLayout()
        self.chk_hist_log = QtWidgets.QCheckBox("Log Y")
        self.chk_hist_log.setChecked(False)
        self.lbl_hist_info = QtWidgets.QLabel("Left click: select histogram pixel(s). Right click: exclude/include fit pixel.")
        self.lbl_hist_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.cmb_spad_view = self._make_view_combo()
        self.btn_load_wl_cal = QtWidgets.QPushButton("Load λ-cal…")
        self.btn_load_wl_cal.setFixedWidth(100)
        self.btn_load_wl_cal.setToolTip(
            "Wavelength calibration of the SPAD's spectral line: a JSON saved by the SPAD23 / SPAD93 v2 GUI\n"
            "(Counts tab, spectral mode). It puts a wavelength axis on the maps and spectral plots, is copied into\n"
            "every sweep folder (WLCAL_*.json) and reloaded on the next start. Without one the axes are in µm.")
        self.btn_clear_wl_cal = QtWidgets.QPushButton("Clear")
        self.btn_clear_wl_cal.setFixedWidth(60)
        self.lbl_wl_cal = QtWidgets.QLabel("λ-cal: none")
        self.lbl_wl_cal.setStyleSheet("color:#6b7280")
        hist_ctrl.addWidget(self.chk_hist_log)
        hist_ctrl.addSpacing(8)
        hist_ctrl.addWidget(QtWidgets.QLabel("View:"))
        hist_ctrl.addWidget(self.cmb_spad_view)
        hist_ctrl.addWidget(self.btn_load_wl_cal)
        hist_ctrl.addWidget(self.btn_clear_wl_cal)
        hist_ctrl.addWidget(self.lbl_wl_cal)
        hist_ctrl.addSpacing(8)
        hist_ctrl.addWidget(self.lbl_hist_info, 1)
        hist_layout.addLayout(hist_ctrl)
        self._build_trpl_view_options(hist_layout)
        # left: the two maps (count rate, 2D fit); right: the three line plots (histograms, radial, sigma^2(t))
        hist_cols = QtWidgets.QHBoxLayout()
        hist_cols.setSpacing(8)
        # the detector switch swaps the two maps in this layout (_rebuild_spad_views)
        self._hist_grid = maps_col = QtWidgets.QVBoxLayout()
        maps_col.setSpacing(6)
        plots_holder = QtWidgets.QWidget()
        plots_col = QtWidgets.QVBoxLayout(plots_holder)
        plots_col.setContentsMargins(0, 0, 0, 0)
        plots_col.setSpacing(6)
        self.spadHistMap = self._det.PixelMapPlot("Count-Rate Map")
        self.histFitPlot = self._det.Gauss2DFitPlot()
        self.histTracePlot = HistogramPlot()
        self.histTracePlot.cursor.hide()  # v1 has no time cursor (the reference line is shown)
        self.histRadialPlot = RadialFitPlot()
        maps_col.addWidget(self.spadHistMap, 1)
        # below the map: the 2D fit (2D) or the time-resolved spectrum image (spectral)
        self._hist_left_stack = QtWidgets.QStackedWidget()
        fit_holder = QtWidgets.QWidget()
        self._hist_fit_layout = QtWidgets.QVBoxLayout(fit_holder)
        self._hist_fit_layout.setContentsMargins(0, 0, 0, 0)
        self._hist_fit_layout.addWidget(self.histFitPlot)
        self._hist_left_stack.addWidget(fit_holder)
        self.spectralHist = SpectralHistPage(self._spec_geom, self._spectral_fit_allowed)
        self._hist_left_stack.addWidget(self.spectralHist.image_panel)
        maps_col.addWidget(self._hist_left_stack, 1)
        plots_col.addWidget(self.histTracePlot, 1)
        plots_col.addWidget(self.histRadialPlot, 1)
        # sigma^2(t): sequential 2D Gaussian fit of every frame from the reference (on request, after collection)
        sig_row = QtWidgets.QHBoxLayout()
        sig_row2 = QtWidgets.QHBoxLayout()
        self.btn_fit_sigma = QtWidgets.QPushButton("Fit σ²(t)")
        self.btn_fit_sigma.setFixedWidth(100)
        self.btn_fit_sigma.setToolTip(
            "Fit a 2D Gaussian to every frame from the reference time to the histogram end (each frame starting\n"
            "from the previous fit) and plot sigma_eq^2(t); D comes from sigma^2 = sigma0^2 + 2 D (t - t_ref).\n"
            "Uses the settings above (time-zero alignment, background, Gaussian offset, excluded pixels).\n"
            "Only for a finished histogram: disabled while a scan or a TRPL acquisition runs.")
        self.btn_cancel_sigma = QtWidgets.QPushButton("Cancel")
        self.btn_cancel_sigma.setFixedWidth(70)
        self.prg_sigma = QtWidgets.QProgressBar()
        self.prg_sigma.setFixedWidth(120)
        self.prg_sigma.setValue(0)
        self.spin_sig_d0 = QtWidgets.QDoubleSpinBox()
        self.spin_sig_d1 = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_sig_d0, self.spin_sig_d1):
            sp.setRange(0.0, 1e6)
            sp.setDecimals(3)
            sp.setSuffix(" ns")
            sp.setKeyboardTracking(False)
            sp.setFixedWidth(100)
        self.spin_sig_d1.setSpecialValueText("end")
        self.spin_sig_d0.setToolTip("Line fit for D from this time after the reference ...")
        self.spin_sig_d1.setToolTip("... to this time after the reference (0 = to the end of the fit).")
        self.chk_sig_flagged = QtWidgets.QCheckBox("Include flagged")
        self.chk_sig_flagged.setToolTip("Also use fits at a bound or with low signal (hollow points) in the line fit.")
        self.btn_copy_sigma = QtWidgets.QPushButton("Copy σ²(t)")
        self.btn_copy_sigma.setFixedWidth(90)
        self.btn_copy_sigma.setToolTip("Copy time (ns), sigma^2 (um^2) and flagged (0/1) as tab-separated columns.")
        self.lbl_sigma = QtWidgets.QLabel("")
        self.lbl_sigma.setStyleSheet("color:#6b7280")
        self.lbl_sigma.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        # two short rows, so they fit the half-width column
        sig_row.addWidget(self.btn_fit_sigma)
        sig_row.addWidget(self.btn_cancel_sigma)
        sig_row.addWidget(self.prg_sigma)
        sig_row.addWidget(self.btn_copy_sigma)
        sig_row.addWidget(self.lbl_sigma, 1)
        sig_row2.addWidget(QtWidgets.QLabel("D from"))
        sig_row2.addWidget(self.spin_sig_d0)
        sig_row2.addWidget(QtWidgets.QLabel("to"))
        sig_row2.addWidget(self.spin_sig_d1)
        sig_row2.addWidget(QtWidgets.QLabel("after t_ref"))
        sig_row2.addSpacing(8)
        sig_row2.addWidget(self.chk_sig_flagged)
        sig_row2.addStretch(1)
        self.histSigmaPlot = TracePlot()
        self._sigma_line = self.histSigmaPlot.p1.plot(pen=pg.mkPen("#dc2626", width=1.6, style=QtCore.Qt.DashLine))
        plots_col.addLayout(sig_row)
        plots_col.addLayout(sig_row2)
        plots_col.addWidget(self.histSigmaPlot, 1)
        # right column: the 2D plots or the spectral page
        self._hist_right_stack = QtWidgets.QStackedWidget()
        self._hist_right_stack.addWidget(plots_holder)
        self._hist_right_stack.addWidget(self.spectralHist)
        hist_cols.addLayout(maps_col, 1)
        hist_cols.addWidget(self._hist_right_stack, 1)
        hist_layout.addLayout(hist_cols, 1)
        self.tabs.addTab(hist_tab, "Histogram")
        self._hist_tab = hist_tab

        # Lifetime tab
        lifetime_tab = QtWidgets.QWidget()
        lifetime_layout = QtWidgets.QVBoxLayout(lifetime_tab)
        lifetime_layout.setContentsMargins(8, 8, 8, 8)
        lifetime_layout.setSpacing(6)
        lifetime_ctrl1 = QtWidgets.QHBoxLayout()
        lifetime_ctrl1.addWidget(QtWidgets.QLabel("Select V:"))
        self.lifetimeVoltageSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lifetimeVoltageSlider.setRange(0, 0)
        self.lifetimeVoltageSlider.setEnabled(False)
        self.lifetimeVoltageSelLbl = QtWidgets.QLabel("Voltage: --")
        lifetime_ctrl1.addWidget(self.lifetimeVoltageSlider, 1)
        lifetime_ctrl1.addWidget(self.lifetimeVoltageSelLbl)
        lifetime_layout.addLayout(lifetime_ctrl1)

        lifetime_ctrl2 = QtWidgets.QHBoxLayout()
        lifetime_ctrl2.addWidget(QtWidgets.QLabel("Select power:"))
        self.lifetimePowerSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lifetimePowerSlider.setRange(0, 0)
        self.lifetimePowerSlider.setEnabled(False)
        self.lifetimePowerSelLbl = QtWidgets.QLabel("Power: --")
        self.lifetimeScaleCombo = QtWidgets.QComboBox()
        self.lifetimeScaleCombo.addItems(["Log", "Linear"])
        lifetime_ctrl2.addWidget(self.lifetimePowerSlider, 1)
        lifetime_ctrl2.addWidget(self.lifetimePowerSelLbl)
        lifetime_ctrl2.addSpacing(12)
        lifetime_ctrl2.addWidget(QtWidgets.QLabel("Colormap scale:"))
        lifetime_ctrl2.addWidget(self.lifetimeScaleCombo)
        lifetime_layout.addLayout(lifetime_ctrl2)

        lifetime_plots = QtWidgets.QHBoxLayout()
        lifetime_plots.setSpacing(8)
        self.lifetimePowerPlot = LifetimeMapPlot()
        self.lifetimeVoltagePlot = LifetimeMapPlot()
        lifetime_plots.addWidget(self.lifetimePowerPlot, 1)
        lifetime_plots.addWidget(self.lifetimeVoltagePlot, 1)
        lifetime_layout.addLayout(lifetime_plots, 1)
        self.tabs.addTab(lifetime_tab, "Lifetime")
        self._lifetime_tab = lifetime_tab

        parent_layout.addWidget(self.tabs, 1)
        self.tabs.setCurrentIndex(0)
        self._init_spad_live_map_plot()
        self._init_spad_hist_map_plot()

    def _build_trpl_view_options(self, parent_layout) -> None:
        """TRPL view options from the SPAD v2 GUIs (display and fit only; saved data stays raw)."""
        row1 = QtWidgets.QHBoxLayout()
        self.chk_t0_align = QtWidgets.QCheckBox("Align time zero on each pixel's first peak")
        self.chk_t0_align.setChecked(False)
        self.chk_t0_align.setToolTip(
            "View only (as in the SPAD23 / SPAD93 v2 GUIs): shift each pixel by whole bins so the rising edge of\n"
            "its first peak lines up (align_first_peaks). Applies to the histogram traces, the\n"
            "2D fit frame and the lifetime maps. Saved data stays raw.\n"
            "The time axis is then shifted so t = 0 ns is the signal peak (maximum of the brightest pixel) of every\n"
            "histogram; Start / End / Reference / pre-t0 times are converted when this is switched.")
        self.lbl_t0_align = QtWidgets.QLabel("")
        self.lbl_t0_align.setStyleSheet("color:#6b7280")
        self.lbl_t0_align.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row1.addWidget(self.chk_t0_align)
        row1.addWidget(self.lbl_t0_align, 1)
        parent_layout.addLayout(row1)

        row_dark = QtWidgets.QHBoxLayout()
        self.spin_dark_acq = QtWidgets.QDoubleSpinBox()
        self.spin_dark_acq.setRange(0.5, 3600.0)
        self.spin_dark_acq.setDecimals(1)
        self.spin_dark_acq.setValue(30.0)
        self.spin_dark_acq.setSuffix(" s")
        self.spin_dark_acq.setFixedWidth(90)
        self.spin_dark_acq.setToolTip("Acquisition time of the dark reference (longer = less noise).")
        self.btn_take_dark = QtWidgets.QPushButton("Take dark…")
        self.btn_take_dark.setToolTip(
            "Dark reference: a TRPL acquisition with the excitation beam blocked (you are asked to block it),\n"
            "with the TCSPC bin size above. It can be subtracted before the 2D fit instead of the pre-t0 offset,\n"
            "which also holds signal left over from earlier pulses when the lifetime is long. It is scaled to\n"
            "each histogram's acquisition time and bin size, saved under DATA_DIR/dark_references and copied\n"
            "into every sweep folder (DARK_TRPL_*.csv) for the offline viewer.")
        self.btn_load_dark = QtWidgets.QPushButton("Load dark…")
        self.btn_clear_dark = QtWidgets.QPushButton("Clear")
        for b in (self.btn_take_dark, self.btn_load_dark):
            b.setFixedWidth(100)
        self.btn_clear_dark.setFixedWidth(60)
        self.lbl_dark_ref = QtWidgets.QLabel("No dark reference.")
        self.lbl_dark_ref.setStyleSheet("color:#6b7280")
        self.lbl_dark_ref.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row_dark.addWidget(QtWidgets.QLabel("Dark reference:"))
        row_dark.addWidget(self.spin_dark_acq)
        row_dark.addWidget(self.btn_take_dark)
        row_dark.addWidget(self.btn_load_dark)
        row_dark.addWidget(self.btn_clear_dark)
        row_dark.addWidget(self.lbl_dark_ref, 1)
        parent_layout.addLayout(row_dark)

        row2 = QtWidgets.QHBoxLayout()
        self.cmb_hist_bg = QtWidgets.QComboBox()
        self.cmb_hist_bg.addItem("No background", "none")
        self.cmb_hist_bg.addItem("Subtract pre-t0 offset", "pre_t0")
        self.cmb_hist_bg.addItem("Subtract dark reference", "dark_ref")
        self.cmb_hist_bg.setToolTip(
            "2D fit only: per-pixel background subtracted from the reference frame (the traces and lifetime maps\n"
            "are not changed).\n"
            "Pre-t0 offset: each pixel's mean count per bin over the pre-t0 window (dark counts plus any signal\n"
            "left over from earlier pulses).\n"
            "Dark reference: the beam-blocked acquisition, scaled to this histogram (take or load one first).")
        self.spin_pre_t0_start = QtWidgets.QDoubleSpinBox()
        self.spin_pre_t0_end = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_pre_t0_start, self.spin_pre_t0_end):
            sp.setRange(-1e6, 1e9)
            sp.setDecimals(3)
            sp.setSuffix(" ns")
            sp.setKeyboardTracking(False)
            sp.setFixedWidth(110)
            sp.setToolTip("Window before time zero used for the per-pixel offset.")
        self.btn_pre_t0_auto = QtWidgets.QPushButton("Auto")
        self.btn_pre_t0_auto.setFixedWidth(60)
        self.btn_pre_t0_auto.setToolTip("Window from the first bin to just before the rise of the summed trace.")
        self.chk_fix_offset = QtWidgets.QCheckBox("Fix Gaussian offset at 0")
        self.chk_fix_offset.setChecked(False)
        self.chk_fix_offset.setToolTip(
            "Hold the constant offset of the 2D Gaussian at 0 instead of fitting it.\n"
            "Use it with the pre-t0 subtraction, when the remaining background should be zero.")
        self.lbl_pre_t0 = QtWidgets.QLabel("")
        self.lbl_pre_t0.setStyleSheet("color:#6b7280")
        self.lbl_pre_t0.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row2.addWidget(self.cmb_hist_bg)
        row2.addWidget(QtWidgets.QLabel("Pre-t0 window"))
        row2.addWidget(self.spin_pre_t0_start)
        row2.addWidget(QtWidgets.QLabel("to"))
        row2.addWidget(self.spin_pre_t0_end)
        row2.addWidget(self.btn_pre_t0_auto)
        row2.addSpacing(12)
        row2.addWidget(self.chk_fix_offset)
        row2.addSpacing(12)
        row2.addWidget(self.lbl_pre_t0, 1)
        parent_layout.addLayout(row2)

    def _wire_signals(self):
        self.initBtn.clicked.connect(self.on_connect_all)
        self.disconnectBtn.clicked.connect(self.on_disconnect_all)
        self.camConnBtn.clicked.connect(lambda: self._toggle_device("camera"))
        self.specConnBtn.clicked.connect(lambda: self._toggle_device("spec"))
        self.stageConnBtn.clicked.connect(lambda: self._toggle_device("A"))
        self.stageBConnBtn.clicked.connect(lambda: self._toggle_device("B"))
        self.frontConnBtn.clicked.connect(lambda: self._toggle_device("front"))
        self.backConnBtn.clicked.connect(lambda: self._toggle_device("back"))
        self.tcspcConnBtn.clicked.connect(lambda: self._toggle_device("spad"))
        self.flipperConnBtn.clicked.connect(lambda: self._toggle_device("flipper"))
        self.pump2OffBtn.clicked.connect(lambda: self._move_pump2(False))
        self.pump2OnBtn.clicked.connect(lambda: self._move_pump2(True))
        self.pump2PosCombo.currentIndexChanged.connect(self._on_pump2_pos_changed)
        self.didpChk.toggled.connect(lambda _v: self._sync_controls())
        self.stopCountsBtn.clicked.connect(self.on_toggle_live_counts)
        self.takeHistogramBtn.clicked.connect(self.on_take_histogram)
        self.tcspcCalibAlignBtn.clicked.connect(self.on_spad_calibrate_align)
        self.measureCombo.currentIndexChanged.connect(lambda _i: self._sync_controls())
        self.takeCurrentBtn.clicked.connect(self.on_take_current)
        self.sweepBtn.clicked.connect(self.on_sweep)
        self.abortBtn.clicked.connect(self.on_abort)
        self.calibBrowseBtn.clicked.connect(self.on_browse_calib)
        self.calibLoadBtn.clicked.connect(self.on_load_calib)
        self.saveNameNowBtn.clicked.connect(self.on_refresh_save_name)
        self.applySpecBtn.clicked.connect(self.on_apply_spec)
        self.stageHomeBtn.clicked.connect(lambda: self._stage_home("A"))
        self.stageMoveBtn.clicked.connect(lambda: self._stage_move("A"))
        self.stageDetectBtn.clicked.connect(self.on_stage_detect)
        self.stageBHomeBtn.clicked.connect(lambda: self._stage_home("B"))
        self.stageBMoveBtn.clicked.connect(lambda: self._stage_move("B"))
        self.stageBDetectBtn.clicked.connect(self.on_stage_detect)
        self.stopLiveBtn.clicked.connect(self.on_stop_live)
        self.seriesCombo.currentIndexChanged.connect(self._update_series_info)
        self.ndWheelAngleCombo.currentIndexChanged.connect(self._on_hwp_nd_angle_changed)
        self.hwpMinSpin.valueChanged.connect(self._on_hwp_nd_angle_changed)
        self.hwpMaxSpin.valueChanged.connect(self._on_hwp_nd_angle_changed)
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpDashLbl, self.hwpMaxSpin):
            w.setVisible(False)
        self._set_wheel_range_visible(False)
        self.wheelStartSpin.valueChanged.connect(self._on_wheel_range_changed)
        self.wheelEndSpin.valueChanged.connect(self._on_wheel_range_changed)
        self.wheelFullBtn.clicked.connect(self._set_full_wheel_range)
        self.cropApplyBtn.clicked.connect(self.on_apply_crop)
        self.readoutRateCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.preampCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.homeBothBtn.clicked.connect(self.on_home_both_gates)
        self.frontApplyBtn.clicked.connect(self.on_front_apply)
        self.backApplyBtn.clicked.connect(self.on_back_apply)
        self.sweepVoltageChk.toggled.connect(self._update_gate_sweep_controls)
        self.sweepPowerChk.toggled.connect(self._sync_controls)
        self.angleListChk.toggled.connect(self._on_angle_list_toggled)
        self.angleListEdit.textChanged.connect(self._on_angle_list_changed)
        self.powerListChk.toggled.connect(self._on_power_list_toggled)
        self.powerListEdit.textChanged.connect(lambda _t: self._power_list_timer.start())
        self.powerListUnitCombo.currentIndexChanged.connect(lambda _i: self._power_list_timer.start())
        self.powerListTolSpin.valueChanged.connect(lambda _v: self._power_list_timer.start())
        self._power_list_timer.timeout.connect(self._on_hwp_nd_angle_changed)
        self.powerListDetailsBtn.clicked.connect(self._show_power_list_details)
        self.sweepGateCombo.currentIndexChanged.connect(self._update_gate_sweep_controls)
        self.gridSweepChk.toggled.connect(self._update_gate_sweep_controls)
        self.frontV0Spin.valueChanged.connect(self._update_vend_labels)
        self.frontStepSpin.valueChanged.connect(self._update_vend_labels)
        self.backV0Spin.valueChanged.connect(self._update_vend_labels)
        self.backStepSpin.valueChanged.connect(self._update_vend_labels)
        self.sweepStepsSpin.valueChanged.connect(self._update_vend_labels)
        self.backStepsSpin.valueChanged.connect(self._update_vend_labels)
        self._update_vend_labels()
        self.voltageSlider.valueChanged.connect(self._on_voltage_slider)
        self.powerSlider.valueChanged.connect(self._on_power_slider)
        self.aSelSlider.valueChanged.connect(self._on_a_slider)
        self.mapCminEdit.editingFinished.connect(self._on_map_clim_changed)
        self.mapCmaxEdit.editingFinished.connect(self._on_map_clim_changed)
        self.vmapCminEdit.editingFinished.connect(self._on_vmap_clim_changed)
        self.vmapCmaxEdit.editingFinished.connect(self._on_vmap_clim_changed)
        self.lifetimeVoltageSlider.valueChanged.connect(self._on_voltage_slider)
        self.lifetimePowerSlider.valueChanged.connect(self._on_power_slider)
        self.lifetimeScaleCombo.currentIndexChanged.connect(lambda _i: self._update_lifetime_tab())
        self.chk_hist_log.toggled.connect(lambda _v: self._update_histogram_tab())
        self.histStartNsSpin.valueChanged.connect(lambda _v: (self._update_histogram_tab(), self._update_lifetime_tab()))
        self.histEndNsSpin.valueChanged.connect(lambda _v: (self._update_histogram_tab(), self._update_lifetime_tab()))
        self.histRefNsSpin.valueChanged.connect(lambda _v: self._on_hist_ref_edited())
        self.chk_hist_ref_auto.toggled.connect(lambda _v: self._update_histogram_tab())
        self.chk_t0_align.toggled.connect(lambda _v: (self._update_histogram_tab(), self._update_lifetime_tab()))
        self.cmb_hist_bg.currentIndexChanged.connect(lambda _i: self._on_hist_bg_changed())
        self.btn_take_dark.clicked.connect(self.on_take_dark)
        self.btn_load_dark.clicked.connect(self.on_load_dark)
        self.btn_clear_dark.clicked.connect(lambda: self._set_dark_reference(None, "cleared"))
        self._update_dark_controls()
        self.spin_pre_t0_start.valueChanged.connect(self._on_pre_t0_window_edited)
        self.spin_pre_t0_end.valueChanged.connect(self._on_pre_t0_window_edited)
        self.btn_pre_t0_auto.clicked.connect(self._on_pre_t0_auto)
        self.chk_fix_offset.toggled.connect(lambda _v: self._update_histogram_tab())
        self.btn_fit_sigma.clicked.connect(self.on_fit_sigma)
        self.btn_cancel_sigma.clicked.connect(self._cancel_sigma)
        self.btn_copy_sigma.clicked.connect(self._copy_sigma)
        self.spin_sig_d0.valueChanged.connect(lambda _v: self._draw_sigma())
        self.spin_sig_d1.valueChanged.connect(lambda _v: self._draw_sigma())
        self.chk_sig_flagged.toggled.connect(lambda _v: self._draw_sigma())
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # spectral mode
        for cmb in (self.cmb_spad_view, self.cmb_spad_view_live):
            cmb.currentIndexChanged.connect(lambda _i, c=cmb: self._set_spad_view_mode(c.currentData()))
        self.btn_load_wl_cal.clicked.connect(self.on_load_wl_cal)
        self.btn_clear_wl_cal.clicked.connect(lambda: self._set_wl_calibration(None, "", "cleared"))
        self.spectralLive.line_clicked.connect(self._on_spec_live_line_clicked)
        self.spectralHist.line_clicked.connect(self._toggle_spec_line)
        self.spectralHist.time_clicked.connect(self._on_spec_time_clicked)
        self.spectralHist.status.connect(lambda text: self.statusBar().showMessage(text))
        self.spectralHist.fit_state_changed.connect(self._sync_controls)
        self.spectralHist.dark_sub_changed.connect(self._update_lifetime_tab)
        self._set_wl_calibration(None, "", persist=False)
        self._restore_wl_calibration()
        self._apply_spad_view()

    def _on_tab_changed(self, _index: int) -> None:
        """The Histogram / Lifetime tabs are redrawn only while visible; catch up when shown."""
        w = self.tabs.currentWidget()
        if w is self._hist_tab and self._hist_dirty:
            self._update_histogram_tab()
        elif w is self._lifetime_tab and self._lifetime_dirty:
            self._update_lifetime_tab()

    # -----------------
    # Helpers
    # -----------------
    def _make_dot(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setFixedSize(10, 10)
        lbl.setStyleSheet("border-radius:5px; background-color:#c62828;")
        return lbl

    def _make_conn_btn(self) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton("Connect")
        btn.setFixedWidth(100)
        return btn

    def _init_spad_live_map_plot(self) -> None:
        self.spadLiveMap.hover_text_fn = self._spad_hover_text
        self.spadLiveMap.pixel_clicked.connect(self._on_spad_live_map_click)
        self.spadLiveMap.set_values(self._spad_map_rates, unit="Mcps")
        self.spadLiveMap.set_selected([self._spad_selected_pixel_live])

    def _init_spad_hist_map_plot(self) -> None:
        self.spadHistMap.hover_text_fn = self._spad_hover_text
        self.spadHistMap.pixel_clicked.connect(self._on_spad_hist_map_click)
        self._update_spad_hist_markers()

    def _spad_hover_text(self, pix: int) -> str:
        snap = self._spad_last_count_snapshot
        if snap is None:
            return ""
        try:
            return f"{float(np.asarray(snap.pixel_rates_mcps)[int(pix)]):.3f} Mcps"
        except Exception:
            return ""

    def _update_spad_hist_markers(self) -> None:
        if not hasattr(self, "spadHistMap"):
            return
        self.spadHistMap.set_selected(self._spad_selected_hist_pixels)
        self.spadHistMap.set_excluded(self._spad_excluded_fit_pixels)
        # excluded pixels are drawn grey: re-apply the values after a change
        self.spadHistMap.set_values(self._spad_map_rates, unit="Mcps")

    def _update_tcspc_status(self, connected: bool, detail: str = "") -> None:
        self._set_dot(self.tcspcDot, bool(connected))
        text = detail or ("connected" if connected else "disconnected")
        if connected:  # always say which detector is in use (the combo may just read "Auto")
            how = "detected" if self._det_detected else "selected"
            text = f"{self._det.key} ({how}) | {text}"
        self.tcspcStatusLbl.setText(text)

    def _update_spad_count_maps(self, snap) -> None:
        rates = np.asarray(snap.pixel_rates_mcps, dtype=np.float64)
        # Auto levels give v1's min-max normalized colours; the colorbar shows Mcps.
        self._spad_map_rates = rates
        if hasattr(self, "spadLiveMap"):
            self.spadLiveMap.set_values(rates, unit="Mcps")
            self.spadLiveMap.set_selected([self._spad_selected_pixel_live])
            self.lbl_spad_live_info.setText(
                f"Pixel {self._spad_selected_pixel_live}: {rates[self._spad_selected_pixel_live]:.3f} Mcps | "
                f"Total {snap.total_rate_mcps:.3f} Mcps"
            )
            if self._spectral:
                self.spectralLive.update_frame(rates, self._spad_excluded_fit_pixels, self._wl_calib,
                                               self._spec_active_line, "Mcps")
        if hasattr(self, "spadHistMap"):
            self._update_spad_hist_markers()

    def _on_spad_live_map_click(self, pix: int, _button: int) -> None:
        # left or right click selects (v1: any button)
        self._spad_selected_pixel_live = int(pix)
        self._spad_selected_hist_pixels.add(int(pix))
        self.spadLiveMap.set_selected([self._spad_selected_pixel_live])
        if self._spectral:  # also selects the pixel's wavelength line
            line = self._spec_geom.line_of(pix)
            self._spec_active_line = line
            self._spec_lines.add(line)
            self._update_spec_highlights()
            self._refresh_spectral_live()
        self._update_spad_hist_markers()
        self._update_histogram_tab()
        self._update_lifetime_tab()

    def _on_spad_hist_map_click(self, pix: int, button: int) -> None:
        pix = int(pix)
        if button == 2:  # right click (PixelMapPlot button 2): exclude/include fit pixel
            if pix in self._spad_excluded_fit_pixels:
                self._spad_excluded_fit_pixels.remove(pix)
            else:
                self._spad_excluded_fit_pixels.add(pix)
            self._refresh_spectral_live()
            self._update_histogram_tab()
            self._update_lifetime_tab()
            return
        line = self._spec_geom.line_of(pix)
        if pix in self._spad_selected_hist_pixels:
            self._spad_selected_hist_pixels.remove(pix)
            # spectral: a line stays selected while another selected pixel is in it (at least one stays)
            g = self._spec_geom
            if (self._spectral and line in self._spec_lines and len(self._spec_lines) > 1
                    and not any(g.line_of(p) == line for p in self._spad_selected_hist_pixels)):
                self._spec_lines.discard(line)
                if self._spec_active_line == line:
                    self._spec_active_line = max(self._spec_lines)
        else:
            self._spad_selected_hist_pixels.add(pix)
            if self._spectral:
                self._spec_lines.add(line)
                self._spec_active_line = line
        if self._spectral:
            self._update_spec_highlights()
            self._refresh_spectral_live()
            self._update_lifetime_tab()
        self._update_histogram_tab()

    # ----------------- spectral mode (spectral.py): the SPAD's center line = wavelength axis
    def _make_view_combo(self) -> QtWidgets.QComboBox:
        cmb = QtWidgets.QComboBox()
        for key, label in VIEW_MODES:
            cmb.addItem(label, key)
        cmb.setCurrentIndex(max(0, cmb.findData(self._spad_view_mode)))
        cmb.setFixedWidth(100)
        cmb.setToolTip(
            "2D space: the SPAD images the sample (count map, 2D Gaussian fit, σ²(t)).\n"
            "Spectral: the SPAD sits behind the spectrometer; its center line (SPAD93: center row, 17 columns;\n"
            "SPAD23: center column, 5 rows) is a wavelength axis: spectrum, time-resolved spectrum, decays per\n"
            "wavelength and the width fit σ(t) of each line. View only: saved TRPL files stay raw.")
        return cmb

    @property
    def _spectral(self) -> bool:
        return self._spad_view_mode == MODE_SPECTRAL

    def _set_spad_view_mode(self, mode) -> None:
        if mode not in (MODE_2D, MODE_SPECTRAL):
            return
        if mode != self._spad_view_mode:
            self._spad_view_mode = mode
            self._det_settings.setValue(SPAD_VIEW_KEY, mode)
        self._apply_spad_view()
        self._update_histogram_tab()
        self._update_lifetime_tab()

    def _apply_spad_view(self) -> None:
        """Show the pages and map overlays of the current view mode."""
        on = self._spectral
        for cmb in (self.cmb_spad_view, self.cmb_spad_view_live):
            cmb.blockSignals(True)
            cmb.setCurrentIndex(max(0, cmb.findData(self._spad_view_mode)))
            cmb.blockSignals(False)
        self.spectralLive.setVisible(on)
        self._hist_left_stack.setCurrentIndex(1 if on else 0)
        self._hist_right_stack.setCurrentIndex(1 if on else 0)
        for m in (self.spadLiveMap, self.spadHistMap):
            m.set_spectral(on, self._wl_calib)
        self._update_spec_highlights()
        w = self._spec_geom.word
        self.lbl_hist_info.setText(
            f"Left click: select pixel(s) and their {w}s. Right click: exclude/include pixel." if on else
            "Left click: select histogram pixel(s). Right click: exclude/include fit pixel.")
        self._refresh_spectral_live()

    def _update_spec_highlights(self) -> None:
        g = self._spec_geom
        if self._spectral:
            g.highlight(self.spadLiveMap, [self._spec_active_line])
            g.highlight(self.spadHistMap, self._spec_lines)
        self.spectralHist.set_lines(self._spec_lines, self._spec_active_line)

    def _refresh_spectral_live(self) -> None:
        snap = self._spad_last_count_snapshot
        if self._spectral and snap is not None:
            self.spectralLive.update_frame(np.asarray(snap.pixel_rates_mcps, dtype=np.float64),
                                           self._spad_excluded_fit_pixels, self._wl_calib,
                                           self._spec_active_line, "Mcps")

    def _on_spec_live_line_clicked(self, line: int) -> None:
        """Click on the live spectrum: that line becomes the active one."""
        self._spec_active_line = int(line)
        self._spec_lines.add(int(line))
        self._update_spec_highlights()
        self._refresh_spectral_live()
        self._update_histogram_tab()
        self._update_lifetime_tab()

    def _toggle_spec_line(self, line: int) -> None:
        """Click on the spectrum at t_ref: add the line, or drop the active one (one always stays)."""
        line = int(line)
        if line in self._spec_lines and len(self._spec_lines) > 1 and line == self._spec_active_line:
            self._spec_lines.discard(line)
            self._spec_active_line = max(self._spec_lines)
        else:
            self._spec_lines.add(line)
            self._spec_active_line = line
        self._update_spec_highlights()
        self._refresh_spectral_live()
        self._update_lifetime_tab()

    def _on_spec_time_clicked(self, t_ns: float) -> None:
        """Click on the spectral image / decays: that time becomes the reference (spectrum at t_ref)."""
        self.histRefNsSpin.setValue(float(t_ns))

    def on_load_wl_cal(self) -> None:
        start = os.path.dirname(self._wl_calib_path) if self._wl_calib_path else DATA_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load {self._det.key} wavelength calibration", start, "JSON files (*.json);;All files (*)")
        if path:
            self._load_wl_calibration(path)

    def _load_wl_calibration(self, path: str, quiet: bool = False) -> bool:
        try:
            calib = self._spec_geom.load_calibration(path)
        except Exception as exc:
            msg = f"Could not load the wavelength calibration {os.path.basename(path)}: {exc}"
            if quiet:
                self.statusBar().showMessage(msg)
            else:
                QtWidgets.QMessageBox.warning(self, "Wavelength calibration", msg)
            return False
        self._set_wl_calibration(calib, path, "loaded")
        return True

    def _set_wl_calibration(self, calib, path: str, how: str = "", persist: bool = True) -> None:
        """Make ``calib`` (None = none) the wavelength calibration of the spectral views."""
        self._wl_calib = calib
        self._wl_calib_path = path if calib is not None else ""
        if persist:
            self._det_settings.setValue(WLCAL_PATH_KEY + self._det.key, self._wl_calib_path)
        g = self._spec_geom
        self.lbl_wl_cal.setText(g.calib_short(calib))
        tip = "" if calib is None else calib.describe() + (f"\n{self._wl_calib_path}" if self._wl_calib_path else "")
        if calib is not None and not calib.geometry_matches:
            tip += "\nWarning: made with a different sensor orientation (flip) than this detector uses."
        self.lbl_wl_cal.setToolTip(tip)
        self.btn_clear_wl_cal.setEnabled(calib is not None)
        for m in (self.spadLiveMap, self.spadHistMap):
            m.set_spectral(self._spectral, calib)
        self._update_spec_highlights()
        self.spectralHist.set_calibration(calib)
        self._refresh_spectral_live()
        if how:
            self.statusBar().showMessage(f"Wavelength calibration {how}: {g.calib_short(calib)}")
        self._update_lifetime_tab()

    def _restore_wl_calibration(self) -> None:
        """The last calibration JSON of the active detector (QSettings), if it still exists."""
        path = str(self._det_settings.value(WLCAL_PATH_KEY + self._det.key, "", type=str) or "")
        if path and os.path.exists(path):
            self._load_wl_calibration(path, quiet=True)

    def _copy_wlcal_to_sweep_folder(self) -> None:
        """Put the wavelength calibration JSON in the sweep folder as WLCAL_*.json."""
        self._sweep_wlcal_file = None
        if self._wl_calib is None or not self._save_dir:
            return
        src = self._wl_calib_path
        name = os.path.basename(src) if src else f"{self._det.key.lower()}_wavelength_calibration.json"
        if not name.startswith("WLCAL_"):
            name = "WLCAL_" + name
        dst = os.path.join(self._save_dir, name)
        try:
            if src and os.path.exists(src):
                if os.path.abspath(dst) != os.path.abspath(src):
                    shutil.copy2(src, dst)
            else:
                self._det.io.save_calibration_json(dst, self._wl_calib)
            self._sweep_wlcal_file = name
        except Exception as exc:
            self.statusBar().showMessage(f"Could not copy the wavelength calibration into the sweep folder: {exc}")

    def _wl_calib_meta(self) -> Optional[dict]:
        c = self._wl_calib
        if c is None:
            return None
        return {"device": self._det.key, "slope_nm_per_um": float(c.slope_nm_per_um),
                "intercept_nm": float(c.intercept_nm), "created": c.created,
                "line_wavelengths_nm": [round(float(v), 4) for v in self._spec_geom.wavelengths(c)]}

    def _spectral_key(self, snapshot) -> tuple:
        """Everything the spectral width fit of ``snapshot`` depends on (besides the page's options)."""
        start_idx, end_idx, _ = self._hist_time_indices(snapshot)
        return (id(snapshot), self.chk_t0_align.isChecked(), self._hist_bg_key(), start_idx, end_idx,
                tuple(sorted(self._spad_excluded_fit_pixels)), self._det.key)

    def _spectral_fit_allowed(self) -> bool:
        return self._sigma_fit_allowed()

    def _update_spectral_hist(self) -> None:
        """Hand the latest histogram to the spectral page (Histogram tab in spectral mode)."""
        snapshot = self._spad_latest_hist_snapshot
        if snapshot is None or snapshot.counts_matrix.shape[0] == 0:
            self.spectralHist.set_context(None)
            return
        t_ns, counts = self._hist_view(snapshot)
        start_idx, end_idx, ref_idx = self._hist_time_indices(snapshot)
        bg = self._hist_background(snapshot, t_ns, counts)  # also updates the background label
        dark, dark_text = self._dark_ref_scaled(snapshot)
        bg_key = self.cmb_hist_bg.currentData()
        bg_text = {"pre_t0": "pre-t0 offset subtracted", "dark_ref": "dark reference subtracted"}.get(
            bg_key, "each frame's 10th percentile subtracted")
        if bg is None and bg_key != "none":
            bg_text = f"each frame's 10th percentile subtracted ({self.cmb_hist_bg.currentText()} unavailable)"
        meta = {
            "device": self._det.key,
            "source": "PL+TRPL v2 Histogram tab (latest histogram)",
            "acquisition_s": float(snapshot.measurement_ms) / 1000.0,
            "bin_width_ps": float(snapshot.bin_width_ps),
            "time_zero_alignment": "first peak per pixel" if self.chk_t0_align.isChecked() else "off (raw)",
            "excluded_pixels": sorted(int(p) for p in self._spad_excluded_fit_pixels),
            "wavelength_calibration_file": self._wl_calib_path or None,
        }
        self.spectralHist.set_context(SpectralContext(
            key=self._spectral_key(snapshot), snapshot=snapshot, time_ns=t_ns, counts=counts,
            excluded=tuple(sorted(self._spad_excluded_fit_pixels)),
            start_ns=float(t_ns[start_idx]), end_ns=float(t_ns[end_idx]), ref_ns=float(t_ns[ref_idx]),
            dark=dark, dark_text=dark_text, fit_bg=bg, bg_text=bg_text, log_y=self.chk_hist_log.isChecked(),
            meta=meta, save_dir=self._save_dir or DATA_DIR,
            base_name=f"{self._det.key.lower()}_hist_{time.strftime('%Y%m%d_%H%M%S')}",
        ))

    def _hist_time_indices(self, snapshot) -> Tuple[int, int, int]:
        t_ns = self._hist_view(snapshot)[0]
        if self._hist_rel and self.chk_hist_ref_auto.isChecked() and t_ns.size:
            self._set_spin_silently(self.histRefNsSpin, float(t_ns[t0_ref_index(t_ns)]))
        start_ns = float(self.histStartNsSpin.value())
        end_ns = float(self.histEndNsSpin.value())
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns
        start_idx = int(np.searchsorted(t_ns, start_ns, side="left"))
        end_idx = int(np.searchsorted(t_ns, end_ns, side="right") - 1)
        start_idx = int(np.clip(start_idx, 0, snapshot.counts_matrix.shape[0] - 1))
        end_idx = int(np.clip(end_idx, start_idx, snapshot.counts_matrix.shape[0] - 1))
        ref_ns = float(self.histRefNsSpin.value())
        ref_idx = int(np.argmin(np.abs(t_ns - ref_ns)))
        ref_idx = int(np.clip(ref_idx, start_idx, end_idx))
        return start_idx, end_idx, ref_idx

    def _spad_fit_inputs(self, values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        mask = np.ones(self._det.num_pixels, dtype=bool)
        if self._spad_excluded_fit_pixels:
            idx = np.array(sorted(self._spad_excluded_fit_pixels), dtype=int)
            mask[idx] = False
        if int(np.sum(mask)) < int(self._det.an.MIN_FIT_PIXELS_2D):
            return None
        return self._spad_fit_x_um[mask], self._spad_fit_y_um[mask], values[mask], mask

    def _on_histogram_update(self, snapshot_obj: object) -> None:
        snapshot = snapshot_obj if is_trpl_snapshot(snapshot_obj) else None
        if snapshot is None:
            return
        if self._dark_pending:  # a dark reference is never shown as a histogram
            self.statusBar().showMessage(f"Dark reference: {snapshot.elapsed_s:.1f} / "
                                         f"{snapshot.measurement_ms / 1000.0:.1f} s, {snapshot.n_events} events")
            return
        self._show_live_hist(snapshot)

    LIVE_HIST_MIN_S = 0.4
    LIVE_HIST_COST_X = 3.0  # the GUI thread spends at most ~1/3 of its time on live histogram redraws

    def _show_live_hist(self, snapshot) -> None:
        """A live (partial) TRPL snapshot: keep the newest, redraw when the throttle allows."""
        self._spad_latest_hist_snapshot = snapshot
        if not self._live_hist_timer.isActive():
            delay = max(0.0, self._live_hist_next - time.monotonic())
            self._live_hist_timer.start(int(delay * 1000.0))

    def _flush_live_hist(self) -> None:
        t0 = time.monotonic()
        try:
            self._update_histogram_tab()
        finally:
            cost = time.monotonic() - t0
            self._live_hist_next = time.monotonic() + max(self.LIVE_HIST_MIN_S, self.LIVE_HIST_COST_X * cost)

    def _apply_crop_to_liveview(self) -> None:
        if self.liveView is None:
            return
        raw_top = int(self._crop_bottom)
        raw_bottom = int(self._crop_top)
        self.liveView.set_crop(raw_top, raw_bottom, int(self._crop_left), int(self._crop_right))

    def _set_dot(self, lbl: QtWidgets.QLabel, ok: bool) -> None:
        color = "#2e7d32" if ok else "#c62828"
        lbl.setStyleSheet(f"border-radius:5px; background-color:{color};")

    def _default_save_name(self) -> str:
        return time.strftime("pl_trpl_%Y%m%d_%H%M%S")

    def on_refresh_save_name(self) -> None:
        """Put the current date/time into the subfolder name (keeps any text around the timestamp)."""
        name = self.saveNameEdit.text().strip()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if re.search(r"\d{8}_\d{6}", name):
            name = re.sub(r"\d{8}_\d{6}", stamp, name, count=1)
        else:
            name = self._default_save_name()
        self.saveNameEdit.setText(name)

    def _resolve_save_dir(self) -> str:
        name = self.saveNameEdit.text().strip()
        if not name:
            name = self._default_save_name()
            self.saveNameEdit.setText(name)
        base = os.path.join(DATA_DIR, name)
        os.makedirs(base, exist_ok=True)
        self._save_dir = base
        return base

    def _apply_camera_defaults(self) -> None:
        """Camera defaults on connect (worker thread; the PL GUI's ``_apply_camera_defaults``)."""
        try:
            self.cam.set_frame_api("Snap")
            self.cam.set_acquisition_mode("Single")
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_fastest_readout_default()
        except Exception:
            pass
        try:
            if andor_cfg is not None:
                self.cam.set_temperature_setpoint(float(andor_cfg.DEFAULT_SETPOINT_C))
                self.cam.set_cooler(bool(andor_cfg.DEFAULT_COOLER_ON))
                self.cam.set_baseline_clamp(bool(andor_cfg.DEFAULT_BASELINE_CLAMP))
        except Exception:
            pass

    def _camera_setup_io(self) -> Dict:
        """Worker thread, right after ``cam.connect()``: defaults, amp info and choices."""
        self._apply_camera_defaults()
        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None
        choices = ([], [], [])
        try:
            choices = self.cam.get_amp_mode_choices()
        except Exception:
            pass
        return {"info": info, "choices": choices}

    def _refresh_amp_choices(self, choices) -> None:
        """Fill the rate / preamp combos from ``get_amp_mode_choices()`` (read in the worker)."""
        current_gain = self.preampCombo.currentText().strip()
        current_rate = self.readoutRateCombo.currentText().strip()
        _, rates, gains = [], [], []
        try:
            _, rates, gains = choices
        except Exception:
            pass
        if gains:
            self.preampCombo.blockSignals(True)
            self.preampCombo.clear()
            self.preampCombo.addItems(gains)
            if str(DEFAULT_PREAMP_GAIN) in gains:
                self.preampCombo.setCurrentText(str(DEFAULT_PREAMP_GAIN))
            elif current_gain and current_gain in gains:
                self.preampCombo.setCurrentText(current_gain)
            else:
                self.preampCombo.setCurrentIndex(0)
            self.preampCombo.blockSignals(False)
        if rates:
            self.readoutRateCombo.blockSignals(True)
            self.readoutRateCombo.clear()
            self.readoutRateCombo.addItems(rates)
            if str(DEFAULT_READOUT_RATE) in rates:
                self.readoutRateCombo.setCurrentText(str(DEFAULT_READOUT_RATE))
            elif current_rate and current_rate in rates:
                self.readoutRateCombo.setCurrentText(current_rate)
            else:
                self.readoutRateCombo.setCurrentIndex(0)
            self.readoutRateCombo.blockSignals(False)

    def _apply_amp_settings(self) -> None:
        """Set amp mode / readout rate / preamp gain in the camera's worker thread."""
        if not self.camDev.is_connected:
            return
        req_gain = self.preampCombo.currentText().strip()
        req_rate = self.readoutRateCombo.currentText().strip() or self._readout_rate_label
        req_amp = self._output_amp_label
        if not self.camDev.call(lambda: self._amp_apply_io(req_amp, req_rate, req_gain), self._on_amp_applied):
            self.statusBar().showMessage("Camera busy: preamp not applied")

    def _amp_apply_io(self, req_amp: str, req_rate: str, req_gain: str):
        """Worker thread: the PL GUI's amp-setting logic; returns (ok, info, error text)."""
        err = ""

        def _gain_value(label: str):
            m = re.search(r"(\\d+(?:\\.\\d+)?)", str(label))
            if m:
                return float(m.group(1))
            return None

        want_gain = _gain_value(req_gain)
        ok = True
        try:
            ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
        except Exception as exc:
            ok = False
            err = f"Preamp set failed: {exc}"

        info = None
        try:
            info = self.cam.get_amp_mode_info()
        except Exception:
            info = None

        if want_gain is not None:
            cur_gain = None
            if info is not None:
                cur_gain = info.get("preamp_gain")
                if cur_gain is None:
                    cur_gain = _gain_value(info.get("gain_label"))
            if cur_gain is None or abs(float(cur_gain) - float(want_gain)) > 0.05:
                try:
                    ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
                except Exception:
                    ok = False
                try:
                    info = self.cam.get_amp_mode_info()
                except Exception:
                    info = None
        try:  # readout time at these settings, for the scan time estimate
            rt = self.cam.cam.get_readout_time() if self.cam.cam is not None else None
            if rt:
                info = dict(info or {}, readout_s=float(rt))
        except Exception:
            pass
        return ok, info, err

    def _on_amp_applied(self, res, call_err: str) -> None:
        if call_err or res is None:
            self.statusBar().showMessage(f"Preamp set failed: {call_err}")
            return
        ok, info, err = res
        if err:
            self.statusBar().showMessage(err)
        self._apply_amp_info(info)
        if info and info.get("readout_s"):
            self._cam_readout_s[self._readout_rate_label] = float(info["readout_s"])
        if not ok:
            self.statusBar().showMessage("Preamp gain not applied")

    def _apply_amp_info(self, info) -> None:
        if info:
            self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = info.get("amp_label") or self._output_amp_label
            rate_label = info.get("rate_label")
            if rate_label:
                self.readoutRateCombo.blockSignals(True)
                self.readoutRateCombo.setCurrentText(str(rate_label))
                self.readoutRateCombo.blockSignals(False)
            gain_label = info.get("gain_label")
            if gain_label:
                self.preampCombo.blockSignals(True)
                self.preampCombo.setCurrentText(str(gain_label))
                self.preampCombo.blockSignals(False)

    def _spec_params(self) -> Dict:
        return {
            "grating": int(self.gratingSpin.value()),
            "center_nm": float(self.centerSpin.value()),
            "slit_um": float(self.slitSpin.value()),
        }

    def _apply_spec_settings(self) -> None:
        """Grating / center / slit in the spectrograph's worker, then the wavelength axis."""
        if not self.specDev.is_connected:
            return
        params = self._spec_params()
        if not self.specDev.call(lambda: self.specDev.apply_settings(params), self._on_spec_applied):
            self.statusBar().showMessage("Spectrograph busy")

    def _on_spec_applied(self, res, err: str) -> None:
        if err:
            self.statusBar().showMessage(f"Spectrograph settings failed: {err}")
            return
        self._wl_read_at = None  # the next readback is the new baseline
        self._set_wavelength_axis(res)
        self._refresh_linecut_maps()
        self.statusBar().showMessage("Spectrograph settings applied")

    def _set_wavelength_axis(self, res) -> None:
        """PL GUI rule: wavelength axis on unless the center is 0 nm (axis read in the worker)."""
        if self.liveView is None:
            return
        center = float(self.centerSpin.value())
        wl = res.get("wavelength_axis") if isinstance(res, dict) else None
        self.liveView.set_wavelength_axis_data(None if abs(center) <= 1e-9 else wl)
        if abs(center) > 1e-9 and wl is None:
            self._wl_retry_at = time.monotonic() + 2.0  # the spectrograph poll retries
            self._warn_no_wavelength()

    def _warn_no_wavelength(self) -> None:
        if not self._wl_warned:
            self._wl_warned = True
            why = self.specDev.last_wl_error
            self.statusBar().showMessage("No wavelength calibration from the spectrograph"
                                         + (f" ({why})" if why else "") + ": image and linecut stay in pixels (retrying)")

    def _check_wavelength_axis(self, d: dict) -> None:
        """Keep the wavelength axis in step with the spectrograph (v1 re-reads it on every frame).

        From the 1 s spectrograph poll: read it when it is missing or the grating / center moved since
        it was read; no axis while the center is 0 nm (the PL GUI rule)."""
        if self.liveView is None or not self.specDev.is_connected or self._wl_fetch_pending:
            return
        gr, cw = d.get("grating"), d.get("center_nm")
        if cw is None:
            return
        have = self.liveView.get_wavelength_axis() is not None
        if abs(float(cw)) <= 1e-9:
            self._wl_read_at = (gr, float(cw))
            if have:
                self.liveView.set_wavelength_axis_data(None)
                self._refresh_linecut_maps()
            return
        if have and self._wl_read_at is None:
            self._wl_read_at = (gr, float(cw))  # read at connect / Apply Spec
            return
        moved = self._wl_read_at is not None and (gr != self._wl_read_at[0]
                                                  or abs(float(cw) - self._wl_read_at[1]) > 0.05)  # readback jitter is not a move
        if have and not moved:
            return
        if not have and not moved and time.monotonic() < self._wl_retry_at:
            return
        if self.specDev.call_running:
            return
        self._wl_fetch_pending = True

        def done(wl, err: str) -> None:
            self._wl_fetch_pending = False
            if wl is None:
                # keep an axis we already have (a failed re-read must not drop it); retry later
                self._wl_retry_at = time.monotonic() + 5.0
                self._warn_no_wavelength()
                return
            self._wl_read_at = (gr, float(cw))
            if self._wl_warned:
                self._wl_warned = False
                self.statusBar().showMessage("Wavelength axis read from the spectrograph")
            self.liveView.set_wavelength_axis_data(wl)
            self._refresh_linecut_maps()

        if not self.specDev.call(self.specDev.read_wavelength_axis, done):
            self._wl_fetch_pending = False

    def _prepare_live_mode(self) -> None:  # worker thread (camera call)
        try:
            self.cam.set_frame_api("Stream+Buffer")
        except Exception:
            pass
        try:
            self.cam.set_acquisition_mode("Run till abort")
        except Exception:
            pass
        try:
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        except Exception:
            pass

    def _prepare_single_mode(self) -> None:  # worker thread (camera call)
        try:
            if hasattr(self.cam, "stop_stream"):
                self.cam.stop_stream()
        except Exception:
            pass
        try:
            self.cam.set_frame_api("Snap")
        except Exception:
            pass
        try:
            self.cam.set_acquisition_mode("Single")
        except Exception:
            pass
        try:
            self.cam.set_trigger_mode("Internal")
        except Exception:
            pass
        try:
            self.cam.set_exposure_ms(float(self.exposureSpin.value()))
        except Exception:
            pass

    def _parse_angle_list(self) -> List[float]:
        """Parse the angle list text field into a sorted list of unique floats."""
        vals: List[float] = []
        for tok in self.angleListEdit.text().replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        seen: set = set()
        unique: List[float] = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def _on_angle_list_changed(self) -> None:
        angles = self._parse_angle_list()
        if angles:
            preview = f"{len(angles)} angles: {[f'{a:.3g}' for a in angles[:5]]}"
            if len(angles) > 5:
                preview += " …"
        else:
            preview = "(0 angles)"
        self.angleListPreviewLbl.setText(preview)
        self._sync_controls()

    def _on_angle_list_toggled(self, on: bool) -> None:
        if on and self.powerListChk.isChecked():
            self.powerListChk.setChecked(False)  # the two lists are exclusive
        self._on_angle_list_changed()

    # ----------------- power list (HWP+ND calibration)
    def _power_list_active(self) -> bool:
        return self.powerListChk.isChecked() and self._calib_mode == "hwp_nd"

    def _on_power_list_toggled(self, on: bool) -> None:
        if on and self.angleListChk.isChecked():
            self.angleListChk.setChecked(False)
        self._power_list_timer.stop()
        if self._calib_mode == "hwp_nd":
            self._on_hwp_nd_angle_changed()  # power list on: solve; off: back to the ND-angle entries
        else:
            self._update_power_list_preview()
            self._sync_controls()

    def _drop_power_list(self) -> None:
        """A calibration without HWP+ND data was loaded: the power list does not apply."""
        self._power_plan = []
        self.powerListChk.blockSignals(True)
        self.powerListChk.setChecked(False)
        self.powerListChk.blockSignals(False)
        self._update_power_list_preview()

    def _apply_power_list(self) -> None:
        """``_dual_entries`` = the solved (HWP = A, ND = B) pair of every reachable target, in list order."""
        data = self._hwp_nd_calib_data
        if data is None:
            return
        targets, self._power_list_bad = hwp_planner.parse_power_list(
            self.powerListEdit.text(), self.powerListUnitCombo.currentText())
        rows: List[hwp_planner.PlanRow] = []
        err = ""
        if targets:
            try:
                grid = hwp_planner.grid_from_hwp_nd(data, hwp_lim=(self.hwpMinSpin.value(), self.hwpMaxSpin.value()))
                rows = hwp_planner.solve(grid, targets, self.powerListTolSpin.value() / 100.0)
            except ValueError as exc:
                err = str(exc)
        self._power_plan = rows
        entries = [DualWheelPowerEntry(a_deg=r.hwp_deg, b_deg=r.nd_deg, power_w=r.power_w,
                                       intensity=float("nan"), index=i)
                   for i, r in enumerate(r for r in rows if r.ok)]
        self._dual_entries = entries
        if entries:
            self._calib_unit, self._calib_scale = hwp_planner.dual_unit(max(e.power_w for e in entries))
        self._build_power_display()
        self._total_entries = len(entries)
        self._reset_sweep_data()
        self._update_series_info()
        self._update_power_list_preview(err)
        self._sync_controls()
        if hasattr(self, "_scan_est_timer"):
            self._scan_est_timer.start()

    def _update_power_list_preview(self, err: str = "") -> None:
        bad = f"  ignored: {' '.join(self._power_list_bad)}" if self._power_list_bad else ""
        if not self._power_list_active():
            self.powerListPreviewLbl.setStyleSheet("color: #555;")
            self.powerListPreviewLbl.setText(
                "(load a wheel+HWP calibration)" if self._calib_mode != "hwp_nd" else "(off)")
            return
        rows = self._power_plan
        if err or not rows:
            self.powerListPreviewLbl.setStyleSheet("color: #b00020;" if err else "color: #555;")
            self.powerListPreviewLbl.setText((err or "(0 powers)") + bad)
            return
        ok = [r for r in rows if r.ok]
        txt = f"{len(rows)} powers → {len(ok)} within ±{self.powerListTolSpin.value():g} %"
        if ok:
            txt += f" (worst {max(abs(r.err) for r in ok) * 100:.2f} %)"
        n_bad = len(rows) - len(ok)
        if n_bad:
            txt += f";  {n_bad} unreachable, skipped"
        self.powerListPreviewLbl.setStyleSheet("color: #b00020;" if n_bad or bad else "color: #555;")
        self.powerListPreviewLbl.setText(txt + bad)

    def _show_power_list_details(self) -> None:
        rows = self._power_plan
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Power list: solved HWP / ND pairs")
        lay = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(rows), 7)
        table.setHorizontalHeaderLabels(
            ["Target", "Predicted", "Error (%)", "±1σ", "HWP = A (°)", "ND = B (°)", "Note"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        bad_bg = QtGui.QBrush(QtGui.QColor("#fde2e1"))
        for i, r in enumerate(rows):
            cells = [hwp_planner.fmt_power(r.target_w), hwp_planner.fmt_power(r.power_w), f"{r.err * 100:+.3f}",
                     hwp_planner.fmt_power(r.sigma_w, 3), f"{r.hwp_deg:.2f}", f"{r.nd_deg:.2f}",
                     r.note or "swept"]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if not r.ok:
                    item.setBackground(bad_bg)
                table.setItem(i, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table)
        if self._hwp_nd_calib_data is not None:
            lay.addWidget(QtWidgets.QLabel(f"Calibration: {os.path.basename(self._hwp_nd_calib_data.path)}"))
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.resize(720, 420)
        dlg.exec_()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._update_device_pauses()
        self._sync_controls()

    def _acquisition_busy(self) -> bool:
        live = self._live_thread is not None and self._live_thread.isRunning()
        accum = self._accum_thread is not None and self._accum_thread.isRunning()
        sweep = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return self._busy or live or accum or sweep

    def _update_device_pauses(self) -> None:
        """v1's rule (``_set_busy`` stopped the poll timers): no readback polling while an
        acquisition owns the hardware; gate polls also pause while a gate ramps."""
        acq = self._acquisition_busy()
        for dev in (self.camDev, self.specDev, self.stageDevs["A"], self.stageDevs["B"], self.flipperDev):
            dev.pause() if acq else dev.resume()
        gates_busy = acq or self._gate_threads_running()
        for dev in self.gateDevs.values():
            dev.pause() if gates_busy else dev.resume()

    def _start_when_quiet(self, devices, fn: Callable[[], None]) -> None:
        """Run ``fn`` once the polls of ``devices`` have stopped touching the hardware."""
        t0 = time.monotonic()

        def check():
            if all(d.quiet for d in devices) or time.monotonic() - t0 > QUIET_WAIT_MAX_S:
                fn()
            else:
                QtCore.QTimer.singleShot(20, check)

        check()

    def _measure_mode(self) -> str:
        return self.measureCombo.currentText().strip()

    def _measure_pl(self) -> bool:
        return self._measure_mode() in (MEASURE_PL, MEASURE_BOTH)

    def _measure_trpl(self) -> bool:
        return self._measure_mode() in (MEASURE_TRPL, MEASURE_BOTH)

    def _spad_ok(self) -> bool:
        return self.spadDev.state in (spad_det.IDLE, spad_det.LIVE)

    def _sync_conn_buttons(self, busy: bool) -> None:
        """Connect / Disconnect button of every device (text follows the state)."""
        used = self._sweep_thread is not None and self._sweep_thread.isRunning()
        rows = (
            (self.camConnBtn, self.camDev.state, busy),
            (self.specConnBtn, self.specDev.state, busy),
            (self.stageConnBtn, self.stageDevs["A"].state, busy or self._stage_busy_by["A"]),
            (self.stageBConnBtn, self.stageDevs["B"].state, busy or self._stage_busy_by["B"]),
            (self.frontConnBtn, self.gateDevs["front"].state, used or self._gate_threads_running()),
            (self.backConnBtn, self.gateDevs["back"].state, used or self._gate_threads_running()),
            (self.flipperConnBtn, self.flipperDev.state, busy),
        )
        for btn, state, locked in rows:
            if state == CONNECTED:
                btn.setText("Disconnect")
                btn.setEnabled(not locked)
            elif state == DISCONNECTED:
                btn.setText("Connect")
                btn.setEnabled(True)
            else:
                btn.setText("Connecting..." if state == CONNECTING else "Closing...")
                btn.setEnabled(False)
        if self.specDev.state == DISCONNECTED and not self.camDev.is_connected:
            self.specConnBtn.setEnabled(False)
        st = self.spadDev.state
        detecting = self._detect_worker is not None
        # the detector can only change while the pSPAD server is not in use
        self.detectorCombo.setEnabled(st == spad_det.DISCONNECTED and not detecting and not used)
        if st in (spad_det.IDLE, spad_det.LIVE):
            self.tcspcConnBtn.setText("Disconnect")
            self.tcspcConnBtn.setEnabled(not used and not self.spadDev.trpl_running)
        elif st == spad_det.DISCONNECTED and detecting:
            self.tcspcConnBtn.setText("Detecting...")
            self.tcspcConnBtn.setEnabled(False)
        elif st == spad_det.DISCONNECTED:
            self.tcspcConnBtn.setText("Connect")
            self.tcspcConnBtn.setEnabled(True)
        else:
            self.tcspcConnBtn.setText("Connecting..." if st == spad_det.CONNECTING else "Busy...")
            self.tcspcConnBtn.setEnabled(False)

    # -----------------
    # Dynamic sweep TRPL time
    # -----------------
    def _dyn_on(self) -> bool:
        return self.trplTimeCombo.currentIndex() == 1

    def _pl_time_s(self) -> float:
        return float(self.exposureSpin.value()) * int(self.accumSpin.value()) / 1000.0

    def _dyn_first_from_pl(self) -> bool:
        return self.dynFirstPlChk.isChecked() and self._measure_pl()

    def _dyn_first_s(self) -> float:
        return max(0.1, self._pl_time_s()) if self._dyn_first_from_pl() else float(self.dynFirstSpin.value())

    def _sync_dyn_controls(self, busy: Optional[bool] = None) -> None:
        if busy is None:
            busy = self._is_busy()
        self.dynCtrlWidget.setVisible(self._dyn_on())
        self.trplTimeCombo.setEnabled(not busy)
        for w in (self.dynTargetSpin, self.dynFirstPlChk, self.dynStepSpin, self.dynMaxSpin):
            w.setEnabled(not busy)
        from_pl = self._dyn_first_from_pl()
        self.dynFirstSpin.setEnabled(not busy and not from_pl)
        if from_pl:
            self.dynFirstSpin.blockSignals(True)  # shows the PL time; the saved own value is kept
            self.dynFirstSpin.setValue(self._dyn_first_s())
            self.dynFirstSpin.blockSignals(False)
            self.dynFirstSpin.setToolTip(f"= PL time ({self._pl_time_s():g} s = exposure × accumulations)")
        else:
            self.dynFirstSpin.setToolTip("Length of the first chunk")

    def _save_dyn_settings(self, *_args) -> None:
        ds = self._det_settings
        ds.setValue("trpl_time_mode", "dynamic" if self._dyn_on() else "fixed")
        ds.setValue("trpl_dyn_target", int(self.dynTargetSpin.value()))
        if not self._dyn_first_from_pl():
            ds.setValue("trpl_dyn_first_s", float(self.dynFirstSpin.value()))
        ds.setValue("trpl_dyn_first_pl", bool(self.dynFirstPlChk.isChecked()))
        ds.setValue("trpl_dyn_step_s", float(self.dynStepSpin.value()))
        ds.setValue("trpl_dyn_max_s", float(self.dynMaxSpin.value()))

    def _wire_dyn_controls(self) -> None:
        self.trplTimeCombo.currentIndexChanged.connect(lambda _i: self._sync_dyn_controls())
        self.dynFirstPlChk.toggled.connect(lambda _b: self._sync_dyn_controls())
        for sig in (self.exposureSpin.valueChanged, self.accumSpin.valueChanged,
                    self.measureCombo.currentIndexChanged):
            sig.connect(lambda *_a: self._sync_dyn_controls())
        for sig in (self.trplTimeCombo.currentIndexChanged, self.dynTargetSpin.valueChanged,
                    self.dynFirstSpin.valueChanged, self.dynFirstPlChk.toggled, self.dynStepSpin.valueChanged,
                    self.dynMaxSpin.valueChanged):
            sig.connect(self._save_dyn_settings)
        self._sync_dyn_controls()

    def _dyn_usable_dark(self):
        ref = self._dark_ref
        return ref if ref is not None and ref.num_pixels == self._det.num_pixels else None

    def _dyn_settings(self, with_background: bool = True) -> Optional["tdyn.DynamicSettings"]:
        """The dynamic TRPL settings of the sweep (None = Fixed time, or TRPL not measured)."""
        if not (self._dyn_on() and self._measure_trpl()):
            return None
        kw = {}
        if with_background:
            win = None if self._pre_t0_auto else (float(self.spin_pre_t0_start.value()),
                                                  float(self.spin_pre_t0_end.value()))
            if win is not None and self._hist_rel and self._hist_last_off is not None:
                win = (win[0] + self._hist_last_off, win[1] + self._hist_last_off)  # raw times (worker)
            kw = {"dark_ref": self._dyn_usable_dark(), "pre_t0_window": win,
                  "excluded": tuple(sorted(int(p) for p in self._spad_excluded_fit_pixels))}
        return tdyn.DynamicSettings(target_counts=float(self.dynTargetSpin.value()), first_s=self._dyn_first_s(),
                                    step_s=float(self.dynStepSpin.value()), max_s=float(self.dynMaxSpin.value()),
                                    **kw)

    def _dyn_record(self, payload: dict) -> None:
        """One dynamic spot came in (trpl_done): keep it for the summary, report it."""
        dyn = payload.get("dyn")
        if not isinstance(dyn, dict):
            return
        fp = payload.get("filepath")
        self._dyn_results.append(dict(dyn, file=os.path.basename(fp) if fp else "", pump2=payload.get("pump2")))
        if payload.get("pump2"):
            return
        spots = [r for r in self._dyn_results if not r.get("pump2")]
        low = sum(not r.get("reached") for r in spots)
        flag = "" if dyn.get("reached") else "  ** BELOW TARGET (max time) **"
        self.statusBar().showMessage(
            f"TRPL spot: {dyn.get('total_s', 0):g} s in {dyn.get('n_chunks', 0)} chunk(s), peak "
            f"{dyn.get('peak', 0):.0f}/{dyn.get('target', 0):g}{flag}  |  {low} of {len(spots)} spot(s) below target")

    def _write_dynamic_summary(self, status: str) -> None:
        """Append the dynamic TRPL summary to session.md (also after an abort)."""
        st = getattr(self, "_sweep_dyn", None)
        if st is None or not self._save_dir:
            return
        spots = [r for r in self._dyn_results if not r.get("pump2")]
        repeats = [r for r in self._dyn_results if r.get("pump2")]
        if not self._session_saved:
            self._write_session_metadata()
        totals = [float(r.get("total_s", 0.0)) for r in spots]
        low = [r for r in spots if not r.get("reached")]
        lines = ["", "## TRPL dynamic", f"- Status: {status}",
                 f"- Spots decided: {len(spots)}"
                 + (f" (+ {len(repeats)} pump-2-on acquisitions that repeat pump 2 off's time)" if repeats else ""),
                 f"- Reached the target ({st.target_counts:g} counts): {len(spots) - len(low)}; "
                 f"below target (stopped at the max {st.max_s:g} s): {len(low)}"]
        if totals:
            lines.append(f"- TRPL time per spot (s): min {min(totals):g}, median {float(np.median(totals)):g}, "
                         f"max {max(totals):g}; total {sum(totals) + sum(float(r.get('total_s', 0)) for r in repeats):g}")
        lines.append(f"- Per-spot log: {tdyn.LOG_NAME}; every chunk: {tdyn.CHUNK_DIR}/")
        if low:
            lines.append("- Below target: " + ", ".join(r.get("file") or "?" for r in low[:40])
                         + (" …" if len(low) > 40 else ""))
        try:
            with open(os.path.join(self._save_dir, "session.md"), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            self.statusBar().showMessage(f"TRPL dynamic summary not written: {exc}")

    def _write_problems_summary(self, problems: List[dict]) -> None:
        """Failed TRPL attempts of the sweep (retried, or the point kept without TRPL) in session.md."""
        if not problems or not self._save_dir:
            return
        if not self._session_saved:
            self._write_session_metadata()
        gave_up = [p for p in problems if p.get("outcome") != "retried"]
        lines = ["", "## Problems",
                 f"- {len(problems)} failed TRPL attempt(s), each followed by a retry except {len(gave_up)} where it "
                 f"gave up (point kept without TRPL, or a dynamic spot ended with the chunks it had)",
                 "- Details: scan_problems.txt; partial data of failed attempts: trpl_failed/"]
        lines += [f"- {p['time']} {p['point']} {p['what']} attempt {p['attempt']}: {p['error']} -> {p['outcome']}"
                  for p in problems[:30]]
        try:
            with open(os.path.join(self._save_dir, "session.md"), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            self.statusBar().showMessage(f"Problems summary not written: {exc}")

    def _sync_controls(self) -> None:
        cam_ok = self.camDev.is_connected
        stage_a_ok = self.stage is not None
        stage_b_ok = self.stage_b is not None
        spec_ok = self.specDev.is_connected
        busy = self._is_busy()
        hard_busy = busy and not self._aborting

        sweep_power = self.sweepPowerChk.isChecked()
        sweep_voltage = self.sweepVoltageChk.isChecked()
        need_dual = self._calib_mode in ("dual", "hwp_nd")

        angle_list_mode = sweep_power and self.angleListChk.isChecked()
        calib_ok = True
        stages_ok = True
        if sweep_power:
            if angle_list_mode:
                calib_ok = bool(self._parse_angle_list())
                stages_ok = stage_a_ok
            else:
                calib_ok = bool(self._dual_entries) if need_dual else self._calib_data is not None
                stages_ok = stage_a_ok and (stage_b_ok if need_dual else True)

        gate_ok = True
        if sweep_voltage:
            mode = self._sweep_gate_mode()
            gate_ok = True
            if mode in ("front", "dual"):
                gate_ok = gate_ok and (self.front_dev is not None)
            if mode in ("back", "dual"):
                gate_ok = gate_ok and (self.back_dev is not None)
        sweep_ok = sweep_power or sweep_voltage

        need_cam = self._measure_pl()
        need_spad = self._measure_trpl()
        spad_ok = self._spad_ok()
        detectors_ok = (cam_ok or not need_cam) and (spad_ok or not need_spad)

        self.initBtn.setEnabled(not busy)
        self.disconnectBtn.setEnabled(not hard_busy)
        self.takeCurrentBtn.setEnabled(cam_ok and not busy)
        self.stopLiveBtn.setEnabled(self._threads_running() and self._live_active)
        flipper_ok = self.flipperDev.is_connected or not self.didpChk.isChecked()
        self.sweepBtn.setEnabled(detectors_ok and flipper_ok and sweep_ok and stages_ok and calib_ok and gate_ok
                                 and not busy)
        sweep_running = self._sweep_thread is not None and self._sweep_thread.isRunning()
        self.takeHistogramBtn.setEnabled(spad_ok and not sweep_running and not self.spadDev.trpl_running)
        self.btn_take_dark.setEnabled(spad_ok and not sweep_running and not self.spadDev.trpl_running
                                      and self._sigma_worker is None and not self.spectralHist.busy)
        self.btn_fit_sigma.setEnabled(self._sigma_fit_allowed())
        self.btn_cancel_sigma.setEnabled(self._sigma_worker is not None)
        self.spectralHist.btn_fit.setEnabled(self._spectral_fit_allowed())
        self.spectralHist.btn_cancel.setEnabled(self.spectralHist.busy)
        self.btn_load_dark.setEnabled(not self._dark_pending)
        self.btn_clear_dark.setEnabled(self._dark_ref is not None and not self._dark_pending)
        self.tcspcCalibAlignBtn.setEnabled(spad_ok and not busy)
        self.tcspcAcqSpin.setEnabled(not self.spadDev.trpl_running and not busy)
        self.tcspcBinSpin.setEnabled(not self.spadDev.trpl_running and not busy)
        self._sync_dyn_controls(busy)
        self.measureCombo.setEnabled(not busy)
        self.didpChk.setEnabled(not busy)
        self.flipperSettleSpin.setEnabled(not busy)
        flip_idle = self.flipperDev.is_connected and not busy and not self.flipperDev.call_running
        self.pump2OffBtn.setEnabled(flip_idle)
        self.pump2OnBtn.setEnabled(flip_idle)
        self.pump2PosCombo.setEnabled(not busy)
        self.flipperSerialEdit.setEnabled(self.flipperDev.state == DISCONNECTED)
        self._sync_conn_buttons(busy)
        self.abortBtn.setEnabled(busy)
        self.applySpecBtn.setEnabled(spec_ok and not busy)
        self.stageHomeBtn.setEnabled(stage_a_ok and not busy)
        self.stageMoveBtn.setEnabled(stage_a_ok and not busy)
        self.stageBHomeBtn.setEnabled(stage_b_ok and not busy)
        self.stageBMoveBtn.setEnabled(stage_b_ok and not busy)
        self.stageSerialCombo.setEnabled(self.stageDevs["A"].state == DISCONNECTED)
        self.stageBSerialCombo.setEnabled(self.stageDevs["B"].state == DISCONNECTED)
        self.stageDetectBtn.setEnabled(not busy)
        self.stageBDetectBtn.setEnabled(not busy)
        self.frontResourceEdit.setEnabled(self.gateDevs["front"].state == DISCONNECTED)
        self.backResourceEdit.setEnabled(self.gateDevs["back"].state == DISCONNECTED)
        self.readoutRateCombo.setEnabled(cam_ok and not busy)
        self.preampCombo.setEnabled(cam_ok and not busy)
        self.seriesCombo.setEnabled(sweep_power and (not need_dual) and (not angle_list_mode) and calib_ok and not busy)
        for w in (self.wheelStartSpin, self.wheelEndSpin, self.wheelFullBtn):
            w.setEnabled(sweep_power and (not need_dual) and (not angle_list_mode) and calib_ok and not busy)
        power_list = self._power_list_active()
        self.powerSkipSpin.setEnabled(sweep_power and (not angle_list_mode) and (not power_list) and not busy)
        self.angleListEdit.setEnabled(sweep_power and self.angleListChk.isChecked() and not busy)
        self.powerListChk.setEnabled(sweep_power and self._calib_mode == "hwp_nd" and not busy)
        for w in (self.powerListEdit, self.powerListUnitCombo, self.powerListTolSpin):
            w.setEnabled(sweep_power and power_list and not busy)
        self.powerListDetailsBtn.setEnabled(power_list and bool(self._power_plan))
        self.ndWheelAngleCombo.setEnabled(not power_list and not busy)
        both_sweep = sweep_power and sweep_voltage
        self.sweepOrderPowerOuterRb.setEnabled(both_sweep and not busy)
        self.sweepOrderVoltageOuterRb.setEnabled(both_sweep and not busy)

        front_busy = self._front_ramp_thread is not None and self._front_ramp_thread.isRunning()
        back_busy = self._back_ramp_thread is not None and self._back_ramp_thread.isRunning()
        self.frontApplyBtn.setEnabled(self.front_dev is not None and (not hard_busy) and (not front_busy))
        self.backApplyBtn.setEnabled(self.back_dev is not None and (not hard_busy) and (not back_busy))
        either_gate = (self.front_dev is not None) or (self.back_dev is not None)
        self.homeBothBtn.setEnabled(either_gate and (not hard_busy) and (not front_busy) and (not back_busy))

    def _update_gate_sweep_controls(self) -> None:
        enabled = self.sweepVoltageChk.isChecked()
        self.sweepGateCombo.setEnabled(enabled)
        self.sweepStepsSpin.setEnabled(enabled)
        mode = self._sweep_gate_mode()
        grid = self._grid_sweep_selected()
        self.frontV0Spin.setEnabled(enabled and mode in ("front", "dual"))
        self.frontStepSpin.setEnabled(enabled and mode in ("front", "dual"))
        self.backV0Spin.setEnabled(enabled and mode in ("back", "dual"))
        self.backStepSpin.setEnabled(enabled and mode in ("back", "dual"))
        # Grid sweep is only meaningful for the dual gate mode.
        self.gridSweepChk.setEnabled(enabled and mode == "dual")
        self.backStepsSpin.setEnabled(enabled and grid)
        self.backStepsLbl.setEnabled(enabled and grid)
        self.stepsLbl.setText("Steps (A):" if grid else "Steps:")
        self._update_vend_labels()
        self._sync_controls()

    def _grid_sweep_selected(self) -> bool:
        return (
            self.sweepVoltageChk.isChecked()
            and self._sweep_gate_mode() == "dual"
            and self.gridSweepChk.isChecked()
        )

    def _current_voltage_key(self) -> Optional[float]:
        if not self._voltage_keys:
            return None
        idx = int(self.voltageSlider.value())
        idx = max(0, min(idx, len(self._voltage_keys) - 1))
        return self._voltage_keys[idx]

    def _current_power_key(self) -> Optional[float]:
        if not self._power_view_keys:
            return None
        idx = int(self.powerSlider.value())
        idx = max(0, min(idx, len(self._power_view_keys) - 1))
        return self._power_view_keys[idx]

    def _current_a_key(self) -> Optional[float]:
        if not self._a_keys:
            return None
        idx = int(self.aSelSlider.value())
        idx = max(0, min(idx, len(self._a_keys) - 1))
        return self._a_keys[idx]

    def _a_label_for_index(self, idx: int) -> str:
        if not self._a_keys:
            return "--"
        idx = max(0, min(idx, len(self._a_keys) - 1))
        if idx < len(self._a_labels):
            return self._a_labels[idx]
        return f"{self._a_keys[idx]:.6g} V"

    def _update_a_slider(self) -> None:
        n = len(self._a_keys)
        if n == 0:
            self.aSelSlider.setRange(0, 0)
            self.aSelSlider.setEnabled(False)
            self.aSelLbl.setText("Gate A: --")
            return
        self.aSelSlider.setRange(0, max(0, n - 1))
        self.aSelSlider.setEnabled(self._grid_active and n > 1)
        idx = min(int(self.aSelSlider.value()), n - 1)
        self.aSelSlider.setValue(idx)
        self.aSelLbl.setText(f"Gate A: {self._a_label_for_index(idx)}")

    def _on_a_slider(self, _value: int) -> None:
        self._update_a_slider()
        self._refresh_linecut_maps()
        self._update_lifetime_tab()

    def _voltage_label_for_index(self, idx: int) -> str:
        if not self._voltage_keys:
            return "--"
        idx = max(0, min(idx, len(self._voltage_keys) - 1))
        if idx < len(self._voltage_labels):
            return self._voltage_labels[idx]
        key = self._voltage_keys[idx]
        return f"{key:.6g} V"

    def _power_label_for_index(self, idx: int) -> str:
        if (not self._power_view_keys) or (not self.sweepPowerChk.isChecked()):
            return "--"
        idx = max(0, min(idx, len(self._power_view_keys) - 1))
        key = self._power_view_keys[idx]
        if key is None or not np.isfinite(float(key)):
            return "--"
        if self._angle_list_mode:
            return f"{key:.3f} °"
        return f"{key:.6g} {self._calib_unit}"

    @staticmethod
    def _normalize_voltage_key(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return round(v, 9)

    def _voltage_label_for_key(self, key: Optional[float]) -> str:
        if key is None:
            return "--"
        try:
            k = float(key)
        except Exception:
            return "--"
        for idx, v in enumerate(self._voltage_keys):
            try:
                if abs(float(v) - k) <= 1e-9:
                    return self._voltage_label_for_index(idx)
            except Exception:
                continue
        return f"{k:.6g} V"

    def _update_voltage_slider(self) -> None:
        n = len(self._voltage_keys)
        if n == 0:
            for slider, lbl in ((self.voltageSlider, self.voltageSelLbl),
                                (self.lifetimeVoltageSlider, self.lifetimeVoltageSelLbl)):
                slider.setRange(0, 0)
                slider.setEnabled(False)
                lbl.setText("Voltage: --")
            return
        enable = (n > 1) and (
            self._grid_active
            or (self.sweepPowerChk.isChecked() and self.sweepVoltageChk.isChecked())
        )
        idx = min(int(self.voltageSlider.value()), n - 1)
        label = "Gate B" if self._grid_active else "Voltage"
        for slider, lbl in ((self.voltageSlider, self.voltageSelLbl),
                            (self.lifetimeVoltageSlider, self.lifetimeVoltageSelLbl)):
            slider.blockSignals(True)
            slider.setRange(0, max(0, n - 1))
            slider.setEnabled(enable)
            slider.setValue(idx)
            slider.blockSignals(False)
            lbl.setText(f"{label}: {self._voltage_label_for_index(idx)}")

    def _update_power_slider(self) -> None:
        n = len(self._power_view_keys)
        if n == 0:
            for slider, lbl in ((self.powerSlider, self.powerSelLbl), (self.lifetimePowerSlider, self.lifetimePowerSelLbl)):
                slider.setRange(0, 0)
                slider.setEnabled(False)
                lbl.setText("Power: --")
            return
        enable = (n > 1) and (
            self._grid_active
            or (self.sweepPowerChk.isChecked() and self.sweepVoltageChk.isChecked())
        )
        idx = min(int(self.powerSlider.value()), n - 1)
        for slider, lbl in ((self.powerSlider, self.powerSelLbl), (self.lifetimePowerSlider, self.lifetimePowerSelLbl)):
            slider.blockSignals(True)
            slider.setRange(0, max(0, n - 1))
            slider.setEnabled(enable)
            slider.setValue(idx)
            slider.blockSignals(False)
            lbl.setText(f"Power: {self._power_label_for_index(idx)}")

    def _on_voltage_slider(self, _value: int) -> None:
        # the Spectrum and Lifetime tabs share the selection
        if self.sender() is self.lifetimeVoltageSlider:
            self.voltageSlider.blockSignals(True)
            self.voltageSlider.setValue(self.lifetimeVoltageSlider.value())
            self.voltageSlider.blockSignals(False)
        self._update_voltage_slider()
        self._refresh_linecut_power_map()
        self._update_lifetime_tab()

    def _on_power_slider(self, _value: int) -> None:
        if self.sender() is self.lifetimePowerSlider:
            self.powerSlider.blockSignals(True)
            self.powerSlider.setValue(self.lifetimePowerSlider.value())
            self.powerSlider.blockSignals(False)
        self._update_power_slider()
        self._refresh_linecut_voltage_map()
        self._update_lifetime_tab()

    @staticmethod
    def _parse_optional_float(text: str) -> Optional[float]:
        s = str(text).strip()
        if not s:
            return None
        try:
            v = float(s)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return float(v)

    def _apply_map_clim(self, data: np.ndarray, artist, cmin_edit: QtWidgets.QLineEdit, cmax_edit: QtWidgets.QLineEdit) -> None:
        try:
            auto_min = float(np.nanmin(data))
            auto_max = float(np.nanmax(data))
        except Exception:
            return
        if not np.isfinite(auto_min) or not np.isfinite(auto_max):
            return
        if auto_max <= auto_min:
            auto_max = auto_min + 1.0

        vmin_user = self._parse_optional_float(cmin_edit.text())
        vmax_user = self._parse_optional_float(cmax_edit.text())
        vmin = auto_min if vmin_user is None else vmin_user
        vmax = auto_max if vmax_user is None else vmax_user
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = auto_min, auto_max
        artist.set_levels(float(vmin), float(vmax))

    def _on_map_clim_changed(self) -> None:
        self._refresh_linecut_power_map()

    def _on_vmap_clim_changed(self) -> None:
        self._refresh_linecut_voltage_map()

    def _entry_key(self, entry) -> Tuple:
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return (
                "dual",
                power_key(entry.power_w, scale=self._calib_scale),
                float(entry.a_deg),
                float(entry.b_deg),
                int(getattr(entry, "index", 0)),
            )
        if entry.power_w is None:  # angle-list SimpleNamespace entries
            return (entry.series, entry.position_deg)
        return (entry.series, power_key(entry.power_w, scale=self._calib_scale), entry.position_deg)

    def _entry_rank(self, entry):
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            return int(getattr(entry, "index", 0))
        if not hasattr(entry, "is_base"):  # angle-list SimpleNamespace entries
            return 0
        return entry_priority(entry)

    def _format_power(self, power_w: float) -> str:
        val = power_w / float(self._calib_scale or 1e-9)
        return f"{val:.6g} {self._calib_unit}"

    def _format_power_key(self, key: float) -> str:
        return f"{key:.6g} {self._calib_unit}"

    def _safe_series_label(self, series: str) -> str:
        clean = series.strip().lower().replace(" ", "_").replace("#", "")
        clean = "".join(ch for ch in clean if ch.isalnum() or ch in ("_", "+"))
        return clean or "series"

    def _sweep_gate_mode(self) -> str:
        text = self.sweepGateCombo.currentText().strip().lower()
        if "front" in text:
            return "front"
        if "back" in text:
            return "back"
        return "dual"

    def _next_filename(self, base: str, ext: str = ".asc") -> str:
        idx = self._file_counters.get(base, 0)
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            name = f"{base}{suffix}{ext}"
            path = os.path.join(self._save_dir or ".", name)
            if not os.path.exists(path):
                self._file_counters[base] = idx + 1
                return path
            idx += 1

    def _build_metadata_base(self) -> dict:
        return {
            "exposure_ms": float(self.exposureSpin.value()),
            "accum_n": int(self.accumSpin.value()),
            "readout_rate": self._readout_rate_label,
            "preamp_gain": self._preamp_gain_label,
            "output_amp": self._output_amp_label,
            "grating": int(self.gratingSpin.value()),
            "center_wl_nm": float(self.centerSpin.value()),
            "slit_id": DEFAULT_SLIT_ID,
            "slit_um": float(self.slitSpin.value()),
            "stage_serial": self.stage.serial if self.stage else "",
            "stage_serial_a": self.stage.serial if self.stage else "",
            "stage_serial_b": self.stage_b.serial if self.stage_b else "",
            "calib_mode": self._calib_mode,
            "power_calib_file": self.calibPathEdit.text().strip(),
            **({"wheel_range_deg": "%g->%g" % self._wheel_range()} if self._calib_mode == "single" else {}),
            "crop_top": self._crop_top,
            "crop_bottom": self._crop_bottom,
            "crop_left": self._crop_left,
            "crop_right": self._crop_right,
            "linecut_row": int(self.liveView.linecut_row() or 0) if self.liveView is not None else 0,
            "linecut_width": int(self.liveView.linecut_width()) if self.liveView is not None else 1,
            "front_resource": self.frontResourceEdit.text().strip(),
            "back_resource": self.backResourceEdit.text().strip(),
            "front_vset": float(self.frontVsetSpin.value()),
            "back_vset": float(self.backVsetSpin.value()),
            "front_icomp_nA": float(self.frontIcompSpin.value()),
            "back_icomp_nA": float(self.backIcompSpin.value()),
            "front_output_on": bool(self._front_output_on),
            "back_output_on": bool(self._back_output_on),
            "sweep_power": bool(self.sweepPowerChk.isChecked()),
            "sweep_voltage": bool(self.sweepVoltageChk.isChecked()),
            "sweep_gate_mode": self._sweep_gate_mode(),
            "grid_sweep": bool(self._grid_sweep_selected()),
            "front_v0": float(self.frontV0Spin.value()),
            "front_step": float(self.frontStepSpin.value()),
            "back_v0": float(self.backV0Spin.value()),
            "back_step": float(self.backStepSpin.value()),
            "sweep_steps": int(self.sweepStepsSpin.value()),
            "sweep_steps_a": int(self.sweepStepsSpin.value()),
            "sweep_steps_b": int(self.backStepsSpin.value()),
            "power_skip_n": int(self.powerSkipSpin.value()),
            "sweep_order": "voltage_outer" if self.sweepOrderVoltageOuterRb.isChecked() else "power_outer",
            "measure_mode": self._measure_mode(),
            "spad_detector": self._det.key,
            "spad_num_pixels": int(self._det.num_pixels),
            "spad_detector_detected": bool(self._det_detected),
            "dark_reference_file": self._sweep_dark_file,
            "spad_view_mode": self._spad_view_mode,
            "spad_wavelength_calibration_file": self._sweep_wlcal_file,
            "spad_wavelength_calibration": self._wl_calib_meta(),
            "dark_reference_taken_at": None if self._dark_ref is None else self._dark_ref.taken_at,
            "tcspc_host": DEFAULT_TCSPC_HOST,
            "tcspc_port": int(DEFAULT_TCSPC_PORT),
            "tcspc_acq_s": "dynamic" if self._dyn_settings(False) is not None else float(self.tcspcAcqSpin.value()),
            "tcspc_bin_ps": int(self.tcspcBinSpin.value()),
            **self._dyn_metadata(),
            "hist_start_ns": float(self.histStartNsSpin.value()),
            "hist_end_ns": float(self.histEndNsSpin.value()),
            "hist_ref_ns": float(self.histRefNsSpin.value()),
            "hist_times_after_t0": bool(self._hist_rel),  # start/end/ref relative to time zero (t = 0)
            "hist_ref_auto": bool(self.chk_hist_ref_auto.isChecked()),
            "didp": bool(self.didpChk.isChecked()),
            "pump2_on_position": int(self._pump2_on_position()),
            "flipper_serial": self.flipperSerialEdit.text().strip(),
            "flipper_settle_ms": int(self.flipperSettleSpin.value()),
            "ramp_micro_read": bool(self.rampReadChk.isChecked()),
            "home_gates_after_scan": bool(self.homeAfterScanChk.isChecked()),
        }

    def _dyn_metadata(self) -> dict:
        st = self._dyn_settings(False)
        if st is None:
            return {"trpl_time_mode": "fixed"}
        return {"trpl_time_mode": "dynamic", "trpl_dyn_target_counts": st.target_counts,
                "trpl_dyn_first_s": st.first_s, "trpl_dyn_first_is_pl_time": self._dyn_first_from_pl(),
                "trpl_dyn_step_s": st.step_s, "trpl_dyn_max_s": st.max_s,
                "trpl_dyn_background": "dark reference" if self._dyn_usable_dark() is not None else "pre-t0",
                "trpl_dyn_log": tdyn.LOG_NAME}

    def _reset_sweep_data(self) -> None:
        self._pv_images.clear()
        self._grid_images.clear()
        self._grid_active = False
        self._a_keys = []
        self._a_labels = []
        self._b_keys = []
        self._b_labels = []
        self._voltage_keys = []
        self._voltage_labels = []
        self._voltage_pairs = {}
        self._taken_powers.clear()
        self._taken_entries.clear()
        self._power_done_counts.clear()
        self._power_target_counts.clear()
        self._power_view_keys = list(self._power_list_keys)
        self._session_saved = False
        self._file_counters.clear()
        self._spad_trpl_data.clear()
        self._refresh_power_table()
        self._update_voltage_slider()
        self._update_power_slider()
        self._update_a_slider()
        self._refresh_linecut_maps()
        self._update_lifetime_tab()

    # -----------------
    # Devices: connect / disconnect / readbacks (all hardware calls run in worker threads)
    # -----------------
    def _wire_devices(self) -> None:
        self.camDev.state_changed.connect(lambda _s: self._on_cam_state())
        self.camDev.connected.connect(self._on_cam_connected)
        self.camDev.readback.connect(self._on_cam_readback)
        self.specDev.state_changed.connect(lambda _s: self._on_spec_state())
        self.specDev.connected.connect(self._on_spec_connected)
        self.specDev.readback.connect(self._on_spec_readback)
        for which, dev in self.stageDevs.items():
            dev.state_changed.connect(lambda _s, w=which: self._on_stage_state(w))
            dev.connected.connect(lambda info, w=which: self._on_stage_connected(w, info))
            dev.readback.connect(lambda d, w=which: self._on_stage_readback(w, d))
        for which, dev in self.gateDevs.items():
            dev.state_changed.connect(lambda _s, w=which: self._on_gate_state(w))
            dev.connected.connect(lambda info, w=which: self._on_gate_connected(w, info))
            dev.readback.connect(lambda d, w=which: self._on_gate_readback(w, d))
        for dev in (self.camDev, self.specDev, *self.stageDevs.values(), *self.gateDevs.values()):
            dev.message.connect(self.statusBar().showMessage)
        self.flipperDev.state_changed.connect(lambda _s: self._on_flipper_state())
        self.flipperDev.connected.connect(self._on_flipper_connected)
        self.flipperDev.readback.connect(self._on_flipper_readback)
        self.flipperDev.message.connect(self.statusBar().showMessage)
        self._wire_spad_device()
        self._on_cam_state()
        self._on_spec_state()
        for w in ("A", "B"):
            self._on_stage_state(w)
        for w in ("front", "back"):
            self._on_gate_state(w)
        self._on_spad_state(self.spadDev.state)

    def _wire_spad_device(self) -> None:
        """Signals of ``self.spadDev`` (again after a detector switch replaced the controller)."""
        dev = self.spadDev
        dev.state_changed.connect(self._on_spad_state)
        dev.connected.connect(self._on_spad_connected)
        dev.message.connect(self.statusBar().showMessage)
        dev.count_frame.connect(self._on_spad_count_frame)
        dev.live_failed.connect(lambda msg: self._update_tcspc_status(True, f"live counts stopped: {msg}"))
        dev.command_done.connect(self._on_spad_command_done)
        dev.trpl_progress.connect(self._on_histogram_update)
        dev.trpl_done.connect(self._on_spad_trpl_done)

    def _retain(self, th: QtCore.QThread) -> QtCore.QThread:
        """Keep a parentless worker thread referenced until Qt reports it ``finished``.

        The workers (rot MoveThread / HomeThread, GateRampThread, the sweep) emit
        ``done`` as the last statement of ``run()``, and the done slot drops the
        widget's reference (``self._stage_threads[which] = None``).  When that slot
        runs in the instant between ``run()`` returning and Qt marking the thread
        finished, the last reference goes and Qt aborts the program: "QThread:
        Destroyed while thread is still running" (seen while turning an ND wheel;
        reproduced within 3000 short moves, none with this).
        """
        self._bg_threads.add(th)
        th.finished.connect(lambda th=th: self._bg_threads.discard(th))
        return th

    def _run_bg(self, fn: Callable[[], object], on_done: Callable[[object, str], None]) -> None:
        """One blocking call that belongs to no device (e.g. listing Kinesis serials)."""
        th = CallThread(fn, self)
        self._bg_threads.add(th)
        th.done.connect(on_done)
        th.finished.connect(lambda th=th: self._bg_threads.discard(th))
        th.start()

    def _toggle_device(self, which: str) -> None:
        if which == "camera":
            self._disconnect_camera() if self.camDev.is_connected else self._connect_camera()
        elif which == "spec":
            self.specDev.disconnect_device() if self.specDev.is_connected else self._connect_spectrograph()
        elif which in ("A", "B"):
            dev = self.stageDevs[which]
            self._disconnect_stage(which) if dev.is_connected else self._connect_stage(which)
        elif which in ("front", "back"):
            dev = self.gateDevs[which]
            self._disconnect_gate(which) if dev.is_connected else self._connect_gate(which)
        elif which == "spad":
            if self._spad_ok():
                self.spadDev.disconnect_device()
            else:
                self._connect_spad()
        elif which == "flipper":
            if self.flipperDev.is_connected:
                self.flipperDev.disconnect_device()
            else:
                self._connect_flipper()
        self._sync_controls()

    def on_connect_all(self):
        if self._busy:
            return
        if self.camDev.state == DISCONNECTED:
            self._pending_after_cam.append(self._connect_spectrograph)
            self._connect_camera()
        elif self.camDev.is_connected and self.specDev.state == DISCONNECTED:
            self._connect_spectrograph()
        if self.stageDevs["A"].state == DISCONNECTED and self.stageSerialCombo.currentText().strip():
            # stages one after the other (both on the Kinesis FTDI driver)
            self._pending_after_stage_a.append(lambda: self._connect_stage("B"))
            self._connect_stage("A")
        elif self.stageDevs["B"].state == DISCONNECTED:
            self._connect_stage("B")
        for which in ("front", "back"):
            if self.gateDevs[which].state == DISCONNECTED:
                self._connect_gate(which)
        if self.spadDev.state == spad_det.DISCONNECTED:
            self._connect_spad()
        if self.flipperDev.state == DISCONNECTED and self.flipperSerialEdit.text().strip():
            self._connect_flipper()
        self.statusBar().showMessage("Connecting all devices...")
        self._sync_controls()

    def on_disconnect_all(self):
        self.on_abort()
        # gates ramp to 0 V first: start once the aborted sweep / ramps no longer drive them
        self._after_gate_users_stop(lambda: [self._disconnect_gate(which) for which in ("front", "back")])
        for which in ("A", "B"):
            self._disconnect_stage(which)
        self._live_active = False
        self._disconnect_camera()
        self._detect_cancelled = self._detect_worker is not None  # do not connect when detection returns
        if self._spad_ok():
            self.spadDev.disconnect_device()
        self.flipperDev.disconnect_device()
        self.statusBar().showMessage("Disconnecting...")
        self._sync_controls()

    # ---- Andor camera
    def _connect_camera(self) -> None:
        self.camDev.connect_device()

    def _disconnect_camera(self) -> None:
        if self.camDev.state != CONNECTED:
            return
        if self.specDev.state != DISCONNECTED:
            # the spectrograph uses the camera: close it first
            self._pending_after_spec_off.append(self._disconnect_camera)
            self.specDev.disconnect_device()
            return
        self.camDev.disconnect_device()

    def _on_cam_state(self) -> None:
        state = self.camDev.state
        self._set_dot(self.camDot, state == CONNECTED)
        if state == DISCONNECTED:
            self.camStatusLbl.setText("Disconnected")
            self._pending_after_cam.clear()
        elif state != CONNECTED:
            self.camStatusLbl.setText(state.capitalize() + "...")
        self._sync_controls()

    def _on_cam_connected(self, info) -> None:
        amp = info.get("info") if isinstance(info, dict) else None
        if amp:
            # the PL GUI's on_initialize: labels from the camera, then its choices, then apply
            self._readout_rate_label = amp.get("rate_label") or self._readout_rate_label
            self._preamp_gain_label = amp.get("gain_label") or self._preamp_gain_label
            self._output_amp_label = amp.get("amp_label") or self._output_amp_label
            rate_label = amp.get("rate_label")
            if rate_label:
                self.readoutRateCombo.blockSignals(True)
                self.readoutRateCombo.setCurrentText(str(rate_label))
                self.readoutRateCombo.blockSignals(False)
        self._refresh_amp_choices(info.get("choices") if isinstance(info, dict) else None)
        self._apply_amp_settings()
        self.camStatusLbl.setText("Connected")
        pending, self._pending_after_cam = self._pending_after_cam, []
        for fn in pending:
            fn()
        self._sync_controls()

    def _on_cam_readback(self, d: dict) -> None:
        if "error" in d:
            self.camStatusLbl.setText(f"Read error: {d['error'][:40]}")
            return
        temp = d.get("temp")
        temp_s = f"{temp:.1f} C" if temp is not None else "--"
        exp_ms = float(self.exposureSpin.value())
        acc = int(self.accumSpin.value())
        ro = self._readout_rate_label or "--"
        gain = self._preamp_gain_label or self.preampCombo.currentText().strip() or "--"
        self.camStatusLbl.setText(f"T={temp_s} | Exp {exp_ms:g} ms x{acc} | RO {ro} | Gain {gain}")

    # ---- spectrograph
    def _connect_spectrograph(self) -> None:
        if not self.camDev.is_connected:
            self.statusBar().showMessage("Connect the Andor camera before the spectrograph.")
            return
        self.specDev.connect_device(**self._spec_params())

    def _on_spec_state(self) -> None:
        state = self.specDev.state
        self._set_dot(self.specDot, state == CONNECTED)
        if state == DISCONNECTED:
            self.specStatusLbl.setText("Disconnected")
            if self.liveView is not None:
                self.liveView.set_wavelength_axis_data(None)
            self._wl_read_at = None
            self._wl_fetch_pending = False
            pending, self._pending_after_spec_off = self._pending_after_spec_off, []
            for fn in pending:
                fn()
        elif state != CONNECTED:
            self.specStatusLbl.setText(state.capitalize() + "...")
        self._sync_controls()

    def _on_spec_connected(self, info) -> None:
        self._set_wavelength_axis(info)
        self._refresh_linecut_maps()
        self.specStatusLbl.setText("Connected")
        self._sync_controls()

    def _on_spec_readback(self, d: dict) -> None:
        if "error" in d:
            self.specStatusLbl.setText(f"Read error: {d['error'][:40]}")
            return
        gr, cw, slit = d.get("grating"), d.get("center_nm"), d.get("slit_um")
        gr_s = str(gr) if gr is not None else "--"
        cw_s = f"{cw:.3f}" if cw is not None else "--"
        slit_s = f"{slit:.1f}" if slit is not None else "--"
        self.specStatusLbl.setText(f"Gr {gr_s} | Center {cw_s} nm | Slit {slit_s} um")
        self._check_wavelength_axis(d)

    # ---- rotation stages
    def _make_stage(self, serial: str) -> RotationStage:
        return RotationStage(serial, scale=DEFAULT_STAGE_SCALE)

    def _stage_widgets(self, which: str):
        if which == "A":
            return self.stageDot, self.stageStatusLbl, self.stageSerialCombo, "Stage A"
        return self.stageBDot, self.stageBStatusLbl, self.stageBSerialCombo, "Stage B"

    def _connect_stage(self, which: str) -> None:
        dev = self.stageDevs[which]
        if dev.state != DISCONNECTED:
            return
        _, lbl, combo, tag = self._stage_widgets(which)
        serial = combo.currentText().strip()
        if not serial:
            return
        other = self.stageDevs["B" if which == "A" else "A"]
        other_serial = (self.stageSerialCombo if which == "B" else self.stageBSerialCombo).currentText().strip()
        if other.state != DISCONNECTED and serial == other_serial:
            self.statusBar().showMessage(f"{tag}: serial {serial} is already used by the other stage")
            return
        dev.connect_device(serial=serial, home=self.homeOnConnectChk.isChecked())

    def _disconnect_stage(self, which: str) -> None:
        if self._stage_busy_by[which]:
            ctrl = self._stage_controllers[which]
            if ctrl is not None:
                ctrl.abort()
        self.stageDevs[which].disconnect_device()

    def _on_stage_state(self, which: str) -> None:
        dot, lbl, _, _ = self._stage_widgets(which)
        dev = self.stageDevs[which]
        state = dev.state
        self._set_dot(dot, state == CONNECTED)
        if state == DISCONNECTED:
            lbl.setText("Disconnected")
            if which == "A":
                self.stage = None
                pending, self._pending_after_stage_a = self._pending_after_stage_a, []
                for fn in pending:
                    fn()
            else:
                self.stage_b = None
        elif state != CONNECTED:
            lbl.setText(state.capitalize() + "...")
            if which == "A":
                self.stage = None
            else:
                self.stage_b = None
        self._sync_controls()

    def _on_stage_connected(self, which: str, info) -> None:
        stage = self.stageDevs[which].stage
        if which == "A":
            self.stage = stage
        else:
            self.stage_b = stage
        _, lbl, _, tag = self._stage_widgets(which)
        lbl.setText(f"SN {stage.serial if stage else '--'} | connected")
        vel = info.get("velocity") if isinstance(info, dict) else None
        self._stage_velocity[which] = (float(vel[2]), float(vel[1])) if vel and vel[1] > 0 and vel[2] > 0 else None
        if isinstance(info, dict) and info.get("home"):
            self._stage_home(which)
        if which == "A":
            pending, self._pending_after_stage_a = self._pending_after_stage_a, []
            for fn in pending:
                fn()
        self._sync_controls()

    def _on_stage_readback(self, which: str, d: dict) -> None:
        if not d or "error" in d:
            return
        _, lbl, _, _ = self._stage_widgets(which)
        pos, homed = d.get("pos"), d.get("homed")
        pos_s = f"{pos:.3f} deg" if pos is not None else "--"
        home_s = "yes" if homed else ("no" if homed is False else "?")
        status = "Moving..." if (self._stage_busy_by[which] or d.get("moving")) else "Ready"
        lbl.setText(f"SN {d.get('serial', '--')} | Homed {home_s} | {status} | Angle {pos_s}")

    # ---- gates
    def _gate_widgets(self, which: str):
        if which == "front":
            return (self.frontGateDot, self.frontGateStatusLbl, self.frontVsetSpin, self.frontIcompSpin,
                    self.frontResourceEdit)
        return (self.backGateDot, self.backGateStatusLbl, self.backVsetSpin, self.backIcompSpin,
                self.backResourceEdit)

    def _render_gate_status(self, which: str) -> None:
        """Gate status line: readings, output and the last ramp / poll message (like the Andor line)."""
        lbl = self._gate_widgets(which)[1]
        state = self._gate_state[which]
        if (self.front_dev if which == "front" else self.back_dev) is None:
            lbl.setText(state)
            return
        vi = self._gate_vi[which]
        out = self._front_output_on if which == "front" else self._back_output_on
        parts = [f"V={vi[0]:.6g} V" if vi else "V=--", f"I={vi[1] * 1e9:.6g} nA" if vi else "I=--",
                 f"Output {'ON' if out else 'OFF'}"]
        if state and state != "Connected":
            parts.append(state)
        lbl.setText(" | ".join(parts))

    def _set_gate_state(self, which: str, state: str) -> None:
        self._gate_state[which] = str(state)
        self._render_gate_status(which)

    def _set_gate_vi(self, which: str, v: float, iA: float) -> None:
        self._gate_vi[which] = (float(v), float(iA))
        self._render_gate_status(which)

    def _ensure_rm(self):
        with self._rm_lock:
            if self.rm is None:
                self.rm = make_resource_manager(VISA_DLL)
            return self.rm

    def _open_gate_io(self, which: str, resource: str, icomp_a: Optional[float]) -> Dict:
        """Worker thread: the PL GUI's ``_connect_gate_device`` device steps."""
        resource = (resource or "").strip()
        if not resource:
            raise RuntimeError("no VISA resource")
        rm = self._ensure_rm()
        dev = KeithleySMU(rm, resource, timeout_ms=20000, query_delay_s=0.0, verbose=False)
        idn = dev.open()
        if icomp_a is not None:
            try:
                dev.set_compliance(float(icomp_a))
            except Exception:
                pass
        v_now = None
        try:
            # Sync the Vset spinbox to the instrument's actual source voltage
            v_now = float(dev.query(":SOUR:VOLT?").strip())
        except Exception:
            v_now = None
        output_on = False
        try:
            dev.set_output(True)
            output_on = True
        except Exception:
            output_on = False
        return {"dev": dev, "idn": idn, "v_now": v_now, "output_on": output_on}

    def _connect_gate(self, which: str) -> None:
        dev = self.gateDevs[which]
        if dev.state != DISCONNECTED:
            return
        widgets = self._gate_widgets(which)
        resource = widgets[4].text().strip()
        if not resource:
            return
        icomp_a = float(widgets[3].value()) * 1e-9
        dev.connect_device(resource=resource, icomp_a=icomp_a)

    def _disconnect_gate(self, which: str) -> None:
        """Ramp to 0 V, output off and close (in the gate's worker thread)."""
        self.gateDevs[which].leave_output_on = False
        self.gateDevs[which].disconnect_device()

    def _after_gate_users_stop(self, fn: Callable[[], None], timeout_s: float = 30.0) -> None:
        """Run ``fn`` once no sweep / gate ramp thread is still driving a gate (after an abort)."""
        t0 = time.monotonic()

        def check():
            if not (self._threads_running() or self._gate_threads_running()) or time.monotonic() - t0 > timeout_s:
                fn()
            else:
                QtCore.QTimer.singleShot(50, check)

        check()

    def gates_at_voltage(self) -> Dict[str, float]:
        """Connected gates not at 0 V: {which: last V reading (NaN if none yet)}."""
        out = {}
        for which, dev in self.gateDevs.items():
            if dev.state == DISCONNECTED:
                continue
            vi = self._gate_vi.get(which)
            v = vi[0] if vi else float("nan")
            if not (abs(v) <= 5e-3):
                out[which] = v
        return out

    def ramp_down_gates(self, then: Callable[[Dict[str, str]], None]) -> None:
        """Window close: abort, ramp every connected gate to 0 V and switch it off, then ``then(errors)``."""
        self.on_abort()
        self.statusBar().showMessage("Ramping the gates to 0 V before closing (Abort leaves them on)...")

        def start():
            for which in ("front", "back"):
                self._disconnect_gate(which)

            def wait():
                if any(d.state != DISCONNECTED for d in self.gateDevs.values()):
                    QtCore.QTimer.singleShot(100, wait)
                    return
                then({w: d.last_close_error for w, d in self.gateDevs.items() if d.last_close_error})

            wait()

        self._after_gate_users_stop(start)

    def leave_gate_outputs_on(self) -> None:
        """Window close without ramping: the gates keep their voltage (outputs not switched off)."""
        for dev in self.gateDevs.values():
            dev.leave_output_on = True

    def _gate_ui_disconnected(self, which: str, reason: str) -> None:
        dot = self._gate_widgets(which)[0]
        if which == "front":
            self.front_dev = None
            self._front_output_on = False
        else:
            self.back_dev = None
            self._back_output_on = False
        self._gate_vi[which] = None
        self._set_gate_state(which, reason)
        self._set_dot(dot, False)

    def _on_gate_state(self, which: str) -> None:
        state = self.gateDevs[which].state
        if state == DISCONNECTED:
            self._gate_ui_disconnected(which, "Disconnected")
        elif state != CONNECTED:
            self._set_gate_state(which, state.capitalize() + "...")
        self._sync_controls()

    def _gate_ui_connected(self, which: str, info: dict) -> None:
        dot, _, vset_spin = self._gate_widgets(which)[:3]
        dev = info.get("dev")
        output_on = bool(info.get("output_on"))
        if which == "front":
            self.front_dev = dev
            self._front_output_on = output_on
        else:
            self.back_dev = dev
            self._back_output_on = output_on
        v_now = info.get("v_now")
        if v_now is not None:
            vset_spin.blockSignals(True)
            vset_spin.setValue(float(v_now))
            vset_spin.blockSignals(False)
        self._set_gate_state(which, (info.get("idn") or "Connected")[:60])
        self._set_dot(dot, True)

    def _on_gate_connected(self, which: str, info) -> None:
        if isinstance(info, dict):
            self._gate_ui_connected(which, info)
        self._sync_controls()

    def _on_gate_readback(self, which: str, d: dict) -> None:
        if not d:
            return
        if "v" in d:
            self._apply_gate_readout(which, d["v"], d["i"])
        elif "poll_error" in d:
            self._set_gate_state(which, "Poll error")
        elif d.get("reconnected"):
            self._gate_ui_connected(which, d.get("info") or {})
            self.statusBar().showMessage(f"{which.capitalize()} gate reconnected")
            self._sync_controls()
        elif d.get("lost"):
            reason = "Reconnect failed" if "reconnect_failed" in d else "Device not responding"
            self._gate_ui_disconnected(which, reason)
            self._sync_controls()

    def _apply_gate_readout(self, which: str, v: Optional[float], iA: Optional[float]) -> None:
        if v is None or iA is None:
            return
        self._gate_last_v[which] = float(v)
        self._gate_state[which] = "Connected"
        self._set_gate_vi(which, v, iA)
        if which == "front":
            self.frontPlot.add_point(v, iA)
            self._front_poll_failures = 0
            self._set_dot(self.frontGateDot, True)
        else:
            self.backPlot.add_point(v, iA)
            self._back_poll_failures = 0
            self._set_dot(self.backGateDot, True)

    # -----------------
    # Calibration
    # -----------------
    def on_browse_calib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select calibration file",
            DATA_DIR,
            "Calibration files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
        )
        if path:
            self.calibPathEdit.setText(path)

    def on_load_calib(self):
        path = self.calibPathEdit.text().strip()
        try:
            if is_hwp_nd_calibration(path):
                data = load_hwp_nd_calibration(path)
                self._load_hwp_nd_calibration(data)
            elif is_dual_wheel_calibration(path):
                data = load_dual_wheel_calibration(path)
                self._load_dual_calibration(data)
            else:
                data = load_power_calibration(path)
                self._load_single_calibration(data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        self.statusBar().showMessage(f"Loaded calibration: {os.path.basename(path)}")
        self._sync_controls()

    def _load_single_calibration(self, data):
        self._calib_mode = "single"
        self._calib_data = data
        self._dual_calib_data = None
        self._hwp_nd_calib_data = None
        self._dual_entries = []
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpDashLbl, self.hwpMaxSpin):
            w.setVisible(False)
        self._calib_unit = data.unit
        self._calib_scale = data.scale
        self._series_entries.clear()
        for entry in data.entries:
            self._series_entries.setdefault(entry.series, []).append(entry)
        for series in self._series_entries:
            self._series_entries[series].sort(key=lambda e: (e.power_w, e.position_deg))

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItems(data.series_names)
        if "base" in data.series_names:
            self.seriesCombo.setCurrentText("base")
        self.seriesCombo.blockSignals(False)
        self.calibModeLbl.setText("Mode: single wheel + ND filters")
        self._drop_power_list()
        self._set_full_wheel_range()
        self._set_wheel_range_visible(True)

        self._build_power_display()
        self._total_entries = len(data.entries)
        self._reset_sweep_data()
        self._update_series_info()

    # Single-wheel angle range ------------------------------------------------
    def _set_wheel_range_visible(self, on: bool) -> None:
        self.wheelRangeLbl.setVisible(on)
        self.wheelRangeBox.setVisible(on)

    def _set_full_wheel_range(self) -> None:
        """Start / end = the lowest / highest wheel angle in the calibration file."""
        data = self._calib_data
        if data is None or not data.entries:
            return
        pos = [e.position_deg for e in data.entries]
        lo, hi = min(pos), max(pos)
        for spin, val in ((self.wheelStartSpin, lo), (self.wheelEndSpin, hi)):
            spin.blockSignals(True)
            spin.setRange(min(0.0, lo), max(360.0, hi))
            spin.setValue(val)
            spin.blockSignals(False)
        self._on_wheel_range_changed()

    def _wheel_range(self) -> Tuple[float, float]:
        return self.wheelStartSpin.value(), self.wheelEndSpin.value()

    def _single_series_entries(self, series: str) -> List[PowerCalibEntry]:
        """The series' calibration points inside the wheel range (inclusive), by power.

        The range runs from start towards increasing angle and wraps through 360° = 0°:
        100 -> 20 is 100..360 plus 0..20, while 20 -> 100 is 20..100.
        """
        start, end = self._wheel_range()
        span = (end - start) % 360.0
        if span == 0.0 and end != start:  # e.g. 0 -> 360: the whole wheel
            span = 360.0
        return [e for e in self._series_entries.get(series, [])
                if (e.position_deg - start + 1e-6) % 360.0 <= span + 2e-6]

    def _on_wheel_range_changed(self, *_args) -> None:
        if self._calib_mode == "single":
            self._update_series_info()

    def _load_dual_calibration(self, data: DualWheelPowerData):
        self._calib_mode = "dual"
        self._dual_calib_data = data
        self._hwp_nd_calib_data = None
        self._calib_data = None
        self._series_entries.clear()
        self._dual_entries = list(data.entries)
        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpDashLbl, self.hwpMaxSpin):
            w.setVisible(False)
        self._calib_unit = data.unit
        self._calib_scale = data.scale

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItem("dual_wheel")
        self.seriesCombo.setCurrentIndex(0)
        self.seriesCombo.blockSignals(False)
        self.calibModeLbl.setText("Mode: dual wheel (file order sweep)")
        self._drop_power_list()
        self._set_wheel_range_visible(False)

        self._build_power_display()
        self._total_entries = len(data.entries)
        self._reset_sweep_data()
        self._update_series_info()

    def _load_hwp_nd_calibration(self, data: HWPNDCalibData) -> None:
        self._calib_mode = "hwp_nd"
        self._hwp_nd_calib_data = data
        self._dual_calib_data = None
        self._calib_data = None
        self._series_entries.clear()

        # Populate ND angle selector
        self.ndWheelAngleCombo.blockSignals(True)
        self.ndWheelAngleCombo.clear()
        for a in data.nd_angles:
            self.ndWheelAngleCombo.addItem(f"{a:.2f}°")
        self.ndWheelAngleCombo.blockSignals(False)
        # Set HWP angle limits to full range of measured angles
        if data.hwp_angles:
            lo, hi = min(data.hwp_angles), max(data.hwp_angles)
            self.hwpMinSpin.blockSignals(True)
            self.hwpMaxSpin.blockSignals(True)
            self.hwpMinSpin.setValue(lo)
            self.hwpMaxSpin.setValue(hi)
            self.hwpMinSpin.blockSignals(False)
            self.hwpMaxSpin.blockSignals(False)

        for w in (self.ndWheelAngleLbl, self.ndWheelAngleCombo,
                  self.hwpAngleLbl, self.hwpMinSpin, self.hwpDashLbl, self.hwpMaxSpin):
            w.setVisible(True)
        self._set_wheel_range_visible(False)

        self.seriesCombo.blockSignals(True)
        self.seriesCombo.clear()
        self.seriesCombo.addItem("hwp_nd")
        self.seriesCombo.setCurrentIndex(0)
        self.seriesCombo.blockSignals(False)

        self.calibModeLbl.setText(
            f"Mode: HWP+ND  |  {len(data.hwp_angles)} HWP steps"
            + (f", {len(data.nd_angles)} ND angles" if data.nd_angles else "")
        )

        # Entries: the power list if it is on, else the first ND angle
        self._on_hwp_nd_angle_changed()

    def _on_hwp_nd_angle_changed(self) -> None:
        if self._calib_mode != "hwp_nd":
            return
        if self._power_list_active():
            self._apply_power_list()
        else:
            self._apply_hwp_nd_angle()
            self._update_power_list_preview()

    def _apply_hwp_nd_angle(self) -> None:
        data = self._hwp_nd_calib_data
        if data is None:
            return
        idx = self.ndWheelAngleCombo.currentIndex()
        if idx < 0:
            return
        nd_angle = data.nd_angles[idx] if data.nd_angles else 0.0
        entries, unit, scale = compute_hwp_nd_entries(data, nd_angle)

        # Apply HWP angle range filter
        lo = self.hwpMinSpin.value()
        hi = self.hwpMaxSpin.value()
        entries = [e for e in entries if lo - 1e-9 <= e.a_deg <= hi + 1e-9]

        self._dual_entries = entries
        self._calib_unit = unit
        self._calib_scale = scale

        self._build_power_display()
        self._total_entries = len(entries)
        self._reset_sweep_data()
        self._update_series_info()
        self._sync_controls()

    def _build_power_display(self):
        self._power_display.clear()
        if self._calib_mode in ("dual", "hwp_nd"):
            if not self._dual_entries:
                return
            for entry in self._dual_entries:
                key = power_key(entry.power_w, scale=self._calib_scale)
                rank = int(getattr(entry, "index", 0))
                cur = self._power_display.get(key)
                if cur is None or rank < cur["rank"]:
                    self._power_display[key] = {"entry": entry, "rank": rank}
        else:
            if self._calib_data is None:
                return
            for entry in self._calib_data.entries:
                key = power_key(entry.power_w, scale=self._calib_scale)
                rank = entry_priority(entry)
                cur = self._power_display.get(key)
                if cur is None or rank < cur["rank"]:
                    self._power_display[key] = {"entry": entry, "rank": rank}

        self._power_list_keys = sorted(self._power_display.keys())
        self._power_view_keys = list(self._power_list_keys)
        self._refresh_power_table()
        self._update_power_slider()

    def _update_series_info(self):
        if self._calib_mode in ("dual", "hwp_nd"):
            n = len(self._dual_entries)
            label = "hwp_nd" if self._calib_mode == "hwp_nd" else "dual_wheel"
            self.seriesInfoLbl.setText(f"{label} ({n} entries)")
            return
        series = self.seriesCombo.currentText().strip()
        n = len(self._series_entries.get(series, []))
        n_in = len(self._single_series_entries(series))
        self.seriesInfoLbl.setText(f"{series} ({n} powers)" if n_in == n
                                   else f"{series} ({n_in} of {n} powers in wheel range)")

    def _refresh_power_table(self):
        unit = self._calib_unit
        self.powerTable.setRowCount(len(self._power_list_keys))
        self.powerTable.setHorizontalHeaderLabels([f"Power ({unit})", "Taken"])
        for row, key in enumerate(self._power_list_keys):
            if self.sweepPowerChk.isChecked():
                power_text = f"{key:.6g}"
            else:
                power_text = "--"
            power_item = QtWidgets.QTableWidgetItem(power_text)
            taken_item = QtWidgets.QTableWidgetItem("")
            taken_item.setFlags(QtCore.Qt.ItemIsEnabled)
            taken_item.setCheckState(QtCore.Qt.Checked if key in self._taken_powers else QtCore.Qt.Unchecked)
            self.powerTable.setItem(row, 0, power_item)
            self.powerTable.setItem(row, 1, taken_item)
        self.powerTable.resizeColumnsToContents()

    def _mark_power_taken(self, key: float) -> None:
        self._taken_powers.add(key)
        for row, k in enumerate(self._power_list_keys):
            if k == key:
                item = self.powerTable.item(row, 1)
                if item is not None:
                    item.setCheckState(QtCore.Qt.Checked)
                break

    def on_apply_spec(self):
        self._apply_spec_settings()  # maps refresh when the worker reports back

    def on_preamp_change(self):
        if not self.camDev.is_connected:
            self._preamp_gain_label = self.preampCombo.currentText().strip()
            return
        self._apply_amp_settings()

    def on_apply_crop(self):
        self._crop_top = int(self.cropTopSpin.value())
        self._crop_bottom = int(self.cropBottomSpin.value())
        self._crop_left = int(self.cropLeftSpin.value())
        self._crop_right = int(self.cropRightSpin.value())
        self._apply_crop_to_liveview()
        self._refresh_linecut_map()
        self.statusBar().showMessage("Crop updated")

    def on_stage_detect(self):
        self.statusBar().showMessage("Searching for Kinesis stages...")
        self._run_bg(list_kinesis_serials, self._on_stage_detect_done)

    def _on_stage_detect_done(self, serials, err: str):
        if err:
            self.statusBar().showMessage(f"Detect failed: {err}")
            return
        if not serials:
            self.statusBar().showMessage("No stages detected")
            return
        current_a = self.stageSerialCombo.currentText().strip()
        current_b = self.stageBSerialCombo.currentText().strip()
        self.stageSerialCombo.blockSignals(True)
        self.stageBSerialCombo.blockSignals(True)
        self.stageSerialCombo.clear()
        self.stageBSerialCombo.clear()
        self.stageSerialCombo.addItems(serials)
        self.stageBSerialCombo.addItems(serials)
        if current_a:
            self.stageSerialCombo.setCurrentText(current_a)
        if current_b:
            self.stageBSerialCombo.setCurrentText(current_b)
        self.stageSerialCombo.blockSignals(False)
        self.stageBSerialCombo.blockSignals(False)
        self.statusBar().showMessage(f"Detected {len(serials)} stage(s)")

    def _on_gate_ramp_progress(self, which: str, v: float, iA: float) -> None:
        self._set_gate_vi(which, v, iA)
        (self.frontPlot if which == "front" else self.backPlot).add_point(v, iA)

    def _on_gate_ramp_done(self, which: str, status: str) -> None:
        if which == "front":
            self._front_ramp_thread = None
            if status == "ok":
                self._set_gate_state("front", "Applied (ramped)")
            elif status == "aborted":
                self._set_gate_state("front", "Ramp aborted")
            else:
                self._set_gate_state("front", status)
        else:
            self._back_ramp_thread = None
            if status == "ok":
                self._set_gate_state("back", "Applied (ramped)")
            elif status == "aborted":
                self._set_gate_state("back", "Ramp aborted")
            else:
                self._set_gate_state("back", status)
        self._update_device_pauses()
        self._sync_controls()

    def _start_gate_ramp(self, which: str, target_v: Optional[float] = None) -> None:
        dev = self.front_dev if which == "front" else self.back_dev
        if dev is None:
            QtWidgets.QMessageBox.warning(self, "Gate", f"{which} gate not connected.")
            return

        vset = target_v if target_v is not None else float(self.frontVsetSpin.value() if which == "front" else self.backVsetSpin.value())
        icomp_nA = float(self.frontIcompSpin.value() if which == "front" else self.backIcompSpin.value())
        icomp_a = icomp_nA * 1e-9

        controller = SweepController()
        thread = self._retain(GateRampThread(dev, vset, icomp_a, controller, RAMP_STEP_V, RAMP_DWELL_S))
        if which == "front":
            self._front_ramp_controller = controller
            self._front_ramp_thread = thread
            thread.progress.connect(lambda v, i: self._on_gate_ramp_progress("front", v, i))
            thread.status.connect(lambda msg: self._set_gate_state("front", msg[:60]))
            thread.done.connect(lambda status: self._on_gate_ramp_done("front", status))
            self._front_output_on = True
        else:
            self._back_ramp_controller = controller
            self._back_ramp_thread = thread
            thread.progress.connect(lambda v, i: self._on_gate_ramp_progress("back", v, i))
            thread.status.connect(lambda msg: self._set_gate_state("back", msg[:60]))
            thread.done.connect(lambda status: self._on_gate_ramp_done("back", status))
            self._back_output_on = True

        thread.start()
        self._update_device_pauses()  # gate polls pause while a gate ramps (as v1)
        self._sync_controls()

    def _update_vend_labels(self) -> None:
        n = int(self.sweepStepsSpin.value())
        n_back = int(self.backStepsSpin.value()) if self._grid_sweep_selected() else n
        front_end = float(self.frontV0Spin.value()) + (n - 1) * float(self.frontStepSpin.value())
        back_end = float(self.backV0Spin.value()) + (n_back - 1) * float(self.backStepSpin.value())
        self.frontVendLbl.setText(f"End: {front_end:.6g} V")
        self.backVendLbl.setText(f"End: {back_end:.6g} V")

    def on_home_both_gates(self):
        if self.front_dev is not None:
            self._start_gate_ramp("front", target_v=0.0)
        if self.back_dev is not None:
            self._start_gate_ramp("back", target_v=0.0)

    def on_front_apply(self):
        self._start_gate_ramp("front")

    def on_back_apply(self):
        self._start_gate_ramp("back")

    # -----------------
    # Stage control
    # -----------------
    def _stage_dev_obj(self, which: str):
        return self.stage if which == "A" else self.stage_b

    def _start_stage_thread(self, which: str, thread: QtCore.QThread) -> None:
        tag = "Stage A" if which == "A" else "Stage B"
        self._stage_busy_by[which] = True
        self._stage_threads[which] = self._retain(thread)
        thread.status.connect(self.statusBar().showMessage)
        thread.done.connect(lambda s, w=which, t=tag: self._on_stage_done(s, t, w))
        thread.start()
        self._sync_controls()

    def _stage_home(self, which: str):
        stage = self._stage_dev_obj(which)
        if stage is None or self._stage_busy_by[which]:
            return
        ctrl = MotionController()
        self._stage_controllers[which] = ctrl
        self._start_stage_thread(which, HomeThread(stage, ctrl))

    def _stage_move(self, which: str):
        stage = self._stage_dev_obj(which)
        if stage is None or self._stage_busy_by[which]:
            return
        ctrl = MotionController()
        self._stage_controllers[which] = ctrl
        target = float((self.stageTargetSpin if which == "A" else self.stageBTargetSpin).value())
        self._start_stage_thread(
            which, MoveThread(stage, target, DEFAULT_RAMP_STEP_DEG, DEFAULT_STAGE_ACCEL, ctrl)
        )

    # PL GUI entry points (one stage each)
    def on_stage_home(self):
        self._stage_home("A")

    def on_stage_move(self):
        self._stage_move("A")

    def on_stage_home_b(self):
        self._stage_home("B")

    def on_stage_move_b(self):
        self._stage_move("B")

    def _on_stage_done(self, status: str, tag: str = "Stage", which: str = "A"):
        self._stage_busy_by[which] = False
        self._stage_threads[which] = None
        if status == "ok":
            self.statusBar().showMessage(f"{tag} ready")
        elif status == "aborted":
            self.statusBar().showMessage(f"{tag} motion aborted")
        else:
            self.statusBar().showMessage(str(status))
        self._sync_controls()

    # -----------------
    # Acquisition
    # -----------------
    def on_take_current(self):
        if self._busy:
            return
        if not self.camDev.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera", "Camera not connected.")
            return
        # Polls let go of the camera, the mode is set in the camera's worker thread,
        # then the acquisition thread starts.
        self._set_busy(True)
        live = int(self.accumSpin.value()) <= 1
        prepare = self._prepare_live_mode if live else self._prepare_single_mode

        def start():
            if not self.camDev.call(prepare, lambda _r, _e: self._start_take_current()):
                self._set_busy(False)

        self._start_when_quiet([self.camDev, self.specDev], start)

    def _start_take_current(self):
        accum_n = int(self.accumSpin.value())
        if accum_n <= 1:
            self._live_thread = self._retain(LiveAcqThread(self.cam))
            self._live_thread.frame_ready.connect(self.on_live_frame)
            self._live_thread.status.connect(self.statusBar().showMessage)
            self._live_thread.finished.connect(self._on_live_done)
            self._live_thread.start()
            self._live_active = True
            self._set_busy(True)
            self.statusBar().showMessage("Live started")
        else:
            self._accum_thread = self._retain(SingleAcqThread(self.cam, accum_n))
            self._accum_thread.frame_ready.connect(self.on_accum_frame)
            self._accum_thread.status.connect(self.statusBar().showMessage)
            self._accum_thread.finished.connect(self.on_accum_done)
            self._accum_thread.start()
            self._set_busy(True)
            self.statusBar().showMessage("Accumulation started")

    def _build_sweep_plan(self, sweep_power: bool, angle_list_mode: bool) -> Tuple[List[dict], list, str]:
        """(plan, entries, error) of a sweep: the stage entries with their file base names.

        No popups, connection checks or side effects (also used for the scan time estimate).
        Angle-list mode returns the angles as ``entries``.
        """
        # Skip N does not apply to a power list (the targets are already the chosen points)
        power_stride = 1 if self._power_list_active() else max(1, int(self.powerSkipSpin.value()) + 1)
        plan: List[dict] = []
        if sweep_power and angle_list_mode:
            angles = self._parse_angle_list()
            if not angles:
                return [], [], "Enter at least one angle in the angle list."
            for angle in angles:
                a_label = f"{angle:.3f}".replace(".", "p").replace("-", "m")
                base = f"PL_A{a_label}deg"
                entry = _types.SimpleNamespace(position_deg=angle, power_w=None, series="angle_list")
                plan.append({"entry": entry, "base": base})
            return plan, angles, ""
        if not sweep_power:
            return [{"entry": None, "base": "PL_manual"}], [], ""
        if self._calib_mode in ("dual", "hwp_nd"):
            if not self._dual_entries:
                return [], [], "Load a calibration file first."
            entries = list(self._dual_entries)
            if power_stride > 1:
                entries = entries[::power_stride]
            if not entries:
                return [], [], "No entries in dual-wheel calibration file."
            for entry in entries:
                power_val = power_key(entry.power_w, scale=self._calib_scale)
                power_label = f"{power_val:.6g}".replace(".", "p")
                a_label = f"{entry.a_deg:.3f}".replace(".", "p")
                b_label = f"{entry.b_deg:.3f}".replace(".", "p")
                base = f"PL_A{a_label}_B{b_label}_P{power_label}{self._calib_unit}"
                plan.append({"entry": entry, "base": base})
            return plan, entries, ""
        if self._calib_data is None:
            return [], [], "Load a power calibration file first."
        series = self.seriesCombo.currentText().strip()
        entries = sorted(self._single_series_entries(series), key=lambda e: e.power_w)
        if power_stride > 1:
            entries = entries[::power_stride]  # stride from the lowest power so it is always kept
        # single wheel: sweep from the highest to the lowest power (the scan ends at the lowest)
        entries = entries[::-1]
        if not entries:
            start, end = self._wheel_range()
            return [], [], f"No entries for series '{series}' in the ND wheel range {start:g}° → {end:g}°."
        for entry in entries:
            power_val = power_key(entry.power_w, scale=self._calib_scale)
            power_label = f"{power_val:.6g}".replace(".", "p")
            series_label = self._safe_series_label(entry.series)
            base = f"PL_{series_label}_{power_label}{self._calib_unit}"
            plan.append({"entry": entry, "base": base})
        return plan, entries, ""

    def on_sweep(self):
        if self._busy:
            return

        sweep_power = self.sweepPowerChk.isChecked()
        sweep_voltage = self.sweepVoltageChk.isChecked()
        if not (sweep_power or sweep_voltage):
            QtWidgets.QMessageBox.warning(self, "Sweep", "Enable sweep power and/or sweep voltage.")
            return
        measure_pl = self._measure_pl()
        measure_trpl = self._measure_trpl()
        if measure_pl and not self.camDev.is_connected:
            QtWidgets.QMessageBox.warning(self, "Sweep", "Camera must be connected (Measure includes PL).")
            return
        if measure_trpl and not self._spad_ok():
            QtWidgets.QMessageBox.warning(self, "Sweep", "TCSPC detector must be connected (Measure includes TRPL).")
            return
        didp = self.didpChk.isChecked()
        if didp and not self.flipperDev.is_connected:
            QtWidgets.QMessageBox.warning(self, "Sweep", "dI/dP needs the pump-2 flipper connected.")
            return
        center = float(self.centerSpin.value())
        if (measure_pl and abs(center) > 1e-9 and self.liveView is not None
                and self.liveView.get_wavelength_axis() is None and getattr(self.cam, "_wavelength_axis_nm", None) is None):
            if not self.specDev.is_connected:
                why = "the spectrograph is not connected"
            else:
                why = "the spectrograph gave no calibration" + (
                    f" ({self.specDev.last_wl_error})" if self.specDev.last_wl_error else "")
            ans = QtWidgets.QMessageBox.question(
                self, "Sweep",
                f"No wavelength axis at {center:g} nm: {why}.\n\n"
                "The sweep will try the spectrograph once more; if that fails too, the spectra are saved "
                "against pixel columns (no wavelength column).\n\nStart the sweep anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return

        dyn = self._dyn_settings() if measure_trpl else None
        if dyn is not None:
            err = dyn.validate()
            if err:
                QtWidgets.QMessageBox.warning(self, "Sweep", f"Dynamic TRPL time: {err}.")
                return
            if dyn.dark_ref is None:
                win = ("automatic window" if dyn.pre_t0_window is None
                       else "%g-%g ns" % tuple(sorted(dyn.pre_t0_window)))
                ans = QtWidgets.QMessageBox.question(
                    self, "Sweep",
                    f"No {self._det.key} dark reference for the dynamic TRPL time.\n\n"
                    f"The peak check subtracts each pixel's pre-t0 level instead ({win}).\n\n"
                    "Start the sweep anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if ans != QtWidgets.QMessageBox.Yes:
                    return
        self._sweep_dyn = dyn
        self._dyn_results = []

        self._reset_sweep_data()

        mode = self._sweep_gate_mode()
        grid_active = sweep_voltage and mode == "dual" and self.gridSweepChk.isChecked()
        nsteps = int(self.sweepStepsSpin.value())
        # In grid mode gate A and gate B use independent step counts.
        nsteps_a = int(self.sweepStepsSpin.value())
        nsteps_b = int(self.backStepsSpin.value())
        front_v0 = float(self.frontV0Spin.value())
        front_step = float(self.frontStepSpin.value())
        back_v0 = float(self.backV0Spin.value())
        back_step = float(self.backStepSpin.value())

        if sweep_voltage:
            if mode in ("front", "dual") and self.front_dev is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Front gate must be connected.")
                return
            if mode in ("back", "dual") and self.back_dev is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Back gate must be connected.")
                return
            if grid_active:
                if nsteps_a < 1 or nsteps_b < 1:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Gate A and B steps must be >= 1.")
                    return
            elif nsteps < 1:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Voltage steps must be >= 1.")
                return

            def _check_v_limits(v0: float, step: float, n: int, label: str):
                vend = v0 + (n - 1) * step
                if abs(v0) > KEITHLEY_V_LIMIT or abs(vend) > KEITHLEY_V_LIMIT:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Voltage limit",
                        f"{label} start/end exceed +/-{KEITHLEY_V_LIMIT:.0f} V",
                    )
                    return False
                return True

            if grid_active:
                if not _check_v_limits(front_v0, front_step, nsteps_a, "Front gate (A)"):
                    return
                if not _check_v_limits(back_v0, back_step, nsteps_b, "Back gate (B)"):
                    return
            else:
                if mode in ("front", "dual") and not _check_v_limits(front_v0, front_step, nsteps, "Front gate"):
                    return
                if mode in ("back", "dual") and not _check_v_limits(back_v0, back_step, nsteps, "Back gate"):
                    return

        self._angle_list_mode = sweep_power and self.angleListChk.isChecked()

        plan, entries, plan_err = self._build_sweep_plan(sweep_power, self._angle_list_mode)
        if plan_err:
            QtWidgets.QMessageBox.warning(self, "Sweep", plan_err)
            return
        if sweep_power and not self._angle_list_mode and self._power_list_active():
            skipped = [r for r in self._power_plan if not r.ok]
            if skipped:
                ans = QtWidgets.QMessageBox.question(
                    self, "Sweep",
                    f"{len(skipped)} of {len(self._power_plan)} target powers cannot be reached within "
                    f"±{self.powerListTolSpin.value():g} % and will be skipped:\n"
                    + ", ".join(hwp_planner.fmt_power(r.target_w) for r in skipped[:10])
                    + (" …" if len(skipped) > 10 else "") + "\n\nStart the sweep anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if ans != QtWidgets.QMessageBox.Yes:
                    return
        if sweep_power and self._angle_list_mode:
            if self.stage is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Stage A must be connected for angle list sweep.")
                return
            angles = entries
            entries = []
            # use angles as pseudo-power keys for map display
            self._power_target_counts = {a: 1 for a in angles}
            self._power_done_counts = {a: 0 for a in angles}
            self._total_entries = len(angles)
            self._power_view_keys = list(angles)
        elif sweep_power:
            if self._calib_mode in ("dual", "hwp_nd"):
                if self.stage is None or self.stage_b is None:
                    QtWidgets.QMessageBox.warning(self, "Sweep", "Both stages must be connected for dual sweep.")
                    return
            elif self.stage is None:
                QtWidgets.QMessageBox.warning(self, "Sweep", "Stage must be connected.")
                return
        else:
            self._power_view_keys = [0.0]

        if not self._angle_list_mode:
            self._power_target_counts.clear()
            if sweep_power:
                for entry in entries:
                    key = power_key(entry.power_w, scale=self._calib_scale)
                    self._power_target_counts[key] = self._power_target_counts.get(key, 0) + 1
            self._power_done_counts = {k: 0 for k in self._power_target_counts}
            self._total_entries = len(entries) if sweep_power else 0
            if sweep_power:
                self._power_view_keys = sorted(self._power_target_counts.keys())


        self._voltage_keys = []
        self._voltage_labels = []
        self._voltage_pairs = {}
        self._grid_active = grid_active
        self._a_keys = []
        self._a_labels = []
        self._b_keys = []
        self._b_labels = []
        if grid_active:
            # Gate A values (the fixed axis) and gate B values (the swept axis
            # / x-axis of the linecut-vs-voltage map).
            for k in range(nsteps_a):
                va = front_v0 + k * front_step
                key = self._normalize_voltage_key(va)
                if key is None or key in self._a_keys:
                    continue
                self._a_keys.append(key)
                self._a_labels.append(f"{va:.6g} V")
            for k in range(nsteps_b):
                vb = back_v0 + k * back_step
                key = self._normalize_voltage_key(vb)
                if key is None or key in self._b_keys:
                    continue
                self._b_keys.append(key)
                self._b_labels.append(f"{vb:.6g} V")
            # The voltage slider / labels drive the gate-B axis.
            self._voltage_keys = list(self._b_keys)
            self._voltage_labels = list(self._b_labels)
            self._voltage_pairs = {b: (None, b) for b in self._b_keys}
        elif sweep_voltage:
            for k in range(nsteps):
                vf = front_v0 + k * front_step if mode in ("front", "dual") else None
                vb = back_v0 + k * back_step if mode in ("back", "dual") else None
                key_raw = vf if mode in ("front", "dual") else vb
                key = self._normalize_voltage_key(key_raw)
                if key is None:
                    continue
                if mode == "dual":
                    label = f"Vf={vf:.6g}, Vb={vb:.6g} V"
                elif mode == "front":
                    label = f"{vf:.6g} V"
                else:
                    label = f"{vb:.6g} V"
                if key in self._voltage_pairs:
                    continue
                self._voltage_keys.append(key)
                self._voltage_labels.append(label)
                self._voltage_pairs[key] = (vf, vb)
        else:
            self._voltage_keys = [0.0]
            self._voltage_labels = ["--"]
            self._voltage_pairs = {0.0: (None, None)}

        self._update_voltage_slider()
        self._update_power_slider()
        self._update_a_slider()

        # (the sweep worker puts the camera in Snap / Single mode itself)
        self._resolve_save_dir()
        self._copy_dark_to_sweep_folder()
        self._copy_wlcal_to_sweep_folder()

        meta = self._build_metadata_base()
        wl_axis = self.liveView.get_wavelength_axis() if self.liveView is not None else None

        front_icomp_a = float(self.frontIcompSpin.value()) * 1e-9 if sweep_voltage and mode in ("front", "dual") else None
        back_icomp_a = float(self.backIcompSpin.value()) * 1e-9 if sweep_voltage and mode in ("back", "dual") else None

        sweep_order = "voltage_outer" if self.sweepOrderVoltageOuterRb.isChecked() else "power_outer"

        self._sweep_thread = PowerVoltageSweepThread(
            self.cam,
            self.stage,
            self.stage_b,
            plan,
            int(self.accumSpin.value()),
            float(self.exposureSpin.value()),
            float(DEFAULT_RAMP_STEP_DEG),
            meta,
            wl_axis,
            sweep_power=sweep_power,
            sweep_voltage=sweep_voltage,
            sweep_mode=mode,
            front_dev=self.front_dev,
            back_dev=self.back_dev,
            front_v0=front_v0,
            front_step=front_step,
            back_v0=back_v0,
            back_step=back_step,
            nsteps=nsteps,
            front_icomp_a=front_icomp_a,
            back_icomp_a=back_icomp_a,
            ramp_step_v=float(RAMP_STEP_V),
            ramp_dwell_s=float(RAMP_DWELL_S),
            settle_ms=float(SWEEP_SETTLE_MS),
            zero_v_eps=float(ZERO_V_EPS),
            zero_v_extra_settle_ms=float(ZERO_V_EXTRA_SETTLE_MS),
            save_dir=self._save_dir,
            sweep_order=sweep_order,
            grid_sweep=grid_active,
            nsteps_a=nsteps_a,
            nsteps_b=nsteps_b,
            readout_rate=self._readout_rate_label,
            preamp_gain=self._preamp_gain_label,
            output_amp=self._output_amp_label,
            measure_pl=measure_pl,
            measure_trpl=measure_trpl,
            tcspc_host=DEFAULT_TCSPC_HOST,
            tcspc_port=int(DEFAULT_TCSPC_PORT),
            tcspc_acq_s=float(self.tcspcAcqSpin.value()),
            tcspc_bin_ps=int(self.tcspcBinSpin.value()),
            spad_count_integration_ms=SPAD_SWEEP_COUNT_MS,
            detector=self._det.key,
            didp=didp,
            flipper=self.flipperDev.flipper if didp else None,
            pump2_on_position=self._pump2_on_position(),
            flipper_settle_ms=float(self.flipperSettleSpin.value()),
            pump2_tags=PUMP2_TAGS,
            ramp_micro_read=self.rampReadChk.isChecked(),
            long_ramp_steps=LONG_RAMP_STEPS,
            trpl_dynamic=dyn,
        )
        self._sweep_thread.frame_ready.connect(self.on_sweep_frame)
        self._sweep_thread.vi_ready.connect(self.on_sweep_vi)
        self._sweep_thread.point_done.connect(self.on_sweep_point)
        self._sweep_thread.power_done.connect(self.on_sweep_power_done)
        self._sweep_thread.spad_count_ready.connect(self.on_sweep_spad_count)
        self._sweep_thread.trpl_update.connect(self.on_sweep_trpl_update)
        self._sweep_thread.trpl_done.connect(self.on_sweep_trpl_done)
        self._sweep_thread.status.connect(self.statusBar().showMessage)
        self._sweep_thread.timing.connect(self._on_sweep_timing)
        self._sweep_thread.done.connect(self.on_sweep_done)
        self._start_scan_timing(self._sweep_thread)
        self._sweep_pending = True
        self._set_busy(True)
        self.statusBar().showMessage("Sweeping...")
        thread = self._sweep_thread

        def start():
            # start only if not aborted while the device polls were letting go
            if self._sweep_pending and self._sweep_thread is thread:
                self._sweep_pending = False
                self._retain(thread).start()
                self._sync_controls()

        def when_quiet():
            if measure_trpl:
                self.spadDev.stop_live(then=start)  # the sweep's TRPL clients get the pSPAD server
            else:
                start()

        devices = [self.camDev, self.specDev, self.stageDevs["A"], self.stageDevs["B"], *self.gateDevs.values(),
                   self.flipperDev]
        self._start_when_quiet(devices, when_quiet)

    def on_abort(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
        if self._accum_thread is not None and self._accum_thread.isRunning():
            self._accum_thread.stop()
        if self._sweep_thread is not None and self._sweep_thread.isRunning():
            self._sweep_thread.stop()
        if getattr(self, "_sweep_pending", False):
            # aborted before the sweep thread started (device polls were still letting go)
            self._sweep_pending = False
            self._sweep_thread = None
            self._finish_scan_timing("aborted")
        if self.spadDev.trpl_running:
            self.spadDev.stop_trpl()
        if self._front_ramp_controller is not None:
            try:
                self._front_ramp_controller.abort()
            except Exception:
                pass
        if self._back_ramp_controller is not None:
            try:
                self._back_ramp_controller.abort()
            except Exception:
                pass
        for ctrl in self._stage_controllers.values():
            if ctrl is not None:
                try:
                    ctrl.abort()
                except Exception:
                    pass
        for dev in self.gateDevs.values():
            dev.abort_rampdown()  # a disconnect ramp to 0 V stops; that gate's output stays on
        for stage in (self.stage, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass
        for which in ("A", "B"):
            self._stage_busy_by[which] = False
        try:
            if self.cam is not None and self.cam.cam is not None:
                self.cam.stop_stream()
        except Exception:
            pass
        self._live_active = False
        self._aborting = True
        self._set_busy(False)
        self.statusBar().showMessage("Aborted")

    def on_stop_live(self):
        # stop() then poll (never thread.wait() in the GUI thread); _on_live_done runs on finished
        if self._live_thread is not None and self._live_thread.isRunning():
            self._live_thread.stop()
            self.statusBar().showMessage("Stopping live...")
        elif self._live_thread is not None:
            self._on_live_done()

    def on_live_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def on_accum_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        accum_idx = fr.get("accum_idx")
        accum_n = fr.get("accum_n")
        if accum_idx and accum_n:
            self.statusBar().showMessage(f"Accumulated {accum_idx}/{accum_n}")
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def on_accum_done(self):
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Accumulation done")

    def on_sweep_frame(self, fr: dict):
        raw = fr.get("image")
        if raw is None:
            return
        accum_idx = fr.get("accum_idx")
        accum_n = fr.get("accum_n")
        if accum_idx and accum_n:
            self.statusBar().showMessage(f"Accumulated {accum_idx}/{accum_n}")
        if self.liveView is not None:
            self.liveView.update_frame(fr)

    def on_sweep_vi(self, payload: dict):
        v_front = payload.get("v_front")
        i_front = payload.get("i_front")
        v_back = payload.get("v_back")
        i_back = payload.get("i_back")
        if v_front is not None and i_front is not None:
            self._apply_gate_readout("front", float(v_front), float(i_front))
        if v_back is not None and i_back is not None:
            self._apply_gate_readout("back", float(v_back), float(i_back))

    def _on_live_done(self):
        if self._live_thread is not None and self._live_thread.isRunning():
            return
        self._live_thread = None
        self._live_active = False
        self._set_busy(False)
        self.statusBar().showMessage("Live stopped")

    def on_sweep_point(self, payload: dict):
        if payload.get("pump2"):
            return  # dI/dP: the maps show the pump-2-off data (pump-2-on is saved, not displayed)
        entry = payload.get("entry")
        raw = payload.get("image")
        if raw is None:
            return

        if self.liveView is None:
            return

        power_key_val = 0.0
        if entry is not None:
            try:
                if self._angle_list_mode and hasattr(entry, "position_deg"):
                    power_key_val = float(entry.position_deg)
                else:
                    power_key_val = power_key(entry.power_w, scale=self._calib_scale)
            except Exception:
                power_key_val = 0.0

        voltage_key = self._normalize_voltage_key(payload.get("voltage_key"))
        if voltage_key is None:
            voltage_key = 0.0

        if voltage_key not in self._voltage_keys:
            self._voltage_keys.append(voltage_key)
            self._voltage_labels.append(f"{voltage_key:.6g} V")
            self._update_voltage_slider()

        try:
            raw_store = np.asarray(raw).copy()
        except Exception:
            raw_store = np.asarray(raw)
        rank = self._entry_rank(entry) if entry is not None else 0

        if self._grid_active and payload.get("grid"):
            # Grid data is indexed power -> A -> B (B == voltage_key).
            a_key = self._normalize_voltage_key(payload.get("a_voltage"))
            b_key = self._normalize_voltage_key(payload.get("b_voltage"))
            if a_key is None:
                a_key = 0.0
            if b_key is None:
                b_key = voltage_key
            if a_key not in self._a_keys:
                self._a_keys.append(a_key)
                self._a_labels.append(f"{a_key:.6g} V")
                self._update_a_slider()
            if b_key not in self._b_keys:
                self._b_keys.append(b_key)
                self._b_labels.append(f"{b_key:.6g} V")
            a_map = self._grid_images.setdefault(power_key_val, {})
            b_map = a_map.setdefault(a_key, {})
            cur = b_map.get(b_key)
            if cur is None or rank <= cur.get("rank", rank):
                b_map[b_key] = {"rank": rank, "raw": raw_store}
            self._refresh_linecut_maps()
            return

        entry_map = self._pv_images.setdefault(power_key_val, {})
        cur = entry_map.get(voltage_key)
        if cur is None or rank <= cur.get("rank", rank):
            entry_map[voltage_key] = {"rank": rank, "raw": raw_store}

        self._refresh_linecut_maps()

    def on_sweep_power_done(self, payload: dict):
        entry = payload.get("entry")
        if entry is None:
            return

        if self._angle_list_mode and hasattr(entry, "position_deg"):
            key = float(entry.position_deg)
        else:
            key = power_key(entry.power_w, scale=self._calib_scale)
        self._taken_entries.add(self._entry_key(entry))
        count = self._power_done_counts.get(key, 0) + 1
        self._power_done_counts[key] = count
        if count >= self._power_target_counts.get(key, 1):
            self._mark_power_taken(key)

        if self._total_entries and len(self._taken_entries) >= self._total_entries:
            self._write_session_metadata()

    def _power_key_of(self, entry) -> float:
        """Map key of a sweep entry (power key, or the stage angle in angle-list mode)."""
        if entry is None:
            return 0.0
        try:
            if self._angle_list_mode and hasattr(entry, "position_deg"):
                return float(entry.position_deg)
            return power_key(entry.power_w, scale=self._calib_scale)
        except Exception:
            return 0.0

    def on_sweep_spad_count(self, payload: dict):
        if payload.get("pump2"):
            return
        snap = payload.get("snapshot")
        if is_count_snapshot(snap):
            self._spad_last_count_snapshot = snap
            if self._spad_live_enabled:
                self._update_spad_count_maps(snap)

    def on_sweep_trpl_update(self, payload: dict):
        if payload.get("pump2"):
            return
        snap = payload.get("snapshot")
        if not is_trpl_snapshot(snap):
            return
        self._show_live_hist(snap)

    def on_sweep_trpl_done(self, payload: dict):
        self._dyn_record(payload)
        if payload.get("pump2"):
            return
        snap = payload.get("snapshot")
        if not is_trpl_snapshot(snap):
            return
        self._spad_latest_hist_snapshot = snap
        voltage_key = self._normalize_voltage_key(payload.get("voltage_key"))
        if voltage_key is None:
            voltage_key = 0.0
        a_key = None
        if self._grid_active and payload.get("grid"):
            a_key = self._normalize_voltage_key(payload.get("a_voltage"))
        self._store_trpl_snapshot(self._power_key_of(payload.get("entry")), float(voltage_key), snap,
                                  payload.get("filepath"), a_key=a_key)
        self._update_histogram_tab()
        self._update_lifetime_tab()

    def on_sweep_done(self, status: str):
        self._aborting = False
        problems = list(getattr(self._sweep_thread, "trpl_problems", []) or [])
        self._sweep_thread = None
        self._finish_scan_timing(status)
        self._write_dynamic_summary(status)
        self._write_problems_summary(problems)
        self._sweep_dyn = None
        self._set_busy(False)
        self._update_lifetime_tab()
        if self._spad_live_enabled and self.spadDev.state == spad_det.IDLE:
            self.spadDev.start_live()
        if status == "ok":
            self.statusBar().showMessage("Sweep complete")
        elif status == "aborted":
            self.statusBar().showMessage("Sweep aborted")
        elif status == "trip":
            self.statusBar().showMessage("Sweep TRIPPED (current limit exceeded)")
        else:
            self.statusBar().showMessage(str(status))
        if self.homeAfterScanChk.isChecked() and status != "aborted":
            self._home_gates_after_scan(status)

    def _home_gates_after_scan(self, status: str, tries: int = 300) -> None:
        """Home Both Gates once the finished sweep thread no longer drives them (``done`` is its last call)."""
        if self._busy:
            return  # a new sweep started in the meantime: leave the gates to it
        if self._gate_threads_running():
            if tries > 0:
                QtCore.QTimer.singleShot(50, lambda: self._home_gates_after_scan(status, tries - 1))
            else:
                self.statusBar().showMessage("Gates not homed after the scan: a gate ramp is still running")
            return
        gates = [w for w, d in (("front", self.front_dev), ("back", self.back_dev)) if d is not None]
        done = "Sweep complete" if status == "ok" else f"Sweep ended ({status})"
        if not gates:
            self.statusBar().showMessage(f"{done}; no gate connected to home")
            return
        self.on_home_both_gates()
        self.statusBar().showMessage(f"{done}; homing {' and '.join(gates)} gate to 0 V")
        self._sync_controls()

    # -----------------
    # Scan time estimate (status bar, bottom right)
    # -----------------
    def _wire_scan_estimate(self) -> None:
        self._scan_est_timer = QtCore.QTimer(self)
        self._scan_est_timer.setSingleShot(True)
        self._scan_est_timer.setInterval(250)
        self._scan_est_timer.timeout.connect(self._update_scan_estimate)
        # readbacks, connects and calibration loads change the estimate too
        self._scan_idle_timer = QtCore.QTimer(self)
        self._scan_idle_timer.setInterval(2000)
        self._scan_idle_timer.timeout.connect(self._update_scan_estimate)
        self._scan_idle_timer.start()

        def later(*_args):
            self._scan_est_timer.start()

        for spin in (self.exposureSpin, self.accumSpin, self.tcspcAcqSpin, self.flipperSettleSpin,
                     self.frontV0Spin, self.frontStepSpin, self.backV0Spin, self.backStepSpin,
                     self.sweepStepsSpin, self.backStepsSpin, self.powerSkipSpin, self.hwpMinSpin, self.hwpMaxSpin,
                     self.wheelStartSpin, self.wheelEndSpin):
            spin.valueChanged.connect(later)
        for combo in (self.readoutRateCombo, self.sweepGateCombo, self.measureCombo, self.seriesCombo,
                      self.ndWheelAngleCombo):
            combo.currentIndexChanged.connect(later)
        for chk in (self.sweepPowerChk, self.sweepVoltageChk, self.gridSweepChk, self.angleListChk, self.didpChk,
                    self.sweepOrderPowerOuterRb, self.sweepOrderVoltageOuterRb, self.rampReadChk):
            chk.toggled.connect(later)
        self.angleListEdit.textChanged.connect(later)
        for sig in (self.trplTimeCombo.currentIndexChanged, self.dynFirstSpin.valueChanged,
                    self.dynFirstPlChk.toggled, self.dynStepSpin.valueChanged, self.dynMaxSpin.valueChanged):
            sig.connect(later)
        self.powerListChk.toggled.connect(later)  # edits re-solve first (_apply_power_list kicks the estimate)
        later()

    def _scan_start_state(self) -> Dict:
        """Hardware state the estimate starts from (cached readbacks only: no device I/O)."""

        def angle(stage) -> Optional[float]:
            try:
                pos = float(stage.get_position_cached()) if stage is not None else float("nan")
            except Exception:
                return None
            return None if np.isnan(pos) else pos

        vel = self._stage_velocity["A"] or self._stage_velocity["B"]
        label = str(self._readout_rate_label)
        gates = [d for d in (self.front_dev, self.back_dev) if d is not None]
        return {
            "gate_2400": all(bool(getattr(d, "is_2400", False)) for d in gates) if gates else None,
            "readout_label": label,
            "readout_s": self._cam_readout_s.get(label),
            "stage_velocity": vel if vel and 0.1 <= vel[0] <= 1000.0 else None,
            "start_a": angle(self.stage),
            "start_b": angle(self.stage_b),
            "start_vf": self._gate_last_v["front"] if self.front_dev is not None else None,
            "start_vb": self._gate_last_v["back"] if self.back_dev is not None else None,
        }

    def _scan_spec_from_ui(self, plan: List[dict]) -> ScanSpec:
        """The spec ``on_sweep`` would hand the worker with the current settings."""
        sweep_power = self.sweepPowerChk.isChecked()
        sweep_voltage = self.sweepVoltageChk.isChecked()
        mode = self._sweep_gate_mode()
        grid = sweep_voltage and mode == "dual" and self.gridSweepChk.isChecked()
        # a gate the sweep needs counts as connected (it must be before the sweep starts)
        front = self.front_dev is not None or (sweep_voltage and mode in ("front", "dual"))
        back = self.back_dev is not None or (sweep_voltage and mode in ("back", "dual"))
        return ScanSpec(
            targets=[ScanSpec.entry_target(item.get("entry")) for item in plan] if sweep_power else [None],
            sweep_power=sweep_power, sweep_voltage=sweep_voltage, mode=mode, grid=grid,
            order="voltage_outer" if self.sweepOrderVoltageOuterRb.isChecked() else "power_outer",
            nsteps=int(self.sweepStepsSpin.value()), nsteps_a=int(self.sweepStepsSpin.value()),
            nsteps_b=int(self.backStepsSpin.value()),
            front_v0=float(self.frontV0Spin.value()), front_step=float(self.frontStepSpin.value()),
            back_v0=float(self.backV0Spin.value()), back_step=float(self.backStepSpin.value()),
            front_present=front, back_present=back,
            ramp_step_v=float(RAMP_STEP_V), ramp_dwell_s=float(RAMP_DWELL_S), settle_ms=float(SWEEP_SETTLE_MS),
            zero_v_eps=float(ZERO_V_EPS), zero_v_extra_settle_ms=float(ZERO_V_EXTRA_SETTLE_MS),
            ramp_step_deg=float(DEFAULT_RAMP_STEP_DEG),
            measure_pl=self._measure_pl(), measure_trpl=self._measure_trpl(),
            exposure_ms=float(self.exposureSpin.value()), accum_n=int(self.accumSpin.value()),
            tcspc_acq_s=float(self.tcspcAcqSpin.value()), didp=self.didpChk.isChecked(),
            flipper_settle_ms=float(self.flipperSettleSpin.value()),
            ramp_micro_read=self.rampReadChk.isChecked(),
            long_ramp_steps=LONG_RAMP_STEPS, detector=self._det.key,
            **ScanSpec.dynamic_fields(self._dyn_settings(False)),
            **self._scan_start_state(),
        )

    def _schedule_for(self, spec: ScanSpec) -> Schedule:
        key = spec.key()
        if self._scan_sched_cache is None or self._scan_sched_cache[0] != key:
            self._scan_sched_cache = (key, simulate(spec))
        return self._scan_sched_cache[1]

    def _update_scan_estimate(self) -> None:
        if self._scan_tracker is not None or getattr(self, "_sweep_pending", False):
            return  # the running scan owns the readout
        w = self.scanTimeWidget
        try:
            sweep_power = self.sweepPowerChk.isChecked()
            if not (sweep_power or self.sweepVoltageChk.isChecked()):
                w.set_unavailable("enable sweep power and/or voltage")
                return
            plan, _, err = self._build_sweep_plan(sweep_power, sweep_power and self.angleListChk.isChecked())
            if err:
                w.set_unavailable(err.rstrip(".").lower())
                return
            spec = self._scan_spec_from_ui(plan)
            sched = self._schedule_for(spec)
            pred = self._timing_model.predict(sched)
        except Exception as exc:
            w.set_unavailable(f"error: {exc}")
            return
        notes = []
        if spec.sweep_power and spec.start_a is None:
            notes.append("Stage A position unknown (not connected): the move to the first point is not counted.")
        if spec.sweep_voltage and (self.front_dev is None and spec.mode in ("front", "dual")
                                   or self.back_dev is None and spec.mode in ("back", "dual")):
            notes.append("Gate not connected: its first ramp is counted from 0 V.")
        if spec.measure_pl and spec.readout_s is None and not self.camDev.is_connected:
            notes.append("Andor not connected: readout time not known yet.")
        w.set_estimate(pred, spec, self._timing_model, notes)

    def _start_scan_timing(self, thread: PowerVoltageSweepThread) -> None:
        spec = ScanSpec.from_worker(thread, **self._scan_start_state())
        sched = self._schedule_for(spec)
        self._timing_model.reset_live()  # this scan learns from the saved values
        tracker = ScanTracker(sched, self._timing_model, time.monotonic())
        self._scan_tracker = tracker
        self.scanTimeWidget.start(tracker)
        self._scan_csv_path = None
        if self._save_dir:
            path = os.path.join(self._save_dir, "scan_timing.csv")
            idx = 1
            while os.path.exists(path):
                path = os.path.join(self._save_dir, f"scan_timing_{idx}.csv")
                idx += 1
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(SCAN_TIMING_CSV_HEADER)
                self._scan_csv_path = path
            except Exception as exc:
                self.statusBar().showMessage(f"Scan timing log not written: {exc}")

    def _on_sweep_timing(self, ev: dict) -> None:
        tr = self._scan_tracker
        if tr is None or not isinstance(ev, dict):
            return
        row = tr.on_event(ev)
        if row and self._scan_csv_path:
            try:
                with open(self._scan_csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except Exception:
                self._scan_csv_path = None
        self.scanTimeWidget.refresh_running()

    def _finish_scan_timing(self, status: str) -> None:
        tr, self._scan_tracker = self._scan_tracker, None
        if tr is None:
            return
        learned = tr.finalize(status)
        self.scanTimeWidget.finish(status)
        self._write_scan_timing_summary(tr, status, learned)

    def _write_scan_timing_summary(self, tr: ScanTracker, status: str, learned: bool) -> None:
        """Append the scan's timing to session.md (written first if the sweep did not finish)."""
        if not self._save_dir or tr.done == 0:
            return
        if not self._session_saved:
            self._write_session_metadata()
        actual = tr.elapsed_s()
        dev = 100.0 * (actual - tr.estimate_s) / tr.estimate_s if tr.estimate_s > 0 else 0.0
        lines = [
            "",
            "## Timing",
            f"- Status: {status}",
            f"- Points: {tr.done}/{tr.n_points}" + (" (dI/dP: 2 acquisitions per point)" if tr.sched.spec.didp else ""),
            f"- Estimated at start: {fmt_duration(tr.estimate_s)} ({tr.estimate_s:.0f} s)",
            f"- Actual: {fmt_duration(actual)} ({actual:.0f} s), {dev:+.1f}% vs the estimate",
            f"- Per-point log: {os.path.basename(self._scan_csv_path) if self._scan_csv_path else '--'}",
            f"- Learned timing saved for the next estimates: {'yes' if learned else 'no'}",
        ]
        try:
            with open(os.path.join(self._save_dir, "session.md"), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            self.statusBar().showMessage(f"Timing summary not written: {exc}")

    # -----------------
    # Pump-2 flipper (dI/dP)
    # -----------------
    def _pump2_on_position(self) -> int:
        return 2 if self.pump2PosCombo.currentIndex() == 1 else 1

    def _pump2_text(self, pos: Optional[int]) -> str:
        if pos is None:
            return "in transit"
        return f"Position {pos} | pump 2 {'ON' if pos == self._pump2_on_position() else 'OFF'}"

    def _on_pump2_pos_changed(self, _idx: int) -> None:
        self._flipper_settings.setValue(FLIPPER_SETTINGS[2], self._pump2_on_position())
        if self.flipperDev.is_connected:
            self.flipperStatusLbl.setText(self._pump2_text(self._flipper_position))

    def _connect_flipper(self) -> None:
        serial = self.flipperSerialEdit.text().strip()
        if serial:
            self.flipperDev.connect_device(serial=serial)

    def _on_flipper_state(self) -> None:
        state = self.flipperDev.state
        self._set_dot(self.flipperDot, state == CONNECTED)
        if state == DISCONNECTED:
            self._flipper_position = None
            self.flipperStatusLbl.setText("Disconnected")
        elif state != CONNECTED:
            self.flipperStatusLbl.setText(state.capitalize() + "...")
        self._sync_controls()

    def _on_flipper_connected(self, info) -> None:
        if isinstance(info, dict):
            self._flipper_position = info.get("position")
            self.flipperStatusLbl.setText(self._pump2_text(self._flipper_position))
        self._sync_controls()

    def _on_flipper_readback(self, d: dict) -> None:
        if not d:
            return
        if "error" in d:
            self.flipperStatusLbl.setText(f"Read error: {d['error'][:40]}")
            return
        self._flipper_position = d.get("position")
        self.flipperStatusLbl.setText(self._pump2_text(self._flipper_position))

    def _move_pump2(self, on: bool) -> None:
        ff = self.flipperDev.flipper
        if ff is None:
            return
        target = self._pump2_on_position() if on else (1 if self._pump2_on_position() == 2 else 2)
        self.flipperStatusLbl.setText(f"moving to position {target}...")
        if self.flipperDev.call(lambda: ff.move_to(target), self._on_pump2_moved):
            self._sync_controls()

    def _on_pump2_moved(self, res, err: str) -> None:
        if err:
            self.statusBar().showMessage(f"Flipper move failed: {err}")
        elif res is not None:
            self._flipper_position = res.position
            self.flipperStatusLbl.setText(self._pump2_text(res.position))
            self.statusBar().showMessage(f"Flipper at position {res.position} ({res.duration_s:.2f} s)")
        QtCore.QTimer.singleShot(0, self._sync_controls)

    # -----------------
    # SPAD23 / SPAD93 (pSPAD) device: detect, connect, live counts, calibrate + align, Take Histogram
    # -----------------
    def _connect_spad(self) -> None:
        """Ask the server which detector it runs (DetectWorker), then connect (``_on_detect_done``)."""
        if self._detect_worker is not None or self.spadDev.state != spad_det.DISCONNECTED:
            return
        self._det_choice = self.detectorCombo.currentText()
        self._det_settings.setValue(DETECTOR_SETTINGS_KEY, self._det_choice)
        w = DetectWorker(DEFAULT_TCSPC_HOST, int(DEFAULT_TCSPC_PORT), self)
        self._detect_cancelled = False
        w.done.connect(self._on_detect_done)
        w.finished.connect(w.deleteLater)
        self._detect_worker = w
        self._update_tcspc_status(False, "detecting the SPAD (one short count frame)...")
        w.start()
        self._sync_controls()

    def _on_detect_done(self, key: str, msg: str, ok: bool) -> None:
        self._detect_worker = None
        if self._detect_cancelled:
            self._detect_cancelled = False
            self._update_tcspc_status(False, "disconnected")
            self._sync_controls()
            return
        choice = self._det_choice
        target = None
        if not ok:
            QtWidgets.QMessageBox.warning(self, "TCSPC detector", msg)
        elif choice == DETECTOR_AUTO:
            if key:
                target = key
            else:
                QtWidgets.QMessageBox.warning(
                    self, "TCSPC detector", msg + "\n\nSet Detector to SPAD23 or SPAD93 and connect again.")
        else:
            target = choice
            if key and key != choice:
                ans = QtWidgets.QMessageBox.question(
                    self, "TCSPC detector",
                    f"{choice} is selected, but the pSPAD server looks like a {key}:\n{msg}.\n\nUse {key}?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Yes)
                if ans == QtWidgets.QMessageBox.Cancel:
                    target = None
                elif ans == QtWidgets.QMessageBox.Yes:
                    target = key
        if target is None or not self._apply_detector(target):
            self._update_tcspc_status(False, "disconnected")
            self._sync_controls()
            return
        self._det_detected = key == target
        self._det_settings.setValue(DETECTOR_SETTINGS_KEY + "_last", target)
        self.statusBar().showMessage(msg)
        self._spad_align_after_connect = True  # v1's Initialize ran calibrate + align
        self.spadDev.connect_device()
        self._sync_controls()

    def _apply_detector(self, key: str) -> bool:
        """Make ``key`` the active detector: new controller, pixel maps and 2D fit plot (False = cancelled)."""
        if key == self._det.key:
            return True
        if self.spadDev.state != spad_det.DISCONNECTED:
            return False
        if self._spad_trpl_data:
            ans = QtWidgets.QMessageBox.question(
                self, "TCSPC detector",
                f"Switching from {self._det.key} to {key} clears the {len(self._spad_trpl_data)} TRPL histogram(s) "
                "shown in this window (saved files are not touched). Continue?")
            if ans != QtWidgets.QMessageBox.Yes:
                return False
            self._spad_trpl_data.clear()
        old = self.spadDev
        try:
            old.shutdown()
        except Exception:
            pass
        wait_threads(old.findChildren(QtCore.QThread), 3.0)
        old.deleteLater()
        self._det = spad_det.profile(key)
        self.spadDev = self._det.device_controller(DEFAULT_TCSPC_HOST, int(DEFAULT_TCSPC_PORT), self)
        self.spadDev.set_integration_ms(SPAD_LIVE_INTEGRATION_MS)
        self._wire_spad_device()
        self._rebuild_spad_views()
        self._on_spad_state(self.spadDev.state)
        self.statusBar().showMessage(f"TCSPC detector: {self._det.label}")
        return True

    def _rebuild_spad_views(self) -> None:
        """Pixel maps and 2D fit plot for the active detector's geometry; per-detector state reset."""
        d = self._det
        self._spad_selected_pixel_live = d.center_pixel
        self._spad_selected_hist_pixels = {d.center_pixel}
        self._spad_excluded_fit_pixels = set()
        self._spad_map_rates = np.zeros(d.num_pixels, dtype=np.float64)
        self._spad_fit_x_um, self._spad_fit_y_um = d.pixel_x_um.copy(), d.pixel_y_um.copy()
        self._spad_last_count_snapshot = None
        self._spad_latest_hist_snapshot = None
        self._hist_t0_cache.clear()
        self._hist_fit_result = None
        self._fit_job = 0  # a fit of the old detector still running is dropped (_on_fit_done)
        self._fit_ctx = None
        swaps = (("spadLiveMap", d.PixelMapPlot(f"{d.key} Count-Rate Map"), self._spad_live_layout),
                 ("spadHistMap", d.PixelMapPlot("Count-Rate Map"), self._hist_grid),
                 ("histFitPlot", d.Gauss2DFitPlot(), self._hist_fit_layout))
        for name, new, layout in swaps:
            old = getattr(self, name)
            layout.replaceWidget(old, new)
            old.hide()
            old.deleteLater()
            setattr(self, name, new)
        self.spad_live_box.setTitle(f"{d.key} Count Rate Map")
        self.lbl_spad_live_info.setText(f"Pixel {d.center_pixel} selected")
        self._init_spad_live_map_plot()
        self._init_spad_hist_map_plot()
        if self._dark_ref is not None and self._dark_ref.num_pixels != d.num_pixels:
            self._dark_ref = None  # taken with the other detector
        # spectral views: the other detector's line layout and its own wavelength calibration
        self._spec_geom = g = spad_spec.geometry(d.key)
        self._spec_lines = {g.center_line}
        self._spec_active_line = g.center_line
        self.spectralLive.set_geometry(g)
        self.spectralHist.set_geometry(g)
        self._set_wl_calibration(None, "", persist=False)
        self._restore_wl_calibration()
        self._apply_spad_view()
        self._update_dark_controls()
        self._update_histogram_tab()
        self._update_lifetime_tab()

    def _on_spad_state(self, state: str) -> None:
        connected = state in (spad_det.IDLE, spad_det.LIVE, spad_det.BUSY)
        self._set_dot(self.tcspcDot, connected)
        if state == spad_det.DISCONNECTED:
            self._update_tcspc_status(False, "disconnected")
        elif state == spad_det.CONNECTING:
            self._update_tcspc_status(False, "connecting...")
        elif state == spad_det.BUSY:
            self.tcspcStatusLbl.setText("busy (TRPL / command)")
        self._sync_controls()

    def _on_spad_connected(self, banner: str, tdc: str) -> None:
        self._update_tcspc_status(True, f"connected | {tdc}"[:90])
        if self._spad_align_after_connect:
            self._spad_align_after_connect = False
            self._spad_calibrate_align(show_popup=False)
        elif self._spad_live_enabled:
            self.spadDev.start_live()

    def _spad_calibrate_align(self, *, show_popup: bool = False) -> bool:
        self._spad_popup_pending = bool(show_popup)

        def run(client):
            msgs = client.calibrate_and_align(align_ms=TRPL_ALIGN_MS, calibrate_if_invalid=True)
            return " | ".join(msgs) if msgs else f"{self._det.key} calibrate+align done."

        return self.spadDev.run_command("calibrate", run)

    def on_spad_calibrate_align(self) -> None:
        self._spad_calibrate_align(show_popup=True)

    def _on_spad_command_done(self, name: str, ok: bool, msg: str) -> None:
        popup, self._spad_popup_pending = self._spad_popup_pending, False
        if not ok:
            msg = f"SPAD calibrate+align failed: {msg}" if name == "calibrate" else msg
        self._update_tcspc_status(True, msg[:90])
        self.statusBar().showMessage(msg)
        if popup:
            box = QtWidgets.QMessageBox.information if ok else QtWidgets.QMessageBox.warning
            box(self, self._det.key, msg)
        if self._spad_live_enabled:
            # DeviceController resumes live only if it was live before the command
            QtCore.QTimer.singleShot(0, self._resume_spad_live)

    def _resume_spad_live(self, tries: int = 50) -> None:
        """Start live counts once the SPAD controller is idle (it finishes a job just after reporting it)."""
        if not self._spad_live_enabled:
            return
        sweep_running = self._sweep_thread is not None and self._sweep_thread.isRunning()
        if sweep_running or getattr(self, "_sweep_pending", False):
            return
        state = self.spadDev.state
        if state == spad_det.IDLE:
            self.spadDev.start_live()
        elif state == spad_det.BUSY and tries > 0:
            QtCore.QTimer.singleShot(100, lambda: self._resume_spad_live(tries - 1))

    def _on_spad_count_frame(self, snap) -> None:
        if not is_count_snapshot(snap):
            return
        self._spad_last_count_snapshot = snap
        now = time.monotonic()
        self._spad_count_times = [t for t in self._spad_count_times if now - t < 5.0] + [now]
        rate = ""
        if len(self._spad_count_times) >= 2:
            span = self._spad_count_times[-1] - self._spad_count_times[0]
            if span > 0:
                rate = f" | {(len(self._spad_count_times) - 1) / span:.1f} frames/s"
        self._update_spad_count_maps(snap)
        self._update_tcspc_status(True, f"connected | live | total {snap.total_rate_mcps:.3f} Mcps{rate}")

    def on_toggle_live_counts(self) -> None:
        self._spad_live_enabled = not self._spad_live_enabled
        self.stopCountsBtn.setText("Pause Live Counts" if self._spad_live_enabled else "Resume Live Counts")
        if self._spad_live_enabled:
            self._resume_spad_live()
            self.statusBar().showMessage("Live SPAD counts resumed")
        else:
            self.spadDev.stop_live()
            self.statusBar().showMessage("Live SPAD counts paused")

    def on_take_histogram(self) -> None:
        if self.spadDev.trpl_running or not self._spad_ok():
            return
        acq_s = float(self.tcspcAcqSpin.value())
        bin_ps = int(self.tcspcBinSpin.value())
        if self.spadDev.start_trpl(int(round(acq_s * 1000.0)), bin_ps):
            self.statusBar().showMessage(f"TRPL histogram acquiring ({acq_s:.3g}s, {bin_ps} ps bins)...")
        self._sync_controls()

    def _on_spad_trpl_done(self, snapshot, err: str) -> None:
        if self._dark_pending:
            self._dark_pending = False
            self._on_dark_done(snapshot, err)
            QtCore.QTimer.singleShot(0, self._sync_controls)
            return
        if err:
            if is_trpl_snapshot(snapshot):
                self._spad_latest_hist_snapshot = snapshot
                self._update_histogram_tab()
            self._on_histogram_error(err)
        elif is_trpl_snapshot(snapshot) and getattr(snapshot, "cancelled", False):
            self._spad_latest_hist_snapshot = snapshot
            self._update_histogram_tab()
            self.statusBar().showMessage("Histogram cancelled")
        else:
            self._on_histogram_done(snapshot)
        QtCore.QTimer.singleShot(0, self._sync_controls)

    def _on_histogram_done(self, snapshot_obj: object) -> None:
        snapshot = snapshot_obj if is_trpl_snapshot(snapshot_obj) else None
        if snapshot is None:
            return
        self._spad_latest_hist_snapshot = snapshot
        self._store_trpl_snapshot(power_key=0.0, voltage_key=0.0, snapshot=snapshot, filepath=None)
        self._update_histogram_tab()
        self._update_lifetime_tab()
        self.statusBar().showMessage(
            f"Histogram done: bins={snapshot.counts_matrix.shape[0]}, events={snapshot.total_counts}"
        )

    def _on_histogram_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Histogram error: {msg}")

    def _store_trpl_snapshot(self, power_key: float, voltage_key: float, snapshot,
                             filepath: Optional[str], a_key: Optional[float] = None) -> None:
        key = (float(power_key), float(voltage_key), a_key)
        self._spad_trpl_data[key] = {
            "time_ns": snapshot.time_axis_ps.astype(np.float64) / 1000.0,
            "counts": snapshot.counts_matrix.copy(),
            "filepath": filepath,
            "measurement_ms": float(snapshot.measurement_ms),  # scales the dark reference (spectral Lifetime)
            "bin_width_ps": float(snapshot.bin_width_ps),
        }

    def _normalize_lifetime_columns(self, data: np.ndarray, *, log_mode: bool) -> np.ndarray:
        out = np.full_like(data, np.nan, dtype=np.float64)
        for j in range(data.shape[1]):
            col = np.asarray(data[:, j], dtype=np.float64)
            cmax = float(np.nanmax(col)) if np.any(np.isfinite(col)) else 0.0
            if cmax <= 0:
                out[:, j] = np.nan
                continue
            ncol = col / cmax
            if log_mode:
                ncol = np.clip(ncol, 1e-3, 1.0)
            else:
                ncol = np.clip(ncol, 0.0, 1.0)
            out[:, j] = ncol
        return out

    T0_LIVE_REFRESH_S = 10.0  # live view: full time-zero alignment at most this often (else the last shifts)

    def _live_t0_alignment(self, counts: np.ndarray, t_ns: np.ndarray):
        """While a histogram is still acquiring: the last alignment's per-pixel shifts applied to the new counts
        (the shifts barely change within an acquisition); None = time for a full alignment."""
        hit = self._hist_t0_cache.get("live")
        if hit is None or hit["n_pix"] != counts.shape[1] or t_ns.size < 2:
            return None
        if abs(float(t_ns[1] - t_ns[0]) - hit["dt"]) > 1e-9:
            return None
        if time.monotonic() - hit["t"] > max(self.T0_LIVE_REFRESH_S, 10.0 * hit.get("cost", 0.0)):
            return None
        res = hit["res"]
        out = np.array(counts, copy=True)
        for p in np.flatnonzero(res.shift_bins):
            out[:, p] = np.roll(counts[:, p], int(res.shift_bins[p]))
        return dataclasses.replace(res, counts=out)

    def _t0_alignment(self, counts: np.ndarray, time_ns: np.ndarray, holder: dict, key: str):
        """``align_first_peaks`` of ``counts``, cached in ``holder[key]`` with the array it belongs to."""
        hit = holder.get(key)
        if hit is not None and hit[0] is counts:
            return hit[1]
        self._t0_started = time.monotonic()
        try:
            res = align_first_peaks(counts, time_ns)
        except Exception as exc:
            res = exc
        holder[key] = (counts, res)
        return res

    @staticmethod
    def _set_spin_silently(spin, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _on_hist_ref_edited(self) -> None:
        """A reference typed (or clicked in the spectral view): it no longer follows t0 + N bins."""
        self.chk_hist_ref_auto.blockSignals(True)
        self.chk_hist_ref_auto.setChecked(False)
        self.chk_hist_ref_auto.blockSignals(False)
        self._update_histogram_tab()

    def _sync_hist_frame(self, off: Optional[float]) -> None:
        """The view switches between raw times and times after time zero: convert the times set in ns."""
        rel = off is not None
        if rel == self._hist_rel:
            return
        d = -float(off) if rel else float(self._hist_last_off or 0.0)
        spins = [self.histStartNsSpin, self.histEndNsSpin, self.histRefNsSpin]
        if not self._pre_t0_auto:
            spins += [self.spin_pre_t0_start, self.spin_pre_t0_end]
        for sp in spins:
            self._set_spin_silently(sp, sp.value() + d)
        self._hist_rel = rel
        self._lifetime_dirty = True

    def _peak_offset(self, t_raw: np.ndarray, res) -> Optional[float]:
        """Raw time of the signal peak (brightest pixel) of an aligned histogram; None if none."""
        if res is None or isinstance(res, Exception):
            return None
        return peak_display_offset(res.counts, t_raw, self._spad_excluded_fit_pixels)

    def _view_offset(self, t_raw: np.ndarray, res, remember: bool = True, peak=False) -> Optional[float]:
        """Raw time shown as t = 0 with the alignment on: the signal peak (None = raw times).  A histogram
        without a peak (alignment failed, or the start of a live one) keeps the last offset (axis stays put).
        ``peak``: the peak time if already known (False = find it)."""
        if not self.chk_t0_align.isChecked():
            off = None
        else:
            off = self._peak_offset(t_raw, res) if peak is False else peak
            if off is None:
                off = self._hist_last_off
        if off is not None and (remember or self._hist_last_off is None):
            self._hist_last_off = off  # (the Histogram tab's: converts the spins back to its raw times)
        self._sync_hist_frame(off)
        return off

    def _hist_view(self, snapshot) -> Tuple[np.ndarray, np.ndarray]:
        """(time ns, counts) shown for ``snapshot``: raw, or time-zero aligned with t = 0 at time zero."""
        align = self.chk_t0_align.isChecked()
        c = self._hist_view_cache
        if c is not None and c[0] is snapshot and c[1] == align:
            return c[2], c[3]
        t_raw = snapshot.time_axis_ps.astype(np.float64) / 1000.0
        counts, res = self._hist_aligned(snapshot, t_raw)
        off = self._view_offset(t_raw, res)
        t = t_raw if off is None else t_raw - off
        if res is not None and not isinstance(res, Exception) and off is not None:
            self.lbl_t0_align.setText(res.message + f" Time axis: t = 0 at the signal peak (brightest pixel, "
                                      f"raw {off:.2f} ns).")
        self._hist_view_cache = (snapshot, align, t, counts)
        return t, counts

    def _hist_view_counts(self, snapshot, t_ns: np.ndarray) -> np.ndarray:
        """Counts shown in the Histogram tab: raw, or time-zero aligned (view only)."""
        return self._hist_aligned(snapshot, t_ns)[0]

    def _hist_aligned(self, snapshot, t_ns: np.ndarray):
        """(counts, alignment result / exception / None when off) of ``snapshot`` (raw ``t_ns``)."""
        counts = snapshot.counts_matrix
        if not self.chk_t0_align.isChecked():
            self.lbl_t0_align.setText("")
            return counts, None
        live = not getattr(snapshot, "done", True) and not getattr(snapshot, "cancelled", False)
        res = self._live_t0_alignment(counts, t_ns) if live else None
        if res is None:
            res = self._t0_alignment(counts, t_ns, self._hist_t0_cache, "hist")
            if live and not isinstance(res, Exception):
                self._hist_t0_cache["live"] = {"res": res, "t": time.monotonic(), "n_pix": counts.shape[1],
                                               "dt": float(t_ns[1] - t_ns[0]) if t_ns.size > 1 else 0.0,
                                               "cost": time.monotonic() - getattr(self, "_t0_started", time.monotonic())}
        if isinstance(res, Exception):
            self.lbl_t0_align.setText(f"Time-zero alignment failed ({res}); raw data shown.")
            return counts, res
        self.lbl_t0_align.setText(res.message)
        return res.counts, res

    def _record_view(self, rec: dict) -> Tuple[np.ndarray, np.ndarray]:
        """(time ns, counts) of a stored TRPL record for the Lifetime tab: raw, or time-zero aligned with
        t = 0 at the record's own signal peak (brightest pixel), so all records line up."""
        t_raw = np.asarray(rec["time_ns"], dtype=np.float64)
        counts = rec["counts"]
        res, peak = None, None
        if self.chk_t0_align.isChecked():
            res = self._t0_alignment(counts, t_raw, rec, "_t0")
            if not isinstance(res, Exception):
                counts = res.counts
            key = (id(res), tuple(sorted(self._spad_excluded_fit_pixels)))
            hit = rec.get("_t0_peak")
            if hit is None or hit[0] != key or hit[1] is not res:
                hit = (key, res, self._peak_offset(t_raw, res))  # cached: redrawn for every record
                rec["_t0_peak"] = hit
            peak = hit[2]
        off = self._view_offset(t_raw, res, remember=False, peak=peak)
        return (t_raw if off is None else t_raw - off), counts

    def _record_view_counts(self, rec: dict) -> np.ndarray:
        """Counts of a stored TRPL record for the Lifetime tab: raw, or time-zero aligned."""
        return self._record_view(rec)[1]

    def _set_pre_t0_window(self, w0: float, w1: float) -> None:
        for sp, v in ((self.spin_pre_t0_start, w0), (self.spin_pre_t0_end, w1)):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)

    def _on_pre_t0_window_edited(self, _v=None) -> None:
        self._pre_t0_auto = False
        self._update_histogram_tab()

    def _on_pre_t0_auto(self) -> None:
        self._pre_t0_auto = True
        self._update_histogram_tab()

    def _pre_t0_offset(self, t_ns: np.ndarray, counts: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """Per-pixel mean count per bin over the pre-t0 window, and its description (None if no bins)."""
        if self._pre_t0_auto:
            win = pre_pulse_window(t_ns, np.asarray(counts).sum(axis=1))
            if win is not None:
                self._set_pre_t0_window(*win)
        w0, w1 = sorted((float(self.spin_pre_t0_start.value()), float(self.spin_pre_t0_end.value())))
        sel = (t_ns >= w0) & (t_ns <= w1)
        n = int(np.count_nonzero(sel))
        if n == 0:
            return None, f"no bins in the pre-t0 window {w0:.2f}-{w1:.2f} ns: offset not subtracted"
        dark = np.asarray(counts)[sel].mean(axis=0).astype(np.float64)
        how = "auto" if self._pre_t0_auto else "manual"
        text = (f"pre-t0 offset = per-pixel mean over {w0:.2f}-{w1:.2f} ns ({how}, {n} bins), "
                f"{dark.mean():.3g} counts/bin on average, max {dark.max():.3g} (pixel {int(np.argmax(dark))})")
        return dark, text

    def _dark_ref_scaled(self, snapshot) -> Tuple[Optional[np.ndarray], str]:
        """The dark reference scaled to ``snapshot`` (its requested acquisition time and bin size)."""
        ref = self._dark_ref
        if ref is None:
            return None, "no dark reference: take or load one"
        n_pix = int(snapshot.counts_matrix.shape[1])
        if ref.num_pixels != n_pix:
            return None, f"the dark reference has {ref.num_pixels} pixels, the histogram {n_pix}"
        acq_s = float(snapshot.measurement_ms) / 1000.0
        dark = ref.per_bin(acq_s, float(snapshot.bin_width_ps))
        return dark, (f"dark reference scaled to {acq_s:g} s and {snapshot.bin_width_ps:g} ps bins: "
                      f"{dark.mean():.3g} counts/bin on average, max {dark.max():.3g} (pixel {int(np.argmax(dark))})")

    def _hist_background(self, snapshot, t_ns: np.ndarray, counts: np.ndarray) -> Optional[np.ndarray]:
        """Per-pixel background subtracted before the 2D fit (combo), with the pre-t0 vs dark comparison."""
        key = self.cmb_hist_bg.currentData()
        pre, pre_text = self._pre_t0_offset(t_ns, counts)
        ref, ref_text = self._dark_ref_scaled(snapshot)
        text = {"pre_t0": pre_text, "dark_ref": ref_text}.get(key, "no per-pixel background subtracted")
        if pre is not None and ref is not None:
            cmp = self._det.an.dark_compare_text(pre, ref)
            if cmp:
                text += " | " + cmp
        self.lbl_pre_t0.setText(text)
        self.lbl_pre_t0.setToolTip(text)
        return pre if key == "pre_t0" else ref if key == "dark_ref" else None

    def _on_hist_bg_changed(self) -> None:
        pre = self.cmb_hist_bg.currentData() == "pre_t0"
        for w in (self.spin_pre_t0_start, self.spin_pre_t0_end, self.btn_pre_t0_auto):
            w.setEnabled(pre)
        self._update_histogram_tab()

    # ---- dark reference (beam blocked)
    def _update_dark_controls(self) -> None:
        """Dark choice enabled only with a dark reference of the active detector."""
        i = self.cmb_hist_bg.findData("dark_ref")
        ok = self._dark_ref is not None and self._dark_ref.num_pixels == self._det.num_pixels
        self.cmb_hist_bg.model().item(i).setEnabled(ok)
        if not ok and self.cmb_hist_bg.currentIndex() == i:
            self.cmb_hist_bg.setCurrentIndex(self.cmb_hist_bg.findData("pre_t0"))
        if self._dark_ref is None:
            text = "No dark reference (the pre-t0 offset can be used instead)."
        else:
            ref = self._dark_ref
            text = ref.describe() + (f" — {os.path.basename(ref.source)}" if ref.source else "")
        self.lbl_dark_ref.setText(text)
        self.lbl_dark_ref.setToolTip(text if self._dark_ref is None or not self._dark_ref.source
                                     else f"{text}\n{self._dark_ref.source}")
        self.btn_clear_dark.setEnabled(self._dark_ref is not None)
        self._on_hist_bg_changed()

    def _set_dark_reference(self, ref, how: str = "") -> None:
        if ref is not None and ref.num_pixels != self._det.num_pixels:
            QtWidgets.QMessageBox.warning(
                self, "Dark reference",
                f"The dark reference has {ref.num_pixels} pixels; the {self._det.key} has {self._det.num_pixels}.")
            return
        self._dark_ref = ref
        self._update_dark_controls()
        if how:
            self.statusBar().showMessage(f"Dark reference {how}: {self.lbl_dark_ref.text()}")

    def _dark_dir(self) -> str:
        return os.path.join(DATA_DIR, "dark_references")

    def on_take_dark(self) -> None:
        if self.spadDev.trpl_running or not self._spad_ok():
            return
        acq_s, bin_ps = float(self.spin_dark_acq.value()), int(self.tcspcBinSpin.value())
        ans = QtWidgets.QMessageBox.question(
            self, "Dark reference",
            "Block the excitation beam (keep the laser running, so the TCSPC timing stays synchronized), "
            f"then press OK.\n\nA {acq_s:g} s TRPL acquisition with {bin_ps} ps bins is taken as the "
            f"{self._det.key} dark reference.",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Ok)
        if ans != QtWidgets.QMessageBox.Ok:
            return
        if self.spadDev.start_trpl(int(round(acq_s * 1000.0)), bin_ps):
            self._dark_pending = True
            self.statusBar().showMessage(f"Dark reference acquiring ({acq_s:g} s, {bin_ps} ps bins)...")
        self._sync_controls()

    def _on_dark_done(self, snapshot, err: str) -> None:
        if err or not is_trpl_snapshot(snapshot) or snapshot.cancelled or snapshot.counts_matrix.shape[0] == 0:
            why = err or ("cancelled" if is_trpl_snapshot(snapshot) and snapshot.cancelled else "no data")
            self.statusBar().showMessage(f"Dark reference not taken: {why}")
            QtWidgets.QMessageBox.warning(self, "Dark reference",
                                          f"Dark reference not taken: {why}.\n\nUnblock the excitation beam.")
            return
        ref = self._det.an.DarkReference.from_counts(snapshot.counts_matrix, snapshot.bin_width_ps,
                                                     snapshot.measurement_ms, detector=self._det.key)
        path = os.path.join(self._dark_dir(), f"DARK_TRPL_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            self._det.io.save_dark_csv(path, snapshot.time_axis_ns, snapshot.counts_matrix, ref,
                                       {"device": self._det.key, "elapsed_s": round(snapshot.elapsed_s, 3),
                                        "n_events": snapshot.n_events})
            ref.source = path
            saved = f"Saved to {path}; it is copied into every sweep folder."
        except Exception as exc:
            saved = f"Could not save it ({exc}): it is used in this session only and not copied to sweep folders."
        self._set_dark_reference(ref, "taken")
        QtWidgets.QMessageBox.information(self, "Dark reference",
                                          f"{ref.describe()}.\n\n{saved}\n\nUnblock the excitation beam.")

    def on_load_dark(self) -> None:
        start = self._dark_dir() if os.path.isdir(self._dark_dir()) else DATA_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load dark reference", start,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            ref = self._det.io.load_dark_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load dark reference", f"Could not load {os.path.basename(path)}:\n{exc}")
            return
        self._set_dark_reference(ref, "loaded")

    def _copy_dark_to_sweep_folder(self) -> None:
        """The dark reference next to the sweep data (DARK_TRPL_*.csv: not a TRPL_* point for the viewers)."""
        self._sweep_dark_file = None
        ref = self._dark_ref
        if ref is None or not self._save_dir or not ref.source or not os.path.isfile(ref.source):
            return
        name = os.path.basename(ref.source)
        if not name.startswith("DARK_TRPL_"):
            name = "DARK_TRPL_" + name
        try:
            dst = os.path.join(self._save_dir, name)
            if os.path.abspath(dst) != os.path.abspath(ref.source):
                shutil.copy2(ref.source, dst)
            self._sweep_dark_file = name
        except Exception as exc:
            self.statusBar().showMessage(f"Could not copy the dark reference into the sweep folder: {exc}")

    def _fix_offset(self) -> Optional[float]:
        return 0.0 if self.chk_fix_offset.isChecked() else None

    def _update_histogram_tab(self) -> None:
        if self.tabs.currentWidget() is not self._hist_tab:
            self._hist_dirty = True  # redrawn when the tab is shown
            return
        self._hist_dirty = False
        self._update_spad_hist_markers()
        if self._spectral:
            self._fit_job = 0  # the 2D fit is not shown
            self._update_spectral_hist()
            return
        snapshot = self._spad_latest_hist_snapshot
        excluded = sorted(self._spad_excluded_fit_pixels)
        if snapshot is None or snapshot.counts_matrix.shape[0] == 0:
            self.histTracePlot.set_traces([])
            self.histTracePlot.set_title("")
            self.histTracePlot.show_message("No histogram yet. Click 'Take Histogram'.")
            self.histFitPlot.set_data(np.full(self._det.num_pixels, np.nan), None, excluded, "", message="No fit data")
            self.histRadialPlot.clear("No fit data", title="")
            self._fit_job = 0  # drop a fit still in flight
            self._draw_sigma()
            return

        t_ns, counts = self._hist_view(snapshot)
        start_idx, end_idx, ref_idx = self._hist_time_indices(snapshot)
        start_ns = float(t_ns[start_idx])
        end_ns = float(t_ns[end_idx])
        ref_ns = float(t_ns[ref_idx])

        selected = sorted(self._spad_selected_hist_pixels)
        self.histTracePlot.set_log(self.chk_hist_log.isChecked())
        self.histTracePlot.set_traces(
            [(f"Px {pix}", t_ns, counts[:, pix], trace_color(k)) for k, pix in enumerate(selected)]
        )
        self.histTracePlot.set_region(start_ns, end_ns)
        if end_ns <= start_ns:
            self.histTracePlot.plot.setXRange(start_ns, start_ns + 1.0, padding=0.01)
        self.histTracePlot.set_reference(ref_ns)
        self.histTracePlot.set_title("Histogram at selected pixel(s)")
        self.histTracePlot.plot.setLabel("left", "Counts")
        if not selected:
            self.histTracePlot.show_message("No pixel selected.")

        self._draw_sigma()
        frame_ref = counts[ref_idx, :].astype(np.float64)
        dark = self._hist_background(snapshot, t_ns, counts)
        if dark is not None:
            frame_ref = frame_ref - dark
        fit_inputs = self._spad_fit_inputs(frame_ref)
        if fit_inputs is None:
            self._fit_job = 0
            self._hist_fit_result = None
            self._draw_fit(None, {"frame_ref": frame_ref, "mask": np.ones(self._det.num_pixels, dtype=bool),
                                  "excluded": excluded, "ref_ns": ref_ns, "enough_pixels": False})
            return
        x_fit, y_fit, z_fit, mask = fit_inputs
        fix_offset = self._fix_offset()
        self._fit_ctx = {"frame_ref": frame_ref, "mask": mask, "excluded": excluded, "ref_ns": ref_ns,
                         "enough_pixels": True}
        self._fit_job = self._fit_worker.submit(
            # the detector's own fit: its pitches set the bounds and start widths
            lambda x=x_fit, y=y_fit, z=z_fit, fo=fix_offset, fit=self._det.an.fit_gaussian_2d:
            fit(x, y, z, fix_offset=fo)
        )

    # ---- sigma^2(t) of the histogram (on request)
    def _sigma_key(self, snapshot) -> tuple:
        """Everything the sigma^2(t) fit of ``snapshot`` depends on (a result is shown only while it matches)."""
        start_idx, end_idx, ref_idx = self._hist_time_indices(snapshot)
        return (id(snapshot), self.chk_t0_align.isChecked(), self._hist_bg_key(), start_idx, end_idx, ref_idx,
                tuple(sorted(self._spad_excluded_fit_pixels)), self.chk_fix_offset.isChecked(), self._det.key)

    def _hist_bg_key(self) -> tuple:
        bg = self.cmb_hist_bg.currentData()
        return ((bg, self._pre_t0_auto, self.spin_pre_t0_start.value(), self.spin_pre_t0_end.value()) if bg == "pre_t0"
                else (bg, id(self._dark_ref)) if bg == "dark_ref" else (bg,))

    def _sigma_fit_allowed(self) -> bool:
        """Only on a finished histogram: never while a scan or a TRPL acquisition (or a dark) runs."""
        snap = self._spad_latest_hist_snapshot
        sweep_running = self._sweep_thread is not None and self._sweep_thread.isRunning()
        return (snap is not None and snap.counts_matrix.shape[0] > 0 and not sweep_running
                and not self.spadDev.trpl_running and not self._dark_pending and self._sigma_worker is None
                and not self.spectralHist.busy)

    def on_fit_sigma(self) -> None:
        if not self._sigma_fit_allowed():
            return
        snapshot = self._spad_latest_hist_snapshot
        t_ns, counts = self._hist_view(snapshot)
        start_idx, end_idx, ref_idx = self._hist_time_indices(snapshot)
        if ref_idx >= end_idx:
            self.lbl_sigma.setText("The reference is at the end of the histogram range: nothing to fit.")
            return
        mask = np.ones(self._det.num_pixels, dtype=bool)
        ex = [p for p in self._spad_excluded_fit_pixels if p < mask.size]
        if ex:
            mask[ex] = False
        if int(mask.sum()) < int(self._det.an.MIN_FIT_PIXELS_2D):
            self.lbl_sigma.setText(f"Not enough enabled pixels (need {self._det.an.MIN_FIT_PIXELS_2D}).")
            return
        frames = np.asarray(counts[: end_idx + 1], dtype=np.float64)
        bg = self._hist_background(snapshot, t_ns, counts)
        if bg is not None:
            frames = frames - bg
        key = self._sigma_key(snapshot)
        self._sigma_job += 1
        kw = dict(frames=frames, x=self._spad_fit_x_um, y=self._spad_fit_y_um, mask=mask, ref_idx=ref_idx,
                  start=start_idx, end=end_idx, fix_offset=self._fix_offset())
        w = FitAllWorker(self._sigma_job, self._det.an.fit_all_2d_sequential, kw, self)
        w.progress.connect(lambda d, n: (self.prg_sigma.setMaximum(max(n, 1)), self.prg_sigma.setValue(d)))
        w.done.connect(lambda job, res, err, k=key, s=snapshot, t=t_ns: self._on_sigma_done(job, res, err, k, s, t))
        w.finished.connect(self._on_sigma_worker_finished)
        self._sigma_worker = w
        self.prg_sigma.setValue(0)
        self.lbl_sigma.setText(f"Fitting {end_idx - ref_idx + 1} frames ...")
        self._sync_controls()
        w.start()

    def _cancel_sigma(self) -> None:
        if self._sigma_worker is not None:
            self._sigma_worker.cancel()

    def _on_sigma_done(self, job: int, res, err: str, key, snapshot, t_ns) -> None:
        if job != self._sigma_job:
            return
        if err or res is None:
            self._sigma_result = None
            self.lbl_sigma.setText(f"σ²(t) fit failed: {err or 'no result'}")
            self.histSigmaPlot.clear("σ_eq²(t)", f"fit failed: {err}")
            self._sigma_line.setData([], [])
            return
        self._sigma_result = (key, snapshot, res, t_ns)
        self._draw_sigma()

    def _on_sigma_worker_finished(self) -> None:
        w, self._sigma_worker = self._sigma_worker, None
        if w is not None:
            w.deleteLater()
        self._sync_controls()

    def _sigma_series(self):
        """(t_ns, sigma^2, flagged, t_ref, result) of the current result, or None if it no longer applies."""
        r = self._sigma_result
        snap = self._spad_latest_hist_snapshot
        if r is None or snap is None or r[1] is not snap or r[0] != self._sigma_key(snap):
            return None
        _, _, res, t_ns = r
        idx = sorted(res.results)
        fits = [res.results[i] for i in idx]
        s2 = np.array([f.sigma_eq ** 2 for f in fits])
        flagged = np.array([bool(f.at_bound) or not f.quality_ok for f in fits])
        return t_ns[idx], s2, flagged, float(t_ns[res.ref_idx]), res

    @staticmethod
    def _sigma_line_fit(t_rel: np.ndarray, s2: np.ndarray, use: np.ndarray):
        """Least-squares line sigma^2 = s0 + slope * t_rel -> (slope, slope_err, s0, s0_err, n) or None."""
        x, y = t_rel[use], s2[use]
        n = int(x.size)
        if n < 3:
            return None
        xm, ym = float(x.mean()), float(y.mean())
        sxx = float(((x - xm) ** 2).sum())
        if sxx <= 0:
            return None
        slope = float(((x - xm) * (y - ym)).sum() / sxx)
        s0 = ym - slope * xm
        var = float(((y - s0 - slope * x) ** 2).sum()) / (n - 2)
        return slope, float(np.sqrt(var / sxx)), s0, float(np.sqrt(var * (1.0 / n + xm ** 2 / sxx))), n

    def _draw_sigma(self) -> None:
        if not hasattr(self, "histSigmaPlot"):
            return
        busy = self._sigma_worker is not None
        ser = self._sigma_series()
        if ser is None:
            self._sigma_line.setData([], [])
            if not busy:
                stale = self._sigma_result is not None
                msg = ("settings or histogram changed: click Fit σ²(t) again" if stale else
                       "click Fit σ²(t) after the histogram is collected")
                self.histSigmaPlot.clear("σ_eq²(t)", msg)
                self.lbl_sigma.setText("")
            return
        t, s2, flagged, t_ref, res = ser
        tr = t - t_ref
        d0, d1 = float(self.spin_sig_d0.value()), float(self.spin_sig_d1.value())
        use = np.isfinite(s2) & (tr >= d0 - 1e-9) & ((tr <= d1 + 1e-9) if d1 > 0 else True)
        if not self.chk_sig_flagged.isChecked():
            use &= ~flagged
        line = self._sigma_line_fit(tr, s2, use)
        bg = self.cmb_hist_bg.currentText().replace("Subtract ", "").lower()
        sub = (f"t_ref {t_ref:.3f} ns; {len(t)} frames, {int(flagged.sum())} flagged (hollow), "
               f"{res.failures} failed; background: {bg}")
        if line is None:
            self._sigma_line.setData([], [])
            text = "line fit needs at least 3 unflagged points in the D window"
        else:
            slope, slope_err, s0, s0_err, n = line
            k = 10.0 / 2.0  # 1 um^2/ns = 10 cm^2/s; sigma^2 = sigma0^2 + 2 D t
            xs = np.array([tr[use].min(), tr[use].max()])
            self._sigma_line.setData(xs + t_ref, s0 + slope * xs)
            text = (f"D = {slope * k:.4g} ± {slope_err * k:.2g} cm²/s, σ0² = {s0:.4g} ± {s0_err:.2g} µm² "
                    f"({n} points)")
        self.histSigmaPlot.set_series(t, s2, flagged, "σ_eq² (µm²)", f"σ_eq²(t) — {text}", subtitle=sub)
        self.lbl_sigma.setText(text)

    def _copy_sigma(self) -> None:
        ser = self._sigma_series()
        if ser is None:
            self.statusBar().showMessage("Nothing to copy: fit σ²(t) first.")
            return
        t, s2, flagged, _, _ = ser
        rows = ["time_ns\tsigma2_um2\tflagged"] + [f"{a:.6g}\t{b:.6g}\t{int(c)}" for a, b, c in zip(t, s2, flagged)]
        QtWidgets.QApplication.clipboard().setText("\n".join(rows))
        self.statusBar().showMessage(f"Copied σ²(t): {len(t)} rows (time_ns, sigma2_um2, flagged)")

    def _on_fit_done(self, job_id: int, fit, err: str) -> None:
        if job_id != self._fit_job or self._fit_ctx is None:
            return  # superseded by a newer snapshot / setting
        self._hist_fit_result = fit
        self._draw_fit(fit, self._fit_ctx)

    def _draw_fit(self, fit, ctx: dict) -> None:
        """2D fit map and radial plot (the drawing part of v1's _update_histogram_tab)."""
        frame_ref, mask, excluded, ref_ns = ctx["frame_ref"], ctx["mask"], ctx["excluded"], ctx["ref_ns"]
        fit_inputs = True if ctx.get("enough_pixels", True) else None
        if fit is None:
            self.histFitPlot.set_data(
                frame_ref, None, excluded, f"2D Gaussian fit @ ref {ref_ns:.3f} ns (unavailable)",
                message="Fit unavailable" if fit_inputs is not None else "Fewer than 6 enabled pixels",
            )
            self.histRadialPlot.clear("Fit unavailable", title="Gaussian radial fit")
            return

        popt = fit.popt
        note = "" if fit.quality_ok else f"low signal (amplitude {fit.snr:.1f} σ)"
        sub = f"x0={popt[1]:.1f} um, y0={popt[2]:.1f} um, sigma_eq={fit.sigma_eq:.1f} um" + (f" — {note}" if note else "")
        self.histFitPlot.set_data(frame_ref, fit, excluded, f"2D Gaussian fit @ ref {ref_ns:.3f} ns", sub)

        x_used = self._spad_fit_x_um[mask]
        y_used = self._spad_fit_y_um[mask]
        z_used = frame_ref[mask]
        r = np.sqrt((x_used - popt[1]) ** 2 + (y_used - popt[2]) ** 2)
        idx_sort = np.argsort(r)
        r_sorted = r[idx_sort]
        z_sorted = z_used[idx_sort]
        r_sym = np.concatenate([-r_sorted[::-1], r_sorted])
        z_sym = np.concatenate([z_sorted[::-1], z_sorted])
        rmax = float(np.max(r_sorted)) if r_sorted.size else 1.0
        rline = np.linspace(-rmax, rmax, 400)
        sigma_safe = max(float(fit.sigma_eq), 1e-6)
        fline = popt[0] * np.exp(-(rline**2) / (2.0 * sigma_safe**2)) + popt[5]
        self.histRadialPlot.set_data(
            r_sym, z_sym, rline, fline, "Gaussian radial fit", f"σ = {fit.sigma_eq:.1f} µm", "Counts"
        )

    def _lifetime_columns(self, keys, pixel: int, t_start: float, t_end: float):
        """Decay columns of ``keys`` (TRPL record keys) in [t_start, t_end] on a common time axis.

        Records can have different bin counts (the stream grows the histogram to the last
        event), so shorter columns are padded with NaN instead of failing in column_stack.
        """
        cols, used, time_axis = [], [], None
        for x, key in keys:
            rec = self._spad_trpl_data.get(key)
            if rec is None:
                continue
            t = self._record_view(rec)[0]
            y = self._lifetime_trace(rec, pixel)
            mask = (t >= t_start) & (t <= t_end)
            if not np.any(mask):
                continue
            if time_axis is None or t[mask].size > time_axis.size:
                time_axis = t[mask]
            cols.append(y[mask])
            used.append(float(x))
        if not cols:
            return None, None, None
        n = time_axis.size
        data = np.full((n, len(cols)), np.nan, dtype=np.float64)
        for j, c in enumerate(cols):
            data[: min(n, c.size), j] = c[:n]
        return data, used, time_axis

    def _lifetime_trace(self, rec: dict, pixel: int) -> np.ndarray:
        """Decay of one TRPL record: the pixel (2D) or the active line's center-line trace (spectral)."""
        counts = self._record_view_counts(rec)
        if not self._spectral:
            return np.asarray(counts[:, pixel], dtype=np.float64)
        g, ex, line = self._spec_geom, sorted(self._spad_excluded_fit_pixels), int(self._spec_active_line)
        y = g.spectrum_matrix(counts, ex)[:, line]
        ref = self._dark_ref
        ms, bin_ps = rec.get("measurement_ms"), rec.get("bin_width_ps")
        if (self.spectralHist.dark_sub_requested() and ref is not None and ms and bin_ps
                and ref.num_pixels == counts.shape[1]):
            dark = ref.per_bin(float(ms) / 1000.0, float(bin_ps))
            y = y - g.spectrum_matrix(dark[None, :], ex)[0][line]
        return y

    def _lifetime_what(self) -> str:
        if not self._spectral:
            return f"Px {int(self._spad_selected_pixel_live)}"
        return self._spec_geom.line_label(self._spec_active_line, self._wl_calib)

    def _update_lifetime_tab(self) -> None:
        if self.tabs.currentWidget() is not self._lifetime_tab:
            self._lifetime_dirty = True  # redrawn when the tab is shown
            return
        self._lifetime_dirty = False
        if not self._spad_trpl_data:
            self.lifetimePowerPlot.clear_message("No TRPL sweep data yet.")
            self.lifetimeVoltagePlot.clear_message("No TRPL sweep data yet.")
            return

        pixel = int(self._spad_selected_pixel_live)
        self._record_view(next(iter(self._spad_trpl_data.values())))  # Start / End in the current coordinates
        t_start = float(self.histStartNsSpin.value())
        t_end = float(self.histEndNsSpin.value())
        if t_end < t_start:
            t_start, t_end = t_end, t_start
        log_mode = self.lifetimeScaleCombo.currentText().strip().lower().startswith("log")
        # grid sweep: records are per gate-A value; follow the Select A slider
        a_sel = self._current_a_key() if self._grid_active else None

        # Lifetime vs power (selected voltage)
        selected_v = self._normalize_voltage_key(self._current_voltage_key())
        v_sel = float(selected_v if selected_v is not None else 0.0)
        pkeys = sorted({k[0] for k in self._spad_trpl_data.keys() if k[2] == a_sel})
        data, p_used, time_axis = self._lifetime_columns([(p, (p, v_sel, a_sel)) for p in pkeys], pixel, t_start, t_end)
        if data is not None:
            ndata = self._normalize_lifetime_columns(data, log_mode=log_mode)
            pmin = float(min(p_used))
            pmax = float(max(p_used))
            if pmax <= pmin:
                pmax = pmin + 1e-9
            xlabel = "Stage A angle (°)" if self._angle_list_mode else f"Power ({self._calib_unit})"
            self.lifetimePowerPlot.set_image(
                ndata,
                (pmin, pmax, float(time_axis[0]), float(time_axis[-1])),
                log_mode,
                f"Lifetime vs {xlabel.split(' (')[0]} @ V={v_sel:.4g} ({self._lifetime_what()})",
                xlabel,
                "Time (ns)",
            )
        else:
            self.lifetimePowerPlot.clear_message("No data for selected voltage.")

        # Lifetime vs voltage (selected power)
        selected_p = self._current_power_key()
        p_sel = float(selected_p if selected_p is not None else 0.0)
        vkeys = sorted({k[1] for k in self._spad_trpl_data.keys() if k[2] == a_sel})
        data, v_used, time_axis = self._lifetime_columns([(v, (p_sel, v, a_sel)) for v in vkeys], pixel, t_start, t_end)
        if data is not None:
            ndata = self._normalize_lifetime_columns(data, log_mode=log_mode)
            vmin = float(min(v_used))
            vmax = float(max(v_used))
            if vmax <= vmin:
                vmax = vmin + 1e-9
            self.lifetimeVoltagePlot.set_image(
                ndata,
                (vmin, vmax, float(time_axis[0]), float(time_axis[-1])),
                log_mode,
                f"Lifetime vs Voltage @ P={p_sel:.4g} ({self._lifetime_what()})",
                "Gate B voltage (V)" if self._grid_active else "Voltage (V)",
                "Time (ns)",
            )
        else:
            self.lifetimeVoltagePlot.clear_message("No data for selected power.")

    # -----------------
    # Image handling
    # -----------------
    def _prepare_display_image(self, raw):
        # Vertical flip only; horizontal (dispersive) axis stays in native
        # order so map columns align with the wavelength axis.
        disp = np.flipud(np.asarray(raw))
        h, w = disp.shape
        top = max(0, min(self._crop_top, h - 1))
        bottom = max(0, min(self._crop_bottom, h - 1))
        left = max(0, min(self._crop_left, w - 1))
        right = max(0, min(self._crop_right, w - 1))
        y1 = max(top, 0)
        y2 = max(y1 + 1, h - bottom)
        x1 = max(left, 0)
        x2 = max(x1 + 1, w - right)
        return disp[y1:y2, x1:x2]

    def _display_image_from_entry(self, info: Optional[dict]) -> Optional[np.ndarray]:
        if not info:
            return None
        raw = info.get("raw")
        if raw is not None:
            try:
                return self._prepare_display_image(raw)
            except Exception:
                return None
        img = info.get("image")
        if img is None:
            return None
        try:
            arr = np.asarray(img)
        except Exception:
            return None
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _linecut_from_display(self, img, row: int) -> Optional[np.ndarray]:
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim != 2:
            return None
        h, _ = a.shape
        width = 1
        if self.liveView is not None:
            width = max(1, int(self.liveView.linecut_width()))
        half = width // 2
        r1 = max(0, int(row) - half)
        r2 = min(h, int(row) + half + (1 if width % 2 else 0))
        sl = a[r1:r2, :]
        return sl.sum(axis=0)

    def _refresh_linecut_map(self):
        self._refresh_linecut_maps()

    def _refresh_linecut_maps(self) -> None:
        self._refresh_linecut_power_map()
        self._refresh_linecut_voltage_map()

    def _linecut_map_yaxis(self, width: int):
        x_axis = self.liveView.get_linecut_axis()
        if x_axis is not None and len(x_axis) == width:
            y0 = float(x_axis[0])
            y1 = float(x_axis[-1])
            ylabel = "Wavelength (nm)" if self.liveView.has_wavelength_axis() else "Pixel"
        else:
            y0 = 0.0
            y1 = float(width)
            ylabel = "Pixel"
        return y0, y1, ylabel

    def _refresh_grid_power_map(self) -> None:
        # Linecut vs power at the currently selected gate-A and gate-B values.
        if not self._power_view_keys or not self._grid_images:
            self.powerMap.show_placeholder()
            return

        a_key = self._current_a_key()
        if a_key is None and self._a_keys:
            a_key = self._a_keys[0]
        b_key = self._normalize_voltage_key(self._current_voltage_key())
        if b_key is None and self._b_keys:
            b_key = sorted(self._b_keys)[0]

        def _img_at(pkey):
            if a_key is None or b_key is None:
                return None
            return self._display_image_from_entry(
                self._grid_images.get(pkey, {}).get(a_key, {}).get(b_key)
            )

        first_image = None
        for pkey in self._power_view_keys:
            first_image = _img_at(pkey)
            if first_image is not None:
                break
        if first_image is None:
            self.powerMap.show_placeholder()
            return

        _, width = first_image.shape
        data = np.full((width, len(self._power_view_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, pkey in enumerate(self._power_view_keys):
            img = _img_at(pkey)
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        powers = self._power_view_keys
        pmin = float(powers[0]) if powers else 0.0
        pmax = float(powers[-1]) if powers else 1.0
        if pmin == pmax:
            pmax = pmin + 1.0

        y0, y1, ylabel = self._linecut_map_yaxis(width)

        x_label = "Stage A angle (°)" if self._angle_list_mode else (f"Power ({self._calib_unit})" if self._calib_unit else "Power")
        parts = []
        a_label = self._voltage_label_for_key(a_key) if a_key is not None else "--"
        if a_label and a_label != "--":
            parts.append(f"A={a_label}")
        b_label = self._voltage_label_for_key(b_key) if b_key is not None else "--"
        if b_label and b_label != "--":
            parts.append(f"B={b_label}")
        title = f"Linecut vs {x_label.split(' (')[0]}"
        if parts:
            title = f"{title} at {', '.join(parts)}"

        self.powerMap.set_image(data, (pmin, pmax, y0, y1), title, x_label, ylabel)
        self._apply_map_clim(data, self.powerMap, self.mapCminEdit, self.mapCmaxEdit)

    def _refresh_linecut_power_map(self) -> None:
        if self.liveView is None:
            return
        if self._grid_active:
            self._refresh_grid_power_map()
            return
        if not self._power_view_keys or not self._pv_images:
            self.powerMap.show_placeholder()
            return

        selected_voltage_key = self._normalize_voltage_key(self._current_voltage_key())
        voltage_key = selected_voltage_key

        def _find_first_image(vkey: Optional[float]):
            if vkey is None:
                return None
            for pkey in self._power_view_keys:
                entry = self._pv_images.get(pkey, {})
                img = self._display_image_from_entry(entry.get(vkey) if entry else None)
                if img is not None:
                    return img
            return None

        first_image = _find_first_image(voltage_key)
        if first_image is None:
            for vkey in self._voltage_keys:
                first_image = _find_first_image(vkey)
                if first_image is not None:
                    voltage_key = vkey
                    break

        if first_image is None:
            self.powerMap.show_placeholder()
            return

        _, width = first_image.shape
        data = np.full((width, len(self._power_view_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, power_key in enumerate(self._power_view_keys):
            info = self._pv_images.get(power_key, {}).get(voltage_key)
            if not info:
                continue
            img = self._display_image_from_entry(info)
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        powers = self._power_view_keys
        pmin = float(powers[0]) if powers else 0.0
        pmax = float(powers[-1]) if powers else 1.0
        if pmin == pmax:
            pmax = pmin + 1.0

        x_axis = self.liveView.get_linecut_axis()
        if x_axis is not None and len(x_axis) == width:
            y0 = float(x_axis[0])
            y1 = float(x_axis[-1])
            ylabel = "Wavelength (nm)" if self.liveView.has_wavelength_axis() else "Pixel"
        else:
            y0 = 0.0
            y1 = float(width)
            ylabel = "Pixel"

        x_label = "Stage A angle (°)" if self._angle_list_mode else (f"Power ({self._calib_unit})" if self._calib_unit else "Power")
        title = f"Linecut vs {x_label.split(' (')[0]}"
        vlabel = self._voltage_label_for_key(voltage_key)
        if vlabel and vlabel != "--":
            title = f"{title} at {vlabel}"

        self.powerMap.set_image(data, (pmin, pmax, y0, y1), title, x_label, ylabel)
        self._apply_map_clim(data, self.powerMap, self.mapCminEdit, self.mapCmaxEdit)

    def _refresh_grid_voltage_map(self) -> None:
        # Linecut vs gate-B voltage, with gate A held at the current value.
        if not self._b_keys or not self._grid_images:
            self.voltageMap.show_placeholder()
            return

        power_key = self._current_power_key()
        if power_key is None and self._power_view_keys:
            power_key = self._power_view_keys[0]
        if power_key is None:
            power_key = 0.0

        a_key = self._current_a_key()
        if a_key is None and self._a_keys:
            a_key = self._a_keys[0]

        b_keys = sorted(self._b_keys)
        b_map = self._grid_images.get(power_key, {}).get(a_key, {}) if a_key is not None else {}

        first_image = None
        for bkey in b_keys:
            first_image = self._display_image_from_entry(b_map.get(bkey))
            if first_image is not None:
                break
        if first_image is None:
            self.voltageMap.show_placeholder()
            return

        _, width = first_image.shape
        data = np.full((width, len(b_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, bkey in enumerate(b_keys):
            img = self._display_image_from_entry(b_map.get(bkey))
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        bmin = float(b_keys[0])
        bmax = float(b_keys[-1])
        if bmin == bmax:
            bmax = bmin + 1.0

        y0, y1, ylabel = self._linecut_map_yaxis(width)

        title = "Linecut vs Gate B"
        parts = []
        a_label = self._voltage_label_for_key(a_key) if a_key is not None else "--"
        if a_label and a_label != "--":
            parts.append(f"A={a_label}")
        power_label = self._format_power_key(float(power_key))
        if power_label and power_label != "--" and self.sweepPowerChk.isChecked():
            parts.append(power_label)
        if parts:
            title = f"Linecut vs Gate B at {', '.join(parts)}"

        self.voltageMap.set_image(data, (bmin, bmax, y0, y1), title, "Gate B voltage (V)", ylabel)
        self._apply_map_clim(data, self.voltageMap, self.vmapCminEdit, self.vmapCmaxEdit)

    def _refresh_linecut_voltage_map(self) -> None:
        if self.liveView is None:
            return
        if self._grid_active:
            self._refresh_grid_voltage_map()
            return
        if not self._voltage_keys or not self._pv_images:
            self.voltageMap.show_placeholder()
            return

        power_key = self._current_power_key()
        if power_key is None and self._power_view_keys:
            power_key = self._power_view_keys[0]
        if power_key is None:
            self.voltageMap.show_placeholder()
            return

        entry = self._pv_images.get(power_key, {})
        first_image = None
        for vkey in self._voltage_keys:
            first_image = self._display_image_from_entry(entry.get(vkey) if entry else None)
            if first_image is not None:
                break
        if first_image is None:
            for pkey in self._power_view_keys:
                candidate = self._pv_images.get(pkey, {})
                for vkey in self._voltage_keys:
                    first_image = self._display_image_from_entry(candidate.get(vkey) if candidate else None)
                    if first_image is not None:
                        power_key = pkey
                        entry = candidate
                        break
                if first_image is not None:
                    break
        if first_image is None:
            self.voltageMap.show_placeholder()
            return

        _, width = first_image.shape
        data = np.full((width, len(self._voltage_keys)), np.nan, dtype=float)
        linecut_row = self.liveView.linecut_row()
        if linecut_row is None:
            return

        for col, vkey in enumerate(self._voltage_keys):
            info = entry.get(vkey)
            if not info:
                continue
            img = self._display_image_from_entry(info)
            if img is None or img.shape[1] != width:
                continue
            line = self._linecut_from_display(img, linecut_row)
            if line is None or line.size != width:
                continue
            data[:, col] = line

        vmin = float(self._voltage_keys[0]) if self._voltage_keys else 0.0
        vmax = float(self._voltage_keys[-1]) if self._voltage_keys else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        x_axis = self.liveView.get_linecut_axis()
        if x_axis is not None and len(x_axis) == width:
            y0 = float(x_axis[0])
            y1 = float(x_axis[-1])
            ylabel = "Wavelength (nm)" if self.liveView.has_wavelength_axis() else "Pixel"
        else:
            y0 = 0.0
            y1 = float(width)
            ylabel = "Pixel"

        power_label = self._format_power_key(float(power_key))
        title = "Linecut vs Voltage"
        if power_label and power_label != "--" and self.sweepPowerChk.isChecked():
            title = f"Linecut vs Voltage at {power_label}"

        self.voltageMap.set_image(data, (vmin, vmax, y0, y1), title, "Voltage (V)", ylabel)
        self._apply_map_clim(data, self.voltageMap, self.vmapCminEdit, self.vmapCmaxEdit)

    # -----------------
    # Session metadata
    # -----------------
    def _wavelength_axis_text(self) -> str:
        wl = self.liveView.get_wavelength_axis() if self.liveView is not None else None
        if wl is None:
            return "none (spectra saved against pixel columns)"
        return (f"{wl[0]:.4f} to {wl[-1]:.4f} nm over {wl.size} columns (first column of every .asc, "
                f"'Columns: wavelength_nm, ...')")

    def _write_session_metadata(self) -> None:
        if self._session_saved or not self._save_dir:
            return
        path = os.path.join(self._save_dir, "session.md")
        meta = self._build_metadata_base()
        lines = [
            "# PL + TRPL power/voltage sweep session",
            "",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Save folder: {self._save_dir}",
            "",
            "## Hardware",
            f"- Stage A serial: {meta.get('stage_serial_a', '')}",
            f"- Stage B serial: {meta.get('stage_serial_b', '')}",
            f"- Exposure (ms): {meta.get('exposure_ms', '')}",
            f"- Accum N: {meta.get('accum_n', '')}",
            f"- Readout rate: {meta.get('readout_rate', '')}",
            f"- Preamp gain: {meta.get('preamp_gain', '')}",
            f"- Output amp: {meta.get('output_amp', '')}",
            f"- Grating: {meta.get('grating', '')}",
            f"- Center (nm): {meta.get('center_wl_nm', '')}",
            f"- Wavelength axis: {self._wavelength_axis_text()}",
            f"- Slit ({meta.get('slit_id', '')}) um: {meta.get('slit_um', '')}",
            "",
            "## Gates",
            f"- Front resource: {meta.get('front_resource', '')}",
            f"- Back resource: {meta.get('back_resource', '')}",
            f"- Front Vset (V): {meta.get('front_vset', '')}",
            f"- Back Vset (V): {meta.get('back_vset', '')}",
            f"- Front Icomp (nA): {meta.get('front_icomp_nA', '')}",
            f"- Back Icomp (nA): {meta.get('back_icomp_nA', '')}",
            f"- Front output: {meta.get('front_output_on', '')}",
            f"- Back output: {meta.get('back_output_on', '')}",
            "",
            "## Sweep",
            f"- Sweep power: {meta.get('sweep_power', '')}",
            f"- Sweep voltage: {meta.get('sweep_voltage', '')}",
            f"- Gate mode: {meta.get('sweep_gate_mode', '')}",
            f"- Grid (fix A, sweep B): {meta.get('grid_sweep', '')}",
            f"- Front V0/step: {meta.get('front_v0', '')}, {meta.get('front_step', '')}",
            f"- Skip N power points: {meta.get('power_skip_n', 0)}",
            f"- Back V0/step: {meta.get('back_v0', '')}, {meta.get('back_step', '')}",
            f"- Steps: {meta.get('sweep_steps', '')}",
            f"- Grid steps A x B: {meta.get('sweep_steps_a', '')} x {meta.get('sweep_steps_b', '')}",
            f"- Measure: {meta.get('measure_mode', '')}",
            f"- dI/dP (pump 2 off / on per point): {meta.get('didp', '')}; pump 2 passes at flipper position "
            f"{meta.get('pump2_on_position', '')} (S/N {meta.get('flipper_serial', '')}), settle "
            f"{meta.get('flipper_settle_ms', '')} ms; files tagged _{PUMP2_TAGS[False]} / _{PUMP2_TAGS[True]}",
            f"- Gate ramps: {RAMP_STEP_V * 1000:g} mV microsteps, {RAMP_DWELL_S * 1000:g} ms dwell, V/I read after "
            f"every microstep: {meta.get('ramp_micro_read', '')}",
            f"- Home both gates after scan: {meta.get('home_gates_after_scan', '')}",
            "",
            "## TCSPC",
            f"- Detector: {meta.get('spad_detector', '')} ({meta.get('spad_num_pixels', '')} pixels, "
            f"{'detected from the server' if meta.get('spad_detector_detected') else 'selected manually'})",
            f"- Host: {meta.get('tcspc_host', '')}:{meta.get('tcspc_port', '')}",
            (f"- Acquisition (s): dynamic (per file, see {tdyn.LOG_NAME})" if meta.get("trpl_time_mode") == "dynamic"
             else f"- Acquisition (s): {meta.get('tcspc_acq_s', '')}"),
        ] + ([
            f"- TRPL time: dynamic; first chunk {meta['trpl_dyn_first_s']:g} s"
            + (" (= PL time)" if meta.get("trpl_dyn_first_is_pl_time") else "")
            + f", then {meta['trpl_dyn_step_s']:g} s chunks, max {meta['trpl_dyn_max_s']:g} s; stop when the brightest"
            f" pixel's peak minus background reaches {meta['trpl_dyn_target_counts']:g} counts (background: "
            f"{meta['trpl_dyn_background']}); TRPL_*.csv = sum of the chunks, chunks in {tdyn.CHUNK_DIR}/",
        ] if meta.get("trpl_time_mode") == "dynamic" else []) + [
            f"- Bin size (ps): {meta.get('tcspc_bin_ps', '')}",
            f"- Dark reference: {self._sweep_dark_file or 'none'}"
            + (f" ({self._dark_ref.describe()})" if self._sweep_dark_file and self._dark_ref is not None else ""),
            f"- Histogram start/end/ref (ns{', t0 = 0' if meta.get('hist_times_after_t0') else ''}): "
            f"{meta.get('hist_start_ns', '')}, {meta.get('hist_end_ns', '')}, {meta.get('hist_ref_ns', '')}",
            f"- Reference: {('t0 + %d bins (auto)' % T0_REF_BINS) if meta.get('hist_ref_auto') else 'manual'}",
            f"- SPAD view: {'spectral (center line = wavelength axis)' if self._spectral else '2D space'}",
            f"- Wavelength calibration: {self._sweep_wlcal_file or 'none'}"
            + (f" ({self._wl_calib.describe()})" if self._sweep_wlcal_file and self._wl_calib is not None else ""),
            "",
            "## TRPL view (display and fit only; saved data are raw)",
            f"- Time-zero alignment (first peak): {'on' if self.chk_t0_align.isChecked() else 'off'}",
            f"- Background subtracted before the 2D fit: {self.cmb_hist_bg.currentText()}",
            f"- Pre-t0 window (ns): {self.spin_pre_t0_start.value():g}, {self.spin_pre_t0_end.value():g} "
            f"({'auto' if self._pre_t0_auto else 'manual'})",
            f"- Gaussian offset: {'fixed at 0' if self.chk_fix_offset.isChecked() else 'fitted'}",
            f"- Spectral views: rebin ×{self.spectralHist.spin_rebin.value()}, dark reference subtracted: "
            f"{'yes' if self.spectralHist.dark_sub_requested() and self._dark_ref is not None else 'no'}",
            "",
            "## Calibration",
            f"- File: {meta.get('power_calib_file', '')}",
            f"- Mode: {meta.get('calib_mode', '')}",
        ] + ([f"- ND wheel range (deg): {meta['wheel_range_deg']}"] if meta.get("wheel_range_deg") else []) + [
            "",
            "## Display Crop",
            f"- Top: {meta.get('crop_top', '')}",
            f"- Bottom: {meta.get('crop_bottom', '')}",
            f"- Left: {meta.get('crop_left', '')}",
            f"- Right: {meta.get('crop_right', '')}",
            "",
            "## Linecut",
            f"- Row: {meta.get('linecut_row', '')}",
            f"- Width: {meta.get('linecut_width', '')}",
        ]
        if meta.get("sweep_power") and not self._angle_list_mode and self._power_list_active() and self._power_plan:
            n_bad = sum(not r.ok for r in self._power_plan)
            lines += [
                "",
                "## Power list (HWP+ND)",
                f"- {len(self._power_plan)} targets, tolerance {self.powerListTolSpin.value():g} %, "
                f"HWP range {self.hwpMinSpin.value():g}–{self.hwpMaxSpin.value():g}°, {n_bad} skipped; swept in list order",
                "",
                "| Target (W) | Predicted (W) | Error (%) | HWP = A (°) | ND = B (°) | Note |",
                "|---|---|---|---|---|---|",
            ] + [
                f"| {r.target_w:.6g} | {r.power_w:.6g} | {r.err * 100:+.3f} | {r.hwp_deg:.4f} | {r.nd_deg:.4f} "
                f"| {r.note or 'swept'} |"
                for r in self._power_plan
            ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._session_saved = True
            self.statusBar().showMessage("Session metadata saved")
        except Exception as exc:
            self.statusBar().showMessage(f"Metadata save failed: {exc}")

    # -----------------
    # Window close
    # -----------------
    def shutdown(self) -> None:
        """Abort, then close every device (window close: bounded waits are fine here)."""
        try:
            self.on_abort()
        except Exception:
            pass
        self.scanTimeWidget.close_mini()  # parentless always-on-top window
        # worker threads still running; _bg_threads are children of this widget and die with it
        wait_threads([self._live_thread, self._accum_thread, self._sweep_thread, self._front_ramp_thread,
                      self._back_ramp_thread, *self._stage_threads.values(), *self._bg_threads], 10.0)
        for dev in self.gateDevs.values():
            dev.shutdown()
        for dev in self.stageDevs.values():
            dev.shutdown()
        self.flipperDev.shutdown()
        self.specDev.shutdown()
        self.camDev.shutdown()
        try:
            self.spadDev.shutdown()
        except Exception:
            pass
        # spadDev.shutdown() only asks its workers to stop (a refused connect blocks ~2 s here)
        wait_threads(self.spadDev.findChildren(QtCore.QThread), 5.0)
        if self._detect_worker is not None:
            self._detect_worker.done.disconnect()
            wait_threads([self._detect_worker], 6.0)
        if self._sigma_worker is not None:
            self._sigma_worker.cancel()
            wait_threads([self._sigma_worker], 5.0)
        self.spectralHist.wait_fit(5.0)
        self._fit_worker.stop()
        wait_threads([self._fit_worker], 3.0)
