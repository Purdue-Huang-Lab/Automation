import os
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from andor.andor_wrapper import AndorSystem
from andor.gui_v2.workers import SingleAcqThread

from .config import (
    BTN_W,
    DATA_DIR,
    DEFAULT_ACQ_NUMBER,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_GRATING,
    DEFAULT_LINECUT_WIDTH,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_READOUT_RATE,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_SPEC_INDEX,
    POLL_MS,
    andor_cfg,
)
from .plots import PLMapPanel, SpectrumPanel, band_rows

LABELS = {"on": "ON", "off": "OFF"}


def _gain_value(label) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)", str(label))
    return float(m.group(1)) if m else None


class PLCompareWidget(QtWidgets.QWidget):
    """Collect two PL images (ON / OFF) with identical Andor + Shamrock settings and
    compare row-band spectra at a clicked pixel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = AndorSystem(verbose=True)

        self._acq_thread: Optional[SingleAcqThread] = None
        self._acq_which: Optional[str] = None
        self._acq_result: Optional[np.ndarray] = None
        self._acq_wl: Optional[np.ndarray] = None
        self._acq_meta: dict = {}
        self._aborting = False

        # which -> {"raw": full-frame sum, "wl": axis or None, "meta": dict, "t": str}
        self._maps: Dict[str, Optional[dict]] = {"on": None, "off": None}
        self._cursor: Optional[Tuple[int, int]] = None  # display (row, col)

        self._crop_top = int(DEFAULT_CROP_TOP)
        self._crop_bottom = int(DEFAULT_CROP_BOTTOM)
        self._crop_left = int(DEFAULT_CROP_LEFT)
        self._crop_right = int(DEFAULT_CROP_RIGHT)

        self._readout_rate_label = str(DEFAULT_READOUT_RATE)
        self._preamp_gain_label = str(DEFAULT_PREAMP_GAIN)
        self._output_amp_label = str(DEFAULT_OUTPUT_AMP)

        self._build_ui()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_status)
        self._poll_timer.start(int(POLL_MS))
        self._sync_controls()

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self._build_dashboard(vbox)
        self._build_panels(vbox)

        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)
        self.statusBar().showMessage("Idle")

        self._wire_signals()

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    def _build_dashboard(self, parent_layout):
        box = QtWidgets.QGroupBox("Dashboard")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(9)

        # Andor line
        self.camDot = self._make_dot()
        self.camStatusLbl = QtWidgets.QLabel("Disconnected")
        self.camStatusLbl.setFont(mono)

        self.exposureSpin = QtWidgets.QDoubleSpinBox()
        self.exposureSpin.setDecimals(3)
        self.exposureSpin.setRange(0.001, 1.0e7)
        self.exposureSpin.setValue(float(DEFAULT_EXPOSURE_MS))
        self.exposureSpin.setFixedWidth(90)

        self.accumSpin = QtWidgets.QSpinBox()
        self.accumSpin.setRange(1, 100000)
        self.accumSpin.setValue(int(DEFAULT_ACQ_NUMBER))
        self.accumSpin.setFixedWidth(70)

        self.readoutRateCombo = QtWidgets.QComboBox()
        self.readoutRateCombo.setEditable(False)
        self.readoutRateCombo.addItem(str(DEFAULT_READOUT_RATE))
        self.readoutRateCombo.setFixedWidth(150)

        self.preampCombo = QtWidgets.QComboBox()
        self.preampCombo.setEditable(False)
        self.preampCombo.addItem(str(DEFAULT_PREAMP_GAIN))
        self.preampCombo.setFixedWidth(70)
        self.preampInfoLbl = QtWidgets.QLabel("Gain: --")
        self.preampInfoLbl.setFont(mono)

        line1 = QtWidgets.QHBoxLayout()
        line1.addWidget(self.camDot)
        line1.addWidget(QtWidgets.QLabel("Andor:"))
        line1.addWidget(self.camStatusLbl, 1)
        line1.addStretch()
        line1.addWidget(QtWidgets.QLabel("Exp (ms):"))
        line1.addWidget(self.exposureSpin)
        line1.addWidget(QtWidgets.QLabel("Acq N:"))
        line1.addWidget(self.accumSpin)
        line1.addWidget(QtWidgets.QLabel("Rate:"))
        line1.addWidget(self.readoutRateCombo)
        line1.addWidget(QtWidgets.QLabel("Preamp:"))
        line1.addWidget(self.preampCombo)
        line1.addWidget(self.preampInfoLbl)
        grid.addLayout(line1, 0, 0)

        # Spectrograph line
        self.specDot = self._make_dot()
        self.specStatusLbl = QtWidgets.QLabel("Disconnected")
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
        self.applySpecBtn.setFixedWidth(BTN_W)

        line2 = QtWidgets.QHBoxLayout()
        line2.addWidget(self.specDot)
        line2.addWidget(QtWidgets.QLabel("Spectrograph:"))
        line2.addWidget(self.specStatusLbl, 1)
        line2.addStretch()
        line2.addWidget(QtWidgets.QLabel("Grating:"))
        line2.addWidget(self.gratingSpin)
        line2.addWidget(QtWidgets.QLabel("Center (nm):"))
        line2.addWidget(self.centerSpin)
        line2.addWidget(QtWidgets.QLabel("Slit (um):"))
        line2.addWidget(self.slitSpin)
        line2.addWidget(self.applySpecBtn)
        grid.addLayout(line2, 1, 0)

        # Session line: save folder, crop, linecut width
        self.saveNameEdit = QtWidgets.QLineEdit(self._default_save_name())
        self.saveNameEdit.setPlaceholderText("subfolder name under DATA_DIR")
        self.saveNameEdit.setMinimumWidth(220)

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
        self.cropApplyBtn.setFixedWidth(BTN_W)

        self.linecutWidthSpin = QtWidgets.QSpinBox()
        self.linecutWidthSpin.setRange(1, 999)
        self.linecutWidthSpin.setValue(int(DEFAULT_LINECUT_WIDTH))
        self.linecutWidthSpin.setFixedWidth(70)

        line3 = QtWidgets.QHBoxLayout()
        line3.addWidget(QtWidgets.QLabel("Save subfolder:"))
        line3.addWidget(self.saveNameEdit, 1)
        line3.addSpacing(12)
        line3.addWidget(QtWidgets.QLabel("Crop T/B/L/R:"))
        line3.addWidget(self.cropTopSpin)
        line3.addWidget(self.cropBottomSpin)
        line3.addWidget(self.cropLeftSpin)
        line3.addWidget(self.cropRightSpin)
        line3.addWidget(self.cropApplyBtn)
        line3.addSpacing(12)
        line3.addWidget(QtWidgets.QLabel("Linecut width (px):"))
        line3.addWidget(self.linecutWidthSpin)
        grid.addLayout(line3, 2, 0)

        # Action line
        self.initBtn = QtWidgets.QPushButton("Initialize")
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.collectOnBtn = QtWidgets.QPushButton("Collect PL1 (ON)")
        self.collectOffBtn = QtWidgets.QPushButton("Collect PL2 (OFF)")
        self.abortBtn = QtWidgets.QPushButton("Abort")
        self.saveMapsBtn = QtWidgets.QPushButton("Save maps")
        self.saveSpectraBtn = QtWidgets.QPushButton("Save spectra")
        for btn in (
            self.initBtn, self.disconnectBtn, self.collectOnBtn, self.collectOffBtn,
            self.abortBtn, self.saveMapsBtn, self.saveSpectraBtn,
        ):
            btn.setFixedWidth(BTN_W + 20)

        line4 = QtWidgets.QHBoxLayout()
        line4.addStretch()
        line4.addWidget(self.initBtn)
        line4.addWidget(self.disconnectBtn)
        line4.addSpacing(16)
        line4.addWidget(self.collectOnBtn)
        line4.addWidget(self.collectOffBtn)
        line4.addWidget(self.abortBtn)
        line4.addSpacing(16)
        line4.addWidget(self.saveMapsBtn)
        line4.addWidget(self.saveSpectraBtn)
        grid.addLayout(line4, 3, 0)

        parent_layout.addWidget(box)

    def _build_panels(self, parent_layout):
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.onPanel = PLMapPanel("PL1 — ON")
        self.offPanel = PLMapPanel("PL2 — OFF")
        self.specPanel = SpectrumPanel("Spectra at cursor")
        self._panels = {"on": self.onPanel, "off": self.offPanel}
        for i, w in enumerate((self.onPanel, self.offPanel, self.specPanel)):
            self.splitter.addWidget(w)
            self.splitter.setStretchFactor(i, 1)
        parent_layout.addWidget(self.splitter, 1)

    def _wire_signals(self):
        self.initBtn.clicked.connect(self.on_initialize)
        self.disconnectBtn.clicked.connect(self.on_disconnect)
        self.applySpecBtn.clicked.connect(self.on_apply_spec)
        self.collectOnBtn.clicked.connect(lambda: self._collect("on"))
        self.collectOffBtn.clicked.connect(lambda: self._collect("off"))
        self.abortBtn.clicked.connect(self.on_abort)
        self.saveMapsBtn.clicked.connect(self.on_save_maps)
        self.saveSpectraBtn.clicked.connect(self.on_save_spectra)
        self.cropApplyBtn.clicked.connect(self.on_apply_crop)
        self.readoutRateCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.preampCombo.currentIndexChanged.connect(self.on_preamp_change)
        self.linecutWidthSpin.valueChanged.connect(lambda _: self._refresh_cursor())
        self.onPanel.pixel_clicked.connect(self._on_pixel_clicked)
        self.offPanel.pixel_clicked.connect(self._on_pixel_clicked)

    # -----------------
    # Helpers
    # -----------------
    def _make_dot(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setFixedSize(10, 10)
        lbl.setStyleSheet("border-radius:5px; background-color:#c62828;")
        return lbl

    def _set_dot(self, lbl: QtWidgets.QLabel, ok: bool) -> None:
        color = "#2e7d32" if ok else "#c62828"
        lbl.setStyleSheet(f"border-radius:5px; background-color:{color};")

    def _default_save_name(self) -> str:
        return time.strftime("plcmp_%Y%m%d_%H%M%S")

    def _resolve_save_dir(self) -> str:
        name = self.saveNameEdit.text().strip()
        if not name:
            name = self._default_save_name()
            self.saveNameEdit.setText(name)
        base = os.path.join(DATA_DIR, name)
        os.makedirs(base, exist_ok=True)
        return base

    @staticmethod
    def _next_path(folder: str, stem: str, ext: str) -> str:
        pat = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(ext)}$", re.IGNORECASE)
        n = 0
        try:
            for f in os.listdir(folder):
                m = pat.match(f)
                if m:
                    n = max(n, int(m.group(1)))
        except Exception:
            pass
        return os.path.join(folder, f"{stem}_{n + 1:03d}{ext}")

    def _acq_running(self) -> bool:
        return self._acq_thread is not None and self._acq_thread.isRunning()

    def _sync_controls(self) -> None:
        cam_ok = self.cam.cam is not None
        spec_ok = self.cam.spec is not None
        busy = self._acq_running()
        have_map = any(m is not None for m in self._maps.values())

        self.initBtn.setEnabled(not busy and not cam_ok)
        self.disconnectBtn.setEnabled(not busy and cam_ok)
        self.collectOnBtn.setEnabled(cam_ok and not busy)
        self.collectOffBtn.setEnabled(cam_ok and not busy)
        self.abortBtn.setEnabled(busy and not self._aborting)
        self.applySpecBtn.setEnabled(spec_ok and not busy)
        for w in (self.exposureSpin, self.accumSpin, self.readoutRateCombo, self.preampCombo,
                  self.gratingSpin, self.centerSpin, self.slitSpin):
            w.setEnabled(not busy)
        self.saveMapsBtn.setEnabled(have_map and not busy)
        self.saveSpectraBtn.setEnabled(have_map and self._cursor is not None and not busy)

    # -----------------
    # Device setup (mirrors pl_power_voltage_widget)
    # -----------------
    def _apply_camera_defaults(self) -> None:
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

    def _refresh_amp_choices(self) -> None:
        if self.cam.cam is None:
            return
        current_gain = self.preampCombo.currentText().strip()
        current_rate = self.readoutRateCombo.currentText().strip()
        rates, gains = [], []
        try:
            _, rates, gains = self.cam.get_amp_mode_choices()
        except Exception:
            pass
        if gains:
            self.preampCombo.blockSignals(True)
            self.preampCombo.clear()
            self.preampCombo.addItems(gains)
            if str(DEFAULT_PREAMP_GAIN) in gains:
                self.preampCombo.setCurrentText(str(DEFAULT_PREAMP_GAIN))
            elif current_gain in gains:
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
            elif current_rate in rates:
                self.readoutRateCombo.setCurrentText(current_rate)
            else:
                self.readoutRateCombo.setCurrentIndex(0)
            self.readoutRateCombo.blockSignals(False)

    def _sync_amp_labels(self, info: Optional[dict]) -> None:
        if not info:
            return
        self._readout_rate_label = info.get("rate_label") or self._readout_rate_label
        self._preamp_gain_label = info.get("gain_label") or self._preamp_gain_label
        self._output_amp_label = info.get("amp_label") or self._output_amp_label
        for combo, key in ((self.readoutRateCombo, "rate_label"), (self.preampCombo, "gain_label")):
            label = info.get(key)
            if label:
                combo.blockSignals(True)
                combo.setCurrentText(str(label))
                combo.blockSignals(False)

    def _apply_amp_settings(self) -> None:
        if self.cam.cam is None:
            return
        req_gain = self.preampCombo.currentText().strip()
        req_rate = self.readoutRateCombo.currentText().strip() or self._readout_rate_label
        req_amp = self._output_amp_label
        want_gain = _gain_value(req_gain)

        try:
            ok = bool(self.cam.set_amp_mode_by_labels(req_amp, req_rate, req_gain, force_preamp=True))
        except Exception as exc:
            ok = False
            self.statusBar().showMessage(f"Preamp set failed: {exc}")
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

        self._sync_amp_labels(info)
        self._update_preamp_label()
        if not ok:
            self.statusBar().showMessage("Preamp gain not applied")

    def _update_preamp_label(self) -> None:
        if self.cam.cam is None:
            self.preampInfoLbl.setText("Gain: --")
            return
        label = self._preamp_gain_label or self.preampCombo.currentText().strip()
        self.preampInfoLbl.setText(f"Gain: {label}")

    def _apply_spec_settings(self) -> None:
        if self.cam.spec is None:
            return
        try:
            self.cam.spec_set_grating(int(self.gratingSpin.value()))
        except Exception:
            pass
        try:
            self.cam.set_center_wavelength_nm(float(self.centerSpin.value()))
        except Exception:
            pass
        try:
            self.cam.spec_set_slit_width_um(DEFAULT_SLIT_ID, float(self.slitSpin.value()))
        except Exception:
            pass

    def _prepare_single_mode(self) -> None:
        for fn, arg in (
            ("stop_stream", None),
            ("set_frame_api", "Snap"),
            ("set_acquisition_mode", "Single"),
            ("set_trigger_mode", "Internal"),
            ("set_exposure_ms", float(self.exposureSpin.value())),
        ):
            try:
                if arg is None:
                    getattr(self.cam, fn)()
                else:
                    getattr(self.cam, fn)(arg)
            except Exception:
                pass

    def _current_wavelength_axis(self) -> Optional[np.ndarray]:
        if self.cam.spec is None or abs(float(self.centerSpin.value())) <= 1e-9:
            return None
        try:
            wl = self.cam.get_wavelength_axis(force=False)
        except Exception:
            wl = None
        if wl is None:
            return None
        arr = np.asarray(wl, dtype=float).ravel()
        if arr.size == 0 or np.allclose(arr, 0.0):
            return None
        return arr.copy()

    def _snapshot_meta(self) -> dict:
        gr = cw = slit = None
        if self.cam.spec is not None:
            gr = self.cam.spec_get_grating()
            cw = self.cam.get_center_wavelength_nm()
            slit = self.cam.spec_get_slit_width_um(DEFAULT_SLIT_ID)
        return {
            "Exposure_ms": f"{float(self.exposureSpin.value()):g}",
            "Accumulations": str(int(self.accumSpin.value())),
            "Readout_rate": str(self._readout_rate_label),
            "Preamp_gain": str(self._preamp_gain_label),
            "Output_amp": str(self._output_amp_label),
            "Grating": str(gr if gr is not None else int(self.gratingSpin.value())),
            "Center_nm": f"{cw if cw is not None else float(self.centerSpin.value()):g}",
            "Slit_um": f"{slit if slit is not None else float(self.slitSpin.value()):g}",
        }

    # -----------------
    # Status polling
    # -----------------
    def _poll_status(self) -> None:
        cam_ok = self.cam.cam is not None
        spec_ok = self.cam.spec is not None
        self._set_dot(self.camDot, cam_ok)
        self._set_dot(self.specDot, spec_ok)
        if self._acq_running():
            # Don't touch the SDK while a frame is being acquired.
            return

        if cam_ok:
            temp = self.cam.get_temperature_c()
            temp_s = f"{temp:.1f} C" if temp is not None else "--"
            exp_ms = float(self.exposureSpin.value())
            acc = int(self.accumSpin.value())
            ro = self._readout_rate_label or "--"
            self.camStatusLbl.setText(f"T={temp_s} | Exp {exp_ms:g} ms x{acc} | RO {ro}")
        else:
            self.camStatusLbl.setText("Disconnected")
        self._update_preamp_label()

        if spec_ok:
            gr = self.cam.spec_get_grating()
            cw = self.cam.get_center_wavelength_nm()
            slit = self.cam.spec_get_slit_width_um(DEFAULT_SLIT_ID)
            gr_s = str(gr) if gr is not None else "--"
            cw_s = f"{cw:.3f}" if cw is not None else "--"
            slit_s = f"{slit:.1f}" if slit is not None else "--"
            self.specStatusLbl.setText(f"Gr {gr_s} | Center {cw_s} nm | Slit {slit_s} um")
        else:
            self.specStatusLbl.setText("Disconnected")

    # -----------------
    # Initialize / Disconnect
    # -----------------
    def on_initialize(self):
        if self._acq_running():
            return
        try:
            self.cam.connect()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Andor connect", str(exc))
            return
        try:
            self.cam.connect_spectrograph(int(DEFAULT_SPEC_INDEX))
        except Exception:
            pass

        self._apply_camera_defaults()
        self._apply_spec_settings()
        try:
            self._sync_amp_labels(self.cam.get_amp_mode_info())
        except Exception:
            pass
        self._refresh_amp_choices()
        self._apply_amp_settings()

        spec_msg = "" if self.cam.spec is not None else " (spectrograph not found)"
        self.statusBar().showMessage("Initialized" + spec_msg)
        self._poll_status()
        self._sync_controls()

    def on_disconnect(self):
        if self._acq_running():
            self.statusBar().showMessage("Abort the acquisition before disconnecting")
            return
        try:
            self.cam.disconnect_spectrograph()
        except Exception:
            pass
        try:
            self.cam.disconnect()
        except Exception:
            pass
        self.statusBar().showMessage("Disconnected")
        self._poll_status()
        self._sync_controls()

    def on_apply_spec(self):
        if self._acq_running():
            return
        self._apply_spec_settings()
        self._poll_status()
        self.statusBar().showMessage("Spectrograph settings applied (stored maps keep their own axis)")

    def on_preamp_change(self):
        if self.cam.cam is None:
            self._preamp_gain_label = self.preampCombo.currentText().strip()
            self._readout_rate_label = self.readoutRateCombo.currentText().strip()
            self._update_preamp_label()
            return
        self._apply_amp_settings()

    def on_apply_crop(self):
        self._crop_top = int(self.cropTopSpin.value())
        self._crop_bottom = int(self.cropBottomSpin.value())
        self._crop_left = int(self.cropLeftSpin.value())
        self._crop_right = int(self.cropRightSpin.value())
        self._render_maps(keep_view=False)
        self.statusBar().showMessage("Crop updated")

    # -----------------
    # Acquisition
    # -----------------
    def _collect(self, which: str) -> None:
        if self._acq_running():
            return
        if self.cam.cam is None:
            QtWidgets.QMessageBox.warning(self, "Camera", "Camera not connected.")
            return
        self._prepare_single_mode()
        self._acq_which = which
        self._acq_result = None
        self._acq_wl = self._current_wavelength_axis()
        self._acq_meta = self._snapshot_meta()
        self._aborting = False

        n = int(self.accumSpin.value())
        self._acq_thread = SingleAcqThread(self.cam, n)
        self._acq_thread.frame_ready.connect(self._on_acq_frame)
        self._acq_thread.status.connect(self.statusBar().showMessage)
        self._acq_thread.finished.connect(self._on_acq_finished)
        self._acq_thread.start()
        self._panels[which].set_info(f"{LABELS[which]} — acquiring 0/{n}...")
        self.statusBar().showMessage(f"Collecting {LABELS[which]} ({n} x {self.exposureSpin.value():g} ms)")
        self._sync_controls()

    def _on_acq_frame(self, fr: dict) -> None:
        which = self._acq_which
        img = fr.get("image")
        if which is None or img is None:
            return
        raw = np.array(img, dtype=float, copy=True)
        idx = int(fr.get("accum_idx") or 1)
        total = int(fr.get("accum_n") or 1)
        panel = self._panels[which]
        disp, x = self._display_data(raw, self._acq_wl)
        panel.set_image(disp, x, keep_view=True)
        panel.set_info(f"{LABELS[which]} — acquiring {idx}/{total}...")
        if self._cursor is not None:
            panel.set_cursor(*self._cursor, self._linecut_width())
        if idx >= total:
            self._acq_result = raw

    def _on_acq_finished(self) -> None:
        if self._acq_running():
            # `finished` can arrive a moment before isRunning() flips; check again shortly.
            QtCore.QTimer.singleShot(50, self._on_acq_finished)
            return
        if self._acq_thread is None:
            return
        which = self._acq_which
        self._acq_thread = None
        self._acq_which = None
        aborted = self._aborting
        self._aborting = False
        if which is not None and self._acq_result is not None:
            self._maps[which] = {
                "raw": self._acq_result,
                "wl": self._acq_wl,
                "meta": dict(self._acq_meta),
                "t": time.strftime("%H:%M:%S"),
            }
            self.statusBar().showMessage(f"{LABELS[which]} collected")
        elif which is not None:
            why = "aborted" if aborted else "failed"
            self.statusBar().showMessage(f"{LABELS[which]} acquisition {why}; previous map kept")
        self._acq_result = None
        self._render_maps(keep_view=True)
        self._sync_controls()

    def on_abort(self):
        if not self._acq_running():
            return
        self._aborting = True
        self._acq_thread.stop()
        self.statusBar().showMessage("Aborting after the current frame...")
        self._sync_controls()

    def shutdown(self) -> None:
        """Stop acquisition and release hardware (called from MainWindow.closeEvent)."""
        self._poll_timer.stop()
        if self._acq_running():
            self._aborting = True
            self._acq_thread.stop()
            timeout_ms = int(max(5000.0, float(self.exposureSpin.value()) * 1.5 + 3000.0))
            self._acq_thread.wait(timeout_ms)
        try:
            self.cam.disconnect_spectrograph()
        except Exception:
            pass
        try:
            self.cam.disconnect()
        except Exception:
            pass

    # -----------------
    # Display / linecuts
    # -----------------
    def _linecut_width(self) -> int:
        return max(1, int(self.linecutWidthSpin.value()))

    def _display_data(self, raw: np.ndarray, wl: Optional[np.ndarray]):
        """flipud + crop (display orientation); wavelength axis cropped to match or None."""
        disp = np.flipud(np.asarray(raw))
        h, w = disp.shape
        top = max(0, min(self._crop_top, h - 1))
        bottom = max(0, min(self._crop_bottom, h - 1))
        left = max(0, min(self._crop_left, w - 1))
        right = max(0, min(self._crop_right, w - 1))
        y2 = max(top + 1, h - bottom)
        x2 = max(left + 1, w - right)
        disp = disp[top:y2, left:x2]
        x = None
        if wl is not None:
            arr = np.asarray(wl, dtype=float).ravel()
            if arr.size == w:
                x = arr[left:x2]
        return disp, x

    def _map_display(self, which: str):
        m = self._maps.get(which)
        if m is None:
            return None, None
        return self._display_data(m["raw"], m["wl"])

    def _render_maps(self, keep_view: bool = True) -> None:
        for which, panel in self._panels.items():
            m = self._maps[which]
            if m is None:
                panel.clear(f"{LABELS[which]} — (not collected)")
                continue
            disp, x = self._map_display(which)
            panel.set_image(disp, x, keep_view=keep_view)
            meta = m["meta"]
            panel.set_info(
                f"{LABELS[which]} — {meta.get('Exposure_ms')} ms x{meta.get('Accumulations')} | "
                f"Gr {meta.get('Grating')} | {meta.get('Center_nm')} nm | {m['t']}"
            )
        self._refresh_cursor()

    def _on_pixel_clicked(self, row: int, col: int) -> None:
        self._cursor = (int(row), int(col))
        self._refresh_cursor()
        self._sync_controls()

    def _refresh_cursor(self) -> None:
        if self._cursor is None:
            return
        width = self._linecut_width()
        for which, panel in self._panels.items():
            if panel.disp_shape() is not None:
                panel.set_cursor(*self._cursor, width)
            else:
                panel.hide_cursor()
        self._refresh_spectra()

    def _compute_spectra(self) -> Optional[dict]:
        if self._cursor is None:
            return None
        row, col = self._cursor
        width = self._linecut_width()
        out = {"row": row, "col": col, "width": width, "warn": ""}
        shapes = {}
        for which in ("on", "off"):
            disp, x = self._map_display(which)
            if disp is None:
                out[which] = None
                out[f"x_{which}"] = None
                out[f"wl_{which}"] = False
                continue
            h, w = disp.shape
            r = max(0, min(h - 1, row))
            r1, r2 = band_rows(r, width, h)
            out[which] = disp[r1:r2, :].sum(axis=0)
            out[f"rows_{which}"] = (r1, r2)
            out[f"wl_{which}"] = x is not None
            out[f"x_{which}"] = x if x is not None else np.arange(w, dtype=float)
            shapes[which] = disp.shape

        out["x_diff"] = None
        out["diff"] = None
        if out["on"] is not None and out["off"] is not None:
            if shapes["on"] != shapes["off"]:
                out["warn"] = "ON/OFF image shapes differ — no difference shown"
            else:
                out["diff"] = out["on"] - out["off"]
                out["x_diff"] = out["x_on"]
                if out["wl_on"] != out["wl_off"] or not np.allclose(out["x_on"], out["x_off"]):
                    out["warn"] = "ON/OFF wavelength axes differ — difference taken pixel-by-pixel"
        return out

    def _refresh_spectra(self) -> None:
        sp = self._compute_spectra()
        if sp is None:
            self.specPanel.clear()
            return
        if sp["on"] is None and sp["off"] is None:
            self.specPanel.clear("No maps collected yet")
            return
        use_wl = sp["wl_on"] if sp["on"] is not None else sp["wl_off"]
        xlabel = "Wavelength (nm)" if use_wl else "Pixel"
        rows = sp.get("rows_on") or sp.get("rows_off")
        info = (
            f"Cursor (row, col) = {sp['row']}, {sp['col']} | width {sp['width']} px | "
            f"rows {rows[0]}-{rows[1] - 1}"
        )
        if sp["warn"]:
            info += f"  [{sp['warn']}]"
            self.statusBar().showMessage(sp["warn"])
        self.specPanel.update_spectra(
            sp["x_on"], sp["on"], sp["x_off"], sp["off"], sp["x_diff"], sp["diff"],
            xlabel=xlabel, info=info,
        )

    # -----------------
    # Saving
    # -----------------
    def _crop_text(self) -> str:
        return f"{self._crop_top},{self._crop_bottom},{self._crop_left},{self._crop_right}"

    def on_save_maps(self):
        if not any(m is not None for m in self._maps.values()):
            return
        try:
            folder = self._resolve_save_dir()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save maps", str(exc))
            return
        saved = []
        for which in ("on", "off"):
            m = self._maps[which]
            if m is None:
                continue
            path = self._next_path(folder, f"pl_{which}", ".asc")
            meta = {"Label": LABELS[which], "Collected": m["t"], **m["meta"],
                    "Display_crop_TBLR": self._crop_text()}
            wl = m["wl"] if m["wl"] is not None else np.zeros(0)  # empty -> pixel-only export
            try:
                self.cam.save_ascii(path, m["raw"], metadata=meta, wavelength_axis_nm=wl)
                saved.append(os.path.basename(path))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Save maps", f"{LABELS[which]}: {exc}")
        if saved:
            self.statusBar().showMessage(f"Saved {', '.join(saved)} -> {folder}")

    def on_save_spectra(self):
        sp = self._compute_spectra()
        if sp is None or (sp["on"] is None and sp["off"] is None):
            return
        try:
            folder = self._resolve_save_dir()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save spectra", str(exc))
            return
        n = max(len(a) for a in (sp["on"], sp["off"]) if a is not None)

        def _pad(a):
            out = np.full(n, np.nan)
            if a is not None:
                out[: len(a)] = a
            return out

        x = sp["x_on"] if sp["on"] is not None else sp["x_off"]
        cols = np.column_stack([_pad(x), _pad(sp["on"]), _pad(sp["off"]), _pad(sp["diff"])])
        use_wl = sp["wl_on"] if sp["on"] is not None else sp["wl_off"]
        lines = [
            "PL ON/OFF comparison spectra",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cursor_row_col (display, after crop): {sp['row']},{sp['col']}",
            f"Linecut_width_px: {sp['width']}",
            f"Display_crop_TBLR: {self._crop_text()}",
        ]
        for which in ("on", "off"):
            m = self._maps[which]
            if m is not None:
                meta = ", ".join(f"{k}={v}" for k, v in m["meta"].items())
                lines.append(f"{LABELS[which]} ({m['t']}): {meta}")
        if sp["warn"]:
            lines.append(f"Warning: {sp['warn']}")
        xname = "wavelength_nm" if use_wl else "pixel"
        lines.append(f"{xname},on,off,on_minus_off")
        path = self._next_path(folder, "spectra", ".csv")
        try:
            np.savetxt(path, cols, delimiter=",", fmt="%.10g", header="\n".join(lines))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save spectra", str(exc))
            return
        self.statusBar().showMessage(f"Saved {os.path.basename(path)} -> {folder}")
