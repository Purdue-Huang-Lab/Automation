"""Main SPAD93 widget: top bar, Counts / TRPL tabs and the status bar."""

import os
import time

from PyQt5 import QtCore, QtWidgets

from spad93.gui_v2.config import (
    BTN_W,
    CLOSE_TIMEOUT_S,
    LIVE_INTEGRATION_MS,
    SETTINGS_APP,
    SETTINGS_ORG,
    STATUS_W,
)
from spad93.gui_v2.counts_panel import CountsPanel
from spad93.gui_v2.device import BUSY, CONNECTING, DISCONNECTED, IDLE, LIVE, DeviceController
from spad93.gui_v2.state import MODE_2D, MODE_SPECTRAL, SharedState
from spad93.gui_v2.trpl_panel import TrplPanel


class Spad93Widget(QtWidgets.QWidget):
    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.state = SharedState(self)
        self.device = DeviceController(host, port, self)
        self._closing = False
        self._force_close = False
        self._close_t0 = 0.0
        self._build_ui()
        self._wire()
        self._restore_settings()
        self._sync_controls()

    # =============================================================== UI
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_start = QtWidgets.QPushButton("Start live")
        self.btn_stop = QtWidgets.QPushButton("Stop live")
        self.btn_si = QtWidgets.QPushButton("Reset (SI)")
        self.btn_si.setToolTip("Send SI to leave the overexposure shutdown state")
        for b in (self.btn_connect, self.btn_disconnect, self.btn_start, self.btn_stop, self.btn_si):
            b.setMinimumWidth(BTN_W - 20)
            top.addWidget(b)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Mode"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItem("2D space", MODE_2D)
        self.cmb_mode.addItem("Spectral", MODE_SPECTRAL)
        top.addWidget(self.cmb_mode)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Integration (ms)"))
        self.spin_int = QtWidgets.QSpinBox()
        self.spin_int.setRange(10, 10000)
        self.spin_int.setSingleStep(50)
        self.spin_int.setValue(LIVE_INTEGRATION_MS)
        top.addWidget(self.spin_int)
        top.addSpacing(12)
        self.lbl_cal = QtWidgets.QLabel("λ-cal: none")
        top.addWidget(self.lbl_cal)
        top.addStretch(1)
        self.lbl_state = QtWidgets.QLabel("Disconnected")
        self.lbl_state.setMinimumWidth(STATUS_W)
        self.lbl_state.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.lbl_state)
        outer.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()
        self.counts = CountsPanel(self.state, lambda: self.device.state == LIVE)
        self.trpl = TrplPanel(self.state, self.device, calib_loader=self.counts.calib.load_file)
        self.tabs.addTab(self.counts, "Counts")
        self.tabs.addTab(self.trpl, "TRPL")
        outer.addWidget(self.tabs, 1)

        self.status_bar = QtWidgets.QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        outer.addWidget(self.status_bar)
        self.status_bar.showMessage("Disconnected. Press Connect (pSPAD server must be running).")

    def _wire(self) -> None:
        d = self.device
        self.btn_connect.clicked.connect(d.connect_device)
        self.btn_disconnect.clicked.connect(d.disconnect_device)
        self.btn_start.clicked.connect(d.start_live)
        self.btn_stop.clicked.connect(lambda: d.stop_live())
        self.btn_si.clicked.connect(self._on_si)
        self.spin_int.valueChanged.connect(d.set_integration_ms)
        self.cmb_mode.currentIndexChanged.connect(lambda _i: self.state.set_mode(self.cmb_mode.currentData()))

        d.state_changed.connect(lambda _s: self._sync_controls())
        d.message.connect(self.show_status)
        d.connected.connect(self._on_connected)
        d.count_frame.connect(self._on_frame)
        d.live_failed.connect(self._on_live_failed)
        d.command_done.connect(self._on_command_done)

        self.state.mode_changed.connect(self._on_mode_changed)
        self.state.calibration_changed.connect(self._on_calibration_changed)
        self.state.excluded_changed.connect(self._on_excluded_changed)
        self.counts.status.connect(self.show_status)
        self.trpl.status.connect(self.show_status)
        self.counts.calib.calibration_saved.connect(lambda p: self.settings.setValue("calibration_path", p))

        self._frames = 0
        self._fps_t0 = time.monotonic()
        self._last_ratio = 0.0

    def show_status(self, text: str) -> None:
        self.status_bar.showMessage(text, 15000)

    # ======================================================== settings
    def _restore_settings(self) -> None:
        raw = self.settings.value("excluded_pixels", "")
        try:
            ex = [int(v) for v in str(raw).split(",") if v.strip()]
        except ValueError:
            ex = []
        if ex:
            self.state.set_excluded(ex)
        mode = self.settings.value("mode", MODE_2D)
        if mode in (MODE_2D, MODE_SPECTRAL):
            self.state.set_mode(mode)
        path = str(self.settings.value("calibration_path", "") or "")
        if path and os.path.exists(path):
            if self.counts.calib.load_file(path, quiet=True):
                self.show_status(f"Loaded the last wavelength calibration: {path}")

    def _on_excluded_changed(self) -> None:
        self.settings.setValue("excluded_pixels", ",".join(str(p) for p in sorted(self.state.excluded)))

    def _on_mode_changed(self, mode: str) -> None:
        i = self.cmb_mode.findData(mode)
        if i >= 0 and i != self.cmb_mode.currentIndex():
            self.cmb_mode.blockSignals(True)
            self.cmb_mode.setCurrentIndex(i)
            self.cmb_mode.blockSignals(False)
        self.settings.setValue("mode", mode)

    def _on_calibration_changed(self) -> None:
        calib = self.state.calibration
        if calib is None:
            self.lbl_cal.setText("λ-cal: none")
            self.lbl_cal.setToolTip("")
        else:
            lam0, lam1 = (float(v) for v in calib.column_wavelengths()[[0, -1]])
            self.lbl_cal.setText(f"λ-cal: {lam0:.1f}–{lam1:.1f} nm ({calib.nm_per_column:+.3f} nm/col)")
            self.lbl_cal.setToolTip(calib.describe() + (f"\n{self.state.calibration_path}" if self.state.calibration_path else ""))

    # ========================================================= device
    def _sync_controls(self) -> None:
        s = self.device.state
        self.btn_connect.setEnabled(s == DISCONNECTED)
        self.btn_disconnect.setEnabled(s in (IDLE, LIVE))
        self.btn_start.setEnabled(s == IDLE)
        self.btn_stop.setEnabled(s == LIVE)
        self.btn_si.setEnabled(s in (IDLE, LIVE))
        if s == DISCONNECTED:
            self.lbl_state.setText("Disconnected")
        elif s == CONNECTING:
            self.lbl_state.setText("Connecting …")
        elif s == IDLE:
            self.lbl_state.setText(f"Connected to {self.device.host}:{self.device.port}")
        elif s == BUSY:
            self.lbl_state.setText("TRPL running" if self.device.trpl_running else "Busy …")
        self.lbl_state.setStyleSheet("")
        self.counts.sync_controls(s == LIVE)
        self.trpl.sync_controls(s)

    def _on_connected(self, banner: str, tdc: str) -> None:
        self.show_status(f"Connected: {banner or 'no banner'} | {tdc}")
        if "invalid" in tdc.lower():
            self.show_status(f"Connected: {banner} | {tdc} — run Calibrate + Align in the TRPL tab before TRPL.")

    def _on_frame(self, snap) -> None:
        self.counts.on_frame(snap)
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            fps = self._frames / (now - self._fps_t0)
            self._frames = 0
            self._fps_t0 = now
            ratio = snap.overload_ratio
            self.lbl_state.setText(f"Live: {fps:.2f} fps | max pixel {100.0 * ratio:.0f}% of limit")
            self.lbl_state.setStyleSheet("color:#b91c1c; font-weight:bold" if ratio >= 0.8 else "")

    def _on_live_failed(self, msg: str) -> None:
        self.show_status(msg)
        if "SI" in msg or "shut" in msg.lower():
            QtWidgets.QMessageBox.warning(
                self, "SPAD93 shutdown",
                f"{msg}\n\nReduce the light level, then press Reset (SI) and Start live.")

    def _on_si(self) -> None:
        self.device.run_command("si", lambda c: c.reset_shutdown())

    def _on_command_done(self, name: str, ok: bool, msg: str) -> None:
        if name == "si":
            self.show_status(("SI: " if ok else "SI failed: ") + msg)

    # ========================================================== close
    def _quiet(self) -> bool:
        return self.device.idle and not self.trpl.busy_background

    def ready_to_close(self) -> bool:
        """True when the window may close now; otherwise stops work and closes later."""
        if self._force_close:
            return True
        if self._closing:
            return self._quiet()
        if self.device.trpl_running:
            ans = QtWidgets.QMessageBox.question(
                self, "Close SPAD93",
                "A TRPL acquisition is running. Stop it, save the data so far and close?")
            if ans != QtWidgets.QMessageBox.Yes:
                return False
            self.trpl.chk_autosave.setChecked(True)
        elif self.trpl.unsaved and not self.trpl.saving:
            ans = QtWidgets.QMessageBox.question(
                self, "Close SPAD93", "The TRPL data has not been saved. Close anyway?")
            if ans != QtWidgets.QMessageBox.Yes:
                return False
        self._closing = True
        self._close_t0 = time.monotonic()
        self.device.shutdown()
        self.trpl.cancel_background()
        if self._quiet():
            return True
        self.show_status("Stopping acquisitions before closing …")
        QtCore.QTimer.singleShot(200, self._check_close)
        return False

    def _check_close(self) -> None:
        if self._quiet():
            self.window().close()
            return
        if time.monotonic() - self._close_t0 > CLOSE_TIMEOUT_S:
            self._force_close = True
            self.window().close()
            return
        QtCore.QTimer.singleShot(200, self._check_close)
