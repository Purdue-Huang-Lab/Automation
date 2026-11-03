"""Filter flipper v2 widget: connect, move between positions 1/2, transit time, cycle test."""

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from flipper.flipper_wrapper import (
    FilterFlipper,
    MoveController,
    MoveResult,
    count_busy_devices,
    list_kinesis_devices,
)
from . import config as cfg
from .config import (
    BTN_W,
    DEFAULT_CYCLES,
    DEFAULT_DWELL_S,
    DEFAULT_SERIAL,
    DEFAULT_TRANSIT_S,
    OFF_COLOR,
    ON_COLOR,
    POLL_ERROR_LIMIT,
    POLL_MS,
    PUMP2_ON_DEFAULT,
    STATUS_W,
    TRANSIT_TIME_RANGE_S,
)
from .plots import MoveTimePlot, PositionHistoryPlot
from .workers import CallThread, CycleThread, PollThread, make_move_call


def _grid(box: QtWidgets.QGroupBox) -> QtWidgets.QGridLayout:
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return g


class FilterFlipperWidget(QtWidgets.QWidget):
    def __init__(self, simulate: bool = False, parent=None):
        super().__init__(parent)
        self.flipper: Optional[FilterFlipper] = None
        self._connected = False
        self._job: Optional[QtCore.QThread] = None
        self._job_kind = ""
        self._poll: Optional[PollThread] = None
        self._move_ctrl: Optional[MoveController] = None
        self._position: Optional[int] = None
        self._poll_errors = 0
        self._disconnecting = False
        self._settings = QtCore.QSettings(cfg.SETTINGS_ORG, cfg.SETTINGS_APP)

        self._build_ui(simulate)
        self._sync_controls()
        self._update_position_display()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, simulate: bool) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 4)
        outer.setSpacing(8)

        # 1. top bar
        top = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_pos1 = QtWidgets.QPushButton("Position 1")
        self.btn_pos2 = QtWidgets.QPushButton("Position 2")
        self.btn_toggle = QtWidgets.QPushButton("Toggle")
        self.btn_identify = QtWidgets.QPushButton("Identify")
        self.btn_identify.setToolTip("Blink the LED on the flipper")
        for b in (self.btn_connect, self.btn_disconnect, self.btn_pos1, self.btn_pos2, self.btn_toggle,
                  self.btn_identify):
            b.setFixedWidth(BTN_W)
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_disconnect)
        top.addSpacing(16)
        top.addWidget(self.btn_pos1)
        top.addWidget(self.btn_pos2)
        top.addWidget(self.btn_toggle)
        top.addWidget(self.btn_identify)
        top.addStretch()
        self.stateLbl = QtWidgets.QLabel("Disconnected")
        self.stateLbl.setFixedWidth(STATUS_W)
        self.stateLbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.stateLbl)
        outer.addLayout(top)

        # 2. settings panels
        middle = QtWidgets.QHBoxLayout()
        middle.setSpacing(8)

        dev_box = QtWidgets.QGroupBox("Device")
        g = _grid(dev_box)
        self.serialCombo = QtWidgets.QComboBox()
        self.serialCombo.setEditable(True)
        self.serialCombo.addItem(DEFAULT_SERIAL)
        self.serialCombo.setMinimumWidth(130)
        self.btn_search = QtWidgets.QPushButton("Search")
        self.btn_search.setFixedWidth(BTN_W)
        self.chk_simulate = QtWidgets.QCheckBox("Simulate (no hardware)")
        self.chk_simulate.setChecked(bool(simulate))
        self.infoLbl = QtWidgets.QLabel("--")
        self.infoLbl.setWordWrap(True)
        g.addWidget(QtWidgets.QLabel("Serial:"), 0, 0)
        g.addWidget(self.serialCombo, 0, 1)
        g.addWidget(self.btn_search, 0, 2)
        g.addWidget(self.chk_simulate, 1, 0, 1, 3)
        g.addWidget(self.infoLbl, 2, 0, 1, 3)
        g.setRowStretch(3, 1)
        middle.addWidget(dev_box, 2)

        pos_box = QtWidgets.QGroupBox("Position")
        g = _grid(pos_box)
        self.posLbl = QtWidgets.QLabel("--")
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.posLbl.setFont(font)
        self.pumpLbl = QtWidgets.QLabel("")
        self.pump2Combo = QtWidgets.QComboBox()
        self.pump2Combo.addItems(["Position 1", "Position 2"])
        self.pump2Combo.setToolTip("Which flipper position lets pump 2 through (saved for the PL + TRPL GUI)")
        on_pos = self._settings.value("pump2_on_position", PUMP2_ON_DEFAULT, type=int)
        self.pump2Combo.setCurrentIndex(1 if int(on_pos) == 2 else 0)
        self.transitSpin = QtWidgets.QDoubleSpinBox()
        self.transitSpin.setRange(*TRANSIT_TIME_RANGE_S)
        self.transitSpin.setDecimals(3)
        self.transitSpin.setSingleStep(0.1)
        self.transitSpin.setSuffix(" s")
        self.transitSpin.setValue(DEFAULT_TRANSIT_S)
        self.btn_transit = QtWidgets.QPushButton("Apply")
        self.btn_transit.setFixedWidth(BTN_W)
        self.lastMoveLbl = QtWidgets.QLabel("Last move: --")
        g.addWidget(self.posLbl, 0, 0, 1, 3)
        g.addWidget(self.pumpLbl, 1, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("Pump 2 passes at:"), 2, 0)
        g.addWidget(self.pump2Combo, 2, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("Transit time:"), 3, 0)
        g.addWidget(self.transitSpin, 3, 1)
        g.addWidget(self.btn_transit, 3, 2)
        g.addWidget(self.lastMoveLbl, 4, 0, 1, 3)
        middle.addWidget(pos_box, 2)

        cyc_box = QtWidgets.QGroupBox("Cycle test")
        g = _grid(cyc_box)
        self.cyclesSpin = QtWidgets.QSpinBox()
        self.cyclesSpin.setRange(1, 100000)
        self.cyclesSpin.setValue(DEFAULT_CYCLES)
        self.cyclesSpin.setToolTip("Number of moves (each move toggles the position)")
        self.dwellSpin = QtWidgets.QDoubleSpinBox()
        self.dwellSpin.setRange(0.0, 3600.0)
        self.dwellSpin.setDecimals(2)
        self.dwellSpin.setSuffix(" s")
        self.dwellSpin.setValue(DEFAULT_DWELL_S)
        self.btn_cycle_start = QtWidgets.QPushButton("Start")
        self.btn_cycle_stop = QtWidgets.QPushButton("Stop")
        self.btn_cycle_clear = QtWidgets.QPushButton("Clear plot")
        for b in (self.btn_cycle_start, self.btn_cycle_stop, self.btn_cycle_clear):
            b.setFixedWidth(BTN_W)
        self.cycleLbl = QtWidgets.QLabel("--")
        g.addWidget(QtWidgets.QLabel("Moves:"), 0, 0)
        g.addWidget(self.cyclesSpin, 0, 1)
        g.addWidget(QtWidgets.QLabel("Dwell:"), 1, 0)
        g.addWidget(self.dwellSpin, 1, 1)
        g.addWidget(self.btn_cycle_start, 0, 2)
        g.addWidget(self.btn_cycle_stop, 1, 2)
        g.addWidget(self.btn_cycle_clear, 2, 2)
        g.addWidget(self.cycleLbl, 3, 0, 1, 3)
        middle.addWidget(cyc_box, 2)
        outer.addLayout(middle)

        # 3. live display
        plots = QtWidgets.QHBoxLayout()
        self.historyPlot = PositionHistoryPlot()
        self.movePlot = MoveTimePlot()
        plots.addWidget(self.historyPlot, 3)
        plots.addWidget(self.movePlot, 2)
        outer.addLayout(plots, 1)

        # 4. status bar
        self._status_bar = QtWidgets.QStatusBar(self)
        outer.addWidget(self._status_bar)
        self._status_bar.showMessage("Idle")

        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_disconnect.clicked.connect(self.on_disconnect)
        self.btn_pos1.clicked.connect(lambda: self.on_move(1))
        self.btn_pos2.clicked.connect(lambda: self.on_move(2))
        self.btn_toggle.clicked.connect(lambda: self.on_move(None))
        self.btn_identify.clicked.connect(self.on_identify)
        self.btn_search.clicked.connect(self.on_search)
        self.btn_transit.clicked.connect(self.on_apply_transit)
        self.btn_cycle_start.clicked.connect(self.on_cycle_start)
        self.btn_cycle_stop.clicked.connect(self.on_cycle_stop)
        self.btn_cycle_clear.clicked.connect(self.on_cycle_clear)
        self.pump2Combo.currentIndexChanged.connect(self._on_pump2_changed)

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    # ------------------------------------------------------------ state
    @property
    def pump2_on_position(self) -> int:
        return 2 if self.pump2Combo.currentIndex() == 1 else 1

    def _job_running(self) -> bool:
        return self._job is not None and self._job.isRunning()

    def _sync_controls(self) -> None:
        busy = self._job_running() or self._disconnecting
        conn = self._connected
        cycling = self._job_kind == "cycle" and self._job_running()
        self.btn_connect.setEnabled(not conn and not busy)
        self.btn_disconnect.setEnabled(conn and not self._disconnecting)
        for b in (self.btn_pos1, self.btn_pos2, self.btn_toggle, self.btn_identify, self.btn_transit,
                  self.btn_cycle_start):
            b.setEnabled(conn and not busy)
        self.btn_cycle_stop.setEnabled(cycling)
        self.btn_search.setEnabled(not conn and not busy)
        self.serialCombo.setEnabled(not conn and not busy)
        self.chk_simulate.setEnabled(not conn and not busy)
        self.transitSpin.setEnabled(conn and not busy)
        self.cyclesSpin.setEnabled(not cycling)
        self.dwellSpin.setEnabled(not cycling)

    def _position_name(self, pos: Optional[int]) -> str:
        if pos is None:
            return "in transit" if self._connected else "--"
        return f"Position {pos}"

    def _pump2_text(self, pos: Optional[int]) -> str:
        if pos is None:
            return ""
        return "pump 2 ON" if pos == self.pump2_on_position else "pump 2 OFF"

    def _update_position_display(self) -> None:
        pos = self._position
        self.posLbl.setText(self._position_name(pos))
        p2 = self._pump2_text(pos)
        self.pumpLbl.setText(p2)
        self.pumpLbl.setStyleSheet(f"color:{ON_COLOR if p2 == 'pump 2 ON' else OFF_COLOR}; font-weight:bold;")
        on = self.pump2_on_position
        self.historyPlot.set_labels(
            "Position 1" + (" (pump 2 on)" if on == 1 else ""),
            "Position 2" + (" (pump 2 on)" if on == 2 else ""),
        )
        if not self._connected:
            self.stateLbl.setText("Disconnected")
        else:
            serial = self.flipper.serial if self.flipper is not None else ""
            extra = f" | {p2}" if p2 else ""
            self.stateLbl.setText(f"Connected {serial} | {self._position_name(pos)}{extra}")

    def _on_pump2_changed(self, _idx: int) -> None:
        self._settings.setValue("pump2_on_position", self.pump2_on_position)
        self._update_position_display()

    def _start_job(self, kind: str, thread: QtCore.QThread, on_done) -> bool:
        if self._job_running():
            return False
        self._job = thread
        self._job_kind = kind
        thread.done.connect(on_done)
        thread.finished.connect(self._on_job_finished)
        thread.start()
        self._sync_controls()
        return True

    def _on_job_finished(self) -> None:
        if self._job is not None and not self._job.isRunning():
            self._job = None
            self._job_kind = ""
        self._sync_controls()

    # ------------------------------------------------------------ connect
    def on_search(self) -> None:
        def _search():
            return list_kinesis_devices(), count_busy_devices()

        self.statusBar().showMessage("Searching for Kinesis devices...")
        self._start_job("search", CallThread(_search, self), self._on_search_done)

    def _on_search_done(self, res, err: str) -> None:
        if err:
            self.statusBar().showMessage(f"Search failed: {err}")
            return
        devs, busy = res
        flippers = [s for s, _ in devs if s.startswith("37")]
        current = self.serialCombo.currentText().strip()
        self.serialCombo.blockSignals(True)
        self.serialCombo.clear()
        self.serialCombo.addItems(flippers or [current or DEFAULT_SERIAL])
        if current:
            self.serialCombo.setCurrentText(current)
        self.serialCombo.blockSignals(False)
        msg = f"Found {len(flippers)} free flipper(s)"
        if busy:
            msg += f"; {busy} Kinesis device(s) are open in another program (Kinesis app?)"
        self.statusBar().showMessage(msg)

    def on_connect(self) -> None:
        if self._connected or self._job_running():
            return
        serial = self.serialCombo.currentText().strip()
        flipper = FilterFlipper(serial, simulate=self.chk_simulate.isChecked(), verbose=False)

        def _open():
            info = flipper.open()
            try:
                transit = flipper.get_transit_time_s()
            except Exception:
                transit = None
            return flipper, info, transit

        self.statusBar().showMessage(f"Connecting to {serial}...")
        self._start_job("connect", CallThread(_open, self), self._on_connect_done)

    def _on_connect_done(self, res, err: str) -> None:
        if err:
            self.statusBar().showMessage(err)
            QtWidgets.QMessageBox.warning(self, "Filter flipper", err)
            return
        flipper, info, transit = res
        self.flipper = flipper
        self._connected = True
        self._poll_errors = 0
        self._position = flipper.last_position
        self.infoLbl.setText(info.text())
        if transit is not None:
            self.transitSpin.setValue(float(transit))
        self.historyPlot.clear()
        self._poll = PollThread(flipper, POLL_MS / 1000.0, self)
        self._poll.sample.connect(self._on_poll_sample)
        self._poll.start()
        self.statusBar().showMessage(f"Connected to {flipper.serial}")
        self._update_position_display()
        self._sync_controls()

    def on_disconnect(self) -> None:
        if not self._connected or self._disconnecting:
            return
        self._disconnecting = True
        if self._move_ctrl is not None:
            self._move_ctrl.abort()
        if isinstance(self._job, CycleThread):
            self._job.stop()
        if self._poll is not None:
            self._poll.stop()
        self.statusBar().showMessage("Disconnecting...")
        self._sync_controls()
        QtCore.QTimer.singleShot(200, self._check_stop)

    def _check_stop(self) -> None:
        if (self._poll is not None and self._poll.isRunning()) or self._job_running():
            QtCore.QTimer.singleShot(200, self._check_stop)
            return
        self._poll = None
        flipper = self.flipper
        self._start_job("close", CallThread(flipper.close if flipper is not None else (lambda: None), self),
                        self._on_closed)

    def _on_closed(self, _res, err: str) -> None:
        self.flipper = None
        self._connected = False
        self._disconnecting = False
        self._position = None
        self.infoLbl.setText("--")
        self.statusBar().showMessage("Disconnected" + (f" ({err})" if err else ""))
        self._update_position_display()
        self._sync_controls()

    # ------------------------------------------------------------ polling
    def _on_poll_sample(self, t: float, pos, err: str) -> None:
        if not self._connected or self._disconnecting:
            return
        if err:
            self._poll_errors += 1
            self.statusBar().showMessage(f"Read error ({self._poll_errors}/{POLL_ERROR_LIMIT}): {err}")
            if self._poll_errors >= POLL_ERROR_LIMIT:
                self.statusBar().showMessage("Flipper not responding; disconnecting")
                self.on_disconnect()
            return
        self._poll_errors = 0
        self._position = pos
        self.historyPlot.add_sample(t, pos)
        self._update_position_display()

    # ------------------------------------------------------------ commands
    def on_move(self, target: Optional[int]) -> None:
        if not self._connected or self.flipper is None:
            return
        self._move_ctrl = MoveController()
        what = "Toggling" if target is None else f"Moving to position {target}"
        self.statusBar().showMessage(f"{what}...")
        self._start_job("move", CallThread(make_move_call(self.flipper, target, self._move_ctrl), self),
                        self._on_move_done)

    def _on_move_done(self, res, err: str) -> None:
        self._move_ctrl = None
        if err:
            self.statusBar().showMessage(f"Move failed: {err}")
            return
        if isinstance(res, MoveResult):
            self._record_move(res)
            if res.aborted:
                self.statusBar().showMessage("Move wait aborted")
            else:
                self.statusBar().showMessage(f"At position {res.position} ({res.duration_s:.2f} s)")

    def _record_move(self, res: MoveResult) -> None:
        if res.aborted or res.position is None:
            return
        self._position = res.position
        self.lastMoveLbl.setText(f"Last move: to {res.position} in {res.duration_s:.2f} s")
        self.movePlot.add_move(res.duration_s)
        self._update_position_display()

    def on_identify(self) -> None:
        if self.flipper is None:
            return
        self._start_job("identify", CallThread(self.flipper.identify, self), self._on_identify_done)

    def _on_identify_done(self, _res, err: str) -> None:
        self.statusBar().showMessage(f"Identify failed: {err}" if err else "LED blinking")

    def on_apply_transit(self) -> None:
        if self.flipper is None:
            return
        value = float(self.transitSpin.value())
        flipper = self.flipper
        self._start_job("transit", CallThread(lambda: flipper.set_transit_time_s(value), self),
                        self._on_transit_done)

    def _on_transit_done(self, res, err: str) -> None:
        if err:
            self.statusBar().showMessage(f"Transit time not set: {err}")
            return
        self.transitSpin.setValue(float(res))
        self.statusBar().showMessage(f"Transit time {float(res):.3f} s")

    # ------------------------------------------------------------ cycle test
    def on_cycle_start(self) -> None:
        if self.flipper is None:
            return
        n = int(self.cyclesSpin.value())
        thread = CycleThread(self.flipper, n, float(self.dwellSpin.value()), self)
        thread.progress.connect(self._on_cycle_progress)
        self.cycleLbl.setText(f"0/{n}")
        if self._start_job("cycle", thread, self._on_cycle_done):
            self.statusBar().showMessage(f"Cycle test: {n} moves")

    def on_cycle_stop(self) -> None:
        if isinstance(self._job, CycleThread):
            self._job.stop()
            self.statusBar().showMessage("Stopping cycle test...")

    def _on_cycle_progress(self, k: int, n: int, res) -> None:
        if isinstance(res, MoveResult):
            self._record_move(res)
            self.cycleLbl.setText(f"{k}/{n}: to {res.position} in {res.duration_s:.2f} s")

    def _on_cycle_done(self, status: str, err: str) -> None:
        n, mean, std, mx = self.movePlot.stats()
        stats = f"{n} moves, {mean:.3f} ± {std:.3f} s (max {mx:.3f} s)" if n else "no moves"
        if status == "ok":
            self.statusBar().showMessage(f"Cycle test done: {stats}")
        elif status == "aborted":
            self.statusBar().showMessage(f"Cycle test stopped: {stats}")
        else:
            self.statusBar().showMessage(f"Cycle test failed: {err}")
        self.cycleLbl.setText(stats)

    def on_cycle_clear(self) -> None:
        self.movePlot.clear()
        self.cycleLbl.setText("--")

    # ------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        """Stop the threads and close the device (window close: short bounded waits are fine here)."""
        if self._move_ctrl is not None:
            self._move_ctrl.abort()
        if isinstance(self._job, CycleThread):
            self._job.stop()
        if self._poll is not None:
            self._poll.stop()
        for th in (self._poll, self._job):
            if th is not None and th.isRunning():
                th.wait(5000)
        if self.flipper is not None:
            self.flipper.close()
        self.flipper = None
        self._connected = False
