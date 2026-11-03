"""DeviceController: owns the hardware workers and gives one of them the server at a time."""

from typing import Callable, Optional

from PyQt5 import QtCore

from spad93.spad93_tcspc_wrapper import Spad93TcspcClient
from spad93.gui_v2.config import (
    LIVE_INTEGRATION_MS,
    LIVE_MIN_INTERVAL_S,
    TRPL_CANCEL_WAIT_S,
    TRPL_UPDATE_S,
)
from spad93.gui_v2.workers import CommandWorker, ConnectWorker, CountLiveWorker, TrplAcquireWorker

DISCONNECTED = "disconnected"
CONNECTING = "connecting"
IDLE = "idle"
LIVE = "live"
BUSY = "busy"


class DeviceController(QtCore.QObject):
    """
    States: DISCONNECTED -> CONNECTING -> IDLE <-> LIVE, and BUSY while a TDC/SI
    command or a TRPL acquisition runs.  Live counts are stopped before a job
    starts and resumed when it ends.  Threads are never waited on in the GUI
    thread; state changes follow ``QThread.finished``.
    """

    state_changed = QtCore.pyqtSignal(str)
    message = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(str, str)  # banner, TDC status
    count_frame = QtCore.pyqtSignal(object)
    live_failed = QtCore.pyqtSignal(str)
    command_done = QtCore.pyqtSignal(str, bool, str)
    trpl_progress = QtCore.pyqtSignal(object)
    trpl_done = QtCore.pyqtSignal(object, str)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self._state = DISCONNECTED
        self._integration_ms = LIVE_INTEGRATION_MS
        self._connect_worker: Optional[ConnectWorker] = None
        self._live: Optional[CountLiveWorker] = None
        self._job = None
        self._after_live_stop: Optional[Callable[[], None]] = None
        self._resume_live = False
        self._closing = False

    # ------------------------------------------------------------ state
    @property
    def state(self) -> str:
        return self._state

    @property
    def trpl_running(self) -> bool:
        return isinstance(self._job, TrplAcquireWorker)

    @property
    def idle(self) -> bool:
        return self._live is None and self._job is None and self._connect_worker is None

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    # ------------------------------------------------------- connection
    def connect_device(self) -> None:
        if self._state != DISCONNECTED or self._connect_worker is not None:
            return
        self._set_state(CONNECTING)
        self.message.emit(f"Connecting to {self.host}:{self.port} ...")
        w = ConnectWorker(self.host, self.port, self)
        w.done.connect(self._on_connect_done)
        w.finished.connect(w.deleteLater)
        self._connect_worker = w
        w.start()

    def _on_connect_done(self, ok: bool, banner: str, info: str) -> None:
        self._connect_worker = None
        if ok:
            self._set_state(IDLE)
            self.connected.emit(banner, info)
        else:
            self._set_state(DISCONNECTED)
            self.message.emit(f"Connection failed: {info}")

    def disconnect_device(self) -> None:
        if self._job is not None:
            self.message.emit("Stop the running job before disconnecting.")
            return
        self._resume_live = False
        self.stop_live(then=lambda: self._set_state(DISCONNECTED))

    # ------------------------------------------------------- live counts
    def set_integration_ms(self, ms: int) -> None:
        self._integration_ms = max(10, int(ms))
        if self._live is not None:
            self._live.set_integration_ms(self._integration_ms)

    def start_live(self) -> None:
        if self._state != IDLE or self._live is not None or self._closing:
            return
        w = CountLiveWorker(self.host, self.port, self._integration_ms, LIVE_MIN_INTERVAL_S, self)
        w.frame_ready.connect(self.count_frame)
        w.status.connect(self.message)
        w.failed.connect(self._on_live_failed)
        w.finished.connect(self._on_live_finished)
        self._live = w
        self._set_state(LIVE)
        w.start()

    def stop_live(self, then: Optional[Callable[[], None]] = None) -> None:
        """Stop live counts; ``then`` runs once the thread has finished (or at once)."""
        if self._live is None:
            if then is not None:
                then()
            return
        self._after_live_stop = then
        self._live.stop()

    def _on_live_failed(self, msg: str) -> None:
        self.live_failed.emit(msg)

    def _on_live_finished(self) -> None:
        w, self._live = self._live, None
        if w is not None:
            w.deleteLater()
        if self._state == LIVE:
            self._set_state(IDLE)
        then, self._after_live_stop = self._after_live_stop, None
        if then is not None:
            then()

    # -------------------------------------------------------------- jobs
    def _run_job(self, start_fn: Callable[[], None]) -> bool:
        if self._state not in (IDLE, LIVE) or self._job is not None:
            self.message.emit("The device is busy.")
            return False
        self._resume_live = self._state == LIVE
        self._set_state(BUSY)
        self._job = "pending"
        self.stop_live(then=start_fn)
        return True

    def _start_worker(self, w: QtCore.QThread) -> None:
        w.finished.connect(self._on_job_finished)
        self._job = w
        w.start()
        self.state_changed.emit(self._state)  # the job is running now: let the UI enable Stop

    def _on_job_finished(self) -> None:
        w, self._job = self._job, None
        if isinstance(w, QtCore.QThread):
            w.deleteLater()
        if self._state == BUSY:
            self._set_state(IDLE)
        if self._resume_live and not self._closing:
            self.start_live()

    def run_command(self, name: str, fn: Callable[[Spad93TcspcClient], str]) -> bool:
        def start() -> None:
            w = CommandWorker(self.host, self.port, name, fn, self)
            w.done.connect(self.command_done)
            self._start_worker(w)

        return self._run_job(start)

    def start_trpl(self, measurement_ms: int, bin_width_ps: int) -> bool:
        def start() -> None:
            w = TrplAcquireWorker(self.host, self.port, measurement_ms, bin_width_ps, TRPL_UPDATE_S,
                                  TRPL_CANCEL_WAIT_S, self)
            w.progress.connect(self.trpl_progress)
            w.done.connect(self.trpl_done)
            self._start_worker(w)

        return self._run_job(start)

    def stop_trpl(self) -> None:
        if isinstance(self._job, TrplAcquireWorker):
            self._job.stop()
            self.message.emit("Stopping TRPL (cancel sent) ...")

    # ------------------------------------------------------------ close
    def shutdown(self) -> None:
        """Ask every worker to stop (used when the window closes)."""
        self._closing = True
        self._resume_live = False
        if self._live is not None:
            self._live.stop()
        if isinstance(self._job, TrplAcquireWorker):
            self._job.stop()
