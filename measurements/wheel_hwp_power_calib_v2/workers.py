"""Power-meter thread of the wheel + HWP v2 GUI.

The sweep itself runs in the v1 ``StageSweepWorker`` (re-exported here unchanged);
the stages use ``StageDevice`` of the PL + TRPL v2 GUI.
"""

import queue
import time

from PyQt5 import QtCore

from PM100A.pm100a_wrapper import PM100A
from measurements.wheel_hwp_power_calib_workers import StageSweepWorker  # noqa: F401

from .config import PM_ZERO_TIMEOUT_S


class PMThread(QtCore.QThread):
    """Owns one PM100A session for its whole life and runs queued commands on it.

    TLPM sessions have thread affinity (every call must come from the thread that
    opened the session), so the session is opened in ``run()`` and every command is
    executed here, never in the GUI thread.

    Commands (``submit(cmd, arg)``): ``wavelength`` (nm), ``averaging`` (count), ``read``,
    ``zero`` (dark adjust, waits until done).  ``result(cmd, value, error)`` reports each;
    ``open`` and ``closed`` are reported for the session itself.
    """

    result = QtCore.pyqtSignal(str, object, str)

    def __init__(self, resource: str, parent=None):
        super().__init__(parent)
        self.resource = resource
        self._queue: "queue.Queue" = queue.Queue()

    def submit(self, cmd: str, arg=None) -> None:
        self._queue.put((cmd, arg))

    def stop(self) -> None:
        self._queue.put(("close", None))

    def run(self) -> None:
        try:
            pm = PM100A(self.resource)
        except Exception as exc:
            self.result.emit("open", None, str(exc) or type(exc).__name__)
            return
        self.result.emit("open", self.resource, "")
        try:
            while True:
                cmd, arg = self._queue.get()
                if cmd == "close":
                    break
                try:
                    val = self._run_cmd(pm, cmd, arg)
                except Exception as exc:
                    self.result.emit(cmd, None, str(exc) or type(exc).__name__)
                else:
                    self.result.emit(cmd, val, "")
        finally:
            try:
                pm.close()
            except Exception:
                pass
            self.result.emit("closed", None, "")

    @staticmethod
    def _run_cmd(pm: PM100A, cmd: str, arg):
        if cmd == "wavelength":
            pm.set_wavelength(float(arg))
            return float(arg)
        if cmd == "averaging":
            pm.set_averaging(int(arg))
            return int(arg)
        if cmd == "read":
            return float(pm.measure_power())
        if cmd == "zero":
            pm.start_dark_adjust()
            deadline = time.monotonic() + PM_ZERO_TIMEOUT_S
            while pm.is_dark_adjust_busy():
                if time.monotonic() > deadline:
                    raise TimeoutError("dark adjust timed out")
                time.sleep(0.2)
            return None
        raise ValueError(f"unknown power meter command {cmd!r}")
