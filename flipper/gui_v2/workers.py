"""QThreads for the filter flipper GUI: every device call runs off the GUI thread."""

import time
from typing import Callable, Optional

from PyQt5 import QtCore

from flipper.flipper_wrapper import FilterFlipper, FilterFlipperError, MoveController


class CallThread(QtCore.QThread):
    """Run one blocking call (connect, move, identify, set transit time) and report the result."""

    done = QtCore.pyqtSignal(object, str)  # result, error text ("" on success)

    def __init__(self, fn: Callable[[], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            res = self._fn()
        except Exception as exc:
            self.done.emit(None, str(exc) or type(exc).__name__)
            return
        self.done.emit(res, "")


class PollThread(QtCore.QThread):
    """Read the flipper position every ``interval_s`` until ``stop()``."""

    sample = QtCore.pyqtSignal(float, object, str)  # monotonic time, position (1/2/None), error text

    def __init__(self, flipper: FilterFlipper, interval_s: float, parent=None):
        super().__init__(parent)
        self._flipper = flipper
        self._min_interval_s = float(interval_s)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self):
        while not self._stop:
            loop_start = time.monotonic()
            try:
                pos = self._flipper.get_position()
                self.sample.emit(loop_start, pos, "")
            except Exception as exc:
                self.sample.emit(loop_start, None, str(exc) or type(exc).__name__)
            elapsed = time.monotonic() - loop_start
            # sleep in short slices so stop() is honoured quickly
            remaining = self._min_interval_s - elapsed
            while remaining > 0 and not self._stop:
                time.sleep(min(0.05, remaining))
                remaining = self._min_interval_s - (time.monotonic() - loop_start)


class CycleThread(QtCore.QThread):
    """Toggle the flipper ``n_cycles`` times (each cycle = one move), dwelling ``dwell_s`` after each move."""

    progress = QtCore.pyqtSignal(int, int, object)  # move index (1-based), total moves, MoveResult
    done = QtCore.pyqtSignal(str, str)  # "ok" | "aborted" | "error", error text

    def __init__(self, flipper: FilterFlipper, n_cycles: int, dwell_s: float, parent=None):
        super().__init__(parent)
        self._flipper = flipper
        self._n = int(n_cycles)
        self._dwell_s = float(dwell_s)
        self._controller = MoveController()

    def stop(self) -> None:
        self._controller.abort()

    def run(self):
        try:
            for k in range(1, self._n + 1):
                if self._controller.is_aborted():
                    self.done.emit("aborted", "")
                    return
                res = self._flipper.toggle(controller=self._controller)
                self.progress.emit(k, self._n, res)
                if res.aborted:
                    self.done.emit("aborted", "")
                    return
                t_end = time.monotonic() + self._dwell_s
                while time.monotonic() < t_end:
                    if self._controller.is_aborted():
                        self.done.emit("aborted", "")
                        return
                    time.sleep(0.02)
        except FilterFlipperError as exc:
            self.done.emit("error", str(exc))
            return
        except Exception as exc:
            self.done.emit("error", str(exc) or type(exc).__name__)
            return
        self.done.emit("ok", "")


def make_move_call(flipper: FilterFlipper, target: Optional[int], controller: MoveController) -> Callable[[], object]:
    """Callable for CallThread: move to ``target`` (``None`` = toggle) and wait."""
    if target is None:
        return lambda: flipper.toggle(controller=controller)
    return lambda: flipper.move_to(int(target), controller=controller)
