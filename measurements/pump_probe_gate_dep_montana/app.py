import math
import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# The package __init__ pins PyQt5, imports it before pyqtgraph and sets the pg config options.
import measurements.pump_probe_gate_dep_montana  # noqa: E402,F401

from PyQt5 import QtWidgets  # noqa: E402

from measurements.pl_trpl_power_voltage_v2.ui import apply_ui_font, enable_high_dpi, scale_fixed_sizes  # noqa: E402
from measurements.pump_probe_gate_dep_montana.widget import PumpProbeMontanaWidget  # noqa: E402


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pump-Probe Gate & Power Dependence — Montana (SMC100)")
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1800, int(avail.width() * 0.95)), min(1000, int(avail.height() * 0.92)))
        self.widget = PumpProbeMontanaWidget()
        scale_fixed_sizes(self.widget)
        self.setCentralWidget(self.widget)
        self._close_state = None  # None, "ramping", "ready"

    def closeEvent(self, event):
        if self._close_state == "ramping":
            event.ignore()
            return
        if self._close_state is None:
            at_v = self.widget.gates_at_voltage()
            if at_v:
                gates = ", ".join(f"{w} gate at {'?' if math.isnan(v) else f'{v:.4g}'} V" for w, v in at_v.items())
                ans = QtWidgets.QMessageBox.question(
                    self, "Close",
                    f"{gates[0].upper()}{gates[1:]}.\n\n"
                    "Yes: ramp the gates to 0 V, switch the outputs off, then close.\n"
                    "No: close now and leave the outputs ON at their voltage.\n"
                    "Cancel: keep the window open.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Yes)
                if ans == QtWidgets.QMessageBox.Cancel:
                    event.ignore()
                    return
                if ans == QtWidgets.QMessageBox.Yes:
                    self._close_state = "ramping"
                    self.widget.ramp_down_gates(self._after_ramp_down)
                    event.ignore()
                    return
                self.widget.leave_gate_outputs_on()
        try:
            self.widget.shutdown()
        except Exception:
            pass
        event.accept()

    def _after_ramp_down(self, errors) -> None:
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Close", "\n".join(f"{w.capitalize()} gate: {e}" for w, e in errors.items()))
        self._close_state = "ready"
        self.close()


_ERROR_LOG = None  # open file: faulthandler writes native crash tracebacks into it


def install_error_handlers(get_widget) -> str:
    """Keep the GUI alive on uncaught Python exceptions, and leave a trace of native crashes.

    PyQt5 aborts the program on an exception that escapes a slot or a thread; this hook
    logs it (stderr, ``ERROR_LOG_NAME`` in DATA_DIR, the Sweep log tab) instead.  A crash
    inside a driver DLL (camera, stage) cannot be caught: ``faulthandler`` writes its
    Python traceback to the same file, and the sweep state file allows 'Resume sweep'.
    Returns the log path ('' if no file could be opened).
    """
    import faulthandler
    import tempfile
    import threading
    import traceback
    from datetime import datetime

    from measurements.pump_probe_gate_dep_montana import config as C

    global _ERROR_LOG
    path = ""
    for folder in (C.DATA_DIR, tempfile.gettempdir()):
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, C.ERROR_LOG_NAME)
            _ERROR_LOG = open(path, "a", encoding="utf-8")
            _ERROR_LOG.write(f"\n==== GUI started {datetime.now():%Y-%m-%d %H:%M:%S} (pid {os.getpid()})\n")
            _ERROR_LOG.flush()
            faulthandler.enable(file=_ERROR_LOG, all_threads=True)
            break
        except Exception:
            _ERROR_LOG, path = None, ""

    def hook(exc_type, exc, tb):
        try:
            text = "".join(traceback.format_exception(exc_type, exc, tb))
            if sys.__stderr__ is not None:
                sys.__stderr__.write(text)
            if _ERROR_LOG is not None:
                _ERROR_LOG.write(f"---- {datetime.now():%Y-%m-%d %H:%M:%S} uncaught exception "
                                 f"({threading.current_thread().name})\n{text}")
                _ERROR_LOG.flush()
            w = get_widget()
            if w is not None:  # signal: queued to the GUI thread from any thread
                w.error_reported.emit(f"{exc_type.__name__}: {exc}" + (f"  (details in {path})" if path else ""))
        except Exception:
            pass

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)
    return path


def main():
    enable_high_dpi()
    app = QtWidgets.QApplication(sys.argv)
    apply_ui_font(app)
    holder = {}
    install_error_handlers(lambda: getattr(holder.get("win"), "widget", None))
    win = MainWindow()
    holder["win"] = win
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
