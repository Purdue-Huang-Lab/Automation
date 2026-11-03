"""SPAD93 GUI entry points: integrated counts + TRPL app, and the offline TRPL viewer."""

import argparse
import sys
import time
import traceback

import spad93.gui_v2  # noqa: F401  (pins pyqtgraph to PyQt5 and sets the white theme)
from PyQt5 import QtCore, QtWidgets

from spad93.gui_v2.config import DEFAULT_HOST, DEFAULT_PORT
from spad93.gui_v2.state import SharedState
from spad93.gui_v2.trpl_panel import TrplPanel
from spad93.gui_v2.widget import Spad93Widget


def _install_excepthook(report) -> None:
    """PyQt5 aborts the process on an exception in a slot unless a hook is installed."""

    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            report(f"Unexpected error: {exc_type.__name__}: {exc}")
        except Exception:
            pass

    sys.excepthook = hook


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        super().__init__()
        self.setWindowTitle(f"SPAD93 Counts + TRPL v2 ({host}:{port})")
        self.resize(1800, 1000)
        self.widget = Spad93Widget(host, port)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        # Qt sends the close event to the window, not the central widget.
        try:
            ok = self.widget.ready_to_close()
        except Exception:
            ok = True
        if ok:
            event.accept()
        else:
            event.ignore()


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPAD93 TRPL Viewer v2")
        self.resize(1800, 1000)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 8)
        self.state = SharedState(self)
        self.panel = TrplPanel(self.state, device=None)
        lay.addWidget(self.panel, 1)
        self.status_bar = QtWidgets.QStatusBar(central)
        self.status_bar.setSizeGripEnabled(False)
        lay.addWidget(self.status_bar)
        self.setCentralWidget(central)
        self.panel.status.connect(lambda t: self.status_bar.showMessage(t, 15000))
        self.status_bar.showMessage("Load a SPAD93 TRPL CSV (time_ns, pixel_0 … pixel_92).")
        self._close_t0 = None

    def closeEvent(self, event):
        # Let a running Fit All / save finish so no QThread is destroyed while running.
        if not self.panel.busy_background:
            event.accept()
            return
        if self._close_t0 is None:
            self._close_t0 = time.monotonic()
            self.panel.cancel_background()
        if time.monotonic() - self._close_t0 > 15.0:
            event.accept()
            return
        event.ignore()
        QtCore.QTimer.singleShot(200, self.close)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPAD93 counts + TRPL GUI (v2)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="host of the pSPAD command server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port of the pSPAD command server")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.host, args.port)
    _install_excepthook(win.widget.show_status)
    win.show()
    sys.exit(app.exec_())


def viewer_main() -> None:
    parser = argparse.ArgumentParser(description="SPAD93 TRPL offline viewer (v2)")
    parser.add_argument("--file", default="", help="TRPL CSV to open at startup")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = ViewerWindow()
    _install_excepthook(lambda t: win.status_bar.showMessage(t, 15000))
    win.show()
    if args.file:
        QtCore.QTimer.singleShot(0, lambda: win.panel.load_file(args.file))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
