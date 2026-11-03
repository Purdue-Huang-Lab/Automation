import os
import sys

# Allow running as a script inside the package directory
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# The package __init__ pins PyQt5, imports it before pyqtgraph and sets the pg config options.
import measurements.pl_trpl_power_voltage_v2  # noqa: E402,F401

import math  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402

from measurements.pl_trpl_power_voltage_v2.ui import apply_ui_font, enable_high_dpi, scale_fixed_sizes  # noqa: E402
from measurements.pl_trpl_power_voltage_v2.widget import PLTRPLPowerVoltageWidgetV2  # noqa: E402


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power/Voltage Dependent PL + TRPL v2")
        # 1700 x 950, but never larger than the screen (at 150 % a 1080p screen is 1280 x 720)
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1700, int(avail.width() * 0.95)), min(950, int(avail.height() * 0.92)))
        self.widget = PLTRPLPowerVoltageWidgetV2()
        scale_fixed_sizes(self.widget)
        # The controls need about 1590 x 1050 px (more at 125 / 150 %): on a smaller screen the window
        # could not shrink to fit and its bottom went off-screen (QWindowsWindow::setGeometry warnings).
        # Scroll instead; the status bar (scan time estimate) moves to the window and stays visible.
        self.setStatusBar(self.widget.statusBar())
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)  # as before when the window is large enough: no scrollbars
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.widget)
        self.setCentralWidget(scroll)
        self._close_state = None  # None, "ramping" (gates ramping to 0 V), "ready"

    def closeEvent(self, event):
        if self._close_state == "ramping":
            event.ignore()  # wait for the ramp to 0 V (Abort stops it and leaves the outputs on)
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
            self.widget.shutdown()  # abort, then close every device
        except Exception:
            pass
        event.accept()

    def _after_ramp_down(self, errors) -> None:
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Close", "\n".join(f"{w.capitalize()} gate: {e}" for w, e in errors.items()))
        self._close_state = "ready"
        self.close()


def make_application(argv) -> QtWidgets.QApplication:
    """QApplication with high-DPI scaling and the Windows UI font (see ui.py)."""
    enable_high_dpi()
    app = QtWidgets.QApplication(argv)
    apply_ui_font(app)
    return app


def main():
    app = make_application(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
