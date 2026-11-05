import argparse
import os
import sys

# Allow running as a script inside the package directory
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# The package __init__ pins PyQt5, imports it before pyqtgraph and sets the pg config options.
import measurements.wheel_hwp_power_calib_v2  # noqa: E402,F401

from PyQt5 import QtWidgets  # noqa: E402

from measurements.pl_trpl_power_voltage_v2.ui import apply_ui_font, enable_high_dpi, scale_fixed_sizes  # noqa: E402
from measurements.wheel_hwp_power_calib_v2.widget import WheelHWPCalibWidget  # noqa: E402


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheel + HWP Power Calibration v2")
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, int(avail.width() * 0.95)), min(950, int(avail.height() * 0.92)))
        self.widget = WheelHWPCalibWidget()
        scale_fixed_sizes(self.widget)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        try:
            self.widget.shutdown()
        except Exception:
            pass
        event.accept()


def main():
    ap = argparse.ArgumentParser(description="Wheel + HWP power calibration v2")
    ap.add_argument("calib", nargs="?", help="calibration CSV (or ND-only .xlsx) to load on start")
    args, qt_args = ap.parse_known_args()
    enable_high_dpi()
    app = QtWidgets.QApplication(sys.argv[:1] + qt_args)
    apply_ui_font(app)
    win = MainWindow()
    win.show()
    if args.calib:
        win.widget.load_file(args.calib)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
