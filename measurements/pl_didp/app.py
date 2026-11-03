import os
import sys

# Allow running as a script inside the package directory
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# PyQt5 must be imported before pyqtgraph so pyqtgraph uses it as its Qt binding
# (PySide6 is installed too).
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
from PyQt5 import QtWidgets

import pyqtgraph as pg
# imageAxisOrder stays at pyqtgraph's default (col-major): the SingleFrameView
# reused from pl_power_gate_dep_viewer_v2 transposes images for that order.
pg.setConfigOptions(background="w", foreground="k")

from measurements.pl_didp.ui import apply_ui_font, enable_high_dpi
from measurements.pl_didp.widget import PLdIdPWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL / TRPL two-pump dI/dP viewer")
        # 1900 × 1000, but never larger than the screen (at 150 % a 1440p screen is 1707 × 960)
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1900, int(avail.width() * 0.95)), min(1000, int(avail.height() * 0.92)))
        self.widget = PLdIdPWidget()
        self.setCentralWidget(self.widget)


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
