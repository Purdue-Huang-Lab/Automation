import os
import sys

# Allow running as a script inside the package directory
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# PyQt5 must be imported before pyqtgraph so pyqtgraph uses it as its Qt binding.
from PyQt5 import QtWidgets

import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k')

from measurements.pl_compare.widget import PLCompareWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL ON/OFF Comparison")
        self.resize(1800, 950)
        self.widget = PLCompareWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        try:
            self.widget.shutdown()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
