import sys

from PyQt5 import QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")

from .dual_widget import DualKeithleyWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Keithley Controller v2 (Safe ramps, microstep plotting, leakage test)")
        self.resize(1400, 820)

        self.widget = DualKeithleyWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        # Qt delivers the close event to the window, not to its children, so the
        # central widget's own closeEvent (stop timer, close logs, disconnect
        # SMUs) only runs if we forward it.
        try:
            self.widget.close()
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
