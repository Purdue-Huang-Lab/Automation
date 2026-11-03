import argparse
import sys

import flipper.gui_v2  # noqa: F401  (pins PyQt5 for pyqtgraph, sets the pg config options)

from PyQt5 import QtWidgets

from .widget import FilterFlipperWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, simulate: bool = False):
        super().__init__()
        self.setWindowTitle("Thorlabs Filter Flipper v2 (MFF101/102)")
        self.resize(1150, 650)
        self.widget = FilterFlipperWidget(simulate=simulate)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        try:
            self.widget.shutdown()
        except Exception:
            pass
        event.accept()


def main():
    ap = argparse.ArgumentParser(description="Thorlabs filter flipper GUI")
    ap.add_argument("--simulate", action="store_true", help="start with the simulated flipper selected")
    args, qt_args = ap.parse_known_args()
    app = QtWidgets.QApplication(sys.argv[:1] + qt_args)
    win = MainWindow(simulate=args.simulate)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
