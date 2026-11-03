"""Filter flipper v2 GUI (PyQt5 + pyqtgraph, white background)."""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from PyQt5 import QtWidgets  # noqa: E402,F401  (PyQt5 must be imported before pyqtgraph)
import pyqtgraph as pg  # noqa: E402

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")
