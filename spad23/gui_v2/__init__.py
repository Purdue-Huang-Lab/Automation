"""SPAD23 v2 GUI (PyQt5 + pyqtgraph, white background).

pyqtgraph is pinned to PyQt5 here: PySide6 is installed as well (the spad23
GUIs use it) and pyqtgraph would otherwise prefer it.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from PyQt5 import QtWidgets  # noqa: E402,F401  (PyQt5 must be imported before pyqtgraph)
import pyqtgraph as pg  # noqa: E402

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")
