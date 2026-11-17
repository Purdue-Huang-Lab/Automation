"""Gate- and power-dependent pump-probe (TA) measurement, Montana cryostat setup (v2 GUI).

Same TA acquisition as measurements/pump_probe_gate_dep (PixeLink camera, pump ON / OFF
split by the reference ROI, -1000 log10(ON/OFF), *para.txt + *sum.txt files), with a
Newport SMC100 delay stage, one ND wheel for power and the PL+TRPL v2 dual-gate sweep.

pyqtgraph is pinned to PyQt5 (PySide6 is installed as well).
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from PyQt5 import QtWidgets  # noqa: E402,F401  (PyQt5 must be imported before pyqtgraph)
import pyqtgraph as pg  # noqa: E402

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")
