from collections import deque

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .config import PLOT_MAX_POINTS


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.pw = pg.PlotWidget(background="w")
        self.pw.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.pw)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Left axis: voltage
        self.pw.setLabel("left", "V", units="V")
        self.pw.setLabel("bottom", "Samples")
        self.pw.showAxis("right")
        self.pw.setLabel("right", "I (nA)")
        for ax in ("bottom", "left", "right"):
            self.pw.getAxis(ax).setPen(pg.mkPen("k"))
            self.pw.getAxis(ax).setTextPen(pg.mkPen("k"))

        # Second ViewBox for I (right axis)
        self.vb2 = pg.ViewBox()
        self.pw.scene().addItem(self.vb2)
        self.pw.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(self.pw.plotItem)
        self.pw.plotItem.vb.sigResized.connect(self._sync_views)

        self.curveV = self.pw.plot(pen=pg.mkPen("steelblue", width=1.5))
        self.curveI = pg.PlotCurveItem(pen=pg.mkPen("tomato", width=1.2, style=QtCore.Qt.DashLine))
        self.vb2.addItem(self.curveI)

        self.vs: deque = deque(maxlen=PLOT_MAX_POINTS)
        self.is_nA: deque = deque(maxlen=PLOT_MAX_POINTS)

    def _sync_views(self):
        self.vb2.setGeometry(self.pw.plotItem.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(self.pw.plotItem.vb, self.vb2.XAxis)

    def add_point(self, v_volts: float, i_amps: float):
        self.vs.append(v_volts)
        self.is_nA.append(i_amps * 1e9)
        self._redraw()

    def clear(self):
        self.vs.clear()
        self.is_nA.clear()
        self._redraw()

    def _redraw(self):
        x = list(range(len(self.vs)))
        self.curveV.setData(x, list(self.vs))
        self.curveI.setData(x, list(self.is_nA))


class LeakageIVPlot(QtWidgets.QWidget):
    """Mean current +/- std per voltage bin, for the leakage test.

    Redrawn on demand (Update button / auto-update timer), never per sample.
    No log-|I| mode: pg.ErrorBarItem is a raw GraphicsObject that ignores
    setLogMode, so the bars would land in the wrong place.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.pw = pg.PlotWidget(background="w")
        self.pw.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.pw)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.pw.setLabel("bottom", "V", units="V")
        self.pw.setLabel("left", "I (nA)")
        self.pw.showGrid(x=True, y=True, alpha=0.25)
        for ax in ("bottom", "left"):
            self.pw.getAxis(ax).setPen(pg.mkPen("k"))
            self.pw.getAxis(ax).setTextPen(pg.mkPen("k"))

        self.errItem = pg.ErrorBarItem(pen=pg.mkPen("k", width=1))
        self.pw.addItem(self.errItem)

        self.meanCurve = self.pw.plot(pen=pg.mkPen("tomato", width=1.2))
        self.meanScatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush("tomato"), size=6)
        self.pw.addItem(self.meanScatter)

    def set_data(self, v_volts, mean_amps, std_amps):
        # ErrorBarItem.setData interweaves arrays internally: lists break it.
        v = np.asarray(v_volts, dtype=float)
        m = np.asarray(mean_amps, dtype=float) * 1e9
        e = np.asarray(std_amps, dtype=float) * 1e9

        if v.size == 0:
            self.clear()
            return

        ok = np.isfinite(v) & np.isfinite(m) & np.isfinite(e)
        v, m, e = v[ok], m[ok], e[ok]
        if v.size == 0:
            self.clear()
            return

        # Beam width tracks the bin width so caps stay proportionate.
        beam = 0.005
        if v.size > 1:
            spacing = float(np.median(np.diff(v)))
            beam = max(1e-3, 0.25 * spacing)

        self.errItem.setData(x=v, y=m, top=e, bottom=e, beam=beam)
        self.meanCurve.setData(v, m)
        self.meanScatter.setData(v, m)

    def clear(self):
        self.errItem.setData(x=None, y=None)
        self.meanCurve.setData([], [])
        self.meanScatter.setData([], [])
