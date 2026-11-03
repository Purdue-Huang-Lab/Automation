"""pyqtgraph plots of the PL + TRPL v2 GUI that the SPAD23 v2 GUI has no counterpart for.

The SPAD plots (count maps, 2D Gaussian fit, radial fit, histogram traces) are
the spad23.gui_v2 widgets; these follow their style (spad23 ``_PlotBase``:
black axes, centered grey message).
"""

from collections import deque
from typing import Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from keithley.gui.config import PLOT_MAX_POINTS
from spad23.gui_v2.plots import _PlotBase, set_title
from measurements.pl_trpl_power_voltage_v2.config import GATE_LINE_COLOR, LIFETIME_CMAP, LINECUT_CMAP

Qt = QtCore.Qt

Extent = Tuple[float, float, float, float]  # matplotlib order: (x0, x1, y0, y1)


def _black_axis(axis: pg.AxisItem) -> None:
    axis.setPen(pg.mkPen("k"))
    axis.setTextPen(pg.mkPen("k"))


class _ImageMapPlot(_PlotBase):
    """An image placed over a matplotlib-style extent (origin at the bottom), with a colorbar.

    ``setRect`` scales relative to the current image shape, so it is applied after
    every ``setImage``.  The view is reset to the extent on every update (matplotlib's
    ``set_extent``), and a reversed extent shows an inverted axis as in matplotlib.
    NaN cells are transparent.
    """

    def __init__(self, cmap: str, parent=None):
        super().__init__(parent)
        self.plot.showGrid(x=False, y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.plot.addItem(self.img)
        self.cmap = pg.colormap.get(cmap)
        self.cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=self.cmap, interactive=False, width=14,
                                    colorMapMenu=False, pen="k")
        self.cbar.setImageItem(self.img, insert_in=self.plot)
        for ax in ("left", "right", "top", "bottom"):
            _black_axis(self.cbar.getAxis(ax))
        self._extent: Extent = (-0.5, 9.5, -0.5, 9.5)

    def _place(self, extent: Extent) -> None:
        x0, x1, y0, y1 = (float(v) for v in extent)
        self._extent = (x0, x1, y0, y1)
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.plot.invertX(x1 < x0)
        self.plot.invertY(y1 < y0)
        self.plot.setXRange(min(x0, x1), max(x0, x1), padding=0)
        self.plot.setYRange(min(y0, y1), max(y0, y1), padding=0)

    def _show(self, data: np.ndarray, extent: Extent) -> None:
        self.img.setImage(np.asarray(data, dtype=np.float64), autoLevels=False)
        self._place(extent)
        self.img.show()

    def set_levels(self, lo: float, hi: float) -> None:
        self.cbar.setLevels((float(lo), float(hi)))
        self.img.setLevels((float(lo), float(hi)))


class LinecutMapPlot(_ImageMapPlot):
    """Linecut vs power / voltage (viridis).

    ``show_placeholder`` mirrors v1's ``set_data(zeros((10, 10)))``: the zeros fill
    the last extent and the levels, title and labels are kept.
    """

    def __init__(self, title: str, xlabel: str, ylabel: str, parent=None):
        super().__init__(LINECUT_CMAP, parent)
        self._show(np.zeros((10, 10)), self._extent)
        self.set_levels(0.0, 1.0)
        self.set_labels(title, xlabel, ylabel)

    def set_labels(self, title: str, xlabel: str, ylabel: str) -> None:
        set_title(self.plot, title)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)

    def set_image(self, data: np.ndarray, extent: Extent, title: str, xlabel: str, ylabel: str) -> None:
        self._show(data, extent)
        self.set_labels(title, xlabel, ylabel)

    def show_placeholder(self) -> None:
        self._show(np.zeros((10, 10)), self._extent)


class LifetimeMapPlot(_ImageMapPlot):
    """Normalized decay columns vs power / voltage (inferno).

    Log mode shows log10 of the normalized data on levels (-3, 0), which is
    matplotlib's ``LogNorm(vmin=1e-3, vmax=1)``; linear mode uses levels (0, 1).
    Blank until the first update (v1 never draws it in ``__init__``).
    """

    def __init__(self, parent=None):
        super().__init__(LIFETIME_CMAP, parent)
        self.img.hide()
        self.cbar.hide()

    def set_image(self, ndata: np.ndarray, extent: Extent, log_mode: bool, title: str, xlabel: str,
                  ylabel: str) -> None:
        z = np.asarray(ndata, dtype=np.float64)
        if log_mode:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log10(z)
        self._show(z, extent)
        self.set_levels(*((-3.0, 0.0) if log_mode else (0.0, 1.0)))
        self.cbar.getAxis("left").setLabel("log10(normalized)" if log_mode else "Normalized")
        self.cbar.show()
        set_title(self.plot, title)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        self.show_message("")

    def clear_message(self, text: str) -> None:
        """v1's ``ax.clear()`` + centered text: no image, title or labels."""
        self.img.hide()
        self.cbar.hide()
        self.plot.setTitle(None)
        self.plot.setLabel("bottom", None)
        self.plot.setLabel("left", None)
        self.plot.invertX(False)
        self.plot.invertY(False)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0)
        self.show_message(text)


class GateVIPlot(QtWidgets.QWidget):
    """V and I (nA) of one gate vs sample index, on two y axes (v1 keithley.gui.plots.PlotWidget).

    Both curves are matplotlib C0 (I dashed); the x range is (0, max(1, n-1)) and
    both y axes autoscale with matplotlib's 5% margins.  Static, like v1.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(background="w")
        self.pw.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.pw)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        plot = self.pw.getPlotItem()
        plot.setLabel("bottom", "Samples")
        plot.setLabel("left", "V (V)")
        plot.showAxis("right")
        plot.setLabel("right", "I (nA)")
        for ax in ("bottom", "left", "right"):
            _black_axis(plot.getAxis(ax))
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)
        plot.hideButtons()

        # Second ViewBox for I on the right axis (pattern from keithley/gui_v2/plots.py)
        self.vb2 = pg.ViewBox()
        self.vb2.setMouseEnabled(x=False, y=False)
        self.vb2.setMenuEnabled(False)
        self.pw.scene().addItem(self.vb2)
        plot.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(plot)
        plot.vb.sigResized.connect(self._sync_views)
        for vb in (plot.vb, self.vb2):
            vb.setDefaultPadding(0.05)
            vb.enableAutoRange(axis="y", enable=True)

        self.curveV = plot.plot(pen=pg.mkPen(GATE_LINE_COLOR, width=1.5))
        self.curveI = pg.PlotCurveItem(pen=pg.mkPen(GATE_LINE_COLOR, width=1.2, style=Qt.DashLine))
        self.vb2.addItem(self.curveI)

        self.vs: deque = deque(maxlen=PLOT_MAX_POINTS)
        self.is_nA: deque = deque(maxlen=PLOT_MAX_POINTS)
        self._plot = plot
        self._redraw()

    def _sync_views(self) -> None:
        self.vb2.setGeometry(self._plot.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(self._plot.vb, self.vb2.XAxis)

    def add_point(self, v_volts: float, i_amps: float) -> None:
        self.vs.append(v_volts)
        self.is_nA.append(i_amps * 1e9)
        self._redraw()

    def clear(self) -> None:
        self.vs.clear()
        self.is_nA.clear()
        self._redraw()

    def _redraw(self) -> None:
        n = len(self.vs)
        x = np.arange(n, dtype=np.float64)
        self.curveV.setData(x, np.asarray(self.vs, dtype=np.float64))
        self.curveI.setData(x, np.asarray(self.is_nA, dtype=np.float64))
        self._plot.setXRange(0.0, float(max(1, n - 1)), padding=0)
