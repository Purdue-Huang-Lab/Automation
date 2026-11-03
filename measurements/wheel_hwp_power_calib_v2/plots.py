"""pyqtgraph plots of the wheel + HWP v2 GUI."""

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from .calib import best_unit
from .config import BAD_COLOR, CURVE_COLOR, OK_COLOR, RANGE_COLOR


def _style(plot: pg.PlotItem) -> None:
    for ax in ("bottom", "left"):
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))
    plot.showGrid(x=True, y=True, alpha=0.25)


class CurvePlot(QtWidgets.QWidget):
    """Power (± std) vs stage angle; the power unit follows the data."""

    def __init__(self, title: str, x_label: str, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(background="w")
        lay.addWidget(self.pw)
        self.plot = self.pw.getPlotItem()
        _style(self.plot)
        self.plot.setTitle(title)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", "Power")
        self.err = pg.ErrorBarItem(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0),
                                   pen=pg.mkPen(CURVE_COLOR, width=0.8), beam=0.0)
        self.plot.addItem(self.err)
        self.curve = self.plot.plot(pen=pg.mkPen(CURVE_COLOR, width=1.2), symbol="o", symbolSize=4,
                                    symbolBrush=CURVE_COLOR, symbolPen=None)

    def set_title(self, title: str) -> None:
        self.plot.setTitle(title)

    def set_data(self, angles: Sequence[float], powers_w: Sequence[float], stds_w: Sequence[float]) -> None:
        a = np.asarray(angles, dtype=float)
        p = np.asarray(powers_w, dtype=float)
        s = np.nan_to_num(np.asarray(stds_w, dtype=float), nan=0.0)
        if a.size == 0:
            self.clear()
            return
        finite = p[np.isfinite(p)]
        unit, scale = best_unit(float(np.max(np.abs(finite))) if finite.size else 0.0)
        self.plot.setLabel("left", f"Power ({unit})")
        self.curve.setData(a, p / scale)
        self.err.setData(x=a, y=p / scale, height=2.0 * s / scale)

    def clear(self) -> None:
        self.curve.setData([], [])
        self.err.setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
        self.plot.setLabel("left", "Power")


class CoveragePlot(QtWidgets.QWidget):
    """Reachable power (log) per ND angle: a bar from the HWP minimum to its maximum, plus solved points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(background="w")
        lay.addWidget(self.pw)
        self.plot = self.pw.getPlotItem()
        _style(self.plot)
        self.plot.setTitle("Reachable power per ND angle")
        self.plot.setLabel("bottom", "ND wheel angle (°)")
        self.plot.setLabel("left", "Power (W)")
        self.plot.setLogMode(x=False, y=True)
        self.plot.getAxis("left").enableAutoSIPrefix(False)  # log ticks read 1e-08 ... in W
        self.plot.addLegend(offset=(-10, 10))
        self.bars = pg.PlotDataItem(pen=pg.mkPen(RANGE_COLOR, width=5), connect="pairs", name="HWP range")
        self.plot.addItem(self.bars)
        self.ok_pts = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7, symbolBrush=OK_COLOR,
                                      symbolPen=None, name="planned")
        self.bad_pts = pg.PlotDataItem(pen=None, symbol="x", symbolSize=9, symbolBrush=BAD_COLOR,
                                       symbolPen=pg.mkPen(BAD_COLOR), name="outside tolerance")
        self.plot.addItem(self.ok_pts)
        self.plot.addItem(self.bad_pts)

    def set_ranges(self, nd_deg: Sequence[float], p_min: Sequence[float], p_max: Sequence[float]) -> None:
        nd = np.asarray(nd_deg, dtype=float)
        x = np.repeat(nd, 2)
        y = np.column_stack([np.asarray(p_min, dtype=float), np.asarray(p_max, dtype=float)]).ravel()
        self.bars.setData(x, y)
        self.set_points([], [], None)

    def set_points(self, nd_deg: Sequence[float], power_w: Sequence[float], ok: Optional[Sequence[bool]]) -> None:
        nd = np.asarray(nd_deg, dtype=float)
        p = np.asarray(power_w, dtype=float)
        good = np.asarray(ok, dtype=bool) if ok is not None else np.ones(nd.shape, dtype=bool)
        self.ok_pts.setData(nd[good], p[good])
        self.bad_pts.setData(nd[~good], p[~good])

    def clear(self) -> None:
        self.bars.setData([], [])
        self.set_points([], [], None)
