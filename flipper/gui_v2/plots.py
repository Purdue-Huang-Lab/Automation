"""pyqtgraph plots of the filter flipper GUI."""

from collections import deque
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .config import HISTORY_S, MEAN_COLOR, MOVE_COLOR, POS_COLOR

MOVING_Y = 1.5  # where "in transit" samples are drawn, between the two positions


def _style(plot: pg.PlotItem) -> None:
    for ax in ("bottom", "left"):
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))
    plot.showGrid(x=True, y=True, alpha=0.25)


class PositionHistoryPlot(QtWidgets.QWidget):
    """Flipper position vs time over the last ``HISTORY_S`` seconds (transit drawn at 1.5)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(background="w")
        lay.addWidget(self.pw)
        self.plot = self.pw.getPlotItem()
        _style(self.plot)
        self.plot.setTitle("Position history")
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setYRange(0.8, 2.2, padding=0)
        self.plot.setMouseEnabled(x=True, y=False)
        self.curve = self.plot.plot(pen=pg.mkPen(POS_COLOR, width=1.6), symbol="o", symbolSize=3,
                                    symbolBrush=POS_COLOR, symbolPen=None)
        self._t: deque = deque()
        self._y: deque = deque()
        self._t0: Optional[float] = None
        self.set_labels("Position 1", "Position 2")

    def set_labels(self, label1: str, label2: str) -> None:
        self.plot.getAxis("left").setTicks([[(1.0, label1), (MOVING_Y, "moving"), (2.0, label2)]])

    def add_sample(self, t_monotonic: float, position: Optional[int]) -> None:
        if self._t0 is None:
            self._t0 = float(t_monotonic)
        t = float(t_monotonic) - self._t0
        self._t.append(t)
        self._y.append(MOVING_Y if position is None else float(position))
        while self._t and self._t[0] < t - HISTORY_S:
            self._t.popleft()
            self._y.popleft()
        self.curve.setData(np.asarray(self._t), np.asarray(self._y))
        self.plot.setXRange(max(0.0, t - HISTORY_S), max(t, 1.0), padding=0.01)

    def clear(self) -> None:
        self._t.clear()
        self._y.clear()
        self._t0 = None
        self.curve.setData([], [])


class MoveTimePlot(QtWidgets.QWidget):
    """Duration of each completed move, with the running mean."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(background="w")
        lay.addWidget(self.pw)
        self.plot = self.pw.getPlotItem()
        _style(self.plot)
        self.plot.setTitle("Move duration")
        self.plot.setLabel("bottom", "Move #")
        self.plot.setLabel("left", "Duration (s)")
        self.scatter = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(MOVE_COLOR))
        self.plot.addItem(self.scatter)
        self.mean_line = pg.InfiniteLine(angle=0, movable=False,
                                         pen=pg.mkPen(MEAN_COLOR, width=1, style=QtCore.Qt.DashLine))
        self.mean_line.hide()
        self.plot.addItem(self.mean_line, ignoreBounds=True)
        self.durations = []

    def add_move(self, duration_s: float) -> None:
        self.durations.append(float(duration_s))
        n = np.arange(1, len(self.durations) + 1)
        self.scatter.setData(n, np.asarray(self.durations))
        self.mean_line.setValue(float(np.mean(self.durations)))
        self.mean_line.show()

    def stats(self):
        """(n, mean, std, max) of the recorded durations."""
        d = np.asarray(self.durations, dtype=float)
        if d.size == 0:
            return 0, float("nan"), float("nan"), float("nan")
        return int(d.size), float(d.mean()), float(d.std()), float(d.max())

    def clear(self) -> None:
        self.durations = []
        self.scatter.setData([], [])
        self.mean_line.hide()
