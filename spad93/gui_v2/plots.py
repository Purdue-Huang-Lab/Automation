"""pyqtgraph x-y plot widgets for the SPAD93 GUI (white background, black axes)."""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from spad93.spad93_analysis import (
    CalibPoint,
    DoubleFitResult,
    FWHM_FACTOR,
    Gauss1DResult,
    WavelengthCalibration,
    gauss1d,
)
from spad93.spad93_geometry import COLUMN_X_UM, NUM_PIXELS, PITCH_X_UM, PIXEL_X_UM, PIXEL_Y_UM, X_SPAN_UM, nearest_column
from spad93.gui_v2.config import (
    COMPONENT_COLORS,
    DATA_COLOR,
    FIT_LINE_COLOR,
    REF_COLOR,
    SELECT_COLOR,
    TRACE_COLORS,
)

Qt = QtCore.Qt


def style_plot(plot: pg.PlotItem, grid: bool = True) -> None:
    """Black axis pens/text (CLAUDE.md v2) and a light grid."""
    for ax in ("bottom", "left", "top", "right"):
        axis = plot.getAxis(ax)
        axis.setPen(pg.mkPen("k"))
        axis.setTextPen(pg.mkPen("k"))
    if grid:
        plot.showGrid(x=True, y=True, alpha=0.25)


def set_title(plot: pg.PlotItem, title: str, subtitle: str = "") -> None:
    html = f"<span style='color:#111827;font-size:10pt'>{title}</span>"
    if subtitle:
        html += f"<br><span style='color:#374151;font-size:8pt'>{subtitle}</span>"
    plot.setTitle(html)


class WavelengthAxisItem(pg.AxisItem):
    """Axis whose view coordinate is x (um) but whose ticks show wavelength.

    Ticks sit at round wavelengths of the linear calibration; without a
    calibration it is an ordinary x (um) axis.
    """

    def __init__(self, orientation: str = "bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self._calib: Optional[WavelengthCalibration] = None
        self.enableAutoSIPrefix(False)
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def set_calibration(self, calib: Optional[WavelengthCalibration], label_um: str = "x (µm)") -> None:
        self._calib = calib if (calib is not None and calib.slope_nm_per_um != 0) else None
        self.setLabel("Wavelength (nm)" if self._calib is not None else label_um)
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        c = self._calib
        if c is None:
            return super().tickValues(minVal, maxVal, size)
        l0, l1 = sorted(float(v) for v in c.wavelength([minVal, maxVal]))
        k = abs(c.slope_nm_per_um)
        return [(spacing / k, [float(c.x_of(v)) for v in values])
                for spacing, values in super().tickValues(l0, l1, size)]

    def tickStrings(self, values, scale, spacing):
        c = self._calib
        if c is None:
            return super().tickStrings(values, scale, spacing)
        dl = spacing * abs(c.slope_nm_per_um)
        places = max(0, int(math.ceil(-math.log10(dl)))) if dl > 0 else 1
        return [f"{float(c.wavelength(v)):.{places}f}" for v in values]


class _PlotBase(QtWidgets.QWidget):
    """A ``pg.PlotWidget`` plus a centered message overlay ("Fit unavailable", ...)."""

    def __init__(self, parent=None, axis_items: Optional[Dict[str, pg.AxisItem]] = None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget(axisItems=axis_items)
        self.plot: pg.PlotItem = self.pw.getPlotItem()
        style_plot(self.plot)
        lay.addWidget(self.pw)
        self._overlay = QtWidgets.QLabel(self.pw)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("color:#6b7280; background: transparent; font-size: 10pt;")
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._overlay.hide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._overlay.setGeometry(self.pw.rect())

    def show_message(self, text: str) -> None:
        self._overlay.setText(text)
        self._overlay.setGeometry(self.pw.rect())
        self._overlay.setVisible(bool(text))
        self._overlay.raise_()

    def set_title(self, title: str, subtitle: str = "") -> None:
        set_title(self.plot, title, subtitle)


class RadialFitPlot(_PlotBase):
    """Intensity versus distance from the fitted center, with the Gaussian fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "Distance from fitted center (µm)")
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(37, 99, 235, 190))
        self.plot.addItem(self.scatter)
        self.line = self.plot.plot(pen=pg.mkPen("#111827", width=1.7))

    def set_data(self, r, z, r_line, z_line, title: str, subtitle: str, ylabel: str) -> None:
        self.scatter.setData(x=np.asarray(r, float), y=np.asarray(z, float))
        self.line.setData(np.asarray(r_line, float), np.asarray(z_line, float))
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("")

    def clear(self, message: str, title: str = "Gaussian radial fit") -> None:
        self.scatter.setData([], [])
        self.line.setData([], [])
        self.set_title(title)
        self.show_message(message)


class DoubleRadialPlot(_PlotBase):
    """The double 2D Gaussian as two 1D radial profiles (a single radial plot needs one center).

    For Gaussian k every enabled pixel is plotted at its distance from center k
    (mirrored to +-r, as in the single-Gaussian radial plot) with the value minus the
    other Gaussian and the offset, so it lies on A_k exp(-r^2 / 2 sigma_k^2).  Points
    are filled where Gaussian k is the larger of the two at that pixel and hollow where
    the other one dominates (their value is mostly what the subtraction left).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "Distance from the Gaussian's center (µm)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.items = []
        for k, color in enumerate(COMPONENT_COLORS):
            c = pg.mkColor(color)
            line = self.plot.plot(pen=pg.mkPen(color, width=1.8), name=f"Gaussian {k + 1}")
            own = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(c.red(), c.green(), c.blue(), 210))
            other = pg.ScatterPlotItem(size=7, pen=pg.mkPen(color, width=1.0), brush=pg.mkBrush(None))
            self.plot.addItem(own)
            self.plot.addItem(other)
            self.items.append((line, own, other))
        self.zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen("#9ca3af", width=1, style=Qt.DotLine))
        self.plot.addItem(self.zero, ignoreBounds=True)

    def set_data(self, values, fit: Optional[DoubleFitResult], excluded, ylabel: str, title: str,
                 subtitle: str = "", message: str = "Fit unavailable") -> None:
        if fit is None:
            for line, own, other in self.items:
                line.setData([], [])
                own.setData([], [])
                other.setData([], [])
            self.legend.clear()
            self.plot.setLabel("left", ylabel)
            self.set_title(title, subtitle)
            self.show_message(message)
            return
        v = np.asarray(values, dtype=np.float64)
        ok = np.ones(NUM_PIXELS, dtype=bool)
        ok[[int(q) for q in excluded]] = False
        x, y, z = PIXEL_X_UM[ok], PIXEL_Y_UM[ok], v[ok]
        comps = []
        for k in range(2):
            A, x0, y0, sig = fit.component(k)
            r = np.hypot(x - x0, y - y0)
            comps.append((A, abs(sig), r, A * np.exp(-(r**2) / (2.0 * max(abs(sig), 1e-6) ** 2))))
        rmax = max(float(np.max(c[2])) for c in comps) if z.size else 1.0
        r_line = np.linspace(-rmax, rmax, 401)
        self.legend.clear()
        for k, (line, own, other) in enumerate(self.items):
            A, sig, r, model_k = comps[k]
            rest = z - comps[1 - k][3] - fit.offset  # this Gaussian's share of the data
            mine = model_k >= comps[1 - k][3]
            line.setData(r_line, A * np.exp(-(r_line**2) / (2.0 * max(sig, 1e-6) ** 2)))
            own.setData(np.concatenate([-r[mine], r[mine]]), np.concatenate([rest[mine], rest[mine]]))
            other.setData(np.concatenate([-r[~mine], r[~mine]]), np.concatenate([rest[~mine], rest[~mine]]))
            self.legend.addItem(line, f"Gaussian {k + 1}: σ = {sig:.1f} µm")
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("")


class LinecutPlot(_PlotBase):
    """'-o' line plot used for linecuts."""

    def __init__(self, color: str = DATA_COLOR, parent=None):
        super().__init__(parent)
        self.curve = self.plot.plot(
            pen=pg.mkPen(color, width=1.4), symbol="o", symbolSize=6, symbolBrush=color, symbolPen=None,
            connect="finite",
        )

    def set_data(self, x, y, xlabel: str, ylabel: str, title: str, subtitle: str = "") -> None:
        self.curve.setData(np.asarray(x, float), np.asarray(y, float))
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("" if np.isfinite(np.asarray(y, float)).any() else "No data")


class SpectrumPlot(_PlotBase):
    """Center-row intensity versus x (um) with a wavelength axis; click selects a column."""

    column_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.axis = WavelengthAxisItem("bottom")
        super().__init__(parent, axis_items={"bottom": self.axis})
        self.axis.set_calibration(None)
        self.curve = self.plot.plot(
            pen=pg.mkPen(DATA_COLOR, width=1.4), symbol="o", symbolSize=6, symbolBrush=DATA_COLOR,
            symbolPen=None, connect="finite",
        )
        self.fit_curve = self.plot.plot(pen=pg.mkPen(FIT_LINE_COLOR, width=1.6, style=Qt.DashLine))
        self.sel_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(SELECT_COLOR, width=2))
        self.sel_line.hide()
        self.plot.addItem(self.sel_line, ignoreBounds=True)
        self.plot.setXRange(-0.5 * PITCH_X_UM, X_SPAN_UM + 0.5 * PITCH_X_UM, padding=0.02)
        self.pw.scene().sigMouseClicked.connect(self._on_click)

    def set_calibration(self, calib: Optional[WavelengthCalibration]) -> None:
        self.axis.set_calibration(calib)

    def set_data(self, values, fit: Optional[Gauss1DResult], ylabel: str, title: str, subtitle: str = "") -> None:
        vals = np.asarray(values, dtype=np.float64)
        self.curve.setData(COLUMN_X_UM, vals)
        if fit is not None:
            xs = np.linspace(-0.5 * PITCH_X_UM, X_SPAN_UM + 0.5 * PITCH_X_UM, 400)
            self.fit_curve.setData(xs, gauss1d(xs, fit.amp, fit.center, fit.sigma, fit.offset))
        else:
            self.fit_curve.setData([], [])
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("" if np.isfinite(vals).any() else "No data")

    def set_selected_column(self, col: Optional[int]) -> None:
        if col is None:
            self.sel_line.hide()
        else:
            self.sel_line.setValue(float(COLUMN_X_UM[int(col)]))
            self.sel_line.show()

    def _on_click(self, ev) -> None:
        if ev.double() or ev.button() != Qt.LeftButton:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.column_clicked.emit(nearest_column(vb.mapSceneToView(ev.scenePos()).x()))


class ColumnProfilePlot(_PlotBase):
    """y-linecut of one column with its 1D Gaussian fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "y (µm)")
        self.points = self.plot.plot(
            pen=pg.mkPen(DATA_COLOR, width=1.0), symbol="o", symbolSize=7, symbolBrush=DATA_COLOR, symbolPen=None
        )
        self.fit_curve = self.plot.plot(pen=pg.mkPen(FIT_LINE_COLOR, width=1.7))

    def set_data(self, y_um, z, fit: Optional[Gauss1DResult], title: str, subtitle: str, ylabel: str) -> None:
        y_um = np.asarray(y_um, float)
        self.points.setData(y_um, np.asarray(z, float))
        if fit is not None and y_um.size:
            pad = 23.0
            ys = np.linspace(min(y_um.min(), fit.center) - pad, max(y_um.max(), fit.center) + pad, 300)
            self.fit_curve.setData(ys, fit.evaluate(ys))
        else:
            self.fit_curve.setData([], [])
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("" if y_um.size else "No enabled pixels in this column")



class MultiProfilePlot(_PlotBase):
    """y-profiles of several columns with their Gaussian fits (one color per column).

    Rejected fits are drawn dashed; columns without a fit show the points only.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "y (µm)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self._items = []

    def set_profiles(self, profiles, title: str, subtitle: str, ylabel: str) -> None:
        """``profiles``: (label, y_um, values, fit or None, color, accepted)."""
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []
        self.legend.clear()
        for label, y, z, fit, color, accepted in profiles:
            y = np.asarray(y, float)
            pts = self.plot.plot(y, np.asarray(z, float), pen=None, symbol="o", symbolSize=7,
                                 symbolBrush=color, symbolPen=None, name=label)
            self._items.append(pts)
            if fit is not None and y.size:
                ys = np.linspace(min(y.min(), fit.center) - 23.0, max(y.max(), fit.center) + 23.0, 300)
                style = Qt.SolidLine if accepted else Qt.DashLine
                line = self.plot.plot(ys, fit.evaluate(ys), pen=pg.mkPen(color, width=1.6, style=style))
                self._items.append(line)
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.show_message("" if profiles else "Select columns: left-click pixels on the map")


class MultiTracePlot(_PlotBase):
    """One fit parameter versus time for several columns, with a time cursor."""

    time_clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "Time (ns)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#111827", width=1, style=Qt.DashLine))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self._items = []
        self.pw.scene().sigMouseClicked.connect(self._on_click)

    def set_series(self, series, ylabel: str, title: str, subtitle: str = "", message: str = "") -> None:
        """``series``: (label, t, y, color); only the points given are drawn."""
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []
        self.legend.clear()
        for label, t, y, color in series:
            item = self.plot.plot(np.asarray(t, float), np.asarray(y, float), pen=pg.mkPen(color, width=1.1),
                                  symbol="o", symbolSize=5, symbolBrush=color, symbolPen=None, name=label)
            self._items.append(item)
        self.plot.setLabel("left", ylabel)
        self.set_title(title, subtitle)
        self.plot.enableAutoRange(enable=True)
        self.show_message(message)

    def set_cursor(self, t_ns: float) -> None:
        self.cursor.setValue(float(t_ns))

    def _on_click(self, ev) -> None:
        if ev.double() or ev.button() != Qt.LeftButton:
            return
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(ev.scenePos()):
            self.time_clicked.emit(float(vb.mapSceneToView(ev.scenePos()).x()))

class HistogramPlot(_PlotBase):
    """Decay traces with a time cursor, a reference line and the selected range."""

    time_clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setLabel("bottom", "Time (ns)")
        self.plot.setLabel("left", "Counts per bin")
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(191, 219, 254, 50), pen=pg.mkPen(None))
        self.region.setZValue(-20)
        self.plot.addItem(self.region, ignoreBounds=True)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#111827", width=1, style=Qt.DashLine))
        self.ref_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(REF_COLOR, width=1.4, style=Qt.DashDotLine))
        self.ref_line.hide()
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.plot.addItem(self.ref_line, ignoreBounds=True)
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._log = False
        self.pw.scene().sigMouseClicked.connect(self._on_click)

    def set_traces(self, traces: Sequence[Tuple[str, np.ndarray, np.ndarray, str]]) -> None:
        keep = set()
        for label, x, y, color in traces:
            keep.add(label)
            yy = np.asarray(y, dtype=np.float64)
            if self._log:
                yy = np.where(yy > 0, yy, np.nan)
            curve = self._curves.get(label)
            if curve is None:
                curve = self.plot.plot(pen=pg.mkPen(color, width=1.2), name=label, connect="finite")
                self._curves[label] = curve
            curve.setData(np.asarray(x, dtype=np.float64), yy)
        for label in list(self._curves):
            if label not in keep:
                self.plot.removeItem(self._curves.pop(label))
                self.legend.removeItem(label)
        self.show_message("" if traces else "Left-click the map to add traces")

    def set_log(self, on: bool) -> None:
        if on != self._log:
            self._log = on
            self.plot.setLogMode(x=False, y=on)

    def set_cursor(self, t_ns: float) -> None:
        self.cursor.setValue(float(t_ns))

    def set_reference(self, t_ns: Optional[float]) -> None:
        if t_ns is None:
            self.ref_line.hide()
        else:
            self.ref_line.setValue(float(t_ns))
            self.ref_line.show()

    def set_region(self, t0: float, t1: float) -> None:
        self.region.setRegion((float(t0), float(t1)))
        if t1 > t0:
            self.plot.setXRange(float(t0), float(t1), padding=0.01)

    def _on_click(self, ev) -> None:
        if ev.double() or ev.button() != Qt.LeftButton:
            return
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(ev.scenePos()):
            self.time_clicked.emit(float(vb.mapSceneToView(ev.scenePos()).x()))


class TracePlot(QtWidgets.QWidget):
    """One or two stacked x-linked traces (fit parameter versus time).

    Flagged points (fit at a bound or low signal) are drawn hollow.  Each panel
    can hold a second series (``y_b`` / ``y2_b``, e.g. the second Gaussian of a
    double fit); both are then drawn in COMPONENT_COLORS with a legend.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.p1 = self.glw.addPlot(row=0, col=0)
        self.p2 = self.glw.addPlot(row=1, col=0)
        self.p2.setXLink(self.p1)
        self.items = []
        self.items_b = []
        self.legends = []
        color_b = COMPONENT_COLORS[1]
        for p in (self.p1, self.p2):
            style_plot(p)
            line = p.plot(pen=pg.mkPen(DATA_COLOR, width=1.1))
            good = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(DATA_COLOR))
            bad = pg.ScatterPlotItem(size=6, pen=pg.mkPen(DATA_COLOR, width=1.2), brush=pg.mkBrush(None))
            cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#111827", width=1, style=Qt.DashLine))
            p.addItem(good)
            p.addItem(bad)
            p.addItem(cursor, ignoreBounds=True)
            self.items.append((line, good, bad, cursor))
            items_b = (p.plot(pen=pg.mkPen(color_b, width=1.1)),
                       pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(color_b)),
                       pg.ScatterPlotItem(size=6, pen=pg.mkPen(color_b, width=1.2), brush=pg.mkBrush(None)))
            p.addItem(items_b[1])
            p.addItem(items_b[2])
            self.items_b.append(items_b)
            self.legends.append(p.addLegend(offset=(-10, 10)))
        self.legend = self.legends[0]
        self.p2.hide()
        self.p1.setLabel("bottom", "Time (ns)")

    @staticmethod
    def _set_points(items, t, y, flagged, color: str) -> None:
        line, good, bad = items[:3]
        line.setPen(pg.mkPen(color, width=1.1))
        good.setBrush(pg.mkBrush(color))
        bad.setPen(pg.mkPen(color, width=1.2))
        # The line joins only good fits; flagged ones (noise bins, bound hits) stay hollow and
        # do not set the y range.
        line.setData(t[~flagged], y[~flagged])
        good.setData(t[~flagged], y[~flagged])
        bad.setData(t[flagged], y[flagged])

    def _fill(self, k: int, t, y, flagged, y_b=None, labels=("", "")) -> None:
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        flagged = np.asarray(flagged, bool) if flagged is not None else np.zeros(t.size, bool)
        two = y_b is not None
        self._set_points(self.items[k], t, y, flagged, COMPONENT_COLORS[0] if two else DATA_COLOR)
        plot = self.p1 if k == 0 else self.p2
        yg = y[~flagged & np.isfinite(y)]
        yb = np.asarray(y_b, float) if two else np.zeros(0)
        self._set_points(self.items_b[k], t if two else np.zeros(0), yb, flagged if two else np.zeros(0, bool),
                         COMPONENT_COLORS[1])
        legend = self.legends[k]
        legend.clear()
        if two:
            yg = np.concatenate([yg, yb[~flagged & np.isfinite(yb)]])
            legend.addItem(self.items[k][0], labels[0])
            legend.addItem(self.items_b[k][0], labels[1])
        if yg.size >= 2:
            lo, hi = float(np.percentile(yg, 1)), float(np.percentile(yg, 99))
            pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-6)
            plot.enableAutoRange(axis="y", enable=False)
            plot.setYRange(lo - pad, hi + pad, padding=0)
            plot.enableAutoRange(axis="x", enable=True)
        else:
            plot.enableAutoRange(enable=True)

    def set_series(self, t, y, flagged, ylabel: str, title: str, y2=None, ylabel2: str = "", subtitle: str = "",
                   y_b=None, label_a: str = "", label_b: str = "",
                   y2_b=None, label2_a: str = "", label2_b: str = "") -> None:
        self._fill(0, t, y, flagged, y_b, (label_a, label_b))
        self.p1.setLabel("left", ylabel)
        set_title(self.p1, title, subtitle)
        if y2 is None:
            self._fill(1, [], [], [])
            self.p2.hide()
            self.p1.setLabel("bottom", "Time (ns)")
        else:
            self.p2.show()  # before the legend is filled: legend entries of hidden curves are drawn grey
            self._fill(1, t, y2, flagged, y2_b, (label2_a, label2_b))
            self.p2.setLabel("left", ylabel2)
            self.p2.setLabel("bottom", "Time (ns)")
            self.p1.setLabel("bottom", "")

    def clear(self, title: str, subtitle: str = "") -> None:
        for k in range(2):
            self._fill(k, [], [], [])
        set_title(self.p1, title, subtitle)
        self.p2.hide()

    def set_cursor(self, t_ns: float) -> None:
        for _, _, _, cursor in self.items:
            cursor.setValue(float(t_ns))


class CalibrationPlot(QtWidgets.QWidget):
    """Captured center-row profiles (normalized, with their peaks) and lambda versus x."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.p_prof = self.glw.addPlot(row=0, col=0)
        self.p_cal = self.glw.addPlot(row=1, col=0)
        for p in (self.p_prof, self.p_cal):
            style_plot(p)
            p.setLabel("bottom", "x (µm)")
            p.setXRange(-0.5 * PITCH_X_UM, X_SPAN_UM + 0.5 * PITCH_X_UM, padding=0.02)
        self.p_prof.setLabel("left", "Normalized")
        self.p_cal.setLabel("left", "Wavelength (nm)")
        set_title(self.p_prof, "Captured center-row profiles")
        set_title(self.p_cal, "Calibration")
        self.p_prof.addLegend(offset=(-10, 10))

    def set_points(self, points: Sequence[Optional[CalibPoint]], calib: Optional[WavelengthCalibration]) -> None:
        self.p_prof.clear()
        self.p_cal.clear()
        legend = self.p_prof.legend
        if legend is not None:
            legend.clear()
        xs = np.linspace(-0.5 * PITCH_X_UM, X_SPAN_UM + 0.5 * PITCH_X_UM, 400)
        cal_x, cal_l = [], []
        for k, pt in enumerate(points):
            if pt is None or not pt.profile:
                continue
            color = TRACE_COLORS[k % len(TRACE_COLORS)]
            prof = np.asarray(pt.profile, dtype=np.float64)
            lo, hi = np.nanmin(prof), np.nanmax(prof)
            norm = (prof - lo) / (hi - lo) if hi > lo else prof * 0.0
            name = f"{pt.wavelength_nm:g} nm" if np.isfinite(pt.wavelength_nm) else f"row {k + 1}"
            self.p_prof.plot(COLUMN_X_UM, norm, pen=None, symbol="o", symbolSize=6, symbolBrush=color,
                             symbolPen=None, name=name)
            if np.isfinite(pt.sigma_um) and pt.sigma_um > 0:
                self.p_prof.plot(xs, np.exp(-((xs - pt.x_um) ** 2) / (2.0 * pt.sigma_um**2)),
                                 pen=pg.mkPen(color, width=1.3))
            vline = pg.InfiniteLine(pos=pt.x_um, angle=90, pen=pg.mkPen(color, width=1, style=Qt.DotLine))
            self.p_prof.addItem(vline, ignoreBounds=True)
            if np.isfinite(pt.wavelength_nm):
                cal_x.append(pt.x_um)
                cal_l.append(pt.wavelength_nm)
        if cal_x:
            self.p_cal.plot(cal_x, cal_l, pen=None, symbol="o", symbolSize=9, symbolBrush="#111827", symbolPen=None)
        if calib is not None:
            xx = np.array([0.0, X_SPAN_UM])
            self.p_cal.plot(xx, calib.wavelength(xx), pen=pg.mkPen(FIT_LINE_COLOR, width=1.6))
            set_title(
                self.p_cal, "Calibration",
                f"{calib.slope_nm_per_um:.5f} nm/µm ({calib.nm_per_column:.4f} nm/column), "
                f"rms residual {calib.rms_residual_nm:.3f} nm",
            )
        else:
            set_title(self.p_cal, "Calibration", "not computed")


def fwhm_text(sigma_um: float, calib: Optional[WavelengthCalibration]) -> str:
    fw = FWHM_FACTOR * sigma_um
    if calib is not None:
        return f"FWHM {fw * abs(calib.slope_nm_per_um):.2f} nm"
    return f"FWHM {fw:.1f} µm"


def double_fit_text(fit: DoubleFitResult, centers_fixed: bool = False) -> str:
    """One-line summary of a double 2D Gaussian fit.

    It flags low signal, and, unless the centers were held fixed, a second Gaussian that
    is not resolved.
    """
    parts = []
    for k in range(2):
        _, x0, y0, s = fit.component(k)
        parts.append(f"{k + 1}: ({x0:.1f}, {y0:.1f}) µm, σ={abs(s):.1f}")
    w1, w2 = fit.weights
    ratio = w2 / w1 if w1 > 0 else float("nan")
    text = "; ".join(parts) + f"; d={fit.separation:.1f} µm, I2/I1={ratio:.2f}"
    if not fit.quality_ok:
        text += f" — low signal ({min(fit.snr1, fit.snr2):.1f} σ)"
    elif not centers_fixed and not fit.resolved:
        text += " — second Gaussian not resolved"
    if centers_fixed:
        text += " (centers fixed)"
    return text


def trace_color(k: int) -> str:
    return TRACE_COLORS[k % len(TRACE_COLORS)]
