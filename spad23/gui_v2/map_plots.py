"""Pixel map, 2D Gaussian fit map and time-resolved spectrum image (pyqtgraph).

The map and the fit map invert y so row 0 (pixels 0-4, y = 0) is drawn at the
top, as in spad23_counts_gui.  In spectral mode the wavelength axis is y: the
map shows it on its right axis and highlights rows.
"""

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from spad23.spad23_analysis import FWHM_FACTOR, DoubleFitResult, FitResult, WavelengthCalibration, gauss2d, gauss2d_double
from spad23.spad23_geometry import (
    CENTER_X_UM,
    INACTIVE_XY,
    LINE_PITCH_UM,
    LINE_SPAN_UM,
    N_ROWS,
    NUM_PIXELS,
    PITCH_X_UM,
    PITCH_Y_UM,
    PIXEL_DIAMETER_UM,
    PIXEL_X_UM,
    PIXEL_Y_UM,
    ROW_Y_UM,
    SITE_PITCH_UM,
    nearest_pixel,
)
from spad23.gui_v2.config import (
    COMPONENT_COLORS,
    EXCLUDE_COLOR,
    FIT_CMAP,
    INACTIVE_COLOR,
    MAP_CMAP,
    NAN_COLOR,
    SELECT_COLOR,
)
from spad23.gui_v2.plots import WavelengthAxisItem, set_title, style_plot

Qt = QtCore.Qt

_ALL_XY = np.vstack([np.column_stack((PIXEL_X_UM, PIXEL_Y_UM)), INACTIVE_XY])
_X_LO, _X_HI = float(_ALL_XY[:, 0].min()), float(_ALL_XY[:, 0].max())
_Y_LO, _Y_HI = float(_ALL_XY[:, 1].min()), float(_ALL_XY[:, 1].max())
_MARGIN = 0.6 * SITE_PITCH_UM


def _brushes(cmap: pg.ColorMap, values: np.ndarray, levels, nan_color: str = NAN_COLOR):
    """One brush per value; NaN values get ``nan_color``."""
    lo, hi = levels
    span = (hi - lo) if hi > lo else 1.0
    out = [pg.mkBrush(nan_color)] * values.size
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size:
        norm = np.clip((values[finite] - lo) / span, 0.0, 1.0)
        colors = cmap.map(norm, mode="qcolor")
        for i, c in zip(finite, colors):
            out[i] = pg.mkBrush(c)
    return out

class _NanSafeImageItem(pg.ImageItem):
    """ImageItem whose NaN cells stay transparent at every downsampled size.

    pyqtgraph 0.13.7 caches the NaN positions of the (auto-downsampled) image and reuses them after a zoom or
    resize changes the downsampled shape: "IndexError: index N is out of bounds" in paint, and the image is not
    drawn.  Recomputing them at each render is cheap (renders happen only when the image or view changes).
    """

    def render(self, *args, **kwargs):
        self._imageNanLocations = None
        return super().render(*args, **kwargs)



class PixelMapPlot(QtWidgets.QWidget):
    """The 23-pixel map: one circle per pixel colored by value.

    Left click emits ``pixel_clicked(pixel, 1)``, right click ``(pixel, 2)``.
    """

    pixel_clicked = QtCore.pyqtSignal(int, int)
    hovered = QtCore.pyqtSignal(int)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lam_axis = WavelengthAxisItem("right")
        self.pw = pg.PlotWidget(axisItems={"right": self.lam_axis})
        lay.addWidget(self.pw)
        self.plot: pg.PlotItem = self.pw.getPlotItem()
        style_plot(self.plot, grid=False)
        self.plot.setLabel("bottom", "x (µm)")
        self.plot.setLabel("left", "y (µm)")
        self.plot.showAxis("right", False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setAspectLocked(True)
        vb.invertY(True)
        self.plot.setXRange(_X_LO - _MARGIN, _X_HI + _MARGIN, padding=0)
        self.plot.setYRange(_Y_LO - _MARGIN, _Y_HI + _MARGIN, padding=0)
        if title:
            set_title(self.plot, title)

        self.row_rects = []
        for k in range(N_ROWS):
            rect = QtWidgets.QGraphicsRectItem(
                _X_LO - 0.5 * SITE_PITCH_UM, ROW_Y_UM[k] - 0.5 * LINE_PITCH_UM, _X_HI - _X_LO + SITE_PITCH_UM, LINE_PITCH_UM
            )
            rect.setPen(pg.mkPen(None))
            rect.setBrush(pg.mkBrush(6, 182, 212, 45))
            rect.setZValue(-20)
            rect.hide()
            self.plot.addItem(rect, ignoreBounds=True)
            self.row_rects.append(rect)

        self.inactive = pg.ScatterPlotItem(
            x=INACTIVE_XY[:, 0], y=INACTIVE_XY[:, 1], size=PIXEL_DIAMETER_UM, pxMode=False,
            pen=pg.mkPen(INACTIVE_COLOR, width=1.5), brush=pg.mkBrush(None),
        )
        self.plot.addItem(self.inactive)
        # the center column along which the spectrum is read
        self.col_line = pg.InfiniteLine(pos=CENTER_X_UM, angle=90, movable=False,
                                        pen=pg.mkPen("#6b7280", width=1, style=Qt.DashLine))
        self.col_line.setZValue(-5)
        self.col_line.hide()
        self.plot.addItem(self.col_line, ignoreBounds=True)
        self.pixels = pg.ScatterPlotItem(
            x=PIXEL_X_UM, y=PIXEL_Y_UM, size=PIXEL_DIAMETER_UM, pxMode=False,
            pen=pg.mkPen("#4b5563", width=1), brush=pg.mkBrush(NAN_COLOR),
        )
        self.plot.addItem(self.pixels)
        self.sel = pg.ScatterPlotItem(size=PIXEL_DIAMETER_UM + 5.0, pxMode=False,
                                      pen=pg.mkPen(SELECT_COLOR, width=3), brush=pg.mkBrush(None))
        self.plot.addItem(self.sel)
        self.excl = pg.ScatterPlotItem(symbol="x", size=14, pxMode=True,
                                       pen=pg.mkPen(EXCLUDE_COLOR, width=2.5), brush=pg.mkBrush(EXCLUDE_COLOR))
        self.plot.addItem(self.excl)

        font = QtGui.QFont()
        font.setPointSize(9)
        self.labels = []
        for p in range(NUM_PIXELS):
            t = pg.TextItem(str(p), color="#f9fafb", anchor=(0.5, 0.5))
            t.setFont(font)
            t.setPos(PIXEL_X_UM[p], PIXEL_Y_UM[p])
            t.setZValue(5)
            self.plot.addItem(t, ignoreBounds=True)
            self.labels.append(t)

        self.hover_text = pg.TextItem("", color="#111827", anchor=(0, 1),
                                      fill=pg.mkBrush(255, 255, 255, 235), border=pg.mkPen("#9ca3af"))
        self.hover_text.setZValue(20)
        self.hover_text.hide()
        self.plot.addItem(self.hover_text, ignoreBounds=True)

        self.cmap = pg.colormap.get(MAP_CMAP)
        self.cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=self.cmap, interactive=False,
                                    width=14, colorMapMenu=False, pen="k")
        self.plot.layout.addItem(self.cbar, 2, 5)
        self.plot.layout.setColumnFixedWidth(4, 5)
        for ax in ("left", "right", "top", "bottom"):
            self.cbar.getAxis(ax).setPen(pg.mkPen("k"))
            self.cbar.getAxis(ax).setTextPen(pg.mkPen("k"))

        self._values = np.full(NUM_PIXELS, np.nan)
        self._excluded: frozenset = frozenset()
        self._hover = -1
        self.hover_text_fn: Optional[Callable[[int], str]] = None
        self.pw.scene().sigMouseClicked.connect(self._on_click)
        self.pw.scene().sigMouseMoved.connect(self._on_move)

    # ------------------------------------------------------------------ API
    def set_title(self, title: str, subtitle: str = "") -> None:
        set_title(self.plot, title, subtitle)

    def set_values(self, values: np.ndarray, levels=None, unit: str = "") -> None:
        v = np.asarray(values, dtype=np.float64).copy()
        if self._excluded:
            v[list(self._excluded)] = np.nan
        self._values = v
        finite = v[np.isfinite(v)]
        if levels is None:
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        else:
            lo, hi = float(levels[0]), float(levels[1])
        if not hi > lo:
            hi = lo + (abs(lo) * 1e-3 if lo else 1.0)
        self.pixels.setBrush(_brushes(self.cmap, v, (lo, hi)))
        self.cbar.setLevels((lo, hi))
        self.cbar.getAxis("left").setLabel(unit)
        norm = np.where(np.isfinite(v), (v - lo) / (hi - lo), 0.0)
        for p, t in enumerate(self.labels):
            t.setColor("#111827" if norm[p] > 0.6 else "#f9fafb")
        if self._hover >= 0:
            self._update_hover_text()

    def set_selected(self, pixels: Iterable[int]) -> None:
        ids = np.asarray(sorted(int(p) for p in pixels), dtype=np.int64)
        self.sel.setData(x=PIXEL_X_UM[ids], y=PIXEL_Y_UM[ids])

    def set_excluded(self, pixels: Iterable[int]) -> None:
        self._excluded = frozenset(int(p) for p in pixels)
        ids = np.asarray(sorted(self._excluded), dtype=np.int64)
        self.excl.setData(x=PIXEL_X_UM[ids], y=PIXEL_Y_UM[ids])

    def set_highlight_rows(self, rows: Iterable[int]) -> None:
        rows = set(int(r) for r in rows)
        for k, rect in enumerate(self.row_rects):
            rect.setVisible(k in rows)

    def set_spectral(self, on: bool, calib: Optional[WavelengthCalibration] = None) -> None:
        self.col_line.setVisible(on)
        self.plot.showAxis("right", bool(on and calib is not None))
        self.lam_axis.set_calibration(calib if on else None)
        if not on:
            self.set_highlight_rows(())

    def set_show_ids(self, on: bool) -> None:
        for t in self.labels:
            t.setVisible(on)

    # ---------------------------------------------------------------- mouse
    def _view_pos(self, scene_pos):
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        return vb.mapSceneToView(scene_pos)

    def _on_click(self, ev) -> None:
        if ev.double():
            return
        p = self._view_pos(ev.scenePos())
        if p is None:
            return
        pix = nearest_pixel(p.x(), p.y())
        if pix is None:
            return
        if ev.button() == Qt.LeftButton:
            self.pixel_clicked.emit(pix, 1)
        elif ev.button() == Qt.RightButton:
            self.pixel_clicked.emit(pix, 2)

    def _on_move(self, scene_pos) -> None:
        p = self._view_pos(scene_pos)
        pix = -1
        if p is not None:
            near = nearest_pixel(p.x(), p.y())
            pix = -1 if near is None else near
        if pix == self._hover:
            return
        self._hover = pix
        if pix < 0:
            self.hover_text.hide()
        else:
            self._update_hover_text()
        self.hovered.emit(pix)

    def _update_hover_text(self) -> None:
        pix = self._hover
        body = self.hover_text_fn(pix) if self.hover_text_fn is not None else ""
        self.hover_text.setText(f"px {pix}" + (f"\n{body}" if body else ""))
        # y is inverted: a smaller y is higher on the screen
        self.hover_text.setPos(PIXEL_X_UM[pix] + 0.4 * PIXEL_DIAMETER_UM, PIXEL_Y_UM[pix] - 0.4 * PIXEL_DIAMETER_UM)
        self.hover_text.show()


class Gauss2DFitPlot(QtWidgets.QWidget):
    """Fitted 2D Gaussian surface (viridis) with the measured pixels on top.

    ``set_data`` shows a single Gaussian (white FWHM ellipse); ``set_data_double``
    shows two (FWHM circles and centers in COMPONENT_COLORS, labelled 1 and 2).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pw = pg.PlotWidget()
        lay.addWidget(self.pw)
        self.plot: pg.PlotItem = self.pw.getPlotItem()
        style_plot(self.plot, grid=False)
        self.plot.setLabel("bottom", "x (µm)")
        self.plot.setLabel("left", "y (µm)")
        self.plot.getViewBox().setAspectLocked(True)
        self.plot.getViewBox().invertY(True)
        self.cmap = pg.colormap.get(FIT_CMAP)
        self.img = pg.ImageItem()
        self.img.setLookupTable(self.cmap.getLookupTable(nPts=256))
        self.img.setZValue(-30)
        self.plot.addItem(self.img)
        self.dots = pg.ScatterPlotItem(x=PIXEL_X_UM, y=PIXEL_Y_UM, size=9, pxMode=True, pen=pg.mkPen("k", width=1))
        self.plot.addItem(self.dots)
        self.ellipse = pg.PlotCurveItem(pen=pg.mkPen("w", width=1.6, style=Qt.DashLine))
        self.plot.addItem(self.ellipse)
        self.center = pg.ScatterPlotItem(symbol="x", size=14, pxMode=True, pen=pg.mkPen("w", width=2.5),
                                         brush=pg.mkBrush("w"))
        self.plot.addItem(self.center)
        self.comp_items = []
        for k, color in enumerate(COMPONENT_COLORS):
            circle = pg.PlotCurveItem(pen=pg.mkPen(color, width=1.8, style=Qt.DashLine))
            mark = pg.ScatterPlotItem(symbol="x", size=14, pxMode=True, pen=pg.mkPen(color, width=2.5),
                                      brush=pg.mkBrush(color))
            label = pg.TextItem(str(k + 1), color=color, anchor=(-0.2, 1.1))
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(True)
            label.setFont(font)
            label.hide()
            for item in (circle, mark, label):
                self.plot.addItem(item, ignoreBounds=True)
            self.comp_items.append((circle, mark, label))
        self.excl = pg.ScatterPlotItem(symbol="x", size=12, pxMode=True, pen=pg.mkPen(EXCLUDE_COLOR, width=2),
                                       brush=pg.mkBrush(EXCLUDE_COLOR))
        self.plot.addItem(self.excl)
        xpad, ypad = 0.35 * PITCH_X_UM, 0.35 * PITCH_Y_UM
        self._xr = (float(PIXEL_X_UM.min()) - xpad, float(PIXEL_X_UM.max()) + xpad)
        self._yr = (float(PIXEL_Y_UM.min()) - ypad, float(PIXEL_Y_UM.max()) + ypad)
        self.plot.setXRange(*self._xr, padding=0)
        self.plot.setYRange(*self._yr, padding=0)
        self._overlay = QtWidgets.QLabel(self.pw)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("color:#6b7280; background: transparent; font-size: 10pt;")
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._overlay.hide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._overlay.setGeometry(self.pw.rect())

    def set_data(
        self,
        values: np.ndarray,
        fit: Optional[FitResult],
        excluded: Iterable[int],
        title: str,
        subtitle: str = "",
        message: str = "Fit unavailable",
    ) -> None:
        self._show_components(None)
        if fit is not None:
            popt = fit.popt
            levels = self._draw_surface(lambda gxx, gyy: gauss2d((gxx, gyy), *popt))
            t = np.linspace(0, 2 * np.pi, 120)
            self.ellipse.setData(popt[1] + 0.5 * FWHM_FACTOR * popt[3] * np.cos(t),
                                 popt[2] + 0.5 * FWHM_FACTOR * popt[4] * np.sin(t))
            self.center.setData([popt[1]], [popt[2]])
        else:
            levels = None
            self.ellipse.setData([], [])
            self.center.setData([], [])
        self._finish(values, excluded, levels, title, subtitle, message)

    def set_data_double(
        self,
        values: np.ndarray,
        fit: Optional[DoubleFitResult],
        excluded: Iterable[int],
        title: str,
        subtitle: str = "",
        message: str = "Fit unavailable",
    ) -> None:
        self.ellipse.setData([], [])
        self.center.setData([], [])
        levels = None
        if fit is not None:
            popt = fit.popt
            levels = self._draw_surface(lambda gxx, gyy: gauss2d_double((gxx, gyy), *popt))
        self._show_components(fit)
        self._finish(values, excluded, levels, title, subtitle, message)

    # ------------------------------------------------------------ helpers
    def _draw_surface(self, model):
        gx = np.linspace(*self._xr, 160)
        gy = np.linspace(*self._yr, 140)
        gxx, gyy = np.meshgrid(gx, gy)
        zz = model(gxx, gyy)
        lo, hi = float(zz.min()), float(zz.max())
        if not hi > lo:
            hi = lo + 1e-9
        dx, dy = gx[1] - gx[0], gy[1] - gy[0]
        self.img.setImage(zz, autoLevels=False, levels=(lo, hi))
        self.img.setRect(QtCore.QRectF(gx[0] - dx / 2, gy[0] - dy / 2, gx[-1] - gx[0] + dx, gy[-1] - gy[0] + dy))
        self.img.show()
        return lo, hi

    def _show_components(self, fit: Optional[DoubleFitResult]) -> None:
        t = np.linspace(0, 2 * np.pi, 120)
        for k, (circle, mark, label) in enumerate(self.comp_items):
            if fit is None:
                circle.setData([], [])
                mark.setData([], [])
                label.hide()
                continue
            _, x0, y0, s = fit.component(k)
            r = 0.5 * FWHM_FACTOR * abs(s)
            circle.setData(x0 + r * np.cos(t), y0 + r * np.sin(t))
            mark.setData([x0], [y0])
            label.setPos(x0, y0)
            label.show()

    def _finish(self, values, excluded, levels, title: str, subtitle: str, message: str) -> None:
        v = np.asarray(values, dtype=np.float64)
        ex = sorted(int(p) for p in excluded)
        shown = v.copy()
        if ex:
            shown[ex] = np.nan
        if levels is None:
            self.img.hide()
            finite = shown[np.isfinite(shown)]
            levels = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            self._overlay.setText(message)
            self._overlay.setGeometry(self.pw.rect())
            self._overlay.show()
        else:
            self._overlay.hide()
        self.dots.setBrush(_brushes(self.cmap, shown, levels))
        ids = np.asarray(ex, dtype=np.int64)
        self.excl.setData(x=PIXEL_X_UM[ids], y=PIXEL_Y_UM[ids])
        set_title(self.plot, title, subtitle)


class TimeSpectrumImage(QtWidgets.QWidget):
    """I(row, t) image: x = row position y (wavelength axis), y = time."""

    time_clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.axis = WavelengthAxisItem("bottom")
        self.pw = pg.PlotWidget(axisItems={"bottom": self.axis})
        lay.addWidget(self.pw)
        self.plot: pg.PlotItem = self.pw.getPlotItem()
        style_plot(self.plot, grid=False)
        self.axis.set_calibration(None)
        self.plot.setLabel("left", "Time (ns)")
        self.cmap = pg.colormap.get(FIT_CMAP)
        self.img = _NanSafeImageItem(autoDownsample=True)
        self.plot.addItem(self.img)
        self.cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=self.cmap, interactive=False, width=14,
                                    colorMapMenu=False, pen="k")
        self.cbar.setImageItem(self.img, insert_in=self.plot)
        for ax in ("left", "right", "top", "bottom"):
            self.cbar.getAxis(ax).setPen(pg.mkPen("k"))
            self.cbar.getAxis(ax).setTextPen(pg.mkPen("k"))
        self.cursor = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("w", width=1.2, style=Qt.DashLine))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.row_lines = []
        for _ in range(N_ROWS):
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(SELECT_COLOR, width=1.5))
            line.hide()
            self.plot.addItem(line, ignoreBounds=True)
            self.row_lines.append(line)
        set_title(self.plot, "Time-resolved spectrum (center column)")
        self.pw.scene().sigMouseClicked.connect(self._on_click)

    def set_calibration(self, calib: Optional[WavelengthCalibration]) -> None:
        self.axis.set_calibration(calib)

    def set_title(self, title: str, subtitle: str = "") -> None:
        set_title(self.plot, title, subtitle)

    def set_data(self, time_ns: np.ndarray, matrix: np.ndarray, log: bool = False, levels=None,
                 label: Optional[str] = None) -> None:
        """NaN cells are left blank (transparent)."""
        if matrix.size == 0:
            self.img.clear()
            return
        z = np.asarray(matrix, dtype=np.float64)
        if log:
            z = np.log10(1.0 + np.clip(z, 0.0, None))
        finite = z[np.isfinite(z)]
        if levels is not None:
            lo, hi = float(levels[0]), float(levels[1])
        else:
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if not hi > lo:
            hi = lo + 1.0
        self.img.setImage(z, autoLevels=False, levels=(lo, hi))
        t = np.asarray(time_ns, dtype=np.float64)
        dt = float(t[1] - t[0]) if t.size > 1 else 1.0
        self.img.setRect(QtCore.QRectF(-0.5 * LINE_PITCH_UM, float(t[0]) - dt / 2, LINE_SPAN_UM + LINE_PITCH_UM, float(t[-1] - t[0]) + dt))
        self.cbar.setLevels((lo, hi))
        self.cbar.getAxis("left").setLabel(label if label is not None else ("log10(1 + counts)" if log else "Counts"))

    def set_cursor(self, t_ns: float) -> None:
        self.cursor.setValue(float(t_ns))

    def set_time_range(self, t0: float, t1: float) -> None:
        """Show all rows over [t0, t1]; right-click > View All shows the whole image."""
        rows = self.img.image.shape[0] if self.img.image is not None else 0
        dt = (t1 - t0) / (rows - 1) if rows > 1 else 0.0
        self.plot.setXRange(-0.5 * LINE_PITCH_UM, LINE_SPAN_UM + 0.5 * LINE_PITCH_UM, padding=0)
        if t1 > t0:
            self.plot.setYRange(t0 - 0.5 * dt, t1 + 0.5 * dt, padding=0)

    def set_rows(self, rows: Sequence[int]) -> None:
        rows = set(int(r) for r in rows)
        for k, line in enumerate(self.row_lines):
            if k in rows:
                line.setValue(float(ROW_Y_UM[k]))
                line.show()
            else:
                line.hide()

    def _on_click(self, ev) -> None:
        if ev.double() or ev.button() != Qt.LeftButton:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        self.time_clicked.emit(float(p.y()))
