from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .config import DIFF_COLOR, MAP_CMAP, OFF_COLOR, ON_COLOR


def _pg_cmap(name: str) -> pg.ColorMap:
    for source in ("matplotlib", None):
        try:
            return pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
        except Exception:
            pass
    return pg.colormap.get("viridis")


def _black_axes(plot: pg.PlotItem, axes=("bottom", "left")) -> None:
    for ax in axes:
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))


def band_rows(row: int, width: int, h: int) -> Tuple[int, int]:
    """Row band [r1, r2) summed for a linecut of `width` rows centred on `row`."""
    width = max(1, int(width))
    half = width // 2
    r1 = max(0, int(row) - half)
    r2 = min(int(h), int(row) + half + (1 if width % 2 else 0))
    return r1, max(r1 + 1, r2)


class PLMapPanel(QtWidgets.QGroupBox):
    """PL CCD image (display orientation) with colour bar, crosshair and linecut band."""

    pixel_clicked = QtCore.pyqtSignal(int, int)  # display (row, col)

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._disp: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None
        self._x0 = 0.0
        self._dx = 1.0
        self._build_ui()

    def _build_ui(self) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)
        vlay.setSpacing(4)

        self.infoLbl = QtWidgets.QLabel("(not collected)")
        self.cursorLbl = QtWidgets.QLabel("Cursor: --")
        vlay.addWidget(self.infoLbl)
        vlay.addWidget(self.cursorLbl)

        self.glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self.glw, 1)

        self.img_plot = self.glw.addPlot(row=0, col=0)
        self.img_plot.invertY(True)  # display row 0 at top
        self.img_plot.setLabel("bottom", "Pixel")
        self.img_plot.setLabel("left", "Spatial pixel")
        _black_axes(self.img_plot)

        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        try:
            self.hist.gradient.setColorMap(_pg_cmap(MAP_CMAP))
        except Exception:
            pass
        self.glw.addItem(self.hist, row=0, col=1)
        self.glw.ci.layout.setColumnStretchFactor(0, 5)
        self.glw.ci.layout.setColumnMaximumWidth(1, 90)

        self.band = pg.LinearRegionItem(
            values=(0, 1),
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 50),
            pen=pg.mkPen("w", width=0.8, style=QtCore.Qt.DashLine),
        )
        self.band.setZValue(10)
        self.band.setVisible(False)
        self.img_plot.addItem(self.band)

        self.cross_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("r", width=1.0))
        self.cross_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1.0))
        for ln in (self.cross_h, self.cross_v):
            ln.setZValue(11)
            ln.setVisible(False)
            self.img_plot.addItem(ln)

        self.img_plot.scene().sigMouseClicked.connect(self._on_mouse_click)

    # -----------------
    # Public API
    # -----------------
    def set_info(self, text: str) -> None:
        self.infoLbl.setText(str(text))

    def disp_shape(self) -> Optional[Tuple[int, int]]:
        return None if self._disp is None else self._disp.shape

    def set_image(self, disp: np.ndarray, x_axis: Optional[np.ndarray], *, keep_view: bool = False) -> None:
        """Show `disp` (display orientation, h x w). `x_axis` is the wavelength per column or None."""
        disp = np.asarray(disp, dtype=float)
        h, w = disp.shape
        is_wl = x_axis is not None and np.asarray(x_axis).size == w and w > 1
        x = np.asarray(x_axis, dtype=float).ravel() if is_wl else np.arange(w, dtype=float)
        same_geom = (
            keep_view
            and self._disp is not None
            and self._disp.shape == disp.shape
            and self._x is not None
            and np.allclose(self._x, x)
        )
        self._disp = disp
        self._x = x
        self._dx = float(x[-1] - x[0]) / (w - 1) if w > 1 else 1.0
        if self._dx == 0.0:
            self._dx = 1.0
        self._x0 = float(x[0]) - self._dx / 2.0  # pixel centres land on x[i]

        safe = disp if np.isfinite(disp).any() else np.zeros_like(disp)
        self.img_item.setImage(safe, autoLevels=not same_geom)
        self.img_item.setRect(QtCore.QRectF(self._x0, 0.0, self._dx * w, float(h)))
        self.img_plot.setLabel("bottom", "Wavelength (nm)" if is_wl else "Pixel")
        if not same_geom:
            lo, hi = sorted((self._x0, self._x0 + self._dx * w))
            self.img_plot.setXRange(lo, hi, padding=0)
            self.img_plot.setYRange(0, float(h), padding=0)

    def clear(self, info: str = "(not collected)") -> None:
        self._disp = None
        self._x = None
        self.img_item.clear()
        self.set_info(info)
        self.hide_cursor()

    def hide_cursor(self) -> None:
        for item in (self.band, self.cross_h, self.cross_v):
            item.setVisible(False)
        self.cursorLbl.setText("Cursor: --")

    def set_cursor(self, row: int, col: int, width: int) -> None:
        if self._disp is None:
            self.hide_cursor()
            return
        h, w = self._disp.shape
        row = max(0, min(h - 1, int(row)))
        col = max(0, min(w - 1, int(col)))
        r1, r2 = band_rows(row, width, h)
        self.band.setRegion((float(r1), float(r2)))
        self.cross_h.setPos(row + 0.5)
        self.cross_v.setPos(float(self._x[col]))
        for item in (self.band, self.cross_h, self.cross_v):
            item.setVisible(True)
        val = float(self._disp[row, col])
        self.cursorLbl.setText(
            f"Cursor: (row, col) = {row}, {col} | x = {self._x[col]:.3f} | I = {val:g} | rows {r1}-{r2 - 1}"
        )

    # -----------------
    # Events
    # -----------------
    def _on_mouse_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton or self._disp is None:
            return
        pos = event.scenePos()
        vb = self.img_plot.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        pt = vb.mapSceneToView(pos)
        h, w = self._disp.shape
        row = int(np.floor(pt.y()))
        col = int(np.floor((pt.x() - self._x0) / self._dx))
        if not (0 <= row < h and 0 <= col < w):
            return
        self.pixel_clicked.emit(row, col)


class SpectrumPanel(QtWidgets.QGroupBox):
    """ON / OFF spectra (top) and ON - OFF (bottom), x-linked."""

    def __init__(self, title: str = "Spectra at cursor", parent=None):
        super().__init__(title, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)
        vlay.setSpacing(4)

        self.infoLbl = QtWidgets.QLabel("Click a pixel on either map")
        vlay.addWidget(self.infoLbl)

        self.glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self.glw, 1)

        self.spec_plot = self.glw.addPlot(row=0, col=0)
        self.spec_plot.setLabel("left", "Intensity (counts)")
        self.spec_plot.setLabel("bottom", "Pixel")
        self.spec_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spec_plot.addLegend(offset=(-10, 10))
        _black_axes(self.spec_plot)
        self.on_curve = self.spec_plot.plot([], [], pen=pg.mkPen(ON_COLOR, width=1.4), name="PL1 (ON)")
        self.off_curve = self.spec_plot.plot([], [], pen=pg.mkPen(OFF_COLOR, width=1.4), name="PL2 (OFF)")

        self.diff_plot = self.glw.addPlot(row=1, col=0)
        self.diff_plot.setLabel("left", "ON − OFF")
        self.diff_plot.setLabel("bottom", "Pixel")
        self.diff_plot.showGrid(x=True, y=True, alpha=0.2)
        self.diff_plot.setXLink(self.spec_plot)
        _black_axes(self.diff_plot)
        self.diff_curve = self.diff_plot.plot([], [], pen=pg.mkPen(DIFF_COLOR, width=1.4))
        self.zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, movable=False, pen=pg.mkPen("#888888", width=1.0, style=QtCore.Qt.DashLine)
        )
        self.diff_plot.addItem(self.zero_line)

    def update_spectra(
        self,
        x_on: Optional[np.ndarray],
        on: Optional[np.ndarray],
        x_off: Optional[np.ndarray],
        off: Optional[np.ndarray],
        x_diff: Optional[np.ndarray],
        diff: Optional[np.ndarray],
        *,
        xlabel: str = "Pixel",
        info: str = "",
    ) -> None:
        for curve, x, y in (
            (self.on_curve, x_on, on),
            (self.off_curve, x_off, off),
            (self.diff_curve, x_diff, diff),
        ):
            if x is None or y is None:
                curve.setData([], [])
            else:
                curve.setData(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self.spec_plot.setLabel("bottom", xlabel)
        self.diff_plot.setLabel("bottom", xlabel)
        if info:
            self.infoLbl.setText(info)
        self.spec_plot.enableAutoRange()
        self.diff_plot.enableAutoRange()

    def clear(self, info: str = "Click a pixel on either map") -> None:
        for curve in (self.on_curve, self.off_curve, self.diff_curve):
            curve.setData([], [])
        self.infoLbl.setText(info)
