"""pyqtgraph panels of the Montana pump-probe GUI.

* ``TAImagePanel``  -- log(ON/OFF) image (RdBu_r, symmetric or fixed levels) with a
  click crosshair, horizontal (spectral) and vertical (spatial) linecuts and the
  reference-ROI rectangle (2 x 2 GraphicsLayoutWidget, CLAUDE.md layout).
* ``MonitorPanel``  -- pump ON / OFF averages and the per-frame reference intensity.
* ``GaussianPanel`` -- Gaussian fit of the spatial linecut + width history
  (``GaussianFitter`` of the old GUI).
* ``GateVIPanel``   -- front / back gate V and I (keithley.gui_v2 PlotWidget).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from keithley.gui_v2.plots import PlotWidget as KeithleyVIPlot
from measurements.pump_probe_gate_dep.gaussian_fitting import GaussianFitter

Qt = QtCore.Qt


def _make_rdbu_r_lut():
    """Same diverging colormap as the old GUI."""
    pos = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    colors = np.array([
        [5, 48, 97, 255], [33, 102, 172, 255], [67, 147, 195, 255], [146, 197, 222, 255],
        [209, 229, 240, 255], [247, 247, 247, 255], [253, 219, 199, 255], [244, 165, 130, 255],
        [214, 96, 77, 255], [178, 24, 43, 255], [103, 0, 31, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


RDBU_R = _make_rdbu_r_lut()
RDBU_R_LUT = RDBU_R.getLookupTable(nPts=256)


def _black_axes(plot, axes=("bottom", "left")) -> None:
    for ax in axes:
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))


def _grid_box(title: str) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
    box = QtWidgets.QGroupBox(title)
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return box, g


# ============================================================================= main image
class TAImagePanel(QtWidgets.QGroupBox):
    """log(Pump ON / Pump OFF) with crosshair linecuts.

    ``set_image(lr)`` takes the (cropped) log-ratio array (rows = space, columns = wavelength);
    the raw array is kept for the linecut arithmetic.  ``crosshair_moved(x, y)`` fires on a click.
    """

    crosshair_moved = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__("log(Pump ON / Pump OFF)  [-1000 log10]", parent)
        self.data: Optional[np.ndarray] = None
        self.cx: Optional[int] = None
        self.cy: Optional[int] = None
        self.auto_levels = True
        self.levels = (-10.0, 10.0)
        self.wl_k, self.wl_b = 1.0, 0.0
        self.space_scale = 0.07
        self.space_unit = "um"
        self.roi_offset_x = 0  # crop x0: the reference ROI is in full-frame columns

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.infoLbl = QtWidgets.QLabel("No data")
        bar.addWidget(self.infoLbl)
        bar.addStretch(1)
        bar.addWidget(QtWidgets.QLabel("Linecut width (px):"))
        self.bandSpin = QtWidgets.QSpinBox()
        self.bandSpin.setRange(1, 200)
        self.bandSpin.setValue(1)
        self.bandSpin.setToolTip("Average this many rows / columns around the crosshair for the linecuts")
        self.bandSpin.valueChanged.connect(self._update_cuts)
        bar.addWidget(self.bandSpin)
        vbox.addLayout(bar)

        self.glw = pg.GraphicsLayoutWidget()
        self.img_plot = self.glw.addPlot(row=0, col=0)
        self.vcut_plot = self.glw.addPlot(row=0, col=1)
        self.hcut_plot = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setColumnStretchFactor(0, 4)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        for p in (self.img_plot, self.vcut_plot, self.hcut_plot):
            _black_axes(p)
            p.showGrid(x=True, y=True, alpha=0.25)

        self.img_plot.setLabel("bottom", "X (pixels)")
        self.img_plot.setLabel("left", "Y (pixels)")
        self.img_plot.invertY(True)
        self.img_plot.setAspectLocked(False)
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(RDBU_R_LUT)
        self.img_plot.addItem(self.img_item)
        self.cbar = pg.ColorBarItem(values=self.levels, colorMap=RDBU_R, interactive=False, width=12)
        self.cbar.setImageItem(self.img_item, insert_in=self.img_plot)

        self.roi_curve = pg.PlotCurveItem(pen=pg.mkPen("y", width=2))
        self.img_plot.addItem(self.roi_curve)
        self.hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("k", width=1, style=Qt.DashLine))
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1, style=Qt.DashLine))
        for ln in (self.hline, self.vline):
            ln.setVisible(False)
            self.img_plot.addItem(ln)
        self.img_plot.scene().sigMouseClicked.connect(self._on_click)

        self.vcut_plot.setLabel("bottom", "Signal")
        self.vcut_plot.setLabel("left", "Space (um)")
        self.vcut_plot.invertY(True)
        self.vcut_curve = self.vcut_plot.plot(pen=pg.mkPen("r", width=2))
        self.vcut_marker = pg.ScatterPlotItem(pen=None, brush="k", size=8)
        self.vcut_plot.addItem(self.vcut_marker)
        self.vcut_plot.addItem(pg.InfiniteLine(angle=90, pen=pg.mkPen("gray", style=Qt.DashLine)))

        self.hcut_plot.setLabel("bottom", "Wavelength / pixel")
        self.hcut_plot.setLabel("left", "Signal")
        self.hcut_curve = self.hcut_plot.plot(pen=pg.mkPen("b", width=2))
        self.hcut_marker = pg.ScatterPlotItem(pen=None, brush="k", size=8)
        self.hcut_plot.addItem(self.hcut_marker)
        self.hcut_plot.addItem(pg.InfiniteLine(angle=0, pen=pg.mkPen("gray", style=Qt.DashLine)))
        vbox.addWidget(self.glw, 1)

    # ------------------------------------------------------------------ settings
    def set_levels(self, auto: bool, vmin: float = -10.0, vmax: float = 10.0) -> None:
        self.auto_levels = bool(auto)
        self.levels = (float(vmin), float(vmax))
        if self.data is not None:
            self._apply_levels()

    def set_calibration(self, wl_k: float, wl_b: float, space_scale: float) -> None:
        self.wl_k, self.wl_b, self.space_scale = float(wl_k), float(wl_b), float(space_scale)
        self.space_unit = "um" if self.space_scale != 1.0 else "px"
        self.vcut_plot.setLabel("left", f"Space ({self.space_unit})")
        self.hcut_plot.setLabel("bottom", "Wavelength (nm)" if (self.wl_k, self.wl_b) != (1.0, 0.0) else "X (pixel)")
        self._update_cuts()

    def set_roi(self, roi, crop_x0: int = 0) -> None:
        x, y, w, h = roi
        x = x - crop_x0
        self.roi_curve.setData([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])

    # ------------------------------------------------------------------ data
    def set_image(self, lr: np.ndarray, info: str = "") -> None:
        self.data = lr
        self.img_item.setImage(lr, autoLevels=False)
        self._apply_levels()
        h, w = lr.shape
        if self.cx is None or self.cy is None:
            self.cx, self.cy = w // 2, h // 2
        self.cx = int(np.clip(self.cx, 0, w - 1))
        self.cy = int(np.clip(self.cy, 0, h - 1))
        self.vline.setValue(self.cx + 0.5)
        self.hline.setValue(self.cy + 0.5)
        self.vline.setVisible(True)
        self.hline.setVisible(True)
        if info:
            self.infoLbl.setText(info)
        self._update_cuts()

    def _apply_levels(self) -> None:
        if self.auto_levels:
            finite = self.data[np.isfinite(self.data)]
            vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            lv = (-vmax, vmax)
        else:
            lv = self.levels
        self.img_item.setLevels(lv)
        self.cbar.setLevels(values=lv)

    def _on_click(self, event) -> None:
        if self.data is None:
            return
        pos = event.scenePos()
        if not self.img_plot.vb.sceneBoundingRect().contains(pos):
            return
        pt = self.img_plot.vb.mapSceneToView(pos)
        h, w = self.data.shape
        x, y = int(np.floor(pt.x())), int(np.floor(pt.y()))
        if 0 <= x < w and 0 <= y < h:
            self.cx, self.cy = x, y
            self.vline.setValue(x + 0.5)
            self.hline.setValue(y + 0.5)
            self._update_cuts()
            self.crosshair_moved.emit(x, y)

    def set_crosshair(self, x: int, y: int) -> None:
        self.cx, self.cy = int(x), int(y)
        if self.data is not None:
            self.set_image(self.data)

    def x_coords(self, w: int) -> np.ndarray:
        return self.wl_k * np.arange(w) + self.wl_b

    def y_coords(self, h: int) -> np.ndarray:
        return self.space_scale * np.arange(h)

    def space_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(coords, profile) of the vertical linecut at the crosshair (Gaussian fit input)."""
        if self.data is None or self.cx is None:
            return None
        h, w = self.data.shape
        half = (self.bandSpin.value() - 1) // 2
        c1, c2 = max(0, self.cx - half), min(w, self.cx + half + 1)
        return self.y_coords(h), self.data[:, c1:c2].mean(axis=1)

    def _update_cuts(self, *_a) -> None:
        if self.data is None or self.cx is None:
            return
        lr = self.data
        h, w = lr.shape
        half = (self.bandSpin.value() - 1) // 2
        r1, r2 = max(0, self.cy - half), min(h, self.cy + half + 1)
        c1, c2 = max(0, self.cx - half), min(w, self.cx + half + 1)
        xprof = lr[r1:r2, :].mean(axis=0)
        yprof = lr[:, c1:c2].mean(axis=1)
        xc, yc = self.x_coords(w), self.y_coords(h)
        self.hcut_curve.setData(xc, xprof)
        self.hcut_marker.setData([xc[self.cx]], [xprof[self.cx]])
        self.vcut_curve.setData(yprof, yc)
        self.vcut_marker.setData([yprof[self.cy]], [yc[self.cy]])
        self.infoLbl.setText(
            f"x={self.cx} ({xc[self.cx]:.2f})  y={self.cy} ({yc[self.cy]:.2f} {self.space_unit})  "
            f"value={lr[self.cy, self.cx]:.4g}   range {np.nanmin(lr):.3g} .. {np.nanmax(lr):.3g}")


# ============================================================================= monitoring
class MonitorPanel(QtWidgets.QWidget):
    """Pump ON / OFF averages (grayscale) and the reference ROI intensity of every frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.auto_levels = True
        self.levels = (3000.0, 8000.0)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.glw = pg.GraphicsLayoutWidget()
        self.on_plot = self.glw.addPlot(row=0, col=0, title="Pump ON")
        self.off_plot = self.glw.addPlot(row=0, col=1, title="Pump OFF")
        self.ref_plot = self.glw.addPlot(row=1, col=0, colspan=2, title="Reference ROI intensity")
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.on_item, self.off_item = pg.ImageItem(), pg.ImageItem()
        self.on_text = pg.TextItem("N=0", color="r", anchor=(0, 0))
        self.off_text = pg.TextItem("N=0", color="b", anchor=(0, 0))
        self.on_roi = pg.PlotCurveItem(pen=pg.mkPen("y", width=1))
        self.off_roi = pg.PlotCurveItem(pen=pg.mkPen("y", width=1))
        for plot, item, txt, roi in ((self.on_plot, self.on_item, self.on_text, self.on_roi),
                                     (self.off_plot, self.off_item, self.off_text, self.off_roi)):
            _black_axes(plot)
            plot.invertY(True)
            plot.addItem(item)
            plot.addItem(roi)
            plot.addItem(txt)
        _black_axes(self.ref_plot)
        self.ref_plot.setLabel("bottom", "Frame")
        self.ref_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trace = self.ref_plot.plot(pen=pg.mkPen("k", width=0.5))
        self.sc_on = pg.ScatterPlotItem(pen=None, brush="r", size=5)
        self.sc_off = pg.ScatterPlotItem(pen=None, brush="b", size=5)
        self.thr = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", style=Qt.DashLine))
        for it in (self.sc_on, self.sc_off, self.thr):
            self.ref_plot.addItem(it)
        lay.addWidget(self.glw, 1)

    def set_levels(self, auto: bool, vmin: float, vmax: float) -> None:
        self.auto_levels = bool(auto)
        self.levels = (float(vmin), float(vmax))

    def update_results(self, results: dict, roi, crop) -> None:
        """``results``: the old analysis dict; images shown full-frame (the ROI is in full-frame pixels)."""
        x, y, w, h = roi
        for item, txt, roi_c, img, n in ((self.on_item, self.on_text, self.on_roi, results.get("avg_pump_on"),
                                          results.get("n_pump_on", 0)),
                                         (self.off_item, self.off_text, self.off_roi, results.get("avg_pump_off"),
                                          results.get("n_pump_off", 0))):
            if img is None:
                continue
            if self.auto_levels:
                item.setImage(img, autoLevels=True)
            else:
                item.setImage(img, levels=self.levels)
            txt.setText(f"N={n}")
            roi_c.setData([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])
        ref = np.asarray(results.get("ref_intensities", []), dtype=float)
        if ref.size:
            fr = np.arange(1, ref.size + 1)
            thr = float(results.get("threshold", np.nan))
            on = ref > thr
            self.trace.setData(fr, ref)
            self.sc_on.setData(fr[on], ref[on])
            self.sc_off.setData(fr[~on], ref[~on])
            self.thr.setValue(thr)


# ============================================================================= Gaussian fit
class GaussianPanel(QtWidgets.QWidget):
    """Gaussian fit of the spatial linecut at the crosshair (old Gaussian Fit tab)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fitter = GaussianFitter()
        self.widths = []
        self.unit = "um"
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        box, g = _grid_box("Fit")
        self.enableChk = QtWidgets.QCheckBox("Enable Gaussian fit")
        self.enableChk.setChecked(True)
        self.autoChk = QtWidgets.QCheckBox("Auto initial guess")
        self.autoChk.setChecked(True)
        self.ampSpin = self._spin(-1e6, 1e6, 1.0, 4)
        self.cenSpin = self._spin(-1e6, 1e6, 64.0, 3)
        self.widSpin = self._spin(0.0, 1e6, 10.0, 3)
        self.offSpin = self._spin(-1e6, 1e6, 0.0, 4)
        g.addWidget(self.enableChk, 0, 0, 1, 2)
        g.addWidget(self.autoChk, 1, 0, 1, 2)
        for r, (lbl, sp) in enumerate((("Amplitude:", self.ampSpin), ("Center:", self.cenSpin),
                                       ("Width sigma:", self.widSpin), ("Offset:", self.offSpin)), start=2):
            g.addWidget(QtWidgets.QLabel(lbl), r, 0)
            g.addWidget(sp, r, 1)
        self.clearBtn = QtWidgets.QPushButton("Clear history")
        self.clearBtn.clicked.connect(self.clear_history)
        g.addWidget(self.clearBtn, 6, 0, 1, 2)
        self.resultLbl = QtWidgets.QLabel("")
        self.resultLbl.setWordWrap(True)
        g.addWidget(self.resultLbl, 7, 0, 1, 2)
        g.setRowStretch(8, 1)
        box.setFixedWidth(240)
        lay.addWidget(box)

        self.glw = pg.GraphicsLayoutWidget()
        self.fit_plot = self.glw.addPlot(row=0, col=0, title="Spatial profile + Gaussian fit")
        self.hist_plot = self.glw.addPlot(row=1, col=0, title="Width history")
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        for p in (self.fit_plot, self.hist_plot):
            _black_axes(p)
            p.showGrid(x=True, y=True, alpha=0.3)
        self.data_curve = self.fit_plot.plot(pen=None, symbol="o", symbolBrush="b", symbolSize=4)
        self.fit_curve = self.fit_plot.plot(pen=pg.mkPen("r", width=2))
        self.center_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("g", style=Qt.DashLine))
        self.fwhm = pg.LinearRegionItem(values=(0, 1), brush=(255, 255, 0, 50), movable=False)
        for it in (self.center_line, self.fwhm):
            it.setVisible(False)
            self.fit_plot.addItem(it)
        self.hist_curve = self.hist_plot.plot(pen=pg.mkPen("b", width=2), symbol="o", symbolSize=5)
        self.hist_plot.setLabel("bottom", "Update")
        lay.addWidget(self.glw, 1)

    @staticmethod
    def _spin(lo, hi, val, dec):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setValue(val)
        return sp

    def clear_history(self) -> None:
        self.widths = []
        self.hist_curve.setData([], [])

    def update_fit(self, coords: np.ndarray, profile: np.ndarray, unit: str = "um") -> None:
        if not self.enableChk.isChecked() or coords is None:
            return
        self.unit = unit
        init = None
        if not self.autoChk.isChecked():
            init = (self.ampSpin.value(), self.cenSpin.value(), self.widSpin.value(), self.offSpin.value())
        res = self.fitter.fit(profile, coords, initial_params=init, use_auto=self.autoChk.isChecked())
        self.data_curve.setData(coords, profile)
        self.fit_plot.setLabel("bottom", f"Y ({unit})")
        if not res["success"]:
            self.fit_curve.setData([], [])
            self.center_line.setVisible(False)
            self.fwhm.setVisible(False)
            self.resultLbl.setText(f"Fit failed: {res['error_message']}")
            return
        amp, cen, wid, off = res["params"]
        self.fit_curve.setData(coords, res["fit_curve"])
        self.center_line.setValue(cen)
        self.center_line.setVisible(True)
        self.fwhm.setRegion((cen - res["fwhm"] / 2, cen + res["fwhm"] / 2))
        self.fwhm.setVisible(True)
        self.resultLbl.setText(f"Center {cen:.3f} {unit}\nSigma {abs(wid):.3f} {unit}\nFWHM {res['fwhm']:.3f} {unit}\n"
                               f"Amp {amp:.4g}  Offset {off:.4g}\nR² {res['r_squared']:.4f}")
        self.widths.append(abs(wid))
        self.widths = self.widths[-100:]
        self.hist_curve.setData(np.arange(len(self.widths)), self.widths)
        self.hist_plot.setLabel("left", f"sigma ({unit})")


# ============================================================================= gate V / I
class GateVIPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.plots = {}
        for which in ("front", "back"):
            lay.addWidget(QtWidgets.QLabel(f"{which.capitalize()} gate V (solid) / I nA (dashed)"))
            p = KeithleyVIPlot()
            for ax in ("bottom", "left"):
                p.pw.getAxis(ax).setPen(pg.mkPen("k"))
                p.pw.getAxis(ax).setTextPen(pg.mkPen("k"))
            self.plots[which] = p
            lay.addWidget(p, 1)

    def add_point(self, which: str, v: float, i_a: float) -> None:
        if which in self.plots and np.isfinite(v) and np.isfinite(i_a):
            self.plots[which].add_point(v, i_a)

    def clear(self) -> None:
        for p in self.plots.values():
            p.clear()
