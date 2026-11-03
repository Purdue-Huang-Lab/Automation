"""pyqtgraph widgets for the two-pump dI/dP viewer.

Images follow the pyqtgraph default (column-major) axis order, matching the
SingleFrameView reused from pl_power_gate_dep_viewer_v2, so maps are passed
to ImageItem transposed.
"""

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .config import (
    CURSOR_X_COLOR, CURSOR_Y_COLOR, DIFF_COLOR, MAP_DISPLAY_COLS, REF_COLOR,
    SET1_COLOR, SET2_COLOR,
)
from .ui import MIN_PT, px


def pg_cmap(name: str) -> pg.ColorMap:
    for source in ("matplotlib", None):
        try:
            return pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
        except Exception:
            pass
    return pg.colormap.get("viridis")


def black_axes(plot: pg.PlotItem, axes=("bottom", "left")) -> None:
    for ax in axes:
        plot.getAxis(ax).setPen(pg.mkPen("k"))
        plot.getAxis(ax).setTextPen(pg.mkPen("k"))


def _fmt(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.4g}" if np.isfinite(v) else "--"


class SweepAxisItem(pg.AxisItem):
    """Axis whose log-mode ticks are labelled only at 1, 2 and 5 × 10ⁿ.

    Over less than a decade pyqtgraph labels every minor tick (2…9 × 10ⁿ), and
    the labels run into each other ("0.0010.0010.001").
    """

    def logTickStrings(self, values, scale, spacing):
        strings = super().logTickStrings(values, scale, spacing)
        out = []
        for v, s in zip(values, strings):
            x = 10.0 ** float(v) * float(scale)
            if not (np.isfinite(x) and x > 0):
                out.append("")
                continue
            mant = x / 10.0 ** np.floor(np.log10(x) + 1e-9)
            out.append(s if any(abs(mant - m) < 1e-3 * m for m in (1.0, 2.0, 5.0)) else "")
        return out


def _compact(v: float) -> str:
    """3 significant digits; 1e4 and up as 2.5e5 (long numbers run into each other)."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e4:
        return f"{v:.3g}"
    mant, exp = f"{v:.2e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


class CompactAxisItem(pg.AxisItem):
    """Linear axis with short tick labels (see ``_compact``)."""

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return super().tickStrings(values, scale, spacing)
        return [_compact(float(v) * float(scale)) for v in values]


def _column_edges(u: np.ndarray) -> np.ndarray:
    """Bin edges around sorted, finite centres u (half-way to each neighbour)."""
    if u.size == 1:
        return np.array([u[0] - 0.5, u[0] + 0.5])
    mid = 0.5 * (u[1:] + u[:-1])
    return np.concatenate([[u[0] - (mid[0] - u[0])], mid, [u[-1] + (u[-1] - mid[-1])]])


class DiffMapWidget(QtWidgets.QGroupBox):
    """Map (sweep × energy) with crosshair cursor, colour bar and both linecuts.

    The sweep axis is usually non-uniform (log-spaced powers), so the image is
    drawn on a fine uniform display grid where every measured column fills the
    span half-way to its neighbours. Linecuts always use the measured points.
    """

    cursor_changed = QtCore.pyqtSignal(int, int)   # (x index, y index)

    def __init__(self, title: str = "Map", parent=None):
        super().__init__(title, parent)
        self._x = self._y = self._z = None
        self._x_eval: Optional[np.ndarray] = None
        self._ref: Optional[np.ndarray] = None
        self._u: Optional[np.ndarray] = None
        self._log_x = False
        self._ix: Optional[int] = None
        self._iy: Optional[int] = None
        self._cmap_name = ""
        self._manual_max: Optional[float] = None
        self._x_label = self._y_label = self._z_label = ""
        self._ref_in_legend = True
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)
        vlay.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        self.cursorLbl = QtWidgets.QLabel("Cursor: --")
        top.addWidget(self.cursorLbl, 1)
        self.lockLevelsCheck = QtWidgets.QCheckBox("Lock colours")
        self.lockLevelsCheck.setToolTip("Keep the current colour range when the map is redrawn.")
        top.addWidget(self.lockLevelsCheck)
        self.exportMapBtn = QtWidgets.QPushButton("Export map")
        self.exportCutsBtn = QtWidgets.QPushButton("Export linecuts")
        top.addWidget(self.exportMapBtn)
        top.addWidget(self.exportCutsBtn)
        vlay.addLayout(top)

        self.glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self.glw, 1)

        self.mapPlot = self.glw.addPlot(row=0, col=0, axisItems={"bottom": SweepAxisItem("bottom")})
        black_axes(self.mapPlot)
        self.imgItem = pg.ImageItem()
        self.mapPlot.addItem(self.imgItem)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.imgItem)
        self.glw.addItem(self.hist, row=0, col=1)

        self.vline = pg.InfiniteLine(angle=90, movable=True,
                                     pen=pg.mkPen(CURSOR_X_COLOR, width=1.2),
                                     hoverPen=pg.mkPen(CURSOR_X_COLOR, width=2.5))
        self.hline = pg.InfiniteLine(angle=0, movable=True,
                                     pen=pg.mkPen(CURSOR_Y_COLOR, width=1.2),
                                     hoverPen=pg.mkPen(CURSOR_Y_COLOR, width=2.5))
        self.mapPlot.addItem(self.vline)
        self.mapPlot.addItem(self.hline)
        self.vline.sigPositionChangeFinished.connect(self._on_vline_moved)
        self.hline.sigPositionChangeFinished.connect(self._on_hline_moved)

        # Right: spectrum (quantity vs energy) at the selected sweep column
        self.specPlot = self.glw.addPlot(row=0, col=2, axisItems={"bottom": CompactAxisItem("bottom")})
        black_axes(self.specPlot)
        self.specPlot.setYLink(self.mapPlot)
        self.specPlot.showGrid(x=True, y=True, alpha=0.2)
        self.specRef = self.specPlot.plot(pen=pg.mkPen(REF_COLOR, width=1.2, style=QtCore.Qt.DashLine))
        self.specCurve = self.specPlot.plot(pen=pg.mkPen(DIFF_COLOR, width=1.4))
        self.specZero = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#999", width=0.8))
        self.specCursor = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(CURSOR_Y_COLOR, width=0.8))
        self.specPlot.addItem(self.specZero)
        self.specPlot.addItem(self.specCursor)

        # Bottom: trace (quantity vs sweep) at the selected energy row
        self.cutPlot = self.glw.addPlot(row=1, col=0, axisItems={"bottom": SweepAxisItem("bottom"),
                                                                 "left": CompactAxisItem("left")})
        black_axes(self.cutPlot)
        self.cutPlot.setXLink(self.mapPlot)
        self.cutPlot.showGrid(x=True, y=True, alpha=0.2)
        self.cutLegend = self.cutPlot.addLegend(offset=(-10, 10))
        self.cutRef = self.cutPlot.plot(
            pen=pg.mkPen(REF_COLOR, width=1.2, style=QtCore.Qt.DashLine),
            symbol="s", symbolSize=5, symbolBrush=REF_COLOR, symbolPen=None, name="numerical (set 1)")
        self.cutCurve = self.cutPlot.plot(
            pen=pg.mkPen(DIFF_COLOR, width=1.4),
            symbol="o", symbolSize=6, symbolBrush=DIFF_COLOR, symbolPen=None, name="measured")
        self.cutZero = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#999", width=0.8))
        self.cutCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(CURSOR_X_COLOR, width=0.8))
        self.cutPlot.addItem(self.cutZero)
        self.cutPlot.addItem(self.cutCursor)

        lay = self.glw.ci.layout
        lay.setRowStretchFactor(0, 3)
        lay.setRowStretchFactor(1, 2)
        lay.setColumnStretchFactor(0, 4)
        lay.setColumnStretchFactor(2, 2)
        lay.setColumnMaximumWidth(1, px(110))

        self.mapPlot.scene().sigMouseClicked.connect(self._on_scene_click)
        self.exportMapBtn.clicked.connect(self._on_export_map)
        self.exportCutsBtn.clicked.connect(self._on_export_cuts)

    # ── public API ────────────────────────────────────────────────────

    def set_manual_max(self, value: Optional[float]) -> None:
        """Half-range of a symmetric colour scale; None / ≤0 → automatic."""
        self._manual_max = float(value) if value and value > 0 else None

    def selected(self) -> Tuple[Optional[int], Optional[int]]:
        return self._ix, self._iy

    def set_data(self, x, y, z, *, x_label: str, y_label: str, z_label: str,
                 log_x: bool, symmetric: bool, center: float = 0.0, cmap: str = "viridis",
                 x_eval=None, ref=None) -> None:
        """x: (n_x,) sweep values  y: (n_y,) uniform ascending  z: (n_y, n_x).

        x_eval : sweep positions at which the measured quantity applies
                 (P₁ + P₂/2 for derivatives); the map still uses x.
        ref    : optional (n_y, n_x) reference overlaid dashed in both linecuts.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float)
        if not (x.size and y.size) or z.shape != (y.size, x.size):
            self.clear("No data")
            return
        log_x = bool(log_x) and bool(np.all(x[np.isfinite(x)] > 0))
        self._x, self._y, self._z = x, y, z
        self._x_eval = None if x_eval is None else np.asarray(x_eval, dtype=float).ravel()
        self._ref = None if ref is None else np.asarray(ref, dtype=float)
        if self._ref is not None and self._ref.shape != z.shape:
            self._ref = None
        self._x_label, self._y_label, self._z_label = x_label, y_label, z_label

        if log_x != self._log_x:
            self._log_x = log_x
            for p in (self.mapPlot, self.cutPlot):
                p.setLogMode(x=log_x, y=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._u = np.log10(x) if log_x else x.copy()

        if cmap != self._cmap_name:
            self._cmap_name = cmap
            try:
                self.hist.gradient.setColorMap(pg_cmap(cmap))
            except Exception:
                pass

        self._draw_image(symmetric, center)
        self.mapPlot.setLabel("bottom", x_label)
        self.mapPlot.setLabel("left", y_label)
        self.cutPlot.setLabel("bottom", x_label)
        self.cutPlot.setLabel("left", z_label)
        self.specPlot.setLabel("bottom", z_label)
        self.cutZero.setValue(center if symmetric else 0.0)
        self.specZero.setValue(center if symmetric else 0.0)
        self.cutZero.setVisible(symmetric)
        self.specZero.setVisible(symmetric)
        has_ref = self._ref is not None
        self.cutRef.setVisible(has_ref)
        self.specRef.setVisible(has_ref)
        if has_ref != self._ref_in_legend:
            if has_ref:
                self.cutLegend.addItem(self.cutRef, "numerical (set 1)")
            else:
                self.cutLegend.removeItem(self.cutRef)
            self._ref_in_legend = has_ref

        n_x, n_y = x.size, y.size
        self._ix = n_x // 2 if self._ix is None else max(0, min(self._ix, n_x - 1))
        self._iy = n_y // 2 if self._iy is None else max(0, min(self._iy, n_y - 1))
        self._apply_cursor(emit=False)

    def set_cursor(self, ix: Optional[int] = None, iy: Optional[int] = None, *, emit: bool = False) -> None:
        if self._x is None:
            return
        if ix is not None:
            self._ix = max(0, min(int(ix), self._x.size - 1))
        if iy is not None:
            self._iy = max(0, min(int(iy), self._y.size - 1))
        self._apply_cursor(emit=emit)

    def clear(self, message: str = "") -> None:
        self._x = self._y = self._z = self._ref = self._u = None
        self.imgItem.clear()
        for c in (self.cutCurve, self.cutRef, self.specCurve, self.specRef):
            c.setData([], [])
        self.cursorLbl.setText(message or "Cursor: --")

    # ── drawing ───────────────────────────────────────────────────────

    def _draw_image(self, symmetric: bool, center: float) -> None:
        u, y, z = self._u, self._y, self._z
        ok = np.isfinite(u)
        cols = np.flatnonzero(ok)
        if not cols.size:
            self.imgItem.clear()
            return
        order = cols[np.argsort(u[cols])]
        edges = _column_edges(u[order])
        n_disp = max(MAP_DISPLAY_COLS, 4 * order.size)
        centres = np.linspace(edges[0], edges[-1], n_disp, endpoint=False) + 0.5 * (edges[-1] - edges[0]) / n_disp
        pick = order[np.clip(np.searchsorted(edges, centres, side="right") - 1, 0, order.size - 1)]
        disp = z[:, pick]                                # (n_y, n_disp)

        dy = float(y[1] - y[0]) if y.size > 1 else 1.0
        levels = self._levels(symmetric, center)
        img = disp.T if np.isfinite(disp).any() else np.zeros_like(disp.T)
        self.imgItem.setImage(img, autoLevels=False, levels=levels)
        self.imgItem.setRect(QtCore.QRectF(float(edges[0]), float(y[0] - dy / 2),
                                           float(edges[-1] - edges[0]), float(dy * y.size)))
        self.hist.setLevels(*levels)

    def _levels(self, symmetric: bool, center: float) -> Tuple[float, float]:
        if self.lockLevelsCheck.isChecked():
            lo, hi = self.hist.getLevels()
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return float(lo), float(hi)
        finite = self._z[np.isfinite(self._z)]
        if not finite.size:
            return 0.0, 1.0
        if symmetric:
            half = self._manual_max or float(np.nanpercentile(np.abs(finite - center), 98))
            if not np.isfinite(half) or half <= 0:
                half = float(np.nanmax(np.abs(finite - center))) or 1.0
            return center - half, center + half
        lo, hi = (float(v) for v in np.nanpercentile(finite, [1, 99]))
        if not hi > lo:
            lo, hi = float(finite.min()), float(finite.max())
        if not hi > lo:
            lo, hi = lo - 0.5, lo + 0.5
        return lo, hi

    def _apply_cursor(self, *, emit: bool) -> None:
        if self._x is None or self._ix is None or self._iy is None:
            return
        ix, iy = self._ix, self._iy
        u_sel = float(self._u[ix]) if np.isfinite(self._u[ix]) else float(np.nanmean(self._u))
        y_sel = float(self._y[iy])
        self.vline.setValue(u_sel)
        self.hline.setValue(y_sel)
        self.cutCursor.setValue(u_sel)
        self.specCursor.setValue(y_sel)

        x_meas = self._x_eval if (self._x_eval is not None and self._x_eval.size == self._x.size) else self._x
        self._set_curve(self.cutCurve, x_meas, self._z[iy, :])
        self._set_curve(self.specCurve, self._z[:, ix], self._y)
        if self._ref is not None:
            self._set_curve(self.cutRef, self._x, self._ref[iy, :])
            self._set_curve(self.specRef, self._ref[:, ix], self._y)

        z_sel = self._z[iy, ix]
        self.cursorLbl.setText(
            f"{self._x_label} = {_fmt(self._x[ix])}    {self._y_label} = {_fmt(y_sel)}    "
            f"{self._z_label} = {_fmt(z_sel)}")
        if emit:
            self.cursor_changed.emit(ix, iy)

    def _set_curve(self, curve, a, b) -> None:
        """Plot finite (a, b) pairs; in log-x mode drop non-positive x."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if self._log_x and curve in (self.cutCurve, self.cutRef):
            ok &= a > 0
        curve.setData(a[ok], b[ok])

    # ── interaction ───────────────────────────────────────────────────

    def _nearest_ix(self, u_val: float) -> int:
        d = np.abs(self._u - u_val)
        d[~np.isfinite(d)] = np.inf
        return int(np.argmin(d))

    def _on_vline_moved(self) -> None:
        if self._x is None:
            return
        self._ix = self._nearest_ix(float(self.vline.value()))
        self._apply_cursor(emit=True)

    def _on_hline_moved(self) -> None:
        if self._y is None:
            return
        self._iy = int(np.argmin(np.abs(self._y - float(self.hline.value()))))
        self._apply_cursor(emit=True)

    def _on_scene_click(self, event) -> None:
        if self._x is None or event.button() != QtCore.Qt.LeftButton:
            return
        pos = event.scenePos()
        if not self.mapPlot.sceneBoundingRect().contains(pos):
            return
        pt = self.mapPlot.getViewBox().mapSceneToView(pos)
        self._ix = self._nearest_ix(float(pt.x()))
        self._iy = int(np.argmin(np.abs(self._y - float(pt.y()))))
        self._apply_cursor(emit=True)

    # ── export ────────────────────────────────────────────────────────

    def _on_export_map(self) -> None:
        if self._z is None:
            QtWidgets.QMessageBox.warning(self, "Export map", "No data.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save map CSV", "", "CSV (*.csv)")
        if not path:
            return
        x, y, z = self._x, self._y, self._z
        out = np.full((y.size + 1, x.size + 1), np.nan)
        out[0, 0] = 0.0
        out[0, 1:] = x
        out[1:, 0] = y
        out[1:, 1:] = z
        hdr = f"row 0: {self._x_label}; column 0: {self._y_label}; values: {self._z_label}"
        _save_csv(self, path, out, header=hdr)

    def _on_export_cuts(self) -> None:
        if self._z is None or self._ix is None:
            QtWidgets.QMessageBox.warning(self, "Export linecuts", "No data.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save linecuts (base name)", "", "CSV (*.csv)")
        if not path:
            return
        base = path[:-4] if path.lower().endswith(".csv") else path
        ix, iy = self._ix, self._iy
        ref = self._ref if self._ref is not None else np.full_like(self._z, np.nan)
        x_meas = self._x_eval if (self._x_eval is not None and self._x_eval.size == self._x.size) else self._x
        _save_csv(
            self, f"{base}_vs_sweep.csv",
            np.column_stack([self._x, x_meas, self._z[iy, :], ref[iy, :]]),
            comments="",
            header=f"# {self._y_label} = {_fmt(self._y[iy])}\n"
                   f"{self._x_label},eval position,{self._z_label},numerical reference")
        _save_csv(
            self, f"{base}_spectrum.csv",
            np.column_stack([self._y, self._z[:, ix], ref[:, ix]]),
            comments="",
            header=f"# {self._x_label} = {_fmt(self._x[ix])}\n"
                   f"{self._y_label},{self._z_label},numerical reference")


class RawCurvesPanel(QtWidgets.QGroupBox):
    """I₁ and I₁₊₂ at the cursor: spectra, their difference, and I vs sweep."""

    def __init__(self, parent=None):
        super().__init__("Raw sets at cursor", parent)
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.setContentsMargins(6, 6, 6, 6)
        self.glw = pg.GraphicsLayoutWidget()
        vlay.addWidget(self.glw)

        self.specPlot = self.glw.addPlot(row=0, col=0)
        self.diffPlot = self.glw.addPlot(row=1, col=0)
        self.sweepPlot = self.glw.addPlot(
            row=2, col=0, axisItems={"bottom": SweepAxisItem("bottom"), "left": SweepAxisItem("left")})
        for p in (self.specPlot, self.diffPlot, self.sweepPlot):
            black_axes(p)
            p.showGrid(x=True, y=True, alpha=0.2)
        self.diffPlot.setXLink(self.specPlot)

        self.specPlot.addLegend(offset=(-10, 10))
        self.spec1 = self.specPlot.plot(pen=pg.mkPen(SET1_COLOR, width=1.3), name="I₁ (pump 1)")
        self.spec2 = self.specPlot.plot(pen=pg.mkPen(SET2_COLOR, width=1.3), name="I₁₊₂ (pump 1+2)")
        self.specPlot.setLabel("left", "Counts")

        self.diffCurve = self.diffPlot.plot(pen=pg.mkPen(DIFF_COLOR, width=1.3))
        self.diffPlot.addItem(pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#999", width=0.8)))
        self.diffPlot.setLabel("left", "ΔI (counts)")
        self.specCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(CURSOR_Y_COLOR, width=0.8))
        self.diffCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(CURSOR_Y_COLOR, width=0.8))
        self.specPlot.addItem(self.specCursor)
        self.diffPlot.addItem(self.diffCursor)

        self.sweepPlot.addLegend(offset=(-10, 10))
        self.sweep1 = self.sweepPlot.plot(pen=pg.mkPen(SET1_COLOR, width=1.2), symbol="o", symbolSize=5,
                                          symbolBrush=SET1_COLOR, symbolPen=None, name="I₁")
        self.sweep2 = self.sweepPlot.plot(pen=pg.mkPen(SET2_COLOR, width=1.2), symbol="o", symbolSize=5,
                                          symbolBrush=SET2_COLOR, symbolPen=None, name="I₁₊₂")
        self.sweepPlot.setLabel("left", "Counts")
        self.sweepCursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(CURSOR_X_COLOR, width=0.8))
        self.sweepPlot.addItem(self.sweepCursor)
        self._log = False

    def update_cursor(self, *, x, y, m1, m2, ix: int, iy: int,
                      x_label: str, y_label: str, log_xy: bool) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        c1, c2 = m1[:, ix], m2[:, ix]
        self.spec1.setData(*_finite(y, c1))
        self.spec2.setData(*_finite(y, c2))
        self.diffCurve.setData(*_finite(y, c2 - c1))
        self.specPlot.setTitle(f"Spectra at {x_label} = {_fmt(x[ix])}", size=f"{MIN_PT}pt")
        self.diffPlot.setLabel("bottom", y_label)
        for line in (self.specCursor, self.diffCursor):
            line.setValue(float(y[iy]))

        r1, r2 = m1[iy, :], m2[iy, :]
        log_xy = bool(log_xy) and bool(np.all(x[np.isfinite(x)] > 0))
        if log_xy != self._log:
            self._log = log_xy
            self.sweepPlot.setLogMode(x=log_xy, y=log_xy)
        keep1 = np.isfinite(x) & np.isfinite(r1)
        keep2 = np.isfinite(x) & np.isfinite(r2)
        if log_xy:
            keep1 &= r1 > 0
            keep2 &= r2 > 0
        self.sweep1.setData(x[keep1], r1[keep1])
        self.sweep2.setData(x[keep2], r2[keep2])
        self.sweepPlot.setLabel("bottom", x_label)
        self.sweepPlot.setTitle(f"vs sweep at {y_label} = {_fmt(y[iy])}", size=f"{MIN_PT}pt")
        u = np.log10(x[ix]) if log_xy else x[ix]
        self.sweepCursor.setValue(float(u) if np.isfinite(u) else 0.0)

    def clear(self) -> None:
        for c in (self.spec1, self.spec2, self.diffCurve, self.sweep1, self.sweep2):
            c.setData([], [])


def _save_csv(parent: QtWidgets.QWidget, path: str, data: np.ndarray, **kwargs) -> bool:
    """np.savetxt as UTF-8 with BOM: headers contain P₁, µW, … which the Windows
    default codec cannot encode, and the BOM lets Excel show them correctly."""
    try:
        np.savetxt(path, data, delimiter=",", fmt="%.10g", encoding="utf-8-sig", **kwargs)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(parent, "Export failed", f"{path}\n\n{exc}")
        return False
    return True


def _finite(a: np.ndarray, b: np.ndarray):
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]
