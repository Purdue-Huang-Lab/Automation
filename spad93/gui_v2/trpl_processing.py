"""Processing sub-tab of the TRPL panel: double-pump linecuts (2D mode).

For data with two pump spots (two Gaussians), the map is cut along the line through
the two Gaussian centers.  The linecut is drawn at the times typed in (ns, view time
axis), each one the mean over a time window around that time (one bin by default),
optionally normalized to its maximum.  Pixels within width / 2 of the line are used;
each is placed at its projection on the line (s = 0 at Gaussian 1, s = d at Gaussian 2)
and pixels in the same bin along the line are averaged.

The centers come from a double 2D Gaussian fit of the map summed over the selected
time range (automatic for new data until edited), from the Fitting tab's method 3
result, or are typed in.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from spad93 import spad93_analysis as an
from spad93.spad93_geometry import PIXEL_X_UM, PIXEL_Y_UM
from spad93.spad93_io import save_columns_csv, timestamp
from spad93.gui_v2.config import BTN_W, COMPONENT_COLORS, save_dir
from spad93.gui_v2.map_plots import PixelMapPlot
from spad93.gui_v2.plots import set_title, style_plot, trace_color

Qt = QtCore.Qt
DOUBLE = "2d_double_sequential"
DEFAULT_TIMES = "0, 0.5, 1, 2, 5"


def _dspin(lo: float, hi: float, val: float, dec: int = 2, step: float = 1.0, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    return s


class TrplProcessingTab(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self._centers_manual = False  # True once the user typed / took centers: no automatic refit
        self._auto_key = None  # (raw object, time range) of the last automatic center fit
        self._source = None  # file of the data the centers belong to (a new file re-enables the auto fit)
        self._fit_msg = ""
        self._last: Optional[Tuple[np.ndarray, List[Tuple[float, int, int, np.ndarray]], an.LineCut]] = None
        self._curves: List[pg.PlotDataItem] = []
        self._build_ui()

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Linecut through the two Gaussian centers (double pump)")
        g = QtWidgets.QGridLayout(box)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        g.setContentsMargins(8, 8, 8, 8)

        self.spin_x1, self.spin_y1, self.spin_x2, self.spin_y2 = (
            _dspin(-1000.0, 1000.0, v, 1, 1.0) for v in (100.0, 92.0, 220.0, 92.0))
        self.btn_fit_centers = QtWidgets.QPushButton("Fit centers")
        self.btn_fit_centers.setToolTip(
            "Fit a double 2D Gaussian to the map summed over the selected time range\n"
            "(background as in the Fitting tab) and take its two centers.")
        self.btn_from_fitting = QtWidgets.QPushButton("From Fitting tab")
        self.btn_from_fitting.setToolTip("Take the centers of the Fitting tab's method 3 (double Gaussian) result.")
        g.addWidget(QtWidgets.QLabel("Gaussian 1 x, y (µm)"), 0, 0)
        g.addWidget(self.spin_x1, 0, 1)
        g.addWidget(self.spin_y1, 0, 2)
        g.addWidget(QtWidgets.QLabel("Gaussian 2 x, y (µm)"), 0, 3)
        g.addWidget(self.spin_x2, 0, 4)
        g.addWidget(self.spin_y2, 0, 5)
        g.addWidget(self.btn_fit_centers, 0, 6)
        g.addWidget(self.btn_from_fitting, 0, 7)

        self.edit_times = QtWidgets.QLineEdit(DEFAULT_TIMES)
        self.edit_times.setToolTip("Times in ns (view time axis), separated by commas, spaces or semicolons.\n"
                                   "Press Enter to update.")
        self.spin_window = _dspin(0.0, 1e6, 0.0, 3, 0.05, " ns")
        self.spin_window.setSpecialValueText("1 bin")
        self.spin_window.setToolTip("Average the bins within ± window / 2 of each time (0 = the nearest bin only).")
        g.addWidget(QtWidgets.QLabel("Times (ns)"), 1, 0)
        g.addWidget(self.edit_times, 1, 1, 1, 5)
        g.addWidget(QtWidgets.QLabel("Time window"), 1, 6)
        g.addWidget(self.spin_window, 1, 7)

        self.spin_width = _dspin(1.0, 500.0, 23.0, 1, 1.0, " µm")
        self.spin_width.setToolTip("Pixels whose distance from the line is ≤ width / 2 are used\n"
                                   "(pixel pitch: 19.92 µm along x, 23 µm along y).")
        self.spin_bin = _dspin(0.0, 200.0, 10.0, 1, 1.0, " µm")
        self.spin_bin.setSpecialValueText("per pixel")
        self.spin_bin.setToolTip("Pixels whose positions along the line fall in the same bin of this size are averaged\n"
                                 "(0 = only pixels at the same position).")
        self.chk_norm = QtWidgets.QCheckBox("Normalize (max = 1)")
        self.chk_norm.setToolTip("Divide each linecut by its maximum.")
        self.chk_bg = QtWidgets.QCheckBox("Subtract background (Fitting tab)")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip("Subtract the per-pixel background chosen in the Fitting tab\n"
                               "(pre-t0 offset or dark reference) before cutting.")
        self.btn_copy = QtWidgets.QPushButton("Copy")
        self.btn_copy.setToolTip("Copy the linecuts as tab-separated columns (s, then one column per time).")
        self.btn_export = QtWidgets.QPushButton("Export…")
        for b in (self.btn_fit_centers, self.btn_from_fitting, self.btn_copy, self.btn_export):
            b.setMinimumWidth(BTN_W - 30)
        g.addWidget(QtWidgets.QLabel("Linecut width"), 2, 0)
        g.addWidget(self.spin_width, 2, 1)
        g.addWidget(QtWidgets.QLabel("Bin along line"), 2, 2)
        g.addWidget(self.spin_bin, 2, 3)
        g.addWidget(self.chk_norm, 2, 4)
        g.addWidget(self.chk_bg, 2, 5)
        g.addWidget(self.btn_copy, 2, 6)
        g.addWidget(self.btn_export, 2, 7)
        g.setColumnStretch(8, 1)
        root.addWidget(box)

        self.lbl_state = QtWidgets.QLabel("")
        self.lbl_state.setWordWrap(True)
        root.addWidget(self.lbl_state)

        split = QtWidgets.QSplitter(Qt.Horizontal)
        self.map = PixelMapPlot("Map summed over the time range")
        self.map.set_show_ids(False)
        self.line_item = pg.PlotCurveItem(pen=pg.mkPen("#111827", width=1.6))
        self.band_items = [pg.PlotCurveItem(pen=pg.mkPen("#111827", width=1.0, style=Qt.DashLine)) for _ in range(2)]
        self.center_items = []
        for color in COMPONENT_COLORS:
            self.center_items.append(pg.ScatterPlotItem(symbol="x", size=16, pxMode=True,
                                                        pen=pg.mkPen(color, width=3), brush=pg.mkBrush(color)))
        for it in [self.line_item, *self.band_items, *self.center_items]:
            it.setZValue(10)
            self.map.plot.addItem(it, ignoreBounds=True)
        split.addWidget(self.map)

        self.pw = pg.PlotWidget()
        self.plot = self.pw.getPlotItem()
        style_plot(self.plot)
        self.plot.setLabel("bottom", "Position along the line, s (µm; 0 = Gaussian 1)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.vlines = []
        for k, color in enumerate(COMPONENT_COLORS):
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color, width=1.4, style=Qt.DashLine),
                                 label=f"G{k + 1}", labelOpts=dict(position=0.95, color=color))
            ln.hide()
            self.plot.addItem(ln, ignoreBounds=True)
            self.vlines.append(ln)
        self.zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen("#9ca3af", width=1, style=Qt.DotLine))
        self.plot.addItem(self.zero, ignoreBounds=True)
        split.addWidget(self.pw)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        root.addWidget(split, 1)

        for s in (self.spin_x1, self.spin_y1, self.spin_x2, self.spin_y2):
            s.valueChanged.connect(self._on_centers_edited)
        for s in (self.spin_window, self.spin_width, self.spin_bin):
            s.valueChanged.connect(lambda _v: self.refresh())
        self.edit_times.editingFinished.connect(self.refresh)
        self.chk_norm.toggled.connect(lambda _v: self.refresh())
        self.chk_bg.toggled.connect(lambda _v: self.refresh())
        self.btn_fit_centers.clicked.connect(self._on_fit_centers)
        self.btn_from_fitting.clicked.connect(self._on_from_fitting)
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_export.clicked.connect(self._on_export)

    # ------------------------------------------------------- helpers
    def invalidate(self) -> None:
        self._last = None

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def centers(self) -> Tuple[float, float, float, float]:
        return (self.spin_x1.value(), self.spin_y1.value(), self.spin_x2.value(), self.spin_y2.value())

    def _set_centers(self, c, manual: bool) -> None:
        for sp, v in zip((self.spin_x1, self.spin_y1, self.spin_x2, self.spin_y2), c):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        self._centers_manual = manual

    def _on_centers_edited(self, _v: float) -> None:
        self._centers_manual = True
        self.refresh()

    def _background(self) -> Tuple[Optional[np.ndarray], str]:
        """Per-pixel background (counts per bin) from the Fitting tab's choice, if enabled."""
        if not self.chk_bg.isChecked():
            return None, "no background subtracted"
        fit = self.panel.fitting
        dark = fit._dark_2d()
        if dark is None:
            return None, "no background subtracted (Fitting tab background = none)"
        return dark, fit._background()[1]

    def _summed_map(self, dark: Optional[np.ndarray]) -> np.ndarray:
        p = self.panel
        r0, r1 = p.range
        frames = np.asarray(p.view_counts[r0 : r1 + 1], dtype=np.float64)
        total = frames.sum(axis=0)
        if dark is not None:
            total = total - dark * frames.shape[0]
        return total

    def _fit_centers(self, quiet: bool) -> bool:
        p = self.panel
        if not p.has_data or p.state.spectral:
            return False
        mask = p.state.enabled_mask()
        if int(mask.sum()) < an.MIN_FIT_PIXELS_DOUBLE:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Fit centers", "Not enough enabled pixels for a double fit.")
            return False
        dark, _ = self._background()
        z = self._summed_map(dark)
        fit = an.fit_gaussian_2d_double(PIXEL_X_UM[mask], PIXEL_Y_UM[mask], z[mask],
                                        fix_offset=p.fitting._fix_offset())
        if fit is None:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Fit centers", "The double Gaussian fit failed.")
            return False
        self._set_centers([fit.popt[i] for i in (1, 2, 6, 7)], manual=False)
        msg = "centers fitted: " + "; ".join(
            f"G{k + 1} ({fit.component(k)[1]:.1f}, {fit.component(k)[2]:.1f}) µm, σx={fit.component(k)[3]:.1f}, "
            f"σy={fit.component(k)[4]:.1f}" for k in range(2))
        if not fit.resolved:
            msg += " — second Gaussian not resolved: check the centers"
        self._fit_msg = msg
        return True

    def _on_fit_centers(self) -> None:
        if self._fit_centers(quiet=False):
            self._auto_key = self._data_key()
            self.refresh()

    def _on_from_fitting(self) -> None:
        f = self.panel.fitting
        res = f._result
        if res is None or res.method != DOUBLE or f._result_version != self.panel.data_version:
            QtWidgets.QMessageBox.information(self, "From Fitting tab",
                                              "No current method 3 (double Gaussian) Fit All result.")
            return
        if res.centers is not None:
            c = res.centers
        else:
            c = [float(res.ref_fit.popt[i]) for i in (1, 2, 6, 7)]
        self._set_centers(c, manual=True)
        self._fit_msg = "centers from the Fitting tab (method 3 reference fit)"
        self.refresh()

    def _data_key(self):
        p = self.panel
        return (id(p.raw), p.range)

    # ------------------------------------------------------- refresh
    def refresh(self) -> None:
        if not self.isVisible():
            return
        p = self.panel
        self._last = None
        if p.state.spectral:
            self._clear("Processing works in 2D mode (switch the mode to 2D space).")
            return
        if not p.has_data:
            self._clear("No data loaded.")
            return
        src = p.raw.source if p.raw is not None else ""
        if src != self._source:
            self._source = src
            self._centers_manual = False
        if not self._centers_manual and self._auto_key != self._data_key():
            self._auto_key = self._data_key()
            self._fit_centers(quiet=True)
        mask = p.state.enabled_mask()
        dark, bg_text = self._background()
        summed = self._summed_map(dark)
        self.map.set_excluded(p.state.excluded)
        self.map.set_values(summed, unit="Counts" if dark is None else "Counts − bg")
        c = self.centers()
        p1, p2 = (c[0], c[1]), (c[2], c[3])
        width = self.spin_width.value()
        cut = an.line_cut(PIXEL_X_UM, PIXEL_Y_UM, p1, p2, width, self.spin_bin.value(), mask)
        self._draw_line(p1, p2, width)
        if cut is None:
            self.map.set_selected([])
            self._clear_curves()
            self.lbl_state.setText("The two centers coincide or no enabled pixel is inside the band: "
                                   "change the centers or widen the linecut.")
            return
        self.map.set_selected(cut.band)
        times = an.parse_times(self.edit_times.text())
        rows = []
        skipped = []
        for t in times:
            out = an.time_window_frame(p.view_counts, p.view_time, t, self.spin_window.value())
            if out is None:
                skipped.append(t)
                continue
            frame, i0, i1 = out
            if dark is not None:
                frame = frame - dark
            rows.append((t, i0, i1, cut.values(frame)))
        self._last = (cut.s_um, rows, cut)
        self._draw_curves(cut, rows)
        parts = [f"d = {cut.length_um:.1f} µm", f"{cut.band.size} pixels in the band, {cut.s_um.size} points",
                 bg_text]
        if skipped:
            parts.append("outside the time axis: " + ", ".join(f"{t:g}" for t in skipped) + " ns")
        if not times:
            parts.append("type times (ns) to draw linecuts")
        msg = self._fit_msg
        self.lbl_state.setText((msg + " | " if msg else "") + "; ".join(parts))

    def _draw_line(self, p1, p2, width: float) -> None:
        d = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
        L = float(np.hypot(*d))
        for k, (it, pt) in enumerate(zip(self.center_items, (p1, p2))):
            it.setData([pt[0]], [pt[1]])
        if not L > 0:
            for it in [self.line_item, *self.band_items]:
                it.setData([], [])
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        s = np.array([-600.0, L + 600.0])
        base_x, base_y = p1[0] + s * u[0], p1[1] + s * u[1]
        self.line_item.setData(base_x, base_y)
        for sign, it in zip((1, -1), self.band_items):
            off = sign * 0.5 * width
            it.setData(base_x + off * n[0], base_y + off * n[1])

    def _clear_curves(self) -> None:
        for c in self._curves:
            self.plot.removeItem(c)
        self._curves = []
        self.legend.clear()
        for ln in self.vlines:
            ln.hide()

    def _draw_curves(self, cut: an.LineCut, rows) -> None:
        self._clear_curves()
        norm = self.chk_norm.isChecked()
        p = self.panel
        for k, (t, i0, i1, v) in enumerate(rows):
            y = v
            if norm:
                m = float(np.nanmax(v)) if v.size else 0.0
                y = v / m if m > 0 else np.full_like(v, np.nan)
            color = trace_color(k)
            if i1 > i0:
                label = f"t = {t:g} ns ({p.time_of(i0):.3f}–{p.time_of(i1):.3f})"
            else:
                label = f"t = {t:g} ns (bin {p.time_of(i0):.3f})"
            curve = self.plot.plot(cut.s_um, y, pen=pg.mkPen(color, width=1.6), symbol="o", symbolSize=6,
                                   symbolBrush=color, symbolPen=None, name=label)
            self._curves.append(curve)
        for ln, pos in zip(self.vlines, (0.0, cut.length_um)):
            ln.setPos(pos)
            ln.show()
        dark, _ = self._background()
        ylabel = "Normalized intensity" if norm else ("Counts per bin per pixel" + (" − background" if dark is not None else ""))
        self.plot.setLabel("left", ylabel)
        c = self.centers()
        set_title(self.plot, "Linecuts between the two Gaussians",
                  f"G1 ({c[0]:.1f}, {c[1]:.1f}) → G2 ({c[2]:.1f}, {c[3]:.1f}) µm; width {self.spin_width.value():.1f} µm"
                  + ("; normalized to max" if norm else ""))

    def _clear(self, message: str) -> None:
        self._clear_curves()
        set_title(self.plot, "Linecuts between the two Gaussians")
        self.lbl_state.setText(message)

    # ------------------------------------------------------- output
    def _columns(self):
        if self._last is None or not self._last[1]:
            return None
        s, rows, cut = self._last
        norm = self.chk_norm.isChecked()
        cols = {"s_um": s}
        for t, i0, i1, v in rows:
            y = v
            if norm:
                m = float(np.nanmax(v)) if v.size else 0.0
                y = v / m if m > 0 else np.full_like(v, np.nan)
            name = f"t={t:g}ns"
            while name in cols:
                name += "_"
            cols[name] = y
        cols["n_pixels"] = np.array([ids.size for ids in cut.pixels], dtype=np.float64)
        return cols

    def _on_copy(self) -> None:
        cols = self._columns()
        if cols is None:
            self.panel.status.emit("No linecuts to copy.")
            return
        names = list(cols)
        arrays = [np.asarray(cols[n], dtype=np.float64) for n in names]
        lines = ["\t".join(names)] + ["\t".join(f"{a[i]:.8g}" for a in arrays) for i in range(arrays[0].size)]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines) + "\n")
        self.panel.status.emit(f"Copied {len(names) - 2} linecut(s) to the clipboard.")

    def _on_export(self) -> None:
        cols = self._columns()
        if cols is None:
            QtWidgets.QMessageBox.information(self, "Export linecuts", "No linecuts to export.")
            return
        p = self.panel
        s, rows, cut = self._last
        c = self.centers()
        meta = {"device": "SPAD93", "source": p.raw.source if p.raw else "",
                "gaussian_1_um": [c[0], c[1]], "gaussian_2_um": [c[2], c[3]], "separation_um": cut.length_um,
                "s_definition": "projection on the line G1->G2, 0 at G1",
                "linecut_width_um": self.spin_width.value(), "bin_along_line_um": self.spin_bin.value(),
                "time_window_ns": self.spin_window.value(), "normalized": self.chk_norm.isChecked(),
                "background": self._background()[1], "excluded_pixels": sorted(p.state.excluded),
                "time_bins_used": {f"{t:g}": [p.time_of(i0), p.time_of(i1)] for t, i0, i1, _ in rows},
                "pixels_per_point": [[int(q) for q in ids] for ids in cut.pixels]}
        meta.update(p.processing_metadata())
        base = os.path.splitext(os.path.basename(p.raw.source))[0] if (p.raw and p.raw.source) else f"spad93_trpl_{timestamp()}"
        default = os.path.join(save_dir("trpl"), f"{base}_double_linecuts.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export linecuts", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            save_columns_csv(path, cols, meta)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export linecuts", f"Could not save:\n{exc}")
            return
        p.status.emit(f"Linecuts exported: {path}")
