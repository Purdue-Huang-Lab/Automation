"""Counts tab: live count map with the 2D-space and spectral analysis pages.

2D: the map is fitted with one 2D Gaussian, or with two round ones on a shared
offset ("Double Gaussian" model, ported from spad23/gui_v2).
"""

import os
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from spad93.spad93_analysis import (
    MIN_FIT_PIXELS_DOUBLE,
    DoubleFitResult,
    auto_rate_scale,
    fit_column_profile,
    fit_gaussian_2d_double,
    fit_gaussian_2d_warm,
    format_rate,
    frame_background,
    locate_peak,
)
from spad93.spad93_geometry import (
    CENTER_PIXEL,
    CENTER_Y_UM,
    COLUMN_X_UM,
    PIXEL_COL,
    PIXEL_X_UM,
    PIXEL_Y_UM,
    column_of,
    column_profile,
    row_profile,
)
from spad93.spad93_io import save_counts_snapshot, save_spectrum_csv, timestamp
from spad93.spad93_wrapper import CountSnapshot
from spad93.gui_v2.calib_panel import CalibPanel
from spad93.gui_v2.config import BTN_W, LIMIT_WARN_RATIO, save_dir
from spad93.gui_v2.map_plots import Gauss2DFitPlot, PixelMapPlot
from spad93.gui_v2.plots import (
    ColumnProfilePlot,
    DoubleRadialPlot,
    LinecutPlot,
    RadialFitPlot,
    SpectrumPlot,
    double_fit_text,
    fwhm_text,
)
from spad93.gui_v2.state import SharedState

MODEL_SINGLE = "single"
MODEL_DOUBLE = "double"


class CountsPanel(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)

    def __init__(self, state: SharedState, live_running, parent=None):
        super().__init__(parent)
        self.state = state
        self._live_running = live_running
        self._latest: Optional[CountSnapshot] = None
        self._redraw_pending = False
        self._selected_pixel = CENTER_PIXEL
        self._selected_col = int(PIXEL_COL[CENTER_PIXEL])
        self._prev_popt = None
        self._prev_double = None  # (popt, rate factor) of the last good double fit: live warm start
        self._last_double: Optional[DoubleFitResult] = None
        self._last_unit = ""
        self._hover = -1
        self._build_ui()
        self.state.excluded_changed.connect(self._on_excluded_changed)
        self.state.calibration_changed.connect(self._on_calibration_changed)
        self.state.mode_changed.connect(self._on_mode_changed)
        self._on_excluded_changed()
        self._on_mode_changed(self.state.mode)

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.map = PixelMapPlot("SPAD93 count-rate map")
        self.map.hover_text_fn = self._hover_text
        lv.addWidget(self.map, 1)
        self.lbl_info = QtWidgets.QLabel("Total: --")
        self.lbl_info.setWordWrap(True)
        lv.addWidget(self.lbl_info)
        row = QtWidgets.QHBoxLayout()
        self.chk_ids = QtWidgets.QCheckBox("Show pixel IDs")
        self.chk_ids.setChecked(True)
        self.btn_clear_ex = QtWidgets.QPushButton("Clear exclusions")
        self.btn_snapshot = QtWidgets.QPushButton("Save snapshot…")
        for b in (self.btn_clear_ex, self.btn_snapshot):
            b.setMinimumWidth(BTN_W)
        row.addWidget(self.chk_ids)
        row.addWidget(self.btn_clear_ex)
        row.addStretch(1)
        row.addWidget(self.btn_snapshot)
        lv.addLayout(row)
        hint = QtWidgets.QLabel("Left click: select pixel (2D) / column (spectral) · Right click: exclude pixel")
        hint.setStyleSheet("color:#6b7280")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        splitter.addWidget(left)

        self.pages = QtWidgets.QStackedWidget()
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([760, 980])

        # 2D space page
        page2d = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(page2d)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("Fit model"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItem("Single 2D Gaussian", MODEL_SINGLE)
        self.cmb_model.addItem("Double 2D Gaussian", MODEL_DOUBLE)
        self.cmb_model.setToolTip(
            "Double: two round Gaussians (A, x0, y0, σ each) on one shared offset, fitted to the whole map.\n"
            "The two spots are only resolved if they are about a pixel pitch or more apart;\n"
            "the subtitle says when the second Gaussian is not resolved.")
        model_row.addWidget(self.cmb_model)
        model_row.addStretch(1)
        g.addLayout(model_row, 0, 0, 1, 2)
        self.fit_plot = Gauss2DFitPlot()
        self.radial_plot = RadialFitPlot()
        self.pixel_fit_plot = DoubleRadialPlot()
        self.side_stack = QtWidgets.QStackedWidget()
        self.side_stack.addWidget(self.radial_plot)
        self.side_stack.addWidget(self.pixel_fit_plot)
        self.hcut_plot = LinecutPlot("#2563eb")
        self.vcut_plot = LinecutPlot("#d97706")
        g.addWidget(self.fit_plot, 1, 0)
        g.addWidget(self.side_stack, 1, 1)
        g.addWidget(self.hcut_plot, 2, 0)
        g.addWidget(self.vcut_plot, 2, 1)
        g.setRowStretch(1, 1)
        g.setRowStretch(2, 1)
        self.pages.addWidget(page2d)

        # spectral page
        pagesp = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        th = QtWidgets.QHBoxLayout(top)
        th.setContentsMargins(0, 0, 0, 0)
        self.spec_plot = SpectrumPlot()
        th.addWidget(self.spec_plot, 3)
        colbox = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(colbox)
        cv.setContentsMargins(0, 0, 0, 0)
        self.col_plot = ColumnProfilePlot()
        cv.addWidget(self.col_plot, 1)
        self.chk_free_offset = QtWidgets.QCheckBox("Fit the offset freely (columns with ≥ 7 pixels)")
        cv.addWidget(self.chk_free_offset)
        self.chk_center_row = QtWidgets.QCheckBox(f"Fix y0 at the center row (y = {CENTER_Y_UM:g} µm)")
        cv.addWidget(self.chk_center_row)
        th.addWidget(colbox, 2)
        pagesp.addWidget(top)
        self.calib = CalibPanel(self.state, self._live_running)
        pagesp.addWidget(self.calib)
        pagesp.setStretchFactor(0, 3)
        pagesp.setStretchFactor(1, 2)
        self.pages.addWidget(pagesp)

        self.map.pixel_clicked.connect(self._on_pixel_clicked)
        self.map.hovered.connect(self._on_hovered)
        self.spec_plot.column_clicked.connect(self._select_column)
        self.chk_ids.toggled.connect(self.map.set_show_ids)
        self.cmb_model.currentIndexChanged.connect(lambda _i: self._on_model_changed())
        self.chk_free_offset.toggled.connect(lambda _v: self._schedule_redraw())
        self.chk_center_row.toggled.connect(lambda _v: self._schedule_redraw())
        self.btn_clear_ex.clicked.connect(lambda: self.state.set_excluded(()))
        self.btn_snapshot.clicked.connect(self._on_save_snapshot)
        self.calib.status.connect(self.status)
        self.map.set_selected([self._selected_pixel])

    # ----------------------------------------------------------- frames
    def on_frame(self, snap: CountSnapshot) -> None:
        self._latest = snap
        self.calib.on_frame(snap)
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            QtCore.QTimer.singleShot(0, self._redraw)

    def _redraw(self) -> None:
        self._redraw_pending = False
        snap = self._latest
        if snap is None:
            return
        try:
            mask = self.state.enabled_mask()
            rates = snap.pixel_rates_mcps
            unit, factor = auto_rate_scale(rates[mask] if mask.any() else rates)
            vals = rates * factor
            self.map.set_values(vals, unit=f"Rate ({unit})")
            self._update_info()
            if not self.isVisible():
                return
            if self.state.spectral:
                self._update_spectral(vals, unit, mask)
            elif self.model == MODEL_DOUBLE:
                self._update_2d_double(vals, unit, factor, mask)
                self._update_linecuts(vals, unit)
            else:
                self._update_2d(vals, unit, mask)
                self._update_linecuts(vals, unit)
        except Exception as exc:  # never let a display error kill the live view
            self.status.emit(f"Display error: {exc}")

    def _update_2d(self, vals: np.ndarray, unit: str, mask: np.ndarray) -> None:
        ex = self.state.excluded
        fit = None
        if int(mask.sum()) >= 6:
            fit = fit_gaussian_2d_warm(PIXEL_X_UM[mask], PIXEL_Y_UM[mask], vals[mask], self._prev_popt)
        if fit is not None and not fit.quality_ok:
            note = f"low signal (amplitude {fit.snr:.1f} σ)"
            self._prev_popt = None
        else:
            note = ""
            self._prev_popt = None if fit is None else fit.popt
        if fit is None:
            self.fit_plot.set_data(vals, None, ex, "2D Gaussian fit",
                                   message="Fit unavailable" if mask.sum() >= 6 else "Fewer than 6 enabled pixels")
            self.radial_plot.clear("Fit unavailable")
        else:
            p = fit.popt
            sub = f"x0={p[1]:.1f} µm, y0={p[2]:.1f} µm, σ_eq={fit.sigma_eq:.1f} µm" + (f" — {note}" if note else "")
            self.fit_plot.set_data(vals, fit, ex, "2D Gaussian fit", sub)
            r = np.sqrt((PIXEL_X_UM[mask] - p[1]) ** 2 + (PIXEL_Y_UM[mask] - p[2]) ** 2)
            order = np.argsort(r)
            r_sorted, z_sorted = r[order], vals[mask][order]
            rmax = float(np.max(r)) if r.size else 1.0
            r_line = np.linspace(-rmax, rmax, 401)
            fit_line = p[0] * np.exp(-(r_line**2) / (2.0 * max(fit.sigma_eq, 1e-6) ** 2)) + p[5]
            self.radial_plot.set_data(
                np.concatenate([-r_sorted[::-1], r_sorted]), np.concatenate([z_sorted[::-1], z_sorted]),
                r_line, fit_line, "Gaussian radial fit", f"σ = {fit.sigma_eq:.1f} µm", f"Intensity ({unit})",
            )

    @property
    def model(self) -> str:
        return str(self.cmb_model.currentData())

    def _on_model_changed(self) -> None:
        self._prev_popt = None
        self._prev_double = None
        self._last_double = None
        self.side_stack.setCurrentIndex(1 if self.model == MODEL_DOUBLE else 0)
        self._schedule_redraw()

    def _update_2d_double(self, vals: np.ndarray, unit: str, factor: float, mask: np.ndarray) -> None:
        ex = self.state.excluded
        fit = None
        enough = int(mask.sum()) >= MIN_FIT_PIXELS_DOUBLE
        if enough:
            p0 = None
            if self._prev_double is not None:
                p0 = np.array(self._prev_double[0], dtype=np.float64)
                p0[[0, 4, 8]] *= factor / self._prev_double[1]  # the display unit may have changed
            fit = fit_gaussian_2d_double(PIXEL_X_UM[mask], PIXEL_Y_UM[mask], vals[mask], p0=p0, keep_order_of=p0)
        self._prev_double = (fit.popt, factor) if fit is not None and fit.quality_ok else None
        self._last_double, self._last_unit = fit, unit
        ylabel = f"Intensity ({unit})"
        if fit is None:
            msg = "Fit unavailable" if enough else f"Fewer than {MIN_FIT_PIXELS_DOUBLE} enabled pixels"
            self.fit_plot.set_data_double(vals, None, ex, "Double 2D Gaussian fit", message=msg)
            self.pixel_fit_plot.set_data(vals, None, ex, ylabel, "Radial profiles of the two Gaussians", message=msg)
            return
        self.fit_plot.set_data_double(vals, fit, ex, "Double 2D Gaussian fit", double_fit_text(fit))
        _, _, _, s1 = fit.component(0)
        _, _, _, s2 = fit.component(1)
        self.pixel_fit_plot.set_data(vals, fit, ex, ylabel, "Radial profiles of the two Gaussians",
                                     f"σ1 = {abs(s1):.1f} µm, σ2 = {abs(s2):.1f} µm, offset {fit.offset:.3g}")

    def _update_linecuts(self, vals: np.ndarray, unit: str) -> None:
        ex = self.state.excluded
        y_sel = float(PIXEL_Y_UM[self._selected_pixel])
        h = row_profile(vals, y_sel, ex)
        self.hcut_plot.set_data(COLUMN_X_UM, h, "x (µm)", f"Intensity ({unit})",
                                "Horizontal linecut", f"y = {y_sel:.1f} µm (through pixel {self._selected_pixel})")
        col = column_of(self._selected_pixel)
        _, ys, zs = column_profile(vals, col, ex)
        self.vcut_plot.set_data(ys, zs, "y (µm)", f"Intensity ({unit})",
                                "Vertical linecut", f"column {col}, x = {COLUMN_X_UM[col]:.1f} µm")

    def _update_spectral(self, vals: np.ndarray, unit: str, mask: np.ndarray) -> None:
        ex = self.state.excluded
        calib = self.state.calibration
        prof = row_profile(vals, CENTER_Y_UM, ex)
        peak = locate_peak(COLUMN_X_UM, prof) if np.isfinite(prof).sum() >= 4 else None
        if peak is None:
            sub = "no peak fit"
        else:
            where = f"{float(calib.wavelength(peak.center)):.2f} nm" if calib else f"x = {peak.center:.1f} µm"
            sub = f"peak {where}, {fwhm_text(peak.sigma, calib)}"
            if peak.flags:
                sub += f" ({', '.join(peak.flags)})"
        self.spec_plot.set_data(prof, peak, f"Intensity ({unit})", "Center-row spectrum", sub)
        col = self._selected_col
        bg = frame_background(vals, mask)
        y, z, cfit = fit_column_profile(vals, col, ex, bg, self.chk_free_offset.isChecked(),
                                        CENTER_Y_UM if self.chk_center_row.isChecked() else None)
        where = f"{float(calib.wavelength(COLUMN_X_UM[col])):.2f} nm" if calib else f"x = {COLUMN_X_UM[col]:.1f} µm"
        if cfit is None:
            sub = "too few pixels" if y.size < 3 else "fit failed"
        else:
            sub = (f"y0 = {cfit.center:.1f} µm{' (fixed)' if self.chk_center_row.isChecked() else ''}, "
                   f"σ = {cfit.sigma:.1f} µm, FWHM = {cfit.fwhm:.1f} µm")
            if cfit.flags:
                sub += f" ({', '.join(cfit.flags)})"
        self.col_plot.set_data(y, z, cfit, f"Column {col} y-linecut ({where})", sub, f"Intensity ({unit})")

    # ------------------------------------------------------------ info
    def _hover_text(self, pix: int) -> str:
        snap = self._latest
        if snap is None:
            return ""
        return f"{int(snap.pixel_counts[pix])} counts\n{format_rate(float(snap.pixel_rates_mcps[pix]))}"

    def _on_hovered(self, pix: int) -> None:
        self._hover = pix
        self._update_info()

    def _update_info(self) -> None:
        snap = self._latest
        if snap is None:
            self.lbl_info.setText("Total: --")
            return
        ratio = snap.overload_ratio
        if self._hover < 0:
            text = (f"Total {format_rate(snap.total_rate_mcps)} | Max pixel {snap.max_pixel_index}: "
                    f"{format_rate(snap.max_pixel_rate_mcps)} | Limit ratio {100.0 * ratio:.1f}%")
        else:
            p = self._hover
            text = (f"Pixel {p} (column {int(PIXEL_COL[p])}): {int(snap.pixel_counts[p])} counts, "
                    f"{format_rate(float(snap.pixel_rates_mcps[p]))} | Total {format_rate(snap.total_rate_mcps)}")
        if snap.warnings:
            text += f" | {snap.warnings[0]}"
        self.lbl_info.setText(text)
        self.lbl_info.setStyleSheet("color:#b91c1c" if ratio >= LIMIT_WARN_RATIO else "")

    # ---------------------------------------------------------- clicks
    def _on_pixel_clicked(self, pix: int, button: int) -> None:
        if button == 2:
            self.state.toggle_excluded(pix)
            return
        if self.state.spectral:
            self._select_column(column_of(pix))
        else:
            self._selected_pixel = pix
            self.map.set_selected([pix])
            self._schedule_redraw()

    def _select_column(self, col: int) -> None:
        self._selected_col = int(col)
        self.map.set_highlight_columns([self._selected_col])
        self.spec_plot.set_selected_column(self._selected_col)
        self._schedule_redraw()

    # ---------------------------------------------------- state changes
    def _on_excluded_changed(self) -> None:
        self.map.set_excluded(self.state.excluded)
        self._prev_popt = None
        self._prev_double = None
        self._schedule_redraw()

    def _on_calibration_changed(self) -> None:
        calib = self.state.calibration
        self.spec_plot.set_calibration(calib)
        self.map.set_spectral(self.state.spectral, calib)
        if self.state.spectral:
            self.map.set_highlight_columns([self._selected_col])
        self._schedule_redraw()

    def _on_mode_changed(self, mode: str) -> None:
        spectral = self.state.spectral
        self.pages.setCurrentIndex(1 if spectral else 0)
        self.map.set_spectral(spectral, self.state.calibration)
        if spectral:
            self.map.set_selected([])
            self.map.set_highlight_columns([self._selected_col])
            self.spec_plot.set_selected_column(self._selected_col)
        else:
            self.map.set_selected([self._selected_pixel])
        self._schedule_redraw()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._schedule_redraw()

    def sync_controls(self, live: bool) -> None:
        self.calib.sync_controls(live)

    # ------------------------------------------------------------ save
    def _on_save_snapshot(self) -> None:
        snap = self._latest
        if snap is None:
            QtWidgets.QMessageBox.information(self, "Save snapshot", "No count frame yet.")
            return
        default = os.path.join(save_dir("counts"), f"spad93_counts_{timestamp()}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save count snapshot", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        calib = self.state.calibration
        meta = {
            "device": "SPAD93",
            "integration_ms": snap.integration_ms,
            "mode": self.state.mode,
            "excluded_pixels": sorted(self.state.excluded),
            "wavelength_calibration": None if calib is None else calib.to_dict(),
        }
        fit = self._last_double
        if not self.state.spectral and self.model == MODEL_DOUBLE and fit is not None:
            comps = {}
            for k in range(2):
                A, x0, y0, s = fit.component(k)
                comps[f"gaussian_{k + 1}"] = {"amplitude": A, "x0_um": x0, "y0_um": y0, "sigma_um": abs(s)}
            meta["double_gaussian_fit"] = dict(comps, offset=fit.offset, unit=self._last_unit,
                                               separation_um=fit.separation, resolved=bool(fit.resolved),
                                               amplitude_significance=[fit.snr1, fit.snr2])
        try:
            rates_cps = snap.pixel_rates_mcps * 1e6
            save_counts_snapshot(path, snap.pixel_counts, rates_cps, meta)
            msg = f"Snapshot saved: {path}"
            if self.state.spectral:
                spec = row_profile(rates_cps, CENTER_Y_UM, self.state.excluded)
                lam = None if calib is None else calib.column_wavelengths()
                spec_path = os.path.splitext(path)[0] + "_spectrum.csv"
                save_spectrum_csv(spec_path, spec, lam, meta)
                msg += f" (+ {os.path.basename(spec_path)})"
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save snapshot", f"Could not save:\n{exc}")
            return
        self.status.emit(msg)
