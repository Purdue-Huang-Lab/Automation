"""Lifetime sub-tab of the TRPL panel: baseline subtraction + multi-exponential fits in time.

Each selected pixel (in both 2D and spectral mode),
or their sum, gets a constant baseline removed (its mean over a window before
time zero) and is fitted with a sum of N exponentials convolved with a
Gaussian IRF (exponentially modified Gaussians), weighted by Poisson errors.
"""

import csv
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from spad93 import spad93_analysis as an
from spad93.spad93_io import timestamp
from spad93.gui_v2.config import BTN_W, save_dir
from spad93.gui_v2.plots import set_title, style_plot, trace_color
from spad93.gui_v2.workers import FitAllWorker

Qt = QtCore.Qt


def _spin(decimals: int = 3) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(-1e6, 1e9)
    s.setDecimals(decimals)
    s.setSingleStep(0.1)
    s.setKeyboardTracking(False)
    return s


class TrplLifetimeTab(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self._results: List[an.DecayFitResult] = []
        self._result_key = None
        self._worker: Optional[FitAllWorker] = None
        self._job_id = 0
        self._curves: List[pg.PlotDataItem] = []
        self._auto_busy = False
        self._build_ui()

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Baseline and fit")
        g = QtWidgets.QGridLayout(box)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        g.setContentsMargins(8, 8, 8, 8)
        self.spin_b0, self.spin_b1, self.spin_f0, self.spin_f1 = (_spin() for _ in range(4))
        self.btn_auto = QtWidgets.QPushButton("Auto windows")
        self.spin_nexp = QtWidgets.QSpinBox()
        self.spin_nexp.setRange(1, 4)
        self.spin_nexp.setValue(2)
        self.cmb_traces = QtWidgets.QComboBox()
        self.cmb_traces.addItem("Each selected trace", "each")
        self.cmb_traces.addItem("Sum of selected traces", "sum")
        self.chk_offset = QtWidgets.QCheckBox("Fit residual offset")
        self.chk_log = QtWidgets.QCheckBox("Log y")
        self.chk_log.setChecked(True)
        self.chk_comp = QtWidgets.QCheckBox("Show components")
        self.chk_comp.setChecked(True)
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_export = QtWidgets.QPushButton("Export…")
        for b in (self.btn_auto, self.btn_fit, self.btn_export):
            b.setMinimumWidth(BTN_W - 30)
        g.addWidget(QtWidgets.QLabel("Baseline from (ns)"), 0, 0)
        g.addWidget(self.spin_b0, 0, 1)
        g.addWidget(QtWidgets.QLabel("to"), 0, 2)
        g.addWidget(self.spin_b1, 0, 3)
        g.addWidget(QtWidgets.QLabel("Fit from (ns)"), 0, 4)
        g.addWidget(self.spin_f0, 0, 5)
        g.addWidget(QtWidgets.QLabel("to"), 0, 6)
        g.addWidget(self.spin_f1, 0, 7)
        g.addWidget(self.btn_auto, 0, 8)
        g.addWidget(QtWidgets.QLabel("Exponentials"), 1, 0)
        g.addWidget(self.spin_nexp, 1, 1)
        g.addWidget(QtWidgets.QLabel("Traces"), 1, 2)
        g.addWidget(self.cmb_traces, 1, 3)
        g.addWidget(self.chk_offset, 1, 4, 1, 2)
        g.addWidget(self.chk_log, 1, 6)
        g.addWidget(self.chk_comp, 1, 7)
        g.addWidget(self.btn_fit, 1, 8)
        g.addWidget(self.btn_export, 1, 9)
        g.setColumnStretch(10, 1)
        root.addWidget(box)
        self.lbl_state = QtWidgets.QLabel(
            "Select traces on the map (left click), check the gray baseline window (before time zero) "
            "and the blue fit window, then click Fit.")
        self.lbl_state.setWordWrap(True)
        root.addWidget(self.lbl_state)

        split = QtWidgets.QSplitter(Qt.Vertical)
        self.glw = pg.GraphicsLayoutWidget()
        self.p_decay = self.glw.addPlot(row=0, col=0)
        self.p_res = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.p_res.setXLink(self.p_decay)
        for p in (self.p_decay, self.p_res):
            style_plot(p)
        self.p_decay.setLabel("left", "Counts − baseline")
        self.p_res.setLabel("left", "Weighted residual")
        self.p_res.setLabel("bottom", "Time (ns)")
        self.p_decay.addLegend(offset=(-10, 10))
        set_title(self.p_decay, "Baseline-subtracted decays and fits")
        self.p_res.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#6b7280", width=1)), ignoreBounds=True)
        self.region_base = pg.LinearRegionItem(brush=pg.mkBrush(156, 163, 175, 60), pen=pg.mkPen("#6b7280"))
        self.region_fit = pg.LinearRegionItem(brush=pg.mkBrush(147, 197, 253, 45), pen=pg.mkPen("#3b82f6"))
        for r in (self.region_base, self.region_fit):
            r.setZValue(-20)
            self.p_decay.addItem(r, ignoreBounds=True)
        self.res_scatter = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(30, 64, 175, 170))
        self.p_res.addItem(self.res_scatter)
        split.addWidget(self.glw)

        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self.btn_auto.clicked.connect(self.auto_windows)
        self.btn_fit.clicked.connect(self._on_fit)
        self.btn_export.clicked.connect(self._on_export)
        for s in (self.spin_b0, self.spin_b1, self.spin_f0, self.spin_f1):
            s.valueChanged.connect(self._on_spin_changed)
        self.region_base.sigRegionChangeFinished.connect(lambda: self._on_region(self.region_base, self.spin_b0, self.spin_b1))
        self.region_fit.sigRegionChangeFinished.connect(lambda: self._on_region(self.region_fit, self.spin_f0, self.spin_f1))
        self.chk_log.toggled.connect(lambda _v: self.refresh())
        self.chk_comp.toggled.connect(lambda _v: self.refresh())
        self.cmb_traces.currentIndexChanged.connect(lambda _i: self.refresh())
        self.table.itemSelectionChanged.connect(self.refresh)

    # ------------------------------------------------------------ data
    def _traces(self) -> List[Tuple[str, np.ndarray]]:
        """(label, raw counts over the full view) for the selected traces, or their sum."""
        traces = self.panel.selected_traces()
        if self.cmb_traces.currentData() == "sum" and len(traces) > 1:
            return [("sum of " + ", ".join(lab for lab, _ in traces), np.sum([y for _, y in traces], axis=0))]
        return traces

    def _key(self, traces) -> tuple:
        return (self.panel.data_version, tuple(lab for lab, _ in traces))

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def invalidate(self) -> None:
        self._results = []
        self._result_key = None
        if hasattr(self, "table"):
            self._fill_table()
            self.refresh()

    # --------------------------------------------------------- windows
    def _windows_valid(self) -> bool:
        p = self.panel
        if not p.has_data:
            return True
        t0, t1 = float(p.view_time[0]), float(p.view_time[-1])
        b0, b1, f0, f1 = (s.value() for s in (self.spin_b0, self.spin_b1, self.spin_f0, self.spin_f1))
        return b1 > b0 and f1 > f0 and t0 - 1e-9 <= b0 <= t1 and t0 <= f1 <= t1 + 1e-9

    def auto_windows(self) -> None:
        """Baseline: start of the range to ~1 ns before the peak; fit: just before the peak to the range end."""
        p = self.panel
        if not p.has_data or self._auto_busy:
            return
        traces = self._traces()
        t = p.view_time
        r0, r1 = p.range
        y = np.sum([tr for _, tr in traces], axis=0) if traces else p.view_counts.sum(axis=1)
        seg = np.asarray(y[r0:r1 + 1], dtype=np.float64)
        i_pk = r0 + int(np.argmax(seg)) if seg.size else r0
        bin_ns = float(np.median(np.diff(t))) if t.size > 1 else 0.01
        t_first, t_pk, t_last = float(t[r0]), float(t[i_pk]), float(t[r1])
        b1 = t_pk - max(1.0, 10 * bin_ns)
        if b1 <= t_first + bin_ns:
            b1 = t_first + 0.8 * (t_pk - t_first)
        b1 = max(b1, t_first + bin_ns)
        f0 = min(max(t_pk - max(0.5, 5 * bin_ns), b1), t_last - bin_ns)
        self._set_spins((t_first, b1, f0, t_last))
        self._auto_busy = True
        try:
            self.refresh()
        finally:
            self._auto_busy = False

    def _set_spins(self, values) -> None:
        for s, v in zip((self.spin_b0, self.spin_b1, self.spin_f0, self.spin_f1), values):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)
        self._sync_regions()

    def _sync_regions(self) -> None:
        for r, a, b in ((self.region_base, self.spin_b0, self.spin_b1), (self.region_fit, self.spin_f0, self.spin_f1)):
            r.blockSignals(True)
            r.setRegion((a.value(), b.value()))
            r.blockSignals(False)

    def _on_spin_changed(self, _v) -> None:
        self._sync_regions()
        self.refresh()

    def _on_region(self, region, s0, s1) -> None:
        lo, hi = region.getRegion()
        for s, v in ((s0, lo), (s1, hi)):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)
        self.refresh()

    # -------------------------------------------------------------- fit
    def _on_fit(self) -> None:
        p = self.panel
        if not an.HAS_SCIPY:
            QtWidgets.QMessageBox.warning(self, "Lifetime fit", "scipy is required for fitting.")
            return
        if not p.has_data or self._worker is not None:
            return
        traces = self._traces()
        if not traces:
            QtWidgets.QMessageBox.information(self, "Lifetime fit", "Select at least one trace on the map (left click).")
            return
        if not self._windows_valid():
            self.auto_windows()
        t = p.view_time
        b0, b1 = self.spin_b0.value(), self.spin_b1.value()
        jobs = []
        for label, raw in traces:
            corr, base = an.subtract_baseline(raw, t, b0, b1)
            jobs.append((label, corr[:, 0], np.asarray(raw, dtype=np.float64), float(base[0])))
        kwargs = dict(t=np.array(t, copy=True), traces=jobs, n_exp=self.spin_nexp.value(),
                      fit_window=(self.spin_f0.value(), self.spin_f1.value()), fit_offset=self.chk_offset.isChecked())
        self._job_id += 1
        key = self._key(traces)
        w = FitAllWorker(self._job_id, an.fit_decays, kwargs, self)
        w.done.connect(lambda job, res, err: self._on_done(job, res, err, key))
        w.finished.connect(self._on_worker_finished)
        self._worker = w
        self.btn_fit.setEnabled(False)
        self.lbl_state.setText(f"Fitting {len(jobs)} trace(s) with {self.spin_nexp.value()} exponential(s) …")
        w.start()

    def _on_done(self, job_id: int, res, err: str, key) -> None:
        if job_id != self._job_id:
            return
        if err:
            self.lbl_state.setText(f"Fit failed: {err}")
            return
        if key[0] != self.panel.data_version:
            self.lbl_state.setText("The data changed during the fit; click Fit again.")
            return
        self._results = list(res)
        self._result_key = key
        n_ok = sum(r.ok for r in self._results)
        bad = [f"{r.label}: {r.message}" for r in self._results if r.message]
        self.lbl_state.setText(
            f"Fitted {n_ok}/{len(self._results)} trace(s), {self.spin_nexp.value()} exponential(s), "
            f"window {self.spin_f0.value():.3f}–{self.spin_f1.value():.3f} ns, baseline "
            f"{self.spin_b0.value():.3f}–{self.spin_b1.value():.3f} ns."
            + (" Notes: " + "; ".join(bad) if bad else "")
        )
        self._fill_table()
        if self._results:
            self.table.selectRow(0)
        self.refresh()

    def _on_worker_finished(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.deleteLater()
        self.btn_fit.setEnabled(True)

    def _valid_results(self, traces) -> List[an.DecayFitResult]:
        return self._results if self._result_key == self._key(traces) else []

    # ---------------------------------------------------------- display
    def _fill_table(self) -> None:
        res = self._results
        n = max((r.n_exp for r in res), default=0)
        headers = (["Trace"] + [f"τ{k + 1} (ns)" for k in range(n)] + [f"f{k + 1} (%)" for k in range(n)]
                   + ["⟨τ⟩ int (ns)", "⟨τ⟩ amp (ns)", "t0 (ns)", "σ IRF (ps)", "χ²ᵣ", "Baseline", "Note"])
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(res))
        for row, r in enumerate(res):
            cells = [r.label]
            if r.ok:
                cells += [f"{tau:.4g} ± {err:.2g}" for tau, err in zip(r.taus, r.tau_err)]
                cells += [""] * (n - r.n_exp)
                cells += [f"{100 * f:.1f}" for f in r.fractions] + [""] * (n - r.n_exp)
                cells += [f"{r.tau_avg_intensity:.4g}", f"{r.tau_avg_amplitude:.4g}", f"{r.t0:.4f}",
                          f"{1000 * r.sigma_irf:.1f}", f"{r.chi2_red:.2f}", f"{r.baseline:.3g}", r.message]
            else:
                cells += [""] * (2 * n + 5) + [f"{r.baseline:.3g}", r.message or "fit failed"]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(pg.mkColor(trace_color(row)))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.blockSignals(False)

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else 0

    def refresh(self) -> None:
        if not self.isVisible():
            return
        try:
            self._draw()
        except Exception as exc:
            self.lbl_state.setText(f"Display error: {exc}")

    def _draw(self) -> None:
        p = self.panel
        for c in self._curves:
            self.p_decay.removeItem(c)
        self._curves = []
        self.res_scatter.setData([], [])
        log = self.chk_log.isChecked()
        self.p_decay.setLogMode(x=False, y=log)
        if not p.has_data:
            return
        if not self._windows_valid() and not self._auto_busy:
            self.auto_windows()
            return
        traces = self._traces()
        results = self._valid_results(traces)
        n = p.view_time.size
        r1 = min(p.range[1], n - 1)
        r0 = min(p.range[0], r1)
        t = p.view_time[r0:r1 + 1]
        hi_row = self._selected_row()
        ymax = 0.0
        for _, raw in traces:
            seg = np.asarray(raw, dtype=np.float64)[r0:r1 + 1]
            if seg.size:
                ymax = max(ymax, float(np.nanmax(seg)))
        floor = max(0.3, 1e-5 * ymax)  # log mode: hide curve tails far below the data

        def clip(v):
            return np.where(v > floor, v, np.nan) if log else v

        for k, (label, raw) in enumerate(traces):
            corr, _ = an.subtract_baseline(raw, p.view_time, self.spin_b0.value(), self.spin_b1.value())
            y = corr[r0:r1 + 1, 0]
            if log:
                y = np.where(y > 0, y, np.nan)
            color = trace_color(k)
            c = pg.PlotDataItem(t, y, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                symbolBrush=pg.mkBrush(color), name=label)
            self.p_decay.addItem(c)
            self._curves.append(c)
            fit = results[k] if k < len(results) else None
            if fit is None or not fit.ok:
                continue
            lo, hi = fit.fit_window
            tf = np.linspace(lo, hi, 800)
            yf = fit.evaluate(tf)
            c = self.p_decay.plot(tf, clip(yf), pen=pg.mkPen(color, width=2.2))
            self._curves.append(c)
            if k == hi_row:
                if self.chk_comp.isChecked() and fit.n_exp > 1:
                    for i in range(fit.n_exp):
                        yc = fit.component(tf, i) + fit.offset
                        c = self.p_decay.plot(tf, clip(yc), pen=pg.mkPen(color, width=1.2, style=Qt.DashLine))
                        self._curves.append(c)
                sel = (p.view_time >= lo) & (p.view_time <= hi)
                tr = p.view_time[sel]
                w = np.sqrt(np.clip(np.asarray(raw, dtype=np.float64)[sel], 1.0, None))
                self.res_scatter.setData(tr, (corr[sel, 0] - fit.evaluate(tr)) / w)
        if results and 0 <= hi_row < len(results) and results[hi_row].ok:
            r = results[hi_row]
            taus = ", ".join(f"τ{i + 1} = {tau:.3g} ns ({100 * f:.0f}%)" for i, (tau, f) in enumerate(zip(r.taus, r.fractions)))
            set_title(self.p_decay, f"{r.label}: {taus}",
                      f"⟨τ⟩int = {r.tau_avg_intensity:.3g} ns, σ_IRF = {1000 * r.sigma_irf:.0f} ps, χ²ᵣ = {r.chi2_red:.2f}")
        else:
            set_title(self.p_decay, "Baseline-subtracted decays",
                      "not fitted yet" if traces else "left-click the map to select traces")
        self.p_decay.setXRange(float(t[0]), float(t[-1]), padding=0.01)
        if log and ymax > floor:
            self.p_decay.setYRange(np.log10(floor), np.log10(2.0 * ymax), padding=0)
        else:
            self.p_decay.enableAutoRange(axis="y", enable=True)

    def showEvent(self, ev):
        super().showEvent(ev)
        if self.panel.has_data and not self._windows_valid():
            self.auto_windows()
        self.refresh()

    # ----------------------------------------------------------- export
    def _on_export(self) -> None:
        traces = self._traces()
        results = self._valid_results(traces)
        if not results:
            QtWidgets.QMessageBox.information(self, "Export lifetimes", "Fit first.")
            return
        p = self.panel
        base = (os.path.splitext(os.path.basename(p.raw.source))[0] if (p.raw and p.raw.source)
                else f"spad93_trpl_{timestamp()}")
        default = os.path.join(save_dir("trpl"), f"{base}_lifetimes.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export lifetimes", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        n = max(r.n_exp for r in results)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write(f"# source: {p.raw.source if p.raw else ''}\n")
                for key, value in p.processing_metadata().items():
                    fh.write(f"# {key}: {json.dumps(value)}\n")
                fh.write(f"# baseline_window_ns: {self.spin_b0.value()},{self.spin_b1.value()}\n")
                fh.write(f"# fit_window_ns: {self.spin_f0.value()},{self.spin_f1.value()}\n")
                fh.write(f"# model: sum of {n} exponentials convolved with a Gaussian IRF; "
                         "fractions are intensity (a_i*tau_i) fractions\n")
                wr = csv.writer(fh)
                wr.writerow(["trace", "ok"] + [f"tau{k + 1}_ns" for k in range(n)] + [f"tau{k + 1}_err_ns" for k in range(n)]
                            + [f"amp{k + 1}" for k in range(n)] + [f"frac{k + 1}" for k in range(n)]
                            + ["tau_avg_int_ns", "tau_avg_amp_ns", "t0_ns", "sigma_irf_ns", "offset", "chi2_red",
                               "baseline", "note"])
                for r in results:
                    pad = [""] * (n - r.n_exp)
                    wr.writerow([r.label, int(r.ok)] + list(r.taus) + pad + list(r.tau_err) + pad + list(r.amps) + pad
                                + list(r.fractions if r.ok else []) + pad
                                + [r.tau_avg_intensity, r.tau_avg_amplitude, r.t0, r.sigma_irf, r.offset, r.chi2_red,
                                   r.baseline, r.message])
            curves = os.path.splitext(path)[0] + "_curves.csv"
            t = p.view_time
            cols = [t]
            names = ["time_ns"]
            for (label, raw), r in zip(traces, results):
                corr, _ = an.subtract_baseline(raw, t, self.spin_b0.value(), self.spin_b1.value())
                cols += [corr[:, 0], r.evaluate(t) if r.ok else np.full(t.size, np.nan)]
                names += [f"{label} data", f"{label} fit"]
            with open(curves, "w", newline="", encoding="utf-8") as fh:
                fh.write(",".join(names) + "\n")
                np.savetxt(fh, np.column_stack(cols), fmt="%.6g", delimiter=",")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export lifetimes", f"Could not save:\n{exc}")
            return
        p.status.emit(f"Lifetimes exported: {path} (+ {os.path.basename(curves)})")
