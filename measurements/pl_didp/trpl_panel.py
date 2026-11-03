"""TRPL tab of the dI/dP viewer: pump 2 on − off, pixel by pixel (SPAD23).

Left: the 23-pixel map of the chosen quantity summed over the time window at
the current point (click a pixel to select it, right click to add / remove).
Middle: off / on traces and the quantity for the selected pixels (summed).
Right: the quantity versus time and the swept axis for the selected pixels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from spad23.gui_v2.map_plots import PixelMapPlot
from spad23.gui_v2.plots import HistogramPlot
from spad23.spad23_geometry import NUM_PIXELS
from spad23.spad23_io import save_columns_csv

from . import trpl as tr
from .config import DEFAULT_REBIN, DEFAULT_TRPL_WIN_NS, DIFF_COLOR, DIV_CMAP, SEQ_CMAP, SET1_COLOR, SET2_COLOR
from .data import pair_stem, trpl_of
from .plots import DiffMapWidget, pg_cmap
from .ui import MIN_PT, px

# Importing spad23.gui_v2 switches pyqtgraph to row-major images; the image plots of
# this viewer (SingleFrameView, DiffMapWidget) are written for the column-major default.
# The SPAD widgets used here draw no ImageItem, so they are unaffected.
pg.setConfigOptions(imageAxisOrder="col-major")


def _f(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.4g}" if np.isfinite(v) else "--"


def _set_title(target, title: str, subtitle: str = "") -> None:
    """Plot title at readable sizes (spad23's set_title uses an 8 pt subtitle)."""
    html = f"<span style='color:#111827;font-size:{MIN_PT + 1}pt'>{title}</span>"
    if subtitle:
        html += f"<br><span style='color:#374151;font-size:{MIN_PT}pt'>{subtitle}</span>"
    target.setTitle(html)


@dataclass
class TrplColumn:
    """One sweep column: its x value and the pump-2 off / on TRPL stub frames."""
    x: float
    off: object = None
    on: object = None
    label: str = ""


class TrplDiffPanel(QtWidgets.QWidget):
    column_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cols: List[TrplColumn] = []
        self._cur = 0
        self._x_label = ""
        self._log_x = False
        self._p2 = 0.0
        self._floor = 0.02
        self._cond = ""
        self._cache: Dict[tuple, object] = {}   # (off path, on path, sweep t0) → TrplPair | error text
        self._align: Dict[Tuple[str, str], object] = {}   # (off path, on path) → TimeZeroAlignment | None
        self._sel: List[int] = []
        self._vals = np.full(NUM_PIXELS, np.nan)
        self._unit = ""
        self._map_cmap = ""
        self._pre_auto = True
        self._dirty = False
        self._folder: Optional[Path] = None
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Show:"))
        self.quantityCombo = QtWidgets.QComboBox()
        self.quantityCombo.addItems(tr.QUANTITIES)
        row1.addWidget(self.quantityCombo)
        row1.addSpacing(12)
        row1.addWidget(QtWidgets.QLabel("Map window (ns):"))
        self.winLoSpin, self.winHiSpin = self._ns_spin(DEFAULT_TRPL_WIN_NS[0]), self._ns_spin(DEFAULT_TRPL_WIN_NS[1])
        row1.addWidget(self.winLoSpin)
        row1.addWidget(QtWidgets.QLabel("to"))
        row1.addWidget(self.winHiSpin)
        row1.addSpacing(12)
        row1.addWidget(QtWidgets.QLabel("Rebin:"))
        self.rebinSpin = QtWidgets.QSpinBox()
        self.rebinSpin.setRange(1, 256)
        self.rebinSpin.setValue(DEFAULT_REBIN)
        self.rebinSpin.setToolTip("Sum this many time bins (applied to both traces)")
        row1.addWidget(self.rebinSpin)
        self.logCheck = QtWidgets.QCheckBox("Log counts")
        self.logCheck.setToolTip("Log y for the off / on traces")
        row1.addWidget(self.logCheck)
        row1.addStretch()
        self.exportPointBtn = QtWidgets.QPushButton("Export point CSV…")
        self.exportSweepBtn = QtWidgets.QPushButton("Export sweep…")
        self.exportPointBtn.setToolTip("Off, on and on − off traces of all 23 pixels at the current point")
        self.exportSweepBtn.setToolTip("One CSV per sweep point, plus the window values of every pixel vs sweep")
        row1.addWidget(self.exportPointBtn)
        row1.addWidget(self.exportSweepBtn)
        root.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.alignCheck = QtWidgets.QCheckBox("Align time zero on each pixel's first peak")
        self.alignCheck.setToolTip(
            "Shifts are found on the pump-2-off trace and applied to both traces,\n"
            "so the subtraction stays registered. Every point is then moved as a\n"
            "whole to one time zero shared by the sweep (median of the points').")
        row2.addWidget(self.alignCheck)
        row2.addSpacing(12)
        self.preCheck = QtWidgets.QCheckBox("Subtract pre-t0 offset:")
        self.preCheck.setToolTip(
            "Subtract each trace's own per-pixel mean over this window.\n"
            "Dark counts already cancel in on − off; this also removes a pump-2\n"
            "background that is present before time zero.")
        row2.addWidget(self.preCheck)
        self.preLoSpin, self.preHiSpin = self._ns_spin(0.0), self._ns_spin(0.0)
        row2.addWidget(self.preLoSpin)
        row2.addWidget(QtWidgets.QLabel("to"))
        row2.addWidget(self.preHiSpin)
        self.preAutoBtn = QtWidgets.QPushButton("Auto")
        self.preAutoBtn.setFixedWidth(px(50))
        self.preAutoBtn.setToolTip("Window from the first bin to just before the rise (per point)")
        row2.addWidget(self.preAutoBtn)
        row2.addSpacing(12)
        self.infoLbl = QtWidgets.QLabel("")
        self.infoLbl.setStyleSheet("color: #555;")
        row2.addWidget(self.infoLbl, 1)
        root.addLayout(row2)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.pixMap = PixelMapPlot("Window value")
        self.pixMap.hover_text_fn = lambda p: f"pixel {p}: {_f(self._vals[p])} {self._unit}".strip()
        split.addWidget(self.pixMap)

        traces = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.rawPlot = HistogramPlot()
        self.diffPlot = HistogramPlot()
        self.diffPlot.plot.setXLink(self.rawPlot.plot)
        self.diffZero = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#999", width=0.8))
        self.diffPlot.plot.addItem(self.diffZero, ignoreBounds=True)
        traces.addWidget(self.rawPlot)
        traces.addWidget(self.diffPlot)
        split.addWidget(traces)

        self.mapView = DiffMapWidget("Δ vs time and sweep (selected pixels)")
        split.addWidget(self.mapView)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 4)
        root.addWidget(split, 1)

        self.quantityCombo.currentTextChanged.connect(lambda _: self._redraw())
        for s in (self.winLoSpin, self.winHiSpin):
            s.editingFinished.connect(self._redraw)
        self.logCheck.toggled.connect(lambda _: self._redraw())
        self.rebinSpin.valueChanged.connect(lambda _: self._on_processing())
        self.alignCheck.toggled.connect(lambda _: self._on_processing())
        self.preCheck.toggled.connect(lambda _: self._on_processing())
        for s in (self.preLoSpin, self.preHiSpin):
            s.editingFinished.connect(self._on_pre_window_edited)
        self.preAutoBtn.clicked.connect(self._on_pre_auto)
        self.pixMap.pixel_clicked.connect(self._on_pixel_clicked)
        self.mapView.cursor_changed.connect(self._on_map_cursor)
        self.exportPointBtn.clicked.connect(self._export_point)
        self.exportSweepBtn.clicked.connect(self._export_sweep)

    @staticmethod
    def _ns_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(2)
        s.setRange(-1e6, 1e6)
        s.setValue(value)
        s.setSuffix(" ns")
        s.setFixedWidth(px(95))
        s.setKeyboardTracking(False)
        return s

    # ══════════════════════════════════════════════════════════════════
    # Public API (called by the main widget)
    # ══════════════════════════════════════════════════════════════════

    def clear_data(self, folder: Optional[Path] = None) -> None:
        """New data loaded: drop the processed pairs and the pixel selection."""
        self._cache.clear()
        self._align.clear()
        self._sel = []
        self._folder = folder

    def set_window(self, t0: float, t1: float) -> None:
        self.winLoSpin.setValue(float(t0))
        self.winHiSpin.setValue(float(t1))

    def refresh(self, cols: List[TrplColumn], cur: int, *, x_label: str, log_x: bool,
                p2_uw: float, rel_floor: float, cond: str = "") -> None:
        self._cols = list(cols)
        self._cur = max(0, min(int(cur), len(self._cols) - 1)) if self._cols else 0
        self._x_label, self._log_x = x_label, bool(log_x)
        self._p2, self._floor, self._cond = float(p2_uw), float(rel_floor), cond
        if self.isVisible():
            self._redraw()
        else:
            self._dirty = True

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if self._dirty:
            self._redraw()

    # ══════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════

    def _on_processing(self) -> None:
        self._cache.clear()
        self._redraw()

    def _on_pre_window_edited(self) -> None:
        if self.preLoSpin.value() < self.preHiSpin.value():
            self._pre_auto = False
            if self.preCheck.isChecked():
                self._on_processing()

    def _on_pre_auto(self) -> None:
        self._pre_auto = True
        if self.preCheck.isChecked():
            self._on_processing()

    def _pre_window(self) -> Optional[Tuple[float, float]]:
        if self._pre_auto:
            return None
        return float(self.preLoSpin.value()), float(self.preHiSpin.value())

    def _alignment(self, col: TrplColumn):
        """First-peak alignment of a complete column (cached), or None if it fails."""
        key = (str(col.off.path), str(col.on.path))
        if key not in self._align:
            try:
                self._align[key] = tr.time_zero(trpl_of(col.off), trpl_of(col.on))
            except Exception:
                self._align[key] = None
        return self._align[key]

    def _sweep_t0(self) -> float:
        """Time zero shared by all points of the sweep (NaN if no point has one)."""
        t0s, period = [], None
        for col in self._cols:
            if col.off is None or col.on is None:
                continue
            al = self._alignment(col)
            if al is None:
                continue
            t0s.append(tr.time_zero_ns(al, trpl_of(col.off).time_ns))
            if period is None:
                period = al.counts.shape[0] * al.bin_ns
        return tr.common_time_zero(t0s, period) if period else float("nan")

    def _pair(self, col: Optional[TrplColumn]):
        """Processed pair of a column, or an error text, or None (no pair)."""
        if col is None or col.off is None or col.on is None:
            return None
        align = self.alignCheck.isChecked()
        t0_ref = self._sweep_t0() if align else None
        key = (str(col.off.path), str(col.on.path), None if t0_ref is None else round(t0_ref, 6))
        hit = self._cache.get(key)
        if hit is None:
            try:
                hit = tr.prepare_pair(trpl_of(col.off), trpl_of(col.on), align=align,
                                      rebin=int(self.rebinSpin.value()), pre_t0=self.preCheck.isChecked(),
                                      pre_window=self._pre_window(),
                                      alignment=self._alignment(col) if align else None, t0_ref_ns=t0_ref)
            except Exception as exc:
                hit = f"{col.off.path.name}: {exc}"
            self._cache[key] = hit
        return hit

    # ══════════════════════════════════════════════════════════════════
    # Drawing
    # ══════════════════════════════════════════════════════════════════

    def _redraw(self) -> None:
        self._dirty = False
        q = self.quantityCombo.currentText()
        if not self._cols:
            self._show_empty("No TRPL files (TRPL_*_P2off.csv / _P2on.csv) loaded")
            return
        col = self._cols[self._cur]
        pair = self._pair(col)
        if not isinstance(pair, tr.TrplPair):
            if isinstance(pair, str):
                why = pair
            elif col.off is None and col.on is None:
                why = "no TRPL files at this point"
            else:
                why = f"missing the pump 2 {'on' if col.on is None else 'off'} file at this point"
            self._show_empty(f"{why}   ·   {self._cond}", keep_map=True)
        else:
            self._draw_point(q, col, pair)
        self._draw_map(q)

    def _show_empty(self, msg: str, keep_map: bool = False) -> None:
        self._vals = np.full(NUM_PIXELS, np.nan)
        self.pixMap.set_values(self._vals)
        _set_title(self.pixMap.plot, "Window value", msg)
        self.rawPlot.set_traces([])
        self.diffPlot.set_traces([])
        self.rawPlot.show_message(msg)
        self.diffPlot.show_message("")
        self.infoLbl.setText("")
        if not keep_map:
            self.mapView.clear(msg)

    def _window(self) -> Tuple[float, float]:
        return float(self.winLoSpin.value()), float(self.winHiSpin.value())

    def _set_map_cmap(self, name: str) -> None:
        if name != self._map_cmap:
            self._map_cmap = name
            cm = pg_cmap(name)
            self.pixMap.cmap = cm
            self.pixMap.cbar.setColorMap(cm)

    def _draw_point(self, q: str, col: TrplColumn, pair: tr.TrplPair) -> None:
        t0, t1 = self._window()
        signed = q in tr.SIGNED
        vals = tr.window_values(q, pair, t0, t1, p2_uw=self._p2, rel_floor=self._floor)
        self._vals = vals
        self._unit = tr.UNITS_OF[q]
        if not self._sel:
            # Start on the brightest pixel (whole pump-2-off trace, so a bad window cannot mislead it)
            off_sum = pair.off.sum(axis=0)
            self._sel = [int(np.nanargmax(off_sum))] if np.isfinite(off_sum).any() else [NUM_PIXELS // 2]
        finite = vals[np.isfinite(vals)]
        levels = None
        if signed:
            self._set_map_cmap(DIV_CMAP)
            half = float(np.max(np.abs(finite))) if finite.size else 1.0
            levels = (-half, half) if half > 0 else (-1.0, 1.0)
        else:
            self._set_map_cmap(SEQ_CMAP)
        self.pixMap.set_values(vals, levels=levels, unit=self._unit)
        self.pixMap.set_selected(self._sel)
        _set_title(self.pixMap.plot, f"{q.split('  ')[0]}  ·  Σ {_f(t0)}–{_f(t1)} ns", self._cond)

        sel = self._sel
        off_s = pair.off[:, sel].sum(axis=1, keepdims=True)
        on_s = pair.on[:, sel].sum(axis=1, keepdims=True)
        t = pair.t_ns
        self.rawPlot.set_log(self.logCheck.isChecked())
        self.rawPlot.set_traces([("pump 2 off", t, off_s[:, 0], SET1_COLOR),
                                 ("pump 2 on", t, on_s[:, 0], SET2_COLOR)])
        pix_txt = ", ".join(str(p) for p in sel)
        _set_title(self.rawPlot.plot, f"Pixel{'s' if len(sel) > 1 else ''} {pix_txt}", col.label)
        self.rawPlot.plot.setLabel("left", "Counts per bin")
        y = tr.quantity(q, off_s, on_s, p2_uw=self._p2, rel_floor=self._floor)[:, 0]
        self.diffPlot.set_traces([(q.split("  ")[0], t, y, DIFF_COLOR)])
        self.diffPlot.plot.setLabel("left", f"{q.split('  ')[0]}" + (f" ({self._unit})" if self._unit else ""))
        _set_title(self.diffPlot.plot, q.split("  ")[0], f"pixel{'s' if len(sel) > 1 else ''} {pix_txt}")
        self.diffZero.setVisible(signed)
        if not np.isfinite(y).any():
            self.diffPlot.show_message("Set P₂ > 0" if q == tr.Q_DIDP and self._p2 <= 0 else "No finite values")
        for plot in (self.rawPlot, self.diffPlot):
            plot.region.setRegion((t0, t1))

        info = []
        if pair.align_msg:
            info.append(pair.align_msg)
        if pair.pre_window is not None:
            info.append(f"pre-t0 {_f(pair.pre_window[0])}–{_f(pair.pre_window[1])} ns subtracted")
            if self._pre_auto:
                for s, v in ((self.preLoSpin, pair.pre_window[0]), (self.preHiSpin, pair.pre_window[1])):
                    s.blockSignals(True)
                    s.setValue(v)
                    s.blockSignals(False)
        info.extend(pair.notes)
        self.infoLbl.setText("   ·   ".join(info))
        self.infoLbl.setToolTip("\n".join(info))

    def _draw_map(self, q: str) -> None:
        pairs = [self._pair(c) for c in self._cols]
        ref = next((p for p in pairs if isinstance(p, tr.TrplPair)), None)
        if ref is None or not self._sel:
            self.mapView.clear("No complete off / on TRPL pairs")
            return
        t_ref = ref.t_ns
        z = np.full((t_ref.size, len(self._cols)), np.nan)
        sel = self._sel
        for j, p in enumerate(pairs):
            if not isinstance(p, tr.TrplPair):
                continue
            off_s = p.off[:, sel].sum(axis=1, keepdims=True)
            on_s = p.on[:, sel].sum(axis=1, keepdims=True)
            y = tr.quantity(q, off_s, on_s, p2_uw=self._p2, rel_floor=self._floor)[:, 0]
            z[:, j] = tr.on_time_axis(p.t_ns, y, t_ref)
        x = np.array([c.x for c in self._cols], dtype=float)
        signed = q in tr.SIGNED
        units = tr.UNITS_OF[q]
        z_label = q.split("≈")[0].split("  ")[0].strip() + (f" ({units})" if units else "")
        pix_txt = ", ".join(str(s) for s in sel)
        self.mapView.setTitle(f"{q.split('  ')[0]}   ·   pixel{'s' if len(sel) > 1 else ''} {pix_txt}")
        self.mapView.set_data(x, t_ref, z, x_label=self._x_label, y_label="Time (ns)", z_label=z_label,
                              log_x=self._log_x, symmetric=signed, center=0.0,
                              cmap=DIV_CMAP if signed else SEQ_CMAP)
        self.mapView.set_cursor(ix=self._cur)

    # ══════════════════════════════════════════════════════════════════
    # Interaction
    # ══════════════════════════════════════════════════════════════════

    def _on_pixel_clicked(self, pix: int, button: int) -> None:
        if not 0 <= pix < NUM_PIXELS:
            return
        if button == 2:
            if pix in self._sel:
                if len(self._sel) > 1:
                    self._sel.remove(pix)
            else:
                self._sel.append(pix)
        else:
            self._sel = [pix]
        self._sel.sort()
        self._redraw()

    def _on_map_cursor(self, ix: int, _iy: int) -> None:
        if ix != self._cur:
            self.column_clicked.emit(int(ix))

    # ══════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════

    def _proc_meta(self) -> dict:
        pre = "off"
        if self.preCheck.isChecked():
            pre = "auto (per point)" if self._pre_auto else f"{self.preLoSpin.value():g} to {self.preHiSpin.value():g} ns"
        return {
            "source": "pl_didp viewer: pump 2 on − off per SPAD23 pixel",
            "time_zero_alignment": ("first peak, shifts from pump 2 off applied to both; every point moved "
                                    "to the sweep's common time zero") if self.alignCheck.isChecked() else "off",
            "rebin": int(self.rebinSpin.value()),
            "pre_t0_offset": pre,
        }

    def _point_columns(self, pair: tr.TrplPair) -> Dict[str, np.ndarray]:
        cols: Dict[str, np.ndarray] = {"time_ns": pair.t_ns}
        for p in range(NUM_PIXELS):
            cols[f"off_px{p}"] = pair.off[:, p]
            cols[f"on_px{p}"] = pair.on[:, p]
            cols[f"diff_px{p}"] = pair.on[:, p] - pair.off[:, p]
        return cols

    def _point_meta(self, col: TrplColumn, pair: tr.TrplPair) -> dict:
        meta = {"off_file": col.off.path.name, "on_file": col.on.path.name, "point": col.label,
                self._x_label or "x": col.x}
        meta.update(self._proc_meta())
        if np.isfinite(pair.t0_ns):
            meta["time_zero_ns"] = round(pair.t0_ns, 4)
        if pair.shift_bins is not None:
            meta["time_zero_shift_ns"] = {str(p): round(float(s) * float(np.median(np.diff(
                trpl_of(col.off).time_ns))), 4) for p, s in enumerate(pair.shift_bins) if s}
        if pair.pre_window is not None:
            meta["pre_t0_window_ns"] = [round(pair.pre_window[0], 4), round(pair.pre_window[1], 4)]
        return meta

    def _start_dir(self) -> str:
        return str(self._folder) if self._folder is not None else ""

    def _export_point(self) -> None:
        col = self._cols[self._cur] if self._cols else None
        pair = self._pair(col)
        if not isinstance(pair, tr.TrplPair):
            QtWidgets.QMessageBox.warning(self, "Export point", "No complete off / on TRPL pair at this point.")
            return
        name = f"didp_trpl_{pair_stem(col.off.path.name)}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save TRPL on − off (all pixels)", str(Path(self._start_dir()) / name), "CSV (*.csv)")
        if not path:
            return
        try:
            save_columns_csv(path, self._point_columns(pair), self._point_meta(col, pair))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export point", f"{path}\n\n{exc}")
            return
        self.infoLbl.setText(f"Saved {Path(path).name}")

    def export_sweep_to(self, folder: Path) -> Tuple[int, Path]:
        """Write one CSV per sweep point and the window summary; returns (n points, summary path)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        q = self.quantityCombo.currentText()
        t0, t1 = self._window()
        xs, rows, files = [], [], []
        for col in self._cols:
            pair = self._pair(col)
            if not isinstance(pair, tr.TrplPair):
                continue
            name = f"didp_trpl_{pair_stem(col.off.path.name)}.csv"
            save_columns_csv(str(folder / name), self._point_columns(pair), self._point_meta(col, pair))
            xs.append(col.x)
            rows.append(tr.window_values(q, pair, t0, t1, p2_uw=self._p2, rel_floor=self._floor))
            files.append(name)
        summary = folder / "didp_trpl_window_summary.csv"
        if rows:
            vals = np.vstack(rows)
            cols: Dict[str, np.ndarray] = {self._x_label or "x": np.asarray(xs, dtype=float)}
            for p in range(NUM_PIXELS):
                cols[f"px{p}"] = vals[:, p]
            meta = {"quantity": q, "units": tr.UNITS_OF[q], "window_ns": [t0, t1], "p2_uw": self._p2,
                    "point_files": files}
            meta.update(self._proc_meta())
            save_columns_csv(str(summary), cols, meta)
        return len(rows), summary

    def _export_sweep(self) -> None:
        if not self._cols:
            QtWidgets.QMessageBox.warning(self, "Export sweep", "No TRPL data loaded.")
            return
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the TRPL on − off CSVs", self._start_dir())
        if not chosen:
            return
        try:
            n, summary = self.export_sweep_to(Path(chosen))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export sweep", str(exc))
            return
        self.infoLbl.setText(f"Saved {n} point CSV(s) and {summary.name} to {chosen}")
