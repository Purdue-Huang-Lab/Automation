"""TRPL tab: acquire, view and save SPAD93 time-resolved data (2D space / spectral).

The same widget is the offline viewer (``device=None``): the acquisition
controls are hidden and data is loaded from saved CSV files.
"""

import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from spad93.spad93_analysis import DarkReference, TimeZeroAlignment, align_first_peaks, locate_peak, rebin_time
from spad93.spad93_geometry import CENTER_PIXEL, COLUMN_X_UM, PIXEL_COL, center_row_matrix, column_of
from spad93.spad93_io import (
    TrplData,
    calibration_from_metadata,
    load_calibration_json,
    load_dark_csv,
    load_trpl_csv,
    columns_to_text,
    save_dark_csv,
    save_origin_matrix_csv,
    save_spectral_matrix_csv,
    save_trpl_csv,
    timestamp,
    trim_trailing_empty,
    unique_path,
)
from spad93.spad93_tcspc_wrapper import TrplSnapshot
from spad93.spad93_wrapper import PIXEL_LIMIT_MCPS
from spad93.gui_v2.config import (
    ALIGN_MS,
    BTN_W,
    SLIDER_DEBOUNCE_MS,
    TRPL_DEFAULT_ACQ_S,
    TRPL_DEFAULT_BIN_PS,
    TRPL_DARK_ACQ_S,
    TRPL_SAVE_WORKER_ROWS,
    save_dir,
)
from spad93.gui_v2.device import BUSY, IDLE, LIVE, DeviceController
from spad93.gui_v2.map_plots import PixelMapPlot, TimeSpectrumImage
from spad93.gui_v2.plots import HistogramPlot, SpectrumPlot, fwhm_text, trace_color
from spad93.gui_v2.state import MODE_2D, MODE_SPECTRAL, SharedState
from spad93.gui_v2.trpl_fitting import TrplFittingTab
from spad93.gui_v2.trpl_lifetime import TrplLifetimeTab
from spad93.gui_v2.workers import CallWorker

GEOMETRY_NOTE = (
    "SPAD93 sensor drawing rotated 90 deg CCW: x = 19.92*col um, y = 23*(site + 0.5*(col odd)) um, "
    "pixels 0-2 in column 0"
)


def _grid(box: QtWidgets.QGroupBox) -> QtWidgets.QGridLayout:
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setContentsMargins(8, 8, 8, 8)
    return g


class TrplPanel(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)

    def __init__(self, state: SharedState, device: Optional[DeviceController] = None,
                 calib_loader: Optional[Callable[[str], bool]] = None, parent=None):
        super().__init__(parent)
        self.state = state
        self.device = device
        self.viewer = device is None
        self._calib_loader = calib_loader
        self._raw: Optional[TrplData] = None
        self.leak: Optional[TimeZeroAlignment] = None  # time-zero alignment applied to the view (None = off)
        self.view_time = np.zeros(0)
        self.view_counts = np.zeros((0, 93), dtype=np.int64)
        self._spec: Optional[np.ndarray] = None
        self._rebin = 1
        self.range = (0, 0)
        self._req_range: Optional[Tuple[float, float]] = None  # applied (start, end) ns; None = all data
        self._range_edit_pending = False  # start/end typed but not applied: keep them in the boxes
        self.current = 0
        self.ref_time: Optional[float] = None
        self._sel_pixels = {CENTER_PIXEL}
        self._sel_cols = {int(PIXEL_COL[CENTER_PIXEL])}
        self.active_col = int(PIXEL_COL[CENTER_PIXEL])
        self.data_version = 0
        self._unsaved = False
        self._saved_path = ""
        self._acquiring = False
        self._new_acquisition = False
        self._pending_snapshot: Optional[TrplSnapshot] = None
        self._progress_pending = False
        self._save_worker: Optional[CallWorker] = None
        self.dark_ref: Optional[DarkReference] = None  # dark reference (beam blocked), used by the Fitting tab
        self._dark_pending = False  # the running acquisition is a dark reference

        self._build_ui()
        self._slider_timer = QtCore.QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(SLIDER_DEBOUNCE_MS)
        self._slider_timer.timeout.connect(self.fitting.refresh_current)

        self.state.mode_changed.connect(self._on_mode_changed)
        self.state.excluded_changed.connect(self._on_excluded_changed)
        self.state.calibration_changed.connect(self._on_calibration_changed)
        if device is not None:
            device.trpl_progress.connect(self._on_progress)
            device.trpl_done.connect(self._on_done)
            device.command_done.connect(self._on_command_done)
        self._on_excluded_changed()
        self._on_mode_changed(self.state.mode)
        self._sync_data_controls()

    # =============================================================== UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        acq = QtWidgets.QGroupBox("Acquisition" if not self.viewer else "Data")
        g = _grid(acq)
        self.spin_acq = QtWidgets.QDoubleSpinBox()
        self.spin_acq.setRange(0.1, 36000.0)
        self.spin_acq.setDecimals(2)
        self.spin_acq.setValue(TRPL_DEFAULT_ACQ_S)
        self.spin_bin = QtWidgets.QSpinBox()
        self.spin_bin.setRange(1, 100000)
        self.spin_bin.setValue(TRPL_DEFAULT_BIN_PS)
        self.btn_tdc = QtWidgets.QPushButton("Calibrate + Align")
        self.btn_start = QtWidgets.QPushButton("Start TRPL")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.chk_autosave = QtWidgets.QCheckBox("Auto-save")
        self.chk_autosave.setChecked(True)
        self.btn_save = QtWidgets.QPushButton("Save As…")
        self.btn_load = QtWidgets.QPushButton("Load…")
        self.btn_load_cal = QtWidgets.QPushButton("Load λ-cal…")
        self.btn_export_spec = QtWidgets.QPushButton("Export spectral…")
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItem("2D space", MODE_2D)
        self.cmb_mode.addItem("Spectral", MODE_SPECTRAL)
        for b in (self.btn_tdc, self.btn_start, self.btn_stop, self.btn_save, self.btn_load, self.btn_load_cal,
                  self.btn_export_spec):
            b.setMinimumWidth(BTN_W - 20)
        self.lbl_progress = QtWidgets.QLabel("Idle")
        self.lbl_progress.setWordWrap(True)
        # dark reference (beam blocked): a background for the fits, next to the pre-t0 offset
        self.spin_dark = QtWidgets.QDoubleSpinBox()
        self.spin_dark.setRange(0.5, 3600.0)
        self.spin_dark.setDecimals(1)
        self.spin_dark.setValue(TRPL_DARK_ACQ_S)
        self.spin_dark.setToolTip("Acquisition time of the dark reference (longer = less noise).")
        self.btn_dark = QtWidgets.QPushButton("Take dark…")
        self.btn_dark.setToolTip(
            "Dark reference: a TRPL acquisition with the excitation beam blocked (you are asked to block it).\n"
            "Each pixel's dark counts can then be subtracted in the Fitting tab instead of the pre-t0 offset,\n"
            "which also holds signal left over from earlier pulses when the lifetime is long.\n"
            "It is scaled to each acquisition's time and bin width, saved as a file and embedded in saved data.")
        self.btn_load_dark = QtWidgets.QPushButton("Load dark…")
        self.btn_clear_dark = QtWidgets.QPushButton("Clear dark")
        for b in (self.btn_dark, self.btn_load_dark, self.btn_clear_dark):
            b.setMinimumWidth(BTN_W - 20)
        self.lbl_dark_ref = QtWidgets.QLabel("No dark reference.")
        self.lbl_dark_ref.setStyleSheet("color:#6b7280")
        self.lbl_dark_ref.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        dark_row = QtWidgets.QHBoxLayout()
        dark_row.addWidget(QtWidgets.QLabel("Dark reference"))
        if self.viewer:
            self.spin_dark.hide()
            self.btn_dark.hide()
        else:
            dark_row.addWidget(QtWidgets.QLabel("acq (s)"))
            dark_row.addWidget(self.spin_dark)
            dark_row.addWidget(self.btn_dark)
        dark_row.addWidget(self.btn_load_dark)
        dark_row.addWidget(self.btn_clear_dark)
        dark_row.addWidget(self.lbl_dark_ref, 1)
        if self.viewer:
            g.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
            g.addWidget(self.cmb_mode, 0, 1)
            g.addWidget(self.btn_load, 0, 2)
            g.addWidget(self.btn_save, 0, 3)
            g.addWidget(self.btn_load_cal, 0, 4)
            g.addWidget(self.btn_export_spec, 0, 5)
            g.addWidget(self.lbl_progress, 1, 0, 1, 7)
            g.addLayout(dark_row, 2, 0, 1, 7)
            g.setColumnStretch(6, 1)
            for w in (self.spin_acq, self.spin_bin, self.btn_tdc, self.btn_start, self.btn_stop, self.chk_autosave):
                w.hide()
        else:
            self.cmb_mode.hide()
            g.addWidget(QtWidgets.QLabel("Acquisition (s)"), 0, 0)
            g.addWidget(self.spin_acq, 0, 1)
            g.addWidget(QtWidgets.QLabel("Bin (ps)"), 0, 2)
            g.addWidget(self.spin_bin, 0, 3)
            g.addWidget(self.btn_tdc, 0, 4)
            g.addWidget(self.btn_start, 0, 5)
            g.addWidget(self.btn_stop, 0, 6)
            g.addWidget(self.chk_autosave, 0, 7)
            g.addWidget(self.btn_save, 0, 8)
            g.addWidget(self.btn_load, 0, 9)
            g.addWidget(self.btn_load_cal, 0, 10)
            g.addWidget(self.btn_export_spec, 0, 11)
            g.addWidget(self.lbl_progress, 1, 0, 1, 13)
            g.addLayout(dark_row, 2, 0, 1, 13)
            g.setColumnStretch(12, 1)
        root.addWidget(acq)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.map = PixelMapPlot("Counts at the selected time bin")
        self.map.hover_text_fn = self._hover_text
        lv.addWidget(self.map, 1)
        self.lbl_info = QtWidgets.QLabel("No data.")
        self.lbl_info.setWordWrap(True)
        lv.addWidget(self.lbl_info)
        hint = QtWidgets.QLabel("Left click: toggle pixel (spectral mode: also selects its column) · "
                                "Right click: exclude pixel")
        hint.setStyleSheet("color:#6b7280")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        splitter.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([700, 1040])

        # ---- histogram tab
        hist = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(hist)
        hv.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Time slice")
        tg = _grid(box)
        self.spin_t0 = QtWidgets.QDoubleSpinBox()
        self.spin_t1 = QtWidgets.QDoubleSpinBox()
        for s in (self.spin_t0, self.spin_t1):
            s.setRange(0.0, 1e9)
            s.setDecimals(4)
        self.btn_apply = QtWidgets.QPushButton("Apply time range")
        self.btn_full = QtWidgets.QPushButton("Full range")
        self.btn_apply.setToolTip("Also works during an acquisition; the time axes rescale to the range.")
        self.spin_rebin = QtWidgets.QSpinBox()
        self.spin_rebin.setRange(1, 10000)
        self.spin_rebin.setPrefix("×")
        self.chk_ref = QtWidgets.QCheckBox("Use this frame as reference")
        self.chk_log = QtWidgets.QCheckBox("Log y")
        self.chk_log_img = QtWidgets.QCheckBox("Log image")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lbl_time = QtWidgets.QLabel("t = --")
        tg.addWidget(QtWidgets.QLabel("Start (ns)"), 0, 0)
        tg.addWidget(self.spin_t0, 0, 1)
        tg.addWidget(QtWidgets.QLabel("End (ns)"), 0, 2)
        tg.addWidget(self.spin_t1, 0, 3)
        apply_box = QtWidgets.QHBoxLayout()
        apply_box.addWidget(self.btn_apply)
        apply_box.addWidget(self.btn_full)
        tg.addLayout(apply_box, 0, 4)
        tg.addWidget(QtWidgets.QLabel("Rebin"), 0, 5)
        tg.addWidget(self.spin_rebin, 0, 6)
        tg.addWidget(self.chk_ref, 0, 7)
        tg.addWidget(self.chk_log, 0, 8)
        tg.addWidget(self.chk_log_img, 0, 9)
        tg.addWidget(QtWidgets.QLabel("Time bin"), 1, 0)
        tg.addWidget(self.slider, 1, 1, 1, 8)
        tg.addWidget(self.lbl_time, 1, 9)
        out_row = QtWidgets.QHBoxLayout()
        self.btn_map_origin = QtWidgets.QPushButton("Save map for Origin…")
        self.btn_map_origin.setToolTip(
            "CSV of the time-resolved spectrum in the selected time range, numbers only:\n"
            "first row = wavelength (nm; x in µm without a calibration), first column = time (ns),\n"
            "cells = center-row counts. The top-left cell is empty; missing values are empty cells.")
        self.btn_copy_decays = QtWidgets.QPushButton("Copy decays")
        self.btn_copy_decays.setToolTip(
            "Copy the plotted decay traces (selected range) to the clipboard as tab-separated numbers:\n"
            "time (ns), then one column per trace in legend order. Paste into Origin.")
        self.chk_dark_sub = QtWidgets.QCheckBox("Subtract dark reference")
        self.chk_dark_sub.setChecked(True)
        out_row.addWidget(self.chk_dark_sub)
        out_row.addWidget(self.btn_map_origin)
        out_row.addWidget(self.btn_copy_decays)
        out_row.addStretch(1)
        tg.addLayout(out_row, 2, 0, 1, 10)
        leak_row = QtWidgets.QHBoxLayout()
        self.chk_leak = QtWidgets.QCheckBox("Align time zero on each pixel's first peak")
        self.chk_leak.setToolTip(
            "Fixes pixels the vendor alignment left off by anything from a fraction of a ns to hundreds of ns\n"
            "(including pixels that wrap around the window, and pixels aligned on the second of two peaks).\n"
            "1. Coarse: every trace is cross-correlated (circularly) with a template built from the data, so\n"
            "   large offsets are found whichever peak is higher.\n"
            "2. Fine: time zero of each pixel = the rising edge of its first peak at half height, between bins\n"
            "   (the flat top of the peak is a poor marker).\n"
            "3. Each pixel is shifted by whole bins so its edge lands on the median edge of the largest group.\n"
            "Applied to the view, fits and exports; Save As still writes the raw counts. Hover a pixel for its shift.")
        self.lbl_leak = QtWidgets.QLabel("")
        self.lbl_leak.setWordWrap(True)
        self.lbl_leak.setStyleSheet("color:#6b7280")
        leak_row.addWidget(self.chk_leak)
        leak_row.addWidget(self.lbl_leak, 1)
        tg.addLayout(leak_row, 3, 0, 1, 10)
        hv.addWidget(box)

        self.hist_pages = QtWidgets.QStackedWidget()
        self.hist_plot = HistogramPlot()
        self.hist_pages.addWidget(self.hist_plot)
        sp = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.tspec = TimeSpectrumImage()
        sp.addWidget(self.tspec)
        bottom = QtWidgets.QWidget()
        bh = QtWidgets.QHBoxLayout(bottom)
        bh.setContentsMargins(0, 0, 0, 0)
        self.col_hist = HistogramPlot()
        self.spec_at_t = SpectrumPlot()
        bh.addWidget(self.col_hist, 1)
        bh.addWidget(self.spec_at_t, 1)
        sp.addWidget(bottom)
        sp.setStretchFactor(0, 3)
        sp.setStretchFactor(1, 2)
        self.hist_pages.addWidget(sp)
        hv.addWidget(self.hist_pages, 1)
        self.tabs.addTab(hist, "Histogram")

        # ---- fitting tab
        self.fitting = TrplFittingTab(self)
        self.tabs.addTab(self.fitting, "Fitting")

        # ---- lifetime tab
        self.lifetime = TrplLifetimeTab(self)
        self.tabs.addTab(self.lifetime, "Lifetime")

        # ---- signals
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_tdc.clicked.connect(self._on_tdc)
        self.btn_save.clicked.connect(self._on_save_as)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_dark.clicked.connect(self._on_take_dark)
        self.btn_load_dark.clicked.connect(self._on_load_dark)
        self.btn_clear_dark.clicked.connect(lambda: self.set_dark_reference(None, "cleared"))
        self.btn_load_cal.clicked.connect(self._on_load_cal)
        self.btn_export_spec.clicked.connect(self._on_export_spectral)
        self.btn_map_origin.clicked.connect(self._on_save_map_origin)
        self.btn_copy_decays.clicked.connect(self._on_copy_decays)
        self.cmb_mode.currentIndexChanged.connect(lambda _i: self.state.set_mode(self.cmb_mode.currentData()))
        self.btn_apply.clicked.connect(self._on_apply_range)
        self.btn_full.clicked.connect(self._on_full_range)
        for sp in (self.spin_t0, self.spin_t1):
            sp.valueChanged.connect(self._on_range_edited)
        self.spin_rebin.valueChanged.connect(self._on_rebin)
        self.chk_ref.toggled.connect(self._on_ref_toggled)
        self.chk_leak.toggled.connect(lambda _v: self._on_leak_changed())
        self.chk_log.toggled.connect(lambda _v: self._update_hist())
        self.chk_log_img.toggled.connect(lambda _v: self._update_hist())
        self.chk_dark_sub.toggled.connect(lambda _v: self._on_dark_sub_changed())
        self.slider.valueChanged.connect(self.set_current)
        self.map.pixel_clicked.connect(self._on_pixel_clicked)
        self.hist_plot.time_clicked.connect(self._on_time_clicked)
        self.col_hist.time_clicked.connect(self._on_time_clicked)
        self.tspec.time_clicked.connect(self._on_time_clicked)
        self.spec_at_t.column_clicked.connect(self._toggle_column)
        self.tabs.currentChanged.connect(lambda _i: self._refresh_all())

    # =========================================================== data
    @property
    def has_data(self) -> bool:
        return self.view_counts.shape[0] > 0

    @property
    def raw(self) -> Optional[TrplData]:
        return self._raw

    def spectral_matrix(self) -> np.ndarray:
        """Center-row counts [bins, 17] of the view, minus the dark reference when 'Subtract dark' is on."""
        if self._spec is None:
            spec = center_row_matrix(self.view_counts, self.state.excluded)
            dark = self._spec_dark()
            self._spec = spec if dark is None else spec - dark
        return self._spec

    def _spec_dark(self) -> Optional[np.ndarray]:
        """Center-row dark counts per view bin (17 columns) if the spectral views subtract it, else None."""
        if not self.chk_dark_sub.isChecked():
            return None
        dark, _ = self.dark_per_bin()
        return None if dark is None else center_row_matrix(dark[None, :], self.state.excluded)[0]

    def dark_per_bin(self) -> Tuple[Optional[np.ndarray], str]:
        """The dark reference scaled to the current view (counts per view bin per pixel), or None and why not.

        Scaled to the requested acquisition time; to the elapsed time while
        acquiring or after a cancelled run.
        """
        ref = self.dark_ref
        if ref is None:
            return None, "no dark reference: take or load one in the Acquisition box"
        if not self.has_data or self._raw is None:
            return None, "no data"
        if ref.num_pixels != self.view_counts.shape[1]:
            return None, f"the dark reference has {ref.num_pixels} pixels, the data {self.view_counts.shape[1]}"
        meta = self._raw.metadata
        ms, elapsed = meta.get("measurement_ms"), meta.get("elapsed_s")
        ok_ms = isinstance(ms, (int, float)) and ms > 0
        ok_el = isinstance(elapsed, (int, float)) and elapsed > 0
        if ok_el and (meta.get("live") or meta.get("cancelled") or not ok_ms):
            acq_s, how = float(elapsed), " (elapsed)"
        elif ok_ms:
            acq_s, how = float(ms) / 1000.0, ""
        else:
            return None, "the data has no acquisition time (measurement_ms), so the dark cannot be scaled"
        bin_ps = float(np.median(np.diff(self.view_time))) * 1000.0 if self.view_time.size > 1 else self._raw.bin_width_ps
        dark = ref.per_bin(acq_s, bin_ps)
        return dark, (f"dark reference = {ref.describe()}; scaled to {acq_s:g} s{how} and {bin_ps:g} ps bins: "
                      f"{dark.mean():.3g} counts/bin on average")

    @property
    def dark_subtracted(self) -> bool:
        """The spectral views (image, column decays, spectrum at t, exports) have the dark subtracted."""
        return self._spec_dark() is not None

    def _on_dark_sub_changed(self) -> None:
        self._spec = None
        self._sync_dark_sub()
        self._update_hist()

    def selected_traces(self):
        """(label, counts per bin over the whole view) of the selected pixels, in both modes."""
        if not self.has_data:
            return []
        calib = self.state.calibration if self.state.spectral else None
        out = []
        for p in sorted(self._sel_pixels):
            lab = f"px {p}"
            if calib is not None:
                lab += f" ({float(calib.wavelength(COLUMN_X_UM[column_of(p)])):.1f} nm)"
            out.append((lab, self.view_counts[:, p].astype(np.float64)))
        return out

    def selected_columns(self):
        return sorted(self._sel_cols)

    def ref_index(self) -> Optional[int]:
        if self.ref_time is None or not self.has_data:
            return None
        return int(np.argmin(np.abs(self.view_time - self.ref_time)))

    def time_of(self, idx: int) -> float:
        return float(self.view_time[int(idx)]) if self.has_data else 0.0

    def set_data(self, data: TrplData, keep_view: bool = False) -> None:
        keep_view = keep_view and self.has_data
        old_t = self.time_of(self.current) if keep_view else None
        # keep_view keeps the applied time range (it follows the growing data while acquiring)
        self._raw = data
        self._rebuild_view()
        if not keep_view:
            self._req_range = None
            self._range_edit_pending = False
            self.ref_time = None
        self._apply_requested_range()
        if not keep_view:
            self.chk_ref.blockSignals(True)
            self.chk_ref.setChecked(False)
            self.chk_ref.blockSignals(False)
        self._sync_range_widgets()
        if old_t is not None:
            self.set_current(int(np.argmin(np.abs(self.view_time - old_t))), force=True)
        else:
            totals = self.view_counts.sum(axis=1)
            self.set_current(int(np.argmax(totals)) if totals.size else 0, force=True)
        self._sync_data_controls()
        self._refresh_all()
        if not keep_view:
            self._rescale_time_axes()

    def _rebuild_view(self) -> None:
        raw = self._raw
        self.leak = None
        if raw is None:
            self.view_time = np.zeros(0)
            self.view_counts = np.zeros((0, 93), dtype=np.int64)
        else:
            counts = raw.counts
            if self.chk_leak.isChecked():
                try:
                    self.leak = align_first_peaks(raw.counts, raw.time_ns)
                    counts = self.leak.counts
                except Exception as exc:
                    self.lbl_leak.setText(f"Time-zero alignment failed ({exc}); raw data shown.")
            self.view_counts, self.view_time = rebin_time(counts, raw.time_ns, self._rebin)
        if self.leak is not None:
            self.lbl_leak.setText(self.leak.message)
        elif not self.chk_leak.isChecked():
            self.lbl_leak.setText("")
        self._spec = None
        self.data_version += 1
        self.fitting.invalidate()
        self.lifetime.invalidate()

    def _apply_requested_range(self) -> None:
        """Bin range for the applied (start, end) times; it follows the data while acquiring."""
        n = self.view_counts.shape[0]
        if not n or self._req_range is None:
            self.range = (0, max(n - 1, 0))
            return
        t0, t1 = self._req_range
        r0 = int(np.clip(np.searchsorted(self.view_time, t0 - 1e-9, side="left"), 0, n - 1))
        r1 = int(np.clip(np.searchsorted(self.view_time, t1 + 1e-9, side="right") - 1, r0, n - 1))
        self.range = (r0, r1)

    def _sync_range_widgets(self) -> None:
        n = self.view_counts.shape[0]
        for s in (self.slider, self.fitting.slider):
            s.blockSignals(True)
            s.setRange(self.range[0], self.range[1])
            s.blockSignals(False)
        if n and not self._range_edit_pending:
            for sp, v in ((self.spin_t0, self.time_of(self.range[0])), (self.spin_t1, self.time_of(self.range[1]))):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)

    def _on_range_edited(self, _v: float) -> None:
        self._range_edit_pending = True

    def _rescale_time_axes(self) -> None:
        """Fit the time axes to the selected range (after Apply, a new file or a new acquisition)."""
        if not self.has_data:
            return
        t0, t1 = self.time_of(self.range[0]), self.time_of(self.range[1])
        self.tspec.set_time_range(t0, t1)
        self.fitting.set_time_range(t0, t1)

    def set_current(self, idx: int, force: bool = False) -> None:
        if not self.has_data:
            return
        idx = int(np.clip(idx, self.range[0], self.range[1]))
        if idx == self.current and not force:
            return
        self.current = idx
        for s in (self.slider, self.fitting.slider):
            s.blockSignals(True)
            s.setValue(idx)
            s.blockSignals(False)
        text = f"t = {self.time_of(idx):.4f} ns (bin {idx})"
        self.lbl_time.setText(text)
        self.fitting.lbl_time.setText(text)
        self._update_map()
        self._update_cursors()
        self._slider_timer.start()

    # ============================================================ views
    def _refresh_all(self) -> None:
        try:
            self._update_map()
            self._update_hist()
            self.fitting.refresh()
            self.lifetime.refresh()
        except Exception as exc:
            self.status.emit(f"Display error: {exc}")

    def _update_map(self) -> None:
        if not self.has_data:
            self.lbl_info.setText("No data.")
            return
        frame = self.view_counts[self.current].astype(np.float64)
        self.map.set_values(frame, unit="Counts")
        sel = f"Pixels: {','.join(str(p) for p in sorted(self._sel_pixels)) or 'none'}"
        if self.state.spectral:
            sel += f" | Columns: {','.join(str(c) for c in sorted(self._sel_cols)) or 'none'}"
        excl = ",".join(str(p) for p in sorted(self.state.excluded)) or "none"
        raw = self._raw
        self.lbl_info.setText(
            f"t = {self.time_of(self.current):.4f} ns (bin {self.current}) | "
            f"{sel} | Excluded: {excl} | "
            f"{raw.n_bins} bins × {raw.bin_width_ps:g} ps, rebin ×{self._rebin}, "
            f"{'time-zero aligned, ' if self.leak is not None and self.leak.shift_bins.any() else ''}"
            f"{int(raw.counts.sum())} events | {os.path.basename(raw.source) if raw.source else 'unsaved'}"
        )

    def _update_cursors(self) -> None:
        t = self.time_of(self.current)
        self.hist_plot.set_cursor(t)
        self.col_hist.set_cursor(t)
        self.tspec.set_cursor(t)
        self.fitting.set_cursor(t)
        if self.state.spectral and self.isVisible():
            self._update_spectrum_at_t()

    def _hist_slice(self):
        r0, r1 = self.range
        return slice(r0, r1 + 1)

    def _update_hist(self) -> None:
        if not self.has_data:
            self.hist_plot.set_traces([])
            self.col_hist.set_traces([])
            return
        sl = self._hist_slice()
        t = self.view_time[sl]
        log = self.chk_log.isChecked()
        t_ref = self.ref_time
        if self.state.spectral:
            spec = self.spectral_matrix()
            log_img = self.chk_log_img.isChecked()
            label = None
            if self.dark_subtracted:
                label = "log10(1 + counts − dark)" if log_img else "Counts − dark"
            self.tspec.set_data(t, spec[sl], log=log_img, label=label)
            self.tspec.set_columns(sorted(self._sel_cols))
            calib = self.state.calibration
            traces = []
            for k, c in enumerate(sorted(self._sel_cols)):
                lab = f"col {c} ({float(calib.wavelength(COLUMN_X_UM[c])):.1f} nm)" if calib else f"col {c}"
                traces.append((lab, t, spec[sl, c], trace_color(k)))
            self.col_hist.set_log(log)
            self.col_hist.set_traces(traces)
            self.col_hist.set_region(float(t[0]), float(t[-1]))
            self.col_hist.set_reference(t_ref)
            self.col_hist.set_title("Center-row decays of the selected columns"
                                    + (" (dark subtracted)" if self.dark_subtracted else ""))
            self._update_spectrum_at_t()
        else:
            traces = [(f"px {p}", t, self.view_counts[sl, p], trace_color(k))
                      for k, p in enumerate(sorted(self._sel_pixels))]
            self.hist_plot.set_log(log)
            self.hist_plot.set_traces(traces)
            self.hist_plot.set_region(float(t[0]), float(t[-1]))
            self.hist_plot.set_reference(t_ref)
            self.hist_plot.set_title("Pixel histograms (left-click pixels on the map)")
        self._update_cursors()

    def _update_spectrum_at_t(self) -> None:
        if not self.has_data:
            return
        prof = self.spectral_matrix()[self.current]
        calib = self.state.calibration
        peak = locate_peak(COLUMN_X_UM, prof) if np.isfinite(prof).sum() >= 4 and np.nansum(prof) > 0 else None
        if peak is None:
            sub = "no peak fit"
        else:
            where = f"{float(calib.wavelength(peak.center)):.2f} nm" if calib else f"x = {peak.center:.1f} µm"
            sub = f"peak {where}, {fwhm_text(peak.sigma, calib)}"
        unit = "Counts − dark" if self.dark_subtracted else "Counts"
        self.spec_at_t.set_data(prof, peak, unit, f"Spectrum at t = {self.time_of(self.current):.3f} ns", sub)
        self.spec_at_t.set_selected_column(self.active_col)

    def _hover_text(self, pix: int) -> str:
        if not self.has_data:
            return ""
        text = f"{int(self.view_counts[self.current, pix])} counts in bin\n{int(self.view_counts[:, pix].sum())} total"
        if self.leak is not None:
            text += "\n" + self.leak.pixel_text(pix)
        return text

    # ========================================================== events
    def _on_pixel_clicked(self, pix: int, button: int) -> None:
        if button == 2:
            self.state.toggle_excluded(pix)
            return
        col = column_of(pix)
        if pix in self._sel_pixels:
            self._sel_pixels.discard(pix)
            # a column stays selected while another selected pixel is in it (keep at least one)
            if (self.state.spectral and col in self._sel_cols and len(self._sel_cols) > 1
                    and not any(column_of(p) == col for p in self._sel_pixels)):
                self._sel_cols.discard(col)
                if self.active_col == col:
                    self.active_col = max(self._sel_cols)
        else:
            self._sel_pixels.add(pix)
            if self.state.spectral:
                self._sel_cols.add(col)
                self.active_col = col
        self.map.set_selected(self._sel_pixels)
        if self.state.spectral:
            self.map.set_highlight_columns(self._sel_cols)
        self._update_hist()
        self._update_map()
        if self.state.spectral:
            self.fitting.refresh()
        self.lifetime.refresh()

    def _toggle_column(self, col: int) -> None:
        col = int(col)
        if col in self._sel_cols and len(self._sel_cols) > 1 and col == self.active_col:
            self._sel_cols.discard(col)
            self.active_col = max(self._sel_cols)
        else:
            self._sel_cols.add(col)
            self.active_col = col
        self.map.set_highlight_columns(self._sel_cols)
        self._update_hist()
        self._update_map()
        self.fitting.refresh()
        self.lifetime.refresh()

    def _on_time_clicked(self, t_ns: float) -> None:
        if self.has_data:
            self.set_current(int(np.argmin(np.abs(self.view_time - t_ns))))

    def _on_apply_range(self) -> None:
        if not self.has_data:
            return
        self._req_range = tuple(sorted((self.spin_t0.value(), self.spin_t1.value())))
        self._range_edit_pending = False
        self._set_range_and_redraw()

    def _on_full_range(self) -> None:
        self._req_range = None
        self._range_edit_pending = False
        self._set_range_and_redraw()

    def _set_range_and_redraw(self) -> None:
        if not self.has_data:
            return
        self._apply_requested_range()
        self._sync_range_widgets()
        self.set_current(self.current, force=True)
        self._update_hist()
        self.fitting.refresh()
        self.lifetime.refresh()
        self._rescale_time_axes()
        t0, t1 = self.time_of(self.range[0]), self.time_of(self.range[1])
        self.status.emit(f"Time range {t0:.3f}–{t1:.3f} ns" + (" (all data)" if self._req_range is None else ""))

    def _on_rebin(self, k: int) -> None:
        self._rebin = int(k)
        if self._raw is not None:
            self.set_data(self._raw, keep_view=True)

    def _on_leak_changed(self) -> None:
        if self._raw is None:
            self.lbl_leak.setText("Applied when data is loaded." if self.chk_leak.isChecked() else "")
            return
        self.set_data(self._raw, keep_view=True)
        self.status.emit(self.leak.message if self.leak is not None else "Time-zero alignment off: raw alignment shown.")

    def processing_metadata(self) -> Dict[str, object]:
        """Header rows describing post-processing of the view (for exports made from it)."""
        return {} if self.leak is None else {"time_zero_alignment": self.leak.summary()}

    def _on_ref_toggled(self, checked: bool) -> None:
        if not self.has_data:
            self.ref_time = None
            return
        self.ref_time = self.time_of(self.current) if checked else None
        self.fitting.reference_changed()
        if checked:
            self.status.emit(f"Reference set at t = {self.ref_time:.4f} ns (bin {self.current}).")
        else:
            self.status.emit("Reference cleared.")
        self._update_hist()

    def _on_mode_changed(self, mode: str) -> None:
        spectral = self.state.spectral
        i = self.cmb_mode.findData(mode)
        if i >= 0 and i != self.cmb_mode.currentIndex():
            self.cmb_mode.blockSignals(True)
            self.cmb_mode.setCurrentIndex(i)
            self.cmb_mode.blockSignals(False)
        self.hist_pages.setCurrentIndex(1 if spectral else 0)
        self.btn_map_origin.setVisible(spectral)
        self.chk_log_img.setVisible(spectral)
        self.chk_dark_sub.setVisible(spectral)
        self.map.set_spectral(spectral, self.state.calibration)
        self.map.set_selected(self._sel_pixels)
        if spectral:
            self.map.set_highlight_columns(self._sel_cols)
        self.fitting.set_mode(spectral)
        self._refresh_all()

    def _on_excluded_changed(self) -> None:
        self.map.set_excluded(self.state.excluded)
        self._spec = None
        self.data_version += 1
        self.fitting.invalidate()
        self.lifetime.invalidate()
        self._refresh_all()

    def _on_calibration_changed(self) -> None:
        calib = self.state.calibration
        self.map.set_spectral(self.state.spectral, calib)
        if self.state.spectral:
            self.map.set_highlight_columns(self._sel_cols)
        for w in (self.tspec, self.spec_at_t):
            w.set_calibration(calib)
        self.fitting.set_calibration(calib)
        self._refresh_all()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._refresh_all()

    # ======================================================= acquisition
    def _on_tdc(self) -> None:
        if self.device is None:
            return
        self.lbl_progress.setText("TDC calibration check + channel alignment ...")
        self.device.run_command("tdc", lambda c: " | ".join(c.calibrate_and_align(align_ms=ALIGN_MS)))

    def _on_command_done(self, name: str, ok: bool, msg: str) -> None:
        if name != "tdc":
            return
        self.lbl_progress.setText(("TDC: " if ok else "TDC failed: ") + msg)
        self.status.emit(("Calibrate + Align done: " if ok else "Calibrate + Align failed: ") + msg)

    def _on_start(self) -> None:
        if self.device is None or self._acquiring:
            return
        if self._unsaved and not self.chk_autosave.isChecked():
            ans = QtWidgets.QMessageBox.question(
                self, "Start TRPL", "The current data has not been saved. Discard it and start a new acquisition?")
            if ans != QtWidgets.QMessageBox.Yes:
                return
        ms = int(round(self.spin_acq.value() * 1000.0))
        bin_ps = int(self.spin_bin.value())
        if self.device.start_trpl(ms, bin_ps):
            self._acquiring = True
            self._new_acquisition = True
            calib = self.state.calibration
            self._acq_params = {"measurement_ms": ms, "bin_width_ps": bin_ps,
                                "started_at": datetime.now().isoformat(timespec="seconds"),
                                "mode": self.state.mode,
                                "wavelength_calibration": None if calib is None else calib.to_dict()}
            self.lbl_progress.setText("Starting TRPL acquisition ...")
            self.lbl_progress.setStyleSheet("")

    # ------------------------------------------------- dark reference
    def _on_take_dark(self) -> None:
        if self.device is None or self._acquiring:
            return
        acq_s, bin_ps = float(self.spin_dark.value()), int(self.spin_bin.value())
        ans = QtWidgets.QMessageBox.question(
            self, "Dark reference",
            "Block the excitation beam (keep the laser running, so the TCSPC timing stays synchronized), "
            f"then press OK.\n\nA {acq_s:g} s TRPL acquisition with {bin_ps} ps bins is taken as the dark "
            "reference. The data shown now is kept.",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Ok)
        if ans != QtWidgets.QMessageBox.Ok:
            return
        if self.device.start_trpl(int(round(acq_s * 1000.0)), bin_ps):
            self._acquiring = True
            self._dark_pending = True
            self.lbl_progress.setText("Dark reference: starting ...")
            self.lbl_progress.setStyleSheet("")
            self._sync_data_controls()

    def _on_dark_done(self, snap: Optional[TrplSnapshot], err: str) -> None:
        if err or snap is None or snap.cancelled or snap.counts_matrix.shape[0] == 0:
            why = err or ("cancelled" if snap is not None and snap.cancelled else "no data")
            self.lbl_progress.setText(f"Dark reference not taken: {why}")
            self.lbl_progress.setStyleSheet("color:#b91c1c")
            QtWidgets.QMessageBox.warning(self, "Dark reference",
                                          f"Dark reference not taken: {why}.\n\nUnblock the excitation beam.")
            return
        ref = DarkReference.from_counts(snap.counts_matrix, snap.bin_width_ps, snap.measurement_ms, detector="SPAD93")
        path = unique_path(os.path.join(save_dir("trpl"), f"spad93_dark_{timestamp()}.csv"))
        try:
            save_dark_csv(path, snap.time_axis_ns, snap.counts_matrix, ref,
                          {"device": "SPAD93", "elapsed_s": round(snap.elapsed_s, 3), "n_events": snap.n_events,
                           "host": self.device.host if self.device else None,
                           "port": self.device.port if self.device else None})
            ref.source = path
            saved = f"Saved to {path}."
        except Exception as exc:
            saved = f"Could not save it ({exc}); it is used in this session only."
        self.set_dark_reference(ref, "taken")
        self.lbl_progress.setText(f"Dark reference taken: {snap.elapsed_s:.1f} s, {snap.n_events} events")
        QtWidgets.QMessageBox.information(self, "Dark reference",
                                          f"{ref.describe()}.\n\n{saved}\n\nUnblock the excitation beam.")

    def set_dark_reference(self, ref: Optional[DarkReference], how: str = "") -> None:
        """Make ``ref`` the dark reference of the Fitting tab (None = none)."""
        n_pix = int(self.view_counts.shape[1])
        if ref is not None and ref.num_pixels != n_pix:
            QtWidgets.QMessageBox.warning(self, "Dark reference",
                                          f"The dark reference has {ref.num_pixels} pixels; this detector has {n_pix}.")
            return
        self.dark_ref = ref
        if ref is None:
            text = "No dark reference (the Fitting tab can use the pre-t0 offset)."
        else:
            text = ref.describe() + (f" — {os.path.basename(ref.source)}" if ref.source else "")
        self.lbl_dark_ref.setText(text)
        self.lbl_dark_ref.setToolTip(text if ref is None or not ref.source else f"{text}\n{ref.source}")
        self.fitting.on_dark_changed()
        self._spec = None
        self._sync_data_controls()
        self._update_hist()
        if how:
            self.status.emit(f"Dark reference {how}: {text}")

    def _on_load_dark(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load dark reference", save_dir("trpl"),
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            ref = load_dark_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load dark reference", f"Could not load {os.path.basename(path)}:\n{exc}")
            return
        self.set_dark_reference(ref, "loaded")

    def _on_stop(self) -> None:
        if self.device is not None:
            self.device.stop_trpl()

    def _on_progress(self, snap: TrplSnapshot) -> None:
        self._pending_snapshot = snap
        if not self._progress_pending:
            self._progress_pending = True
            QtCore.QTimer.singleShot(0, self._apply_progress)

    def _apply_progress(self) -> None:
        self._progress_pending = False
        snap = self._pending_snapshot
        if snap is None or not self._acquiring:
            return
        total_s = snap.measurement_ms / 1000.0
        if self._dark_pending:  # the dark reference never replaces the data shown
            self.lbl_progress.setText(f"Dark reference: {snap.elapsed_s:.1f} / {total_s:.1f} s, {snap.n_events} events"
                                      + (" — stopping" if snap.cancelled else ""))
            return
        rate = snap.n_events / snap.elapsed_s if snap.elapsed_s > 0 else 0.0
        self.lbl_progress.setText(
            f"Running: {snap.elapsed_s:.1f} / {total_s:.1f} s, {snap.n_events} events "
            f"({rate / 1e3:.1f} kcps), {snap.counts_matrix.shape[0]} bins"
            + (f", {snap.out_of_window} out of window" if snap.out_of_window else "")
            + (" — stopping" if snap.cancelled else "")
        )
        if snap.counts_matrix.shape[0] == 0:
            return
        data = TrplData(time_ns=snap.time_axis_ns, counts=snap.counts_matrix, bin_width_ps=snap.bin_width_ps,
                        metadata={"live": True, "elapsed_s": snap.elapsed_s}, source="")  # elapsed: scales the dark
        keep = self.has_data and not self._new_acquisition
        self._new_acquisition = False
        if self.isVisible():
            self.set_data(data, keep_view=keep)
        else:
            self._raw = data
            self._rebuild_view()
            if not keep:
                self._req_range = None
                self._range_edit_pending = False
            self._apply_requested_range()
            self._sync_range_widgets()

    def _on_done(self, snap: Optional[TrplSnapshot], err: str) -> None:
        self._acquiring = False
        self._pending_snapshot = None
        if self._dark_pending:
            self._dark_pending = False
            self._on_dark_done(snap, err)
            self._sync_data_controls()
            return
        if snap is not None and snap.counts_matrix.shape[0] > 0 and snap.n_events > 0:
            meta = dict(getattr(self, "_acq_params", {}))
            meta.update({
                "elapsed_s": round(snap.elapsed_s, 3),
                "n_events": snap.n_events,
                "done": snap.done,
                "cancelled": snap.cancelled,
                "out_of_window": snap.out_of_window,
                "markers": {str(k): v for k, v in snap.marker_counts},
                "warnings": list(snap.warnings),
                "error": err or None,
                "host": self.device.host if self.device else None,
                "port": self.device.port if self.device else None,
                "pixel_limit_mcps": PIXEL_LIMIT_MCPS,
            })
            data = TrplData(time_ns=snap.time_axis_ns, counts=snap.counts_matrix, bin_width_ps=snap.bin_width_ps,
                            metadata=meta, source="")
            self._unsaved = True
            self.set_data(data, keep_view=self.has_data and not self._new_acquisition)
            self._new_acquisition = False
            how = "cancelled" if snap.cancelled else ("done" if snap.done else "ended early")
            self.lbl_progress.setText(
                f"TRPL {how}: {snap.elapsed_s:.1f} s, {snap.n_events} events, {snap.counts_matrix.shape[0]} bins"
                + (f" | {snap.warnings[0]}" if snap.warnings else ""))
            if self.chk_autosave.isChecked():
                path = unique_path(os.path.join(save_dir("trpl"), f"spad93_trpl_{timestamp()}.csv"))
                self.save_to(path)
        else:
            self.lbl_progress.setText("TRPL finished without events." if not err else f"TRPL failed: {err}")
        if err:
            self.lbl_progress.setStyleSheet("color:#b91c1c")
            hint = "\n\nThe detector is in overexposure shutdown: use Reset (SI)." if "SI" in err else ""
            QtWidgets.QMessageBox.warning(self, "TRPL acquisition", f"{err}{hint}")
        self._sync_data_controls()

    # ============================================================ files
    def _metadata(self) -> Dict[str, object]:
        """Header rows of a saved TRPL file.

        The mode and wavelength calibration are the ones in effect while
        measuring (kept from the acquisition or from the loaded file).  A
        calibration loaded later is saved too, marked as applied afterwards.
        """
        raw = self._raw
        old: Dict[str, object] = dict(raw.metadata) if raw is not None else {}
        for key in ("device", "mode", "mode_description", "data_columns", "wavelength_calibration_applied",
                    "wavelength_calibration_source", "wavelength_calibration", "saved_at", "geometry",
                    "excluded_pixels", "bin_width_ps", "live"):
            old.pop(key, None)
        mode = raw.metadata.get("mode") if raw is not None else None
        if mode not in (MODE_2D, MODE_SPECTRAL):
            mode = self.state.mode
        file_cal = calibration_from_metadata(raw.metadata) if raw is not None else None  # any header version
        source = raw.metadata.get("wavelength_calibration_source") if raw is not None else None
        if file_cal is not None:
            calib = file_cal.to_dict()
            source = source or "active during the measurement"
        elif self.state.calibration is not None:
            calib = self.state.calibration.to_dict()
            source = "loaded after the measurement (not active while measuring)"
        else:
            calib, source = None, None
        meta: Dict[str, object] = {
            "device": "SPAD93",
            "mode": mode,
            "mode_description": "spectral (center row = wavelength axis)" if mode == MODE_SPECTRAL else "2D space",
            "data_columns": "time_ns, pixel_0 ... pixel_92 = raw counts per bin of all 93 pixels (any mode)",
            "wavelength_calibration_applied": calib is not None,
        }
        if source:
            meta["wavelength_calibration_source"] = source
        meta.update(old)
        if raw is not None:
            meta["bin_width_ps"] = raw.bin_width_ps
        meta.update({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "excluded_pixels": sorted(self.state.excluded),
            "geometry": GEOMETRY_NOTE,
            "wavelength_calibration": calib,
        })
        if self.dark_ref is not None:  # restored on load, so the file carries its dark reference
            meta["dark_reference"] = self.dark_ref.summary()
        else:
            meta.pop("dark_reference", None)
        return meta

    def save_to(self, path: str) -> None:
        raw = self._raw
        if raw is None:
            return
        counts = trim_trailing_empty(raw.counts)
        time_ns = raw.time_ns[: counts.shape[0]]
        meta = self._metadata()

        def job() -> str:
            return save_trpl_csv(path, time_ns, counts, meta)

        if counts.shape[0] > TRPL_SAVE_WORKER_ROWS:
            if self._save_worker is not None:
                self.status.emit("A save is still running; try again in a moment.")
                return
            self.status.emit(f"Saving {counts.shape[0]} rows ...")
            w = CallWorker(job, self)
            w.done.connect(lambda ok, text: self._on_saved(ok, text, path))
            w.finished.connect(self._on_save_worker_finished)
            self._save_worker = w
            w.start()
        else:
            try:
                job()
                self._on_saved(True, path, path)
            except Exception as exc:
                self._on_saved(False, str(exc), path)

    def _on_save_worker_finished(self) -> None:
        w, self._save_worker = self._save_worker, None
        if w is not None:
            w.deleteLater()

    def _on_saved(self, ok: bool, text: str, path: str) -> None:
        if ok:
            self._unsaved = False
            self._saved_path = path
            if self._raw is not None:
                self._raw.source = path
            self.status.emit(f"TRPL data saved: {path}")
            self._update_map()
        else:
            QtWidgets.QMessageBox.critical(self, "Save TRPL", f"Could not save:\n{text}")

    @property
    def saving(self) -> bool:
        return self._save_worker is not None

    @property
    def busy_background(self) -> bool:
        """A save or a Fit All is still running in a worker thread."""
        return self.saving or self.fitting.busy or self.lifetime.busy

    def cancel_background(self) -> None:
        self.fitting.cancel()
        self.lifetime.cancel()

    @property
    def unsaved(self) -> bool:
        return self._unsaved and self._raw is not None

    def _on_save_as(self) -> None:
        if self._raw is None:
            QtWidgets.QMessageBox.information(self, "Save TRPL", "No TRPL data to save.")
            return
        default = os.path.join(save_dir("trpl"), f"spad93_trpl_{timestamp()}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save TRPL CSV", default,
                                                        "CSV files (*.csv);;All files (*)")
        if path:
            self.save_to(path)

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open TRPL CSV", save_dir("trpl"),
                                                        "CSV files (*.csv);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> bool:
        if self._acquiring:
            self.status.emit("Stop the acquisition before loading a file.")
            return False
        try:
            data = load_trpl_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load TRPL", f"Could not load {os.path.basename(path)}:\n{exc}")
            return False
        file_cal = calibration_from_metadata(data.metadata)
        if self.viewer:
            ex = data.metadata.get("excluded_pixels")
            if isinstance(ex, list):
                self.state.set_excluded(ex)
            if file_cal is not None:
                self.state.set_calibration(file_cal, f"{path} (embedded)")
            mode = data.metadata.get("mode")
            if mode in (MODE_2D, MODE_SPECTRAL):
                self.state.set_mode(mode)
        elif file_cal is not None and self.state.calibration is None:
            self.state.set_calibration(file_cal, f"{path} (embedded)")
        self._unsaved = False
        self._saved_path = path
        self.set_data(data)
        self.status.emit(f"Loaded {os.path.basename(path)}: {data.n_bins} bins × {data.bin_width_ps:g} ps")
        emb = data.metadata.get("dark_reference")
        if isinstance(emb, dict):
            try:
                self.set_dark_reference(DarkReference.from_summary(emb), "from the loaded file")
            except Exception as exc:
                self.status.emit(f"The file's dark reference was not used: {exc}")
        self.lbl_progress.setText(f"Loaded {path}")
        return True

    def _on_load_cal(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load wavelength calibration", save_dir("calibration"),
                                                        "JSON files (*.json);;All files (*)")
        if not path:
            return
        if self._calib_loader is not None:
            self._calib_loader(path)
            return
        try:
            self.state.set_calibration(load_calibration_json(path), path)
            self.status.emit(f"Calibration loaded: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load calibration", f"Could not load:\n{exc}")

    def _on_export_spectral(self) -> None:
        if not self.has_data:
            QtWidgets.QMessageBox.information(self, "Export spectral", "No TRPL data.")
            return
        base = os.path.splitext(os.path.basename(self._saved_path))[0] if self._saved_path else f"spad93_trpl_{timestamp()}"
        default = os.path.join(save_dir("trpl"), f"{base}_spectral.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export time-resolved spectrum", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        sl = self._hist_slice()
        calib = self.state.calibration
        meta = self._metadata()
        meta.update({"rebin": self._rebin, "content": "center-row intensity per column (time x lambda)"
                     + (", dark reference subtracted" if self.dark_subtracted else "")})
        if self.dark_subtracted:
            meta["dark_subtraction"] = self.dark_per_bin()[1]
        meta.update(self.processing_metadata())
        try:
            save_spectral_matrix_csv(path, self.view_time[sl], self.spectral_matrix()[sl],
                                     None if calib is None else calib.column_wavelengths(), meta)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export spectral", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Spectral matrix exported: {path}")

    def _base_name(self) -> str:
        return os.path.splitext(os.path.basename(self._saved_path))[0] if self._saved_path else f"spad93_trpl_{timestamp()}"

    def _on_save_map_origin(self) -> None:
        if not self.has_data:
            QtWidgets.QMessageBox.information(self, "Save map", "No TRPL data.")
            return
        default = os.path.join(save_dir("trpl"), f"{self._base_name()}_spectral_map_origin.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save time-resolved spectrum for Origin", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        sl = self._hist_slice()
        calib = self.state.calibration
        x = COLUMN_X_UM if calib is None else calib.column_wavelengths()
        try:
            save_origin_matrix_csv(path, x, self.view_time[sl], self.spectral_matrix()[sl])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save map", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Map saved for Origin ({'wavelength nm' if calib else 'x µm, no calibration'} × time ns, "
                         f"{self.view_time[sl].size} rows{', dark subtracted' if self.dark_subtracted else ''}): {path}")

    def decay_traces(self):
        """(label, time over the selected range, counts) exactly as plotted in the Histogram tab."""
        if not self.has_data:
            return []
        sl = self._hist_slice()
        t = self.view_time[sl]
        if self.state.spectral:
            spec = self.spectral_matrix()
            calib = self.state.calibration
            return [(f"col {c}" + (f" ({float(calib.wavelength(COLUMN_X_UM[c])):.1f} nm)" if calib else ""), t, spec[sl, c])
                    for c in sorted(self._sel_cols)]
        return [(f"px {p}", t, self.view_counts[sl, p]) for p in sorted(self._sel_pixels)]

    def _on_copy_decays(self) -> None:
        traces = self.decay_traces()
        if not traces:
            self.status.emit("Nothing to copy: select traces on the map first.")
            return
        QtWidgets.QApplication.clipboard().setText(columns_to_text([traces[0][1]] + [y for _, _, y in traces]))
        self.status.emit(f"Copied {traces[0][1].size} rows × {len(traces) + 1} columns: time (ns), "
                         + ", ".join(lab for lab, _, _ in traces))

    # ======================================================== controls
    def _sync_data_controls(self) -> None:
        has = self.has_data
        for w in (self.btn_apply, self.btn_full, self.spin_rebin, self.chk_ref, self.slider, self.spin_t0, self.spin_t1,
                  self.btn_export_spec):
            w.setEnabled(has)
        self.btn_save.setEnabled(self._raw is not None and not self._acquiring)
        self.btn_load.setEnabled(not self._acquiring)
        self.btn_load_dark.setEnabled(not self._acquiring)
        self.btn_clear_dark.setEnabled(self.dark_ref is not None and not self._acquiring)
        self._sync_dark_sub()

    def _sync_dark_sub(self) -> None:
        """'Subtract dark reference' is usable only with a dark reference that can be scaled to the data."""
        dark, text = self.dark_per_bin()
        self.chk_dark_sub.setEnabled(dark is not None)
        self.chk_dark_sub.setToolTip(
            "Spectral mode: subtract each pixel's dark counts (beam-blocked dark reference, scaled to this\n"
            "acquisition's time and bin width) from the time-resolved spectrum, the column decays, the spectrum\n"
            "at t and the spectral exports. Pixels with more dark counts no longer look brighter.\n"
            "Save As still writes the raw counts.\n\n" + text)

    def sync_controls(self, device_state: str) -> None:
        """Called by the main widget whenever the device state changes."""
        if self.viewer:
            return
        ready = device_state in (IDLE, LIVE)
        trpl = self.device is not None and self.device.trpl_running
        self.btn_start.setEnabled(ready and not self._acquiring)
        self.btn_tdc.setEnabled(ready and not self._acquiring)
        self.btn_dark.setEnabled(ready and not self._acquiring)
        self.spin_dark.setEnabled(not self._acquiring)
        self.btn_stop.setEnabled(trpl)
        self.spin_acq.setEnabled(not self._acquiring)
        self.spin_bin.setEnabled(not self._acquiring)
        self._sync_data_controls()
        if device_state == BUSY and not trpl and not self._acquiring:
            self.btn_stop.setEnabled(False)
