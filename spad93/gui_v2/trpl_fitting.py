"""Fitting sub-tab of the TRPL panel ("Fit All" over the selected time range).

2D space (ports of spad23_trpl_viewer): 1 sequential 2D Gaussian, 2 fixed-center
radial 1D Gaussian -> sigma_eq^2(t); 3 sequential double 2D Gaussian (two elliptical
spots, sigma_x and sigma_y each, on one offset) -> sigma_x1 sigma_y1 (t), sigma_x2 sigma_y2 (t)
(the sigma_eq^2 of each spot).  By default both centers stay
where the reference fit put them (two pump spots whose signal spreads); the
bottom trace is then each spot's integrated intensity 2 pi A sigma_x sigma_y.  With the
centers free it is their separation d(t).
By default every pixel's offset before time zero (its mean count per bin over a
pre-t0 window, automatic or typed in) is subtracted from its trace first, so
dark counts and hot pixels do not pull the 2D fits.  The Gaussian offset can then
be held at 0 ("Fix Gaussian offset at 0") instead of being fitted.
(Method 3, the pre-t0 subtraction and the fixed offset are ported from spad23/gui_v2.)

Spectral (transport): the y-profile of every column (one wavelength each) is
fitted with a 1D Gaussian in every time bin from time zero (the peak of the summed
decay inside the selected range) to the end of the range; σ² is shown by default.  The page shows (1) the profiles
of the selected columns at the selected time with their fits, (2) the fitted
width versus time for the selected columns and (3) the width as a
wavelength x time map.  Bins that are too noisy are not fitted, and fits that
fail the quality checks are left out of (2) and (3).
"""

import os
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from spad93 import spad93_analysis as an
from spad93.spad93_geometry import CENTER_Y_UM, COLUMN_X_UM, N_COLS, PIXEL_X_UM, PIXEL_Y_UM
from spad93.spad93_io import columns_to_text, save_columns_csv, timestamp
from spad93.gui_v2.config import BTN_W, save_dir
from spad93.gui_v2.map_plots import Gauss2DFitPlot, TimeSpectrumImage
from spad93.gui_v2.plots import (
    DoubleRadialPlot,
    MultiProfilePlot,
    MultiTracePlot,
    RadialFitPlot,
    TracePlot,
    double_fit_text,
    trace_color,
)
from spad93.gui_v2.workers import FitAllWorker

METHODS_2D = [
    ("2d_sequential", "1: sequential 2D Gaussian fit"),
    ("2d_fixed_center", "2: fixed-center radial 1D Gaussian fit"),
    ("2d_double_sequential", "3: sequential double 2D Gaussian fit (two spots)"),
]
DOUBLE = "2d_double_sequential"
METHODS_SPECTRAL = [
    ("widths_free", "1: 1D Gaussian per column and bin (y0 free)"),
    ("widths_fixed_y0", "2: 1D Gaussian per column and bin (y0 fixed from the time-integrated profile)"),
]
# "dark" = the pre-t0 offset (mean before the pulse: dark counts + signal left from earlier pulses);
# "darkref" = the dark reference taken with the beam blocked (TRPL panel), scaled to the data.
DARKREF_LABEL = "Dark reference (beam blocked, per pixel)"
BACKGROUNDS_SPECTRAL = [
    ("dark", "Dark level before t0 (per pixel)"),
    ("darkref", DARKREF_LABEL),
    ("frame", "Frame 10th percentile"),
]
BACKGROUNDS_2D = [
    ("dark", "Subtract each pixel's offset before t0"),
    ("darkref", DARKREF_LABEL),
    ("none", "None (only the Gaussian's own offset)"),
]
QUANTITIES = [
    ("sigma", "σ_y (µm)"),
    ("sigma2", "σ_y² (µm²)"),
    ("fwhm", "FWHM_y (µm)"),
]


class TrplFittingTab(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self._result = None  # FitAllResult (2D) or ColumnWidthMap (spectral)
        self._result_version = -1
        self._job_id = 0
        self._worker: Optional[FitAllWorker] = None
        self._calib = None
        self._spectral = False
        self._dark_cache: Tuple[object, Optional[np.ndarray], str] = (None, None, "")
        self._win_auto = True  # pre-t0 window follows the data until the user edits it
        self._bg_choice = {False: "dark", True: "dark"}  # background key per mode (spectral?)
        self._build_ui()

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        self.cmb_method = QtWidgets.QComboBox()
        self.btn_fit = QtWidgets.QPushButton("Fit All")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_export = QtWidgets.QPushButton("Export fits…")
        for b in (self.btn_fit, self.btn_cancel, self.btn_export):
            b.setMinimumWidth(BTN_W - 30)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setValue(0)
        top.addWidget(QtWidgets.QLabel("Method"))
        top.addWidget(self.cmb_method, 1)
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_cancel)
        top.addWidget(self.progress)
        top.addWidget(self.btn_export)
        root.addLayout(top)

        # method 3 (double Gaussian) constraints
        self.double_opts = QtWidgets.QWidget()
        dro = QtWidgets.QHBoxLayout(self.double_opts)
        dro.setContentsMargins(0, 0, 0, 0)
        self.chk_fix_centers = QtWidgets.QCheckBox("Fix both centers at the reference fit")
        self.chk_fix_centers.setChecked(True)
        self.chk_fix_centers.setToolTip(
            "Fit two Gaussians to the reference frame, then keep both centers there and fit only the\n"
            "amplitudes, widths (and offset) in the other frames, e.g. two pump spots whose signal spreads.\n"
            "Untick to let the centers move from frame to frame.")
        self.chk_fix_ratio = QtWidgets.QCheckBox("Fix A2/A1")
        self.chk_fix_ratio.setToolTip(
            "Hold the amplitude ratio of Gaussian 2 to Gaussian 1 fixed in every frame.\n"
            "'from reference' takes it from the reference fit; type a value to use that instead\n"
            "(e.g. the pump power ratio). A single noisy reference frame gives a noisy ratio.")
        self.spin_ratio = QtWidgets.QDoubleSpinBox()
        self.spin_ratio.setRange(0.0, 1000.0)
        self.spin_ratio.setDecimals(3)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setSpecialValueText("from reference")  # shown at 0
        self.spin_ratio.setValue(0.0)
        self.spin_ratio.setKeyboardTracking(False)
        self.chk_share_sigma = QtWidgets.QCheckBox("Same σ for both")
        self.chk_share_sigma.setToolTip(
            "Fit one σx and one σy shared by both Gaussians (also in the reference fit).")
        dro.addWidget(QtWidgets.QLabel("Method 3:"))
        dro.addWidget(self.chk_fix_centers)
        dro.addSpacing(12)
        dro.addWidget(self.chk_fix_ratio)
        dro.addWidget(self.spin_ratio)
        dro.addSpacing(12)
        dro.addWidget(self.chk_share_sigma)
        dro.addStretch(1)
        root.addWidget(self.double_opts)

        # background: per-pixel offset before t0 (both modes)
        bg_row = QtWidgets.QHBoxLayout()
        self.cmb_bg = QtWidgets.QComboBox()
        self.cmb_bg.setToolTip(
            "Subtracted from every pixel before the fit. The offset before t0 is each pixel's mean count per bin\n"
            "over the pre-t0 window (the least-squares constant of its trace there), subtracted from its whole trace.")
        self.spin_w0 = QtWidgets.QDoubleSpinBox()
        self.spin_w1 = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_w0, self.spin_w1):
            sp.setRange(-1e6, 1e9)
            sp.setDecimals(3)
            sp.setSuffix(" ns")
            sp.setKeyboardTracking(False)
            sp.setToolTip("Window before time zero used for the per-pixel offset.")
        self.btn_auto_win = QtWidgets.QPushButton("Auto")
        self.btn_auto_win.setToolTip("Window from the first bin to just before the rise of the summed trace.")
        self.lbl_dark = QtWidgets.QLabel("")
        self.lbl_dark.setStyleSheet("color:#6b7280")
        # long text must not widen the tab (it would squeeze the map); the tooltip has all of it
        self.lbl_dark.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        bg_row.addWidget(QtWidgets.QLabel("Background"))
        bg_row.addWidget(self.cmb_bg)
        bg_row.addWidget(QtWidgets.QLabel("Pre-t0 window"))
        bg_row.addWidget(self.spin_w0)
        bg_row.addWidget(QtWidgets.QLabel("to"))
        bg_row.addWidget(self.spin_w1)
        bg_row.addWidget(self.btn_auto_win)
        self.chk_fix_offset = QtWidgets.QCheckBox("Fix Gaussian offset at 0")
        self.chk_fix_offset.setToolTip(
            "2D methods: hold the constant offset of the Gaussian model(s) at 0 instead of fitting it.\n"
            "Use it with the pre-t0 subtraction, when the remaining background should be zero.\n"
            "(The spectral width fits already fix their offset at 0 unless 'Free offset' is on.)")
        bg_row.addWidget(self.chk_fix_offset)
        bg_row.addWidget(self.lbl_dark, 1)
        root.addLayout(bg_row)

        # spectral-only options
        self.spec_opts = QtWidgets.QWidget()
        so = QtWidgets.QHBoxLayout(self.spec_opts)
        so.setContentsMargins(0, 0, 0, 0)
        self.chk_free_offset = QtWidgets.QCheckBox("Free offset (cols ≥ 7 px)")
        self.spin_snr = QtWidgets.QDoubleSpinBox()
        self.spin_snr.setRange(0.5, 100.0)
        self.spin_snr.setSingleStep(0.5)
        self.spin_snr.setDecimals(1)
        self.spin_snr.setValue(an.MIN_SNR)
        self.spin_snr.setToolTip(
            "Noise gate. A column is not fitted in a bin when its brightest pixel is less than this many\n"
            "Poisson sigmas above the background, and a fit is dropped when its amplitude is less than\n"
            f"this many standard errors, a parameter is at its bound or σ has > {an.MAX_SIGMA_REL_ERR:.0%} error.\n"
            "Raising it hides more bins at once; lowering it below the Fit All value needs a new Fit All.")
        self.cmb_quantity = QtWidgets.QComboBox()
        for key, label in QUANTITIES:
            self.cmb_quantity.addItem(label, key)
        self.cmb_quantity.setCurrentIndex(self.cmb_quantity.findData("sigma2"))
        self.chk_center_row = QtWidgets.QCheckBox(f"Fix y0 at center row ({CENTER_Y_UM:g} µm)")
        self.chk_center_row.setToolTip("Every column's Gaussian is centered at the center row (y = "
                                       f"{CENTER_Y_UM:g} µm); only amplitude and width are fitted.\n"
                                       "Overrides the y0 choice of the method.")
        self.chk_norm = QtWidgets.QCheckBox("Normalize profiles")
        so.addWidget(self.chk_free_offset)
        so.addWidget(self.chk_center_row)
        so.addWidget(QtWidgets.QLabel("Min SNR"))
        so.addWidget(self.spin_snr)
        so.addWidget(QtWidgets.QLabel("Show"))
        so.addWidget(self.cmb_quantity)
        so.addWidget(self.chk_norm)
        so.addStretch(1)
        root.addWidget(self.spec_opts)

        row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lbl_time = QtWidgets.QLabel("t = --")
        row.addWidget(QtWidgets.QLabel("Time bin"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_time)
        root.addLayout(row)
        self.lbl_state = QtWidgets.QLabel("")
        self.lbl_state.setWordWrap(True)
        root.addWidget(self.lbl_state)

        self.pages = QtWidgets.QStackedWidget()
        # page 0: 2D space
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.trace = TracePlot()
        split.addWidget(self.trace)
        p2 = QtWidgets.QWidget()
        h2 = QtWidgets.QHBoxLayout(p2)
        h2.setContentsMargins(0, 0, 0, 0)
        self.map_fit = Gauss2DFitPlot()
        self.radial = RadialFitPlot()
        self.pixel_fit = DoubleRadialPlot()
        self.side_stack = QtWidgets.QStackedWidget()
        self.side_stack.addWidget(self.radial)
        self.side_stack.addWidget(self.pixel_fit)
        h2.addWidget(self.map_fit, 1)
        h2.addWidget(self.side_stack, 1)
        split.addWidget(p2)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        self.pages.addWidget(split)
        # page 1: spectral transport
        page_s = QtWidgets.QWidget()
        ps = QtWidgets.QVBoxLayout(page_s)
        ps.setContentsMargins(0, 0, 0, 0)
        copy_row = QtWidgets.QHBoxLayout()
        self.btn_copy_profiles = QtWidgets.QPushButton("Copy profiles @ t")
        self.btn_copy_profiles.setToolTip(
            "Copy the spatial profiles at the current time to the clipboard as tab-separated numbers.\n"
            "Per selected column (legend order): y (µm), data, then y (µm), fit curve (if fitted).\n"
            "Columns have different lengths; shorter ones end in empty cells. Paste into Origin.")
        copy_row.addWidget(self.btn_copy_profiles)
        copy_row.addStretch(1)
        ps.addLayout(copy_row)
        vs = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        hs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.profiles = MultiProfilePlot()
        self.width_trace = MultiTracePlot()
        hs.addWidget(self.profiles)
        hs.addWidget(self.width_trace)
        vs.addWidget(hs)
        self.width_map = TimeSpectrumImage()
        vs.addWidget(self.width_map)
        vs.setStretchFactor(0, 1)
        vs.setStretchFactor(1, 1)
        ps.addWidget(vs, 1)
        self.pages.addWidget(page_s)
        root.addWidget(self.pages, 1)

        self.btn_fit.clicked.connect(self._on_fit_all)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_copy_profiles.clicked.connect(self._on_copy_profiles)
        self.cmb_method.currentIndexChanged.connect(lambda _i: self.refresh())
        for w in (self.chk_fix_centers, self.chk_fix_ratio, self.chk_share_sigma):
            w.toggled.connect(lambda _v: self.invalidate())
        self.chk_fix_ratio.toggled.connect(self.spin_ratio.setEnabled)
        self.spin_ratio.setEnabled(False)
        self.spin_ratio.valueChanged.connect(lambda _v: self.invalidate() if self.chk_fix_ratio.isChecked() else None)
        self.cmb_bg.currentIndexChanged.connect(self._on_bg_changed)
        self.spin_w0.valueChanged.connect(self._on_window_edited)
        self.spin_w1.valueChanged.connect(self._on_window_edited)
        self.btn_auto_win.clicked.connect(self._on_auto_window)
        self.chk_fix_offset.toggled.connect(lambda _v: self.invalidate())
        self.chk_free_offset.toggled.connect(lambda _v: self.invalidate())
        self.chk_center_row.toggled.connect(self._on_center_row)
        self.spin_snr.valueChanged.connect(lambda _v: self.refresh())
        self.cmb_quantity.currentIndexChanged.connect(lambda _i: self.refresh())
        self.chk_norm.toggled.connect(lambda _v: self.refresh_current())
        self.slider.valueChanged.connect(lambda v: self.panel.set_current(v))
        self.width_trace.time_clicked.connect(self.panel._on_time_clicked)
        self.width_map.time_clicked.connect(self.panel._on_time_clicked)
        self.set_mode(False)
        self._sync_buttons()

    # -------------------------------------------------------- mode/state
    def set_mode(self, spectral: bool) -> None:
        self._spectral = spectral
        self.cmb_method.blockSignals(True)
        self.cmb_method.clear()
        for key, label in (METHODS_SPECTRAL if spectral else METHODS_2D):
            self.cmb_method.addItem(label, key)
        self.cmb_method.blockSignals(False)
        self.cmb_bg.blockSignals(True)
        self.cmb_bg.clear()
        for key, label in (BACKGROUNDS_SPECTRAL if spectral else BACKGROUNDS_2D):
            self.cmb_bg.addItem(label, key)
        self.cmb_bg.setCurrentIndex(max(0, self.cmb_bg.findData(self._bg_choice[spectral])))
        self.cmb_bg.blockSignals(False)
        self._update_darkref_item()
        self.pages.setCurrentIndex(1 if spectral else 0)
        self.spec_opts.setVisible(spectral)
        self.chk_fix_offset.setVisible(not spectral)
        self.cmb_method.setEnabled(not (spectral and self.chk_center_row.isChecked()))
        self.refresh()

    def set_calibration(self, calib) -> None:
        self._calib = calib
        self.width_map.set_calibration(calib)
        self.refresh()

    def method(self) -> str:
        if self._spectral and self.chk_center_row.isChecked():
            return "widths_center_row"
        return str(self.cmb_method.currentData() or "")

    def method_label(self) -> str:
        if self.method() == "widths_center_row":
            return f"1D Gaussian per column and bin (y0 fixed at the center row, {CENTER_Y_UM:g} µm)"
        return self.cmb_method.currentText().split(": ", 1)[-1]

    def _on_center_row(self, on: bool) -> None:
        self.cmb_method.setEnabled(not on)
        self.refresh()

    def invalidate(self) -> None:
        """Data, rebinning, exclusions or fit settings changed: old fits no longer apply."""
        if self._worker is not None:
            self._worker.cancel()
        self._result = None
        if hasattr(self, "trace"):
            self.refresh()

    def reference_changed(self) -> None:
        """Only the 2D methods use the reference frame."""
        if not self._spectral:
            self.invalidate()

    def set_time_range(self, t0: float, t1: float) -> None:
        self.width_map.set_time_range(t0, t1)
        self.width_trace.plot.setXRange(t0, t1, padding=0.01)

    def shift_times(self, d: float) -> None:
        """The panel's time axis moved by ``d`` ns (raw <-> time zero at 0): a manual pre-t0 window follows."""
        if not self._win_auto:
            self._set_window(self.spin_w0.value() + d, self.spin_w1.value() + d)

    def set_cursor(self, t_ns: float) -> None:
        self.trace.set_cursor(t_ns)
        self.width_trace.set_cursor(t_ns)
        self.width_map.set_cursor(t_ns)

    def _valid_result(self):
        res = self._result
        if res is None or res.method != self.method() or self._result_version != self.panel.data_version:
            return None
        return res

    def _col_label(self, c: int) -> str:
        if self._calib is not None:
            return f"col {c} ({float(self._calib.wavelength(COLUMN_X_UM[c])):.1f} nm)"
        return f"col {c} (x = {COLUMN_X_UM[c]:.0f} µm)"

    # ------------------------------------------------------- background
    def _on_bg_changed(self, _i: int) -> None:
        key = self.cmb_bg.currentData()
        if key is not None:
            self._bg_choice[self._spectral] = str(key)
        self.invalidate()

    def _on_window_edited(self, _v: float) -> None:
        self._win_auto = False
        self.invalidate()

    def _on_auto_window(self) -> None:
        self._win_auto = True
        self.invalidate()

    def _set_window(self, w0: float, w1: float) -> None:
        for sp, v in ((self.spin_w0, w0), (self.spin_w1, w1)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _dark(self) -> Tuple[Optional[np.ndarray], str]:
        """Per-pixel offset before t0 (mean count per bin over the pre-t0 window) of the current view, cached."""
        p = self.panel
        if self._win_auto and p.has_data:
            win = an.pre_pulse_window(p.view_time, p.view_counts.sum(axis=1))
            if win is not None:
                self._set_window(*win)
        w0, w1 = sorted((self.spin_w0.value(), self.spin_w1.value()))
        key = (p.data_version, w0, w1, self._win_auto)
        if self._dark_cache[0] == key:
            return self._dark_cache[1], self._dark_cache[2]
        dark, text = None, "no data"
        if p.has_data:
            sel = (p.view_time >= w0) & (p.view_time <= w1)
            n = int(sel.sum())
            if n == 0:
                text = f"no bins in the pre-t0 window {w0:.2f}–{w1:.2f} ns: offset not subtracted"
            else:
                dark = p.view_counts[sel].mean(axis=0).astype(np.float64)
                how = "auto" if self._win_auto else "manual"
                text = (f"pre-t0 offset = per-pixel mean over {w0:.2f}–{w1:.2f} ns ({how}, {n} bins), "
                        f"{dark.mean():.3g} counts/bin on average, max {dark.max():.3g} (pixel {int(np.argmax(dark))})")
        self._dark_cache = (key, dark, text)
        return dark, text

    # ---- dark reference (beam blocked; panel.dark_ref)
    def _darkref(self) -> Tuple[Optional[np.ndarray], str]:
        """The dark reference scaled to the current view (counts per view bin), or None and why not."""
        return self.panel.dark_per_bin()

    def _update_darkref_item(self) -> None:
        """Enable the dark-reference choice only when it can be used (tooltip says why not)."""
        i = self.cmb_bg.findData("darkref")
        if i < 0:
            return
        dark, text = self._darkref()
        item = self.cmb_bg.model().item(i)
        item.setEnabled(dark is not None)
        item.setToolTip(text)
        if dark is None and self.cmb_bg.currentIndex() == i:
            self.cmb_bg.blockSignals(True)
            self.cmb_bg.setCurrentIndex(self.cmb_bg.findData("dark"))
            self.cmb_bg.blockSignals(False)

    def on_dark_changed(self) -> None:
        """The panel's dark reference was taken, loaded or cleared."""
        self._update_darkref_item()
        if self.cmb_bg.currentData() == "darkref":
            self.invalidate()
        else:
            self.refresh()

    def _background(self) -> Tuple[Optional[np.ndarray], str]:
        """Per-pixel background of the current choice: pre-t0 offset, dark reference or none."""
        key = self.cmb_bg.currentData()
        if key == "dark":
            return self._dark()
        if key == "darkref":
            return self._darkref()
        return None, ""

    def _update_bg_label(self) -> None:
        pre, pre_text = self._dark()
        ref, ref_text = self._darkref()
        text = ref_text if self.cmb_bg.currentData() == "darkref" else pre_text
        cmp = an.dark_compare_text(pre, ref, self.panel.state.enabled_mask()) if ref is not None else ""
        if cmp:
            text += " | " + cmp
        self.lbl_dark.setText(text)
        self.lbl_dark.setToolTip(text)

    def _dark_2d(self) -> Optional[np.ndarray]:
        """The per-pixel background to subtract in the 2D methods (None = not subtracted)."""
        if self._spectral or self.cmb_bg.currentData() not in ("dark", "darkref"):
            return None
        return self._background()[0]

    def _frame_2d(self, idx: int) -> np.ndarray:
        frame = self.panel.view_counts[int(idx)].astype(np.float64)
        dark = self._dark_2d()
        return frame if dark is None else frame - dark

    @staticmethod
    def _constraint_text(res) -> str:
        parts = []
        if getattr(res, "amp_ratio", None) is not None:
            parts.append(f"A2/A1 fixed at {res.amp_ratio:.3g}")
        if getattr(res, "shared_sigma", False):
            parts.append("same σx, σy")
        return ("; " + ", ".join(parts)) if parts else ""

    def _fix_offset(self) -> Optional[float]:
        """Value the 2D Gaussian offset is held at (None = fitted)."""
        return 0.0 if (not self._spectral and self.chk_fix_offset.isChecked()) else None

    def _offset_text(self) -> str:
        return "Gaussian offset fixed at 0" if self._fix_offset() is not None else "Gaussian offset fitted"

    def _counts_label(self) -> str:
        if self._dark_2d() is None:
            return "Counts"
        return "Counts − dark reference" if self.cmb_bg.currentData() == "darkref" else "Counts − pre-t0 offset"

    def _background_for(self, frame: np.ndarray, mask: np.ndarray):
        if self.cmb_bg.currentData() in ("dark", "darkref"):
            dark, _ = self._background()
            if dark is not None:
                return dark
        return an.frame_background(frame, mask)

    # ------------------------------------------------------------ fit all
    def _on_fit_all(self) -> None:
        p = self.panel
        if not an.HAS_SCIPY:
            QtWidgets.QMessageBox.warning(self, "Fit All", "scipy is required for fitting.")
            return
        if not p.has_data:
            QtWidgets.QMessageBox.warning(self, "Fit All", "No data loaded.")
            return
        if self._worker is not None:
            return
        method = self.method()
        r0, r1 = p.range
        mask = p.state.enabled_mask()
        excluded = tuple(sorted(p.state.excluded))
        frames = np.array(p.view_counts[: r1 + 1], copy=True)
        if method.startswith("2d"):
            ref = p.ref_index()
            if ref is None:
                QtWidgets.QMessageBox.warning(self, "Fit All", "Set a reference frame in the Histogram tab first.")
                return
            if max(r0, ref) > r1:
                QtWidgets.QMessageBox.warning(self, "Fit All", "The time range has no frames after the reference.")
                return
            need = an.MIN_FIT_PIXELS_DOUBLE if method == DOUBLE else an.MIN_FIT_PIXELS_2D
            if int(mask.sum()) < need:
                QtWidgets.QMessageBox.warning(self, "Fit All", f"Not enough enabled pixels for fitting (need ≥ {need}).")
                return
            fn = {"2d_sequential": an.fit_all_2d_sequential, "2d_fixed_center": an.fit_all_2d_fixed_center,
                  DOUBLE: an.fit_all_2d_double_sequential}[method]
            dark = self._dark_2d()
            if dark is not None:
                frames = frames.astype(np.float64) - dark
            kw = dict(frames=frames, x=PIXEL_X_UM, y=PIXEL_Y_UM, mask=mask, ref_idx=ref, start=r0, end=r1,
                      fix_offset=self._fix_offset())
            if method == DOUBLE:
                kw.update(fix_centers=self.chk_fix_centers.isChecked(), fix_ratio=self.chk_fix_ratio.isChecked(),
                          ratio=self.spin_ratio.value() or None, share_sigma=self.chk_share_sigma.isChecked())
            n_frames = r1 - max(r0, ref) + 1
        else:
            dark = None
            if self.cmb_bg.currentData() in ("dark", "darkref"):
                dark, _ = self._background()
            # spectral widths start at time zero = the peak of the summed decay inside the range
            t0 = r0 + int(np.argmax(np.asarray(p.view_counts[r0 : r1 + 1])[:, mask].sum(axis=1)))
            if t0 >= r1:
                QtWidgets.QMessageBox.warning(self, "Fit All", "The decay peaks at the end of the time range: "
                                              "nothing to fit after it.")
                return
            fn = an.fit_all_column_widths
            kw = dict(frames=frames, excluded=excluded, mask=mask, start=t0, end=r1,
                      min_snr=float(self.spin_snr.value()), fixed_center=method == "widths_fixed_y0",
                      center_y=CENTER_Y_UM if method == "widths_center_row" else None,
                      free_offset=self.chk_free_offset.isChecked(), time_ns=np.array(p.view_time[: r1 + 1]),
                      dark=None if dark is None else dark.copy())
            n_frames = r1 - t0 + 1
        self._job_id += 1
        version = p.data_version
        w = FitAllWorker(self._job_id, fn, kw, self)
        w.progress.connect(self._on_progress)
        w.done.connect(lambda job, res, err: self._on_done(job, res, err, version))
        w.finished.connect(self._on_worker_finished)
        self._worker = w
        self.progress.setValue(0)
        what = f"{n_frames} frames × {N_COLS} columns" if not method.startswith("2d") else f"{n_frames} frames"
        self.lbl_state.setText(f"Fitting {what} ...")
        self._sync_buttons()
        w.start()

    def _on_cancel(self) -> None:
        self.cancel()

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def _on_done(self, job_id: int, res, err: str, version: int) -> None:
        if job_id != self._job_id:
            return
        if err:
            self.lbl_state.setText(f"Fit All failed: {err}")
            return
        if version != self.panel.data_version:
            self.lbl_state.setText("Data changed during fitting; results discarded. Click Fit All again.")
            return
        self._result = res
        self._result_version = version
        self.refresh()

    def _on_worker_finished(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.deleteLater()
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        running = self._worker is not None
        self.btn_fit.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _state_text(self) -> str:
        res = self._valid_result()
        if self._spectral:
            _, dark_text = self._background()
            bg = dark_text if self.cmb_bg.currentData() in ("dark", "darkref") else "background = frame 10th percentile"
            if res is None:
                return (f"Click Fit All to fit every column from time zero (the peak of the summed decay) to the "
                        f"end of the selected time range. {bg}.")
            n = res.counts()
            parts = [f"{n.get(k, 0)} {an.WIDTH_STATUS_TEXT[k]}" for k in
                     (an.WIDTH_OK, an.WIDTH_NOISY, an.WIDTH_REJECTED, an.WIDTH_FAILED, an.WIDTH_NO_PIXELS)
                     if n.get(k, 0)]
            text = (f"{'Cancelled' if res.cancelled else 'Done'}: {res.time_ns.size} bins × {N_COLS} columns "
                    f"from t0 = {res.time_ns[0]:.3f} ns (peak) — "
                    + ", ".join(parts) + f". Fit All min SNR {res.min_snr:g}; "
                    + ("dark subtracted per pixel" if res.dark else "frame background") + ".")
            if self.spin_snr.value() < res.min_snr:
                text += f" Min SNR is below the Fit All value ({res.min_snr:g}): click Fit All to fit more bins."
            if res.message:
                text += f" {res.message}."
            return text
        bg = ("no per-pixel offset subtracted" if self._dark_2d() is None else
              "dark reference subtracted per pixel" if self.cmb_bg.currentData() == "darkref" else
              "pre-t0 offset subtracted per pixel")
        bg += f", {self._offset_text()}"
        if res is None:
            return f"Set a reference frame in the Histogram tab, then click Fit All ({bg})."
        return (f"{'Cancelled' if res.cancelled else 'Done'}: {len(res.results)} fitted frame(s), "
                f"failures={res.failures}, reference bin {res.ref_idx}; {bg}. Hollow points: "
                + ("second Gaussian not resolved, low signal or a fit at a bound."
                   if res.method == DOUBLE and res.centers is None else "fit at a bound or low signal."))

    # ------------------------------------------------------------ display
    def refresh(self) -> None:
        try:
            self.double_opts.setVisible(not self._spectral and self.method() == DOUBLE)
            self._update_darkref_item()
            self._update_bg_label()  # also refreshes the auto pre-t0 window
            self.lbl_state.setText(self._state_text())
            if self._spectral:
                self._refresh_widths()
            else:
                self._refresh_trace()
            self.refresh_current()
        except Exception as exc:
            self.lbl_state.setText(f"Display error: {exc}")

    def _refresh_trace(self) -> None:
        res = self._valid_result()
        if res is None:
            self.trace.clear("Fit results", "No fit results yet: click Fit All.")
            return
        p = self.panel
        r0, r1 = p.range
        idxs = [i for i in sorted(res.results) if r0 <= i <= r1 and i >= res.ref_idx]
        if not idxs:
            self.trace.clear("Fit results", "No fitted points in the selected range.")
            return
        t = p.view_time[idxs]
        fits = [res.results[i] for i in idxs]
        if res.method == DOUBLE:
            s1 = np.array([f.sigma_eq(0) for f in fits])
            s2 = np.array([f.sigma_eq(1) for f in fits])
            sub = f"{self.cmb_method.currentText()}; ref t = {p.time_of(res.ref_idx):.3f} ns"
            labels = dict(y_b=s2**2, label_a="σx1·σy1 (Gaussian 1)", label_b="σx2·σy2 (Gaussian 2)")
            sub += self._constraint_text(res)
            if res.centers is not None:
                # the centers are fixed: the separation is constant, show each spot's integrated intensity
                flagged = np.array([bool(f.at_bound) or not f.quality_ok for f in fits])
                w = np.array([f.weights for f in fits])
                c = res.centers
                sub += f"; centers fixed at ({c[0]:.1f}, {c[1]:.1f}) and ({c[2]:.1f}, {c[3]:.1f}) µm"
                self.trace.set_series(t, s1**2, flagged, "σx·σy (µm²)", "σx·σy of each Gaussian vs time",
                                      y2=w[:, 0], ylabel2="2πAσxσy (counts·µm²)", subtitle=sub, y2_b=w[:, 1],
                                      label2_a="I1 (Gaussian 1)", label2_b="I2 (Gaussian 2)", **labels)
            else:
                flagged = np.array([bool(f.at_bound) or not f.resolved for f in fits])
                d = np.array([f.separation for f in fits])
                self.trace.set_series(t, s1**2, flagged, "σx·σy (µm²)", "σx·σy of each Gaussian vs time", y2=d,
                                      ylabel2="Separation d (µm)", subtitle=sub, **labels)
            self.trace.set_cursor(p.time_of(p.current))
            return
        flagged = np.array([bool(f.at_bound) or not f.quality_ok for f in fits])
        s = np.array([f.sigma_eq for f in fits])
        self.trace.set_series(t, s**2, flagged, "σ_eq² (µm²)", "σ_eq² vs time",
                              subtitle=f"{self.cmb_method.currentText()}; ref t = {p.time_of(res.ref_idx):.3f} ns")
        self.trace.set_cursor(p.time_of(p.current))

    def _quantity(self, sigma: np.ndarray) -> np.ndarray:
        key = self.cmb_quantity.currentData()
        if key == "sigma2":
            return sigma**2
        if key == "fwhm":
            return an.FWHM_FACTOR * sigma
        return sigma

    def _refresh_widths(self) -> None:
        p = self.panel
        res = self._valid_result()
        qlabel = self.cmb_quantity.currentText()
        qname = qlabel.split(" (")[0]
        cols = p.selected_columns()
        if res is None:
            self.width_trace.set_series([], qlabel, f"{qname} vs time", message="No fit results yet: click Fit All.")
            self.width_map.img.clear()
            self.width_map.set_title(f"Time-resolved transport: fitted {qname}", "click Fit All")
            self.width_map.set_columns(cols)
            return
        acc = res.accepted(float(self.spin_snr.value()))
        q = np.where(acc, self._quantity(res.sigma), np.nan)
        r0, r1 = p.range
        in_range = (np.arange(res.start, res.end + 1) >= r0) & (np.arange(res.start, res.end + 1) <= r1)
        series = []
        for k, c in enumerate(cols):
            ok = acc[:, c] & in_range
            series.append((f"{self._col_label(c)}: {int(ok.sum())} bins", res.time_ns[ok], q[ok, c], trace_color(k)))
        msg = "" if any(s[1].size for s in series) else "No accepted fits for the selected columns."
        self.width_trace.set_series(series, qlabel, f"{qname} vs time (selected columns)",
                                    "too-noisy bins and rejected fits are left out", message=msg)
        if in_range.any():
            rows = np.flatnonzero(in_range)
            sub_q = q[rows]
            finite = sub_q[np.isfinite(sub_q)]
            levels = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else None
            self.width_map.set_data(res.time_ns[rows], sub_q, levels=levels, label=qlabel)
        else:
            self.width_map.img.clear()
        self.width_map.set_columns(cols)
        self.width_map.set_title(f"Time-resolved transport: fitted {qname}",
                                 f"{self.method_label()}; blank = not fitted or rejected")
        self.set_cursor(p.time_of(p.current))

    def refresh_current(self) -> None:
        p = self.panel
        if not p.has_data or not self.isVisible():
            return
        idx = p.current
        frame = p.view_counts[idx].astype(np.float64) if self._spectral else self._frame_2d(idx)
        t = p.time_of(idx)
        mask = p.state.enabled_mask()
        ex = p.state.excluded
        res = self._valid_result()
        if self._spectral:
            self._show_profiles(frame, t, idx, mask, ex, res)
        else:
            self._show_2d(frame, t, idx, mask, ex, res, self.method())

    def _show_2d(self, frame, t, idx, mask, ex, res, method) -> None:
        self.side_stack.setCurrentIndex(1 if method == DOUBLE else 0)
        if method == DOUBLE:
            self._show_2d_double(frame, t, idx, mask, ex, res)
            return
        if method == "2d_fixed_center" and res is not None and res.ref_fit is not None:
            ref_idx = res.ref_idx
            ref_frame = self._frame_2d(ref_idx)
            ref_fit = res.ref_fit
            x0, y0 = float(ref_fit.popt[1]), float(ref_fit.popt[2])
            self.map_fit.set_data(ref_frame, ref_fit, ex, "Reference 2D Gaussian fit (fixed center)",
                                  f"ref t = {self.panel.time_of(ref_idx):.3f} ns, x0={x0:.1f} µm, y0={y0:.1f} µm")
            fit1d = res.results.get(idx)
            r = np.sqrt((PIXEL_X_UM[mask] - x0) ** 2 + (PIXEL_Y_UM[mask] - y0) ** 2)
            order = np.argsort(r)
            if fit1d is None:
                self.radial.clear("No method-2 fit at this time bin", "Radial 1D Gaussian fit")
                return
            A, sig, off = (float(v) for v in fit1d.popt)
            r_line = np.linspace(0.0, float(r.max()), 401)
            self.radial.set_data(r[order], frame[mask][order], r_line,
                                 A * np.exp(-(r_line**2) / (2.0 * max(sig, 1e-6) ** 2)) + off,
                                 f"Radial 1D Gaussian @ t = {t:.3f} ns", f"σ = {fit1d.sigma_eq:.2f} µm", self._counts_label())
            return
        fit = res.results.get(idx) if res is not None else None
        label = "Fit All result"
        if fit is None and int(mask.sum()) >= an.MIN_FIT_PIXELS_2D:
            fit = an.fit_gaussian_2d(PIXEL_X_UM[mask], PIXEL_Y_UM[mask], frame[mask], fix_offset=self._fix_offset())
            label = "free fit (no Fit All result at this bin)"
        if fit is None:
            self.map_fit.set_data(frame, None, ex, f"2D Gaussian fit @ t = {t:.3f} ns")
            self.radial.clear("Fit unavailable")
            return
        p = fit.popt
        q = "" if fit.quality_ok else f", low signal ({fit.snr:.1f} σ)"
        self.map_fit.set_data(frame, fit, ex, f"2D Gaussian fit @ t = {t:.3f} ns",
                              f"x0={p[1]:.1f} µm, y0={p[2]:.1f} µm, σ_eq={fit.sigma_eq:.1f} µm — {label}{q}")
        r = np.sqrt((PIXEL_X_UM[mask] - p[1]) ** 2 + (PIXEL_Y_UM[mask] - p[2]) ** 2)
        order = np.argsort(r)
        rs, zs = r[order], frame[mask][order]
        rmax = float(r.max()) if r.size else 1.0
        r_line = np.linspace(-rmax, rmax, 401)
        self.radial.set_data(np.concatenate([-rs[::-1], rs]), np.concatenate([zs[::-1], zs]), r_line,
                             p[0] * np.exp(-(r_line**2) / (2.0 * max(fit.sigma_eq, 1e-6) ** 2)) + p[5],
                             "Gaussian radial fit", f"σ = {fit.sigma_eq:.1f} µm", self._counts_label())

    def _show_2d_double(self, frame, t, idx, mask, ex, res) -> None:
        fit = res.results.get(idx) if res is not None else None
        centers = res.centers if res is not None else None
        ratio = res.amp_ratio if res is not None else (
            (self.spin_ratio.value() or None) if self.chk_fix_ratio.isChecked() else None)
        share = res.shared_sigma if res is not None else self.chk_share_sigma.isChecked()
        label = "Fit All result"
        if fit is None and int(mask.sum()) >= an.MIN_FIT_PIXELS_DOUBLE:
            fit = an.fit_gaussian_2d_double(PIXEL_X_UM[mask], PIXEL_Y_UM[mask], frame[mask],
                                            fix_offset=self._fix_offset(), centers=centers, amp_ratio=ratio,
                                            share_sigma=share)
            label = "fit of this bin (no Fit All result here)"
        title = f"Double 2D Gaussian fit @ t = {t:.3f} ns"
        if fit is None:
            self.map_fit.set_data_double(frame, None, ex, title)
            self.pixel_fit.set_data(frame, None, ex, self._counts_label(), "Radial profiles of the two Gaussians")
            return
        self.map_fit.set_data_double(frame, fit, ex, title, f"{double_fit_text(fit, centers is not None)} — {label}")
        self.pixel_fit.set_data(frame, fit, ex, self._counts_label(), "Radial profiles of the two Gaussians",
                                f"offset {fit.offset:.3g}; amplitude significance {fit.snr1:.1f} σ / {fit.snr2:.1f} σ")

    def _show_profiles(self, frame, t, idx, mask, ex, res) -> None:
        bg = self._background_for(frame, mask)
        min_snr = float(self.spin_snr.value())
        free_offset = self.chk_free_offset.isChecked()
        norm = self.chk_norm.isChecked()
        use_res = res is not None and res.row_of(idx) is not None
        acc = res.accepted(min_snr) if use_res else None
        profiles = []
        for k, c in enumerate(self.panel.selected_columns()):
            if use_res:
                r = res.row_of(idx)
                # min_snr=inf: only the background-subtracted profile, no fit
                y, z, _, _, _ = an.fit_column_width(frame, c, ex, bg, np.inf, None, free_offset)
                status = int(res.status[r, c])
                fit = res.fit_at(idx, c)
                ok = bool(acc[r, c])
            else:
                fc = CENTER_Y_UM if self.method() == "widths_center_row" else None
                y, z, fit, status, _ = an.fit_column_width(frame, c, ex, bg, min_snr, fc, free_offset)
                ok = status == an.WIDTH_OK
            if fit is not None and ok:
                err = f" ± {fit.sigma_err:.2g}" if np.isfinite(fit.sigma_err) else ""
                desc = f"σ = {fit.sigma:.1f}{err} µm"
            elif fit is not None:
                desc = f"rejected (σ = {fit.sigma:.1f} µm)"
            else:
                desc = an.WIDTH_STATUS_TEXT.get(status, "")
            zz = np.asarray(z, dtype=np.float64)
            if norm:
                if fit is not None and fit.amp > 0:
                    zz = (zz - fit.offset) / fit.amp
                    fit = an.Gauss1DResult(amp=1.0, center=fit.center, sigma=fit.sigma, offset=0.0)
                elif zz.size and np.nanmax(zz) > 0:
                    zz = zz / np.nanmax(zz)
            profiles.append((f"{self._col_label(c)}: {desc}", y, zz, fit, trace_color(k), ok))
        src = ("Fit All result" if use_res else "free fit (no Fit All result at this bin)")
        bg_text = "dark-subtracted" if isinstance(bg, np.ndarray) else f"background {bg:.3g} subtracted"
        if self.method() == "widths_center_row":
            bg_text += f"; y0 fixed at the center row ({CENTER_Y_UM:g} µm)"
        self._shown_profiles = (t, profiles)
        self.profiles.set_profiles(profiles, f"Spatial profiles @ t = {t:.3f} ns", f"{src}; {bg_text}",
                                   "Normalized" if norm else "Counts − background")

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def profile_columns(self):
        """(column labels, arrays) of the spatial profiles shown at the current time."""
        t, profiles = getattr(self, "_shown_profiles", (None, []))
        labels, cols = [], []
        for label, y, z, fit, _color, _ok in profiles:
            name = label.split(":")[0]
            labels += [f"{name} y (µm)", f"{name} data"]
            cols += [np.asarray(y, float), np.asarray(z, float)]
            if fit is not None and np.size(y):
                ys = np.linspace(min(np.min(y), fit.center) - 23.0, max(np.max(y), fit.center) + 23.0, 300)
                labels += [f"{name} fit y (µm)", f"{name} fit"]
                cols += [ys, fit.evaluate(ys)]
        return t, labels, cols

    def _on_copy_profiles(self) -> None:
        t, labels, cols = self.profile_columns()
        if not cols:
            self.panel.status.emit("Nothing to copy: select columns on the map first.")
            return
        QtWidgets.QApplication.clipboard().setText(columns_to_text(cols))
        self.panel.status.emit(f"Copied spatial profiles @ t = {t:.3f} ns, {len(cols)} columns: " + ", ".join(labels))

    # ------------------------------------------------------------ export
    def _on_export(self) -> None:
        res = self._valid_result()
        if res is None:
            QtWidgets.QMessageBox.information(self, "Export fits", "No fit results for the selected method.")
            return
        p = self.panel
        meta = {"device": "SPAD93", "method": res.method, "method_label": self.method_label(),
                "source": p.raw.source if p.raw else "", "rebin_bin_width_ps": float(np.median(np.diff(p.view_time)) * 1000)
                if p.view_time.size > 1 else None, "excluded_pixels": sorted(p.state.excluded)}
        meta.update(p.processing_metadata())
        if not self._spectral:
            meta["background"] = self._background()[1] if self._dark_2d() is not None else "none (Gaussian offset only)"
            if getattr(p, "dark_ref", None) is not None:
                meta["dark_reference"] = {k: v for k, v in p.dark_ref.summary().items() if k != "rate_per_bin"}
            meta["gaussian_offset"] = "fixed at 0" if self._fix_offset() is not None else "fitted"
        if self._spectral:
            cols, extra = self._export_widths(res)
        else:
            cols, extra = self._export_2d(res)
        meta.update(extra)
        base = os.path.splitext(os.path.basename(p.raw.source))[0] if (p.raw and p.raw.source) else f"spad93_trpl_{timestamp()}"
        default = os.path.join(save_dir("trpl"), f"{base}_fit_{res.method}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export fit results", default,
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            save_columns_csv(path, cols, meta)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export fits", f"Could not save:\n{exc}")
            return
        p.status.emit(f"Fit results exported: {path}")

    def _export_2d(self, res):
        p = self.panel
        idxs = sorted(res.results)
        t = p.view_time[idxs]
        fits = [res.results[i] for i in idxs]
        if res.method == DOUBLE:
            comp = [[f.component(k) for f in fits] for k in range(2)]
            cols = {"time_ns": t}
            for k in range(2):
                n = k + 1
                cols.update({f"A{n}": [c[0] for c in comp[k]], f"x{n}_um": [c[1] for c in comp[k]],
                             f"y{n}_um": [c[2] for c in comp[k]], f"sigma{n}_x_um": [c[3] for c in comp[k]],
                             f"sigma{n}_y_um": [c[4] for c in comp[k]],
                             f"sigma{n}_eq_um": [float(np.sqrt(c[3] * c[4])) for c in comp[k]]})
            cols.update({"offset": [f.offset for f in fits], "separation_um": [f.separation for f in fits],
                         "snr1": [f.snr1 for f in fits], "snr2": [f.snr2 for f in fits],
                         "resolved": [float(f.resolved) for f in fits], "at_bound": [float(bool(f.at_bound)) for f in fits],
                         "I1": [f.weights[0] for f in fits], "I2": [f.weights[1] for f in fits]})
            c = res.centers
            return cols, {"failures": res.failures, "reference_time_ns": p.time_of(res.ref_idx),
                          "model": "sum over k=1,2 of Ak exp(-(x-xk)^2/2 sigmak_x^2 - (y-yk)^2/2 sigmak_y^2)"
                                   " + offset; sigma_eq = sqrt(sigma_x sigma_y); I = 2 pi A sigma_x sigma_y",
                          "amplitude_ratio_A2_A1": "free" if res.amp_ratio is None else f"fixed at {res.amp_ratio:.4g}",
                          "sigma": "shared (sigma1_x = sigma2_x, sigma1_y = sigma2_y)" if res.shared_sigma
                          else "separate",
                          "centers": "free" if c is None else
                          f"fixed at the reference fit: ({c[0]:.3f}, {c[1]:.3f}) and ({c[2]:.3f}, {c[3]:.3f}) um"}
        if res.method == "2d_sequential":
            cols = {"time_ns": t, "x0_um": [f.popt[1] for f in fits], "y0_um": [f.popt[2] for f in fits],
                    "sigma_x_um": [f.popt[3] for f in fits], "sigma_y_um": [f.popt[4] for f in fits],
                    "sigma_eq_um": [f.sigma_eq for f in fits], "sigma_eq2_um2": [f.sigma_eq**2 for f in fits],
                    "amplitude": [f.popt[0] for f in fits], "offset": [f.popt[5] for f in fits]}
        else:
            cols = {"time_ns": t, "sigma_um": [f.sigma_eq for f in fits], "sigma2_um2": [f.sigma_eq**2 for f in fits],
                    "amplitude": [f.popt[0] for f in fits], "offset": [f.popt[2] for f in fits]}
        cols["amplitude_significance"] = [getattr(f, "snr", np.nan) for f in fits]
        cols["at_bound"] = [float(bool(f.at_bound)) for f in fits]
        return cols, {"failures": res.failures, "reference_time_ns": p.time_of(res.ref_idx)}

    def _export_widths(self, res):
        min_snr = float(self.spin_snr.value())
        acc = res.accepted(min_snr)
        cols = {"time_ns": res.time_ns}
        for c in range(N_COLS):
            cols[f"sigma_um_col{c}"] = np.where(acc[:, c], res.sigma[:, c], np.nan)
        for c in range(N_COLS):
            cols[f"sigma_err_um_col{c}"] = np.where(acc[:, c], res.sigma_err[:, c], np.nan)
        for c in range(N_COLS):
            cols[f"y0_um_col{c}"] = np.where(acc[:, c], res.center[:, c], np.nan)
        for c in range(N_COLS):
            cols[f"amp_col{c}"] = np.where(acc[:, c], res.amp[:, c], np.nan)
        for c in range(N_COLS):
            status = res.status[:, c].astype(np.float64)
            cols[f"status_col{c}"] = np.where((status == an.WIDTH_OK) & ~acc[:, c], an.WIDTH_REJECTED, status)
        extra = {
            "content": "fitted Gaussian width of each column's y-profile per time bin; NaN = not fitted or rejected",
            "status_codes": {str(k): v for k, v in an.WIDTH_STATUS_TEXT.items()},
            "min_snr": min_snr,
            "fit_all_min_snr": res.min_snr,
            "background": (("dark reference per pixel" if self.cmb_bg.currentData() == "darkref" else "dark per pixel")
                           if res.dark else "frame 10th percentile"),
            "free_offset_cols_ge_7px": res.free_offset,
            "column_x_um": [float(v) for v in COLUMN_X_UM],
            "column_lambda_nm": None if self._calib is None else [float(v) for v in self._calib.wavelength(COLUMN_X_UM)],
        }
        if res.dark:
            extra["dark_window"] = self._background()[1]
        if res.method in ("widths_fixed_y0", "widths_center_row"):
            extra["fixed_y0_um"] = [None if not np.isfinite(v) else float(v) for v in res.fixed_center]
        return cols, extra
