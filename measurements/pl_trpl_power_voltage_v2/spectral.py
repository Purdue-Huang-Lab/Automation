"""Spectral mode of the PL + TRPL v2 GUI: the SPAD's center line is a wavelength axis.

Ported from the SPAD93 / SPAD23 v2 GUIs (Counts tab spectral page, TRPL tab spectral
views, Fitting tab width fits):

* SPAD93: the center row (y = 92 um) crosses 17 columns, one wavelength each.
* SPAD23: the center column (x = 46 um) crosses 5 rows, one wavelength each.

The two geometry modules use the same function names (``row_profile``,
``column_profile``) with swapped meanings, so everything here goes through
``SpectralGeometry`` and never calls them by name.  The widgets (spectrum plot,
time-resolved spectrum image, ...) carry their detector's geometry, so they come
from that detector's ``gui_v2`` package and are rebuilt on a detector switch.

Display and fits only: the saved TRPL files stay raw (all pixels).
"""

import importlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from measurements.pl_trpl_power_voltage_v2 import detectors as spad_det
from spad23.gui_v2.plots import HistogramPlot, trace_color
from spad23.gui_v2.workers import FitAllWorker

MODE_2D = "2d"
MODE_SPECTRAL = "spectral"
VIEW_MODES = ((MODE_2D, "2D space"), (MODE_SPECTRAL, "Spectral"))

QUANTITIES = (("sigma", "σ (µm)"), ("sigma2", "σ² (µm²)"), ("fwhm", "FWHM (µm)"))
# width-fit methods: {c} = the cross axis (y for SPAD93 columns, x for SPAD23 rows)
WIDTH_METHODS = (("free", "{c}0 free"), ("fixed", "{c}0 fixed from the time-integrated profile"),
                 ("center", "{c}0 fixed at the {center}"))
CM2_PER_S = 10.0  # 1 um^2/ns = 10 cm^2/s


def line_fit(x: np.ndarray, y: np.ndarray):
    """Least-squares line y = y0 + slope * x -> (slope, slope_err, y0, y0_err, n) or None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = int(x.size)
    if n < 3:
        return None
    xm, ym = float(x.mean()), float(y.mean())
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return None
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    y0 = ym - slope * xm
    var = float(((y - y0 - slope * x) ** 2).sum()) / (n - 2)
    return slope, float(np.sqrt(var / sxx)), y0, float(np.sqrt(var * (1.0 / n + xm**2 / sxx))), n


# ======================================================================
# Per-detector adapter
# ======================================================================
class SpectralGeometry:
    """The spectral line layout of one detector, with name shims for its widgets and fits."""

    def __init__(self, key: str):
        det = spad_det.profile(key)
        self.key = key
        self.an, self.io = det.an, det.io
        self.num_pixels = int(det.num_pixels)
        self.center_pixel = int(det.center_pixel)
        pkg = key.lower()
        geo = importlib.import_module(f"{pkg}.{pkg}_geometry")
        self.plots = importlib.import_module(f"{pkg}.gui_v2.plots")
        self.maps = importlib.import_module(f"{pkg}.gui_v2.map_plots")
        if key == spad_det.SPAD93:
            self.n_lines = int(geo.N_COLS)
            self.pos_um = np.asarray(geo.COLUMN_X_UM, dtype=np.float64)
            self.word, self.Word, self.cross, self.along = "col", "Column", "y", "x"
            self.center_name, self.center_um = "center row", float(geo.CENTER_Y_UM)
            self._line_of = geo.column_of
            self._matrix = geo.center_row_matrix
            self._profile = lambda v, ex: geo.row_profile(v, geo.CENTER_Y_UM, ex)
            self._fit_linecut = self.an.fit_column_profile
            self._fit_width = self.an.fit_column_width
            self._fit_all = self.an.fit_all_column_widths
            self._center_kw = "center_y"
            self.LinecutPlot = self.plots.ColumnProfilePlot
            self._names = dict(highlight="set_highlight_columns", mark="set_columns", clicked="column_clicked",
                               select="set_selected_column", wl="column_wavelengths", per="nm_per_column",
                               row="row_of")
            self.free_offset_min = 7
        else:
            self.n_lines = int(geo.N_ROWS)
            self.pos_um = np.asarray(geo.ROW_Y_UM, dtype=np.float64)
            self.word, self.Word, self.cross, self.along = "row", "Row", "x", "y"
            self.center_name, self.center_um = "center column", float(geo.CENTER_X_UM)
            self._line_of = geo.row_of
            self._matrix = geo.center_column_matrix
            self._profile = lambda v, ex: geo.column_profile(v, geo.CENTER_X_UM, ex)
            self._fit_linecut = self.an.fit_row_profile
            self._fit_width = self.an.fit_row_width
            self._fit_all = self.an.fit_all_row_widths
            self._center_kw = "center_x"
            self.LinecutPlot = self.plots.RowProfilePlot
            self._names = dict(highlight="set_highlight_rows", mark="set_rows", clicked="row_clicked",
                               select="set_selected_row", wl="row_wavelengths", per="nm_per_row",
                               row="bin_row")
            self.free_offset_min = int(getattr(self.an, "FREE_OFFSET_MIN_PIXELS", 5))
        self.center_line = self.line_of(self.center_pixel)

    # ---- geometry
    def line_of(self, pixel: int) -> int:
        return int(self._line_of(int(pixel)))

    def spectrum_matrix(self, counts: np.ndarray, excluded: Sequence[int] = ()) -> np.ndarray:
        """Center-line profile of every time bin: [bins, pixels] -> [bins, lines]."""
        return self._matrix(np.asarray(counts, dtype=np.float64), tuple(excluded))

    def spectrum(self, values: np.ndarray, excluded: Sequence[int] = ()) -> np.ndarray:
        """Center-line profile of one frame: [pixels] -> [lines]."""
        return np.asarray(self._profile(np.asarray(values, dtype=np.float64), tuple(excluded)), dtype=np.float64)

    def locate_peak(self, profile: np.ndarray):
        prof = np.asarray(profile, dtype=np.float64)
        if np.isfinite(prof).sum() < 4 or not np.nansum(prof) > 0:
            return None
        return self.an.locate_peak(self.pos_um, prof)

    def fit_linecut(self, values, line: int, excluded, background, free_offset: bool, fix_center):
        """(position across the line, values, Gauss1DResult or None) of one line's linecut."""
        return self._fit_linecut(values, int(line), tuple(excluded), background, free_offset, fix_center)

    def fit_width(self, frame, line: int, excluded, background, min_snr: float, fix_center, free_offset: bool):
        """(pos, z - background, fit or None, status, peak SNR): the noise-gated width fit of one line."""
        return self._fit_width(frame, int(line), tuple(excluded), background, min_snr, fix_center, free_offset)

    def fit_all_widths(self, center: Optional[float] = None, **kw):
        kw[self._center_kw] = center
        return self._fit_all(**kw)

    def width_row(self, res, idx: int) -> Optional[int]:
        return getattr(res, self._names["row"])(int(idx))

    # ---- widget name shims
    def highlight(self, pixel_map, lines) -> None:
        getattr(pixel_map, self._names["highlight"])(sorted(int(c) for c in lines))

    def mark_image(self, image, lines) -> None:
        getattr(image, self._names["mark"])(sorted(int(c) for c in lines))

    def spectrum_clicked(self, plot):
        return getattr(plot, self._names["clicked"])

    def select_on_spectrum(self, plot, line: Optional[int]) -> None:
        getattr(plot, self._names["select"])(line)

    # ---- wavelength calibration
    def wavelengths(self, calib) -> Optional[np.ndarray]:
        return None if calib is None else np.asarray(getattr(calib, self._names["wl"])(), dtype=np.float64)

    def line_label(self, line: int, calib=None) -> str:
        lab = f"{self.word} {int(line)}"
        if calib is not None:
            lab += f" ({float(calib.wavelength(self.pos_um[int(line)])):.1f} nm)"
        return lab

    def calib_short(self, calib) -> str:
        if calib is None:
            return "λ-cal: none (axes in µm)"
        lam = self.wavelengths(calib)
        return f"λ-cal: {lam[0]:.1f}–{lam[-1]:.1f} nm ({getattr(calib, self._names['per']):+.3f} nm/{self.word})"

    def load_calibration(self, path: str):
        """A wavelength calibration JSON of the standalone SPAD GUI; refused if made for the other detector."""
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        dev = str(d.get("device", "") or "")
        if dev and dev.upper() != self.key.upper():
            raise ValueError(f"this calibration was made for {dev}; the detector is {self.key}")
        return self.an.WavelengthCalibration.from_dict(d)


@lru_cache(maxsize=None)
def geometry(key: str) -> SpectralGeometry:
    return SpectralGeometry(key)


def _replace(layout: QtWidgets.QLayout, old: QtWidgets.QWidget, new: QtWidgets.QWidget) -> None:
    layout.replaceWidget(old, new)
    old.hide()
    old.deleteLater()


# ======================================================================
# Live tab: center-line spectrum and the linecut of the active line
# ======================================================================
class SpectralLivePanel(QtWidgets.QWidget):
    """Spectrum of the live count frame (peak, FWHM) and the active line's linecut with its Gaussian."""

    line_clicked = QtCore.pyqtSignal(int)

    def __init__(self, geom: SpectralGeometry, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._lay = lay
        self.spec_plot = QtWidgets.QWidget()
        self.cut_plot = QtWidgets.QWidget()
        lay.addWidget(self.spec_plot, 3)
        lay.addWidget(self.cut_plot, 2)
        opts = QtWidgets.QHBoxLayout()
        self.chk_center = QtWidgets.QCheckBox()
        self.chk_free_offset = QtWidgets.QCheckBox()
        opts.addWidget(self.chk_center)
        opts.addWidget(self.chk_free_offset)
        opts.addStretch(1)
        lay.addLayout(opts)
        self._last = None
        self.chk_center.toggled.connect(lambda _v: self._redraw())
        self.chk_free_offset.toggled.connect(lambda _v: self._redraw())
        self.set_geometry(geom)

    def set_geometry(self, geom: SpectralGeometry) -> None:
        self.geom = geom
        spec, cut = geom.plots.SpectrumPlot(), geom.LinecutPlot()
        _replace(self._lay, self.spec_plot, spec)
        _replace(self._lay, self.cut_plot, cut)
        self.spec_plot, self.cut_plot = spec, cut
        geom.spectrum_clicked(spec).connect(self.line_clicked.emit)
        self.chk_center.setText(f"Fix {geom.cross}0 at the {geom.center_name} ({geom.cross} = {geom.center_um:g} µm)")
        self.chk_free_offset.setText(f"Free offset ({geom.word}s with ≥ {geom.free_offset_min} pixels)")
        self._last = None
        self.spec_plot.show_message("No count frame yet")

    def update_frame(self, values: np.ndarray, excluded: Sequence[int], calib, line: int, unit: str) -> None:
        self._last = (np.asarray(values, dtype=np.float64), tuple(sorted(excluded)), calib, int(line), unit)
        if self.isVisible():
            self._redraw()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._redraw()

    def _redraw(self) -> None:
        if self._last is None:
            return
        g = self.geom
        vals, ex, calib, line, unit = self._last
        self.spec_plot.set_calibration(calib)
        prof = g.spectrum(vals, ex)
        peak = g.locate_peak(prof)
        if peak is None:
            sub = "no peak fit"
        else:
            where = f"{float(calib.wavelength(peak.center)):.2f} nm" if calib else f"{g.along} = {peak.center:.1f} µm"
            sub = f"peak {where}, {g.plots.fwhm_text(peak.sigma, calib)}"
            if peak.flags:
                sub += f" ({', '.join(peak.flags)})"
        self.spec_plot.set_data(prof, peak, f"Intensity ({unit})", f"Spectrum ({g.center_name})", sub)
        g.select_on_spectrum(self.spec_plot, line)
        mask = np.ones(vals.size, dtype=bool)
        mask[[p for p in ex if p < vals.size]] = False
        bg = g.an.frame_background(vals, mask)
        fixed = self.chk_center.isChecked()
        pos, z, fit = g.fit_linecut(vals, line, ex, bg, self.chk_free_offset.isChecked(),
                                    g.center_um if fixed else None)
        where = (f"{float(calib.wavelength(g.pos_um[line])):.2f} nm" if calib
                 else f"{g.along} = {g.pos_um[line]:.1f} µm")
        if fit is None:
            sub = "too few pixels" if np.size(pos) < 3 else "fit failed"
        else:
            sub = (f"{g.cross}0 = {fit.center:.1f} µm{' (fixed)' if fixed else ''}, "
                   f"σ = {fit.sigma:.1f} µm, FWHM = {fit.fwhm:.1f} µm")
            if fit.flags:
                sub += f" ({', '.join(fit.flags)})"
        self.cut_plot.set_data(pos, z, fit, f"{g.Word} {line} {g.cross}-linecut ({where})", sub, f"Intensity ({unit})")


# ======================================================================
# Histogram tab: time-resolved spectrum, decays, spectrum at t_ref and the width fit sigma(t)
# ======================================================================
@dataclass
class SpectralContext:
    """What the spectral Histogram page shows (built by the widget from the latest histogram)."""

    key: tuple  # everything the width fit depends on except the page's own options
    snapshot: object
    time_ns: np.ndarray  # raw bins
    counts: np.ndarray  # view counts (time-zero aligned if on), raw bins [bins, pixels]
    excluded: Tuple[int, ...]
    start_ns: float
    end_ns: float
    ref_ns: float
    dark: Optional[np.ndarray]  # dark reference per raw bin and pixel (scaled), or None
    dark_text: str
    fit_bg: Optional[np.ndarray]  # width-fit background per raw bin and pixel; None = frame 10th percentile
    bg_text: str
    log_y: bool = False
    meta: Dict[str, object] = field(default_factory=dict)  # header rows of the exports
    save_dir: str = ""
    base_name: str = "spad_trpl"


@dataclass
class _View:
    """The context rebinned and reduced to the center line (cached per context / options)."""

    t: np.ndarray
    counts: np.ndarray
    spec: np.ndarray
    s: int
    e: int
    ref: int
    k: int
    dark_sub: bool


class SpectralHistPage(QtWidgets.QWidget):
    """Right column of the Histogram tab in spectral mode; ``image_panel`` goes under the count map."""

    status = QtCore.pyqtSignal(str)
    time_clicked = QtCore.pyqtSignal(float)
    line_clicked = QtCore.pyqtSignal(int)
    fit_state_changed = QtCore.pyqtSignal()
    dark_sub_changed = QtCore.pyqtSignal()

    def __init__(self, geom: SpectralGeometry, fit_allowed: Callable[[], bool], parent=None):
        super().__init__(parent)
        self._fit_allowed = fit_allowed
        self.calib = None
        self.lines: List[int] = []
        self.active: Optional[int] = None
        self.ctx: Optional[SpectralContext] = None
        self._view: Optional[_View] = None
        self._view_key = None
        self._range_key = None
        self._width_range_key = None
        self._fit_worker: Optional[FitAllWorker] = None
        self._fit_job = 0
        self._fit_result = None  # (key, result)
        self._d_lines: list = []
        self._shown_profiles: Tuple[Optional[float], list] = (None, [])

        # left: the time-resolved spectrum image
        self.image_panel = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(self.image_panel)
        il.setContentsMargins(0, 0, 0, 0)
        self._img_lay = il
        self.tspec = QtWidgets.QWidget()
        il.addWidget(self.tspec, 1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        self.spin_rebin = QtWidgets.QSpinBox()
        self.spin_rebin.setRange(1, 10000)
        self.spin_rebin.setPrefix("×")
        self.spin_rebin.setFixedWidth(70)
        self.spin_rebin.setToolTip("Sum this many time bins (spectral views and the width fit only).")
        self.chk_log_img = QtWidgets.QCheckBox("Log image")
        self.chk_dark_sub = QtWidgets.QCheckBox("Subtract dark reference")
        self.chk_dark_sub.setChecked(True)
        self.btn_export = QtWidgets.QPushButton("Export spectral…")
        self.btn_origin = QtWidgets.QPushButton("Map for Origin…")
        self.btn_copy_decays = QtWidgets.QPushButton("Copy decays")
        self.btn_export.setToolTip("CSV of the time-resolved spectrum in the histogram start/end range, with a\n"
                                   "header (detector, λ-cal, dark subtraction, rebin): rows = time, columns = lines.")
        self.btn_origin.setToolTip(
            "Numbers only, for Origin: first row = wavelength (nm; position in µm without a λ-cal),\n"
            "first column = time (ns), cells = center-line counts. The top-left cell is empty.")
        self.btn_copy_decays.setToolTip("Copy time (ns) and the decays of the selected lines (tab-separated).")
        row.addWidget(QtWidgets.QLabel("Rebin"))
        row.addWidget(self.spin_rebin)
        row.addWidget(self.chk_log_img)
        row.addWidget(self.chk_dark_sub)
        row.addStretch(1)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_origin)
        row.addWidget(self.btn_copy_decays)
        root.addLayout(row)
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#6b7280")
        self.lbl_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        root.addWidget(self.lbl_info)

        self.sub_tabs = QtWidgets.QTabWidget()
        root.addWidget(self.sub_tabs, 1)
        # ---- decays & spectrum
        t1 = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(t1)
        v1.setContentsMargins(0, 4, 0, 0)
        self._t1_lay = v1
        self.decays = HistogramPlot()
        self.decays.cursor.hide()  # the reference line is the time marker here
        self.spec_at_t = QtWidgets.QWidget()
        v1.addWidget(self.decays, 1)
        v1.addWidget(self.spec_at_t, 1)
        self.sub_tabs.addTab(t1, "Decays && spectrum")
        # ---- width fit sigma(t)
        t2 = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(t2)
        v2.setContentsMargins(0, 4, 0, 0)
        v2.setSpacing(4)
        r1 = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit σ(t)")
        self.btn_fit.setFixedWidth(90)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setFixedWidth(70)
        self.prg = QtWidgets.QProgressBar()
        self.prg.setFixedWidth(110)
        self.prg.setValue(0)
        self.btn_copy_widths = QtWidgets.QPushButton("Copy widths")
        self.btn_export_widths = QtWidgets.QPushButton("Export widths…")
        self.lbl_fit = QtWidgets.QLabel("")
        self.lbl_fit.setStyleSheet("color:#6b7280")
        self.lbl_fit.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        for w in (self.btn_fit, self.btn_cancel, self.prg, self.btn_copy_widths, self.btn_export_widths):
            r1.addWidget(w)
        r1.addWidget(self.lbl_fit, 1)
        v2.addLayout(r1)
        r2 = QtWidgets.QHBoxLayout()
        self.cmb_method = QtWidgets.QComboBox()
        self.chk_free_offset = QtWidgets.QCheckBox("Free offset")
        self.spin_snr = QtWidgets.QDoubleSpinBox()
        self.spin_snr.setRange(0.0, 100.0)
        self.spin_snr.setDecimals(1)
        self.spin_snr.setSingleStep(0.5)
        self.spin_snr.setFixedWidth(60)
        self.spin_snr.setToolTip(
            "A bin is fitted only if the brightest pixel of the line is this many sqrt(counts) above the background,\n"
            "and a fit is kept only if its amplitude is this significant. Raising it after the fit hides more bins.")
        self.cmb_quantity = QtWidgets.QComboBox()
        for k, lab in QUANTITIES:
            self.cmb_quantity.addItem(lab, k)
        self.chk_norm = QtWidgets.QCheckBox("Normalize profiles")
        r2.addWidget(QtWidgets.QLabel("Method"))
        r2.addWidget(self.cmb_method, 1)
        r2.addWidget(self.chk_free_offset)
        r2.addWidget(QtWidgets.QLabel("Min SNR"))
        r2.addWidget(self.spin_snr)
        r2.addWidget(self.cmb_quantity)
        r2.addWidget(self.chk_norm)
        v2.addLayout(r2)
        r3 = QtWidgets.QHBoxLayout()
        self.spin_d0 = QtWidgets.QDoubleSpinBox()
        self.spin_d1 = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_d0, self.spin_d1):
            sp.setRange(0.0, 1e6)
            sp.setDecimals(3)
            sp.setSuffix(" ns")
            sp.setKeyboardTracking(False)
            sp.setFixedWidth(100)
        self.spin_d1.setSpecialValueText("end")
        self.spin_d0.setToolTip("Line fit of σ² for D from this time after time zero (the peak) ...")
        self.spin_d1.setToolTip("... to this time after time zero (0 = to the end of the fit).")
        self.lbl_d = QtWidgets.QLabel("")
        self.lbl_d.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        r3.addWidget(QtWidgets.QLabel("D from"))
        r3.addWidget(self.spin_d0)
        r3.addWidget(QtWidgets.QLabel("to"))
        r3.addWidget(self.spin_d1)
        r3.addWidget(QtWidgets.QLabel("after t0 (peak):"))
        r3.addWidget(self.lbl_d, 1)
        v2.addLayout(r3)
        self.width_trace = QtWidgets.QWidget()
        bottom = QtWidgets.QHBoxLayout()
        self._t2_lay, self._t2_bottom = v2, bottom
        self.profiles = QtWidgets.QWidget()
        self.width_map = QtWidgets.QWidget()
        bottom.addWidget(self.profiles, 1)
        bottom.addWidget(self.width_map, 1)
        v2.addWidget(self.width_trace, 1)
        v2.addLayout(bottom, 1)
        self.sub_tabs.addTab(t2, "Width fit σ(t)")
        self.btn_fit.setToolTip(
            "1D Gaussian fit of every line's profile across the line, in every bin from time zero (the peak of the\n"
            "summed decay inside the histogram start/end range) to the end (rebinned), with the background of the\n"
            "'background' combo above (none = each frame's 10th percentile).\n"
            "D comes from σ² = σ0² + 2 D (t − t0) over the D window.\n"
            "Only for a finished histogram (Take Histogram): disabled while a scan or a TRPL acquisition runs.")

        # signals
        self.spin_rebin.valueChanged.connect(lambda _v: self.redraw())
        self.chk_log_img.toggled.connect(lambda _v: self.redraw())
        self.chk_dark_sub.toggled.connect(lambda _v: (self.redraw(), self.dark_sub_changed.emit()))
        self.btn_export.clicked.connect(self._on_export)
        self.btn_origin.clicked.connect(self._on_origin)
        self.btn_copy_decays.clicked.connect(self._on_copy_decays)
        self.decays.time_clicked.connect(self.time_clicked.emit)
        self.btn_fit.clicked.connect(self.start_fit)
        self.btn_cancel.clicked.connect(self.cancel_fit)
        self.btn_copy_widths.clicked.connect(self._on_copy_widths)
        self.btn_export_widths.clicked.connect(self._on_export_widths)
        for w in (self.cmb_method, self.cmb_quantity):
            w.currentIndexChanged.connect(lambda _i: self._redraw_width())
        for w in (self.chk_free_offset, self.chk_norm):
            w.toggled.connect(lambda _v: self._redraw_width())
        for sp in (self.spin_snr, self.spin_d0, self.spin_d1):
            sp.valueChanged.connect(lambda _v: self._redraw_width())
        self.sub_tabs.currentChanged.connect(lambda _i: self.redraw())
        self.set_geometry(geom)
        self.spin_snr.setValue(float(getattr(geom.an, "MIN_SNR", 3.0)))
        self.cmb_quantity.setCurrentIndex(self.cmb_quantity.findData("sigma2"))

    # ------------------------------------------------------------ setup
    def set_geometry(self, geom: SpectralGeometry) -> None:
        """Rebuild the geometry-bound widgets for ``geom`` (detector switch)."""
        self.cancel_fit()
        self.geom = geom
        tspec, spec, trace = geom.maps.TimeSpectrumImage(), geom.plots.SpectrumPlot(), geom.plots.MultiTracePlot()
        prof, wmap = geom.plots.MultiProfilePlot(), geom.maps.TimeSpectrumImage()
        _replace(self._img_lay, self.tspec, tspec)
        _replace(self._t1_lay, self.spec_at_t, spec)
        _replace(self._t2_lay, self.width_trace, trace)
        _replace(self._t2_bottom, self.profiles, prof)
        _replace(self._t2_bottom, self.width_map, wmap)
        self.tspec, self.spec_at_t, self.width_trace, self.profiles, self.width_map = tspec, spec, trace, prof, wmap
        prof.plot.setLabel("bottom", f"{geom.cross} (µm)")
        tspec.time_clicked.connect(self.time_clicked.emit)
        wmap.time_clicked.connect(self.time_clicked.emit)
        trace.time_clicked.connect(self.time_clicked.emit)
        geom.spectrum_clicked(spec).connect(self.line_clicked.emit)
        self._d_lines = []
        keep = self.cmb_method.currentData()
        self.cmb_method.blockSignals(True)
        self.cmb_method.clear()
        for k, lab in WIDTH_METHODS:
            self.cmb_method.addItem(lab.format(c=geom.cross, center=f"{geom.center_name} ({geom.center_um:g} µm)"), k)
        self.cmb_method.setCurrentIndex(max(0, self.cmb_method.findData(keep)))
        self.cmb_method.blockSignals(False)
        self.chk_free_offset.setToolTip(
            f"Also fit the Gaussian offset ({geom.word}s with ≥ {geom.free_offset_min} pixels); otherwise it is 0 "
            "after the background subtraction.")
        self.chk_dark_sub.setToolTip(
            f"Subtract each pixel's dark counts (the dark reference, scaled to this histogram's acquisition time\n"
            f"and bin size) from the time-resolved spectrum, the {geom.word} decays, the spectrum at t_ref, the\n"
            "exports and the Lifetime tab. Pixels with more dark counts then no longer look brighter.\n"
            "Saved TRPL files stay raw.")
        self._fit_result = None
        self.ctx = None  # the histogram belonged to the other detector
        self._view = self._view_key = self._range_key = self._width_range_key = None
        self.set_calibration(self.calib)

    def set_calibration(self, calib) -> None:
        self.calib = calib
        for w in (self.tspec, self.spec_at_t, self.width_map):
            w.set_calibration(calib)
        self.redraw()

    def set_lines(self, lines, active: Optional[int]) -> None:
        self.lines = sorted(int(c) for c in lines)
        self.active = active
        self.redraw()

    def set_context(self, ctx: Optional[SpectralContext]) -> None:
        self.ctx = ctx
        self.redraw()

    @property
    def dark_subtracted(self) -> bool:
        return self._view is not None and self._view.dark_sub

    def dark_sub_requested(self) -> bool:
        return self.chk_dark_sub.isChecked()

    # ------------------------------------------------------------ data
    def _get_view(self) -> Optional[_View]:
        ctx = self.ctx
        if ctx is None or ctx.counts.shape[0] == 0:
            self._view = None
            return None
        k = int(self.spin_rebin.value())
        dark_sub = self.chk_dark_sub.isChecked() and ctx.dark is not None
        key = (id(ctx), k, dark_sub)
        if self._view is not None and self._view_key == key:
            return self._view
        counts, t = self.geom.an.rebin_time(ctx.counts, ctx.time_ns, k)
        counts = np.asarray(counts, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        spec = self.geom.spectrum_matrix(counts, ctx.excluded)
        if dark_sub:
            spec = spec - self.geom.spectrum_matrix(np.asarray(ctx.dark)[None, :] * k, ctx.excluded)[0]
        n = t.size
        s = int(np.clip(np.searchsorted(t, ctx.start_ns - 1e-9, side="left"), 0, n - 1))
        e = int(np.clip(np.searchsorted(t, ctx.end_ns + 1e-9, side="right") - 1, s, n - 1))
        ref = int(np.clip(int(np.argmin(np.abs(t - ctx.ref_ns))), s, e))
        self._view = _View(t, counts, spec, s, e, ref, k, dark_sub)
        self._view_key = key
        return self._view

    def _counts_label(self, v: _View) -> str:
        return "Counts − dark" if v.dark_sub else "Counts"

    # ------------------------------------------------------------ drawing
    def redraw(self) -> None:
        if not self.isVisible():
            return
        v = self._get_view()
        g = self.geom
        if v is None:
            self.tspec.img.clear()
            self.tspec.set_title(f"Time-resolved spectrum ({g.center_name})", "No histogram yet")
            self.decays.set_traces([])
            self.decays.show_message("No histogram yet. Click 'Take Histogram'.")
            self.spec_at_t.set_data(np.full(g.n_lines, np.nan), None, "Counts", "Spectrum at t_ref")
            self.lbl_info.setText("")
            self._redraw_width()
            return
        ctx = self.ctx
        sl = slice(v.s, v.e + 1)
        t = v.t[sl]
        log_img = self.chk_log_img.isChecked()
        label = None
        if v.dark_sub:
            label = "log10(1 + counts − dark)" if log_img else "Counts − dark"
        self.tspec.set_data(t, v.spec[sl], log=log_img, label=label)
        g.mark_image(self.tspec, self.lines)
        self.tspec.set_cursor(float(v.t[v.ref]))
        self.tspec.set_title(f"Time-resolved spectrum ({g.center_name})", "dark subtracted" if v.dark_sub else "")
        rkey = (id(ctx.snapshot), v.s, v.e, v.k)
        if rkey != self._range_key:
            self._range_key = rkey
            self.tspec.set_time_range(float(t[0]), float(t[-1]))
        traces = [(g.line_label(c, self.calib), t, v.spec[sl, c], trace_color(k)) for k, c in enumerate(self.lines)]
        self.decays.set_log(ctx.log_y)
        self.decays.set_traces(traces)
        self.decays.set_region(float(t[0]), float(t[-1]))
        self.decays.set_reference(float(v.t[v.ref]))
        self.decays.set_title(f"Center-line decays of the selected {g.word}s" + (" (dark subtracted)" if v.dark_sub else ""))
        self.decays.plot.setLabel("left", self._counts_label(v))
        if not self.lines:
            self.decays.show_message(f"Left-click pixels on the map to select {g.word}s")
        prof = v.spec[v.ref]
        peak = g.locate_peak(prof)
        if peak is None:
            sub = "no peak fit"
        else:
            where = (f"{float(self.calib.wavelength(peak.center)):.2f} nm" if self.calib
                     else f"{g.along} = {peak.center:.1f} µm")
            sub = f"peak {where}, {g.plots.fwhm_text(peak.sigma, self.calib)}"
        self.spec_at_t.set_data(prof, peak, self._counts_label(v), f"Spectrum at t_ref = {v.t[v.ref]:.3f} ns", sub)
        g.select_on_spectrum(self.spec_at_t, self.active)
        dark = ctx.dark_text if v.dark_sub else ("dark not subtracted" if ctx.dark is not None else ctx.dark_text)
        self.lbl_info.setText(f"rebin ×{v.k} ({(v.t[1] - v.t[0]) * 1000.0 if v.t.size > 1 else 0:g} ps bins); {dark}")
        self.lbl_info.setToolTip(self.lbl_info.text())
        self._redraw_width()

    # ------------------------------------------------------------ width fit
    def _fit_key(self) -> Optional[tuple]:
        ctx = self.ctx
        if ctx is None:
            return None
        return (ctx.key, int(self.spin_rebin.value()), self.cmb_method.currentData(), self.chk_free_offset.isChecked())

    @property
    def busy(self) -> bool:
        return self._fit_worker is not None

    def start_fit(self) -> None:
        if self.busy or not self._fit_allowed():
            return
        v = self._get_view()
        if v is None:
            return
        ctx, g = self.ctx, self.geom
        if v.e <= v.s:
            self.lbl_fit.setText("The histogram start/end range holds less than two bins.")
            return
        mask = np.ones(g.num_pixels, dtype=bool)
        mask[[p for p in ctx.excluded if p < mask.size]] = False
        # time zero = the peak of the summed decay (enabled pixels) inside the start/end range
        t0_idx = v.s + int(np.argmax(v.counts[v.s : v.e + 1][:, mask].sum(axis=1)))
        if t0_idx >= v.e:
            self.lbl_fit.setText("The decay peaks at the end of the histogram range: nothing to fit after it.")
            return
        method = self.cmb_method.currentData()
        dark = None if ctx.fit_bg is None else np.asarray(ctx.fit_bg, dtype=np.float64) * v.k
        kw = dict(frames=v.counts[: v.e + 1], excluded=tuple(ctx.excluded), mask=mask, start=t0_idx, end=v.e,
                  min_snr=float(self.spin_snr.value()), fixed_center=method == "fixed",
                  free_offset=self.chk_free_offset.isChecked(), time_ns=v.t, dark=dark,
                  center=g.center_um if method == "center" else None)
        self._fit_job += 1
        key = self._fit_key()
        w = FitAllWorker(self._fit_job, g.fit_all_widths, kw, self)
        w.progress.connect(lambda d, n: (self.prg.setMaximum(max(n, 1)), self.prg.setValue(d)))
        w.done.connect(lambda job, res, err, k=key, bg=ctx.bg_text: self._on_fit_done(job, res, err, k, bg))
        w.finished.connect(self._on_fit_finished)
        self._fit_worker = w
        self.prg.setValue(0)
        self.lbl_fit.setText(f"Fitting {g.n_lines} {g.word}s × {v.e - t0_idx + 1} bins from t0 = {v.t[t0_idx]:.3f} ns "
                             "(peak) ...")
        self.fit_state_changed.emit()
        w.start()

    def cancel_fit(self) -> None:
        if self._fit_worker is not None:
            self._fit_worker.cancel()

    def wait_fit(self, timeout_s: float) -> None:
        w = self._fit_worker
        if w is not None:
            w.cancel()
            w.wait(int(timeout_s * 1000))

    def _on_fit_done(self, job: int, res, err: str, key, bg_text: str) -> None:
        if job != self._fit_job:
            return
        if err or res is None:
            self._fit_result = None
            self.lbl_fit.setText(f"Width fit failed: {err or 'no result'}")
            return
        self._fit_result = (key, res, bg_text)
        self._width_range_key = None
        counts = res.counts()
        st = self.geom.an.WIDTH_STATUS_TEXT
        text = ", ".join(f"{n} {st.get(s, s)}" for s, n in sorted(counts.items()))
        t0 = f"t0 = {res.time_ns[0]:.3f} ns (peak) to {res.time_ns[-1]:.3f} ns: " if res.time_ns.size else ""
        self.lbl_fit.setText(("cancelled: " if res.cancelled else "") + t0 + text
                             + (f"; {res.message}" if res.message else ""))
        self.lbl_fit.setToolTip(self.lbl_fit.text())
        self._redraw_width()

    def _on_fit_finished(self) -> None:
        w, self._fit_worker = self._fit_worker, None
        if w is not None:
            w.deleteLater()
        self.fit_state_changed.emit()

    def _valid_result(self):
        r = self._fit_result
        if r is None or r[0] != self._fit_key():
            return None
        return r[1]

    def _quantity(self, sigma: np.ndarray) -> np.ndarray:
        key = self.cmb_quantity.currentData()
        if key == "sigma2":
            return sigma**2
        if key == "fwhm":
            return self.geom.an.FWHM_FACTOR * sigma
        return sigma

    def d_results(self, res):
        """Per selected line: (line, D, D_err, sigma0^2, n, t_window, s2_window_fit) or None per line.

        Times count from t0 = the first fitted bin (the peak of the decay).
        """
        t0 = float(res.time_ns[0])
        acc = res.accepted(float(self.spin_snr.value()))
        tr = res.time_ns - t0
        d0, d1 = float(self.spin_d0.value()), float(self.spin_d1.value())
        win = (tr >= d0 - 1e-9) & ((tr <= d1 + 1e-9) if d1 > 0 else True)
        out = []
        for c in self.lines:
            use = acc[:, c] & win & np.isfinite(res.sigma[:, c])
            fit = line_fit(tr[use], res.sigma[use, c] ** 2)
            if fit is None:
                out.append((c, None))
                continue
            slope, slope_err, s0, _s0_err, n = fit
            xs = np.array([tr[use].min(), tr[use].max()])
            out.append((c, (slope / 2.0 * CM2_PER_S, slope_err / 2.0 * CM2_PER_S, s0, n, xs + t0, s0 + slope * xs)))
        return out

    def _redraw_width(self) -> None:
        if not self.isVisible() or self.sub_tabs.currentIndex() != 1:
            return
        g = self.geom
        for item in self._d_lines:
            self.width_trace.plot.removeItem(item)
        self._d_lines = []
        qlabel = self.cmb_quantity.currentText()
        qname = qlabel.split(" (")[0]
        res = self._valid_result()
        v = self._get_view()
        if res is None:
            busy = self.busy
            stale = self._fit_result is not None
            msg = ("fitting ..." if busy else "settings or histogram changed: click Fit σ(t) again" if stale
                   else "click Fit σ(t) after the histogram is collected")
            self.width_trace.set_series([], qlabel, f"{qname} vs time", message=msg)
            self.width_map.img.clear()
            self.width_map.set_title(f"Fitted {qname}", msg)
            g.mark_image(self.width_map, self.lines)
            self.profiles.set_profiles([], "Profiles at t_ref", "", "Counts − background")
            self.lbl_d.setText("")
            return
        acc = res.accepted(float(self.spin_snr.value()))
        q = np.where(acc, self._quantity(res.sigma), np.nan)
        series = []
        for k, c in enumerate(self.lines):
            ok = acc[:, c]
            series.append((f"{g.line_label(c, self.calib)}: {int(ok.sum())} bins", res.time_ns[ok], q[ok, c],
                           trace_color(k)))
        msg = "" if any(s[1].size for s in series) else f"No accepted fits for the selected {g.word}s."
        self.width_trace.set_series(series, qlabel, f"{qname} vs time (selected {g.word}s)",
                                    "too-noisy bins and rejected fits are left out", message=msg)
        t_ref = float(v.t[v.ref]) if v is not None else float(res.time_ns[0])
        self.width_trace.set_cursor(t_ref)
        d_parts = []
        for k, (c, d) in enumerate(self.d_results(res)):
            if d is None:
                d_parts.append(f"{g.line_label(c)}: < 3 points")
                continue
            dval, derr, s0, n, xs, ys = d
            d_parts.append(f"{g.line_label(c)}: D = {dval:.4g} ± {derr:.2g} cm²/s ({n} pts)")
            if self.cmb_quantity.currentData() == "sigma2":
                line = self.width_trace.plot.plot(xs, ys, pen=pg.mkPen(trace_color(k), width=1.6,
                                                                       style=QtCore.Qt.DashLine))
                self._d_lines.append(line)
        self.lbl_d.setText("; ".join(d_parts) if d_parts else f"select {g.word}s on the map")
        self.lbl_d.setToolTip(self.lbl_d.text() + f"\nσ² = σ0² + 2 D (t − t0), t0 = {res.time_ns[0]:.3f} ns (peak); "
                              "dashed lines (σ² view)")
        finite = q[np.isfinite(q)]
        levels = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else None
        self.width_map.set_data(res.time_ns, q, levels=levels, label=qlabel)
        g.mark_image(self.width_map, self.lines)
        self.width_map.set_cursor(t_ref)
        wkey = (id(res),)
        if wkey != self._width_range_key and res.time_ns.size:
            self._width_range_key = wkey
            self.width_map.set_time_range(float(res.time_ns[0]), float(res.time_ns[-1]))
        bg_text = self._fit_result[2] if self._fit_result else ""
        self.width_map.set_title(f"Fitted {qname}", f"{self.cmb_method.currentText()}; blank = not fitted or rejected")
        if v is not None:
            self._show_profiles(v, res, bg_text)

    def _show_profiles(self, v: _View, res, bg_text: str) -> None:
        g, ctx = self.geom, self.ctx
        idx = v.ref
        frame = v.counts[idx]
        mask = np.ones(g.num_pixels, dtype=bool)
        mask[[p for p in ctx.excluded if p < mask.size]] = False
        bg = (np.asarray(ctx.fit_bg, dtype=np.float64) * v.k if ctx.fit_bg is not None
              else g.an.frame_background(frame, mask))
        min_snr = float(self.spin_snr.value())
        free_offset = self.chk_free_offset.isChecked()
        norm = self.chk_norm.isChecked()
        r = g.width_row(res, idx)
        acc = res.accepted(min_snr)
        profiles = []
        for k, c in enumerate(self.lines):
            if r is not None:
                pos, z, _, _, _ = g.fit_width(frame, c, ctx.excluded, bg, np.inf, None, free_offset)
                status = int(res.status[r, c])
                fit = res.fit_at(idx, c)
                ok = bool(acc[r, c])
            else:
                fc = g.center_um if self.cmb_method.currentData() == "center" else None
                pos, z, fit, status, _ = g.fit_width(frame, c, ctx.excluded, bg, min_snr, fc, free_offset)
                ok = status == g.an.WIDTH_OK
            if fit is not None and ok:
                err = f" ± {fit.sigma_err:.2g}" if np.isfinite(fit.sigma_err) else ""
                desc = f"σ = {fit.sigma:.1f}{err} µm"
            elif fit is not None:
                desc = f"rejected (σ = {fit.sigma:.1f} µm)"
            else:
                desc = g.an.WIDTH_STATUS_TEXT.get(status, "")
            zz = np.asarray(z, dtype=np.float64)
            if norm:
                if fit is not None and fit.amp > 0:
                    zz = (zz - fit.offset) / fit.amp
                    fit = g.an.Gauss1DResult(amp=1.0, center=fit.center, sigma=fit.sigma, offset=0.0)
                elif zz.size and np.nanmax(zz) > 0:
                    zz = zz / np.nanmax(zz)
            profiles.append((f"{g.line_label(c)}: {desc}", pos, zz, fit, trace_color(k), ok))
        t = float(v.t[idx])
        self._shown_profiles = (t, profiles)
        src = "width-fit result" if r is not None else "free fit (t_ref outside the fit range)"
        self.profiles.set_profiles(profiles, f"Profiles @ t_ref = {t:.3f} ns", f"{src}; {bg_text}",
                                   "Normalized" if norm else "Counts − background")

    # ------------------------------------------------------------ exports
    def _ask_path(self, title: str, suffix: str) -> str:
        ctx = self.ctx
        default = os.path.join(ctx.save_dir or "", f"{ctx.base_name}{suffix}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default, "CSV files (*.csv);;All files (*)")
        return path

    def export_meta(self) -> Dict[str, object]:
        v, ctx = self._view, self.ctx
        meta = dict(ctx.meta)
        meta.update({
            "content": f"center-line ({self.geom.center_name}) counts per {self.geom.word} (time x lambda)",
            "rebin": v.k if v is not None else 1,
            "dark_subtracted": bool(v is not None and v.dark_sub),
            "dark_reference": ctx.dark_text if v is not None and v.dark_sub else None,
            "wavelength_calibration": None if self.calib is None else self.calib.to_dict(),
        })
        return meta

    def _on_export(self) -> None:
        v = self._get_view()
        if v is None:
            self.status.emit("Export spectral: no histogram.")
            return
        path = self._ask_path("Export time-resolved spectrum", "_spectral.csv")
        if not path:
            return
        sl = slice(v.s, v.e + 1)
        try:
            self.geom.io.save_spectral_matrix_csv(path, v.t[sl], v.spec[sl], self.geom.wavelengths(self.calib),
                                                  self.export_meta())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export spectral", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Spectral matrix exported{' (dark subtracted)' if v.dark_sub else ''}: {path}")

    def _on_origin(self) -> None:
        v = self._get_view()
        if v is None:
            self.status.emit("Map for Origin: no histogram.")
            return
        path = self._ask_path("Save time-resolved spectrum for Origin", "_spectral_map_origin.csv")
        if not path:
            return
        sl = slice(v.s, v.e + 1)
        x = self.geom.wavelengths(self.calib)
        try:
            self.geom.io.save_origin_matrix_csv(path, self.geom.pos_um if x is None else x, v.t[sl], v.spec[sl])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Map for Origin", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Map saved for Origin ({'wavelength nm' if x is not None else 'µm, no λ-cal'} × time ns"
                         f"{', dark subtracted' if v.dark_sub else ''}): {path}")

    def decay_traces(self):
        v = self._get_view()
        if v is None:
            return []
        sl = slice(v.s, v.e + 1)
        return [(self.geom.line_label(c, self.calib), v.t[sl], v.spec[sl, c]) for c in self.lines]

    def _on_copy_decays(self) -> None:
        traces = self.decay_traces()
        if not traces:
            self.status.emit(f"Nothing to copy: select {self.geom.word}s on the map first.")
            return
        QtWidgets.QApplication.clipboard().setText(
            self.geom.io.columns_to_text([traces[0][1]] + [y for _, _, y in traces]))
        self.status.emit(f"Copied {traces[0][1].size} rows: time (ns), " + ", ".join(lab for lab, _, _ in traces))

    def _on_copy_widths(self) -> None:
        res = self._valid_result()
        if res is None or not self.lines:
            self.status.emit("Nothing to copy: run Fit σ(t) and select lines first.")
            return
        acc = res.accepted(float(self.spin_snr.value()))
        q = np.where(acc, self._quantity(res.sigma), np.nan)
        QtWidgets.QApplication.clipboard().setText(
            self.geom.io.columns_to_text([res.time_ns] + [q[:, c] for c in self.lines]))
        self.status.emit(f"Copied {self.cmb_quantity.currentText()} vs time (ns): "
                         + ", ".join(self.geom.line_label(c, self.calib) for c in self.lines))

    def _on_export_widths(self) -> None:
        res = self._valid_result()
        if res is None:
            self.status.emit("Export widths: run Fit σ(t) first.")
            return
        path = self._ask_path("Export width fit", "_spectral_widths.csv")
        if not path:
            return
        g = self.geom
        cols: Dict[str, Sequence[float]] = {"time_ns": res.time_ns}
        for c in range(g.n_lines):
            cols[f"sigma_um_{g.word}{c}"] = res.sigma[:, c]
            cols[f"sigma_err_um_{g.word}{c}"] = res.sigma_err[:, c]
            cols[f"{g.cross}0_um_{g.word}{c}"] = res.center[:, c]
            cols[f"amp_{g.word}{c}"] = res.amp[:, c]
            cols[f"status_{g.word}{c}"] = res.status[:, c].astype(np.float64)
        v = self._get_view()
        meta = self.export_meta()
        meta.update({
            "content": f"1D Gaussian width of every {g.word}'s {g.cross}-profile versus time",
            "method": self.cmb_method.currentText(),
            "min_snr": float(self.spin_snr.value()),
            "free_offset": self.chk_free_offset.isChecked(),
            "background": self._fit_result[2] if self._fit_result else "",
            "status_codes": {str(k): s for k, s in g.an.WIDTH_STATUS_TEXT.items()},
            f"{g.word}_{g.along}_um": [float(x) for x in g.pos_um],
            f"{g.word}_lambda_nm": None if self.calib is None else [float(x) for x in g.wavelengths(self.calib)],
            "t0_peak_ns": float(res.time_ns[0]),
            "fit_range": "time zero (peak of the summed decay) to the histogram end",
            "d_window_ns_after_t0": [float(self.spin_d0.value()), float(self.spin_d1.value()) or None],
            "D_cm2_per_s": {g.line_label(c): (None if d is None else [round(d[0], 6), round(d[1], 6)])
                            for c, d in self.d_results(res)},
        })
        try:
            g.io.save_columns_csv(path, cols, meta)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export widths", f"Could not save:\n{exc}")
            return
        self.status.emit(f"Width fit exported: {path}")

    def showEvent(self, ev):
        super().showEvent(ev)
        self.redraw()
