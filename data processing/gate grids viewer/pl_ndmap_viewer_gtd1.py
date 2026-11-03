"""
pl_ndmap_viewer_gtd1.py

Interactive n-D map viewer for a GT-D1-style (topWSe2/botWS2) dual-gate PL scan.

What it does
------------
Using the gate ratio r = c_b/c_t (measured by gtd1_full_collapse.py, or loaded from a
calibration file), it re-plots the map in the physical coordinates
    u_n = V_f + r*V_b   (x axis; proportional to carrier density n)
    u_D = V_f - r*V_b   (y axis; proportional to displacement field D)
and colours it by a per-gate-point PL scalar (integrated intensity, peak height,
peak energy, linewidth, cosine distance, or fitted-peak quantities).

Calibration + data are decoupled
--------------------------------
  * "Load calibration..."  reads a gtd1_calibration.json (written by the full collapse)
    and applies its r (and CNP if present) to whatever data is loaded.
  * "Load data folder..."  points the viewer at ANY folder of PL_manual_Vf<>_Vb<>.asc
    frames in the same format (it can be a DIFFERENT dataset than the one the calibration
    was fitted on). If that folder already has a dual_gate_speccube_gtd1.npz cache it loads
    it; otherwise it builds the cache (progress bar) and saves it next to the data.
    If frames are newer than the cache (scan still running) the cache is updated
    incrementally; "Reload" does that for the current folder (Shift+click = full rebuild).
    Frames saved as RAW images (no wavelength column) borrow the wavelength axis from an
    .asc with the same grating/center found nearby (or one you pick).
  * "Calibrated axes" checkbox toggles the main map between BEFORE calibration
    (x,y = raw gate voltages V_front, V_back) and AFTER (x,y = u_n, u_D).

Controls
--------
  * Spatial pixel + integration width  (which slit rows to sum into the linecut)
  * Background: subtract the mean over a reference wavelength window (nm)
  * Peak fit: 1-2 Lorentzian/Gaussian peaks over an energy window (eV)
  * r and the n / D origins (u_n0, u_D0)
  * Map-quantity dropdown ; "Fit all points" to colour by fitted quantities
  * "Find CNP" and "Sweep gen" helpers

Folder layout (self-sufficient; copy the whole folder anywhere)
------------------------------------------------------------
  pl_ndmap_viewer_gtd1.py, build_gtd1_cache.py, dual_gate_asc_io.py, gtd1_full_collapse.py
  calibration\\   OPTIONAL, sample-specific: gtd1_calibration.json, gtd1_collapse.png and the
                 reference dual_gate_speccube_gtd1.npz the calibration was fitted on.
At startup the viewer opens the folder given on the command line, else
calibration\\dual_gate_speccube_gtd1.npz, else asks for a data folder.

Run with the workspace ROOT .venv (3.10 + PyQt5):
  & "C:\\Users\\YouAr\\OneDrive - purdue.edu\\.venv\\Scripts\\python.exe" \
    pl_ndmap_viewer_gtd1.py [data_folder]
"""

import os
import sys
import json
import warnings
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from scipy.optimize import curve_fit as _curve_fit
except ImportError:
    _curve_fit = None

try:
    from scipy.interpolate import LinearNDInterpolator as _LinearND
except Exception:
    _LinearND = None

EV_NM = 1239.841984
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                  # build_gtd1_cache / dual_gate_asc_io
CACHE_NAME = "dual_gate_speccube_gtd1.npz"                     # standard cache filename per data folder
CAL_DIR = HERE / "calibration"                                 # sample-specific files (optional)
CACHE = CAL_DIR / CACHE_NAME                                   # default dataset opened at startup
R_DEFAULT = 0.461                       # GT-D1 measured: full intensity shear-collapse -> r=c_b/c_t=0.461+/-0.021
                                        # (gtd1_full_collapse.py). Energy collapse FAILS: IX energy mixes
                                        # Stark(D)+doping(n). Thick hBN (~2.2x) on the BACK gate -> back=top.
                                        # Calibrate by straightening the INTENSITY/cosine band, not energy.

# physical-units conversion (dual-gate capacitor model): n = c_F*u_n/e,  D/e0 = c_F*u_D/(2 e0),
# with c_F = e0*eps_hBN/d_F.  For u in volts and d_F in nm (K_N = (e0/e)*1e5 / 1e12):
#   n [10^12 cm^-2] = K_N * eps_hBN * u_n / d_F ;   D/e0 [V/nm] = eps_hBN * u_D / (2*d_F)
K_N = 5.5264

# ---- appearance: white figure backgrounds + larger fonts --------------------
pg.setConfigOptions(antialias=False, background="w", foreground="k")
GUI_FONT_PT = 11                        # Qt widget font
LABEL_CSS = {"color": "#000000", "font-size": "13pt", "font-family": "Arial"}   # axis label text
TITLE_CSS = {"color": "#000000", "size": "13pt"}                               # plot title text
TICK_PT = 11                            # axis tick numbers
_AXIS_PEN = pg.mkPen("#000000")

# waterfall (energy vs gate) maps: uniform-eV display window + regular (u_n,u_D) grid
WF_E_LO, WF_E_HI, WF_NEU = 1.28, 1.55, 320   # IX window; excludes the 1.20 eV laser leak
WF_NGRID = 81

# raw (fit-free) map quantities are always available; fitted ones need "Fit all"
RAW_QUANTITIES = [
    "Cosine distance (neutral ref)",
    "Integrated intensity (IX win)",
    "Peak height (max)",
    "Peak energy (argmax)",
    "Centroid energy",
    "Linewidth (2nd moment)",
]
FIT_QUANTITIES = [
    "Fit: peak1 energy", "Fit: peak1 height", "Fit: peak1 FWHM (meV)",
    "Fit: peak1 area", "Fit: peak2 energy", "Fit: peak2 area", "Fit: R^2",
]
_ENERGY_QUANTITIES = {"Peak energy (argmax)", "Centroid energy",
                      "Fit: peak1 energy", "Fit: peak2 energy"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pg_cmap(name):
    for source in ("matplotlib", None):
        try:
            return pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
        except Exception:
            pass
    return pg.colormap.get("viridis")


def _tick_font():
    return QtGui.QFont("Arial", TICK_PT)


def _style_axes(pi, bottom=None, left=None):
    """Enlarge tick numbers / axis lines and (optionally) set bigger axis labels
    on a PlotItem so everything is legible on the white background."""
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        if ax is None:
            continue
        try:
            ax.setTickFont(_tick_font())
        except Exception:
            pass
        try:
            ax.setPen(_AXIS_PEN)
            ax.setTextPen(_AXIS_PEN)
        except Exception:
            pass
    if bottom is not None:
        pi.setLabel("bottom", bottom, **LABEL_CSS)
    if left is not None:
        pi.setLabel("left", left, **LABEL_CSS)


def _gauss(e, a, c, s):
    s = max(float(s), 1e-6)
    return a * np.exp(-0.5 * ((e - c) / s) ** 2)


def _lorentz(e, a, c, g):
    g = max(float(g), 1e-6)
    return a * g * g / ((e - c) ** 2 + g * g)


def _peakfunc(shape):
    return _gauss if shape == "Gaussian" else _lorentz


def _fwhm(shape, w):
    return (2.3548 * w) if shape == "Gaussian" else (2.0 * w)   # w = sigma or gamma


def _area(shape, a, w):
    return (a * w * np.sqrt(2 * np.pi)) if shape == "Gaussian" else (np.pi * a * w)


def fit_spectrum(e, y, npeaks, shape, lo, hi):
    """Fit 1-2 peaks (+constant) over [lo,hi] eV. Returns dict or None."""
    m = (e >= lo) & (e <= hi)
    if m.sum() < 6:
        return None
    ew, yw = e[m], y[m]
    f = _peakfunc(shape)
    off0 = float(np.nanmin(yw))
    ipk = int(np.nanargmax(yw))
    c1, a1 = float(ew[ipk]), float(yw[ipk] - off0)
    a1 = max(a1, 1e-6)
    span = hi - lo

    def model1(x, off, a, c, w):
        return off + f(x, a, c, w)

    def model2(x, off, a, c, w, a2, c2, w2):
        return off + f(x, a, c, w) + f(x, a2, c2, w2)

    try:
        if npeaks >= 2:
            p0 = [off0, a1, c1, 0.02, 0.4 * a1, c1 - 0.05, 0.02]
            lob = [-abs(a1), 0, lo, 0.004, 0, lo, 0.004]
            hib = [abs(a1) + 1, 5 * a1, hi, 0.15, 5 * a1, hi, 0.15]
            p0 = np.clip(p0, np.array(lob) + 1e-9, np.array(hib) - 1e-9)
            popt, _ = _curve_fit(model2, ew, yw, p0=p0, bounds=(lob, hib), maxfev=20000)
            off, pk = popt[0], [(popt[1], popt[2], popt[3]), (popt[4], popt[5], popt[6])]
            ymod = model2(ew, *popt)
        else:
            p0 = [off0, a1, c1, 0.02]
            lob = [-abs(a1), 0, lo, 0.004]
            hib = [abs(a1) + 1, 5 * a1, hi, 0.15]
            p0 = np.clip(p0, np.array(lob) + 1e-9, np.array(hib) - 1e-9)
            popt, _ = _curve_fit(model1, ew, yw, p0=p0, bounds=(lob, hib), maxfev=12000)
            off, pk = popt[0], [(popt[1], popt[2], popt[3])]
            ymod = model1(ew, *popt)
    except Exception:
        return None

    pk.sort(key=lambda t: t[1])                       # order by energy
    sst = float(np.nansum((yw - np.nanmean(yw)) ** 2))
    r2 = 1.0 - float(np.nansum((yw - ymod) ** 2)) / sst if sst > 0 else 0.0
    return dict(offset=float(off), peaks=[(float(a), float(c), float(w)) for a, c, w in pk],
                shape=shape, r2=r2, win=(lo, hi))


# ----------------------------------------------------------------------------
# dual-gate sweep-plan generator
# ----------------------------------------------------------------------------
class SweepGenDialog(QtWidgets.QDialog):
    """Generate a dual-gate sweep plan (both gates stepped synchronously) along a
    pure-doping (constant D) or pure-displacement (constant n) line through the CNP.
    Front step is user-set; back |step| = front step / r. Constant-D co-moves the
    gates (+); constant-n opposes them (-)."""

    def __init__(self, parent, r0, vf0, vb0):
        super().__init__(parent)
        self.setWindowTitle("Dual-gate sweep generator")
        self.resize(640, 560)
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Sweep n  @ constant D   (pure doping)",
                            "Sweep D  @ constant n   (pure displacement)"])
        self.rIn = self._dspin(0.05, 20.0, r0, 3, 0.005)
        self.vf0 = self._dspin(-40, 40, vf0, 3, 0.05, " V")
        self.vb0 = self._dspin(-40, 40, vb0, 3, 0.05, " V")
        self.held = self._dspin(-40, 40, 0.0, 3, 0.1, " V")
        self.fstep = self._dspin(0.001, 5.0, 0.05, 4, 0.005, " V")
        self.nside = QtWidgets.QSpinBox(); self.nside.setRange(1, 5000); self.nside.setValue(20)
        self.nside.setFixedWidth(120)
        self.heldLabel = QtWidgets.QLabel(); self.stepsLabel = QtWidgets.QLabel()
        form.addRow("Sweep type", self.kind)
        form.addRow("r = c_b / c_t", self.rIn)
        form.addRow("CNP  V_front (bottom)", self.vf0)
        form.addRow("CNP  V_back (top)", self.vb0)
        form.addRow(self.heldLabel, self.held)                 # held coordinate (label set by type)
        form.addRow("Front-gate step size", self.fstep)
        form.addRow(self.stepsLabel, self.nside)               # steps/side (label set by type)
        lay.addLayout(form)
        self.out = QtWidgets.QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt")
        lay.addWidget(self.out, 1)
        self.saveBtn = QtWidgets.QPushButton("Save plan to .txt ...")
        lay.addWidget(self.saveBtn)
        for w in (self.rIn, self.vf0, self.vb0, self.held, self.fstep):
            w.valueChanged.connect(self._update)
        self.nside.valueChanged.connect(self._update)
        self.kind.currentIndexChanged.connect(self._on_type)
        self.saveBtn.clicked.connect(self._save)
        self._on_type()

    def _on_type(self):
        if self.kind.currentIndex() == 0:                     # sweep n @ const D
            self.heldLabel.setText("Constant u_D  (V, 0 = CNP)")
            self.stepsLabel.setText("Steps per side of CNP (n=0)")
        else:                                                 # sweep D @ const n
            self.heldLabel.setText("Constant u_n  (V, 0 = CNP)")
            self.stepsLabel.setText("Steps per side of D=0")
        self._update()

    def _dspin(self, lo, hi, val, dec, step, suf=""):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
        s.setSingleStep(step); s.setValue(val); s.setFixedWidth(120)
        if suf:
            s.setSuffix(suf)
        return s

    def _plan(self):
        r = float(self.rIn.value()); vf0 = float(self.vf0.value()); vb0 = float(self.vb0.value())
        h = float(self.held.value()); df = float(self.fstep.value()); N = int(self.nside.value())
        db = df / r; const_D = self.kind.currentIndex() == 0
        # swept coord centered on its CNP value (u_n=0 for n-sweep, u_D=0 for D-sweep);
        # the held coord is offset from the CNP by h.  Front is +df either way.
        f_v0, f_step, f_end = vf0 + h / 2 - N * df, df, vf0 + h / 2 + N * df
        if const_D:                                    # hold u_D=h ; gates co-move (+)
            b_v0, b_step, b_end = vb0 - h / (2 * r) - N * db, db, vb0 - h / (2 * r) + N * db
        else:                                          # hold u_n=h ; gates oppose (-)
            b_v0, b_step, b_end = vb0 + h / (2 * r) + N * db, -db, vb0 + h / (2 * r) - N * db
        return dict(r=r, vf0=vf0, vb0=vb0, h=h, df=df, N=N, db=db, const_D=const_D, npts=2 * N + 1,
                    f_v0=f_v0, f_step=f_step, f_end=f_end, b_v0=b_v0, b_step=b_step, b_end=b_end)

    def _text(self):
        p = self._plan()
        if p["const_D"]:
            kind = "Sweep n @ constant D (pure doping)"
            hold = f"# hold          : u_D = {p['h']:+.3f} V   (0 = CNP's D)"
            axis = "u_n (density)"
        else:
            kind = "Sweep D @ constant n (pure displacement)"
            hold = f"# hold          : u_n = {p['h']:+.3f} V   (0 = CNP;  set off-CNP for r-nulling)"
            axis = "u_D (displacement)"
        span = 2 * p["N"] * p["df"]
        return "\n".join([
            "# GT-D1 dual-gate sweep plan",
            f"# type          : {kind}",
            f"# r = c_b/c_t   : {p['r']:.3f}",
            f"# CNP           : V_front(bottom) = {p['vf0']:+.3f} V , V_back(top) = {p['vb0']:+.3f} V",
            hold,
            f"# front step    : {p['df']:.4f} V   steps/side = {p['N']}   points = {p['npts']}",
            f"# back |step|   : {p['db']:.5f} V   (= front step / r)",
            f"# swept coord   : {axis} = center +/- {span:.3f} V   ({2 * p['df']:.4f} V per point)",
            "#",
            "# gate    V0 (V)       step (V)        points     Vend (V)",
            f"FRONT    {p['f_v0']:+.4f}     {p['f_step']:+.6f}     {p['npts']:>5d}     {p['f_end']:+.4f}",
            f"BACK     {p['b_v0']:+.4f}     {p['b_step']:+.6f}     {p['npts']:>5d}     {p['b_end']:+.4f}",
            "#",
            f"# dual/synchronous scan: for k = 0..{p['npts'] - 1}: "
            "V_front = FRONT.V0 + k*FRONT.step ; V_back = BACK.V0 + k*BACK.step",
        ])

    def _update(self):
        self.out.setPlainText(self._text())

    def _save(self):
        default = "sweep_n_constD.txt" if self.kind.currentIndex() == 0 else "sweep_D_constn.txt"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save sweep plan", default,
                                                      "Text files (*.txt)")
        if fn:
            Path(fn).write_text(self._text(), encoding="utf-8")
            self.setWindowTitle(f"Dual-gate sweep generator  -  saved {Path(fn).name}")


# ----------------------------------------------------------------------------
# main window
# ----------------------------------------------------------------------------
class NDMapViewer(QtWidgets.QMainWindow):
    def __init__(self, data, source=None):
        super().__init__()
        self.calibrated = True          # main map axes: True = (u_n,u_D), False = raw (Vf,Vb)
        self.cal_meta = None            # loaded calibration dict
        self.cal_path = None
        self._source = source           # data folder (for the title)
        self._prof = None               # cached spatial profile (recomputed per dataset)
        self._cnp = None                # (Vf,Vb) from Find CNP / calibration, for the sweep generator
        self._sync = False

        self._set_arrays(data)          # populate all data-derived arrays + state
        self._build_ui()                # controls reference spat_rows / spot_center / nwin
        self._refresh_all()

    # ---- data (re)load ---------------------------------------------------
    def _set_arrays(self, data):
        """Assign every data-derived array + reset derived caches (used at start-up
        and whenever a new data folder is loaded)."""
        self.cube = data["cube"]                        # (nvf,nvb,nwin,nwl) float32
        self.wl = np.asarray(data["wl"], float)
        self.vf = np.asarray(data["vf_vals"], float)
        self.vb = np.asarray(data["vb_vals"], float)
        self.tstamp = np.asarray(data["tstamp"], float)
        self.measured = np.isfinite(self.tstamp)         # incomplete/ongoing grid -> blank unmeasured
        self.spat_rows = np.asarray(data["spat_rows"], int)
        self.spot_center = int(data["spot_center"])
        self.nvf, self.nvb, self.nwin, self.nwl = self.cube.shape
        # per-axis gate steps: a map taken with back-step = front-step/r has dvb != dvf
        self.dvf = float(np.median(np.diff(self.vf))) if self.vf.size > 1 else 0.2   # front-gate step
        self.dvb = float(np.median(np.diff(self.vb))) if self.vb.size > 1 else 0.2   # back-gate step
        self.dv = self.dvf                                                            # back-compat alias
        self.source_dir = str(data["source_dir"]) if "source_dir" in data else None

        self.order = np.argsort(EV_NM / self.wl)         # -> ascending energy
        self.e = (EV_NM / self.wl)[self.order]
        self.VF, self.VB = np.meshgrid(self.vf, self.vb, indexing="ij")

        self.spec_bs = None                              # (nvf,nvb,nwl) eV-ordered, bg-subtracted
        self._cosdist = None                             # cached cosine-distance map
        self.fit = None                                  # dict of fitted-quantity maps
        self.fit_cfg = None
        self.sel = (self.nvf // 2, self.nvb // 2)

        # waterfall state
        self._e_u = np.linspace(WF_E_LO, WF_E_HI, WF_NEU)
        self._spec_u = None                              # (nvf*nvb, WF_NEU) uniform-eV spectra
        self._wf = None                                  # (WF_NGRID,WF_NGRID,WF_NEU) on regular (u_n,u_D)
        self._wf_grid = None
        self._wf_stale = True
        self._dIdx = WF_NGRID // 2                        # swept D slice index (density map)
        self._nIdx = WF_NGRID // 2                        # swept n slice index (D map)

    def _refresh_all(self):
        self._recompute_spectra()
        self._update_spatial_profile()
        self._update_phys_info()
        self._apply_map_axes()
        self._apply_wf_axes()
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_active_tab()
        self._update_slice_lines()
        self._update_title()

    def _update_title(self):
        name = Path(self._source).name if self._source else "GT-D1"
        self.setWindowTitle(f"PL n-D map viewer   ·   {name}   ·   {self.nvf}x{self.nvb} grid")

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        self.resize(1680, 940)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)
        root.addWidget(self._make_controls())

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._make_map_panel())

        self.tabs = QtWidgets.QTabWidget()
        spec_tab = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        spec_tab.addWidget(self._make_spectrum_panel())
        spec_tab.addWidget(self._make_spatial_panel())
        spec_tab.setStretchFactor(0, 3); spec_tab.setStretchFactor(1, 1)
        self.tabs.addTab(spec_tab, "Spectrum")
        self.tabs.addTab(self._make_wf_panel("density"), "Spectrum vs density (sweep D)")
        self.tabs.addTab(self._make_wf_panel("field"), "Spectrum vs D (sweep n)")
        self.tabs.currentChanged.connect(lambda _: self._on_tab_changed())
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        root.addWidget(split, 1)
        self.setCentralWidget(central)

    def _dspin(self, lo, hi, val, dec, step, suf="", width=92):
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(dec); s.setRange(lo, hi); s.setSingleStep(step)
        s.setValue(val); s.setFixedWidth(width)
        if suf:
            s.setSuffix(suf)
        return s

    def _make_controls(self):
        box = QtWidgets.QGroupBox("Controls")
        g = QtWidgets.QGridLayout(box)
        g.setContentsMargins(8, 6, 8, 6); g.setHorizontalSpacing(8); g.setVerticalSpacing(4)

        r0 = self.spat_rows[0]; r1 = self.spat_rows[-1]
        self.spatCenter = QtWidgets.QSpinBox(); self.spatCenter.setRange(r0, r1)
        self.spatCenter.setValue(self.spot_center); self.spatCenter.setFixedWidth(84)
        self.spatWidth = QtWidgets.QSpinBox(); self.spatWidth.setRange(1, self.nwin)
        self.spatWidth.setValue(min(13, self.nwin)); self.spatWidth.setFixedWidth(74)
        self.bgLo = self._dspin(200, 2000, 1050.0, 1, 5, " nm")
        self.bgHi = self._dspin(200, 2000, 1150.0, 1, 5, " nm")
        self.rSpin = self._dspin(0.2, 5.0, R_DEFAULT, 3, 0.005)
        self.nOrig = self._dspin(-20, 20, 0.0, 2, 0.1, " V")
        self.dOrig = self._dspin(-20, 20, 0.0, 2, 0.1, " V")
        self.shapeCombo = QtWidgets.QComboBox(); self.shapeCombo.addItems(["Lorentzian", "Gaussian"])
        self.shapeCombo.setFixedWidth(112)
        self.npeaks = QtWidgets.QSpinBox(); self.npeaks.setRange(1, 2); self.npeaks.setValue(1)
        self.npeaks.setFixedWidth(56)
        self.fitLo = self._dspin(0.8, 2.2, 1.28, 3, 0.01, " eV")   # >1.20 eV laser leak
        self.fitHi = self._dspin(0.8, 2.2, 1.52, 3, 0.01, " eV")
        self.quantity = QtWidgets.QComboBox(); self.quantity.addItems(RAW_QUANTITIES + FIT_QUANTITIES)
        self.quantity.setMinimumWidth(250)
        self.quantity.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.quantity.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.fitAllBtn = QtWidgets.QPushButton("Fit all points"); self.fitAllBtn.setFixedWidth(130)
        self.cnpBtn = QtWidgets.QPushButton("Find CNP"); self.cnpBtn.setFixedWidth(110)
        self.sweepBtn = QtWidgets.QPushButton("Sweep gen"); self.sweepBtn.setFixedWidth(110)
        self.sweepBtn.setToolTip("Generate a dual-gate sweep plan (const-D or const-n through the CNP)")
        self.cnpBtn.setToolTip("Peak IX intensity along u_D=0 -> max = charge neutrality; "
                               "reports gate voltages and re-zeros the density axis there")

        # data + calibration + before/after toggle
        self.loadDataBtn = QtWidgets.QPushButton("Load data folder…")
        self.loadDataBtn.setToolTip("Point the viewer at any folder of PL_manual_Vf<>_Vb<>.asc frames "
                                    "(loads or builds its cache)")
        self.reloadBtn = QtWidgets.QPushButton("Reload")
        self.reloadBtn.setToolTip("Re-read the current data folder (picks up new frames of a scan "
                                  "that is still running; only new frames are read).\n"
                                  "Shift+click: rebuild the cache from scratch.")
        self.loadCalBtn = QtWidgets.QPushButton("Load calibration…")
        self.loadCalBtn.setToolTip("Read a gtd1_calibration.json and apply its r to the loaded data")
        self.calChk = QtWidgets.QCheckBox("Calibrated axes (u_n, u_D)")
        self.calChk.setChecked(True)
        self.calChk.setToolTip("Off = main map x,y are raw gate voltages (V_front, V_back); "
                               "On = x,y are u_n (density) and u_D (displacement)")
        self.calInfo = QtWidgets.QLabel(f"r = {R_DEFAULT:.3f}  (default; no calibration file loaded)")
        self.calInfo.setStyleSheet("color:#1a5aa0")
        # physical-units panel: enter front hBN thickness d_F and eps_hBN -> axes in n, D/e0
        self.dF = self._dspin(0.5, 500.0, 30.0, 1, 1.0, " nm")
        self.eps = self._dspin(1.0, 10.0, 3.4, 2, 0.1)
        self.dF.setToolTip("Front (thin) hBN thickness d_F.  Back hBN d_B = d_F / r.")
        self.eps.setToolTip("Out-of-plane hBN dielectric constant (≈ 3.0–3.4)")
        self.physChk = QtWidgets.QCheckBox("Physical units (n, D/ε₀)")
        self.physChk.setToolTip("Relabel the map + waterfall axes in n [10¹² cm⁻²] and D/ε₀ [V/nm] "
                                "from d_F and ε_hBN (turns on calibrated axes)")
        self.physInfo = QtWidgets.QLabel(""); self.physInfo.setStyleSheet("color:#1a7a3a")
        self.status = QtWidgets.QLabel(""); self.status.setStyleSheet("color:#555")

        def lab(t):
            return QtWidgets.QLabel(t)
        g.addWidget(lab("Spatial px"), 0, 0); g.addWidget(self.spatCenter, 0, 1)
        g.addWidget(lab("width"), 0, 2); g.addWidget(self.spatWidth, 0, 3)
        g.addWidget(lab("Bkg ref"), 0, 4); g.addWidget(self.bgLo, 0, 5)
        g.addWidget(lab("to"), 0, 6); g.addWidget(self.bgHi, 0, 7)
        g.addWidget(lab("r=c_b/c_t"), 0, 8); g.addWidget(self.rSpin, 0, 9)
        g.addWidget(self.cnpBtn, 0, 10, 1, 2)
        g.addWidget(lab("Peaks"), 1, 0); g.addWidget(self.npeaks, 1, 1)
        g.addWidget(self.shapeCombo, 1, 2, 1, 2)
        g.addWidget(lab("Fit win"), 1, 4); g.addWidget(self.fitLo, 1, 5)
        g.addWidget(lab("to"), 1, 6); g.addWidget(self.fitHi, 1, 7)
        g.addWidget(lab("u_n0"), 1, 8); g.addWidget(self.nOrig, 1, 9)
        g.addWidget(self.sweepBtn, 1, 10, 1, 2)
        g.addWidget(lab("Colour by"), 2, 0); g.addWidget(self.quantity, 2, 1, 1, 3)
        g.addWidget(self.fitAllBtn, 2, 4, 1, 2)
        g.addWidget(lab("u_D0"), 2, 6); g.addWidget(self.dOrig, 2, 7)
        g.addWidget(self.calChk, 2, 8, 1, 4)
        # data / calibration row
        g.addWidget(self.loadDataBtn, 3, 0, 1, 2)
        g.addWidget(self.reloadBtn, 3, 2)
        g.addWidget(self.loadCalBtn, 3, 3, 1, 2)
        g.addWidget(self.calInfo, 3, 5, 1, 7)
        # physical-units row
        g.addWidget(lab("d_F"), 4, 0); g.addWidget(self.dF, 4, 1)
        g.addWidget(lab("ε_hBN"), 4, 2); g.addWidget(self.eps, 4, 3)
        g.addWidget(self.physChk, 4, 4, 1, 3)
        g.addWidget(self.physInfo, 4, 7, 1, 5)
        g.addWidget(self.status, 5, 0, 1, 12)

        # wiring
        for w in (self.spatCenter, self.spatWidth):
            w.valueChanged.connect(lambda _: self._on_band_changed())
        for w in (self.bgLo, self.bgHi):
            w.valueChanged.connect(lambda _: self._on_bg_changed())
        for w in (self.fitLo, self.fitHi):
            w.valueChanged.connect(lambda _: self._on_fitcfg_changed())
        self.shapeCombo.currentTextChanged.connect(lambda _: self._on_fitcfg_changed())
        self.npeaks.valueChanged.connect(lambda _: self._on_fitcfg_changed())
        for w in (self.rSpin, self.nOrig, self.dOrig):
            w.valueChanged.connect(lambda _: self._on_coords_changed())
        self.quantity.currentTextChanged.connect(lambda _: self._refresh_map())
        self.fitAllBtn.clicked.connect(self._fit_all)
        self.cnpBtn.clicked.connect(self._find_cnp)
        self.sweepBtn.clicked.connect(self._open_sweep_gen)
        self.loadDataBtn.clicked.connect(self._load_data_folder)
        self.reloadBtn.clicked.connect(self._reload_data_folder)
        self.loadCalBtn.clicked.connect(self._load_calibration)
        self.calChk.toggled.connect(self._apply_display)
        self.physChk.toggled.connect(self._apply_display)
        for w in (self.dF, self.eps):
            w.valueChanged.connect(lambda _: self._on_units_param())
        return box

    def _make_map_panel(self):
        box = QtWidgets.QGroupBox("n - D map")
        lay = QtWidgets.QVBoxLayout(box)
        self._glw = pg.GraphicsLayoutWidget()
        self._mapPlot = self._glw.addPlot(row=0, col=0)
        self._img = pg.ImageItem()
        self._mapPlot.addItem(self._img)
        self._hist = pg.HistogramLUTItem(); self._hist.setImageItem(self._img)
        self._glw.addItem(self._hist, row=0, col=1)
        self._marker = pg.ScatterPlotItem(size=16, pen=pg.mkPen("k", width=2),
                                          brush=pg.mkBrush(None), symbol="x")
        self._mapPlot.addItem(self._marker)
        # slice lines: horizontal = constant-D cut (density sweep); vertical = constant-n cut (D sweep)
        self._sliceH = pg.InfiniteLine(angle=0, movable=True,
                                       pen=pg.mkPen("#00b3cc", width=1.6, style=QtCore.Qt.DashLine))
        self._sliceV = pg.InfiniteLine(angle=90, movable=True,
                                       pen=pg.mkPen("#d43bb0", width=1.6, style=QtCore.Qt.DashLine))
        for ln in (self._sliceH, self._sliceV):
            ln.setVisible(False); self._mapPlot.addItem(ln)
        self._sliceH.sigPositionChangeFinished.connect(self._on_sliceH_drag)
        self._sliceV.sigPositionChangeFinished.connect(self._on_sliceV_drag)
        self._mapPlot.scene().sigMouseClicked.connect(self._on_map_click)
        self._glw.ci.layout.setColumnMaximumWidth(1, 90)
        self._apply_map_axes()
        lay.addWidget(self._glw)
        return box

    def _make_spectrum_panel(self):
        box = QtWidgets.QGroupBox("Spectrum + fit  (click a pixel on the map)")
        lay = QtWidgets.QVBoxLayout(box)
        self._specPlot = pg.PlotWidget()
        self._specPlot.addLegend(offset=(-10, 10))
        self._cRaw = self._specPlot.plot(pen=pg.mkPen("#8a8aa6", width=1.2), name="raw")
        self._cBs = self._specPlot.plot(pen=pg.mkPen("#1f77b4", width=1.8), name="bkg-sub")
        self._cFit = self._specPlot.plot(pen=pg.mkPen("#e08a00", width=1.8,
                                         style=QtCore.Qt.DashLine), name="fit")
        self._cP1 = self._specPlot.plot(pen=pg.mkPen("#2ca02c", width=1.2, style=QtCore.Qt.DotLine))
        self._cP2 = self._specPlot.plot(pen=pg.mkPen("#d62728", width=1.2, style=QtCore.Qt.DotLine))
        self._fitReg = pg.LinearRegionItem(brush=pg.mkBrush(255, 179, 0, 40), movable=False)
        self._fitReg.setZValue(-10); self._specPlot.addItem(self._fitReg)
        _style_axes(self._specPlot.getPlotItem(), bottom="Photon energy (eV)",
                    left="PL (counts, bkg-subtracted)")
        lay.addWidget(self._specPlot)
        return box

    def _make_spatial_panel(self):
        box = QtWidgets.QGroupBox("Spatial profile (drag the band to set integration rows)")
        lay = QtWidgets.QVBoxLayout(box)
        self._spatPlot = pg.PlotWidget()
        self._cProf = self._spatPlot.plot(pen=pg.mkPen("#1f77b4", width=1.5))
        self._bandReg = pg.LinearRegionItem(brush=pg.mkBrush(255, 140, 0, 55))
        self._spatPlot.addItem(self._bandReg)
        self._bandReg.sigRegionChangeFinished.connect(self._on_region_changed)
        _style_axes(self._spatPlot.getPlotItem(), bottom="spatial pixel (along slit)",
                    left="PL weight")
        lay.addWidget(self._spatPlot)
        return box

    # ---- waterfall (energy vs gate) panels -------------------------------
    def _make_wf_panel(self, kind):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        col = "#00b3cc" if kind == "density" else "#d43bb0"
        top.addWidget(QtWidgets.QLabel("sweep:"))
        sld = QtWidgets.QSlider(QtCore.Qt.Horizontal); sld.setRange(0, WF_NGRID - 1)
        sld.setValue(WF_NGRID // 2)
        lbl = QtWidgets.QLabel("—"); lbl.setStyleSheet(f"color:{col}; font-weight:bold; font-size:12pt")
        top.addWidget(sld, 1); top.addWidget(lbl)
        v.addLayout(top)
        glw = pg.GraphicsLayoutWidget()
        plot = glw.addPlot(row=0, col=0)
        _style_axes(plot, bottom="Photon energy (eV)",
                    left="u_n  (V, ∝ density)" if kind == "density" else "u_D  (V, ∝ D)")
        img = pg.ImageItem(); plot.addItem(img)
        hist = pg.HistogramLUTItem(); hist.setImageItem(img)
        glw.addItem(hist, row=0, col=1); glw.ci.layout.setColumnMaximumWidth(1, 90)
        v.addWidget(glw)
        setattr(self, f"_wfImg_{kind}", img); setattr(self, f"_wfHist_{kind}", hist)
        setattr(self, f"_wfPlot_{kind}", plot); setattr(self, f"_wfLbl_{kind}", lbl)
        setattr(self, f"_wfSld_{kind}", sld)
        sld.valueChanged.connect(lambda val, k=kind: self._on_wf_slider(k, val))
        return w

    def _build_wf(self):
        if self._wf is not None and not self._wf_stale:
            return
        if _LinearND is None or self._spec_u is None:
            self._wf = None; return
        r = float(self.rSpin.value())
        un = (self.VF + r * self.VB - float(self.nOrig.value())).ravel()
        uD = (self.VF - r * self.VB - float(self.dOrig.value())).ravel()
        keep = self.measured.ravel()                     # interpolate only measured points
        try:                                             # partial scan: needs >=3 non-collinear pts
            interp = _LinearND(np.column_stack([un[keep], uD[keep]]), self._spec_u[keep])
        except Exception:
            self._wf = None; self._wf_grid = None; return
        ung = np.linspace(un[keep].min(), un[keep].max(), WF_NGRID)
        uDg = np.linspace(uD[keep].min(), uD[keep].max(), WF_NGRID)
        UN, UD = np.meshgrid(ung, uDg, indexing="ij")
        cube = interp(np.column_stack([UN.ravel(), UD.ravel()]))
        self._wf = cube.reshape(WF_NGRID, WF_NGRID, self._e_u.size)
        self._wf_grid = (ung, uDg)
        self._wf_stale = False

    def _set_levels(self, img_item, hist, arr, cmap):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            hist.gradient.setColorMap(_pg_cmap(cmap))
            lv = (float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99)))
            img_item.setLevels(lv); hist.setLevels(*lv)

    def _update_wf(self, kind):
        self._build_wf()
        if self._wf is None:
            self.status.setText("scipy.interpolate needed for waterfall maps" if _LinearND is None
                                else "not enough measured points (yet) for the waterfall maps")
            return
        ung, uDg = self._wf_grid
        phys = self._mode() == "phys"
        sN, sD = self._scale_factors() if phys else (1.0, 1.0)
        img_item = getattr(self, f"_wfImg_{kind}"); hist = getattr(self, f"_wfHist_{kind}")
        plot = getattr(self, f"_wfPlot_{kind}"); lbl = getattr(self, f"_wfLbl_{kind}")
        if kind == "density":
            idx = int(np.clip(self._dIdx, 0, WF_NGRID - 1))
            img = self._wf[:, idx, :]                       # rows = u_n
            yv = ung * sN
            lbl.setText(f"D/ε₀ = {uDg[idx]*sD:+.3f} V/nm" if phys else f"u_D = {uDg[idx]:+.2f} V")
        else:
            idx = int(np.clip(self._nIdx, 0, WF_NGRID - 1))
            img = self._wf[idx, :, :]                       # rows = u_D
            yv = uDg * sD
            lbl.setText(f"n = {ung[idx]*sN:+.3f} ×10¹² cm⁻²" if phys else f"u_n = {ung[idx]:+.2f} V")
        y0, yspan = float(yv[0]), float(yv[-1] - yv[0])
        img_item.setImage(np.asarray(img, float).T, autoLevels=False)   # x=energy, y=gate
        img_item.setRect(QtCore.QRectF(self._e_u[0], y0, self._e_u[-1] - self._e_u[0], yspan))
        self._set_levels(img_item, hist, img, "magma")
        plot.setYRange(y0, y0 + yspan, padding=0.02)
        plot.setXRange(self._e_u[0], self._e_u[-1], padding=0.02)

    def _on_wf_slider(self, kind, val):
        if kind == "density":
            self._dIdx = int(val)
        else:
            self._nIdx = int(val)
        self._update_wf(kind); self._update_slice_lines()

    def _on_tab_changed(self):
        self._refresh_active_tab(); self._update_slice_lines()

    def _refresh_active_tab(self):
        i = self.tabs.currentIndex()
        if i == 1:
            self._update_wf("density")
        elif i == 2:
            self._update_wf("field")

    def _update_slice_lines(self):
        # slice lines are a calibrated (u_n,u_D) concept -> only on the waterfall tabs AND
        # when the map is in calibrated axes.
        i = self.tabs.currentIndex()
        show = self.calibrated
        self._sliceH.setVisible(show and i == 1)
        self._sliceV.setVisible(show and i == 2)
        if self._wf_grid is None:
            return
        ung, uDg = self._wf_grid
        sN, sD = self._xy_scale()                          # map axes are (u_n,u_D) or (n,D/e0)
        self._sync = True
        self._sliceH.setValue(float(uDg[int(np.clip(self._dIdx, 0, WF_NGRID - 1))]) * sD)
        self._sliceV.setValue(float(ung[int(np.clip(self._nIdx, 0, WF_NGRID - 1))]) * sN)
        self._sync = False

    def _sync_slider(self, kind, idx):
        sld = getattr(self, f"_wfSld_{kind}")
        sld.blockSignals(True); sld.setValue(int(idx)); sld.blockSignals(False)

    def _on_sliceH_drag(self):
        if self._sync or self._wf_grid is None:
            return
        _, uDg = self._wf_grid
        _, sD = self._xy_scale()                            # drag value is in map-y units (u_D or D/e0)
        val = self._sliceH.value() / (sD or 1.0)
        self._dIdx = int(np.argmin(np.abs(uDg - val)))
        self._sync_slider("density", self._dIdx)
        self._update_wf("density"); self._update_slice_lines()

    def _on_sliceV_drag(self):
        if self._sync or self._wf_grid is None:
            return
        ung, _ = self._wf_grid
        sN, _ = self._xy_scale()                            # drag value is in map-x units (u_n or n)
        val = self._sliceV.value() / (sN or 1.0)
        self._nIdx = int(np.argmin(np.abs(ung - val)))
        self._sync_slider("field", self._nIdx)
        self._update_wf("field"); self._update_slice_lines()

    def _on_coords_changed(self):
        self._wf_stale = True
        self._refresh_map()
        self._refresh_active_tab()
        self._update_slice_lines()

    # ---- band / spectra --------------------------------------------------
    def _band_slice(self):
        c = int(self.spatCenter.value()); w = int(self.spatWidth.value())
        half = w // 2
        lo = c - half; hi = c + half + (0 if w % 2 == 0 else 0)
        lo = max(self.spat_rows[0], lo); hi = min(self.spat_rows[-1], c + half)
        i0 = int(lo - self.spat_rows[0]); i1 = int(hi - self.spat_rows[0])
        return i0, max(i0, i1)

    def _recompute_spectra(self):
        i0, i1 = self._band_slice()
        band = np.nan_to_num(self.cube[:, :, i0:i1 + 1, :]).sum(axis=2)   # (nvf,nvb,nwl) nm order
        lo, hi = float(self.bgLo.value()), float(self.bgHi.value())
        mask = (self.wl >= lo) & (self.wl <= hi)
        if mask.any():
            band = band - np.nanmean(band[:, :, mask], axis=2, keepdims=True)
        self.spec_bs = band[:, :, self.order]                             # eV-ordered
        self.fit = None; self.fit_cfg = None                             # invalidate fits
        self._cosdist = None                                             # invalidate cosine map
        # uniform-eV spectra for the waterfall maps (same source axis for all pixels)
        flat = self.spec_bs.reshape(-1, self.nwl)
        su = np.empty((flat.shape[0], self._e_u.size), np.float32)
        for k in range(flat.shape[0]):
            su[k] = np.interp(self._e_u, self.e, flat[k])
        self._spec_u = su
        self._wf_stale = True

    # ---- scalar maps -----------------------------------------------------
    def _fit_window_mask(self):
        """Energy mask of the fit window; None (+ status note) if it misses the spectrum."""
        lo, hi = float(self.fitLo.value()), float(self.fitHi.value())
        m = (self.e >= lo) & (self.e <= hi)
        if not m.any():
            self.status.setText(f"fit window {lo:.3f}–{hi:.3f} eV is outside the data "
                                f"({np.nanmin(self.e):.3f}–{np.nanmax(self.e):.3f} eV)")
            return None
        return m

    def _raw_map(self, name):
        if name == "Cosine distance (neutral ref)":
            return self._cosine_map()
        m = self._fit_window_mask()
        if m is None:
            return np.full((self.nvf, self.nvb), np.nan)
        ew = self.e[m]
        Y = self.spec_bs[:, :, m]
        if name == "Integrated intensity (IX win)":
            return Y.sum(-1)
        if name == "Peak height (max)":
            return Y.max(-1)
        if name == "Peak energy (argmax)":
            return ew[np.argmax(Y, axis=-1)]
        Yc = np.clip(Y, 0, None); tot = Yc.sum(-1)
        cen = (Yc * ew).sum(-1) / np.where(tot > 0, tot, 1)
        if name == "Centroid energy":
            return cen
        if name == "Linewidth (2nd moment)":
            var = (Yc * (ew[None, None, :] - cen[..., None]) ** 2).sum(-1) / np.where(tot > 0, tot, 1)
            return np.sqrt(np.clip(var, 0, None)) * 1000.0                # meV
        return np.zeros((self.nvf, self.nvb))

    def _cosine_map(self):
        """Cosine DISTANCE of each area-normalized spectrum to a neutral-region
        reference (fit-free, uses the full spectrum). Low = same character as the
        reference; the dark band traces the neutral / iso-D line. Reference is
        auto-picked from measured points near (Vf,Vb)=(0,0), then refined to the
        lowest-distance nearby points."""
        if self._cosdist is not None:
            return self._cosdist
        m = self._fit_window_mask()                                  # restrict to IX (no laser leak)
        vf, vb, meas = self.VF.ravel(), self.VB.ravel(), self.measured.ravel()
        if m is None or not meas.any():
            return np.full((self.nvf, self.nvb), np.nan)
        S = np.nan_to_num(self.spec_bs[:, :, m].reshape(-1, int(m.sum())).astype(float))
        Sb = np.clip(S - np.median(S, axis=1, keepdims=True), 0, None)
        nrm = np.linalg.norm(Sb, axis=1, keepdims=True)
        U = Sb / np.where(nrm > 0, nrm, 1.0)
        ref_sel = meas & (np.abs(vf) <= 0.6) & (np.abs(vb) <= 0.6)
        if ref_sel.sum() < 3:
            ref_sel = meas & (np.abs(vf) <= 1.2) & (np.abs(vb) <= 1.2)
        if ref_sel.sum() < 1:
            ref_sel = meas
        ref = U[ref_sel].mean(0); ref /= (np.linalg.norm(ref) or 1.0)
        d = 1.0 - U @ ref
        near = np.where(meas & (np.abs(vf) < 2.0) & (np.abs(vb) < 2.0))[0]
        if near.size:
            sel = near[np.argsort(d[near])[:max(15, near.size // 5)]]
            ref = U[sel].mean(0); ref /= (np.linalg.norm(ref) or 1.0)
            d = 1.0 - U @ ref
        self._cosdist = d.reshape(self.nvf, self.nvb)
        return self._cosdist

    def _find_cnp(self):
        """Charge neutrality via the zero-displacement PL line: along u_D=0 the IX
        Stark shift is fixed, so peak intensity vs u_n peaks at n=0 (undoped IX is
        brightest). The neutral region is a broad plateau, so the estimate is the
        intensity-weighted CENTROID of the bright plateau (a raw argmax would snap to
        the single grid vertex lying exactly on u_D=0). Peak intensity is averaged over
        a thin u_D band for robustness; reports the gate voltages with the plateau
        uncertainty and re-zeros the density axis at the CNP."""
        if _LinearND is None:
            self.status.setText("scipy.interpolate needed for the CNP finder"); return
        r = float(self.rSpin.value()); ud0 = float(self.dOrig.value())
        ph = self._raw_map("Peak height (max)")
        UNr = self.VF + r * self.VB; UDr = self.VF - r * self.VB
        meas = self.measured & np.isfinite(ph)
        if meas.sum() < 8:
            self.status.setText("not enough measured points"); return
        try:
            interp = _LinearND(np.column_stack([UNr[meas].ravel(), UDr[meas].ravel()]),
                               ph[meas].ravel())
        except Exception:
            self.status.setText("measured points do not span an area (yet) -> no CNP"); return
        uln = np.linspace(UNr[meas].min(), UNr[meas].max(), 400)
        # average over a thin u_D band around 0 so a single on-line grid vertex can't dominate
        bands = [interp(np.column_stack([uln, np.full(uln.shape, ud0 + dd)]))
                 for dd in (-0.3, -0.15, 0.0, 0.15, 0.3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # off-data stretches are all-NaN
            I = np.nanmean(np.vstack(bands), axis=0)
        ok = np.isfinite(I)
        if ok.sum() < 5:
            self.status.setText("the u_D=0 line is not covered by measured data"); return
        uln, I = uln[ok], I[ok]
        Imin, Imax = float(np.min(I)), float(np.max(I))
        sel = I >= Imin + 0.85 * (Imax - Imin)                 # bright plateau (top 15%)
        un_cnp = float(np.average(uln[sel], weights=(I[sel] - Imin)))
        un_lo, un_hi = float(uln[sel].min()), float(uln[sel].max())
        dun = 0.5 * (un_hi - un_lo)                            # plateau half-width ~ uncertainty
        Vf = (un_cnp + ud0) / 2.0; Vb = (un_cnp - ud0) / (2.0 * r)
        dVf, dVb = dun / 2.0, dun / (2.0 * r)
        self._cnp = (Vf, Vb)                                  # remember for the sweep generator
        self.nOrig.setValue(un_cnp)                            # re-zero density axis at CNP
        msg = (f"CNP @ u_D=0:  u_n={un_cnp:+.2f}+/-{dun:.2f} V  ->  V_front={Vf:+.2f}+/-{dVf:.2f}, "
               f"V_back={Vb:+.2f}+/-{dVb:.2f} V (back=top; broad neutral plateau).")
        self.status.setText(msg); print(msg)
        self._show_cnp_plot(uln - un_cnp, I, un_cnp, un_lo - un_cnp, un_hi - un_cnp,
                            Vf, Vb, dVf, dVb, r)

    def _show_cnp_plot(self, un_rel, I, un_cnp, lo_rel, hi_rel, Vf, Vb, dVf, dVb, r):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("CNP finder  —  IX peak intensity along u_D = 0")
        lay = QtWidgets.QVBoxLayout(dlg)
        pw = pg.PlotWidget()
        _style_axes(pw.getPlotItem(),
                    bottom="u_n - u_n0   (V,  proportional to density n;  0 = CNP)",
                    left="IX peak intensity (counts)")
        reg = pg.LinearRegionItem(values=(lo_rel, hi_rel),
                                  brush=pg.mkBrush(255, 179, 0, 55), movable=False)
        reg.setZValue(-10); pw.addItem(reg)
        pw.plot(un_rel, I, pen=pg.mkPen("#1f77b4", width=2.2))
        pw.addItem(pg.InfiniteLine(pos=0.0, angle=90,
                                   pen=pg.mkPen("r", width=1.8, style=QtCore.Qt.DashLine)))
        lay.addWidget(pw)
        lbl = QtWidgets.QLabel(
            "CNP  (centroid of the bright plateau at zero displacement field):\n"
            f"    u_n  = {un_cnp:+.3f} V\n"
            f"    V_front (bottom gate) = {Vf:+.2f} +/- {dVf:.2f} V\n"
            f"    V_back  (top gate)    = {Vb:+.2f} +/- {dVb:.2f} V\n"
            f"    r = {r:.3f}     shaded = bright plateau; density axis re-zeroed at CNP")
        lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 12pt")
        lay.addWidget(lbl)
        dlg.resize(720, 500)
        dlg.show()
        self._cnp_dlg = dlg                                    # keep a reference (non-modal)

    def _open_sweep_gen(self):
        r = float(self.rSpin.value())
        if self._cnp is not None:
            vf0, vb0 = self._cnp
        else:                                                 # CNP from current origins u_n0/u_D0
            un0, ud0 = float(self.nOrig.value()), float(self.dOrig.value())
            vf0, vb0 = (un0 + ud0) / 2.0, (un0 - ud0) / (2.0 * r)
        dlg = SweepGenDialog(self, r, round(vf0, 3), round(vb0, 3))
        dlg.show()
        self._sweep_dlg = dlg                                 # keep a reference (non-modal)

    def _fit_all(self):
        if _curve_fit is None:
            self.status.setText("scipy not available"); return
        lo, hi = float(self.fitLo.value()), float(self.fitHi.value())
        shape = self.shapeCombo.currentText(); npk = int(self.npeaks.value())
        self.status.setText("fitting all points ..."); QtWidgets.QApplication.processEvents()
        keys = ["Fit: peak1 energy", "Fit: peak1 height", "Fit: peak1 FWHM (meV)",
                "Fit: peak1 area", "Fit: peak2 energy", "Fit: peak2 area", "Fit: R^2"]
        M = {k: np.full((self.nvf, self.nvb), np.nan) for k in keys}
        for i in range(self.nvf):
            for j in range(self.nvb):
                if not self.measured[i, j]:
                    continue
                res = fit_spectrum(self.e, self.spec_bs[i, j], npk, shape, lo, hi)
                if res is None:
                    continue
                p = res["peaks"]
                a1, c1, w1 = p[0]
                M["Fit: peak1 energy"][i, j] = c1
                M["Fit: peak1 height"][i, j] = a1
                M["Fit: peak1 FWHM (meV)"][i, j] = _fwhm(shape, w1) * 1000.0
                M["Fit: peak1 area"][i, j] = _area(shape, a1, w1)
                M["Fit: R^2"][i, j] = res["r2"]
                if len(p) > 1:
                    a2, c2, w2 = p[1]
                    M["Fit: peak2 energy"][i, j] = c2
                    M["Fit: peak2 area"][i, j] = _area(shape, a2, w2)
            self.status.setText(f"fitting ... {i+1}/{self.nvf}"); QtWidgets.QApplication.processEvents()
        self.fit = M; self.fit_cfg = (self._band_slice(), lo, hi, shape, npk)
        self.status.setText("fit done")
        self._refresh_map()

    def _current_map(self, name):
        if name in RAW_QUANTITIES:
            arr, div = self._raw_map(name), (name in _ENERGY_QUANTITIES)
        elif self.fit is None or name not in self.fit:
            return None, False
        else:
            arr, div = self.fit[name], (name in _ENERGY_QUANTITIES)
        arr = np.where(self.measured, np.asarray(arr, float), np.nan)   # blank unmeasured points
        return arr, div

    # ---- display mode + physical-units scaling ---------------------------
    def _mode(self):
        """'raw' (V_f,V_b) | 'gate' (u_n,u_D volts) | 'phys' (n, D/e0)."""
        if not self.calibrated:
            return "raw"
        return "phys" if self.physChk.isChecked() else "gate"

    def _scale_factors(self):
        """(s_n, s_D): converts u_n[V] -> n[10^12 cm^-2] and u_D[V] -> D/e0[V/nm]."""
        dF = max(float(self.dF.value()), 1e-6); eps = float(self.eps.value())
        return K_N * eps / dF, eps / (2.0 * dF)

    def _xy_scale(self):
        """(sx, sy) mapping (u_n,u_D) -> current map scene coords (1,1 unless physical)."""
        if self._mode() == "phys":
            return self._scale_factors()
        return 1.0, 1.0

    # ---- transform (grid indices -> display coords) ----------------------
    def _transform(self):
        """QTransform mapping ImageItem pixel (i,j) -> scene coords, per display mode.
        raw: axis-aligned (V_f,V_b); gate: sheared (u_n,u_D); phys: (u_n,u_D) rescaled to
        (n, D/e0). Uses per-axis steps dvf, dvb so a map with dvb != dvf renders correctly."""
        mode = self._mode()
        if mode == "raw":
            # pixel center (i+0.5, j+0.5) -> (vf[i], vb[j]) ; r is irrelevant to the raw grid
            dx = float(self.vf[0]) - 0.5 * self.dvf
            dy = float(self.vb[0]) - 0.5 * self.dvb
            return QtGui.QTransform(self.dvf, 0.0, 0.0, self.dvb, dx, dy), None
        r = float(self.rSpin.value())
        p, q = self.dvf, r * self.dvb            # u_n += dvf per i, += r*dvb per j
        a_n = float(self.vf[0]) + r * float(self.vb[0]) - float(self.nOrig.value())
        a_D = float(self.vf[0]) - r * float(self.vb[0]) - float(self.dOrig.value())
        # item pixel (i,j) center at (i+0.5, j+0.5); scene = (u_n, u_D)
        dx = a_n - 0.5 * (p + q)
        dy = a_D - 0.5 * (p - q)
        if mode == "phys":                       # scale x by s_n (->n), y by s_D (->D/e0)
            sN, sD = self._scale_factors()
            return (QtGui.QTransform(sN * p, sD * p, sN * q, -sD * q, sN * dx, sD * dy),
                    (p, q, a_n, a_D))
        return QtGui.QTransform(p, p, q, -q, dx, dy), (p, q, a_n, a_D)

    def _uncoords(self, i, j):
        """Physical (u_n, u_D) of grid point (i,j) — always, regardless of display mode."""
        r = float(self.rSpin.value())
        a_n = float(self.vf[0]) + r * float(self.vb[0]) - float(self.nOrig.value())
        a_D = float(self.vf[0]) - r * float(self.vb[0]) - float(self.dOrig.value())
        p, q = self.dvf, r * self.dvb
        return a_n + p * i + q * j, a_D + p * i - q * j

    def _display_xy(self, i, j):
        """Scene coords of grid point (i,j) in the CURRENT display mode (for the marker)."""
        mode = self._mode()
        if mode == "raw":
            return float(self.vf[i]), float(self.vb[j])
        un, uD = self._uncoords(i, j)
        if mode == "phys":
            sN, sD = self._scale_factors()
            return sN * un, sD * uD
        return un, uD

    def _apply_map_axes(self):
        mode = self._mode()
        if mode == "raw":
            xl = "V_front  (bottom gate, V)"
            yl = "V_back  (top gate, V)"
        elif mode == "phys":
            xl = "n   (10¹² cm⁻²)"
            yl = "D/ε₀   (V/nm)"
        else:
            xl = "u_n = V_f + r·V_b   (V,  ∝ density n)"
            yl = "u_D = V_f − r·V_b   (V,  ∝ displacement D)"
        _style_axes(self._mapPlot, bottom=xl, left=yl)

    def _apply_wf_axes(self):
        phys = self._mode() == "phys"
        for kind in ("density", "field"):
            plot = getattr(self, f"_wfPlot_{kind}", None)
            if plot is None:
                continue
            if kind == "density":
                yl = "n   (10¹² cm⁻²)" if phys else "u_n  (V, ∝ density)"
            else:
                yl = "D/ε₀   (V/nm)" if phys else "u_D  (V, ∝ D)"
            _style_axes(plot, left=yl)

    # ---- refresh ---------------------------------------------------------
    def _refresh_map(self):
        name = self.quantity.currentText()
        arr, diverging = self._current_map(name)
        if arr is None:
            self.status.setText(f"'{name}' needs a fit -> click 'Fit all points'")
            return
        tr, _ = self._transform()
        self._img.setImage(np.asarray(arr, float), autoLevels=False)
        self._img.setTransform(tr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if diverging:
                cmap, plo, phi = "plasma", 2, 98
            elif "FWHM" in name or "Linewidth" in name or "Cosine" in name:
                cmap, plo, phi = "magma", 1, 99
            else:
                cmap, plo, phi = "viridis", 1, 99
            # drive the LUT through the histogram gradient (it owns the image LUT)
            self._hist.gradient.setColorMap(_pg_cmap(cmap))
            lv = (float(np.nanpercentile(finite, plo)), float(np.nanpercentile(finite, phi)))
            self._img.setLevels(lv)
            self._hist.setLevels(*lv)
        mode = {"raw": "V_front, V_back", "gate": "u_n, u_D", "phys": "n, D/ε₀"}[self._mode()]
        self._mapPlot.setTitle(f"{name}   [{mode}]", **TITLE_CSS)
        self._mapPlot.autoRange()
        self._update_marker()

    def _update_marker(self):
        i, j = self.sel
        x, y = self._display_xy(i, j)
        self._marker.setData([x], [y])

    def _update_spatial_profile(self):
        if getattr(self, "_prof", None) is None:
            # PL weight per slit row (dark floor removed), averaged over the measured gate
            # points (an ongoing scan's unmeasured ones are all-NaN). Computed once per dataset
            # (band-independent); the ~0.5 GB temporary is transient.
            meas = self.cube[self.measured]                              # (nmeas,nwin,nwl)
            if meas.shape[0]:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)       # NaN rows of short frames
                    pl = meas - np.nanmedian(meas, axis=2, keepdims=True)
                    self._prof = np.nan_to_num(np.nanmean(np.clip(pl, 0, None).sum(axis=2), axis=0))
            else:
                self._prof = np.zeros(self.nwin)
            self._cProf.setData(self.spat_rows, self._prof)
        i0, i1 = self._band_slice()
        self._sync = True
        self._bandReg.setRegion((self.spat_rows[i0] - 0.5, self.spat_rows[i1] + 0.5))
        self._sync = False

    # ---- spectrum panel --------------------------------------------------
    def _show_spectrum(self, i, j):
        self.sel = (i, j)
        y = self.spec_bs[i, j]
        i0, i1 = self._band_slice()
        raw = np.nan_to_num(self.cube[i, j, i0:i1 + 1, :]).sum(0)[self.order]
        self._cRaw.setData(self.e, raw)
        self._cBs.setData(self.e, y)
        lo, hi = float(self.fitLo.value()), float(self.fitHi.value())
        self._fitReg.setRegion((lo, hi))
        res = fit_spectrum(self.e, y, int(self.npeaks.value()),
                           self.shapeCombo.currentText(), lo, hi)
        un, uD = self._uncoords(i, j)
        sN, sD = self._scale_factors()
        head = (f"Vf={self.vf[i]:+.2f} Vb={self.vb[j]:+.2f}  |  u_n={un:+.2f} u_D={uD:+.2f}"
                f"  |  n={sN*un:+.2f}×10¹² D/ε₀={sD*uD:+.3f} V/nm")
        if res is None:
            self._cFit.setData([], []); self._cP1.setData([], []); self._cP2.setData([], [])
            self._specPlot.setTitle(head + "  |  (no fit)", **TITLE_CSS)
        else:
            f = _peakfunc(res["shape"]); m = (self.e >= lo) & (self.e <= hi); ew = self.e[m]
            tot = np.full(ew.shape, res["offset"])
            comps = []
            for a, c, w in res["peaks"]:
                comp = res["offset"] + f(ew, a, c, w); comps.append(comp); tot = tot + f(ew, a, c, w)
            self._cFit.setData(ew, tot)
            self._cP1.setData(ew, comps[0])
            self._cP2.setData(ew, comps[1]) if len(comps) > 1 else self._cP2.setData([], [])
            p = res["peaks"]
            txt = f"E1={p[0][1]:.3f} eV FWHM1={_fwhm(res['shape'], p[0][2])*1000:.0f} meV"
            if len(p) > 1:
                txt += f"  E2={p[1][1]:.3f} eV"
            self._specPlot.setTitle(head + f"  |  {txt}  R²={res['r2']:.3f}", **TITLE_CSS)
        self._update_marker()

    # ---- events ----------------------------------------------------------
    def _on_map_click(self, ev):
        pos = ev.scenePos()
        if not self._mapPlot.sceneBoundingRect().contains(pos):
            return
        pt = self._img.mapFromScene(pos)                 # uses the item transform (both modes)
        i, j = int(np.floor(pt.x())), int(np.floor(pt.y()))
        if 0 <= i < self.nvf and 0 <= j < self.nvb:
            self._show_spectrum(i, j)

    def _on_region_changed(self):
        if self._sync:
            return
        a, b = self._bandReg.getRegion()
        c = int(round(0.5 * (a + b))); w = int(round(b - a))
        self._sync = True
        self.spatCenter.setValue(max(self.spat_rows[0], min(self.spat_rows[-1], c)))
        self.spatWidth.setValue(max(1, min(self.nwin, w)))
        self._sync = False
        self._on_band_changed()

    def _on_band_changed(self):
        if self._sync:
            return
        self._recompute_spectra()
        self._update_spatial_profile()
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_active_tab()

    def _on_bg_changed(self):
        self._recompute_spectra()
        self._refresh_map()
        self._show_spectrum(*self.sel)
        self._refresh_active_tab()

    def _on_fitcfg_changed(self):
        self.fit = None                                 # fitted maps now stale
        self._cosdist = None                            # cosine map uses the fit window
        self._refresh_map()
        self._show_spectrum(*self.sel)

    def _apply_display(self, *args):
        """Re-apply the display mode (called by the Calibrated / Physical-units checkboxes)."""
        if self.physChk.isChecked() and not self.calChk.isChecked():   # phys implies calibrated
            self.calChk.blockSignals(True); self.calChk.setChecked(True); self.calChk.blockSignals(False)
        self.calibrated = self.calChk.isChecked()
        self._update_phys_info()
        self._apply_map_axes()
        self._apply_wf_axes()
        self._refresh_map()
        self._refresh_active_tab()
        self._update_marker()
        self._update_slice_lines()
        self._update_title()

    def _on_units_param(self):
        """d_F / eps_hBN changed: rescale the physical axes without recomputing the scalar map."""
        self._update_phys_info()
        if self._mode() != "phys":
            return
        tr, _ = self._transform()
        self._img.setTransform(tr)
        self._apply_map_axes(); self._mapPlot.autoRange()
        self._apply_wf_axes(); self._refresh_active_tab()
        self._update_marker(); self._update_slice_lines()

    def _update_phys_info(self):
        sN, sD = self._scale_factors()
        self.physInfo.setText(f"n = {sN:.3f}·u_n ×10¹² cm⁻²    ·    D/ε₀ = {sD:.4f}·u_D V/nm")

    # ---- calibration + data pickers --------------------------------------
    def _load_calibration(self):
        start = str(CAL_DIR) if CAL_DIR.is_dir() else self._start_dir()
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration file", start, "Calibration (*.json);;All files (*)")
        if not fn:
            return
        try:
            cal = json.loads(Path(fn).read_text(encoding="utf-8"))
            r = float(cal["r"])
        except Exception as e:
            self.status.setText(f"calibration read failed: {e}")
            QtWidgets.QMessageBox.warning(self, "Load calibration",
                                          f"Could not read a calibration r from:\n{fn}\n\n{e}")
            return
        self.cal_meta = cal; self.cal_path = fn
        sig = cal.get("r_sigma")
        cdata = cal.get("data") or {}
        src = cdata.get("source_dir", "") or ""
        cnp = cal.get("cnp") or {}
        if "v_front" in cnp and "v_back" in cnp:            # optional CNP in the file
            self._cnp = (float(cnp["v_front"]), float(cnp["v_back"]))
        self.rSpin.setValue(r)                              # triggers _on_coords_changed -> refresh
        self.calChk.setChecked(True)                        # applying a calibration -> calibrated view
        if src and ("\\" in src or "/" in src):             # a real folder path was recorded
            prov = Path(src).name
        else:                                               # fall back to grid / point count
            grid = cdata.get("grid"); nm = cdata.get("n_measured")
            bits = []
            if grid:
                bits.append(f"{grid[0]}x{grid[1]}")
            if nm:
                bits.append(f"{nm} pts")
            prov = ", ".join(bits) or "?"
        info = f"r = {r:.3f}" + (f" ± {float(sig):.3f}" if sig is not None else "")
        info += f"   ·  {Path(fn).name}   ·  fitted on {prov}"
        self.calInfo.setText(info)
        self.status.setText(f"calibration loaded: r = {r:.3f}  (from {Path(fn).name})")

    def _load_data_folder(self):
        start = self._start_dir()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select data folder (.asc dual-gate sweep)", start)
        if not folder:
            return
        self._open_data_folder(folder)

    def _reload_data_folder(self):
        folder = self._source
        if not folder or not Path(folder).is_dir():
            QtWidgets.QMessageBox.information(
                self, "Reload", f"The current data folder is not available here:\n{folder}\n\n"
                                "Use 'Load data folder…' instead.")
            return
        rebuild = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        self._open_data_folder(folder, reload=True, rebuild=rebuild)

    def _open_data_folder(self, folder, reload=False, rebuild=False):
        """Load (or reload) a data folder. A reload keeps the view: integration band,
        selected gate point (by voltage), slice positions."""
        self.status.setText("reloading data ..." if reload else "loading data ...")
        QtWidgets.QApplication.processEvents()
        try:
            data, cache, cancelled = self._load_or_build_cube(folder, rebuild, update=reload)
        except Exception as e:
            self.status.setText(f"load failed: {e}")
            QtWidgets.QMessageBox.warning(self, "Load data folder",
                                          f"Could not load:\n{folder}\n\n{e}")
            return
        if cancelled or data is None:
            self.status.setText("data load cancelled"); return
        n_old = int(self.measured.sum())
        keep_band = reload and np.array_equal(np.asarray(data["spat_rows"], int), self.spat_rows)
        band = (self.spatCenter.value(), self.spatWidth.value())
        sel_v = (self.vf[self.sel[0]], self.vb[self.sel[1]])
        wf_idx = (self._dIdx, self._nIdx)
        self._prof = None
        if not reload:
            self._cnp = None
        self._set_arrays(data)
        # refresh control ranges for the new grid / spot window
        r0, r1 = int(self.spat_rows[0]), int(self.spat_rows[-1])
        self.spatCenter.blockSignals(True)
        self.spatCenter.setRange(r0, r1)
        self.spatCenter.setValue(int(band[0] if keep_band else self.spot_center))
        self.spatCenter.blockSignals(False)
        self.spatWidth.blockSignals(True)
        self.spatWidth.setRange(1, self.nwin)
        self.spatWidth.setValue(int(band[1]) if keep_band else min(13, self.nwin))
        self.spatWidth.blockSignals(False)
        if reload:
            self.sel = (int(np.argmin(np.abs(self.vf - sel_v[0]))),
                        int(np.argmin(np.abs(self.vb - sel_v[1]))))
            self._dIdx, self._nIdx = wf_idx
        self._source = str(Path(cache).parent)          # resolved data folder (after any auto-descend)
        self._refresh_all()
        n_meas = int(self.measured.sum())
        what = f"reloaded (+{n_meas - n_old} points)" if reload else "loaded"
        self.status.setText(f"{what} {Path(self._source).name}:  {self.nvf}x{self.nvb} grid, "
                            f"{n_meas} measured points  (cache: {Path(cache).name})")

    def _load_or_build_cube(self, folder, rebuild=False, update=False):
        return load_or_build_cube(folder, self, rebuild, update)

    def _start_dir(self):
        if self._source:
            return str(self._source)
        if self.source_dir and Path(self.source_dir).is_dir():
            return str(self.source_dir)
        return str(HERE)


def _wl_ok(wl):
    """A real wavelength axis: finite, positive and strictly monotonic. (Caches built
    before raw-image frames were handled took a column of COUNTS as the axis.)"""
    wl = np.asarray(wl, float)
    if wl.size < 2 or not np.all(np.isfinite(wl)) or np.any(wl <= 0):
        return False
    d = np.diff(wl)
    return bool(np.all(d > 0) or np.all(d < 0))


def _cache_is_stale(cache, files):
    """True if any .asc frame was written after the cache was built (ongoing scan),
    or the cache's wavelength axis is invalid."""
    if not files:
        return False
    try:
        with np.load(cache, allow_pickle=True) as z:
            if not _wl_ok(z["wl"]):
                return True
            built = float(z["built"]) if "built" in z.files else cache.stat().st_mtime
            if "n_files" in z.files and int(z["n_files"]) != len(files):
                return True
    except Exception:
        return True
    return max(f.stat().st_mtime for f in files) >= built


def load_or_build_cube(folder, parent=None, rebuild=False, update=False):
    """Return (data_dict, cache_path, cancelled). Loads an existing
    dual_gate_speccube_gtd1.npz next to the data, else builds + saves one.
    If .asc frames are newer than the cache (a scan still running), the cache is
    updated incrementally (only new frames are read); update=True always does that
    check (Reload). rebuild=True ignores the cache."""
    folder = Path(folder)
    cache = folder / CACHE_NAME
    files = sorted(folder.glob("*.asc"))
    prev = None
    if cache.exists():
        if not files or not (rebuild or update or _cache_is_stale(cache, files)):
            return _load_cache_dict(cache), cache, False      # cache-only folder, or up to date
        if not rebuild:
            try:
                prev = _load_cache_dict(cache)
            except Exception:
                prev = None                  # unreadable cache -> full rebuild
            if prev is not None and not _wl_ok(prev["wl"]):
                prev = None                  # bad wavelength axis -> full rebuild
    if not files:                            # picked a parent -> descend into the data subfolder
        subs = [d for d in sorted(folder.iterdir())
                if d.is_dir() and (next(d.glob("*.asc"), None) is not None
                                   or (d / CACHE_NAME).exists())]
        if len(subs) == 1:
            return load_or_build_cube(subs[0], parent, rebuild, update)
        if len(subs) > 1:
            names = "\n".join("   • " + s.name for s in subs)
            raise FileNotFoundError(f"No .asc files directly in:\n{folder}\n\n"
                                    f"It has several data subfolders — pick one:\n{names}")
        raise FileNotFoundError(f"no cache and no .asc files in:\n{folder}")
    try:
        import build_gtd1_cache as B
    except Exception as e:
        raise RuntimeError(f"build_gtd1_cache not importable (needed to build a cache):\n{e!r}")
    what = "Updating" if prev is not None else "Building"
    dlg = QtWidgets.QProgressDialog(
        f"{what} spectral cache from {len(files)} .asc files\n{folder.name}",
        "Cancel", 0, len(files), parent)
    dlg.setWindowTitle("Building cache"); dlg.setWindowModality(QtCore.Qt.WindowModal)
    dlg.setMinimumDuration(0); dlg.setValue(0)
    state = {"cancel": False}

    def cb(done, total):
        if dlg.maximum() != total:
            dlg.setMaximum(total)
        if done % 10 == 0 or done == total:
            dlg.setValue(done)
            QtWidgets.QApplication.processEvents()
        if dlg.wasCanceled():
            state["cancel"] = True
            raise RuntimeError("cancelled")

    wl_ref = None
    while True:
        try:
            data = B.build_cube(folder, cache_path=cache, progress=cb, prev=prev, wl_ref=wl_ref)
            break
        except RuntimeError:
            if state["cancel"]:
                dlg.close(); return None, None, True
            raise
        except LookupError as e:             # raw-image frames, no wavelength reference found
            dlg.hide()
            QtWidgets.QMessageBox.information(
                parent, "Wavelength axis needed",
                f"{e}\n\nPick an .asc frame that HAS a wavelength column, taken with the same "
                "grating and center wavelength (e.g. from an earlier scan).")
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent, "Reference .asc with wavelength column", str(folder.parent),
                "Andor ASCII (*.asc);;All files (*)")
            if not fn:
                dlg.close(); return None, None, True
            wl_ref = fn
            dlg.show()
    dlg.setValue(dlg.maximum()); dlg.close()
    return data, cache, False


def _load_cache_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _resolve_data_folder(start):
    """Return a folder that holds the standard cache or .asc files, descending one
    level if the given folder only contains a single such data subfolder; else None."""
    p = Path(start)
    if not p.is_dir():
        return None
    if (p / CACHE_NAME).exists() or next(p.glob("*.asc"), None) is not None:
        return p
    subs = [d for d in sorted(p.iterdir()) if d.is_dir()
            and ((d / CACHE_NAME).exists() or next(d.glob("*.asc"), None) is not None)]
    return subs[0] if len(subs) == 1 else None


def main():
    app = QtWidgets.QApplication(sys.argv[:1])            # keep Qt from parsing our folder arg
    app.setFont(QtGui.QFont("Arial", GUI_FONT_PT))        # larger GUI font

    # startup dataset: folder on the command line > calibration\ reference cache > ask
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    data = src = None
    if arg:
        folder = _resolve_data_folder(arg)
        if folder is None:
            print(f"[warn] no .asc/cache found under: {arg}")
        else:
            data, cache, _ = load_or_build_cube(folder)
            src = str(folder) if data is not None else None
    if data is None and CACHE.exists():
        data = _load_cache_dict(CACHE)
        src = str(data["source_dir"]) if "source_dir" in data else None
    while data is None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select data folder (.asc dual-gate sweep)", str(HERE.parent))
        if not folder:
            sys.exit(0)
        try:
            data, cache, _ = load_or_build_cube(folder)
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Load data folder", f"Could not load:\n{folder}\n\n{e}")
            continue
        if data is not None:
            src = str(Path(cache).parent)

    # keep the default column-major image axis order (array[i,j] -> item x=i, y=j),
    # which is what the ImageItem QTransform and the click-inverse below assume.
    win = NDMapViewer(data, source=src)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
