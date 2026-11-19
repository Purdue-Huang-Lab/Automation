"""Magnification / spot-size calculator for the SPAD23 and SPAD93 detection path.

Optical path::

    sample -> objective (M_nom x, NA) -> tube lens (f_TL) -> SPAD array

* Objective focal length  f_obj = f_TL,design / M_nom   (e.g. 200 mm / 100 = 2 mm)
* Actual magnification    M     = f_TL / f_obj
* Length conversion       L_sample = L_detector / M

Spot model (everything computed on the sample plane, then scaled by M):

* Excitation spot: diffraction-limited focus of a uniformly filled pupil, i.e. an
  Airy pattern  I(r) = [2 J1(v)/v]^2,  v = 2 pi NA r / lambda
  (FWHM = 0.514 lambda/NA, first zero at 0.61 lambda/NA).
* Image on the detector ("laser" spot), two imaging models:
    - emission (incoherent, e.g. PL): excitation spot convolved with the detection
      PSF at the emission wavelength, then magnified by M;
    - reflected laser: the focused spot re-imaged by the same objective = M x Airy.
* "Diffused" spot: the emitting profile is the excitation spot convolved with a
  Gaussian of standard deviation sigma_D (sigma_D^2 = 2 D tau for exciton diffusion),
  imaged through the detection PSF.

All profiles are radially symmetric and computed exactly from their OTFs with a
Hankel transform: the Airy intensity has the "chinese-hat" OTF, the Gaussian a
Gaussian OTF.  Each pixel's signal is the intensity integrated over its active
disk (diameter set in the panel, default = PIXEL_DIAMETER_UM of the geometry file).

Click a detector panel to move the beam center.

Run from the Automation folder:
    python -m spad23.spad_magnification_viewer
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import j0, j1

from PyQt5 import QtCore, QtWidgets  # PyQt5 must come before pyqtgraph
import pyqtgraph as pg

from spad23 import spad23_geometry as g23
from spad93 import spad93_geometry as g93

pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k", antialias=True)


# ----------------------------------------------------------------------------- defaults
DEFAULTS = {
    "wl_exc_nm": 532.0,
    "wl_em_nm": 532.0,
    "na": 0.9,
    "m_nom": 100.0,
    "f_design_mm": 200.0,   # tube lens the objective's nominal M refers to (Nikon/Mitutoyo/Leica 200, Olympus 180, Zeiss 165)
    "f_tl_mm": 200.0,
    "sigma_d_um": 1.0,      # diffusion broadening on the sample (Gaussian sigma)
}

N_NU = 1000         # spatial-frequency samples for the Hankel transform
N_R = 800           # radial samples of each profile
N_DISK = 400        # sample points per pixel disk
IMG_N = 241         # sample-plane image size (pixels per side)

CMAP = pg.colormap.get("inferno")
LUT = CMAP.getLookupTable(nPts=256)

COL_LASER = (31, 119, 180)
COL_DIFF = (214, 39, 40)
COL_PSF = (120, 120, 120)


@dataclass(frozen=True)
class Detector:
    name: str
    x: np.ndarray            # pixel centers on the detector (um)
    y: np.ndarray
    pitch_x: float
    pitch_y: float
    pixel_d: float
    center_pixel: int


DETECTORS: Dict[str, Detector] = {
    "SPAD23": Detector("SPAD23", g23.PIXEL_X_UM.copy(), g23.PIXEL_Y_UM.copy(),
                       g23.PITCH_X_UM, g23.PITCH_Y_UM, g23.PIXEL_DIAMETER_UM, g23.CENTER_PIXEL),
    "SPAD93": Detector("SPAD93", g93.PIXEL_X_UM.copy(), g93.PIXEL_Y_UM.copy(),
                       g93.PITCH_X_UM, g93.PITCH_Y_UM, g93.PIXEL_DIAMETER_UM, g93.CENTER_PIXEL),
}


# ----------------------------------------------------------------------------- optics
def chat(nu: np.ndarray, nu_c: float) -> np.ndarray:
    """OTF of an Airy intensity pattern (cutoff nu_c = 2 NA / lambda), normalized to 1 at 0."""
    s = np.clip(nu / nu_c, 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))


def airy(r: np.ndarray, wl_um: float, na: float) -> np.ndarray:
    """Airy intensity normalized to 1 at r = 0."""
    v = 2.0 * np.pi * na * np.asarray(r, dtype=np.float64) / wl_um
    out = np.ones_like(v)
    nz = v > 1e-9
    out[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return out


def radial_profile(wl_exc: float, wl_em: float, na: float, sigma_d: float,
                   with_det_psf: bool, r: np.ndarray) -> np.ndarray:
    """Radial intensity on the sample plane (um^-2), normalized to unit total power.

    OTF = chat(exc) * [chat(det)] * exp(-2 pi^2 sigma_d^2 nu^2), then
    I(r) = 2 pi int OTF(nu) J0(2 pi nu r) nu dnu.
    """
    nu_c = 2.0 * na / wl_exc
    nu_max = nu_c
    if with_det_psf:
        nu_max = min(nu_max, 2.0 * na / wl_em)
    if sigma_d > 0:
        nu_max = min(nu_max, 1.25 / sigma_d)   # Gaussian OTF < 1e-13 beyond this
    nu = np.linspace(0.0, nu_max, N_NU)
    otf = chat(nu, nu_c)
    if with_det_psf:
        otf = otf * chat(nu, 2.0 * na / wl_em)
    if sigma_d > 0:
        otf = otf * np.exp(-2.0 * np.pi ** 2 * sigma_d ** 2 * nu ** 2)
    w = np.full(N_NU, nu[1] - nu[0])
    w[0] = w[-1] = 0.5 * w[0]
    kern = j0(2.0 * np.pi * np.outer(r, nu))
    return 2.0 * np.pi * kern @ (otf * nu * w)


def width_at(r: np.ndarray, prof: np.ndarray, level: float = 0.5) -> float:
    """Full width where the profile first drops to ``level`` x its peak (0.5 -> FWHM)."""
    thr = level * prof[0]
    below = np.flatnonzero(prof < thr)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    r0, r1, p0, p1 = r[i - 1], r[i], prof[i - 1], prof[i]
    return 2.0 * float(r0 + (thr - p0) * (r1 - r0) / (p1 - p0))


def unit_disk_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sunflower pattern: n equal-area points on the unit disk."""
    k = np.arange(n) + 0.5
    rad = np.sqrt(k / n)
    th = k * np.pi * (3.0 - np.sqrt(5.0))
    return rad * np.cos(th), rad * np.sin(th)


DISK_U, DISK_V = unit_disk_points(N_DISK)


# ----------------------------------------------------------------------------- widgets
class PixelPlot(pg.PlotItem):
    """Detector plane: pixels colored by collected signal, FWHM rings, beam center."""

    def __init__(self, title: str, ring_color):
        super().__init__(title=title)
        self.setAspectLocked(True)
        self.invertY(True)   # row 0 at the top, as in the SPAD GUIs
        self.setLabel("bottom", "x on detector (µm)")
        self.setLabel("left", "y on detector (µm)")
        self.pixels = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen("k", width=1))
        self.addItem(self.pixels)
        self.labels = []
        self.ring = pg.PlotCurveItem(pen=pg.mkPen(ring_color, width=2.5, style=QtCore.Qt.DashLine))
        self.ring_e2 = pg.PlotCurveItem(pen=pg.mkPen(ring_color, width=1.5, style=QtCore.Qt.DotLine))
        self.addItem(self.ring)
        self.addItem(self.ring_e2)
        self.center = pg.ScatterPlotItem(symbol="+", size=22, pen=pg.mkPen((0, 170, 255), width=3), brush=None)
        self.addItem(self.center)

    def set_layout(self, det: Detector, d_um: float):
        self.pixels.setData(x=det.x, y=det.y, size=d_um)
        for t in self.labels:
            self.removeItem(t)
        self.labels = []
        if det.x.size <= 30:
            for i, (xx, yy) in enumerate(zip(det.x, det.y)):
                t = pg.TextItem(str(i), color=(140, 140, 140), anchor=(0.5, 0.5))
                t.setPos(xx, yy)
                t.setZValue(5)
                self.addItem(t)
                self.labels.append(t)
        pad = 0.8 * det.pitch_x
        self.setRange(xRange=(det.x.min() - pad, det.x.max() + pad),
                      yRange=(det.y.min() - pad, det.y.max() + pad), padding=0)

    def update_values(self, vals: np.ndarray, x0: float, y0: float, fwhm: float, w_e2: float):
        vmax = float(vals.max()) if vals.size and vals.max() > 0 else 1.0
        idx = np.clip((vals / vmax * 255).astype(int), 0, 255)
        self.pixels.setBrush([pg.mkBrush(*LUT[i]) for i in idx])
        th = np.linspace(0, 2 * np.pi, 181)
        self.ring.setData(x0 + 0.5 * fwhm * np.cos(th), y0 + 0.5 * fwhm * np.sin(th))
        self.ring_e2.setData(x0 + 0.5 * w_e2 * np.cos(th), y0 + 0.5 * w_e2 * np.sin(th))
        self.center.setData([x0], [y0])


class MagnificationWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.det = DETECTORS["SPAD23"]
        self.x0 = float(self.det.x[self.det.center_pixel])
        self.y0 = float(self.det.y[self.det.center_pixel])
        self._build_ui()
        self._on_detector_changed()

    # ------------------------------------------------------------------ UI
    def _spin(self, val, lo, hi, step, dec, suffix=""):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(self.recompute)
        return s

    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("Optics")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        self.cb_det = QtWidgets.QComboBox()
        self.cb_det.addItems(list(DETECTORS))
        self.cb_det.currentTextChanged.connect(self._on_detector_changed)
        self.sp_wl_exc = self._spin(DEFAULTS["wl_exc_nm"], 200, 2000, 1, 1, " nm")
        self.sp_wl_em = self._spin(DEFAULTS["wl_em_nm"], 200, 2000, 1, 1, " nm")
        self.sp_na = self._spin(DEFAULTS["na"], 0.05, 1.45, 0.05, 3)
        self.sp_mnom = self._spin(DEFAULTS["m_nom"], 1, 200, 1, 1, " ×")
        self.sp_fdes = self._spin(DEFAULTS["f_design_mm"], 50, 400, 5, 1, " mm")
        self.sp_ftl = self._spin(DEFAULTS["f_tl_mm"], 10, 2000, 5, 1, " mm")
        self.sp_sig = self._spin(DEFAULTS["sigma_d_um"], 0.0, 50, 0.1, 2, " µm")
        self.sp_pixd = self._spin(self.det.pixel_d, 1, 23, 0.5, 1, " µm")   # 23 um = nearest-neighbour distance
        self.cb_model = QtWidgets.QComboBox()
        self.cb_model.addItems(["Emission (incoherent): spot ⊗ PSF", "Reflected laser: M × spot"])
        self.cb_model.currentIndexChanged.connect(self.recompute)
        self.cb_sample = QtWidgets.QComboBox()
        self.cb_sample.addItems(["Excitation spot", "Diffused emission"])
        self.cb_sample.currentIndexChanged.connect(self.recompute)
        self.btn_center = QtWidgets.QPushButton("Center on center pixel")
        self.btn_center.clicked.connect(self._center_beam)

        rows = [
            ("Detector", self.cb_det),
            ("Excitation λ", self.sp_wl_exc),
            ("Emission / detection λ", self.sp_wl_em),
            ("Objective NA", self.sp_na),
            ("Objective nominal M", self.sp_mnom),
            ("Design tube-lens f", self.sp_fdes),
            ("Tube-lens f", self.sp_ftl),
            ("Diffusion σ_D = √(2Dτ)", self.sp_sig),
            ("Pixel active diameter", self.sp_pixd),
            ("Laser image model", self.cb_model),
            ("Sample-plane image", self.cb_sample),
        ]
        for i, (lab, w) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(lab), i, 0)
            grid.addWidget(w, i, 1)
        grid.addWidget(self.btn_center, len(rows), 0, 1, 2)
        tip = QtWidgets.QLabel("Click a detector panel to move the beam center.")
        tip.setStyleSheet("color: gray")
        grid.addWidget(tip, len(rows) + 1, 0, 1, 2)
        left.addWidget(box)

        res_box = QtWidgets.QGroupBox("Results")
        res_lay = QtWidgets.QVBoxLayout(res_box)
        self.lbl_res = QtWidgets.QLabel()
        self.lbl_res.setTextFormat(QtCore.Qt.RichText)
        self.lbl_res.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_res.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.lbl_res.setWordWrap(True)
        res_lay.addWidget(self.lbl_res)
        left.addWidget(res_box, 1)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setFixedWidth(440)
        outer.addWidget(scroll)

        self.glw = pg.GraphicsLayoutWidget()
        outer.addWidget(self.glw, 1)

        # sample plane
        self.p_sample = self.glw.addPlot(row=0, col=0, title="Sample plane")
        self.p_sample.setAspectLocked(True)
        self.p_sample.invertY(True)
        self.p_sample.setLabel("bottom", "x on sample (µm)")
        self.p_sample.setLabel("left", "y on sample (µm)")
        self.img = pg.ImageItem()
        self.img.setLookupTable(LUT)
        self.p_sample.addItem(self.img)
        self.foot = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen((0, 220, 255), width=1.2), brush=None)
        self.p_sample.addItem(self.foot)
        self.s_ring_exc = pg.PlotCurveItem(pen=pg.mkPen("w", width=2))
        self.s_ring_las = pg.PlotCurveItem(pen=pg.mkPen(COL_LASER, width=2, style=QtCore.Qt.DashLine))
        self.s_ring_dif = pg.PlotCurveItem(pen=pg.mkPen(COL_DIFF, width=2, style=QtCore.Qt.DashLine))
        for it in (self.s_ring_exc, self.s_ring_las, self.s_ring_dif):
            self.p_sample.addItem(it)

        # detector planes
        self.p_las = PixelPlot("Detector: laser spot", COL_LASER)
        self.p_dif = PixelPlot("Detector: diffused spot", COL_DIFF)
        self.glw.addItem(self.p_las, row=0, col=1)
        self.glw.addItem(self.p_dif, row=1, col=0)
        self.glw.scene().sigMouseClicked.connect(self._on_click)

        # radial profiles
        self.p_rad = self.glw.addPlot(row=1, col=1, title="Radial profiles (normalized)")
        self.p_rad.setLabel("bottom", "radius on detector (µm)")
        self.p_rad.setLabel("left", "I / I(0)")
        self.p_rad.showAxis("top")
        self.ax_top = self.p_rad.getAxis("top")
        self.ax_top.enableAutoSIPrefix(False)
        self.ax_top.setLabel("radius on sample (µm)")
        self.p_rad.showGrid(x=True, y=True, alpha=0.3)
        self.p_rad.addLegend(offset=(-5, 5), labelTextColor="k", brush=pg.mkBrush(255, 255, 255, 200))
        self.c_exc = self.p_rad.plot(pen=pg.mkPen(COL_PSF, width=1.5, style=QtCore.Qt.DotLine), name="excitation × M")
        self.c_las = self.p_rad.plot(pen=pg.mkPen(COL_LASER, width=2), name="laser image")
        self.c_dif = self.p_rad.plot(pen=pg.mkPen(COL_DIFF, width=2), name="diffused image")
        self.s_las = pg.ScatterPlotItem(size=8, pen=pg.mkPen("k", width=0.5), brush=pg.mkBrush(*COL_LASER), name="pixels (area avg)")
        self.s_dif = pg.ScatterPlotItem(size=8, pen=pg.mkPen("k", width=0.5), brush=pg.mkBrush(*COL_DIFF))
        self.p_rad.addItem(self.s_las)
        self.p_rad.addItem(self.s_dif)
        self.pitch_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine)) for _ in range(2)]
        for ln in self.pitch_lines:
            self.p_rad.addItem(ln)

        lay = self.glw.ci.layout
        for i in range(2):
            lay.setColumnStretchFactor(i, 1)
            lay.setRowStretchFactor(i, 1)
        for p in (self.p_sample, self.p_las, self.p_dif, self.p_rad):
            p.setMinimumSize(200, 200)
            p.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            for ax in ("bottom", "left", "top"):
                p.getAxis(ax).setPen(pg.mkPen("k"))
                p.getAxis(ax).setTextPen(pg.mkPen("k"))

    # ------------------------------------------------------------------ events
    def _on_detector_changed(self, *_):
        self.det = DETECTORS[self.cb_det.currentText()]
        self.sp_pixd.blockSignals(True)
        self.sp_pixd.setValue(self.det.pixel_d)
        self.sp_pixd.blockSignals(False)
        self._center_beam()

    def _center_beam(self):
        self.x0 = float(self.det.x[self.det.center_pixel])
        self.y0 = float(self.det.y[self.det.center_pixel])
        self.recompute()

    def _on_click(self, ev):
        for p in (self.p_las, self.p_dif):
            if p.vb.sceneBoundingRect().contains(ev.scenePos()):
                pt = p.vb.mapSceneToView(ev.scenePos())
                self.x0, self.y0 = float(pt.x()), float(pt.y())
                self.recompute()
                return

    # ------------------------------------------------------------------ physics + drawing
    def recompute(self, *_):
        det = self.det
        wl_e = self.sp_wl_exc.value() * 1e-3
        wl_m = self.sp_wl_em.value() * 1e-3
        na = self.sp_na.value()
        f_obj = self.sp_fdes.value() / self.sp_mnom.value()          # mm
        M = self.sp_ftl.value() / f_obj
        sig = self.sp_sig.value()
        d_pix = self.sp_pixd.value()
        emission = self.cb_model.currentIndex() == 0

        # sample-plane radial grid: cover the back-projected array and the spot tails
        dx, dy = det.x - self.x0, det.y - self.y0
        r_arr = (np.hypot(dx, dy).max() + 0.5 * d_pix) / M
        r_max = max(1.2 * r_arr, 6.0 * wl_e / na + 5.0 * sig, 8.0 * wl_m / na)
        r = np.linspace(0.0, r_max, N_R)

        I_exc = airy(r, wl_e, na)                                     # excitation spot (peak-normalized)
        I_las = radial_profile(wl_e, wl_m, na, 0.0, emission, r)      # laser image (unit power)
        I_emit = radial_profile(wl_e, wl_m, na, sig, False, r)        # diffused emitter on the sample
        I_dif = radial_profile(wl_e, wl_m, na, sig, True, r)          # diffused image (unit power)

        fw_exc = width_at(r, I_exc)
        fw_las = width_at(r, I_las)
        fw_emit = width_at(r, I_emit)
        fw_dif = width_at(r, I_dif)
        we2_las = width_at(r, I_las, np.exp(-2))
        we2_dif = width_at(r, I_dif, np.exp(-2))

        # pixel signals: integrate over each pixel's active disk (sample-plane units)
        a = 0.5 * d_pix / M
        px = (dx[:, None] + 0.5 * d_pix * DISK_U[None, :]) / M
        py = (dy[:, None] + 0.5 * d_pix * DISK_V[None, :]) / M
        rp = np.hypot(px, py)
        area = np.pi * a * a
        P_las = np.interp(rp, r, I_las, right=0.0).mean(axis=1) * area
        P_dif = np.interp(rp, r, I_dif, right=0.0).mean(axis=1) * area
        r_pix_det = np.hypot(dx, dy)

        # --- detector panels
        for p in (self.p_las, self.p_dif):
            if p.pixels.data.size != det.x.size or not np.isclose(p.pixels.data["size"][0], d_pix):
                p.set_layout(det, d_pix)
        self.p_las.update_values(P_las, self.x0, self.y0, M * fw_las, M * we2_las)
        self.p_dif.update_values(P_dif, self.x0, self.y0, M * fw_dif, M * we2_dif)
        self.p_las.setTitle(f"Detector, laser: FWHM {M * fw_las:.1f} µm", size="10pt")
        self.p_dif.setTitle(f"Detector, diffused: FWHM {M * fw_dif:.1f} µm", size="10pt")

        # --- sample plane image
        sample_diff = self.cb_sample.currentIndex() == 1
        half = max(1.15 * r_arr, 1.3 * (fw_dif if sample_diff else fw_las), 2.5 * wl_e / na)
        cx, cy = self.x0 / M, self.y0 / M
        ax = np.linspace(-half, half, IMG_N)
        XX, YY = np.meshgrid(ax, ax)
        RR = np.hypot(XX, YY)
        im = np.interp(RR, r, I_emit if sample_diff else I_exc, right=0.0)
        im = im / im.max()
        self.img.setImage(im, levels=(0, 1))
        self.img.setRect(QtCore.QRectF(cx - half, cy - half, 2 * half, 2 * half))
        self.foot.setData(x=det.x / M, y=det.y / M, size=d_pix / M)
        th = np.linspace(0, 2 * np.pi, 181)
        for item, fw in ((self.s_ring_exc, fw_exc), (self.s_ring_las, fw_las), (self.s_ring_dif, fw_dif)):
            item.setData(cx + 0.5 * fw * np.cos(th), cy + 0.5 * fw * np.sin(th))
        self.p_sample.setRange(xRange=(cx - half, cx + half), yRange=(cy - half, cy + half), padding=0)
        self.p_sample.setTitle("Sample: " + ("diffused emitter" if sample_diff else "excitation spot"), size="10pt")

        # --- radial plot (detector units, sample units on the top axis)
        rd = M * r
        self.c_exc.setData(rd, I_exc / I_exc[0])
        self.c_las.setData(rd, I_las / I_las[0])
        self.c_dif.setData(rd, I_dif / I_dif[0])
        self.s_las.setData(r_pix_det, P_las / area / I_las[0])
        self.s_dif.setData(r_pix_det, P_dif / area / I_dif[0])
        self.pitch_lines[0].setValue(min(det.pitch_x, det.pitch_y))
        self.pitch_lines[1].setValue(2 * min(det.pitch_x, det.pitch_y))
        x_hi = max(1.05 * r_pix_det.max() + 0.5 * d_pix, 1.5 * M * fw_dif)
        self.p_rad.setRange(xRange=(0, min(x_hi, rd[-1])), yRange=(-0.02, 1.05), padding=0)
        self.ax_top.setScale(1.0 / M)

        # --- results text
        na_img = na / M
        nm_per_um = 1e3 / M
        bright = int(np.argmax(P_las))
        x_span = det.x.max() - det.x.min() + d_pix
        y_span = det.y.max() - det.y.min() + d_pix

        def row(k, v):
            return f"<tr><td>{k}</td><td align='right'>&nbsp;&nbsp;<b>{v}</b></td></tr>"

        html = "<table cellspacing='1'>"
        html += "<tr><td colspan=2><u>Magnification</u></td></tr>"
        html += row("f_objective = f_design / M_nom", f"{f_obj:.3f} mm")
        html += row("M = f_TL / f_obj", f"{M:.2f} ×")
        html += row("image-side NA = NA / M", f"{na_img:.4f}")
        html += "<tr><td colspan=2><br><u>Length conversion</u></td></tr>"
        html += row("1 µm on detector", f"{nm_per_um:.2f} nm on sample")
        html += row("1 µm on sample", f"{M:.1f} µm on detector")
        html += row(f"pitch x {det.pitch_x:g} µm", f"{det.pitch_x * nm_per_um:.1f} nm")
        html += row(f"pitch y {det.pitch_y:g} µm", f"{det.pitch_y * nm_per_um:.1f} nm")
        html += row(f"pixel Ø {d_pix:g} µm", f"{d_pix * nm_per_um:.1f} nm")
        html += row("array field of view", f"{x_span / M:.2f} × {y_span / M:.2f} µm")
        html += "<tr><td colspan=2><br><u>Spot FWHM: sample → detector</u></td></tr>"
        html += row("excitation (0.514 λ/NA)", f"{fw_exc * 1e3:.0f} nm → {M * fw_exc:.1f} µm")
        html += row("Airy 1st zero radius 0.61 λ/NA", f"{0.61 * wl_e / na * 1e3:.0f} nm → {M * 0.61 * wl_e / na:.1f} µm")
        html += row("laser image" + (" (spot ⊗ PSF)" if emission else " (M × spot)"),
                    f"{fw_las * 1e3:.0f} nm → {M * fw_las:.1f} µm")
        html += row(f"diffused emitter (σ_D = {sig:g} µm)", f"{fw_emit * 1e3:.0f} nm")
        html += row("diffused image (emitter ⊗ PSF)", f"{fw_dif * 1e3:.0f} nm → {M * fw_dif:.1f} µm")
        html += row("laser image FWHM / pitch", f"{M * fw_las / det.pitch_x:.2f} pixels")
        html += row("diffused image FWHM / pitch", f"{M * fw_dif / det.pitch_x:.2f} pixels")
        html += "<tr><td colspan=2><br><u>Collected fraction of image power</u></td></tr>"
        html += row(f"laser: brightest pixel ({bright})", f"{100 * P_las[bright]:.1f} %")
        html += row("laser: all pixels", f"{100 * P_las.sum():.1f} %")
        html += row(f"diffused: brightest pixel ({int(np.argmax(P_dif))})", f"{100 * P_dif.max():.1f} %")
        html += row("diffused: all pixels", f"{100 * P_dif.sum():.1f} %")
        html += "<tr><td colspan=2><br><u>Beam center</u></td></tr>"
        html += row("on detector", f"({self.x0:.1f}, {self.y0:.1f}) µm")
        html += row("on sample", f"({self.x0 / M:.3f}, {self.y0 / M:.3f}) µm")
        html += "</table>"
        html += ("<p style='color:gray'>Sample plane: white ring = excitation FWHM, blue dashed = laser image "
                 "FWHM / M, red dashed = diffused image FWHM / M, cyan circles = pixels projected onto the "
                 "sample (÷ M).<br>Detector panels: dashed ring = FWHM, dotted = 1/e² diameter; pixel color = "
                 "signal integrated over the active disk.<br>Radial plot: dots = each pixel's area-averaged "
                 "intensity vs. its center radius; grey lines = 1 and 2 pitches.</p>")
        self.lbl_res.setText(html)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPAD magnification / spot size")
        self.setCentralWidget(MagnificationWidget(self))
        self.resize(1500, 950)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
