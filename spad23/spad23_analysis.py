"""Gaussian fits and wavelength calibration for the SPAD23 (numpy/scipy only, no Qt).

Copied from ``spad93/spad93_analysis.py``.  The 2D fits and the radial fit
are ports of ``spad23/spad23_trpl_viewer.py`` (same coordinates: x along a row,
y = 19.92 um * row), with moment-based start values, bounded widths and a
quality estimate.

Spectral mode is the SPAD93 one transposed: the wavelength axis is y (one row
per wavelength, ``ROW_Y_UM``) and the spatial profile of a wavelength is the
x-profile of its row.  ``CalibPoint.x_um`` and ``WavelengthCalibration`` keep
their SPAD93 field names but hold the position along that dispersion axis (y).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.special import erfc, erfcx

    _HAS_SCIPY = True
except Exception:
    curve_fit = None  # type: ignore
    erfc = erfcx = None  # type: ignore
    _HAS_SCIPY = False

from spad23.spad23_geometry import (
    LINE_PITCH_UM,
    LINE_SPAN_UM,
    N_ROWS,
    PITCH_X_UM,
    PITCH_Y_UM,
    PIXEL_X_UM,
    ROW_IS_SHORT,
    ROW_Y_UM,
    SITE_PITCH_UM,
    column_weights,
    row_pixels,
)

# Mirror flags kept so calibration files have the same fields as SPAD93 ones (not supported here).
FLIP_X = False
FLIP_Y = False

HAS_SCIPY = _HAS_SCIPY
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
MIN_FIT_PIXELS_2D = 6
MIN_SNR = 3.0

ProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], bool]


# ======================================================================
# Display helpers (ported from spad23_counts_gui)
# ======================================================================
def auto_rate_scale(values_mcps: np.ndarray) -> Tuple[str, float]:
    vals = np.asarray(values_mcps, dtype=np.float64)
    max_mcps = float(np.nanmax(vals)) if vals.size and np.isfinite(vals).any() else 0.0
    if max_mcps < 0.001:
        return "cps", 1e6
    if max_mcps < 0.1:
        return "kcps", 1e3
    return "Mcps", 1.0


def format_rate(rate_mcps: float) -> str:
    if rate_mcps < 0.001:
        return f"{rate_mcps * 1e6:.1f} cps"
    if rate_mcps < 0.1:
        return f"{rate_mcps * 1e3:.3f} kcps"
    return f"{rate_mcps:.3f} Mcps"


def frame_background(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Background level of a frame: 10th percentile of the enabled pixels."""
    vals = np.asarray(frame, dtype=np.float64)
    if mask is not None:
        vals = vals[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 10)) if vals.size else 0.0


def rebin_time(counts: np.ndarray, time_ns: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sum ``factor`` consecutive time bins (a short last group is summed too)."""
    factor = max(1, int(factor))
    if factor == 1 or counts.shape[0] == 0:
        return counts, time_ns
    n = counts.shape[0]
    n_groups = int(np.ceil(n / factor))
    pad = n_groups * factor - n
    c = counts
    if pad:
        c = np.concatenate([counts, np.zeros((pad, counts.shape[1]), dtype=counts.dtype)], axis=0)
    summed = c.reshape(n_groups, factor, counts.shape[1]).sum(axis=1)
    return summed, time_ns[::factor][:n_groups]


# ======================================================================
# Time-zero alignment on each pixel's first peak (post-processing, optional)
# ======================================================================
# The vendor channel alignment can leave pixels off by anything from a fraction of
# a ns to hundreds of ns (some even wrap around the window), and with lasers whose
# traces have two peaks (an IRF-like laser leak, then the signal a fixed delay
# later) some pixels end up aligned on the second peak.  The alignment here:
#   1. coarse: circular cross-correlation of every trace with a template built from
#      the data itself (both peaks), which finds large offsets whichever peak is higher;
#   2. fine: the rising edge of the first peak of the template (half-height crossing,
#      interpolated between bins) in every pixel; the flat top of the peak is a poor marker;
#   3. every pixel is shifted (whole raw bins, circularly) so its edge lands on the
#      median edge time of the largest group of pixels, so most pixels barely move.
T0_SMOOTH_NS = 0.3  # boxcar used before locating peaks and edges
T0_SEARCH_NS = 1.5  # the first peak is searched within +- this of where the coarse step puts it
T0_EDGE_LEVEL = 0.5  # fraction of the first peak's height that marks time zero (on the rising edge)
T0_MIN_SNR = 10.0  # the first peak must be this many noise sigmas above the pixel's background
T0_SECOND_MIN_REL = 0.15  # a second peak counts if at least this fraction of the template maximum
T0_PEAK_SEP_NS = 3.0  # ... and more than this from the maximum
T0_CLUSTER_NS = 1.0  # pixels within this of each other form the group that sets the target

T0_EDGE = 0  # rising edge of the first peak found: aligned on it
T0_COARSE = 1  # first peak too weak for an edge: aligned by the cross-correlation only
T0_NONE = 2  # no usable signal: left unchanged
T0_STATUS_TEXT = {
    T0_EDGE: "aligned on the rising edge of its first peak",
    T0_COARSE: "first peak too weak: aligned by cross-correlation only",
    T0_NONE: "no usable signal: not shifted",
}


@dataclass
class TimeZeroAlignment:
    counts: np.ndarray  # [bins, pixels] after the shifts (same dtype as the input)
    shift_bins: np.ndarray  # per pixel, raw bins (+ = later)
    status: np.ndarray  # per pixel, T0_* code
    edge_ns: np.ndarray  # per pixel, time zero (first-peak edge) before the shift; nan if none
    correlation: np.ndarray  # per pixel, normalized cross-correlation with the template (0..1)
    bin_ns: float
    target_ns: float  # where every first-peak edge is put
    second_delay_ns: float  # delay of the template's second peak after the first; nan if none
    message: str

    def pixel_text(self, pix: int) -> str:
        st = int(self.status[pix])
        s = self.shift_bins[pix] * self.bin_ns
        if st == T0_EDGE:
            return f"time zero: first-peak edge at {self.edge_ns[pix]:.2f} ns, shifted {s:+.2f} ns"
        if st == T0_COARSE:
            return f"{T0_STATUS_TEXT[st]}, shifted {s:+.2f} ns"
        return T0_STATUS_TEXT[st]

    def summary(self) -> Dict[str, object]:
        """Plain-type summary for file headers."""
        def pix(code):
            return [int(p) for p in np.flatnonzero(self.status == code)]
        return {
            "method": f"first-peak rising edge at {T0_EDGE_LEVEL:.0%} height, coarse step by cross-correlation",
            "target_ns": round(self.target_ns, 4),
            "second_peak_delay_ns": None if not np.isfinite(self.second_delay_ns) else round(self.second_delay_ns, 3),
            "shift_ns": {str(p): round(float(self.shift_bins[p] * self.bin_ns), 4)
                         for p in range(self.shift_bins.size) if self.shift_bins[p]},
            "coarse_only_pixels": pix(T0_COARSE),
            "not_aligned_pixels": pix(T0_NONE),
        }


def _circular_boxcar(y: np.ndarray, k: int) -> np.ndarray:
    """Centered k-bin running mean along axis 0, wrapping around the window."""
    if k <= 1:
        return y.astype(np.float64)
    ker = np.zeros(y.shape[0])
    ker[:k] = 1.0 / k
    ker = np.roll(ker, -(k // 2))
    return np.real(np.fft.ifft(np.fft.fft(y, axis=0) * np.fft.fft(ker)[:, None], axis=0))


def _template_peaks(tmpl: np.ndarray, dt: float, m: int) -> List[int]:
    """Real peaks of the template (circular): local maxima within +-0.5 ns, at least
    T0_SECOND_MIN_REL of the maximum ``m``, more than T0_PEAK_SEP_NS away from it (closer
    features, e.g. a small pre-pulse, belong to the same pulse) and separated from it by a
    dip below 70 % of the lower of the two (a decaying tail or noise on it is not a peak)."""
    n = tmpl.size
    w = max(int(round(0.5 / dt)), 1)
    sep = max(int(round(T0_PEAK_SEP_NS / dt)), 2)
    local_max = np.ones(n, dtype=bool)
    for s in range(1, w + 1):
        local_max &= (tmpl >= np.roll(tmpl, s)) & (tmpl >= np.roll(tmpl, -s))
    cand = np.flatnonzero(local_max & (tmpl >= T0_SECOND_MIN_REL * tmpl[m]))
    out = [m]
    for j in cand:
        j = int(j)
        if min((j - m) % n, (m - j) % n) <= sep:
            continue
        lo, hi = (j, m) if (m - j) % n < n // 2 else (m, j)  # the shorter way round
        seg = tmpl[(lo + np.arange((hi - lo) % n + 1)) % n]
        if seg.min() <= 0.7 * min(tmpl[j], tmpl[m]):
            out.append(j)
    return out


def _circ_dist(a, b, period: float):
    """Signed circular difference a - b in (-period/2, period/2]."""
    return (np.asarray(a) - b + period / 2.0) % period - period / 2.0


def align_first_peaks(counts: np.ndarray, time_ns: np.ndarray, edge_level: float = T0_EDGE_LEVEL) -> TimeZeroAlignment:
    """Align every pixel's time zero on the rising edge of its first peak (see the notes above).

    Shifts are whole raw bins and circular: the window is taken to be one laser
    period, so counts pushed off one end come back at the other (as the TDC
    timestamps themselves wrap).
    """
    c = np.asarray(counts)
    t = np.asarray(time_ns, dtype=np.float64)
    n, n_pix = c.shape
    if n < 8:
        raise ValueError("Too few time bins.")
    dt = float(np.median(np.diff(t)))
    period = n * dt
    k = max(1, int(round(T0_SMOOTH_NS / dt))) | 1
    bg = np.median(c, axis=0).astype(np.float64)
    y = _circular_boxcar(c.astype(np.float64) - bg, k)
    norm = np.sqrt((y**2).sum(axis=0))
    yn = y / np.where(norm > 0, norm, 1.0)

    # 1. coarse: template from the data (start: traces rolled to their maxima, median), refined twice
    tmpl = np.median(np.stack([np.roll(yn[:, p], -int(np.argmax(yn[:, p]))) for p in range(n_pix)], axis=1), axis=1)
    F = np.fft.fft(yn, axis=0)
    lag = np.zeros(n_pix, dtype=np.int64)
    corr = np.zeros(n_pix)
    for _ in range(3):
        tn = np.sqrt((tmpl**2).sum())
        xc = np.real(np.fft.ifft(F * np.conj(np.fft.fft(tmpl))[:, None], axis=0))
        lag = np.argmax(xc, axis=0)
        corr = xc[lag, np.arange(n_pix)] / (tn if tn > 0 else 1.0)
        good = corr > 0.3
        if not good.any():
            break
        tmpl = np.mean(np.stack([np.roll(yn[:, p], -int(lag[p])) for p in np.flatnonzero(good)], axis=1), axis=1)

    # the template's first peak: the earliest real peak before its maximum (within half the window)
    m = int(np.argmax(tmpl))
    peaks = _template_peaks(tmpl, dt, m)
    before = [(int((m - j) % n), j) for j in peaks if 0 < (m - j) % n < n // 2]
    first = max(before)[1] if before else m
    after = sorted(int((j - first) % n) for j in peaks if j != first)
    second_delay = after[0] * dt if after and after[0] < n // 2 else float("nan")

    # 2. fine: rising edge of every pixel's first peak
    win = max(int(round(T0_SEARCH_NS / dt)), 2)
    status = np.full(n_pix, T0_NONE, dtype=np.int64)
    edge = np.full(n_pix, np.nan)
    pos = np.full(n_pix, np.nan)  # where the pixel's time zero is (edge, or the coarse estimate)
    for p in range(n_pix):
        if corr[p] <= 0.3:
            continue
        center = (first + int(lag[p])) % n
        idx = (center + np.arange(-win, win + 1)) % n
        seg = y[idx, p]
        i = int(np.argmax(seg))
        h = float(seg[i])
        pos[p] = center * dt  # coarse: the template's first peak, moved by the lag
        status[p] = T0_COARSE
        if h < T0_MIN_SNR * np.sqrt(max(bg[p], 1.0) / k):
            continue
        g, steps = int(idx[i]), 0
        while y[(g - 1) % n, p] >= edge_level * h and steps < 4 * win:
            g -= 1
            steps += 1
        a, b = float(y[(g - 1) % n, p]), float(y[g % n, p])
        frac = (edge_level * h - a) / (b - a) if b != a else 1.0
        edge[p] = ((g - 1 + frac) * dt) % period
        pos[p] = edge[p]
        status[p] = T0_EDGE
    if np.any(status == T0_COARSE):
        # put coarse-only pixels on the same footing: the template's own edge relative to its first peak
        tm = float(tmpl[first])
        g, steps = first, 0
        while tmpl[(g - 1) % n] >= edge_level * tm and steps < 4 * win:
            g -= 1
            steps += 1
        a, b = float(tmpl[(g - 1) % n]), float(tmpl[g % n])
        offset = (g - 1 + ((edge_level * tm - a) / (b - a) if b != a else 1.0) - first) * dt
        coarse = status == T0_COARSE
        pos[coarse] = (pos[coarse] + offset) % period

    # 3. target: median edge of the largest group (circular distances)
    have = np.flatnonzero(np.isfinite(pos))
    shift = np.zeros(n_pix, dtype=np.int64)
    out = c.copy()
    if have.size == 0:
        return TimeZeroAlignment(out, shift, status, edge, corr, dt, float("nan"), second_delay,
                                 "Time-zero alignment: no pixel has a usable first peak; nothing shifted.")
    ref = pos[have[status[have] == T0_EDGE]] if np.any(status[have] == T0_EDGE) else pos[have]
    counts_near = [int(np.sum(np.abs(_circ_dist(ref, v, period)) <= T0_CLUSTER_NS)) for v in ref]
    v0 = float(ref[int(np.argmax(counts_near))])
    group = ref[np.abs(_circ_dist(ref, v0, period)) <= T0_CLUSTER_NS]
    target = float((v0 + np.median(_circ_dist(group, v0, period))) % period)
    for p in have:
        shift[p] = int(np.round(_circ_dist(target, pos[p], period) / dt))
        if shift[p]:
            out[:, p] = np.roll(c[:, p], shift[p])

    moved = shift != 0
    s_ns = shift * dt
    big = np.abs(s_ns) > T0_CLUSTER_NS
    msg = (f"Time zero: first-peak edge of every pixel put at {target:.2f} ns"
           + (f" (second peak {second_delay:.1f} ns later)" if np.isfinite(second_delay) else "")
           + f"; {int(moved.sum())} px shifted")
    if moved.any():
        msg += f" ({int(big.sum())} by more than {T0_CLUSTER_NS:g} ns, up to {np.abs(s_ns).max():.1f} ns)"
    n_coarse, n_none = int(np.sum(status == T0_COARSE)), int(np.sum(status == T0_NONE))
    if n_coarse:
        msg += f"; {n_coarse} px by cross-correlation only (weak first peak)"
    if n_none:
        msg += f"; {n_none} px without usable signal left alone"
    return TimeZeroAlignment(out, shift, status, edge, corr, dt, target, second_delay, msg + ".")


# with the alignment on, the views show t = raw time - this, so time zero is at 0 ns at every point
T0_REF_BINS = 5  # default reference frame: this many (shown) bins after time zero


def t0_display_offset(time_ns: np.ndarray, target_ns: Optional[float]) -> Optional[float]:
    """Raw time (ns) of the bin holding time zero ``target_ns`` (from bin 0, as TimeZeroAlignment.target_ns).

    Subtracted from the time axis it puts time zero at exactly 0 ns on the bin grid (the
    first-peak edge lies within half a bin of 0).  None without a time zero.
    """
    t = np.asarray(time_ns, dtype=np.float64)
    if t.size < 2 or target_ns is None or not np.isfinite(target_ns):
        return None
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    return float(t[int(np.round(float(target_ns) / dt)) % t.size])


T0_PEAK_SMOOTH_NS = 0.5  # boxcar over which the brightest pixel's maximum is taken (the time zero shown)


def peak_display_offset(counts: np.ndarray, time_ns: np.ndarray, excluded=()) -> Optional[float]:
    """Raw time (ns) of the signal peak: the maximum of the brightest pixel (each pixel minus its median,
    smoothed over T0_PEAK_SMOOTH_NS).  With the time-zero alignment on, the views put t = 0 here (the
    pixels themselves stay aligned on their first-peak edge).  None without signal."""
    c = np.asarray(counts, dtype=np.float64)
    t = np.asarray(time_ns, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 4 or t.size != c.shape[0]:
        return None
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    k = max(1, int(round(T0_PEAK_SMOOTH_NS / dt))) | 1
    y = c - np.median(c, axis=0)
    if k > 1:
        cs = np.cumsum(np.vstack([np.zeros((1, y.shape[1])), y]), axis=0)
        y = (cs[k:] - cs[:-k]) / k  # row i = mean of rows i .. i+k-1 (centre i + k // 2)
    ex = [int(p) for p in excluded if 0 <= int(p) < y.shape[1]]
    if ex:
        y[:, ex] = -np.inf
    p = int(np.argmax(y.max(axis=0)))
    i = int(np.argmax(y[:, p]))
    if not y[i, p] > 0:
        return None
    return float(t[i + k // 2])


def t0_ref_index(view_time_ns: np.ndarray, n_bins: int = T0_REF_BINS) -> int:
    """Index of the default reference frame: ``n_bins`` bins after the first bin at or after t = 0."""
    t = np.asarray(view_time_ns, dtype=np.float64)
    if t.size == 0:
        return 0
    i0 = int(np.searchsorted(t, -1e-9, side="left"))
    return int(min(i0 + int(n_bins), t.size - 1))


# ======================================================================
# 2D Gaussian (spatial) fits
# ======================================================================
@dataclass
class FitResult:
    popt: np.ndarray  # [A, x0, y0, sx, sy, offset]
    z_fit: np.ndarray
    sigma_eq: float
    snr: float = float("inf")
    at_bound: bool = False

    @property
    def quality_ok(self) -> bool:
        return self.snr >= MIN_SNR


@dataclass
class RadialFitResult:
    popt: np.ndarray  # [A, sigma, offset]
    sigma_eq: float
    at_bound: bool = False
    snr: float = float("inf")

    @property
    def quality_ok(self) -> bool:
        return self.snr >= MIN_SNR


def gauss2d(coords, A, x0, y0, sx, sy, offset):
    xg, yg = coords
    return A * np.exp(-(((xg - x0) ** 2) / (2.0 * sx**2) + ((yg - y0) ** 2) / (2.0 * sy**2))) + offset


def _curve_fit(*args, **kwargs):
    """``curve_fit`` without the covariance / overflow warnings (bad frames are expected)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return curve_fit(*args, **kwargs)  # type: ignore[misc]


def _curve_fit_fixed(f, xdata, ydata, p0, lower, upper, fixed: Optional[Dict[int, float]] = None,
                     tied: Optional[Dict[int, Tuple[int, float]]] = None, **kwargs):
    """``_curve_fit`` with constrained parameters.

    ``fixed`` = {index: value} holds parameters at those values; ``tied`` =
    {index: (other index, factor)} sets ``p[index] = factor * p[other]`` (the other
    one must be free or fixed).  Returns the full popt and a full-size pcov
    (fixed parameters have zero variance, tied ones follow the parameter they are tied to).
    """
    n = len(p0)
    fixed, tied = fixed or {}, tied or {}
    if not fixed and not tied:
        popt, pcov = _curve_fit(f, xdata, ydata, p0=_clip_into(p0, lower, upper), bounds=(lower, upper), **kwargs)
        return np.asarray(popt, dtype=np.float64), np.asarray(pcov, dtype=np.float64)
    free = [i for i in range(n) if i not in fixed and i not in tied]

    def full(pf):
        p = np.empty(n)
        p[free] = pf
        for i, v in fixed.items():
            p[i] = v
        for i, (j, fac) in tied.items():
            p[i] = fac * p[j]
        return p

    def model(xd, *pf):
        return f(xd, *full(pf))

    lo = [lower[i] for i in free]
    hi = [upper[i] for i in free]
    pf, cf = _curve_fit(model, xdata, ydata, p0=_clip_into([p0[i] for i in free], lo, hi), bounds=(lo, hi), **kwargs)
    popt = full(np.asarray(pf, dtype=np.float64))
    jac = np.zeros((n, len(free)))  # d p_full / d p_free
    jac[free, np.arange(len(free))] = 1.0
    for i, (j, fac) in tied.items():
        jac[i] = fac * jac[j]
    pcov = jac @ np.asarray(cf, dtype=np.float64) @ jac.T
    return popt, pcov


def _unbound(lower, upper, *constrained):
    """Copies of the bounds with the fixed/tied parameters unbounded (so they never count as 'at a bound')."""
    lower, upper = list(lower), list(upper)
    for d in constrained:
        for i in d or {}:
            lower[i], upper[i] = -np.inf, np.inf
    return lower, upper


def _significance(amp: float, pcov: np.ndarray, dof: int) -> float:
    """Amplitude over its standard error (inf when there are no degrees of freedom).

    Unlike amplitude/rms-residual this stays small for fits to pure noise.
    """
    if dof <= 0:
        return float("inf")
    var = float(np.asarray(pcov)[0, 0]) if np.size(pcov) else float("nan")
    if not np.isfinite(var) or var <= 0:
        return 0.0
    return float(amp) / float(np.sqrt(var))


def _clip_into(p0: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> List[float]:
    out = []
    for v, lo, hi in zip(p0, lower, upper):
        v = float(v)
        if np.isfinite(lo) and np.isfinite(hi):
            span = hi - lo
            v = min(max(v, lo + 1e-6 * span), hi - 1e-6 * span)
        elif np.isfinite(lo):
            v = max(v, lo + 1e-9)
        elif np.isfinite(hi):
            v = min(v, hi - 1e-9)
        out.append(v)
    return out


def _hit_bound(p: Sequence[float], lower: Sequence[float], upper: Sequence[float], rel: float = 1e-3) -> bool:
    for v, lo, hi in zip(p, lower, upper):
        if np.isfinite(lo) and np.isfinite(hi):
            tol = rel * (hi - lo)
            if v - lo <= tol or hi - v <= tol:
                return True
    return False


def _bounds_2d(x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float):
    span = max(float(np.ptp(x)), float(np.ptp(y)), pitch_x, pitch_y)
    lower = [0.0, float(np.min(x) - pitch_x), float(np.min(y) - pitch_y), 0.5, 0.5, -np.inf]
    upper = [np.inf, float(np.max(x) + pitch_x), float(np.max(y) + pitch_y), 2.0 * span, 2.0 * span, np.inf]
    return lower, upper


def moments_guess_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, pitch_x: float = PITCH_X_UM, pitch_y: float = PITCH_Y_UM,
                     bg: Optional[float] = None) -> List[float]:
    bg = float(np.percentile(z, 10)) if bg is None else float(bg)
    w = np.clip(z - bg, 0.0, None)
    total = float(w.sum())
    amp = max(float(np.max(z) - bg), 1e-6)
    if total <= 0:
        return [amp, float(np.mean(x)), float(np.mean(y)), 20.0, 20.0, bg]
    x0 = float((w * x).sum() / total)
    y0 = float((w * y).sum() / total)
    sx = float(np.sqrt(max((w * (x - x0) ** 2).sum() / total, (0.5 * pitch_x) ** 2)))
    sy = float(np.sqrt(max((w * (y - y0) ** 2).sum() / total, (0.5 * pitch_y) ** 2)))
    return [amp, x0, y0, sx, sy, bg]


def _make_fit_2d(x, y, z, popt, pcov, lower, upper, n_free: int = 6) -> FitResult:
    popt = np.asarray(popt, dtype=np.float64)
    z_fit = gauss2d((x, y), *popt)
    sigma_eq = float(np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2.0))
    return FitResult(
        popt=popt,
        z_fit=np.asarray(z_fit, dtype=np.float64),
        sigma_eq=sigma_eq,
        snr=_significance(popt[0], pcov, int(z.size) - n_free),
        at_bound=_hit_bound(popt, lower, upper),
    )


def fit_gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pitch_x: float = PITCH_X_UM,
    pitch_y: float = PITCH_Y_UM,
    p0: Optional[Sequence[float]] = None,
    maxfev: int = 20000,
    fix_offset: Optional[float] = None,
) -> Optional[FitResult]:
    """2D Gaussian with a free offset, or with the offset held at ``fix_offset``."""
    if not _HAS_SCIPY or z.size < MIN_FIT_PIXELS_2D:
        return None
    if np.allclose(z, z[0]):
        return None
    lower, upper = _bounds_2d(x, y, pitch_x, pitch_y)
    guess = list(moments_guess_2d(x, y, z, pitch_x, pitch_y, bg=fix_offset) if p0 is None else p0)
    # Fit z/zs (see fit_gaussian_1d); the A and offset bounds are 0 / +-inf, so they need no scaling.
    zs = max(float(np.max(np.abs(z))), 1e-300)
    fixed = None if fix_offset is None else {5: float(fix_offset) / zs}
    if fixed:
        guess[5] = float(fix_offset)
    guess[0] /= zs
    guess[5] /= zs
    try:
        popt, pcov = _curve_fit_fixed(gauss2d, (x, y), z / zs, guess, lower, upper, fixed, maxfev=maxfev)
    except Exception:
        return None
    popt[[0, 5]] *= zs
    pcov = pcov * zs**2  # only pcov[0, 0] is used (amplitude significance)
    return _make_fit_2d(x, y, z, popt, pcov, lower, upper, 6 - len(fixed or {}))


def fit_gaussian_2d_warm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prev_popt: Optional[np.ndarray],
    pitch_x: float = PITCH_X_UM,
    pitch_y: float = PITCH_Y_UM,
    fix_offset: Optional[float] = None,
) -> Optional[FitResult]:
    """Live fit: start from the previous frame's result, fall back to a full fit."""
    if prev_popt is not None:
        fit = fit_gaussian_2d(x, y, z, pitch_x, pitch_y, p0=prev_popt, maxfev=2000, fix_offset=fix_offset)
        if fit is not None and fit.quality_ok:
            return fit
    return fit_gaussian_2d(x, y, z, pitch_x, pitch_y, fix_offset=fix_offset)


def fit_gaussian_2d_constrained_next(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prev_popt: np.ndarray,
    tightness: float = 1.0,
    pitch_x: float = PITCH_X_UM,
    pitch_y: float = PITCH_Y_UM,
    fix_offset: Optional[float] = None,
) -> Optional[FitResult]:
    """Fit the next TRPL frame starting from the previous frame's fit, with small bounds."""
    if not _HAS_SCIPY:
        return None
    if np.allclose(z, z[0]):
        return None

    A_prev, x0_prev, y0_prev, sx_prev, sy_prev, off_prev = [float(v) for v in prev_popt]
    sx_prev = max(sx_prev, 1e-3)
    sy_prev = max(sy_prev, 1e-3)

    dA = max(abs(A_prev) * 0.18 * tightness, 1e-3)
    dx = max(pitch_x * 0.18 * tightness, 0.8)
    dy = max(pitch_y * 0.18 * tightness, 0.8)
    dsx = max(abs(sx_prev) * 0.18 * tightness, 0.5)
    dsy = max(abs(sy_prev) * 0.18 * tightness, 0.5)
    doff = max(abs(off_prev) * 0.25 * tightness, 0.5)

    lower = [
        max(0.0, A_prev - dA),
        x0_prev - dx,
        y0_prev - dy,
        max(1e-3, sx_prev - dsx),
        max(1e-3, sy_prev - dsy),
        off_prev - doff,
    ]
    upper = [A_prev + dA, x0_prev + dx, y0_prev + dy, sx_prev + dsx, sy_prev + dsy, off_prev + doff]
    fixed = None if fix_offset is None else {5: float(fix_offset)}
    lower, upper = _unbound(lower, upper, fixed)
    p0 = [A_prev, x0_prev, y0_prev, sx_prev, sy_prev, off_prev if fixed is None else float(fix_offset)]
    try:
        popt, pcov = _curve_fit_fixed(gauss2d, (x, y), z, p0, lower, upper, fixed, maxfev=20000)
    except Exception:
        return None
    return _make_fit_2d(x, y, z, popt, pcov, lower, upper, 6 - len(fixed or {}))


def fit_radial_gaussian_1d(
    r: np.ndarray,
    z: np.ndarray,
    p0: Optional[np.ndarray] = None,
    sigma_guess: Optional[float] = None,
    fix_offset: Optional[float] = None,
) -> Optional[RadialFitResult]:
    if not _HAS_SCIPY:
        return None
    if r.size < 3 or z.size < 3:
        return None
    if np.allclose(z, z[0]):
        return None

    r = r.astype(np.float64)
    z = z.astype(np.float64)

    def gauss1d_r(rg, A, sigma, offset):
        return A * np.exp(-(rg**2) / (2.0 * sigma**2)) + offset

    sigma_max = max(4.0 * float(np.max(r)), 10.0)
    if p0 is not None and np.asarray(p0).size == 3:
        p0_use = [float(p0[0]), max(float(p0[1]), 1e-3), float(p0[2])]
    else:
        A0 = max(float(np.max(z) - np.min(z)), 1e-6)
        sigma0 = max(float(sigma_guess) if sigma_guess is not None else float(np.std(r)), 1e-3)
        p0_use = [A0, sigma0, float(np.min(z))]

    lower = [0.0, 1e-3, -np.inf]
    upper = [np.inf, sigma_max, np.inf]
    fixed = None if fix_offset is None else {2: float(fix_offset)}
    if fixed:
        p0_use[2] = float(fix_offset)
    try:
        popt, pcov = _curve_fit_fixed(gauss1d_r, r, z, p0_use, lower, upper, fixed, maxfev=20000)
    except Exception:
        return None
    return RadialFitResult(
        popt=popt,
        sigma_eq=float(abs(popt[1])),
        at_bound=_hit_bound(popt, lower, upper),
        snr=_significance(popt[0], pcov, int(r.size) - 3 + len(fixed or {})),
    )


# ======================================================================
# Double 2D Gaussian: two round spots on one shared offset
# ======================================================================
MIN_FIT_PIXELS_DOUBLE = 10  # 9 parameters
MIN_DOUBLE_FRACTION = 0.05  # weaker / brighter integrated intensity for "resolved"
_DOUBLE_SCALED = [0, 4, 8]  # amplitudes and offset: fitted as z / max|z|
_DOUBLE_SWAP = [4, 5, 6, 7, 0, 1, 2, 3, 8]


def gauss2d_double(coords, A1, x1, y1, s1, A2, x2, y2, s2, offset):
    xg, yg = coords
    return (
        A1 * np.exp(-((xg - x1) ** 2 + (yg - y1) ** 2) / (2.0 * s1**2))
        + A2 * np.exp(-((xg - x2) ** 2 + (yg - y2) ** 2) / (2.0 * s2**2))
        + offset
    )


@dataclass
class DoubleFitResult:
    popt: np.ndarray  # [A1, x1, y1, s1, A2, x2, y2, s2, offset]
    z_fit: np.ndarray
    rss: float
    snr1: float = float("inf")
    snr2: float = float("inf")
    at_bound: bool = False

    def component(self, i: int) -> Tuple[float, float, float, float]:
        """``(A, x0, y0, sigma)`` of component ``i`` (0 or 1)."""
        k = 4 * int(i)
        return tuple(float(v) for v in self.popt[k : k + 4])  # type: ignore[return-value]

    @property
    def offset(self) -> float:
        return float(self.popt[8])

    @property
    def separation(self) -> float:
        return float(np.hypot(self.popt[1] - self.popt[5], self.popt[2] - self.popt[6]))

    @property
    def quality_ok(self) -> bool:
        return min(self.snr1, self.snr2) >= MIN_SNR

    @property
    def weights(self) -> Tuple[float, float]:
        """Integrated intensity of each component (2 pi A sigma^2)."""
        return (float(2 * np.pi * self.popt[0] * self.popt[3] ** 2), float(2 * np.pi * self.popt[4] * self.popt[7] ** 2))

    @property
    def resolved(self) -> bool:
        """Both amplitudes significant, the weaker spot >= MIN_DOUBLE_FRACTION of the brighter
        one's light, and the centers at least (sigma1 + sigma2) / 2 apart."""
        w1, w2 = self.weights
        return (self.quality_ok and min(w1, w2) >= MIN_DOUBLE_FRACTION * max(w1, w2)
                and self.separation >= 0.5 * (abs(self.popt[3]) + abs(self.popt[7])))


def _bounds_double(x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float):
    span = max(float(np.ptp(x)), float(np.ptp(y)), pitch_x, pitch_y)
    s_lo, s_hi = 0.25 * min(pitch_x, pitch_y), 2.0 * span
    x_lo, x_hi = float(np.min(x) - pitch_x), float(np.max(x) + pitch_x)
    y_lo, y_hi = float(np.min(y) - pitch_y), float(np.max(y) + pitch_y)
    lower = [0.0, x_lo, y_lo, s_lo, 0.0, x_lo, y_lo, s_lo, -np.inf]
    upper = [np.inf, x_hi, y_hi, s_hi, np.inf, x_hi, y_hi, s_hi, np.inf]
    return lower, upper


def _local_maxima(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float) -> np.ndarray:
    """Indices of pixels at least as bright as all neighbours within ``radius``, brightest first."""
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    near = (d2 <= radius**2) & (d2 > 0)
    idx = np.array([i for i in range(z.size) if np.all(z[i] >= z[near[i]])], dtype=np.int64)
    return idx[np.argsort(-z[idx])] if idx.size else idx


def _double_seeds(x, y, z, pitch_x, pitch_y, single: Optional[FitResult],
                  fix_offset: Optional[float] = None) -> List[List[float]]:
    """Start points: the two brightest local maxima, and the single-Gaussian fit split in two."""
    bg = float(np.percentile(z, 10)) if fix_offset is None else float(fix_offset)
    pitch = max(pitch_x, pitch_y)
    seeds: List[List[float]] = []
    peaks = _local_maxima(x, y, z, 1.15 * pitch)
    if peaks.size >= 2:
        i, j = int(peaks[0]), int(peaks[1])
        seeds.append([z[i] - bg, x[i], y[i], 0.6 * pitch, z[j] - bg, x[j], y[j], 0.6 * pitch, bg])
    if single is not None:
        A, x0, y0, sx, sy, off = (float(v) for v in single.popt)
        s = max(single.sigma_eq, 0.5 * pitch)
        w = np.clip(z - off, 0.0, None)
        if w.sum() > 0:
            cov = np.cov(np.vstack([x, y]), aweights=w)
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, int(np.argmax(evals))]
        else:
            v = np.array([1.0, 0.0])
        d = max(0.8 * s, 0.5 * pitch)
        seeds.append([0.7 * A, x0 + d * v[0], y0 + d * v[1], 0.7 * s,
                      0.7 * A, x0 - d * v[0], y0 - d * v[1], 0.7 * s, off])
        # a weak second spot: the pixel the single fit misses most
        k = int(np.argmax(z - single.z_fit))
        if np.hypot(x[k] - x0, y[k] - y0) > 0.5 * pitch:
            seeds.append([A, x0, y0, s, max(float(z[k] - single.z_fit[k]), 1e-9), x[k], y[k], 0.6 * pitch, off])
    return seeds


def _order_like(popt: np.ndarray, pcov: np.ndarray, ref: np.ndarray):
    """Swap the components so their centers best match those of ``ref``."""
    keep = np.hypot(popt[1] - ref[1], popt[2] - ref[2]) + np.hypot(popt[5] - ref[5], popt[6] - ref[6])
    swap = np.hypot(popt[1] - ref[5], popt[2] - ref[6]) + np.hypot(popt[5] - ref[1], popt[6] - ref[2])
    if swap < keep:
        return popt[_DOUBLE_SWAP], pcov[np.ix_(_DOUBLE_SWAP, _DOUBLE_SWAP)]
    return popt, pcov


def _double_fixed(fix_offset: Optional[float], centers: Optional[Sequence[float]]) -> Dict[int, float]:
    """Fixed parameters of a double fit: the offset and/or the centers (x1, y1, x2, y2)."""
    fixed: Dict[int, float] = {}
    if fix_offset is not None:
        fixed[8] = float(fix_offset)
    if centers is not None:
        fixed.update(zip((1, 2, 5, 6), (float(v) for v in centers)))
    return fixed


def _double_tied(amp_ratio: Optional[float], share_sigma: bool) -> Dict[int, Tuple[int, float]]:
    """Ties of a double fit: A2 = amp_ratio * A1 and/or sigma2 = sigma1."""
    tied: Dict[int, Tuple[int, float]] = {}
    if amp_ratio is not None:
        tied[4] = (0, float(amp_ratio))
    if share_sigma:
        tied[7] = (3, 1.0)
    return tied


def _fit_double_once(x, y, z, guess, lower, upper, maxfev, fixed: Optional[Dict[int, float]] = None,
                     tied: Optional[Dict[int, Tuple[int, float]]] = None):
    """One bounded fit of z/max|z|; ``fixed`` = {index: value in data units}, ``tied`` see _double_tied.

    Returns (popt, pcov, rss) in data units, or None.
    """
    zs = max(float(np.max(np.abs(z))), 1e-300)
    scale = np.ones(9)
    scale[_DOUBLE_SCALED] = zs
    g = np.asarray(guess, dtype=np.float64).copy()
    for i, v in (fixed or {}).items():
        g[i] = v
    g = g / scale
    lo = np.asarray(lower, dtype=np.float64) / scale
    hi = np.asarray(upper, dtype=np.float64) / scale
    fixed = {i: v / scale[i] for i, v in fixed.items()} if fixed else None
    try:
        popt, pcov = _curve_fit_fixed(gauss2d_double, (x, y), z / zs, list(g), list(lo), list(hi), fixed,
                                      tied, maxfev=maxfev)
    except Exception:
        return None
    popt = popt * scale
    pcov = pcov * np.outer(scale, scale)
    rss = float(np.sum((z - gauss2d_double((x, y), *popt)) ** 2))
    return popt, pcov, rss


def _make_double(x, y, z, popt, pcov, rss, lower, upper, n_free: int = 9) -> DoubleFitResult:
    dof = int(z.size) - n_free
    return DoubleFitResult(
        popt=popt,
        z_fit=np.asarray(gauss2d_double((x, y), *popt), dtype=np.float64),
        rss=rss,
        snr1=_significance(popt[0], pcov, dof),
        snr2=_significance(popt[4], pcov[4:, 4:], dof),
        at_bound=_hit_bound(popt, lower, upper),
    )


def fit_gaussian_2d_double(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pitch_x: float = PITCH_X_UM,
    pitch_y: float = PITCH_Y_UM,
    p0: Optional[Sequence[float]] = None,
    keep_order_of: Optional[Sequence[float]] = None,
    maxfev: int = 800,
    fix_offset: Optional[float] = None,
    centers: Optional[Sequence[float]] = None,
    amp_ratio: Optional[float] = None,
    share_sigma: bool = False,
) -> Optional[DoubleFitResult]:
    """Fit two round 2D Gaussians plus one offset; the lowest-RSS fit of several start points wins.

    ``centers`` = (x1, y1, x2, y2) holds both centers fixed (only A, sigma and the offset are
    fitted); ``fix_offset`` holds the offset; ``amp_ratio`` ties A2 = amp_ratio * A1 and
    ``share_sigma`` ties sigma2 = sigma1.

    ``maxfev`` is per start point and kept small: a well-posed fit converges in a
    few hundred evaluations, while noise-only frames would otherwise run to the
    limit on every start point (seconds per frame in the live view).
    Components are ordered brighter first (A * sigma^2), or, with ``keep_order_of``
    (a previous popt), so that their centers match it (stable labels in live view).
    """
    if not _HAS_SCIPY or z.size < MIN_FIT_PIXELS_DOUBLE or np.allclose(z, z[0]):
        return None
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    single = fit_gaussian_2d(x, y, z, pitch_x, pitch_y, maxfev=maxfev, fix_offset=fix_offset)
    lower, upper = _bounds_double(x, y, pitch_x, pitch_y)
    seeds = ([list(p0)] if p0 is not None else []) + _double_seeds(x, y, z, pitch_x, pitch_y, single, fix_offset)
    fixed = _double_fixed(fix_offset, centers)
    tied = _double_tied(amp_ratio, share_sigma)
    lower, upper = _unbound(lower, upper, fixed, tied)
    if centers is not None:
        # with fixed centers the seeds only provide amplitudes and widths; add one from the data
        c = [float(v) for v in centers]
        bg = float(np.percentile(z, 10)) if fix_offset is None else float(fix_offset)
        amp = []
        for k in (0, 2):
            near = int(np.argmin((x - c[k]) ** 2 + (y - c[k + 1]) ** 2))
            amp.append(max(float(z[near]) - bg, 1e-9))
        s0 = 0.6 * max(pitch_x, pitch_y)
        seeds.insert(0, [amp[0], c[0], c[1], s0, amp[1], c[2], c[3], s0, bg])
    best = None
    for guess in seeds:
        out = _fit_double_once(x, y, z, guess, lower, upper, maxfev, fixed, tied)
        if out is not None and (best is None or out[2] < best[2]):
            best = out
    if best is None:
        return None
    popt, pcov, rss = best
    if centers is not None or amp_ratio is not None:
        pass  # the order is given by the centers / the ratio would be inverted by a swap
    elif keep_order_of is not None:
        popt, pcov = _order_like(popt, pcov, np.asarray(keep_order_of, dtype=np.float64))
    elif popt[4] * popt[7] ** 2 > popt[0] * popt[3] ** 2:
        popt, pcov = popt[_DOUBLE_SWAP], pcov[np.ix_(_DOUBLE_SWAP, _DOUBLE_SWAP)]
    return _make_double(x, y, z, popt, pcov, rss, lower, upper, 9 - len(fixed) - len(tied))


def fit_gaussian_2d_double_constrained_next(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prev_popt: np.ndarray,
    tightness: float = 1.0,
    pitch_x: float = PITCH_X_UM,
    pitch_y: float = PITCH_Y_UM,
    fix_offset: Optional[float] = None,
    centers: Optional[Sequence[float]] = None,
    amp_ratio: Optional[float] = None,
    share_sigma: bool = False,
) -> Optional[DoubleFitResult]:
    """Next TRPL frame from the previous double fit with small steps (component order is kept).

    ``centers`` = (x1, y1, x2, y2) holds both centers fixed; ``amp_ratio`` / ``share_sigma``
    tie A2 and sigma2 to A1 and sigma1.
    """
    if not _HAS_SCIPY or z.size < MIN_FIT_PIXELS_DOUBLE or np.allclose(z, z[0]):
        return None
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    p = np.asarray(prev_popt, dtype=np.float64)
    a_ref = max(abs(p[0]), abs(p[4]))
    lower, upper = [], []
    for k in (0, 4):
        A, x0, y0, s = p[k], p[k + 1], p[k + 2], max(abs(p[k + 3]), 1e-3)
        dA = max(a_ref * 0.18 * tightness, 1e-3)
        dx = max(pitch_x * 0.18 * tightness, 0.8)
        dy = max(pitch_y * 0.18 * tightness, 0.8)
        ds = max(s * 0.18 * tightness, 0.5)
        lower += [max(0.0, A - dA), x0 - dx, y0 - dy, max(1e-3, s - ds)]
        upper += [A + dA, x0 + dx, y0 + dy, s + ds]
    off = p[8]
    doff = max(abs(off) * 0.25 * tightness, 0.05 * a_ref * tightness, 0.5)
    lower.append(off - doff)
    upper.append(off + doff)
    fixed = _double_fixed(fix_offset, centers)
    tied = _double_tied(amp_ratio, share_sigma)
    lower, upper = _unbound(lower, upper, fixed, tied)
    out = _fit_double_once(x, y, z, p, lower, upper, 20000, fixed, tied)
    if out is None:
        return None
    # at_bound is judged against the physical limits; with 9 parameters some step limit is
    # nearly always touched, which says nothing about the quality of the fit
    glo, ghi = _unbound(*_bounds_double(x, y, pitch_x, pitch_y), fixed, tied)
    return _make_double(x, y, z, out[0], out[1], out[2], glo, ghi, 9 - len(fixed) - len(tied))


# ======================================================================
# 1D Gaussian fits (row x-linecuts, spectra)
# ======================================================================
@dataclass
class Gauss1DResult:
    amp: float
    center: float
    sigma: float
    offset: float
    gain_odd: float = 0.0
    method: str = "gauss"
    snr: float = float("inf")
    at_bound: bool = False
    dof: int = 0
    flags: Tuple[str, ...] = ()
    sigma_err: float = float("nan")

    @property
    def fwhm(self) -> float:
        return FWHM_FACTOR * self.sigma

    @property
    def area(self) -> float:
        return self.amp * self.sigma * np.sqrt(2.0 * np.pi)

    @property
    def quality_ok(self) -> bool:
        return self.snr >= MIN_SNR

    @property
    def params(self) -> np.ndarray:
        return np.array([self.amp, self.center, self.sigma, self.offset], dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return gauss1d(np.asarray(x, dtype=np.float64), self.amp, self.center, self.sigma, self.offset)


def gauss1d(x, A, x0, sigma, offset):
    return A * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) + offset


def fit_gaussian_1d(
    x: np.ndarray,
    z: np.ndarray,
    *,
    p0: Optional[Sequence[float]] = None,
    fix_offset: Optional[float] = None,
    fix_center: Optional[float] = None,
    sigma_min: float = 1e-3,
    sigma_max: Optional[float] = None,
    center_bounds: Optional[Tuple[float, float]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    maxfev: int = 10000,
) -> Optional[Gauss1DResult]:
    """Fit ``A exp(-(x-x0)^2/2s^2) + B``; ``fix_offset``/``fix_center`` hold B / x0 fixed.

    ``lower``/``upper`` (4 values each, for A, x0, s, B) override the default bounds.
    """
    if not _HAS_SCIPY:
        return None
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(z)
    x, z = x[good], z[good]
    scale = float(np.max(np.abs(z))) if z.size else 0.0
    if scale > 0 and not np.isclose(scale, 1.0):
        # Fit z/scale: the fit is not scale-invariant, and on values of ~100
        # (dark counts in cps) it crawls to maxfev (seconds per frame).
        def sc(v):
            return None if v is None else [float(v[0]) / scale, v[1], v[2], float(v[3]) / scale]

        res = fit_gaussian_1d(
            x, z / scale, p0=sc(p0), fix_offset=None if fix_offset is None else float(fix_offset) / scale,
            fix_center=fix_center, sigma_min=sigma_min, sigma_max=sigma_max, center_bounds=center_bounds,
            lower=sc(lower), upper=sc(upper), maxfev=maxfev,
        )
        return None if res is None else replace(res, amp=res.amp * scale, offset=res.offset * scale)
    fixed: Dict[str, float] = {}
    if fix_offset is not None:
        fixed["B"] = float(fix_offset)
    if fix_center is not None:
        fixed["x0"] = float(fix_center)
    names = ["A", "x0", "s", "B"]
    free = [n for n in names if n not in fixed]
    if x.size < max(len(free), 3) or np.allclose(z, z[0]):
        return None

    span = max(float(np.ptp(x)), 1e-6)
    if sigma_max is None:
        sigma_max = 4.0 * span
    if center_bounds is None:
        center_bounds = (float(x.min()) - span, float(x.max()) + span)
    default_lo = {"A": 0.0, "x0": center_bounds[0], "s": max(sigma_min, 1e-6), "B": -np.inf}
    default_hi = {"A": np.inf, "x0": center_bounds[1], "s": sigma_max, "B": np.inf}
    if lower is not None:
        default_lo.update(dict(zip(names, (float(v) for v in lower))))
    if upper is not None:
        default_hi.update(dict(zip(names, (float(v) for v in upper))))

    if p0 is not None:
        guess = dict(zip(names, (float(v) for v in p0)))
    else:
        bg = fixed.get("B", float(np.percentile(z, 10)))
        w = np.clip(z - bg, 0.0, None)
        total = float(w.sum())
        i_max = int(np.argmax(z))
        c0 = float((w * x).sum() / total) if total > 0 else float(x[i_max])
        s0 = float(np.sqrt((w * (x - c0) ** 2).sum() / total)) if total > 0 else span / 4.0
        guess = {"A": max(float(z[i_max]) - bg, 1e-9), "x0": c0, "s": s0, "B": bg}
    guess.update(fixed)

    lo = [default_lo[n] for n in free]
    hi = [default_hi[n] for n in free]
    start = _clip_into([guess[n] for n in free], lo, hi)

    def model(xx, *pf):
        p = dict(fixed)
        p.update(zip(free, pf))
        return gauss1d(xx, p["A"], p["x0"], p["s"], p["B"])

    try:
        popt, pcov = _curve_fit(model, x, z, p0=start, bounds=(lo, hi), maxfev=maxfev)
    except Exception:
        return None
    p = dict(fixed)
    p.update(zip(free, (float(v) for v in popt)))
    dof = int(x.size - len(free))
    k = free.index("s")
    s_var = float(pcov[k, k]) if dof > 0 and np.all(np.isfinite(pcov)) else float("nan")
    return Gauss1DResult(
        amp=p["A"],
        center=p["x0"],
        sigma=abs(p["s"]),
        offset=p["B"],
        snr=_significance(p["A"], pcov, dof),
        at_bound=_hit_bound(popt, lo, hi),
        dof=dof,
        sigma_err=float(np.sqrt(s_var)) if s_var >= 0 else float("nan"),
    )


def fit_gaussian_1d_constrained_next(
    x: np.ndarray,
    z: np.ndarray,
    prev: Gauss1DResult,
    tightness: float = 1.0,
    pitch: float = SITE_PITCH_UM,
    fix_offset: Optional[float] = None,
) -> Optional[Gauss1DResult]:
    """1D analogue of ``fit_gaussian_2d_constrained_next`` (small steps from the previous frame)."""
    A, c, s, B = prev.amp, prev.center, max(prev.sigma, 1e-3), prev.offset
    dA = max(abs(A) * 0.18 * tightness, 1e-3)
    dc = max(pitch * 0.18 * tightness, 0.8)
    ds = max(abs(s) * 0.18 * tightness, 0.5)
    dB = max(abs(B) * 0.25 * tightness, 0.5)
    lower = [max(0.0, A - dA), c - dc, max(1e-3, s - ds), B - dB]
    upper = [A + dA, c + dc, s + ds, B + dB]
    return fit_gaussian_1d(x, z, p0=[A, c, s, B], fix_offset=fix_offset, lower=lower, upper=upper)


# The offset of a row profile is fitted freely only for rows with this many enabled pixels.
FREE_OFFSET_MIN_PIXELS = 5


def fit_row_profile(
    frame: np.ndarray,
    row: int,
    excluded: Sequence[int] = (),
    background: Optional[float] = None,
    free_offset: bool = False,
    fix_center: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Gauss1DResult]]:
    """Fit the x-linecut of row ``row`` (offset fixed at ``background`` unless free).

    ``fix_center`` holds x0 fixed (e.g. at the center column).

    The offset is only fitted freely for rows with at least
    ``FREE_OFFSET_MIN_PIXELS`` enabled pixels; returns ``(x_um, values, fit)``.
    """
    ids = row_pixels(row, excluded)
    x = PIXEL_X_UM[ids]
    z = np.asarray(frame, dtype=np.float64)[ids]
    if ids.size < 3:
        return x, z, None
    fix = None if (free_offset and ids.size >= FREE_OFFSET_MIN_PIXELS) else (0.0 if background is None else float(background))
    fit = fit_gaussian_1d(
        x,
        z,
        fix_offset=fix,
        fix_center=fix_center,
        sigma_min=0.25 * SITE_PITCH_UM,
        sigma_max=8.0 * SITE_PITCH_UM,
        center_bounds=(float(x.min()) - SITE_PITCH_UM, float(x.max()) + SITE_PITCH_UM),
    )
    if fit is not None:
        flags = []
        if fit.dof == 0:
            flags.append("no degrees of freedom")
        if not fit.quality_ok:
            flags.append("low signal")
        fit.flags = tuple(flags)
    return x, z, fit


# ======================================================================
# Peak location along the spectral (center) column (calibration, spectrum read-out)
# ======================================================================
def _log_parabola(x: np.ndarray, z: np.ndarray, i: int, background: float) -> Optional[Gauss1DResult]:
    if i <= 0 or i >= x.size - 1:
        return None
    a, b, c = z[i - 1] - background, z[i] - background, z[i + 1] - background
    if min(a, b, c) <= 0:
        return None
    la, lb, lc = np.log(a), np.log(b), np.log(c)
    d = la - 2.0 * lb + lc
    if d >= 0:
        return None
    h = float(x[i + 1] - x[i])
    center = float(x[i] + 0.5 * h * (la - lc) / d)
    sigma = float(np.sqrt(-(h**2) / d))
    amp = float(np.exp(lb - (lc - la) ** 2 / (8.0 * d)))
    return Gauss1DResult(amp=amp, center=center, sigma=sigma, offset=background, method="log-parabola")


def locate_peak(
    x: np.ndarray,
    z: np.ndarray,
    pitch: float = LINE_PITCH_UM,
    odd_mask: Optional[np.ndarray] = ROW_IS_SHORT,
    allow_parity: bool = True,
) -> Optional[Gauss1DResult]:
    """Find the peak of a center-column profile (5 rows).

    1. Gaussian fit with sigma >= 0.25 pitch.
    2. If the line is resolved (sigma >= 0.5 pitch) and there are at least 6
       points, refit with a gain for the 4-pixel rows (they average two pixels
       11.5 um off the column, so a spot narrow in x makes the profile
       zig-zag).  With the 5 SPAD23 rows this step is skipped.
    3. If the fit fails, a 3-point log-parabola around the maximum.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(z)
    odd = np.asarray(odd_mask, dtype=bool)[good] if odd_mask is not None else None
    x, z = x[good], z[good]
    if x.size < 4 or np.allclose(z, z[0]):
        return None
    i_max = int(np.argmax(z))
    bounds = (float(x.min()) - pitch, float(x.max()) + pitch)
    res = fit_gaussian_1d(x, z, sigma_min=0.25 * pitch, sigma_max=10.0 * pitch, center_bounds=bounds)

    if (
        res is not None
        and allow_parity
        and odd is not None
        and odd.any()
        and (~odd).any()
        and res.sigma >= 0.5 * pitch
        and x.size >= 6
        and _HAS_SCIPY
    ):
        def model(xx, A, x0, s, B, g):
            return A * np.exp(-((xx - x0) ** 2) / (2.0 * s**2)) * (1.0 + g * odd) + B

        lower = [0.0, bounds[0], 0.25 * pitch, -np.inf, -0.9]
        upper = [np.inf, bounds[1], 10.0 * pitch, np.inf, 3.0]
        zs = max(float(np.max(np.abs(z))), 1e-300)  # fit z/zs, as in fit_gaussian_1d
        start = _clip_into([res.amp / zs, res.center, res.sigma, res.offset / zs, 0.0], lower, upper)
        try:
            popt, pcov = _curve_fit(model, x, z / zs, p0=start, bounds=(lower, upper), maxfev=10000)
            res = Gauss1DResult(
                amp=float(popt[0]) * zs,
                center=float(popt[1]),
                sigma=float(abs(popt[2])),
                offset=float(popt[3]) * zs,
                gain_odd=float(popt[4]),
                method="gauss+parity",
                snr=_significance(popt[0], pcov, int(x.size) - 5),
                at_bound=_hit_bound(popt[:4], lower[:4], upper[:4]),
                dof=int(x.size - 5),
            )
        except Exception:
            pass

    if res is None:
        res = _log_parabola(x, z, i_max, float(np.percentile(z, 10)))
        if res is None:
            return None

    flags: List[str] = []
    if i_max in (0, x.size - 1):
        flags.append("edge row")
    if not (x.min() <= res.center <= x.max()):
        flags.append("peak outside array")
    if res.sigma < 0.5 * pitch:
        flags.append("undersampled")
    if abs(res.gain_odd) > 0.2:
        flags.append(f"odd/even ratio {1.0 + res.gain_odd:.2f} (spot narrow in x)")
    if not res.quality_ok:
        flags.append("low signal")
    return replace(res, flags=tuple(flags))


# ======================================================================
# Wavelength calibration (linear: lambda = intercept + slope * x)
# ======================================================================
@dataclass
class CalibPoint:
    wavelength_nm: float
    x_um: float
    sigma_um: float = float("nan")
    method: str = ""
    flags: Tuple[str, ...] = ()
    profile: Tuple[float, ...] = ()
    captured_at: str = ""

    def to_dict(self) -> dict:
        return {
            "wavelength_nm": float(self.wavelength_nm),
            "x_um": float(self.x_um),
            "sigma_um": None if not np.isfinite(self.sigma_um) else float(self.sigma_um),
            "method": self.method,
            "flags": list(self.flags),
            "profile": [None if not np.isfinite(v) else float(v) for v in self.profile],
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibPoint":
        sig = d.get("sigma_um")
        return cls(
            wavelength_nm=float(d["wavelength_nm"]),
            x_um=float(d["x_um"]),
            sigma_um=float("nan") if sig is None else float(sig),
            method=str(d.get("method", "")),
            flags=tuple(d.get("flags", ())),
            profile=tuple(float("nan") if v is None else float(v) for v in d.get("profile", ())),
            captured_at=str(d.get("captured_at", "")),
        )


@dataclass
class WavelengthCalibration:
    slope_nm_per_um: float
    intercept_nm: float
    points: List[CalibPoint] = field(default_factory=list)
    residuals_nm: Tuple[float, ...] = ()
    created: str = ""
    warnings: Tuple[str, ...] = ()
    flip_x: bool = FLIP_X
    flip_y: bool = FLIP_Y

    @classmethod
    def fit(cls, points: Sequence[CalibPoint]) -> "WavelengthCalibration":
        pts = [p for p in points if np.isfinite(p.wavelength_nm) and np.isfinite(p.x_um)]
        if len(pts) < 2:
            raise ValueError("Need at least 2 captured points with a wavelength.")
        x = np.array([p.x_um for p in pts], dtype=np.float64)
        lam = np.array([p.wavelength_nm for p in pts], dtype=np.float64)
        if float(np.ptp(x)) < 1e-6:
            raise ValueError("All peaks are at the same position; cannot calibrate.")
        if float(np.ptp(lam)) < 1e-9:
            raise ValueError("All points have the same wavelength; cannot calibrate.")
        slope, intercept = np.polyfit(x, lam, 1)
        resid = lam - (slope * x + intercept)

        warnings: List[str] = []
        if len(set(np.round(lam, 6))) < lam.size:
            warnings.append("Two points have the same wavelength.")
        if float(np.ptp(x)) < LINE_SPAN_UM / 3.0:
            warnings.append(
                f"Points span only {100.0 * np.ptp(x) / LINE_SPAN_UM:.0f}% of the array; "
                "wavelengths outside them are extrapolated."
            )
        if len(pts) > 2 and float(np.max(np.abs(resid))) > 0.5 * abs(slope) * LINE_PITCH_UM:
            warnings.append("A residual exceeds half a row; check the captured peaks.")
        for p in pts:
            bad = [f for f in p.flags if f in ("edge row", "peak outside array")]
            if bad:
                warnings.append(f"{p.wavelength_nm:g} nm: {', '.join(bad)}.")
        return cls(
            slope_nm_per_um=float(slope),
            intercept_nm=float(intercept),
            points=list(pts),
            residuals_nm=tuple(float(v) for v in resid),
            created=datetime.now().isoformat(timespec="seconds"),
            warnings=tuple(warnings),
        )

    def wavelength(self, x_um):
        return self.intercept_nm + self.slope_nm_per_um * np.asarray(x_um, dtype=np.float64)

    def x_of(self, wavelength_nm):
        return (np.asarray(wavelength_nm, dtype=np.float64) - self.intercept_nm) / self.slope_nm_per_um

    @property
    def nm_per_row(self) -> float:
        return self.slope_nm_per_um * LINE_PITCH_UM

    @property
    def rms_residual_nm(self) -> float:
        r = np.asarray(self.residuals_nm, dtype=np.float64)
        return float(np.sqrt(np.mean(r**2))) if r.size else 0.0

    def row_wavelengths(self) -> np.ndarray:
        return self.wavelength(ROW_Y_UM)

    def describe(self) -> str:
        lam0, lam1 = (float(v) for v in self.wavelength([0.0, LINE_SPAN_UM]))
        return (
            f"λ = {self.intercept_nm:.3f} + {self.slope_nm_per_um:.5f}·y nm  "
            f"({self.nm_per_row:+.4f} nm/row, array {lam0:.2f} → {lam1:.2f} nm, "
            f"rms residual {self.rms_residual_nm:.3f} nm, {len(self.points)} points)"
        )

    def to_dict(self) -> dict:
        return {
            "device": "SPAD23",
            "model": "linear",
            "slope_nm_per_um": self.slope_nm_per_um,
            "intercept_nm": self.intercept_nm,
            "residuals_nm": list(self.residuals_nm),
            "created": self.created,
            "warnings": list(self.warnings),
            "points": [p.to_dict() for p in self.points],
            "geometry": {
                "pitch_x_um": PITCH_X_UM,
                "pitch_y_um": PITCH_Y_UM,
                "orientation": "rows horizontal, pixels 0-4 in row 0 at y = 0; dispersion along y",
                "position_axis": "y (um); x_um / slope_nm_per_um refer to it",
                "flip_x": self.flip_x,
                "flip_y": self.flip_y,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WavelengthCalibration":
        geo = d.get("geometry", {})
        return cls(
            slope_nm_per_um=float(d["slope_nm_per_um"]),
            intercept_nm=float(d["intercept_nm"]),
            points=[CalibPoint.from_dict(p) for p in d.get("points", [])],
            residuals_nm=tuple(float(v) for v in d.get("residuals_nm", ())),
            created=str(d.get("created", "")),
            warnings=tuple(d.get("warnings", ())),
            flip_x=bool(geo.get("flip_x", FLIP_X)),
            flip_y=bool(geo.get("flip_y", FLIP_Y)),
        )

    @property
    def geometry_matches(self) -> bool:
        return self.flip_x == FLIP_X and self.flip_y == FLIP_Y


# ======================================================================
# TRPL "Fit All" loops (pure functions, run in a worker thread)
# ======================================================================
def _tick(progress: Optional[ProgressFn], done: int, total: int) -> None:
    if progress is not None and (done % 25 == 0 or done == total):
        progress(done, total)


def _cancelled(cancelled: Optional[CancelFn]) -> bool:
    return cancelled is not None and cancelled()


@dataclass
class FitAllResult:
    method: str
    results: Dict[int, object]
    failures: int
    ref_idx: int
    ref_fit: Optional[object] = None
    row: Optional[int] = None
    message: str = ""
    cancelled: bool = False
    centers: Optional[Tuple[float, float, float, float]] = None  # double fit: centers held fixed
    amp_ratio: Optional[float] = None  # double fit: A2/A1 held at this value
    shared_sigma: bool = False  # double fit: sigma2 = sigma1


def fit_all_2d_sequential(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ref_idx: int,
    start: int,
    end: int,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
    fix_offset: Optional[float] = None,
) -> FitAllResult:
    """Method 1: fit the reference frame, then each later frame starting from the previous fit.

    ``fix_offset`` holds the Gaussian offset at that value (e.g. 0 after a pre-t0 subtraction).
    """
    ref = frames[ref_idx].astype(np.float64)
    ref_fit = fit_gaussian_2d(x[mask], y[mask], ref[mask], fix_offset=fix_offset)
    if ref_fit is None:
        raise ValueError("Could not fit the reference frame.")
    results: Dict[int, object] = {ref_idx: ref_fit}
    prev = ref_fit
    failures = 0
    idxs = [i for i in range(max(start, ref_idx), end + 1) if i != ref_idx]
    for n, i in enumerate(idxs, 1):
        if _cancelled(cancelled):
            return FitAllResult("2d_sequential", results, failures, ref_idx, ref_fit, cancelled=True)
        z = frames[i].astype(np.float64)[mask]
        fit = fit_gaussian_2d_constrained_next(x[mask], y[mask], z, prev.popt, tightness=1.0, fix_offset=fix_offset)
        if fit is None:
            fit = fit_gaussian_2d_constrained_next(x[mask], y[mask], z, prev.popt, tightness=1.8,
                                                   fix_offset=fix_offset)
        if fit is None:
            failures += 1
        else:
            results[i] = fit
            prev = fit
        _tick(progress, n, len(idxs))
    return FitAllResult("2d_sequential", results, failures, ref_idx, ref_fit)


def fit_all_2d_fixed_center(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ref_idx: int,
    start: int,
    end: int,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
    fix_offset: Optional[float] = None,
) -> FitAllResult:
    """Method 2: center from the reference 2D fit, then a radial 1D fit per frame."""
    ref = frames[ref_idx].astype(np.float64)
    xm, ym = x[mask], y[mask]
    ref_fit = fit_gaussian_2d(xm, ym, ref[mask], fix_offset=fix_offset)
    if ref_fit is None:
        raise ValueError("Could not fit the reference frame.")
    x0, y0 = float(ref_fit.popt[1]), float(ref_fit.popt[2])
    r = np.sqrt((xm - x0) ** 2 + (ym - y0) ** 2)
    order = np.argsort(r)
    r_sorted = r[order]
    ref_radial = fit_radial_gaussian_1d(r_sorted, ref[mask][order], sigma_guess=ref_fit.sigma_eq,
                                        fix_offset=fix_offset)
    if ref_radial is None:
        raise ValueError("Reference radial 1D fit failed.")
    results: Dict[int, object] = {ref_idx: ref_radial}
    prev = ref_radial
    failures = 0
    idxs = [i for i in range(max(start, ref_idx), end + 1) if i != ref_idx]
    for n, i in enumerate(idxs, 1):
        if _cancelled(cancelled):
            return FitAllResult("2d_fixed_center", results, failures, ref_idx, ref_fit, cancelled=True)
        z = frames[i].astype(np.float64)[mask][order]
        fit = fit_radial_gaussian_1d(r_sorted, z, p0=prev.popt, sigma_guess=prev.sigma_eq, fix_offset=fix_offset)
        if fit is None:
            fit = fit_radial_gaussian_1d(r_sorted, z, p0=None, sigma_guess=prev.sigma_eq, fix_offset=fix_offset)
        if fit is None:
            failures += 1
        else:
            results[i] = fit
            prev = fit
        _tick(progress, n, len(idxs))
    return FitAllResult("2d_fixed_center", results, failures, ref_idx, ref_fit)


def fit_all_2d_double_sequential(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ref_idx: int,
    start: int,
    end: int,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
    fix_offset: Optional[float] = None,
    fix_centers: bool = True,
    fix_ratio: bool = False,
    ratio: Optional[float] = None,
    share_sigma: bool = False,
) -> FitAllResult:
    """Method 3: double 2D Gaussian of the reference frame, then each later frame from the previous fit.

    With ``fix_centers`` both centers stay where the reference fit put them (two pump spots
    whose signal spreads); only the amplitudes, widths and offset follow the frames.
    Otherwise the centers move too; the components keep their labels (a frame that needs
    a free refit is matched to the previous frame's centers).
    With ``fix_ratio`` A2/A1 is held at ``ratio``, or, if ``ratio`` is None, at the value of
    the reference fit; ``share_sigma`` fits one sigma for both Gaussians (reference included).
    """
    method = "2d_double_sequential"
    xm, ym = x[mask], y[mask]
    given = float(ratio) if (fix_ratio and ratio is not None) else None
    ref_fit = fit_gaussian_2d_double(xm, ym, frames[ref_idx].astype(np.float64)[mask], fix_offset=fix_offset,
                                     amp_ratio=given, share_sigma=share_sigma)
    if ref_fit is None:
        raise ValueError("Could not fit two Gaussians to the reference frame.")
    centers = tuple(float(ref_fit.popt[i]) for i in (1, 2, 5, 6)) if fix_centers else None
    amp_ratio = None
    if fix_ratio:
        if given is not None:
            amp_ratio = given
        elif ref_fit.popt[0] > 0 and ref_fit.popt[4] > 0:
            amp_ratio = float(ref_fit.popt[4] / ref_fit.popt[0])
        else:
            raise ValueError("The reference fit has a zero amplitude: A2/A1 cannot be taken from it.")
    extra = dict(centers=centers, amp_ratio=amp_ratio, shared_sigma=bool(share_sigma))
    results: Dict[int, object] = {ref_idx: ref_fit}
    prev = ref_fit
    failures = 0
    idxs = [i for i in range(max(start, ref_idx), end + 1) if i != ref_idx]
    for n, i in enumerate(idxs, 1):
        if _cancelled(cancelled):
            return FitAllResult(method, results, failures, ref_idx, ref_fit, cancelled=True, **extra)
        z = frames[i].astype(np.float64)[mask]
        kw = dict(fix_offset=fix_offset, centers=centers, amp_ratio=amp_ratio, share_sigma=share_sigma)
        fit = fit_gaussian_2d_double_constrained_next(xm, ym, z, prev.popt, 1.0, **kw)
        if fit is None:
            fit = fit_gaussian_2d_double_constrained_next(xm, ym, z, prev.popt, 1.8, **kw)
        if fit is None:
            fit = fit_gaussian_2d_double(xm, ym, z, p0=prev.popt, keep_order_of=prev.popt, **kw)
        if fit is None:
            failures += 1
        else:
            results[i] = fit
            prev = fit
        _tick(progress, n, len(idxs))
    return FitAllResult(method, results, failures, ref_idx, ref_fit, **extra)


def fit_all_row(
    frames: np.ndarray,
    row: int,
    excluded: Sequence[int],
    mask: np.ndarray,
    ref_idx: int,
    start: int,
    end: int,
    fixed_center: bool,
    free_offset: bool = False,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
) -> FitAllResult:
    """Spectral methods 1/2: x-profile of one row per frame.

    Method 1 (``fixed_center=False``) steps from the previous frame's fit;
    method 2 keeps x0 from the reference frame.  The offset is the frame
    background unless ``free_offset`` (rows with >= FREE_OFFSET_MIN_PIXELS pixels).
    """
    method = "row_fixed_center" if fixed_center else "row_sequential"
    ref = frames[ref_idx].astype(np.float64)
    x, _, ref_fit = fit_row_profile(ref, row, excluded, frame_background(ref, mask), free_offset)
    if ref_fit is None:
        raise ValueError(f"Could not fit row {row} in the reference frame.")
    ids = row_pixels(row, excluded)
    use_free = free_offset and ids.size >= FREE_OFFSET_MIN_PIXELS
    results: Dict[int, object] = {ref_idx: ref_fit}
    prev = ref_fit
    failures = 0
    idxs = [i for i in range(max(start, ref_idx), end + 1) if i != ref_idx]
    for n, i in enumerate(idxs, 1):
        if _cancelled(cancelled):
            return FitAllResult(method, results, failures, ref_idx, ref_fit, row=row, cancelled=True)
        frame = frames[i].astype(np.float64)
        z = frame[ids]
        bg = None if use_free else frame_background(frame, mask)
        if fixed_center:
            fit = fit_gaussian_1d(
                x, z, fix_center=ref_fit.center, fix_offset=bg,
                sigma_min=0.25 * SITE_PITCH_UM, sigma_max=8.0 * SITE_PITCH_UM,
                p0=[prev.amp, ref_fit.center, prev.sigma, prev.offset if bg is None else bg],
            )
        else:
            fit = fit_gaussian_1d_constrained_next(x, z, prev, 1.0, SITE_PITCH_UM, fix_offset=bg)
            if fit is None:
                fit = fit_gaussian_1d_constrained_next(x, z, prev, 1.8, SITE_PITCH_UM, fix_offset=bg)
        if fit is None:
            failures += 1
        else:
            results[i] = fit
            prev = fit
        _tick(progress, n, len(idxs))
    return FitAllResult(method, results, failures, ref_idx, ref_fit, row=row)


def fit_all_spectrum(
    frames: np.ndarray,
    excluded: Sequence[int],
    start: int,
    end: int,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
) -> FitAllResult:
    """Spectral method 3: Gaussian peak of the center-column spectrum in every frame."""
    w = column_weights(excluded=excluded)
    results: Dict[int, object] = {}
    failures = 0
    idxs = list(range(start, end + 1))
    for n, i in enumerate(idxs, 1):
        if _cancelled(cancelled):
            return FitAllResult("spectrum_peak", results, failures, start, cancelled=True)
        profile = w @ frames[i].astype(np.float64)
        fit = locate_peak(ROW_Y_UM, profile)
        if fit is None:
            failures += 1
        else:
            results[i] = fit
        _tick(progress, n, len(idxs))
    return FitAllResult("spectrum_peak", results, failures, start)


# ======================================================================
# Spectral transport: Gaussian width of every row's x-profile versus time
# ======================================================================
WIDTH_OK = 0
WIDTH_NOISY = 1  # skipped before fitting: peak below the SNR threshold
WIDTH_FAILED = 2  # the fit did not converge (or x0 could not be fixed)
WIDTH_REJECTED = 3  # fitted, but low significance, at a bound or a large sigma error
WIDTH_NO_PIXELS = 4  # fewer than 3 enabled pixels in the row
WIDTH_NOT_FITTED = 5  # not reached (cancelled)
WIDTH_STATUS_TEXT = {
    WIDTH_OK: "ok",
    WIDTH_NOISY: "too noisy (not fitted)",
    WIDTH_FAILED: "fit failed",
    WIDTH_REJECTED: "rejected (noisy fit)",
    WIDTH_NO_PIXELS: "too few pixels",
    WIDTH_NOT_FITTED: "not fitted",
}
MAX_SIGMA_REL_ERR = 0.5


def row_peak_snr(z: np.ndarray, background) -> float:
    """max_i (z_i - b_i) / sqrt(z_i): Poisson significance of the brightest pixel above background."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0 or not np.isfinite(z).any():
        return float("nan")
    excess = (z - np.asarray(background, dtype=np.float64)) / np.sqrt(np.clip(z, 1.0, None))
    return float(np.nanmax(excess))


def pre_pulse_window(time_ns: np.ndarray, total: np.ndarray, margin_ns: float = 1.0,
                     margin_bins: int = 10) -> Optional[Tuple[float, float]]:
    """Window from the first bin to just before the rise of ``total`` (None if under 3 bins)."""
    t = np.asarray(time_ns, dtype=np.float64)
    if t.size < 4:
        return None
    bin_ns = float(np.median(np.diff(t)))
    t_end = float(t[int(np.argmax(total))]) - max(margin_ns, margin_bins * bin_ns)
    if int(np.count_nonzero(t <= t_end)) < 3:
        return None
    return float(t[0]), t_end


# ======================================================================
# Dark reference (beam blocked): per-pixel dark counts, an alternative to the pre-t0 offset
# ======================================================================
# The pre-t0 offset is the per-pixel level before the pulse; with lifetimes near the laser
# period it also holds signal left over from earlier pulses.  A dark reference is a TRPL
# acquisition with the excitation beam blocked: dark counts are uniform in time, so they are
# kept as counts per bin per second at the reference bin width and scaled to any other
# acquisition time and bin width (the requested times are used: the true exposure is not
# reported by the server).
@dataclass
class DarkReference:
    rate_per_bin: np.ndarray  # per pixel: counts per bin per second, at bin_width_ps
    bin_width_ps: float
    measurement_ms: float
    n_bins: int = 0
    total_counts: int = 0
    detector: str = ""
    taken_at: str = ""
    source: str = ""

    @classmethod
    def from_counts(cls, counts: np.ndarray, bin_width_ps: float, measurement_ms: float, detector: str = "",
                    taken_at: str = "", source: str = "") -> "DarkReference":
        """From a dark histogram [bins, pixels] acquired for ``measurement_ms`` (requested time)."""
        c = np.asarray(counts, dtype=np.float64)
        if c.ndim != 2 or c.shape[0] < 1:
            raise ValueError("The dark acquisition has no time bins.")
        if not measurement_ms > 0:
            raise ValueError("The dark acquisition time is unknown.")
        rate = c.mean(axis=0) / (float(measurement_ms) / 1000.0)
        return cls(rate, float(bin_width_ps), float(measurement_ms), int(c.shape[0]), int(c.sum()), detector,
                   taken_at or datetime.now().isoformat(timespec="seconds"), source)

    @property
    def num_pixels(self) -> int:
        return int(np.asarray(self.rate_per_bin).size)

    def per_bin(self, acq_s: float, bin_ps: float) -> np.ndarray:
        """Expected dark counts per bin of width ``bin_ps`` (e.g. raw bin x rebin) in ``acq_s`` seconds."""
        return np.asarray(self.rate_per_bin, dtype=np.float64) * float(acq_s) * float(bin_ps) / self.bin_width_ps

    def rates_cps(self) -> np.ndarray:
        """Dark count rate of each pixel (counts per second, all bins)."""
        return np.asarray(self.rate_per_bin, dtype=np.float64) * max(self.n_bins, 1)

    def describe(self) -> str:
        cps = self.rates_cps()
        k = int(np.argmax(cps)) if cps.size else 0
        when = self.taken_at.replace("T", " ")
        return (f"{self.detector or 'dark'} dark reference, {self.measurement_ms / 1000.0:g} s at {self.bin_width_ps:g} ps, "
                f"taken {when}: {np.mean(cps):.3g} cps per pixel on average, max {cps[k]:.3g} cps (pixel {k})")

    def summary(self) -> Dict[str, object]:
        """Plain-type dict for file headers (``from_summary`` restores it)."""
        return {"detector": self.detector, "taken_at": self.taken_at, "source": self.source,
                "measurement_ms": self.measurement_ms, "bin_width_ps": self.bin_width_ps, "n_bins": self.n_bins,
                "total_counts": self.total_counts,
                "rate_per_bin": [float(f"{v:.6g}") for v in np.asarray(self.rate_per_bin, dtype=np.float64)]}

    @classmethod
    def from_summary(cls, d: Dict[str, object]) -> "DarkReference":
        return cls(np.asarray(d["rate_per_bin"], dtype=np.float64), float(d["bin_width_ps"]),
                   float(d["measurement_ms"]), int(d.get("n_bins", 0)), int(d.get("total_counts", 0)),
                   str(d.get("detector", "")), str(d.get("taken_at", "")), str(d.get("source", "")))


def dark_compare_text(pre_t0: Optional[np.ndarray], dark: Optional[np.ndarray],
                      mask: Optional[np.ndarray] = None) -> str:
    """How much of the pre-t0 level is not dark (signal left over from earlier pulses)."""
    if pre_t0 is None or dark is None or np.size(pre_t0) != np.size(dark):
        return ""
    p = np.asarray(pre_t0, dtype=np.float64)
    d = np.asarray(dark, dtype=np.float64)
    if mask is not None:
        p, d = p[mask], d[mask]
    sp, sd = float(p.sum()), float(d.sum())
    if not sp > 0:
        return ""
    frac = (sp - sd) / sp
    levels = f"pre-t0 level {sp / p.size:.3g} vs dark {sd / d.size:.3g} counts/bin per pixel"
    below = int(np.sum(p < 0.8 * d))
    if frac < -0.1 or below > 0.5 * p.size:
        return (f"{levels}: the pre-t0 level is BELOW the dark in {below} px - stale dark reference "
                "or a different acquisition time / bin width than assumed?")
    return f"{levels}: {100.0 * max(frac, 0.0):.0f}% of the pre-t0 level is not dark (signal from earlier pulses)"


def fit_row_width(
    frame: np.ndarray,
    row: int,
    excluded: Sequence[int] = (),
    background=0.0,
    min_snr: float = MIN_SNR,
    fix_center: Optional[float] = None,
    free_offset: bool = False,
    max_rel_err: float = MAX_SIGMA_REL_ERR,
) -> Tuple[np.ndarray, np.ndarray, Optional[Gauss1DResult], int, float]:
    """Gaussian fit of one row's x-profile for transport, with a noise gate.

    ``background`` is a constant or one value per pixel (23); it is
    subtracted and the fit offset is 0 (free for rows with >= FREE_OFFSET_MIN_PIXELS
    pixels if ``free_offset``).  Returns ``(x_um, background-subtracted values, fit,
    status, peak_snr)``.  A profile whose brightest pixel is less than
    ``min_snr`` Poisson sigmas above the background is not fitted
    (``WIDTH_NOISY``).  A fit is rejected when its
    amplitude significance is below ``min_snr``, a parameter is at its bound
    or the relative error of sigma exceeds ``max_rel_err``.
    """
    ids = row_pixels(row, excluded)
    y = PIXEL_X_UM[ids]  # position along the row (named ``y`` as in the SPAD93 code)
    raw = np.asarray(frame, dtype=np.float64)[ids]
    b = np.asarray(background, dtype=np.float64)
    zb = b[ids] if b.ndim else np.full(ids.size, float(b))
    z = raw - zb
    if ids.size < 3:
        return y, z, None, WIDTH_NO_PIXELS, float("nan")
    snr = row_peak_snr(raw, zb)
    if not snr >= min_snr:
        return y, z, None, WIDTH_NOISY, snr
    if fix_center is not None and not np.isfinite(fix_center):
        return y, z, None, WIDTH_FAILED, snr
    fix_off = None if (free_offset and ids.size >= FREE_OFFSET_MIN_PIXELS) else 0.0
    fit = fit_gaussian_1d(
        y,
        z,
        fix_offset=fix_off,
        fix_center=fix_center,
        sigma_min=0.25 * SITE_PITCH_UM,
        sigma_max=8.0 * SITE_PITCH_UM,
        center_bounds=(float(y.min()) - SITE_PITCH_UM, float(y.max()) + SITE_PITCH_UM),
    )
    if fit is None:
        return y, z, None, WIDTH_FAILED, snr
    rel = fit.sigma_err / fit.sigma if fit.sigma > 0 else float("inf")
    bad_err = fit.dof > 0 and not rel <= max_rel_err
    if fit.at_bound or fit.snr < min_snr or bad_err:
        return y, z, fit, WIDTH_REJECTED, snr
    return y, z, fit, WIDTH_OK, snr


@dataclass
class RowWidthMap:
    """sigma_x(t, row) from ``fit_all_row_widths``; arrays are [bins, N_ROWS]."""

    method: str
    start: int
    time_ns: np.ndarray
    sigma: np.ndarray
    sigma_err: np.ndarray
    center: np.ndarray
    amp: np.ndarray
    offset: np.ndarray
    peak_snr: np.ndarray
    fit_snr: np.ndarray
    status: np.ndarray
    min_snr: float
    fixed_center: np.ndarray
    free_offset: bool = False
    dark: bool = False
    message: str = ""
    cancelled: bool = False

    @property
    def end(self) -> int:
        return self.start + self.time_ns.size - 1

    def accepted(self, min_snr: Optional[float] = None) -> np.ndarray:
        """Fits to show; a higher ``min_snr`` than the fit used hides more bins."""
        ok = self.status == WIDTH_OK
        if min_snr is not None and min_snr > self.min_snr:
            ok &= (self.peak_snr >= min_snr) & (self.fit_snr >= min_snr)
        return ok

    def bin_row(self, idx: int) -> Optional[int]:
        """Index into the result arrays of time bin ``idx`` (None outside the fitted range)."""
        r = int(idx) - self.start
        return r if 0 <= r < self.time_ns.size else None

    def fit_at(self, idx: int, col: int) -> Optional[Gauss1DResult]:
        r = self.bin_row(idx)
        if r is None or not np.isfinite(self.sigma[r, col]):
            return None
        return Gauss1DResult(
            amp=float(self.amp[r, col]),
            center=float(self.center[r, col]),
            sigma=float(self.sigma[r, col]),
            offset=float(self.offset[r, col]),
            snr=float(self.fit_snr[r, col]),
            sigma_err=float(self.sigma_err[r, col]),
        )

    def counts(self) -> Dict[int, int]:
        vals, n = np.unique(self.status, return_counts=True)
        return {int(v): int(k) for v, k in zip(vals, n)}


def fit_all_row_widths(
    frames: np.ndarray,
    excluded: Sequence[int],
    mask: np.ndarray,
    start: int,
    end: int,
    min_snr: float = MIN_SNR,
    fixed_center: bool = False,
    free_offset: bool = False,
    time_ns: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    center_x: Optional[float] = None,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
) -> RowWidthMap:
    """Fit the x-profile of all 5 rows in every bin ``start..end``.

    With ``fixed_center`` each row's x0 comes from a fit of the profile
    summed over the range, so only the amplitude and sigma vary with time.
    ``center_x`` instead fixes x0 of every row at that x (the center column).
    The background is ``dark`` (per-pixel counts per bin, e.g. the mean
    before the pulse) or, if None, each frame's 10th percentile; the offset
    is additionally free for rows with >= FREE_OFFSET_MIN_PIXELS pixels if ``free_offset``.
    """
    idxs = np.arange(start, end + 1)
    n = idxs.size
    nan = lambda: np.full((n, N_ROWS), np.nan)  # noqa: E731
    out = RowWidthMap(
        method=("widths_center_column" if center_x is not None
                else "widths_fixed_x0" if fixed_center else "widths_free"),
        start=int(start),
        time_ns=(np.asarray(time_ns, dtype=np.float64)[start : end + 1] if time_ns is not None
                 else idxs.astype(np.float64)),
        sigma=nan(), sigma_err=nan(), center=nan(), amp=nan(), offset=nan(), peak_snr=nan(), fit_snr=nan(),
        status=np.full((n, N_ROWS), WIDTH_NOT_FITTED, dtype=np.int8),
        min_snr=float(min_snr),
        fixed_center=np.full(N_ROWS, np.nan),
        free_offset=bool(free_offset),
        dark=dark is not None,
    )
    if center_x is not None:
        out.fixed_center[:] = float(center_x)
    elif fixed_center:
        summed = np.asarray(frames[start : end + 1], dtype=np.float64).sum(axis=0)
        bg = frame_background(summed, mask) if dark is None else np.asarray(dark, dtype=np.float64) * n
        bad = []
        for c in range(N_ROWS):
            _, _, fit, status, _ = fit_row_width(summed, c, excluded, bg, min_snr, None, free_offset)
            if fit is not None and status == WIDTH_OK:
                out.fixed_center[c] = fit.center
            else:
                bad.append(c)
        if bad:
            out.message = "x0 could not be fixed for row(s) " + ", ".join(str(c) for c in bad)
    for r, i in enumerate(idxs):
        if _cancelled(cancelled):
            out.cancelled = True
            break
        frame = np.asarray(frames[i], dtype=np.float64)
        bg = frame_background(frame, mask) if dark is None else dark
        for c in range(N_ROWS):
            fc = float(out.fixed_center[c]) if (fixed_center or center_x is not None) else None
            _, _, fit, status, snr = fit_row_width(frame, c, excluded, bg, min_snr, fc, free_offset)
            out.status[r, c] = status
            out.peak_snr[r, c] = snr
            if fit is not None:
                out.sigma[r, c] = fit.sigma
                out.sigma_err[r, c] = fit.sigma_err
                out.center[r, c] = fit.center
                out.amp[r, c] = fit.amp
                out.offset[r, c] = fit.offset
                out.fit_snr[r, c] = fit.snr
        _tick(progress, r + 1, n)
    return out


# ======================================================================
# Lifetime fits: multi-exponential decay convolved with a Gaussian IRF
# ======================================================================
def emg_unit(t: np.ndarray, t0: float, sigma: float, tau: float) -> np.ndarray:
    """exp(-(t - t0)/tau) for t > t0, convolved with a unit-area Gaussian of width ``sigma``.

    This is the exponentially modified Gaussian scaled so that its amplitude
    is the amplitude of the exponential.  erfcx is used where exp * erfc
    would overflow.
    """
    t = np.asarray(t, dtype=np.float64)
    sigma = max(float(sigma), 1e-9)
    tau = max(float(tau), 1e-9)
    dt = t - t0
    z = (sigma / tau - dt / sigma) / np.sqrt(2.0)
    out = np.empty_like(dt)
    pos = z >= 0
    out[pos] = 0.5 * np.exp(-(dt[pos] ** 2) / (2.0 * sigma**2)) * erfcx(z[pos])
    neg = ~pos
    out[neg] = 0.5 * np.exp(sigma**2 / (2.0 * tau**2) - dt[neg] / tau) * erfc(z[neg])
    return out


@dataclass
class DecayFitResult:
    label: str
    n_exp: int
    ok: bool
    message: str = ""
    t0: float = float("nan")
    sigma_irf: float = float("nan")
    taus: np.ndarray = field(default_factory=lambda: np.zeros(0))
    amps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    offset: float = 0.0
    tau_err: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chi2_red: float = float("nan")
    baseline: float = 0.0
    fit_window: Tuple[float, float] = (float("nan"), float("nan"))

    @property
    def fractions(self) -> np.ndarray:
        """Intensity (area) fraction of each component: a_i tau_i / sum_j a_j tau_j."""
        w = self.amps * self.taus
        total = float(w.sum())
        return w / total if total > 0 else np.full(self.n_exp, np.nan)

    @property
    def tau_avg_intensity(self) -> float:
        w = self.amps * self.taus
        return float((w * self.taus).sum() / w.sum()) if w.sum() > 0 else float("nan")

    @property
    def tau_avg_amplitude(self) -> float:
        return float((self.amps * self.taus).sum() / self.amps.sum()) if self.amps.sum() > 0 else float("nan")

    def component(self, t: np.ndarray, i: int) -> np.ndarray:
        return self.amps[i] * emg_unit(t, self.t0, self.sigma_irf, self.taus[i])

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        y = np.full(t.shape, self.offset, dtype=np.float64)
        for i in range(self.n_exp):
            y += self.component(t, i)
        return y


def subtract_baseline(traces: np.ndarray, time_ns: np.ndarray, t_lo: float, t_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract from each column of ``traces`` [bins, k] its mean over ``t_lo <= t <= t_hi``."""
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim == 1:
        traces = traces[:, None]
    sel = (time_ns >= min(t_lo, t_hi)) & (time_ns <= max(t_lo, t_hi))
    base = np.nanmean(traces[sel], axis=0) if sel.any() else np.zeros(traces.shape[1])
    return traces - base, np.atleast_1d(base)


def fit_decay(
    t: np.ndarray,
    y: np.ndarray,
    var: np.ndarray,
    n_exp: int = 2,
    fit_window: Optional[Tuple[float, float]] = None,
    fit_offset: bool = False,
    label: str = "",
    baseline: float = 0.0,
) -> DecayFitResult:
    """Fit ``sum_i a_i exp(-(t - t0)/tau_i)`` convolved with Gaussian(sigma) (+ offset).

    ``y`` is baseline-subtracted and ``var`` is the Poisson variance of the raw
    counts (weights 1/sqrt(var)).  Several start sets of lifetimes are tried
    and the lowest chi^2 is kept; components are returned sorted by tau.
    """
    n_exp = max(1, int(n_exp))
    res = DecayFitResult(label=label, n_exp=n_exp, ok=False, baseline=float(baseline))
    if not _HAS_SCIPY:
        res.message = "scipy missing"
        return res
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if fit_window is not None:
        lo, hi = sorted(fit_window)
        sel = (t >= lo) & (t <= hi)
        t, y, var = t[sel], y[sel], var[sel]
    good = np.isfinite(y) & np.isfinite(var)
    t, y, var = t[good], y[good], var[good]
    n_par = 2 + 2 * n_exp + (1 if fit_offset else 0)
    if t.size < n_par + 3:
        res.message = "too few points in the fit window"
        return res
    res.fit_window = (float(t[0]), float(t[-1]))
    i_pk = int(np.argmax(y))
    peak = float(y[i_pk])
    if not peak > 0:
        res.message = "no signal"
        return res
    bin_ns = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    w = np.sqrt(np.clip(var, 1.0, None))
    t_pk = float(t[i_pk])
    tau_est = max(float(np.sum(np.clip(y[i_pk:], 0, None)) * bin_ns / peak), 2.0 * bin_ns)

    def model(tt, *p):
        out = np.full(tt.shape, p[-1] if fit_offset else 0.0, dtype=np.float64)
        for k in range(n_exp):
            out += p[2 + 2 * k] * emg_unit(tt, p[0], p[1], p[3 + 2 * k])
        return out

    lower = [t_pk - min(3.0, 0.5 * span), bin_ns / 5.0]
    upper = [t_pk + max(2.0 * bin_ns, 0.2), 3.0]
    for _ in range(n_exp):
        lower += [0.0, bin_ns / 4.0]
        upper += [np.inf, 20.0 * span]
    if fit_offset:
        lower.append(-np.inf)
        upper.append(np.inf)

    best = None
    for spread in ((3.0, 6.0, 15.0) if n_exp > 1 else (1.0, 0.3, 3.0)):
        p0 = [t_pk - bin_ns, max(2.0 * bin_ns, 0.02)]
        for k in range(n_exp):
            p0 += [peak / n_exp, tau_est * spread ** (k - (n_exp - 1) / 2.0)]
        if fit_offset:
            p0.append(0.0)
        try:
            popt, pcov = _curve_fit(model, t, y, p0=_clip_into(p0, lower, upper), sigma=w,
                                    absolute_sigma=True, bounds=(lower, upper), maxfev=20000)
        except Exception:
            continue
        chi2 = float(np.sum(((y - model(t, *popt)) / w) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, popt, pcov)
    if best is None:
        res.message = "fit did not converge"
        return res
    # Weighting by the observed counts biases lifetimes low where the tail has few
    # counts; refit once with the expected counts (model + baseline) as the variance.
    w2 = np.sqrt(np.clip(model(t, *best[1]) + baseline, 1.0, None))
    try:
        popt2, pcov2 = _curve_fit(model, t, y, p0=_clip_into(best[1], lower, upper), sigma=w2,
                                  absolute_sigma=True, bounds=(lower, upper), maxfev=20000)
        best = (float(np.sum(((y - model(t, *popt2)) / w2) ** 2)), popt2, pcov2)
    except Exception:
        pass
    chi2, popt, pcov = best
    amps = np.array([popt[2 + 2 * k] for k in range(n_exp)])
    taus = np.array([popt[3 + 2 * k] for k in range(n_exp)])
    diag = np.diag(pcov) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
    err = np.sqrt(np.clip(diag, 0, None))
    order = np.argsort(taus)
    res.ok = True
    res.t0, res.sigma_irf = float(popt[0]), float(popt[1])
    res.amps = amps[order]
    res.taus = taus[order]
    res.tau_err = np.array([err[3 + 2 * k] for k in range(n_exp)])[order]
    res.offset = float(popt[-1]) if fit_offset else 0.0
    res.chi2_red = chi2 / max(t.size - n_par, 1)
    names = ["t0", "σ_IRF"] + [n for k in range(n_exp) for n in (f"a{k + 1}", f"τ{k + 1}")] + (["offset"] if fit_offset else [])
    pinned = [nm for nm, v, lo, hi in zip(names, popt, lower, upper)
              if (np.isfinite(lo) and abs(v - lo) <= 1e-6 * max(1.0, abs(lo)))
              or (np.isfinite(hi) and abs(v - hi) <= 1e-6 * max(1.0, abs(hi)))]
    if pinned:
        res.message = f"at bound: {', '.join(pinned)}"
    return res


def fit_decays(
    t: np.ndarray,
    traces: Sequence[Tuple[str, np.ndarray, np.ndarray, float]],
    n_exp: int,
    fit_window: Tuple[float, float],
    fit_offset: bool = False,
    progress: Optional[ProgressFn] = None,
    cancelled: Optional[CancelFn] = None,
) -> List[DecayFitResult]:
    """Fit several ``(label, baseline-subtracted y, variance, baseline)`` traces (worker entry)."""
    out: List[DecayFitResult] = []
    for n, (label, y, var, base) in enumerate(traces, 1):
        if _cancelled(cancelled):
            break
        out.append(fit_decay(t, y, var, n_exp, fit_window, fit_offset, label, base))
        if progress is not None:
            progress(n, len(traces))
    return out
