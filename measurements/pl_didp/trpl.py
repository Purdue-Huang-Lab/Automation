"""TRPL pump-2 subtraction, pixel by pixel (numpy only).

For every SPAD23 pixel p and time bin t:

    ΔI(t, p) = I_on(t, p) − I_off(t, p)

Both traces of a point are processed identically before the subtraction:
optional time-zero alignment (shifts found on the pump-2-off trace and applied
to both, so the pair stays registered), optional rebinning and an optional
per-trace pre-t0 offset.

Time-zero alignment puts every pixel's first-peak edge at one time per point;
across a sweep the whole point is then moved (all pixels by the same whole
bins) so that time is the same at every point: the circular median of the
points' own time zeros.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from spad23.spad23_analysis import TimeZeroAlignment, align_first_peaks, pre_pulse_window, rebin_time
from spad23.spad23_io import TrplData

Q_DI = "ΔI = I_on − I_off"
Q_DIDP = "dI/dP ≈ ΔI / P₂"
Q_REL = "ΔI / I_off"
Q_OFF = "I_off  (pump 2 off)"
Q_ON = "I_on  (pump 2 on)"
QUANTITIES = [Q_DI, Q_DIDP, Q_REL, Q_OFF, Q_ON]
SIGNED = (Q_DI, Q_DIDP, Q_REL)
UNITS_OF = {Q_DI: "counts", Q_DIDP: "counts/µW", Q_REL: "", Q_OFF: "counts", Q_ON: "counts"}


@dataclass
class TrplPair:
    """Processed pump-2 off / on traces of one point, float [bins, 23]."""
    t_ns: np.ndarray
    off: np.ndarray
    on: np.ndarray
    shift_bins: Optional[np.ndarray] = None    # time-zero shifts (raw bins), applied to both
    align_msg: str = ""
    t0_ns: float = float("nan")                # time zero after alignment (absolute ns)
    point_shift_bins: int = 0                  # part of shift_bins that moves the whole point to the sweep's t0
    pre_window: Optional[Tuple[float, float]] = None
    off_dark: Optional[np.ndarray] = None      # per-pixel pre-t0 offsets subtracted
    on_dark: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)

    @property
    def bin_ns(self) -> float:
        return float(np.median(np.diff(self.t_ns))) if self.t_ns.size > 1 else 1.0


def roll_bins(counts: np.ndarray, shifts: Sequence[int]) -> np.ndarray:
    """Circular per-pixel shift (as ``align_first_peaks`` applies it)."""
    out = np.array(counts, copy=True)
    for p, s in enumerate(shifts):
        if int(s):
            out[:, p] = np.roll(counts[:, p], int(s))
    return out


def time_zero(off: TrplData, on: TrplData) -> TimeZeroAlignment:
    """First-peak alignment of the pump-2-off trace, cut to the pair's common length."""
    n = min(off.n_bins, on.n_bins)
    return align_first_peaks(off.counts[:n], np.asarray(off.time_ns[:n], dtype=np.float64))


def time_zero_ns(al: TimeZeroAlignment, t_ns: np.ndarray) -> float:
    """Absolute time of an alignment's target (``target_ns`` counts from bin 0)."""
    return float(t_ns[0]) + float(al.target_ns)


def common_time_zero(t0s: Sequence[float], period_ns: float) -> float:
    """Circular median of the points' time zeros (NaN when none is finite)."""
    v = np.asarray([x for x in t0s if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    d = (v - v[0] + period_ns / 2.0) % period_ns - period_ns / 2.0
    return float(v[0] + np.median(d))


def prepare_pair(off: TrplData, on: TrplData, *, align: bool = False, rebin: int = 1,
                 pre_t0: bool = False, pre_window: Optional[Tuple[float, float]] = None,
                 alignment: Optional[TimeZeroAlignment] = None,
                 t0_ref_ns: Optional[float] = None) -> TrplPair:
    """Register and correct one off / on pair; ``pre_window`` None → automatic.

    ``alignment`` is ``time_zero(off, on)`` if already computed; ``t0_ref_ns`` is the
    sweep's common time zero the whole point is moved to (None: the point's own).
    """
    b_off, b_on = float(off.bin_width_ps), float(on.bin_width_ps)
    if abs(b_off - b_on) > 1e-6 * max(b_off, b_on, 1.0):
        raise ValueError(f"bin widths differ ({b_off:g} ps vs {b_on:g} ps)")
    n = min(off.n_bins, on.n_bins)
    if n < 2:
        raise ValueError("too few time bins")
    notes = []
    if off.n_bins != on.n_bins:
        notes.append(f"lengths differ ({off.n_bins} vs {on.n_bins} bins): cut to {n}")
    t = np.asarray(off.time_ns[:n], dtype=np.float64)
    a = np.asarray(off.counts[:n], dtype=np.float64)
    b = np.asarray(on.counts[:n], dtype=np.float64)

    shifts, msg, t0, whole = None, "", float("nan"), 0
    if align:
        al = alignment
        if al is None:
            try:
                al = time_zero(off, on)
            except ValueError as exc:
                msg = f"time-zero alignment skipped: {exc}"
        if al is not None:
            shifts = np.asarray(al.shift_bins, dtype=np.int64).copy()
            msg = al.message
            t0 = time_zero_ns(al, t)
            if t0_ref_ns is not None and np.isfinite(t0_ref_ns) and np.isfinite(t0):
                dt = float(al.bin_ns)
                period = n * dt
                whole = int(np.round(((t0_ref_ns - t0 + period / 2.0) % period - period / 2.0) / dt))
                if whole:
                    shifts += whole
                    msg += f" Whole point moved {whole * dt:+.2f} ns to the sweep's time zero."
                t0 = float(t[0]) + (t0 + whole * dt - float(t[0])) % period
            elif t0_ref_ns is not None:
                notes.append("no time zero at this point: not moved to the sweep's time zero")
            a = roll_bins(a, shifts)
            b = roll_bins(b, shifts)

    if int(rebin) > 1:
        a, t_r = rebin_time(a, t, int(rebin))
        b, _ = rebin_time(b, t, int(rebin))
        t = t_r

    window = off_dark = on_dark = None
    if pre_t0:
        window = pre_window if pre_window is not None else pre_pulse_window(t, a.sum(axis=1))
        if window is not None:
            m = (t >= window[0]) & (t <= window[1])
            if m.any():
                off_dark, on_dark = a[m].mean(axis=0), b[m].mean(axis=0)
                a = a - off_dark
                b = b - on_dark
            else:
                window = None
        if window is None:
            notes.append("pre-t0 window empty: no offset subtracted")
    return TrplPair(t, a, b, shifts, msg, t0, whole, window, off_dark, on_dark, notes)


def quantity(q: str, off: np.ndarray, on: np.ndarray, *, p2_uw: float, rel_floor: float) -> np.ndarray:
    """Quantity ``q`` per bin for each column of off / on ([bins, k]).

    ΔI / I_off is blanked where I_off < rel_floor × max|I_off| of that column.
    """
    off = np.asarray(off, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    if q == Q_OFF:
        return off
    if q == Q_ON:
        return on
    di = on - off
    if q == Q_DI:
        return di
    if q == Q_DIDP:
        return di / float(p2_uw) if p2_uw > 0 else np.full_like(di, np.nan)
    if q == Q_REL:
        floor = float(rel_floor) * np.nanmax(np.abs(off), axis=0, initial=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(off) > floor[None, :], di / off, np.nan)
    raise KeyError(q)


def window_values(q: str, pair: TrplPair, t0: float, t1: float, *,
                  p2_uw: float, rel_floor: float) -> np.ndarray:
    """Per-pixel value of ``q`` over the window [t0, t1] (sums; ΔI/I_off = ΣΔ / ΣI_off)."""
    m = (pair.t_ns >= t0) & (pair.t_ns <= t1)
    s_off = pair.off[m].sum(axis=0)
    s_on = pair.on[m].sum(axis=0)
    return quantity(q, s_off[None, :], s_on[None, :], p2_uw=p2_uw, rel_floor=rel_floor)[0]


def on_time_axis(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """A trace on another (uniform) time axis: same axis → as is, else linear, NaN outside."""
    t_src = np.asarray(t_src, dtype=np.float64)
    if t_src.size == t_dst.size and np.allclose(t_src, t_dst):
        return np.asarray(y, dtype=np.float64)
    out = np.interp(t_dst, t_src, y, left=np.nan, right=np.nan)
    return out
