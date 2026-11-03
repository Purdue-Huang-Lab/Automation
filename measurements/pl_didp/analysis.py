"""Numerics for the two-pump differential PL measurement (no Qt).

Set 1: pump 1 only, power sweep P₁            →  I₁(P₁)
Set 2: pump 1 sweep + weak constant pump 2 P₂  →  I₂(P₁) = I(P₁ + P₂)

    ΔI = I₂ − I₁ = P₂·dI/dP|_{P₁+P₂/2} + O(P₂³)

so ΔI/P₂ is the differential response, most accurately assigned to the
midpoint P₁ + P₂/2 (central-difference argument).  The local power-law
exponent is k = dlnI/dlnP ≈ (ΔI/P₂)·P̄/Ī with P̄ = P₁ + P₂/2, Ī = (I₁+I₂)/2.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from measurements.pl_power_gate_dep_viewer_v2 import _deriv_vs_x, _nm_to_ev

Q_I1 = "I₁  (pump 1)"
Q_I2 = "I₁₊₂  (pump 1 + 2)"
Q_DI = "ΔI = I₁₊₂ − I₁"
Q_REL = "ΔI / I₁"
Q_DIDP = "dI/dP ≈ ΔI / P₂"
Q_DIDP_NUM = "dI/dP  numerical, set 1"
Q_EXP = "dlnI/dlnP ≈ (ΔI/P₂)·P/I"
Q_EXP_NUM = "dlnI/dlnP  numerical, set 1"

QUANTITIES = [Q_DIDP, Q_DI, Q_REL, Q_EXP, Q_I1, Q_I2, Q_DIDP_NUM, Q_EXP_NUM]

# Measured quantity → its numerical counterpart from set 1 alone (overlaid in linecuts)
REFERENCE_OF = {Q_DIDP: Q_DIDP_NUM, Q_EXP: Q_EXP_NUM}

# Quantities whose sweep position is the midpoint P₁ + P₂/2 rather than P₁
MIDPOINT_QUANTITIES = (Q_DIDP, Q_EXP)

# Colour-scale convention: (symmetric, centre)
SCALE_OF = {
    Q_I1: (False, 0.0), Q_I2: (False, 0.0),
    Q_DI: (True, 0.0), Q_REL: (True, 0.0),
    Q_DIDP: (True, 0.0), Q_DIDP_NUM: (True, 0.0),
    Q_EXP: (True, 1.0), Q_EXP_NUM: (True, 1.0),   # k = 1 is linear
}

UNITS_OF = {
    Q_I1: "counts", Q_I2: "counts", Q_DI: "counts", Q_REL: "",
    Q_DIDP: "counts/µW", Q_DIDP_NUM: "counts/µW", Q_EXP: "", Q_EXP_NUM: "",
}


def compute_quantities(
    m1: np.ndarray, m2: np.ndarray, *,
    p1_uw: Optional[np.ndarray], p2_uw: float,
    midpoint: bool, rel_floor: float, power_sweep: bool,
) -> Dict[str, np.ndarray]:
    """All display quantities on the (n_wl, n_x) grid of m1/m2.

    p1_uw       : pump-1 power per column (µW), or None if unknown (uncalibrated angles)
    p2_uw       : pump-2 power (µW); ≤ 0 → quantities that divide by it are NaN
    power_sweep : columns are a power sweep (numerical derivatives need this)
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    nan = np.full_like(m1, np.nan)
    di = m2 - m1
    i_mid = 0.5 * (m1 + m2) if midpoint else m1

    floor = _floor(m1, rel_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(m1) > floor, di / m1, np.nan)

    out = {Q_I1: m1, Q_I2: m2, Q_DI: di, Q_REL: rel,
           Q_DIDP: nan, Q_DIDP_NUM: nan, Q_EXP: nan, Q_EXP_NUM: nan}

    p2 = float(p2_uw)
    if p2 > 0:
        out[Q_DIDP] = di / p2

    if p1_uw is None:
        return out
    p1 = np.asarray(p1_uw, dtype=float).ravel()
    if p1.size != m1.shape[1]:
        return out

    if p2 > 0:
        p_eval = p1 + 0.5 * p2 if midpoint else p1
        with np.errstate(divide="ignore", invalid="ignore"):
            k = out[Q_DIDP] * p_eval[None, :] / i_mid
        out[Q_EXP] = np.where(np.abs(i_mid) > floor, k, np.nan)

    if power_sweep and p1.size >= 2:
        out[Q_DIDP_NUM] = _deriv_vs_x(m1, p1)
        log_i = np.full_like(m1, np.nan)
        keep = np.isfinite(m1) & (m1 > max(floor, 0.0))
        log_i[keep] = np.log(m1[keep])
        log_p = np.full_like(p1, np.nan)
        log_p[p1 > 0] = np.log(p1[p1 > 0])
        out[Q_EXP_NUM] = _deriv_vs_x(log_i, log_p)
    return out


def _floor(m: np.ndarray, rel_floor: float) -> float:
    finite = np.abs(m[np.isfinite(m)])
    if not finite.size:
        return np.inf
    return float(rel_floor) * float(np.nanmax(finite))


def interp_cols(m: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 1 (shared x for every row); NaN outside."""
    m = np.asarray(m, dtype=float)
    xs = np.asarray(x_src, dtype=float).ravel()
    xd = np.asarray(x_dst, dtype=float).ravel()
    out = np.full((m.shape[0], xd.size), np.nan)
    ok = np.isfinite(xs)
    xs, m = xs[ok], m[:, ok]
    if xs.size < 2:
        return out
    order = np.argsort(xs)
    xs, m = xs[order], m[:, order]
    j = np.clip(np.searchsorted(xs, xd) - 1, 0, xs.size - 2)
    x0, x1 = xs[j], xs[j + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (xd - x0) / (x1 - x0)
    inside = (xd >= xs[0]) & (xd <= xs[-1]) & np.isfinite(t)
    vals = m[:, j] * (1.0 - t)[None, :] + m[:, j + 1] * t[None, :]
    out[:, inside] = vals[:, inside]
    return out


def estimate_p2_eff(m1: np.ndarray, m2: np.ndarray, p1_uw: np.ndarray,
                    p2_start_uw: float, *, n_iter: int = 5) -> Tuple[float, int]:
    """Least-squares pump-1-equivalent P₂ such that ΔI ≈ P₂ · (dI₁/dP)|_{P₁+P₂/2}.

    The numerical derivative of set 1 is the reference; ΔI and it are compared
    cell by cell over the whole map.  Returns (p2_eff_uw, n_cells_used).
    """
    m1 = np.asarray(m1, dtype=float)
    di = np.asarray(m2, dtype=float) - m1
    p1 = np.asarray(p1_uw, dtype=float).ravel()
    d_num = _deriv_vs_x(m1, p1)
    p2 = max(float(p2_start_uw), 0.0)
    n = 0
    for _ in range(n_iter):
        d_at = interp_cols(d_num, p1, p1 + 0.5 * p2)
        ok = np.isfinite(d_at) & np.isfinite(di)
        n = int(ok.sum())
        den = float(np.sum(d_at[ok] ** 2))
        if n == 0 or den <= 0:
            return np.nan, 0
        p2_new = float(np.sum(di[ok] * d_at[ok]) / den)
        if not np.isfinite(p2_new):
            return np.nan, n
        converged = abs(p2_new - p2) <= 1e-6 * max(abs(p2_new), 1e-30)
        p2 = p2_new
        if converged or p2 <= 0:
            break
    return p2, n


def to_display_y(wl_nm: np.ndarray, m: np.ndarray, unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a (n_wl, n_x) map onto a uniform, ascending y grid in eV or nm."""
    if unit == "eV":
        return _nm_to_ev(wl_nm, m)
    wl = np.asarray(wl_nm, dtype=float).ravel()
    ok = np.isfinite(wl)
    wl, m = wl[ok], np.asarray(m, dtype=float)[ok, :]
    if wl.size < 2:
        return wl, m
    order = np.argsort(wl)
    wl, m = wl[order], m[order, :]
    y = np.linspace(wl[0], wl[-1], wl.size)
    out = np.full((y.size, m.shape[1]), np.nan)
    for j in range(m.shape[1]):
        col = m[:, j]
        mask = np.isfinite(col)
        if mask.sum() >= 2:
            out[:, j] = np.interp(y, wl[mask], col[mask])
    return y, out
