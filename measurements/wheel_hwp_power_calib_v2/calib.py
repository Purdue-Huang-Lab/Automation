"""Wheel + HWP power model and planner (numpy only, no Qt).

Model (the same as ``compute_hwp_nd_entries``, so the PL GUIs' HWP+ND mode predicts the same power)::

    P(nd, hwp) = P_nd(nd) * P_hwp(hwp) / max(P_hwp)

Only *measured* angles are used: with a 0.3 deg HWP and 5 deg ND sweep the
grid is dense enough (adjacent powers ~1 % apart) that no interpolation is needed.
"""

import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

UNIT_SCALE = {"pW": 1e-12, "nW": 1e-9, "µW": 1e-6, "uW": 1e-6, "mW": 1e-3, "W": 1.0}

# header comments written by save_calibration (read back by read_fixed_angles)
_ND_FIXED_KEY = "# ND fixed angle (HWP sweep):"
_HWP_FIXED_KEY = "# HWP fixed angle (ND sweep):"


# ---------------------------------------------------------------- formatting
def best_unit(watts: float) -> Tuple[str, float]:
    """(unit, scale) so that ``watts / scale`` reads well; scale in W."""
    w = abs(float(watts)) if np.isfinite(watts) else 0.0
    if w < 1e-9:
        return "pW", 1e-12
    if w < 1e-6:
        return "nW", 1e-9
    if w < 1e-3:
        return "µW", 1e-6
    if w < 1.0:
        return "mW", 1e-3
    return "W", 1.0


def dual_unit(p_max_w: float) -> Tuple[str, float]:
    """(unit, scale) as the dual-wheel loaders and ``compute_hwp_nd_entries`` use: nW / uW / mW / W (ASCII "uW")."""
    if p_max_w < 1e-6:
        return "nW", 1e-9
    if p_max_w < 1e-3:
        return "uW", 1e-6
    if p_max_w < 1.0:
        return "mW", 1e-3
    return "W", 1.0


def fmt_power(watts: float, digits: int = 4) -> str:
    if watts is None or not np.isfinite(watts):
        return "NaN"
    unit, scale = best_unit(watts)
    return f"{watts / scale:.{digits}g} {unit}"


# ---------------------------------------------------------------- model
def _curve(angles, powers, stds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid (finite, P > 0) points sorted by angle, one per angle; NaN std -> 0."""
    a = np.asarray(angles, dtype=float)
    p = np.asarray(powers, dtype=float)
    s = np.asarray(stds, dtype=float) if len(stds) == len(a) else np.zeros_like(a)
    ok = np.isfinite(a) & np.isfinite(p) & (p > 0)
    a, p, s = a[ok], p[ok], np.nan_to_num(s[ok], nan=0.0)
    a, idx = np.unique(a, return_index=True)
    return a, p[idx], s[idx]


def _log_slope_pct(angles: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """|d ln P / d angle| in % per degree."""
    if len(angles) < 2:
        return np.zeros_like(powers)
    return np.abs(np.gradient(np.log(powers), angles)) * 100.0


def _within(a: np.ndarray, lim: Optional[Tuple[float, float]]) -> np.ndarray:
    if lim is None:
        return np.ones(a.shape, dtype=bool)
    lo, hi = min(lim), max(lim)
    return (a >= lo - 1e-9) & (a <= hi + 1e-9)


@dataclass
class Grid:
    """Every measured (ND, HWP) combination inside the angle limits."""

    nd_deg: np.ndarray  # (n_nd,)
    hwp_deg: np.ndarray  # (n_hwp,)
    power_w: np.ndarray  # (n_nd, n_hwp)
    sigma_w: np.ndarray  # (n_nd, n_hwp), 1 sigma from the two sweeps' read noise
    sens: np.ndarray  # (n_nd, n_hwp), |d ln P/d nd| + |d ln P/d hwp| in %/deg

    @property
    def size(self) -> int:
        return int(self.power_w.size)

    def flat(self) -> Dict[str, np.ndarray]:
        """All combinations sorted by power."""
        nd, hwp = np.meshgrid(self.nd_deg, self.hwp_deg, indexing="ij")
        order = np.argsort(self.power_w, axis=None, kind="stable")
        return {
            "power_w": self.power_w.ravel()[order],
            "sigma_w": self.sigma_w.ravel()[order],
            "nd_deg": nd.ravel()[order],
            "hwp_deg": hwp.ravel()[order],
            "sens": self.sens.ravel()[order],
        }


def build_grid(hwp_angles, hwp_powers, hwp_stds, nd_angles, nd_powers, nd_stds,
               hwp_lim: Optional[Tuple[float, float]] = None,
               nd_lim: Optional[Tuple[float, float]] = None) -> Grid:
    ha, hp, hs = _curve(hwp_angles, hwp_powers, hwp_stds)
    na, npw, ns = _curve(nd_angles, nd_powers, nd_stds)
    if ha.size == 0:
        raise ValueError("no valid HWP sweep points")
    if na.size == 0:
        raise ValueError("no valid ND sweep points")
    p_ref = float(np.max(hp))  # over the whole HWP sweep, as compute_hwp_nd_entries
    # slopes from the whole curves, so the limits do not create artificial end slopes
    h_sens, n_sens = _log_slope_pct(ha, hp), _log_slope_pct(na, npw)
    hk, nk = _within(ha, hwp_lim), _within(na, nd_lim)
    if not hk.any():
        raise ValueError("no HWP points inside the HWP limits")
    if not nk.any():
        raise ValueError("no ND points inside the ND limits")
    ha, hp, hs, h_sens = ha[hk], hp[hk], hs[hk], h_sens[hk]
    na, npw, ns, n_sens = na[nk], npw[nk], ns[nk], n_sens[nk]
    power = np.outer(npw, hp / p_ref)
    rel = np.sqrt((hs / hp)[None, :] ** 2 + (ns / npw)[:, None] ** 2)
    return Grid(nd_deg=na, hwp_deg=ha, power_w=power, sigma_w=power * rel,
                sens=n_sens[:, None] + h_sens[None, :])


def grid_from_hwp_nd(data, hwp_lim: Optional[Tuple[float, float]] = None,
                     nd_lim: Optional[Tuple[float, float]] = None) -> Grid:
    """``build_grid`` from a ``HWPNDCalibData`` (dual_wheel_power_calibration.load_hwp_nd_calibration)."""
    return build_grid(data.hwp_angles, data.hwp_powers_w, data.hwp_stds_w,
                      data.nd_angles, data.nd_powers_w, data.nd_stds_w, hwp_lim=hwp_lim, nd_lim=nd_lim)


def nd_ranges(grid: Grid) -> Dict[str, np.ndarray]:
    """Per ND angle: lowest / highest power reachable with the HWP, and the HWP angle of each."""
    i_min = np.argmin(grid.power_w, axis=1)
    i_max = np.argmax(grid.power_w, axis=1)
    rows = np.arange(grid.nd_deg.size)
    return {
        "nd_deg": grid.nd_deg,
        "p_min": grid.power_w[rows, i_min],
        "hwp_at_min": grid.hwp_deg[i_min],
        "p_max": grid.power_w[rows, i_max],
        "hwp_at_max": grid.hwp_deg[i_max],
    }


def coverage(ranges: Dict[str, np.ndarray]) -> Tuple[float, float, List[Tuple[float, float]]]:
    """(overall min, overall max, gaps) of the union of the per-ND-angle ranges."""
    lo, hi = np.asarray(ranges["p_min"]), np.asarray(ranges["p_max"])
    order = np.argsort(lo)
    gaps: List[Tuple[float, float]] = []
    cur_hi = hi[order[0]]
    for i in order[1:]:
        if lo[i] > cur_hi:
            gaps.append((float(cur_hi), float(lo[i])))
        cur_hi = max(cur_hi, hi[i])
    return float(lo.min()), float(hi.max()), gaps


# ---------------------------------------------------------------- planner
@dataclass
class PlanRow:
    target_w: float
    power_w: float
    sigma_w: float
    nd_deg: float
    hwp_deg: float
    sens: float
    expected_err: float  # sqrt(err^2 + (sens * angle repeatability)^2), fraction
    note: str  # "" (within tolerance), "outside tolerance", "below range", "above range"

    @property
    def err(self) -> float:
        """Relative error of the predicted power."""
        return self.power_w / self.target_w - 1.0

    @property
    def ok(self) -> bool:
        return not self.note


def solve(grid: Grid, targets_w: Sequence[float], tol: float, angle_err_deg: float = 0.1) -> List[PlanRow]:
    """For each target: among the pairs within ``tol`` (fraction) the one with the smallest expected error.

    Expected error = sqrt(err^2 + (sens * angle_err_deg)^2): the power error of the measured
    point plus what a stage-repeatability error of ``angle_err_deg`` does to it, so a point on a
    flat part of both curves wins over a slightly closer one on a steep part.  With no pair
    within ``tol`` the nearest one (in log power) is taken and flagged.  Rows keep the order of
    ``targets_w``.
    """
    f = grid.flat()
    p, sens = f["power_w"], f["sens"]
    p_min, p_max = float(p[0]), float(p[-1])
    rows: List[PlanRow] = []
    for t in targets_w:
        t = float(t)
        err = p / t - 1.0
        expected = np.hypot(err, sens / 100.0 * angle_err_deg)
        cand = np.flatnonzero(np.abs(err) <= tol)
        if cand.size:
            i = cand[np.argmin(expected[cand])]
            note = ""
        else:
            i = int(np.argmin(np.abs(np.log(p / t))))
            if t < p_min:
                note = "below range"
            elif t > p_max:
                note = "above range"
            else:
                note = "outside tolerance"
        rows.append(PlanRow(target_w=t, power_w=float(p[i]), sigma_w=float(f["sigma_w"][i]),
                            nd_deg=float(f["nd_deg"][i]), hwp_deg=float(f["hwp_deg"][i]),
                            sens=float(sens[i]), expected_err=float(expected[i]), note=note))
    return rows


# ---------------------------------------------------------------- target list input
_NUM_RE = re.compile(
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*(pw|nw|uw|µw|mw|w|p|n|u|µ|m)?(?![a-z0-9.])")
_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "": 1.0}


def parse_power_list(text: str, default_unit: str) -> Tuple[List[float], List[str]]:
    """Powers in W from e.g. ``"1, 2.5uW 300 nW; 4e2"`` and the tokens that could not be read.

    Comma / semicolon / whitespace / newline separated.  A bare number (also ``4e2``) is in
    ``default_unit``; a number with a unit (pW, nW, uW / µW, mW, W, or just the prefix) uses that unit.
    """
    low = text.lower().replace("μ", "µ")  # Greek mu -> micro sign
    vals: List[float] = []
    leftover = []
    pos = 0
    default = UNIT_SCALE[default_unit]
    for m in _NUM_RE.finditer(low):
        leftover.append(low[pos:m.start()])
        pos = m.end()
        num, unit = float(m.group(1)), m.group(2)
        if unit is None:
            scale = default
        else:
            scale = _PREFIX[unit[:-1] if unit.endswith("w") else unit]
        if num > 0:
            vals.append(num * scale)
        else:
            leftover.append(m.group(0))
    leftover.append(low[pos:])
    bad = [tok for tok in re.split(r"[\s,;]+", " ".join(leftover)) if tok]
    return vals, bad


def spaced(start: float, stop: float, n: int, log: bool = True) -> np.ndarray:
    if n <= 1:
        return np.array([float(start)])
    if log:
        return np.geomspace(float(start), float(stop), int(n))
    return np.linspace(float(start), float(stop), int(n))


# ---------------------------------------------------------------- files
def write_dual_wheel_csv(path: str, rows: Sequence[PlanRow], source: str = "") -> str:
    """Write ``rows`` as a dual-wheel calibration (a = HWP, b = ND), in row order.

    PL + TRPL v2 loads it as "dual wheel" and sweeps the rows in file order, with the
    HWP on stage A and the ND wheel on stage B (as its HWP+ND mode).  Returns the unit used.
    """
    unit, scale = dual_unit(max(r.power_w for r in rows))
    with open(path, "w", newline="") as fh:
        fh.write(f"# Power plan from wheel+HWP calibration  {datetime.now().isoformat(timespec='seconds')}\n")
        if source:
            fh.write(f"# Calibration file: {os.path.basename(source)}\n")
        fh.write("# Stage A = HWP (a_deg)   stage B = ND wheel (b_deg)\n")
        fh.write("# Targets (W): " + " ".join(f"{r.target_w:.4g}" for r in rows) + "\n")
        w = csv.writer(fh)
        w.writerow(["a_deg", "b_deg", f"power_{unit}"])
        for r in rows:
            w.writerow([f"{r.hwp_deg:.4f}", f"{r.nd_deg:.4f}", f"{r.power_w / scale:.6g}"])
    return unit


def save_calibration(path: str, header: Sequence[str],
                     hwp: Tuple[Sequence[float], Sequence[float], Sequence[float]],
                     nd: Tuple[Sequence[float], Sequence[float], Sequence[float]],
                     nd_fixed: Optional[float] = None, hwp_fixed: Optional[float] = None) -> None:
    """v1 file format ([HWP_SWEEP] / [ND_SWEEP]); ``header`` lines are written as ``# ...`` comments."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for line in header:
            w.writerow([f"# {line}"])
        if nd_fixed is not None:
            w.writerow([f"{_ND_FIXED_KEY} {nd_fixed:.4f}"])
        if hwp_fixed is not None:
            w.writerow([f"{_HWP_FIXED_KEY} {hwp_fixed:.4f}"])
        if len(hwp[0]):
            hp = np.asarray(hwp[1], dtype=float)
            p_max = float(np.nanmax(hp)) if hp.size else 1.0
            w.writerow(["[HWP_SWEEP]"])
            w.writerow(["hwp_angle_deg", "mean_power_w", "std_power_w", "norm_power"])
            for ang, pw, sd in zip(*hwp):
                norm = pw / p_max if p_max > 0 else float("nan")
                w.writerow([f"{ang:.4f}", f"{pw:.6e}", f"{sd:.6e}", f"{norm:.6f}"])
        if len(nd[0]):
            w.writerow(["[ND_SWEEP]"])
            w.writerow(["nd_angle_deg", "mean_power_w", "std_power_w"])
            for ang, pw, sd in zip(*nd):
                w.writerow([f"{ang:.4f}", f"{pw:.6e}", f"{sd:.6e}"])


def read_fixed_angles(path: str) -> Tuple[Optional[float], Optional[float]]:
    """(ND fixed angle of the HWP sweep, HWP fixed angle of the ND sweep) from a v2 file, else None."""
    nd_fixed = hwp_fixed = None
    try:
        with open(path, "r") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                for key in (_ND_FIXED_KEY, _HWP_FIXED_KEY):
                    if line.startswith(key):
                        try:
                            val = float(line[len(key):].strip())
                        except ValueError:
                            continue
                        if key == _ND_FIXED_KEY:
                            nd_fixed = val
                        else:
                            hwp_fixed = val
    except OSError:
        pass
    return nd_fixed, hwp_fixed


# ---------------------------------------------------------------- ND-wheel-only (.xlsx)
ND_XLSX_DETAILS = "details"


def save_nd_xlsx(path: str, header: Sequence[str],
                 nd: Tuple[Sequence[float], Sequence[float], Sequence[float]]) -> int:
    """ND-wheel-only calibration as ``angle | power (nW)`` on the first sheet; returns the rows written.

    That sheet is what PL + TRPL v2 loads as a single-wheel calibration (and the viewers'
    angle -> power files); points without a valid (finite, > 0) power are left out of it.
    The "details" sheet keeps ``header`` and every point with its std, for reloading here.
    """
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["angle", "power (nW)"])
    n = 0
    for ang, pw in zip(nd[0], nd[1]):
        if np.isfinite(pw) and pw > 0:
            ws.append([round(float(ang), 4), float(f"{pw / 1e-9:.6g}")])
            n += 1
    det = wb.create_sheet(ND_XLSX_DETAILS)
    for line in header:
        det.append([line])
    det.append([])
    det.append(["nd_angle_deg", "mean_power_w", "std_power_w"])
    for ang, pw, sd in zip(*nd):
        det.append([float(ang), float(pw), float(sd)])
    wb.save(path)
    return n


def load_nd_xlsx(path: str) -> Tuple[List[float], List[float], List[float]]:
    """(angles, powers W, stds W) from ``save_nd_xlsx`` or any ``angle | power (unit)`` sheet (std 0)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if ND_XLSX_DETAILS in wb.sheetnames:
            rows = list(wb[ND_XLSX_DETAILS].iter_rows(values_only=True))
            for i, row in enumerate(rows):
                if row and row[0] == "nd_angle_deg":
                    pts = [r[:3] for r in rows[i + 1:]
                           if r and len(r) >= 3 and all(isinstance(v, (int, float)) for v in r[:3])]
                    if pts:
                        a, p, s = zip(*pts)
                        return [float(v) for v in a], [float(v) for v in p], [float(v) for v in s]
    finally:
        wb.close()
    from measurements.power_calibration import load_power_calibration

    data = load_power_calibration(path)
    if not data.entries:
        raise ValueError(f"No angle / power rows in {os.path.basename(path)}")
    return ([e.position_deg for e in data.entries], [e.power_w for e in data.entries],
            [0.0] * len(data.entries))
