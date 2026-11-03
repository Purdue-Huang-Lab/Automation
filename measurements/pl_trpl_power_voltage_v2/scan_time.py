"""Scan time estimate of the PL + TRPL v2 sweep.

``simulate`` replays the loop order of ``PowerVoltageSweepThread`` (power-outer with snake,
voltage-outer, grid, dI/dP) and counts, per acquisition point, what the worker will do before
and at that point: stage move calls / substeps / motion time, gate microsteps and ramp calls,
VI reads, fixed waits, acquisitions and flips.  ``TimingModel`` turns those counts into seconds
with coefficients that start from defaults (stage velocity and Andor readout time read from the
devices when known) and are learned from the worker's ``timing`` events: live during a scan
(for the time left) and across scans (saved in QSettings).

``ScanTracker`` follows a running scan; ``ScanTimeWidget`` is the status-bar readout.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from measurements.pl_trpl_power_voltage_v2.vdesktop import DesktopFollower

SETTINGS = ("Automation", "PLTRPLv2ScanTiming")
SETTINGS_KEY = "model_v1"

# features of one acquisition point (costs since the previous point)
F_STAGE_CALLS, F_STAGE_STEPS, F_STAGE_MOTION, F_GATE_MICRO, F_GATE_CALLS, F_VI_READS, F_WAIT, F_ACQ, F_FLIPS, \
    F_RAMP_READS = range(10)  # F_RAMP_READS: V/I reads inside long ramps (when the microsteps are not read)
N_FEATURES = 10

# Kinesis K10CR1 defaults, used when the stage did not report its velocity
DEFAULT_STAGE_VEL = 10.0  # deg/s
DEFAULT_STAGE_ACC = 10.0  # deg/s^2

DEFAULTS = {
    "stage": [0.05, 0.10, 1.0],  # s per move_to call, s per substep, x motion time
    "frame_oh": 0.5,  # s per Andor frame beyond the exposure (readout + snap overhead), unknown readout
    "snap_oh": 0.15,  # added to the SDK readout time to get the frame overhead
    "save": 0.4,  # s per .asc file
    "trpl_oh": 0.6,  # s per TRPL acquisition beyond its acquisition time
    "acq_oh": 0.02,  # s per acquisition beyond max(PL, TRPL)
    "flip": 0.7,  # s per flip beyond the settle time
    "resid": 0.02,  # s per point not covered by the parts above
    "start": 1.5,  # s from Start Sweep to the first point's work
    "end": 0.3,  # s from the last point to done
}
# Keithley: a V/I read is one READ? on a 2400, two measurements (:MEAS:VOLT?, :MEAS:CURR?) on others
# (2450 over TCPIP, current autoranging at nA): s per V/I read, and per write
GATE_READ_S = {"2400": 0.05, "other": 0.35}
GATE_WRITE_S = 0.01
GATE_SYNC_S = 0.01  # "*OPC?" after each set when the microsteps are not read (sync_each_step)
LIVE_MIN_POINTS = 3  # learned values are saved only from scans with at least this many points


# ---------------------------------------------------------------- geometry (mirrors the wrappers)
def trapezoid_time(d: float, vmax: float, acc: float) -> float:
    """Time of a point-to-point move of ``d`` degrees with a trapezoidal velocity profile."""
    d = abs(float(d))
    if d <= 0.0:
        return 0.0
    if vmax <= 0 or acc <= 0:
        return 0.0
    if d >= vmax * vmax / acc:
        return d / vmax + vmax / acc
    return 2.0 * math.sqrt(d / acc)


def stage_move_geometry(start: Optional[float], target: float, step_deg: float,
                        angle_range: Tuple[float, float] = (0.0, 360.0), wrap: bool = True) -> List[float]:
    """Substep lengths (deg) of ``RotationStage.move_to(target, step_deg=...)`` from ``start``.

    An unknown start (no readback yet) counts as a move to where the stage already is.
    """
    lo, hi = float(angle_range[0]), float(angle_range[1])
    target = max(lo, min(hi, float(target)))
    if start is None or (isinstance(start, float) and math.isnan(start)):
        start = target
    start = float(start)
    step = abs(float(step_deg))
    width = hi - lo
    if wrap and width > 0:
        start = ((start - lo) % width) + lo
        target = ((target - lo) % width) + lo
        delta = target - start
        if delta > 0:
            alt = delta - width
        elif delta < 0:
            alt = delta + width
        else:
            alt = 0.0
        use = alt if abs(alt) < abs(delta) else delta
        if step <= 0:
            step = abs(use)
        if abs(use) <= step + 1e-12:
            return [abs(use)]
        out = []
        remaining = abs(use)
        while remaining > step + 1e-12:
            out.append(step)
            remaining -= step
        out.append(remaining)
        return out
    d = abs(target - start)
    if step <= 0:
        step = d
    if d <= step + 1e-12:
        return [d]
    n_full = int(d // step)
    out = [step] * n_full
    rest = d - n_full * step
    if rest > 1e-12:
        out.append(rest)
    return out


def gate_ramp_microsteps(start: Optional[float], target: float, step_v: float) -> int:
    """Microsteps of ``KeithleySMU.ramp_to_voltage`` (each one: set, dwell, read)."""
    start = 0.0 if start is None else float(start)
    target = float(target)
    step = max(1e-6, abs(float(step_v)))
    d = abs(target - start)
    if d <= step:
        return 1
    n_full = int(d // step)
    return n_full + (1 if abs(d - n_full * step) > 1e-9 else 0)


# ---------------------------------------------------------------- scan description
@dataclass
class ScanSpec:
    """Everything the sweep worker loops over, plus the start state of the hardware."""

    targets: List[Optional[Tuple[float, Optional[float]]]]  # stage (A, B or None) per plan entry
    sweep_power: bool
    sweep_voltage: bool
    mode: str  # "dual" / "front" / "back"
    grid: bool
    order: str  # "power_outer" / "voltage_outer"
    nsteps: int
    nsteps_a: int
    nsteps_b: int
    front_v0: float
    front_step: float
    back_v0: float
    back_step: float
    front_present: bool  # gate passed to the worker (ramped when swept, read at every point)
    back_present: bool
    ramp_step_v: float
    ramp_dwell_s: float
    settle_ms: float
    zero_v_eps: float
    zero_v_extra_settle_ms: float
    ramp_step_deg: float
    measure_pl: bool
    measure_trpl: bool
    exposure_ms: float
    accum_n: int
    tcspc_acq_s: float
    didp: bool
    flipper_settle_ms: float
    ramp_micro_read: bool = False  # V/I read after every ramp microstep
    long_ramp_steps: int = 50  # longer ramps read every microstep anyway
    gate_2400: Optional[bool] = None  # gates are 2400s (one READ? per V/I read); None = not known
    readout_label: str = ""
    readout_s: Optional[float] = None  # Andor SDK readout time at these settings
    stage_velocity: Optional[Tuple[float, float]] = None  # (max velocity deg/s, acceleration deg/s^2)
    start_a: Optional[float] = None
    start_b: Optional[float] = None
    start_vf: Optional[float] = None
    start_vb: Optional[float] = None
    detector: str = "SPAD23"  # TRPL overhead is learned per detector (SPAD23 / SPAD93)

    def key(self) -> tuple:
        return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in self.__dict__.items())

    @staticmethod
    def entry_target(entry) -> Optional[Tuple[float, Optional[float]]]:
        if entry is None:
            return None
        if hasattr(entry, "a_deg"):
            return float(entry.a_deg), float(entry.b_deg)
        return float(entry.position_deg), None

    @classmethod
    def from_worker(cls, th, **extra) -> "ScanSpec":
        """The spec of a constructed ``PowerVoltageSweepThread`` (exactly what it will run)."""
        return cls(
            targets=[cls.entry_target(item.get("entry")) for item in th.plan] if th.sweep_power else [None],
            sweep_power=th.sweep_power, sweep_voltage=th.sweep_voltage, mode=th.sweep_mode,
            grid=th.grid_sweep, order=th.sweep_order, nsteps=th.nsteps, nsteps_a=th.nsteps_a,
            nsteps_b=th.nsteps_b, front_v0=th.front_v0, front_step=th.front_step, back_v0=th.back_v0,
            back_step=th.back_step, front_present=th.front_dev is not None, back_present=th.back_dev is not None,
            ramp_step_v=th.ramp_step_v, ramp_dwell_s=th.ramp_dwell_s, settle_ms=th.settle_ms,
            zero_v_eps=th.zero_v_eps, zero_v_extra_settle_ms=th.zero_v_extra_settle_ms,
            ramp_step_deg=th.ramp_step_deg, measure_pl=th.measure_pl, measure_trpl=th.measure_trpl,
            exposure_ms=th.exposure_ms, accum_n=th.accum_n, tcspc_acq_s=th.tcspc_acq_s, didp=th.didp,
            flipper_settle_ms=th.flipper_settle_ms, ramp_micro_read=th.ramp_micro_read,
            long_ramp_steps=th.long_ramp_steps, detector=th.detector.key, **extra,
        )


@dataclass
class Schedule:
    rows: np.ndarray  # (n_points, N_FEATURES)
    suffix: np.ndarray  # (n_points + 1, N_FEATURES): suffix[k] = sum of rows[k:]
    spec: ScanSpec

    @property
    def n_points(self) -> int:
        return int(self.rows.shape[0])


def _voltage_steps(spec: ScanSpec, reverse: bool = False):
    """``PowerVoltageSweepThread._iter_voltage_steps``."""
    if not spec.sweep_voltage:
        return [(None, None)]
    n = max(1, int(spec.nsteps))
    out = []
    for k in (range(n - 1, -1, -1) if reverse else range(n)):
        vf = float(spec.front_v0 + k * spec.front_step) if spec.mode in ("front", "dual") else None
        vb = float(spec.back_v0 + k * spec.back_step) if spec.mode in ("back", "dual") else None
        out.append((vf, vb))
    return out


def _grid_steps(spec: ScanSpec):
    """``PowerVoltageSweepThread._iter_grid_steps`` (B snaked on odd A rows)."""
    na, nb = max(1, int(spec.nsteps_a)), max(1, int(spec.nsteps_b))
    for ai in range(na):
        vf = float(spec.front_v0 + ai * spec.front_step)
        for bi in (range(nb - 1, -1, -1) if ai % 2 == 1 else range(nb)):
            yield vf, float(spec.back_v0 + bi * spec.back_step)


def simulate(spec: ScanSpec) -> Schedule:
    """Per-point feature rows of the sweep, in the worker's order."""
    vmax, acc = spec.stage_velocity or (DEFAULT_STAGE_VEL, DEFAULT_STAGE_ACC)
    eps = abs(float(spec.zero_v_eps))
    reads = int(bool(spec.front_present)) + int(bool(spec.back_present))
    n_acq = 2 if spec.didp else 1
    flips = 2 if spec.didp else 0
    settle_s = max(0.0, spec.settle_ms) / 1000.0
    zero_s = max(0.0, spec.zero_v_extra_settle_ms) / 1000.0
    rows: List[List[float]] = []
    pending = [0.0] * N_FEATURES
    state = {"a": spec.start_a, "b": spec.start_b, "vf": spec.start_vf, "vb": spec.start_vb}
    motion_cache: Dict[float, float] = {}

    def motion(d: float) -> float:
        t = motion_cache.get(d)
        if t is None:
            t = motion_cache[d] = trapezoid_time(d, vmax, acc)
        return t

    def move_one(key: str, target: float) -> None:
        steps = stage_move_geometry(state[key], target, spec.ramp_step_deg)
        pending[F_STAGE_CALLS] += 1
        pending[F_STAGE_STEPS] += len(steps)
        pending[F_STAGE_MOTION] += sum(motion(d) for d in steps)
        state[key] = target

    def move(target) -> None:
        if not spec.sweep_power or target is None:
            return
        a, b = target
        move_one("a", a)
        if b is not None:
            move_one("b", b)

    def ramp(key: str, target: float) -> None:
        n = gate_ramp_microsteps(state[key], target, spec.ramp_step_v)
        pending[F_GATE_MICRO] += n
        pending[F_GATE_CALLS] += 1
        if not spec.ramp_micro_read and n > spec.long_ramp_steps:
            pending[F_RAMP_READS] += n
        state[key] = target

    def wait(s: float) -> None:
        if s > 0:
            pending[F_WAIT] += s

    def is_zero(vf, vb) -> bool:
        return (vf is not None and abs(vf) <= eps) or (vb is not None and abs(vb) <= eps)

    def acquire() -> None:
        nonlocal pending
        pending[F_VI_READS] += reads
        pending[F_ACQ] += n_acq
        pending[F_FLIPS] += flips
        rows.append(pending)
        pending = [0.0] * N_FEATURES

    plan = spec.targets if spec.sweep_power else [None]
    front, back = spec.front_present, spec.back_present

    if spec.grid:
        for target in plan:
            move(target)
            if front:
                ramp("vf", float(spec.front_v0))
            if back:
                ramp("vb", float(spec.back_v0))
            prev_vf = None
            for vf, vb in _grid_steps(spec):
                if front and (prev_vf is None or vf != prev_vf):
                    ramp("vf", vf)
                    prev_vf = vf
                if back:
                    ramp("vb", vb)
                wait(settle_s)
                if is_zero(vf, vb):
                    wait(zero_s)
                acquire()
    elif spec.order == "voltage_outer":
        for vf, vb in _voltage_steps(spec):
            if spec.sweep_voltage:
                if vf is not None and front:
                    ramp("vf", vf)
                if vb is not None and back:
                    ramp("vb", vb)
                wait(settle_s)
                if is_zero(vf, vb):
                    wait(zero_s)
            for target in plan:
                move(target)
                acquire()
    else:
        snake = bool(spec.sweep_power and spec.sweep_voltage and max(1, int(spec.nsteps)) > 1)
        forward = True
        for target in plan:
            move(target)
            steps = _voltage_steps(spec, reverse=bool(snake and not forward))
            if spec.sweep_voltage:
                vf0, vb0 = steps[0]
                if spec.mode in ("front", "dual") and front:
                    ramp("vf", vf0)
                if spec.mode in ("back", "dual") and back:
                    ramp("vb", vb0)
            for vf, vb in steps:
                if spec.sweep_voltage:
                    if vf is not None and front:
                        ramp("vf", vf)
                    if vb is not None and back:
                        ramp("vb", vb)
                    wait(settle_s)
                if is_zero(vf, vb):
                    wait(zero_s)
                acquire()
            if snake:
                forward = not forward

    arr = np.asarray(rows, dtype=np.float64).reshape(-1, N_FEATURES)
    suffix = np.zeros((arr.shape[0] + 1, N_FEATURES), dtype=np.float64)
    if arr.shape[0]:
        suffix[:-1] = np.cumsum(arr[::-1], axis=0)[::-1]
    return Schedule(rows=arr, suffix=suffix, spec=spec)


# ---------------------------------------------------------------- learning
# A default counts as DEFAULT_WEIGHT observations, a saved value as its observation count (capped):
# a scan's own data take over after a few points, and saved values hold no trace of the defaults.
DEFAULT_WEIGHT = 1.0
SAVED_WEIGHT_CAP = 10.0
COUNT_CAP = 200.0  # saved values keep adapting (older scans fade out)


def _saved(entry, default):
    """(value, observation count) of a saved entry; (default, 0) when there is none."""
    if isinstance(entry, dict) and "v" in entry:
        return entry["v"], float(entry.get("n", 0.0))
    return default, 0.0


class _Mean:
    """Running mean from a prior (saved value or default)."""

    def __init__(self, prior: float, n_prior: float = 0.0):
        self.prior = float(prior)
        self.n_prior = float(n_prior)  # observations behind the prior (0 = default)
        self.w0 = min(self.n_prior, SAVED_WEIGHT_CAP) if self.n_prior > 0 else DEFAULT_WEIGHT
        self.sum = 0.0
        self.n = 0.0

    def add(self, value: float, weight: float = 1.0) -> None:
        if weight > 0 and math.isfinite(value):
            self.sum += float(value) * weight
            self.n += weight

    @property
    def value(self) -> float:
        return (self.prior * self.w0 + self.sum) / (self.w0 + self.n)

    def saved(self) -> dict:
        """Data-only value over this and earlier scans (the default does not count)."""
        n_old = min(self.n_prior, COUNT_CAP)
        if self.n + n_old <= 0:
            return {"v": self.prior, "n": 0.0}
        return {"v": (self.prior * n_old + self.sum) / (n_old + self.n), "n": n_old + self.n}


class _Ridge:
    """Linear fit y = x . c, regularized towards the prior (saved value or default)."""

    def __init__(self, prior: Sequence[float], n_prior: float = 0.0):
        self.prior = np.asarray(prior, dtype=np.float64)
        self.n_prior = float(n_prior)
        self.w0 = min(self.n_prior, SAVED_WEIGHT_CAP) if self.n_prior > 0 else DEFAULT_WEIGHT
        k = self.prior.size
        self.xtx = np.zeros((k, k))
        self.xty = np.zeros(k)
        self.n = 0

    def add(self, x: Sequence[float], y: float) -> None:
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)) or not math.isfinite(y) or not np.any(x):
            return
        self.xtx += np.outer(x, x)
        self.xty += x * float(y)
        self.n += 1

    def _solve(self, w0: float) -> np.ndarray:
        if self.n == 0:
            return self.prior.copy()
        # prior precision scaled to the typical feature size, so it is worth ~w0 observations
        scale = np.maximum(np.diag(self.xtx) / max(1, self.n), 1e-9)
        lam = np.diag(w0 * scale)
        try:
            c = np.linalg.solve(lam + self.xtx, lam @ self.prior + self.xty)
        except np.linalg.LinAlgError:
            return self.prior.copy()
        return np.maximum(c, 0.0)

    @property
    def value(self) -> np.ndarray:
        return self._solve(self.w0)

    def saved(self) -> dict:
        # a small pull to the prior keeps collinear features (moves / substeps) well defined
        n_old = min(self.n_prior, COUNT_CAP)
        v = self._solve(max(n_old, 0.25)) if self.n else self.prior
        return {"v": [float(c) for c in v], "n": n_old + self.n}


class TimingModel:
    """Seconds per counted feature; priors from defaults / saved values, learned from timing events."""

    MEANS = ("save", "trpl_oh", "acq_oh", "flip", "resid", "start", "end")

    def __init__(self, saved: Optional[dict] = None):
        saved = dict(saved or {})
        self.scans = int(saved.get("scans", 0))
        self.saved = saved
        self.reset_live()

    def reset_live(self) -> None:
        """Start a new scan's learning from the saved values."""
        self.stage = _Ridge(*_saved(self.saved.get("stage"), DEFAULTS["stage"]))
        for key in self.MEANS:
            setattr(self, key, _Mean(*_saved(self.saved.get(key), DEFAULTS[key])))
        self._frame_oh: Dict[str, _Mean] = {}
        self._gate: Dict[str, _Ridge] = {}  # by "read" / "noread" (V/I read after every microstep or not)
        self._read: Dict[str, _Mean] = {}  # by gate type "2400" / "other"
        self._trpl: Dict[str, _Mean] = {}  # TRPL overhead by detector; prior = the global trpl_oh

    # ---- persistence
    @classmethod
    def load(cls) -> "TimingModel":
        try:
            raw = QtCore.QSettings(*SETTINGS).value(SETTINGS_KEY, "", type=str)
            return cls(json.loads(raw) if raw else None)
        except Exception:
            return cls()

    def to_dict(self) -> dict:
        frame = dict(self.saved.get("frame_oh_by_rate", {}))
        for label, m in self._frame_oh.items():
            if m.n > 0:
                frame[label] = m.saved()
        gate = dict(self.saved.get("gate_by_mode", {}))
        gate.update({k: m.saved() for k, m in self._gate.items() if m.n > 0})
        read = dict(self.saved.get("read_by_type", {}))
        read.update({k: m.saved() for k, m in self._read.items() if m.n > 0})
        trpl = dict(self.saved.get("trpl_oh_by_det", {}))
        trpl.update({k: m.saved() for k, m in self._trpl.items() if m.n > 0})
        out = {"scans": self.scans, "stage": self.stage.saved(), "frame_oh_by_rate": frame,
               "gate_by_mode": gate, "read_by_type": read, "trpl_oh_by_det": trpl}
        for key in self.MEANS:
            out[key] = getattr(self, key).saved()
        return out

    def commit(self, n_points: int) -> bool:
        """End of a scan: keep what was learned (as the next scan's prior) and save it."""
        if n_points < LIVE_MIN_POINTS:
            return False
        self.scans += 1
        self.saved = self.to_dict()
        self.reset_live()
        try:
            QtCore.QSettings(*SETTINGS).setValue(SETTINGS_KEY, json.dumps(self.saved))
        except Exception:
            pass
        return True

    # ---- per-acquisition times
    def frame_oh(self, spec: ScanSpec) -> float:
        m = self._frame_oh.get(spec.readout_label)
        if m is None:
            prior, n = _saved(self.saved.get("frame_oh_by_rate", {}).get(spec.readout_label), None)
            if prior is None:
                prior = (float(spec.readout_s) + DEFAULTS["snap_oh"]) if spec.readout_s else DEFAULTS["frame_oh"]
            m = self._frame_oh[spec.readout_label] = _Mean(prior, n)
        return m.value

    @staticmethod
    def _gate_type(spec: ScanSpec) -> str:
        return "2400" if spec.gate_2400 else "other"

    def read_model(self, spec: ScanSpec) -> _Mean:
        key = self._gate_type(spec)
        m = self._read.get(key)
        if m is None:
            m = self._read[key] = _Mean(*_saved(self.saved.get("read_by_type", {}).get(key), GATE_READ_S[key]))
        return m

    def gate_model(self, spec: ScanSpec) -> _Ridge:
        """Ramp cost: [s per microstep beyond the dwell, s per ramp call]."""
        key = "read" if spec.ramp_micro_read else "noread"
        m = self._gate.get(key)
        if m is None:
            read = GATE_READ_S[self._gate_type(spec)]
            # microstep: set (+ V/I read, or the *OPC? sync without reads); call: compliance + output writes,
            # start read, verify read
            prior = [GATE_WRITE_S + (read if spec.ramp_micro_read else GATE_SYNC_S), 2.0 * read + 3.0 * GATE_WRITE_S]
            m = self._gate[key] = _Ridge(*_saved(self.saved.get("gate_by_mode", {}).get(key), prior))
        return m

    def pl_s(self, spec: ScanSpec) -> float:
        if not spec.measure_pl:
            return 0.0
        n = max(1, int(spec.accum_n))
        return n * (spec.exposure_ms / 1000.0 + self.frame_oh(spec)) + self.save.value

    def trpl_model(self, spec: ScanSpec) -> _Mean:
        """TRPL overhead of this detector (93 px stream more events than 23); starts from the global value."""
        key = spec.detector
        m = self._trpl.get(key)
        if m is None:
            prior, n = _saved(self.saved.get("trpl_oh_by_det", {}).get(key), None)
            m = self._trpl[key] = _Mean(self.trpl_oh.value, 0.0) if prior is None else _Mean(prior, n)
        return m

    def trpl_s(self, spec: ScanSpec) -> float:
        return (spec.tcspc_acq_s + self.trpl_model(spec).value) if spec.measure_trpl else 0.0

    def acq_s(self, spec: ScanSpec) -> float:
        """One acquisition (PL and TRPL run in parallel: the longer one)."""
        return max(self.pl_s(spec), self.trpl_s(spec)) + self.acq_oh.value

    def weights(self, spec: ScanSpec) -> np.ndarray:
        w = np.zeros(N_FEATURES + 1)
        w[F_STAGE_CALLS:F_STAGE_MOTION + 1] = self.stage.value
        g = self.gate_model(spec).value
        w[F_GATE_MICRO] = spec.ramp_dwell_s + g[0]
        w[F_GATE_CALLS] = g[1]
        w[F_VI_READS] = self.read_model(spec).value
        w[F_RAMP_READS] = w[F_VI_READS]
        w[F_WAIT] = 1.0
        w[F_ACQ] = self.acq_s(spec)
        w[F_FLIPS] = spec.flipper_settle_ms / 1000.0 + self.flip.value
        w[N_FEATURES] = self.resid.value  # per point
        return w

    def end_s(self, spec: ScanSpec) -> float:
        # dI/dP: the worker flips back to pump 2 off at the end (no settle)
        return self.end.value + (self.flip.value if spec.didp else 0.0)

    def predict(self, sched: Schedule) -> dict:
        """Total and breakdown (seconds) of a whole scan."""
        spec = sched.spec
        w = self.weights(spec)
        tot = sched.suffix[0]
        n = sched.n_points
        parts = {
            "acquisition": tot[F_ACQ] * w[F_ACQ],
            "stage": float(tot[F_STAGE_CALLS:F_STAGE_MOTION + 1] @ w[F_STAGE_CALLS:F_STAGE_MOTION + 1]),
            "gate ramps": tot[F_GATE_MICRO] * w[F_GATE_MICRO] + tot[F_GATE_CALLS] * w[F_GATE_CALLS]
            + tot[F_RAMP_READS] * w[F_RAMP_READS],
            "VI reads": tot[F_VI_READS] * w[F_VI_READS],
            "settle waits": tot[F_WAIT],
            "flipper": tot[F_FLIPS] * w[F_FLIPS],
            "overhead": self.start.value + self.end_s(spec) + n * w[N_FEATURES],
        }
        return {"total": float(sum(parts.values())), "parts": parts, "n_points": n,
                "pl_s": self.pl_s(spec), "trpl_s": self.trpl_s(spec), "acq_s": self.acq_s(spec),
                "stage_detail": {"calls": tot[F_STAGE_CALLS], "steps": tot[F_STAGE_STEPS],
                                 "motion_s": tot[F_STAGE_MOTION]},
                "gate_detail": {"microsteps": tot[F_GATE_MICRO], "calls": tot[F_GATE_CALLS]}}

    def point_s(self, sched: Schedule, k: int) -> float:
        w = self.weights(sched.spec)
        return float(sched.rows[k] @ w[:N_FEATURES] + w[N_FEATURES])

    def remaining_s(self, sched: Schedule, done: int) -> float:
        """Seconds after ``done`` points have been acquired (to the end of the scan)."""
        done = max(0, min(done, sched.n_points))
        w = self.weights(sched.spec)
        n_left = sched.n_points - done
        return float(sched.suffix[done] @ w[:N_FEATURES] + n_left * w[N_FEATURES]) + self.end_s(sched.spec)

    # ---- learning from one point's timing event
    def learn_point(self, sched: Schedule, k: int, ev: dict, wall_s: Optional[float]) -> None:
        spec = sched.spec
        x = sched.rows[k]
        # the first move / ramp started from an unknown position when there was no readback
        first_stage_unknown = k == 0 and spec.start_a is None
        first_gate_unknown = k == 0 and (spec.start_vf is None or spec.start_vb is None)
        if x[F_STAGE_CALLS] > 0 and "stage" in ev and not first_stage_unknown:
            self.stage.add(x[F_STAGE_CALLS:F_STAGE_MOTION + 1], float(ev["stage"]))
        if x[F_GATE_CALLS] > 0 and "gate" in ev and not first_gate_unknown:
            self.gate_model(spec).add([x[F_GATE_MICRO], x[F_GATE_CALLS]],
                                      float(ev["gate"]) - x[F_GATE_MICRO] * spec.ramp_dwell_s
                                      - x[F_RAMP_READS] * self.read_model(spec).value)
        if x[F_VI_READS] > 0 and "read" in ev:
            self.read_model(spec).add(float(ev["read"]) / x[F_VI_READS])
        n_acq = max(1.0, x[F_ACQ])
        if spec.measure_pl and "frame" in ev:
            frames = n_acq * max(1, int(spec.accum_n))
            self.frame_oh(spec)  # make sure the per-rate estimate exists
            self._frame_oh[spec.readout_label].add(float(ev["frame"]) / frames - spec.exposure_ms / 1000.0,
                                                   weight=n_acq)
        if spec.measure_pl and "save" in ev:
            self.save.add(float(ev["save"]) / n_acq, weight=n_acq)
        if spec.measure_trpl and "trpl" in ev:
            oh = float(ev["trpl"]) / n_acq - spec.tcspc_acq_s
            self.trpl_model(spec).add(oh, weight=n_acq)
            self.trpl_oh.add(oh, weight=n_acq)  # global value: prior of a detector without data yet
        if "acq" in ev:
            # against this event's own PL / TRPL times (the models of those are still learning)
            pl = (float(ev.get("frame", 0.0)) + float(ev.get("save", 0.0))) / n_acq if spec.measure_pl else 0.0
            trpl = float(ev.get("trpl", 0.0)) / n_acq if spec.measure_trpl else 0.0
            self.acq_oh.add(float(ev["acq"]) / n_acq - max(pl, trpl), weight=n_acq)
        if x[F_FLIPS] > 0 and "flip" in ev:
            self.flip.add(float(ev["flip"]) / x[F_FLIPS] - spec.flipper_settle_ms / 1000.0)
        if wall_s is not None:
            measured = sum(float(ev.get(key, 0.0)) for key in ("stage", "gate", "read", "wait", "acq", "flip"))
            self.resid.add(wall_s - measured)


# ---------------------------------------------------------------- live tracking
def fmt_duration(s: Optional[float]) -> str:
    if s is None or not math.isfinite(s):
        return "--"
    s = max(0, int(round(s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:d}:{m:02d}:{sec:02d}"
    return f"{h:d}:{m:02d}:{sec:02d}"


def fmt_clock(epoch_s: float) -> str:
    now = time.time()
    fmt = "%H:%M" if time.localtime(epoch_s)[:3] == time.localtime(now)[:3] else "%a %H:%M"
    return time.strftime(fmt, time.localtime(epoch_s))


CSV_HEADER = ["idx", "wall_time", "elapsed_s", "point_s", "stage_s", "gate_s", "read_s", "wait_s", "acq_s",
              "flip_s", "predicted_point_s", "left_est_s"]


@dataclass
class ScanTracker:
    """Follows one running scan: elapsed, time left, total, progress; learns as points come in."""

    sched: Schedule
    model: TimingModel
    t_click: float  # time.monotonic() at Start Sweep
    estimate_s: float = 0.0  # total predicted at the start
    done: int = 0
    t_last: Optional[float] = None
    t_end: Optional[float] = None
    end_dt: Optional[float] = None  # last point -> end event, minus timed leftovers
    wall_sum: float = 0.0  # wall time of points 2..done (the first one also holds the start)
    rows: List[list] = field(default_factory=list)

    def __post_init__(self):
        self.estimate_s = self.model.predict(self.sched)["total"]
        self._left_after: float = self.model.remaining_s(self.sched, 0)
        self._next_point_s: float = self.model.point_s(self.sched, 0) if self.sched.n_points else 0.0

    @property
    def n_points(self) -> int:
        return self.sched.n_points

    def on_event(self, ev: dict) -> Optional[list]:
        """Worker ``timing`` payload; returns the CSV row of a point event."""
        kind = ev.get("event")
        t = float(ev.get("t", time.monotonic()))
        if kind == "start":
            return None
        if kind == "end":
            self.t_end = t
            if self.t_last is not None:
                self.end_dt = t - self.t_last - sum(float(ev.get(key, 0.0)) for key in ("stage", "gate", "wait"))
            return None
        if kind != "point":
            return None
        k = self.done
        if k >= self.sched.n_points:  # more points than planned: keep counting, nothing to learn
            self.done += 1
            self.t_last = t
            return None
        predicted = self.model.point_s(self.sched, k)
        if self.t_last is None:
            wall = None  # the first point also holds the start overhead
            measured = sum(float(ev.get(key, 0.0)) for key in ("stage", "gate", "read", "wait", "acq", "flip"))
            self.model.start.add(t - self.t_click - measured)
        else:
            wall = t - self.t_last
            self.wall_sum += wall
        self.model.learn_point(self.sched, k, ev, wall)
        self.done += 1
        self.t_last = t
        c = self.correction()
        end = self.model.end_s(self.sched.spec)
        self._left_after = c * (self.model.remaining_s(self.sched, self.done) - end) + end
        self._next_point_s = c * self.model.point_s(self.sched, self.done) if self.done < self.sched.n_points else 0.0
        row = [self.done, time.strftime("%Y-%m-%d %H:%M:%S"), f"{t - self.t_click:.3f}",
               f"{(wall if wall is not None else t - self.t_click):.3f}"]
        row += [f"{float(ev.get(key, 0.0)):.3f}" for key in ("stage", "gate", "read", "wait", "acq", "flip")]
        row += [f"{predicted:.3f}", f"{self._left_after:.3f}"]
        self.rows.append(row)
        return row

    def correction(self) -> float:
        """Measured / modelled wall time of this scan's points so far (points 2..done, today's coefficients),
        pulled towards 1 by two points' worth: covers what the per-part model has not learned yet."""
        n = min(self.done, self.sched.n_points) - 1
        if n < 1:
            return 1.0
        w = self.model.weights(self.sched.spec)
        p_now = float((self.sched.suffix[1] - self.sched.suffix[n + 1]) @ w[:N_FEATURES] + n * w[N_FEATURES])
        if p_now <= 0:
            return 1.0
        lam = 2.0 * p_now / n
        return min(2.0, max(0.5, (self.wall_sum + lam) / (p_now + lam)))

    def finalize(self, status: str) -> bool:
        """Scan done: learn the end overhead (completed scans only) and save the model."""
        if self.t_end is None:
            self.t_end = time.monotonic()
        if status == "ok" and self.end_dt is not None:
            flip_back = self.model.flip.value if self.sched.spec.didp else 0.0
            self.model.end.add(self.end_dt - flip_back)
        return self.model.commit(min(self.done, self.sched.n_points))

    def elapsed_s(self, now: Optional[float] = None) -> float:
        end = self.t_end if self.t_end is not None else (time.monotonic() if now is None else now)
        return end - self.t_click

    def left_s(self, now: Optional[float] = None) -> float:
        if self.t_end is not None:
            return 0.0
        now = time.monotonic() if now is None else now
        if self.t_last is None:
            return max(self.model.start.value + self._left_after - (now - self.t_click),
                       self._left_after - self._next_point_s)
        # count down through the point in progress, but never below the time after it
        floor = self._left_after - self._next_point_s
        return max(self._left_after - (now - self.t_last), floor)

    def total_s(self, now: Optional[float] = None) -> float:
        return self.elapsed_s(now) + self.left_s(now)


# ---------------------------------------------------------------- widgets
MINI_POS_KEY = "mini_pos"
MINI_W = 290
_MINI_TINT = {"idle": "#ffffff", "running": "#ffffff", "ok": "#dcf2dc", "bad": "#f7dada"}


class ScanMiniWindow(QtWidgets.QWidget):
    """Tiny always-on-top readout of the scan time (opened by right-clicking the status-bar readout).

    Parentless so it stays up when the GUI is minimized; a tool window (no taskbar entry) that never
    takes the focus. Drag it anywhere; the position is remembered. Fixed size, set once: resizing a
    frameless window to its content made Windows grow it by 30 px on scaled multi-monitor setups
    (QWindowsWindow::setGeometry warnings), so the second line is elided and the bar keeps its room."""

    closed = QtCore.pyqtSignal()
    clear_requested = QtCore.pyqtSignal()  # "Show estimate" after a finished scan

    def __init__(self):
        super().__init__(None, QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("Scan progress")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setObjectName("scanMiniFrame")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)
        lay = QtWidgets.QGridLayout(self.frame)
        lay.setContentsMargins(8, 4, 4, 6)
        lay.setHorizontalSpacing(4)
        lay.setVerticalSpacing(2)
        self.big = QtWidgets.QLabel("Scan time: --")
        self.big.setTextFormat(QtCore.Qt.PlainText)
        font = self.big.font()
        font.setPointSizeF(font.pointSizeF() * 1.45)
        font.setBold(True)
        self.big.setFont(font)
        self.small = QtWidgets.QLabel("")
        self.small.setTextFormat(QtCore.Qt.PlainText)
        self.small.setStyleSheet("color: #444;")
        self.small.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.closeBtn = QtWidgets.QToolButton()
        self.closeBtn.setText("×")
        self.closeBtn.setAutoRaise(True)
        self.closeBtn.setFixedSize(18, 18)
        self.closeBtn.setToolTip("Close")
        self.closeBtn.clicked.connect(self.close)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setFixedHeight(8)
        self.bar.setTextVisible(False)
        policy = self.bar.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.bar.setSizePolicy(policy)
        self.bar.setVisible(False)
        lay.addWidget(self.big, 0, 0)
        lay.addWidget(self.closeBtn, 0, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        lay.addWidget(self.small, 1, 0, 1, 2)
        lay.addWidget(self.bar, 2, 0, 1, 2)
        lay.setColumnStretch(0, 1)
        lay.setRowStretch(3, 1)  # any extra height goes below the bar
        self._tip = "Scan progress: drag to move, right-click for options"
        self.setToolTip(self._tip)
        self._kind = "idle"
        self._drag: Optional[QtCore.QPoint] = None
        self._placed = False
        # Windows hides it when the user switches virtual desktops: follow to the current one
        self._follower = DesktopFollower(self)
        self._set_tint("idle")
        self.ensurePolished()  # style sheet + fonts applied, so the hint is the shown height
        self.setFixedSize(MINI_W, self.sizeHint().height())

    # ---- content
    def _set_tint(self, kind: str) -> None:
        self.frame.setStyleSheet(f"QFrame#scanMiniFrame {{ background: {_MINI_TINT.get(kind, '#ffffff')}; "
                                 "border: 1px solid #8a8a8a; }")

    def show_state(self, kind: str, big: str, small: str, done: int = 0, n: int = 0) -> None:
        """kind: "idle", "running", "ok" (finished) or "bad" (aborted / trip / error)."""
        if kind != self._kind:
            self._kind = kind
            self._set_tint(kind)
        self.big.setText(big)
        width = self.small.width() if self.small.width() > 20 else MINI_W - 14
        self.small.setText(self.small.fontMetrics().elidedText(small, QtCore.Qt.ElideRight, width))
        self.setToolTip(f"{small}\n\n{self._tip}" if small else self._tip)
        running = kind == "running"
        if running:
            self.bar.setRange(0, max(1, n))
            self.bar.setValue(min(done, max(1, n)))
        self.bar.setVisible(running)

    # ---- position
    def _default_pos(self) -> QtCore.QPoint:
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        return QtCore.QPoint(avail.right() - self.width() - 16, avail.top() + 16)

    def _restore_pos(self) -> None:
        pos = None
        try:
            raw = QtCore.QSettings(*SETTINGS).value(MINI_POS_KEY)
            if isinstance(raw, QtCore.QPoint):
                pos = raw
        except Exception:
            pos = None
        # back onto a screen if that monitor is gone (keep the top-left corner on some screen)
        if pos is None or QtWidgets.QApplication.screenAt(pos + QtCore.QPoint(20, 10)) is None:
            pos = self._default_pos()
        self.move(pos)

    def _save_pos(self) -> None:
        try:
            QtCore.QSettings(*SETTINGS).setValue(MINI_POS_KEY, self.pos())
        except Exception:
            pass

    def showEvent(self, event) -> None:
        if not self._placed:
            self._placed = True
            self._restore_pos()
        super().showEvent(event)
        self._follower.start()

    def closeEvent(self, event) -> None:
        self._follower.stop()
        self._save_pos()
        event.accept()
        self.closed.emit()

    # ---- drag / menu
    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self._drag)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None and event.button() == QtCore.Qt.LeftButton:
            self._drag = None
            self._save_pos()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QtWidgets.QMenu(self)
        act_clear = menu.addAction("Show estimate") if self._kind in ("ok", "bad") else None
        act_close = menu.addAction("Close")
        chosen = menu.exec_(event.globalPos())
        if chosen is act_close:
            self.close()
        elif act_clear is not None and chosen is act_clear:
            self.clear_requested.emit()


class ScanTimeWidget(QtWidgets.QFrame):
    """Status-bar readout: estimate before a scan, elapsed / left / total / ETA during it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 2, 0)
        lay.setSpacing(6)
        self.label = QtWidgets.QLabel("Scan time: --")
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setFixedWidth(110)
        self.bar.setFixedHeight(14)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%v/%m")
        self.bar.setVisible(False)
        lay.addStretch(1)  # keep the text in the bottom-right corner
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        self._last_summary = ""
        self._last_detail = ""
        self._tracker: Optional[ScanTracker] = None
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self.refresh_running)
        # floating always-on-top copy (right-click): the state last shown, and the estimate while idle
        self._mini: Optional[ScanMiniWindow] = None
        self._mini_state: tuple = ("idle", "Scan time: --", "")
        self._mini_idle: tuple = self._mini_state
        self._mini_sticky = False  # a finished scan's result stays up until the next scan / "Show estimate"
        self.setToolTip("")

    def setToolTip(self, tip: str) -> None:  # noqa: N802 (Qt name)
        hint = "<i>Right-click: floating progress window (always on top)</i>"
        super().setToolTip(f"{tip}<br><br>{hint}" if tip else hint)

    # ---- floating window
    def contextMenuEvent(self, event) -> None:
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Floating progress window")
        act.setCheckable(True)
        act.setChecked(self.mini_visible())
        if menu.exec_(event.globalPos()) is act:
            self.toggle_mini()

    def mini_visible(self) -> bool:
        return self._mini is not None and self._mini.isVisible()

    def toggle_mini(self) -> None:
        if self.mini_visible():
            self.close_mini()
            return
        if self._mini is None:
            self._mini = ScanMiniWindow()
            self._mini.clear_requested.connect(self._on_mini_clear)
        self._mini.show_state(*self._mini_state)
        self._mini.show()

    def close_mini(self) -> None:
        if self._mini is not None:
            self._mini.close()

    def _push_mini(self, *state) -> None:
        self._mini_state = state
        if self._mini is not None and self._mini.isVisible():
            self._mini.show_state(*state)

    def _push_idle(self, big: str, small: str) -> None:
        self._mini_idle = ("idle", big, small)
        if not self._mini_sticky:
            self._push_mini(*self._mini_idle)

    def _on_mini_clear(self) -> None:
        self._mini_sticky = False
        self._push_mini(*self._mini_idle)

    # ---- idle
    def set_unavailable(self, why: str) -> None:
        if self._tracker is not None:
            return
        self.bar.setVisible(False)
        self.label.setText(self._prefix() + f"Est. n/a ({why})")
        self.setToolTip("Scan time estimate: " + why)
        self._push_idle("Est. n/a", why)

    def set_estimate(self, pred: dict, spec: ScanSpec, model: TimingModel, notes: Sequence[str] = ()) -> None:
        if self._tracker is not None:
            return
        total = pred["total"]
        self.bar.setVisible(False)
        done_at = fmt_clock(time.time() + total)
        self.label.setText(self._prefix() + f"Est. {fmt_duration(total)} · {pred['n_points']} pts · "
                           f"done ≈ {done_at} if started now")
        tip = self._breakdown_html(pred, spec, model, notes)
        self.setToolTip(tip + (f"<br><br>{self._last_detail}" if self._last_detail else ""))
        self._push_idle(f"Est. {fmt_duration(total)}", f"{pred['n_points']} pts · done ≈ {done_at} if started now")

    def _prefix(self) -> str:
        return f"{self._last_summary}  │  " if self._last_summary else ""

    @staticmethod
    def _breakdown_html(pred: dict, spec: ScanSpec, model: TimingModel, notes: Sequence[str]) -> str:
        total = max(pred["total"], 1e-9)
        lines = [f"<b>Scan time estimate: {fmt_duration(pred['total'])}</b> ({pred['n_points']} points"
                 + (", dI/dP: 2 acquisitions each" if spec.didp else "") + ")", "<table>"]
        for name, s in sorted(pred["parts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"<tr><td>{name}</td><td align=right>&nbsp;{fmt_duration(s)}</td>"
                         f"<td align=right>&nbsp;{100.0 * s / total:.0f}%</td></tr>")
        lines.append("</table>")
        acq = f"Per acquisition ≈ {pred['acq_s']:.2f} s"
        if spec.measure_pl:
            acq += (f" (PL {pred['pl_s']:.2f} s = {spec.accum_n} × ({spec.exposure_ms / 1000.0:.3g} s exp + "
                    f"{model.frame_oh(spec):.2f} s readout/overhead) + {model.save.value:.2f} s save")
            acq += ")" if not spec.measure_trpl else ""
        if spec.measure_trpl:
            acq += (" (" if not spec.measure_pl else "; ") + f"TRPL {pred['trpl_s']:.2f} s)"
        lines.append(acq)
        sd, gd = pred["stage_detail"], pred["gate_detail"]
        vmax, accel = spec.stage_velocity or (DEFAULT_STAGE_VEL, DEFAULT_STAGE_ACC)
        if spec.sweep_power:
            lines.append(f"Stage: {sd['calls']:.0f} moves, {sd['steps']:.0f} substeps of ≤{spec.ramp_step_deg:g}°, "
                         f"{sd['motion_s']:.0f} s motion at {vmax:g}°/s, {accel:g}°/s²"
                         + ("" if spec.stage_velocity else " (default velocity)"))
        if gd["calls"]:
            lines.append(f"Gates: {gd['calls']:.0f} ramps, {gd['microsteps']:.0f} microsteps of "
                         f"{spec.ramp_step_v * 1000:g} mV with {spec.ramp_dwell_s * 1000:g} ms dwell, "
                         + ("V/I read after each" if spec.ramp_micro_read else "V/I read once per ramp")
                         + f" ({model.gate_model(spec).value[0] + spec.ramp_dwell_s:.2f} s per microstep"
                         + (", Keithley 2400" if spec.gate_2400 else ", Keithley 2450-type" if spec.gate_2400 is False
                            else ", gate model not known yet: 2450-type assumed") + ")")
        lines.append(f"Timing model: learned from {model.scans} scan(s)" if model.scans
                     else "Timing model: defaults (not calibrated yet; improves after the first scan)")
        lines.extend(notes)
        return "<br>".join(lines)

    # ---- running
    def start(self, tracker: ScanTracker) -> None:
        self._tracker = tracker
        self.bar.setRange(0, max(1, tracker.n_points))
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.setToolTip(f"Estimated at start: {fmt_duration(tracker.estimate_s)} ({tracker.n_points} points)")
        self._mini_sticky = False
        self._tick.start()
        self.refresh_running()

    def refresh_running(self) -> None:
        tr = self._tracker
        if tr is None:
            self._tick.stop()
            return
        now = time.monotonic()
        left = tr.left_s(now)
        eta = fmt_clock(time.time() + left)
        elapsed = fmt_duration(tr.elapsed_s(now))
        self.bar.setValue(min(tr.done, self.bar.maximum()))
        self.label.setText(f"Elapsed {elapsed} | Left ≈ {fmt_duration(left)} | "
                           f"Total ≈ {fmt_duration(tr.total_s(now))} | ETA {eta}")
        self._push_mini("running", f"Left ≈ {fmt_duration(left)}",
                        f"{tr.done}/{tr.n_points} pts · ETA {eta} · elapsed {elapsed}", tr.done, tr.n_points)

    def finish(self, status: str) -> None:
        tr, self._tracker = self._tracker, None
        self._tick.stop()
        self.bar.setVisible(False)
        if tr is None:
            return
        actual = tr.elapsed_s()
        dev = 100.0 * (actual - tr.estimate_s) / tr.estimate_s if tr.estimate_s > 0 else 0.0
        if status == "ok":
            self._last_summary = f"Last scan {fmt_duration(actual)} ({dev:+.0f}% vs est.)"
        else:
            word = status if status in ("aborted", "trip") else "error"
            self._last_summary = f"Last scan {word} after {fmt_duration(actual)} ({tr.done}/{tr.n_points} pts)"
        self._last_detail = (f"Last scan: {fmt_duration(actual)}, estimated {fmt_duration(tr.estimate_s)} "
                             f"({dev:+.1f}%); {status}; {tr.done}/{tr.n_points} points")
        self.label.setText(self._last_summary)
        self.setToolTip(self._last_detail)
        ended = f"ended {time.strftime('%H:%M')}"
        if status == "ok":
            self._push_mini("ok", f"Done in {fmt_duration(actual)}",
                            f"{tr.done}/{tr.n_points} pts · {dev:+.0f}% vs est. · {ended}")
        else:
            word = {"aborted": "Aborted", "trip": "Tripped"}.get(status, "Error")
            self._push_mini("bad", f"{word} after {fmt_duration(actual)}", f"{tr.done}/{tr.n_points} pts · {ended}")
        self._mini_sticky = True
