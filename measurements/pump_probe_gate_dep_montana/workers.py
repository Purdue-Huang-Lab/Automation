"""Worker threads of the Montana pump-probe GUI.

* ``LiveCaptureThread``  -- continuous capture + ON/OFF analysis (old ``_CaptureThread``).
* ``DelayScanThread``    -- one delay scan at the current conditions (old ``_DelayScanThread``).
* ``GatePowerSweepThread`` -- delay scan at every (power, gate step, polarization):
  the old ``_GateSweepThread`` delay-scan / save logic, with the PL+TRPL v2 gate steps
  (Dual / Front / Back, V0 + k * step, optional grid) and gate ramp safety, and one ND
  wheel for the power.
* ``GateRampThread``     -- manual "Apply V" ramp (pl_trpl v2).

Every scan thread publishes ``activity = (text, t_start, limit_s)`` before each blocking
hardware call (GUI watchdog), and has ``abandon()``: the Terminate button stops it and
leaves it behind if a driver call never returns; an abandoned thread saves nothing more.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore

from keithley.keithley_wrapper import ComplianceTrip, SetpointError, SweepController, VIReading  # noqa: F401
from measurements.pump_probe_gate_dep_montana import config as C
from measurements.pump_probe_gate_dep_montana import ta_core
from rot.rot_wrapper import MotionController
from smc100.smc100_wrapper import MoveController as StageMoveController

SCAN_LOG_NAME = "scan_log.csv"
SCAN_LOG_HEADER = ["prefix", "start", "end", "n_delays", "power_idx", "wheel_angle_deg", "power_w",
                   "v_front_set", "v_back_set", "v_front_read", "i_front_a", "v_back_read", "i_back_a",
                   "polarization", "pol_angle_deg", "status"]


def append_scan_log(save_dir: str, row: Sequence) -> None:
    """One row of ``scan_log.csv`` (header written with the first row).  Raises on I/O errors."""
    path = os.path.join(save_dir, SCAN_LOG_NAME)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(SCAN_LOG_HEADER)
        w.writerow(list(row) + [""] * (len(SCAN_LOG_HEADER) - len(row)))


def capture_limit_s(n_frames: int) -> float:
    return C.WATCHDOG_CAPTURE_S + C.WATCHDOG_CAPTURE_PER_FRAME_S * max(1, int(n_frames))


def _stop_stage_async(stage) -> None:
    """``stage.stop()`` in a daemon thread: a hung serial port must not freeze the GUI."""
    if stage is None:
        return

    def run():
        try:
            stage.stop()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


class _Abort(RuntimeError):
    pass


class _RampTrip(RuntimeError):
    """A gate ramp reached the current compliance."""


def gate_ramp_microsteps(start: Optional[float], target: float, step_v: float) -> int:
    """Microsteps of ``KeithleySMU.ramp_to_voltage`` (pl_trpl v2 scan_time.gate_ramp_microsteps)."""
    start = 0.0 if start is None else float(start)
    step = max(1e-6, abs(float(step_v)))
    d = abs(float(target) - start)
    if d <= step:
        return 1
    n_full = int(d // step)
    return n_full + (1 if abs(d - n_full * step) > 1e-9 else 0)


# ============================================================================= live capture
class LiveCaptureThread(QtCore.QThread):
    """Continuous capture + analysis in background (old ``_CaptureThread``)."""

    result_ready = QtCore.pyqtSignal(object)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, camera, n_frames: int, roi, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.n_frames = int(n_frames)
        self.roi = tuple(roi)
        self._running = True
        self.activity = None

    def run(self):
        while self._running:
            try:
                self.activity = (f"camera: live capture of {self.n_frames} frames", time.monotonic(),
                                 capture_limit_s(self.n_frames))
                frames = self.camera.capture_frames(self.n_frames)
                results = ta_core.analyze_frames(frames, *self.roi)
                if self._running:
                    self.result_ready.emit(results)
            except Exception as e:
                if self._running:
                    self.error_occurred.emit(str(e))
                break
        self.activity = None

    def stop(self):
        self._running = False

    def abandon(self):
        self.stop()


# ============================================================================= delay scan
class _DelayScanMixin:
    """The delay loop shared by both scan threads (old ``_run_delay_scan``)."""

    activity = None  # (text, monotonic start, limit s) of the blocking call in progress, for the watchdog
    _terminated = False  # abandoned by Terminate: save nothing more

    def _set_activity(self, text: str, limit_s: float) -> None:
        self.activity = (text, time.monotonic(), float(limit_s)) if text else None

    def _delay_loop(self, label: str) -> List[Dict]:
        scan_results = []
        n = len(self.time_delays_ps)
        cap_limit = capture_limit_s(self.n_frames)
        for i, (delay_ps, position_mm) in enumerate(zip(self.time_delays_ps, self.positions_mm)):
            if not self._running:
                self.status.emit(f"Stopped at delay {i}/{n}")
                break
            avg_times = ta_core.avg_for_delay(delay_ps, self.avg_map, self.default_avg)
            self._set_activity(f"delay stage: move to {position_mm:.4f} mm (delay {i+1}/{n}, {delay_ps:.2f} ps)",
                               C.WATCHDOG_MOVE_S)
            try:
                self.stage.move_to(position_mm, wait=True, controller=self._stage_ctrl)
            except Exception:
                if not self._running:  # Stop aborted the move: keep the delays done so far
                    break
                raise
            self.current_delay_ps = delay_ps
            self._set_activity("delay stage: settle", ta_core.STAGE_SETTLE_S + C.WATCHDOG_EXTRA_S)
            time.sleep(ta_core.STAGE_SETTLE_S)

            def on_capture(avg_idx, avg_n, _i=i, _d=delay_ps):
                self.status.emit(f"{label}Delay {_i+1}/{n} ({_d:.2f} ps) | Avg {avg_idx+1}/{avg_n}")
                self._set_activity(f"camera: capture {self.n_frames} frames (delay {_i+1}/{n}, {_d:.2f} ps, "
                                   f"avg {avg_idx+1}/{avg_n})", cap_limit)

            final_results = ta_core.acquire_delay_point(self.camera, self.n_frames, self.roi, avg_times,
                                                        lambda: self._running, on_capture)
            self._set_activity("processing", C.WATCHDOG_EXTRA_S)
            if final_results is None or self._terminated:
                break
            step_info = {'time_delay_ps': delay_ps, 'position_mm': position_mm,
                         'results': final_results, 'avg_times_used': avg_times}
            scan_results.append(step_info)
            self.step_result.emit(final_results, tuple(self.roi), dict(step_info, index=i, total=n))
            self._after_delay_point()
        return scan_results

    def _after_delay_point(self) -> None:
        pass


class DelayScanThread(QtCore.QThread, _DelayScanMixin):
    """One delay scan (old ``_DelayScanThread``).  ``finished_scan(results, completed)``."""

    status = QtCore.pyqtSignal(str)
    step_result = QtCore.pyqtSignal(object, tuple, dict)
    finished_scan = QtCore.pyqtSignal(list, bool)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, camera, stage, time_delays_ps, positions_mm, n_frames, roi, avg_map, default_avg,
                 parent=None):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.time_delays_ps = list(time_delays_ps)
        self.positions_mm = list(positions_mm)
        self.n_frames = int(n_frames)
        self.roi = tuple(roi)
        self.avg_map = dict(avg_map)
        self.default_avg = int(default_avg)
        self.current_delay_ps = float("nan")
        self._running = True
        self._stage_ctrl = StageMoveController()

    def run(self):
        try:
            results = self._delay_loop("")
            if not self._terminated:
                self.finished_scan.emit(results, self._running)
        except Exception as e:
            traceback.print_exc()
            if not self._terminated:
                self.error_occurred.emit(str(e) if self._running else f"stopped ({e})")
        finally:
            self.activity = None

    def stop(self):
        self._running = False
        self._stage_ctrl.abort()

    def abandon(self):
        """Terminate: stop, and save / report nothing if the stuck call ever returns."""
        self._terminated = True
        self.stop()
        _stop_stage_async(self.stage)


# ============================================================================= gate ramp (manual)
class GateRampThread(QtCore.QThread):
    """Manual gate ramp (pl_trpl v2 ``GateRampThread``)."""

    progress = QtCore.pyqtSignal(float, float)  # V, I(A) at every microstep
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)  # ok / aborted / error: ...

    def __init__(self, dev, target_v: float, icomp_a: Optional[float], controller: SweepController,
                 ramp_step_v: float, ramp_dwell_s: float, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.target_v = float(target_v)
        self.icomp_a = icomp_a
        self.controller = controller
        self.ramp_step_v = float(ramp_step_v)
        self.ramp_dwell_s = float(ramp_dwell_s)

    def run(self):
        try:
            if self.icomp_a is not None:
                self.dev.set_compliance(float(self.icomp_a))
            self.dev.set_output(True)
            self.status.emit(f"Ramping to {self.target_v:.6g} V...")
            vi_last = self.dev.ramp_to_voltage(
                self.target_v, ramp_step_v=self.ramp_step_v, ramp_dwell_s=self.ramp_dwell_s,
                controller=self.controller, micro_read=True,
                on_microstep=lambda vi: self.progress.emit(vi.v, vi.i), verify=True)
            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.progress.emit(vi_last.v, vi_last.i)
                self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")


# ============================================================================= gate x power sweep
class GatePowerSweepThread(QtCore.QThread, _DelayScanMixin):
    """Full delay scan (one para/sum pair) at every power x gate step x polarization.

    power_plan: list of {"angle": deg or None, "power_w": W or nan, "idx": i}; ``[None]`` = no power sweep.
    Loop orders: "power_outer" (voltages per power, snaked on alternate powers, as pl_trpl v2),
    "voltage_outer" (powers per voltage), or grid (power -> gate A -> gate B snaked).

    The loops are flattened once into a fixed list of actions (power move / gate ramp /
    delay scan #k), so scan k is the same scan in a resumed sweep.  With ``sweep_progress``
    (``SweepProgress``) the scans already done are skipped; the first scan measured after
    skipped ones first brings the wheel and the gates to its own setpoints.
    """

    status = QtCore.pyqtSignal(str)
    step_result = QtCore.pyqtSignal(object, tuple, dict)
    vi_ready = QtCore.pyqtSignal(object)  # {"v_front", "i_front"} and / or {"v_back", "i_back"}
    scan_saved = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, int)  # delay scans done, total
    done = QtCore.pyqtSignal(str)  # ok / aborted / trip: ... / error: ...

    def __init__(self, *, camera, stage, time_delays_ps, positions_mm, n_frames, roi, avg_map, default_avg,
                 power_plan, wheel, wheel_step_deg, power_settle_s,
                 sweep_voltage, sweep_mode, grid, front_dev, back_dev,
                 front_v0, front_step, back_v0, back_step, nsteps, nsteps_b,
                 front_icomp_a, back_icomp_a, ramp_step_v, ramp_dwell_s, long_ramp_steps,
                 gate_settle_s, order, home_gates_after,
                 pol_enabled, pol_stage, lh_angle, rh_offset, pol_settle_s,
                 save_dir, save_base, timestamp, crop, t0_ps, pixel_size, crop_h, camera_width,
                 power_unit="nW", sweep_progress=None, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.time_delays_ps = list(time_delays_ps)
        self.positions_mm = list(positions_mm)
        self.n_frames = int(n_frames)
        self.roi = tuple(roi)
        self.avg_map = dict(avg_map)
        self.default_avg = int(default_avg)
        self.power_plan = list(power_plan) or [None]
        self.wheel = wheel
        self.wheel_step_deg = float(wheel_step_deg)
        self.power_settle_s = float(power_settle_s)
        self.sweep_voltage = bool(sweep_voltage)
        self.sweep_mode = sweep_mode  # "dual" / "front" / "back"
        self.grid = bool(grid) and self.sweep_voltage and sweep_mode == "dual"
        self.front_dev = front_dev
        self.back_dev = back_dev
        self.front_v0, self.front_step = float(front_v0), float(front_step)
        self.back_v0, self.back_step = float(back_v0), float(back_step)
        self.nsteps, self.nsteps_b = max(1, int(nsteps)), max(1, int(nsteps_b))
        self.front_icomp_a, self.back_icomp_a = front_icomp_a, back_icomp_a
        self.ramp_step_v, self.ramp_dwell_s = float(ramp_step_v), float(ramp_dwell_s)
        self.long_ramp_steps = int(long_ramp_steps)
        self.gate_settle_s = float(gate_settle_s)
        self.order = order
        self.home_gates_after = bool(home_gates_after)
        self.pol_enabled = bool(pol_enabled) and pol_stage is not None
        self.pol_stage = pol_stage
        self.lh_angle = float(lh_angle)
        self.rh_offset = float(rh_offset)
        self.pol_settle_s = float(pol_settle_s)
        self.save_dir = save_dir
        self.save_base = save_base
        self.timestamp = timestamp
        self.crop = crop
        self.t0_ps = float(t0_ps)
        self.pixel_size = float(pixel_size)
        self.crop_h = int(crop_h)
        self.camera_width = int(camera_width)
        self.power_unit = power_unit
        self.sweep_progress = sweep_progress

        self.current_delay_ps = float("nan")
        self._running = True
        self._terminated = False
        self.activity = None
        self._stage_ctrl = StageMoveController()
        self._wheel_ctrl = MotionController()
        self._gate_controller = SweepController()
        self._ramp_trip: Optional[str] = None
        self._last_vi: Dict[str, Tuple[float, float]] = {}
        self._scans_done = 0
        self._entry = None
        self._vset: Tuple[Optional[float], Optional[float]] = (None, None)
        self._actions = self._plan_actions()

    # ------------------------------------------------------------------ plan size
    def gate_steps(self, reverse: bool = False) -> List[Tuple[int, Optional[float], Optional[float]]]:
        """pl_trpl v2 ``_iter_voltage_steps``: (k, vf, vb), None for a gate that is not swept."""
        if not self.sweep_voltage:
            return [(0, None, None)]
        idx = range(self.nsteps - 1, -1, -1) if reverse else range(self.nsteps)
        out = []
        for k in idx:
            vf = float(self.front_v0 + k * self.front_step) if self.sweep_mode in ("front", "dual") else None
            vb = float(self.back_v0 + k * self.back_step) if self.sweep_mode in ("back", "dual") else None
            out.append((k, vf, vb))
        return out

    def grid_steps(self):
        """pl_trpl v2 ``_iter_grid_steps``: hold A (front), sweep B (back), B snaked."""
        counter = 0
        for ai in range(self.nsteps):
            vf = float(self.front_v0 + ai * self.front_step)
            b_idx = range(self.nsteps_b - 1, -1, -1) if (ai % 2 == 1) else range(self.nsteps_b)
            for bi in b_idx:
                yield counter, vf, float(self.back_v0 + bi * self.back_step)
                counter += 1

    def total_scans(self) -> int:
        return sum(1 for a in self._actions if a[0] == "scan")

    def scan_labels(self) -> List[str]:
        """One text per scan, in scan order (the sweep state file and the Resume dialog)."""
        out = []
        for a in self._actions:
            if a[0] == "scan":
                _, _idx, entry, vf, vb, pol_tag, _pa = a
                out.append(self._scan_label(entry, vf, vb, pol_tag, with_power=True) or "single scan")
        return out

    # ------------------------------------------------------------------ plan
    def _plan_actions(self) -> List[tuple]:
        """The sweep loops as a flat list, in execution order:

        ("power", entry) / ("gates", vf, vb, vset) / ("scan", idx, entry, vf, vb, pol_tag, pol_angle).
        ``vf`` / ``vb`` None in a gates action = that gate is not ramped there (grid: A unchanged);
        ``vset`` = the (front, back) setpoints that hold from then on.
        """
        acts: List[tuple] = []
        n_scan = [0]

        def scans(entry, vf, vb):
            for pol_tag, pol_angle in self._pol_pairs():
                acts.append(("scan", n_scan[0], entry, vf, vb, pol_tag, pol_angle))
                n_scan[0] += 1

        if self.grid:
            for entry in self.power_plan:
                acts.append(("power", entry))
                last_vf = None
                for _c, vf, vb in self.grid_steps():
                    acts.append(("gates", vf if vf != last_vf else None, vb, (vf, vb)))
                    last_vf = vf
                    scans(entry, vf, vb)
        elif self.order == "voltage_outer":
            for _k, vf, vb in self.gate_steps():
                acts.append(("gates", vf, vb, (vf, vb)))
                for entry in self.power_plan:
                    acts.append(("power", entry))
                    scans(entry, vf, vb)
        else:
            snake = self.sweep_voltage and self.nsteps > 1 and len(self.power_plan) > 1
            forward = True
            for entry in self.power_plan:
                acts.append(("power", entry))
                for _k, vf, vb in self.gate_steps(reverse=snake and not forward):
                    acts.append(("gates", vf, vb, (vf, vb)))
                    scans(entry, vf, vb)
                if snake:
                    forward = not forward
        return acts

    def _done_count(self) -> int:
        return self.sweep_progress.n_done if self.sweep_progress is not None else self._scans_done

    # ------------------------------------------------------------------ run
    def run(self):
        status = "ok"
        self._trip: Optional[str] = None
        try:
            self.progress.emit(self._done_count(), self.total_scans())
            self._run_plan()
            if not self._running:
                status = "aborted"
            elif self.home_gates_after and self.sweep_voltage:
                self._home_gates()
        except _Abort:
            status = "aborted"
        except (_RampTrip, ComplianceTrip) as exc:
            status = f"trip: {exc}"
        except Exception as exc:
            traceback.print_exc()
            status = "aborted" if not self._running else f"error: {type(exc).__name__}: {exc}"
        finally:
            if self._trip:
                status = f"trip: {self._trip}"
            if self._terminated:
                status = "terminated"
            elif self.sweep_progress is not None:
                self.sweep_progress.end(status)
            self.activity = None
            self.done.emit(status)

    def stop(self):
        self._running = False
        self._stage_ctrl.abort()
        self._wheel_ctrl.abort()
        self._gate_controller.abort()
        _stop_stage_async(self.stage)

    def abandon(self):
        """Terminate: stop, and save / log nothing if the stuck call ever returns."""
        self._terminated = True
        self.stop()

    def _check(self):
        if not self._running:
            raise _Abort()

    # ------------------------------------------------------------------ loops
    def _run_plan(self) -> None:
        """Run the actions; skip the scans already done (resume).

        Power / gate actions are run just before the next measured scan, in plan order.
        After skipped scans (and at the start) the hardware state is unknown, so the wheel
        and both gates are first brought to the setpoints of the scan about to be measured
        (gates before power for "powers per voltage", power first otherwise, as the loops).
        """
        skip = self.sweep_progress.done_set() if self.sweep_progress is not None else set()
        synced = False
        pending: List[tuple] = []
        target_power = None
        target_vset: Tuple[Optional[float], Optional[float]] = (None, None)
        gates_first = self.order == "voltage_outer" and not self.grid
        for act in self._actions:
            self._check()
            if act[0] == "power":
                target_power = act[1]
                pending.append(act)
                continue
            if act[0] == "gates":
                target_vset = act[3]
                pending.append(act)
                continue
            _, idx, entry, vf, vb, pol_tag, pol_angle = act
            if idx in skip:
                synced = False
                pending = []
                continue
            if not synced:
                full = ("gates", target_vset[0], target_vset[1], target_vset)
                pending = [full, ("power", target_power)] if gates_first else [("power", target_power), full]
                synced = True
            for p in pending:
                self._check()
                if p[0] == "power":
                    self._move_power(p[1])
                else:
                    self._ramp_gates(p[1], p[2], vset=p[3])
            pending = []
            self._scan_point(idx, entry, vf, vb, pol_tag, pol_angle)

    # ------------------------------------------------------------------ power / polarization
    def _move_power(self, entry) -> None:
        self._entry = entry
        if entry is None or self.wheel is None or entry.get("angle") is None:
            return
        angle = float(entry["angle"])
        p = entry.get("power_w")
        ptxt = f" ({p / ta_core_unit(self.power_unit):.4g} {self.power_unit})" if p is not None and math.isfinite(p) else ""
        self.status.emit(f"Power {entry.get('idx', 0) + 1}/{len(self.power_plan)}: wheel -> {angle:.3f} deg{ptxt}")
        self._set_activity(f"ND wheel: move to {angle:.3f} deg", C.WATCHDOG_MOVE_S)
        self.wheel.move_to(angle, step_deg=self.wheel_step_deg, controller=self._wheel_ctrl)
        self._check()
        self._set_activity("ND wheel: settle", self.power_settle_s + C.WATCHDOG_EXTRA_S)
        time.sleep(self.power_settle_s)

    def _pol_pairs(self):
        if self.pol_enabled:
            return [('LH', self.lh_angle), ('RH', self.lh_angle + self.rh_offset)]
        return [(None, None)]

    def _move_pol(self, pol_angle):
        if self.pol_enabled and pol_angle is not None:
            self._set_activity(f"polarization stage: move to {pol_angle:.2f} deg", C.WATCHDOG_MOVE_S)
            self.pol_stage.move_to(pol_angle, step_deg=360.0, controller=self._wheel_ctrl)
            self._check()
            self._set_activity("polarization stage: settle", self.pol_settle_s + C.WATCHDOG_EXTRA_S)
            time.sleep(self.pol_settle_s)

    # ------------------------------------------------------------------ one (power, gate, pol) scan
    def _scan_label(self, entry, vf, vb, pol_tag, with_power: bool = False) -> str:
        angle = entry.get("angle") if entry is not None else None
        label = ""
        if vf is not None:
            label += f"Vf={vf:+.3f}V "
        if vb is not None:
            label += f"Vb={vb:+.3f}V "
        label += f"ND {angle:.1f} deg " if angle is not None else ""
        p = entry.get("power_w") if entry is not None else None
        if with_power and p is not None and math.isfinite(p):
            label += f"({p / ta_core_unit(self.power_unit):.4g} {self.power_unit}) "
        label += f"{pol_tag} " if pol_tag else ""
        return label.strip()

    def _scan_point(self, idx, entry, vf, vb, pol_tag, pol_angle) -> None:
        angle = entry.get("angle") if entry is not None else None
        power_w = entry.get("power_w") if entry is not None else None
        self._check()
        self._move_pol(pol_angle)
        t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = self._scan_label(entry, vf, vb, pol_tag)
        label = (label + " | ") if label else ""
        if self.sweep_progress is not None:
            self.sweep_progress.start_scan(idx)
        delay_results = self._delay_loop(label)
        if self._terminated:  # abandoned: this scan is redone on Resume
            return
        if delay_results:
            prefix = ta_core.scan_prefix(self.save_base, vf, vb, angle, pol_tag, self.timestamp,
                                         power_w, self.power_unit)
            self._set_activity("saving", C.WATCHDOG_EXTRA_S)
            try:
                para, summ = ta_core.save_scan(
                    self.save_dir, prefix, delay_results, pixel_size=self.pixel_size, crop=self.crop,
                    camera_width=self.camera_width, crop_h=self.crop_h, t0_ps=self.t0_ps,
                    angle_deg=angle, n_frames=self.n_frames)
                status = "complete" if len(delay_results) == len(self.time_delays_ps) else "partial"
            except Exception as exc:
                para = summ = ""
                status = f"save failed: {exc}"
                self.status.emit(f"Warning: save failed: {exc}")
            info = {"prefix": prefix, "para": para, "sum": summ, "vf": vf, "vb": vb, "angle": angle,
                    "power_w": power_w, "pol": pol_tag, "n_delays": len(delay_results), "status": status,
                    "results": delay_results, "scan_idx": idx, "total": self.total_scans()}
            self._log_scan(info, t_start, pol_angle)
            self._scans_done += 1
            if self.sweep_progress is not None:
                self.sweep_progress.finish_scan(idx, prefix, para, summ, status == "complete")
                if self.sweep_progress.last_save_error:
                    self.status.emit(f"Warning: sweep state file not written: {self.sweep_progress.last_save_error}")
            self.scan_saved.emit(info)
            self.progress.emit(self._done_count(), self.total_scans())
        self._check()

    def _after_delay_point(self) -> None:
        """Read the gates once per delay step (old GUI), check the compliance (pl_trpl v2)."""
        for which, dev, icomp in (("front", self.front_dev, self.front_icomp_a),
                                  ("back", self.back_dev, self.back_icomp_a)):
            if dev is None:
                continue
            self._set_activity(f"{which} gate: read V/I", C.WATCHDOG_EXTRA_S)
            v, i = self._read_vi(dev, icomp)
            if v is None:
                continue
            self._last_vi[which] = (v, i)
            self.vi_ready.emit({f"v_{which}": v, f"i_{which}": i})
            self._check_trip(i, icomp, which.capitalize())

    def _log_scan(self, info: Dict, t_start: str, pol_angle) -> None:
        vfr, ifr = self._last_vi.get("front", (None, None))
        vbr, ibr = self._last_vi.get("back", (None, None))
        entry = self._entry or {}
        row = [info["prefix"], t_start, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), info["n_delays"],
               entry.get("idx", ""), info["angle"] if info["angle"] is not None else "",
               info["power_w"] if info["power_w"] is not None else "",
               info["vf"] if info["vf"] is not None else "", info["vb"] if info["vb"] is not None else "",
               vfr if vfr is not None else "", ifr if ifr is not None else "",
               vbr if vbr is not None else "", ibr if ibr is not None else "",
               info["pol"] or "", pol_angle if pol_angle is not None else "", info["status"]]
        try:
            append_scan_log(self.save_dir, row)
        except Exception as exc:
            self.status.emit(f"Warning: scan log not written: {exc}")

    # ------------------------------------------------------------------ gates (pl_trpl v2 logic)
    def _ramp_gates(self, vf, vb, vset=None) -> None:
        """Ramp the swept gate(s) (front first, then back), then the gate settle (old GUI)."""
        if not self.sweep_voltage:
            return
        self._vset = vset if vset is not None else (vf, vb)
        ramped = False
        if vf is not None and self.front_dev is not None:
            self.status.emit(f"Ramping front gate to {vf:.3f} V ...")
            self._gate_ramp(self.front_dev, vf, self.front_icomp_a, "front")
            ramped = True
        if vb is not None and self.back_dev is not None:
            self.status.emit(f"Ramping back gate to {vb:.3f} V ...")
            self._gate_ramp(self.back_dev, vb, self.back_icomp_a, "back")
            ramped = True
        self._check()
        if ramped:
            self._set_activity("gates: settle", self.gate_settle_s + C.WATCHDOG_EXTRA_S)
            time.sleep(self.gate_settle_s)

    def _home_gates(self) -> None:
        self.status.emit("Ramping the swept gates to 0 V ...")
        if self.sweep_mode in ("front", "dual") and self.front_dev is not None:
            self._gate_ramp(self.front_dev, 0.0, self.front_icomp_a, "front")
        if self.sweep_mode in ("back", "dual") and self.back_dev is not None:
            self._gate_ramp(self.back_dev, 0.0, self.back_icomp_a, "back")

    def _gate_ramp(self, dev, target_v: float, icomp_a: Optional[float], which: str) -> VIReading:
        tag = which.capitalize()
        start = dev.read_source_setpoint() if hasattr(dev, "read_source_setpoint") else \
            float(getattr(dev, "_last_set_v", target_v))
        n_steps = gate_ramp_microsteps(start, target_v, self.ramp_step_v)
        long_ramp = n_steps > self.long_ramp_steps
        self._set_activity(f"{which} gate: ramp to {float(target_v):.6g} V ({n_steps} steps)",
                           n_steps * (self.ramp_dwell_s + 0.5) + C.WATCHDOG_EXTRA_S)
        if long_ramp:
            self.status.emit(f"{tag} gate: ramping {start:.6g} -> {float(target_v):.6g} V ({n_steps} steps of "
                             f"{self.ramp_step_v * 1000:g} mV, V/I read at every step)")

        def emit(vi):
            self.vi_ready.emit({f"v_{which}": vi.v, f"i_{which}": vi.i})

        for attempt in range(2):
            try:
                if icomp_a is not None:
                    dev.set_compliance(float(icomp_a))
                dev.set_output(True)
                callback = emit
                if long_ramp and icomp_a is not None:
                    lim = abs(float(icomp_a))

                    def callback(vi, _lim=lim):
                        emit(vi)
                        if np.isfinite(vi.i) and abs(vi.i) >= 0.98 * _lim and self._ramp_trip is None:
                            self._ramp_trip = (f"{tag} gate at the current compliance during the ramp: "
                                               f"V={vi.v:.6g} V, I={vi.i * 1e9:.4g} nA "
                                               f"(limit {_lim * 1e9:g} nA); ramp stopped")
                            self._gate_controller.abort()
                vi = dev.ramp_to_voltage(target_v, ramp_step_v=self.ramp_step_v, ramp_dwell_s=self.ramp_dwell_s,
                                         controller=self._gate_controller, micro_read=long_ramp,
                                         on_microstep=callback, verify=True, sync_each_step=not long_ramp)
                if self._ramp_trip is not None:
                    self._running = False
                    raise _RampTrip(self._ramp_trip)
                if not long_ramp and vi is not None and np.isfinite(vi.v) and np.isfinite(vi.i):
                    emit(vi)
                return vi
            except _RampTrip:
                raise
            except SetpointError as exc:
                why = ""
                if icomp_a is not None and np.isfinite(exc.i_read) and abs(exc.i_read) >= 0.9 * abs(icomp_a):
                    why = (f" - the current is at the {abs(icomp_a) * 1e9:g} nA compliance (raise I comp, "
                           f"or ramp slower)")
                raise RuntimeError(f"{tag} gate: {exc}{why}") from exc
            except Exception as exc:
                if not self._running or self._gate_controller.is_aborted():
                    raise _Abort() from exc
                if attempt == 0 and self._recover_gate_device(dev, icomp_a, tag, exc):
                    self.status.emit(f"{tag} gate recovered; retrying ramp")
                    continue
                raise RuntimeError(f"{tag} gate ramp to {float(target_v):.6g} V failed: "
                                   f"{type(exc).__name__}: {exc}") from exc

    def _read_vi(self, dev, icomp_a) -> Tuple[Optional[float], Optional[float]]:
        tag = "Front" if dev is self.front_dev else "Back"
        for attempt in range(2):
            try:
                vi = dev.read_vi()
                return float(vi.v), float(vi.i)
            except Exception as exc:
                if attempt == 0 and self._recover_gate_device(dev, icomp_a, tag, exc):
                    continue
                self.status.emit(f"{tag} gate read failed: {exc}")
                return None, None
        return None, None

    def _recover_gate_device(self, dev, icomp_a, tag: str, reason=None) -> bool:
        try:
            why = f" ({type(reason).__name__}: {str(reason)[:120]})" if reason is not None else ""
            self.status.emit(f"{tag} gate comm lost{why}; reconnecting...")
            try:
                dev.close()
            except Exception:
                pass
            dev.open()
            if icomp_a is not None:
                dev.set_compliance(float(icomp_a))
            dev.set_output(True)
            return True
        except Exception as exc:
            self.status.emit(f"{tag} gate reconnect failed: {exc}")
            return False

    def _check_trip(self, i_a, limit_a, tag: str) -> None:
        if i_a is None or limit_a is None:
            return
        if abs(float(i_a)) > abs(float(limit_a)) + 1e-15 and not getattr(self, "_trip", None):
            # end the sweep after this delay point; the delays done so far are saved
            self._trip = f"{tag} TRIP: |I|={abs(i_a) * 1e9:.6g} nA > {abs(limit_a) * 1e9:.6g} nA"
            self.status.emit(self._trip)
            self._running = False
            self._gate_controller.abort()


def ta_core_unit(unit: str) -> float:
    return {"pW": 1e-12, "nW": 1e-9, "uW": 1e-6, "mW": 1e-3, "W": 1.0}.get(unit, 1e-9)
