"""Workers of the PL + TRPL v2 GUI: a copy of measurements/pl_power_voltage_workers.py
(all PL sweep modes unchanged) with the SPAD TRPL acquisition (SPAD23 or SPAD93,
``detector``) added to every sweep point.

measure_pl / measure_trpl select PL only, TRPL only or both.  With both, the
TRPL part (quick count snapshot + S stream over the pSPAD server) runs in a
helper thread at the same time as the Andor accumulation, and both are joined
before the point is reported, so a point takes about max(PL, TRPL).
"""

from __future__ import annotations

import contextlib
import csv
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from keithley.keithley_wrapper import ComplianceTrip, KeithleySMU, SetpointError, SweepController, VIReading
from measurements.dual_wheel_power_calibration import DualWheelPowerEntry
from measurements.power_calibration import PowerCalibEntry
from rot.rot_wrapper import MotionController
from flipper.flipper_wrapper import MoveController as FlipperMoveController
from measurements.pl_trpl_power_voltage_v2.scan_time import gate_ramp_microsteps
from measurements.pl_trpl_power_voltage_v2.detectors import SPAD23, profile
from measurements.pl_trpl_power_voltage_v2 import trpl_dynamic as tdyn
from spad23.spad23_analysis import pre_pulse_window


class _RampTrip(RuntimeError):
    """A sweep ramp reached the current compliance (the sweep ends with status "trip")."""


# TRPL stream failures (e.g. the pSPAD server sent no DONE after a long acquisition) are retried on a new
# connection; a point whose TRPL still fails is kept without TRPL and the sweep goes on.
TRPL_RETRIES = 2              # extra attempts of one stream
TRPL_RETRY_WAIT_S = 3.0       # pause before a retry (lets the server settle)
TRPL_DONE_GRACE_S = 60.0      # silence allowed after the measurement end before "no DONE" (client default 15 s)
MAX_TRPL_FAIL_POINTS = 3      # this many points in a row without TRPL: the sweep stops (server really down)
FAILED_DIR = "trpl_failed"    # partial data of failed attempts (outside the TRPL_*.csv glob)
PROBLEMS_NAME = "scan_problems.txt"


class _TrplFailed(RuntimeError):
    """A TRPL stream failed on every attempt."""


class GateRampThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float)  # V, I(A) at every microstep
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)  # ok/aborted/error:...

    def __init__(
        self,
        dev: KeithleySMU,
        target_v: float,
        icomp_a: Optional[float],
        controller: SweepController,
        ramp_step_v: float,
        ramp_dwell_s: float,
        parent=None,
    ):
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

            def on_microstep(vi: VIReading):
                self.progress.emit(vi.v, vi.i)

            vi_last = self.dev.ramp_to_voltage(
                self.target_v,
                ramp_step_v=self.ramp_step_v,
                ramp_dwell_s=self.ramp_dwell_s,
                controller=self.controller,
                micro_read=True,
                on_microstep=on_microstep,
                verify=True,
            )

            if self.controller.is_aborted():
                self.done.emit("aborted")
            else:
                self.progress.emit(vi_last.v, vi_last.i)
                self.done.emit("ok")
        except Exception as e:
            self.done.emit(f"error: {e}")


class PowerVoltageSweepThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    vi_ready = QtCore.pyqtSignal(object)
    spad_count_ready = QtCore.pyqtSignal(object)
    trpl_update = QtCore.pyqtSignal(object)
    trpl_done = QtCore.pyqtSignal(object)
    point_done = QtCore.pyqtSignal(object)
    power_done = QtCore.pyqtSignal(object)
    timing = QtCore.pyqtSignal(object)  # {"event": "start" | "point" | "end", "t": monotonic, ...} (scan time)
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cam,
        stage_a,
        stage_b,
        plan: List[dict],
        accum_n: int,
        exposure_ms: float,
        ramp_step_deg: float,
        metadata_base: dict,
        wavelength_axis_nm,
        *,
        sweep_power: bool,
        sweep_voltage: bool,
        sweep_mode: str,
        front_dev: Optional[KeithleySMU],
        back_dev: Optional[KeithleySMU],
        front_v0: float,
        front_step: float,
        back_v0: float,
        back_step: float,
        nsteps: int,
        front_icomp_a: Optional[float],
        back_icomp_a: Optional[float],
        ramp_step_v: float,
        ramp_dwell_s: float,
        settle_ms: float,
        zero_v_eps: float,
        zero_v_extra_settle_ms: float,
        save_dir: Optional[str],
        sweep_order: str = "power_outer",
        grid_sweep: bool = False,
        nsteps_a: int = 1,
        nsteps_b: int = 1,
        readout_rate: Optional[str] = None,
        preamp_gain: Optional[str] = None,
        output_amp: Optional[str] = None,
        measure_pl: bool = True,
        measure_trpl: bool = False,
        tcspc_host: str = "127.0.0.1",
        tcspc_port: int = 9998,
        tcspc_acq_s: float = 10.0,
        tcspc_bin_ps: int = 500,
        spad_count_integration_ms: int = 120,
        detector: str = SPAD23,
        didp: bool = False,
        flipper=None,
        pump2_on_position: int = 2,
        flipper_settle_ms: float = 200.0,
        pump2_tags: Optional[Dict[bool, str]] = None,
        ramp_micro_read: bool = False,
        long_ramp_steps: int = 50,
        trpl_dynamic: Optional["tdyn.DynamicSettings"] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.cam = cam
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.plan = plan
        self.accum_n = int(accum_n)
        self.exposure_ms = float(exposure_ms)
        self.ramp_step_deg = float(ramp_step_deg)
        self.metadata_base = metadata_base or {}
        self.wavelength_axis_nm = wavelength_axis_nm
        self.sweep_power = bool(sweep_power)
        self.sweep_voltage = bool(sweep_voltage)
        self.sweep_mode = sweep_mode
        self.front_dev = front_dev
        self.back_dev = back_dev
        self.front_v0 = float(front_v0)
        self.front_step = float(front_step)
        self.back_v0 = float(back_v0)
        self.back_step = float(back_step)
        self.nsteps = int(nsteps)
        self.front_icomp_a = front_icomp_a
        self.back_icomp_a = back_icomp_a
        self.ramp_step_v = float(ramp_step_v)
        self.ramp_dwell_s = float(ramp_dwell_s)
        self.settle_ms = float(settle_ms)
        self.zero_v_eps = float(zero_v_eps)
        self.zero_v_extra_settle_ms = float(zero_v_extra_settle_ms)
        self.save_dir = save_dir
        self.sweep_order = sweep_order  # "power_outer" or "voltage_outer"
        self.grid_sweep = bool(grid_sweep)
        self.nsteps_a = int(nsteps_a)
        self.nsteps_b = int(nsteps_b)
        self.readout_rate = readout_rate
        self.preamp_gain = preamp_gain
        self.output_amp = output_amp
        self.measure_pl = bool(measure_pl)
        self.measure_trpl = bool(measure_trpl)
        self.tcspc_host = str(tcspc_host)
        self.tcspc_port = int(tcspc_port)
        self.tcspc_acq_s = float(tcspc_acq_s)
        self.tcspc_bin_ps = int(tcspc_bin_ps)
        self.spad_count_integration_ms = max(20, int(spad_count_integration_ms))
        self.detector = profile(detector)  # SPAD23 / SPAD93: client classes
        # dI/dP: every point is acquired with pump 2 off, then on (filter flipper)
        self.didp = bool(didp)
        self.flipper = flipper
        self.pump2_on_position = 2 if int(pump2_on_position) == 2 else 1
        self.pump2_off_position = 1 if self.pump2_on_position == 2 else 2
        self.flipper_settle_ms = max(0.0, float(flipper_settle_ms))
        # read V/I after every ramp microstep (only feeds the live V/I plot; on a 2450 each read is two
        # measurements, ~0.4 s).  Off: one reading per ramp (the verify read) is shown instead.
        self.ramp_micro_read = bool(ramp_micro_read)
        # ramps with more microsteps (the long pre-ramps) always read V/I and stop at the compliance
        self.long_ramp_steps = int(long_ramp_steps)
        self._ramp_trip: Optional[str] = None
        self.pump2_tags = dict(pump2_tags or {False: "P2off", True: "P2on"})
        self._flipper_controller = FlipperMoveController()
        self._acq_meta: Dict[str, object] = {}  # per-acquisition header fields (pump 2 state)
        self._done_status: Optional[str] = None
        self._stop = False
        self._stage_controller = MotionController()
        self._gate_controller = SweepController()
        self._file_counters: Dict[str, int] = {}
        # time per part since the last acquired point (stage, gate, read, wait, frame, save, trpl, flip, acq)
        self._tacc: Dict[str, float] = {}
        self._point_count = 0
        # dynamic TRPL time (None = fixed tcspc_acq_s): chunks until the peak reaches the target
        self.trpl_dynamic = trpl_dynamic if self.measure_trpl else None
        self._dyn_log = tdyn.DynamicLog(self.save_dir) if self.trpl_dynamic is not None else None
        self._dyn_schedule: Optional[List[int]] = None  # dI/dP: pump 2 off's chunks (ms), repeated at pump 2 on
        self.trpl_problems: List[dict] = []  # failed TRPL attempts (session.md, scan_problems.txt)
        self._trpl_fail_points = 0  # consecutive points whose TRPL failed on every attempt

    @contextlib.contextmanager
    def _timed(self, key: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._tacc[key] = self._tacc.get(key, 0.0) + (time.monotonic() - t0)

    def _settle(self, ms: float) -> None:
        if ms > 0:
            with self._timed("wait"):
                time.sleep(max(0.0, ms / 1000.0))

    def _emit_point_timing(self) -> None:
        self._point_count += 1
        payload = dict(self._tacc, event="point", idx=self._point_count, t=time.monotonic())
        self._tacc = {}
        self.timing.emit(payload)

    def _set_done(self, status: str) -> None:
        """Record the final status; ``done`` is emitted once, after the cleanup in ``run``
        (so the flipper is back at pump 2 off before the GUI re-enables the controls)."""
        if self._done_status is None:
            self._done_status = str(status)

    def stop(self):
        self._stop = True
        try:
            self._gate_controller.abort()
        except Exception:
            pass
        try:
            self._flipper_controller.abort()
        except Exception:
            pass
        try:
            self._stage_controller.abort()
        except Exception:
            pass
        for stage in (self.stage_a, self.stage_b):
            try:
                if stage is not None:
                    stage.stop()
            except Exception:
                pass

    def _next_filename(self, base: str, ext: str = ".asc") -> str:
        idx = self._file_counters.get(base, 0)
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            name = f"{base}{suffix}{ext}"
            path = os.path.join(self.save_dir or ".", name)
            if not os.path.exists(path):
                self._file_counters[base] = idx + 1
                return path
            idx += 1

    @staticmethod
    def _fmt_v_label(v: Optional[float]) -> str:
        if v is None:
            return "na"
        s = f"{v:.6g}"
        return s.replace("-", "m").replace(".", "p")

    def _gate_ramp(self, dev: KeithleySMU, target_v: float, icomp_a: Optional[float],
                   on_microstep=None) -> VIReading:
        tag = self._gate_tag(dev)
        start = dev.read_source_setpoint() if hasattr(dev, "read_source_setpoint") else \
            float(getattr(dev, "_last_set_v", target_v))
        n_steps = gate_ramp_microsteps(start, target_v, self.ramp_step_v)
        long_ramp = n_steps > self.long_ramp_steps
        if long_ramp:  # a long (pre-)ramp: say so, it can take minutes
            self.status.emit(f"{tag} gate: ramping {start:.6g} -> {float(target_v):.6g} V ({n_steps} steps of "
                             f"{self.ramp_step_v * 1000:g} mV, V/I read at every step)")
        for attempt in range(2):
            try:
                if icomp_a is not None:
                    dev.set_compliance(float(icomp_a))
                dev.set_output(True)
                # long ramps read every microstep (live current, stop at the compliance); short ones
                # read only if "Read V/I during gate ramps" is on
                micro_read = on_microstep is not None and (self.ramp_micro_read or long_ramp)
                callback = on_microstep
                if micro_read and icomp_a is not None:
                    def callback(vi, _cb=on_microstep, _lim=abs(float(icomp_a)), _tag=tag):
                        _cb(vi)
                        if np.isfinite(vi.i) and abs(vi.i) >= 0.98 * _lim and self._ramp_trip is None:
                            self._ramp_trip = (f"{_tag} gate at the current compliance during the ramp: "
                                               f"V={vi.v:.6g} V, I={vi.i * 1e9:.4g} nA "
                                               f"(limit {_lim * 1e9:g} nA); ramp stopped")
                            self._gate_controller.abort()
                vi = dev.ramp_to_voltage(
                    target_v,
                    ramp_step_v=self.ramp_step_v,
                    ramp_dwell_s=self.ramp_dwell_s,
                    controller=self._gate_controller,
                    micro_read=micro_read,
                    on_microstep=callback,
                    verify=True,
                    # no reads: wait for each set to be done, so commands never pile up in the SMU
                    sync_each_step=not micro_read,
                )
                if self._ramp_trip is not None:
                    self.status.emit(self._ramp_trip)
                    self._set_done("trip")
                    self._stop = True
                    raise _RampTrip(self._ramp_trip)
                if on_microstep is not None and not micro_read and vi is not None and np.isfinite(vi.v) \
                        and np.isfinite(vi.i):
                    on_microstep(vi)  # the verify reading, for the V/I plot
                return vi
            except _RampTrip:
                raise  # the compliance stopped the ramp: not a comm problem, no reconnect
            except SetpointError as exc:
                # the link is fine, the output cannot get there: reconnecting would not help
                why = ""
                if icomp_a is not None and np.isfinite(exc.i_read) and abs(exc.i_read) >= 0.9 * abs(icomp_a):
                    why = (f" - the current is at the {abs(icomp_a) * 1e9:g} nA compliance (raise I comp, or "
                           f"ramp slower)")
                raise RuntimeError(f"{tag} gate: {exc}{why}") from exc
            except Exception as exc:
                if self._stop or self._gate_controller.is_aborted():
                    raise
                if attempt == 0 and self._recover_gate_device(dev, icomp_a=icomp_a, tag=tag, reason=exc):
                    self.status.emit(f"{tag} gate recovered; retrying ramp")
                    continue
                raise RuntimeError(f"{tag} gate ramp to {float(target_v):.6g} V failed: "
                                   f"{type(exc).__name__}: {exc}") from exc

    def _read_vi(
        self,
        dev: Optional[KeithleySMU],
        *,
        target_v: Optional[float] = None,
        icomp_a: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        if dev is None:
            return (None, None)

        tag = self._gate_tag(dev)
        for attempt in range(2):
            try:
                vi = dev.read_vi()
                return (float(vi.v), float(vi.i))
            except Exception as exc:
                if attempt == 0 and self._recover_gate_device(dev, icomp_a=icomp_a, tag=tag, reason=exc):
                    if target_v is not None:
                        self.status.emit(f"{tag} gate recovered; restoring {float(target_v):.6g} V")
                        self._gate_ramp(dev, float(target_v), icomp_a)
                    self.status.emit(f"{tag} gate recovered; retrying read")
                    continue
                raise exc

        return (None, None)

    def _gate_tag(self, dev: Optional[KeithleySMU]) -> str:
        if dev is self.front_dev:
            return "Front"
        if dev is self.back_dev:
            return "Back"
        return "Gate"

    def _recover_gate_device(self, dev: Optional[KeithleySMU], *, icomp_a: Optional[float], tag: str,
                             reason: Optional[BaseException] = None) -> bool:
        if dev is None:
            return False
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

    def _check_trip(self, i_a: Optional[float], limit_a: Optional[float], tag: str) -> None:
        if i_a is None or limit_a is None:
            return
        if abs(float(i_a)) > abs(float(limit_a)) + 1e-15:
            msg = f"{tag} TRIP: |I|={abs(i_a)*1e9:.6g} nA > {abs(limit_a)*1e9:.6g} nA"
            raise ComplianceTrip(msg, float(i_a))

    def _acquire_image(self, entry, v_front: Optional[float], v_back: Optional[float], i_front: Optional[float], i_back: Optional[float], base: str):
        accum_n = max(1, int(self.accum_n))
        sum_img = None
        last_fr = None

        for j in range(1, accum_n + 1):
            if self._stop or self._gate_controller.is_aborted():
                return None, "aborted"
            with self._timed("frame"):
                fr, err = self._grab_frame()
            if err:
                if err == "aborted":
                    return None, "aborted"
                return None, err
            img = fr.get("image")
            if img is None:
                return None, "error: acquire failed (no image)"
            # Keep the raw camera (native detector) orientation. The dispersive
            # axis is stored in native order so it stays aligned with the
            # wavelength calibration; display paths only flip vertically.
            img = np.asarray(img, dtype=np.float64)
            if sum_img is None:
                sum_img = img
            else:
                sum_img += img
            last_fr = fr
            fr_out = dict(fr)
            fr_out["image"] = sum_img
            fr_out["accum_idx"] = j
            fr_out["accum_n"] = accum_n
            fr_out["entry"] = entry
            self.frame_ready.emit(fr_out)

        if last_fr is None:
            return None, "error: acquire failed (no frames)"

        if base:
            filepath = self._next_filename(base)
            meta = dict(self.metadata_base)
            if entry is not None:
                if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
                    meta.update({
                        "power_w": entry.power_w,
                        "a_deg": entry.a_deg,
                        "b_deg": entry.b_deg,
                    })
                else:
                    meta.update({
                        "series": entry.series,
                        "power_w": entry.power_w,
                        "position_deg": entry.position_deg,
                    })
            meta.update({
                "front_v_set": v_front,
                "back_v_set": v_back,
                "front_i_read": i_front,
                "back_i_read": i_back,
                "sweep_mode": self.sweep_mode,
                "sweep_order": self.sweep_order,
            })
            meta.update(self._acq_meta)
            try:
                # Native detector column order, matching the wavelength axis
                # (column i <-> wavelength_axis_nm[i]) and the on-screen display.
                # None: save_ascii falls back to the camera's cached calibration (as the PL GUI always has)
                with self._timed("save"):
                    self.cam.save_ascii(filepath, sum_img, metadata=meta, wavelength_axis_nm=self.wavelength_axis_nm)
            except Exception as exc:
                return None, f"error: save failed: {exc}"
        else:
            filepath = None

        return {
            "image": sum_img,
            "accum_n": accum_n,
            "filepath": filepath,
        }, None

    @staticmethod
    def _usable_axis(wl) -> Optional[np.ndarray]:
        """The PL GUI's rule: finite and not all zero (save_ascii also needs one value per column)."""
        if wl is None:
            return None
        try:
            arr = np.asarray(wl, dtype=float).ravel()
        except Exception:
            return None
        if arr.size < 2 or not np.all(np.isfinite(arr)) or np.allclose(arr, 0.0):
            return None
        return arr

    def _resolve_wavelength_axis(self) -> None:
        """No axis from the GUI: read it here (this thread owns the camera; the spectrograph poll is
        paused) from the open spectrograph, else the camera's cache; say which one the files get."""
        source = "GUI"
        wl = self._usable_axis(self.wavelength_axis_nm)
        try:
            center_nm = float(self.metadata_base.get("center_wl_nm", 1.0))
        except Exception:
            center_nm = 1.0
        if wl is None and abs(center_nm) > 1e-9:  # center 0 nm: no axis (the PL GUI rule)
            spec = getattr(self.cam, "spec", None)
            if spec is not None and hasattr(spec, "get_calibration"):
                try:
                    wl = self._usable_axis(np.asarray(spec.get_calibration(), dtype=float) * 1e9)
                    source = "the spectrograph"
                except Exception as exc:
                    self.status.emit(f"Spectrograph calibration read failed: {type(exc).__name__}: {exc}")
                    wl = None
        if wl is None:
            wl = self._usable_axis(getattr(self.cam, "_wavelength_axis_nm", None))
            source = "the camera's cached calibration"
        if wl is None:
            self.status.emit("No wavelength axis: spectra saved against pixel columns")
            return
        self.wavelength_axis_nm = wl
        if source != "GUI":
            self.status.emit(f"Wavelength axis from {source} ({wl[0]:.2f} to {wl[-1]:.2f} nm)")

    def _grab_frame(self):
        retries = 1
        while True:
            if self._stop or self._gate_controller.is_aborted():
                return None, "aborted"
            try:
                fr = self.cam.get_frame()
            except Exception as exc:
                self.status.emit(f"Acquire failed: {exc}")
                return None, f"error: acquire failed: {exc}"
            if fr.get("ok"):
                return fr, None
            err = str(fr.get("err", "Acquire failed"))
            if self._stop:
                return None, "aborted"
            if ("timeout" in err.lower()) and (retries > 0):
                retries -= 1
                try:
                    timeout_scale = float(getattr(self.cam, "_timeout_scale", 2.0))
                    timeout_scale = min(timeout_scale * 2.0, 50.0)
                    self.cam._timeout_scale = timeout_scale
                    self.status.emit(f"Timeout; retrying with scale {timeout_scale:g}")
                    continue
                except Exception:
                    pass
            return None, f"error: acquire failed: {err}"

    def _is_zero_step(self, v_front: Optional[float], v_back: Optional[float]) -> bool:
        eps = abs(float(self.zero_v_eps))
        if v_front is not None and abs(float(v_front)) <= eps:
            return True
        if v_back is not None and abs(float(v_back)) <= eps:
            return True
        return False

    def _iter_voltage_steps(self, reverse: bool = False):
        if not self.sweep_voltage:
            return [(0, None, None)]
        steps = []
        n = max(1, int(self.nsteps))
        indices = range(n - 1, -1, -1) if reverse else range(n)
        for k in indices:
            vf = None
            vb = None
            if self.sweep_mode in ("front", "dual"):
                vf = float(self.front_v0 + k * self.front_step)
            if self.sweep_mode in ("back", "dual"):
                vb = float(self.back_v0 + k * self.back_step)
            steps.append((k, vf, vb))
        return steps

    def _voltage_suffix(self, v_front: Optional[float], v_back: Optional[float]) -> str:
        if not self.sweep_voltage:
            return ""
        if self.sweep_mode == "front":
            return f"_Vf{self._fmt_v_label(v_front)}"
        elif self.sweep_mode == "back":
            return f"_Vb{self._fmt_v_label(v_back)}"
        else:
            return f"_Vf{self._fmt_v_label(v_front)}_Vb{self._fmt_v_label(v_back)}"

    def _voltage_key_from_step(self, v_front: Optional[float], v_back: Optional[float]) -> Optional[float]:
        if self.grid_sweep:
            # Grid: linecut-vs-voltage is against gate B, with gate A held fixed.
            return v_back
        if self.sweep_mode == "front":
            return v_front
        elif self.sweep_mode == "back":
            return v_back
        return v_front

    def _iter_grid_steps(self):
        """Grid sweep: fix gate A at each of its steps, sweep gate B through its
        full list, then advance A. B is snaked (reversed on odd A rows) to reduce
        back-gate travel; A always advances forward. Yields (counter, vf, vb)."""
        na = max(1, int(self.nsteps_a))
        nb = max(1, int(self.nsteps_b))
        counter = 0
        for ai in range(na):
            vf = float(self.front_v0 + ai * self.front_step)
            b_indices = range(nb - 1, -1, -1) if (ai % 2 == 1) else range(nb)
            for bi in b_indices:
                vb = float(self.back_v0 + bi * self.back_step)
                yield counter, vf, vb
                counter += 1

    def _move_stage_for_entry(self, entry, idx: int, total: int) -> bool:
        """Move stage to the position for this entry. Returns False and emits done on error."""
        if not (self.sweep_power and entry is not None):
            return True
        if isinstance(entry, DualWheelPowerEntry) or hasattr(entry, "a_deg"):
            if self.stage_a is None or self.stage_b is None:
                self._set_done("error: stage not connected")
                return False
            self.status.emit(
                f"Move to A={entry.a_deg:.3f}, B={entry.b_deg:.3f} deg ({idx}/{total})"
            )
            try:
                with self._timed("stage"):
                    self.stage_a.move_to(entry.a_deg, step_deg=self.ramp_step_deg, controller=self._stage_controller)
                    self.stage_b.move_to(entry.b_deg, step_deg=self.ramp_step_deg, controller=self._stage_controller)
            except Exception as exc:
                self.status.emit(f"Stage move failed: {exc}")
                self._set_done(f"error: stage move failed: {exc}")
                return False
        else:
            if self.stage_a is None:
                self._set_done("error: stage not connected")
                return False
            self.status.emit(f"Move to {entry.position_deg:.3f} deg ({idx}/{total})")
            try:
                with self._timed("stage"):
                    self.stage_a.move_to(entry.position_deg, step_deg=self.ramp_step_deg,
                                         controller=self._stage_controller)
            except Exception as exc:
                self.status.emit(f"Stage move failed: {exc}")
                self._set_done(f"error: stage move failed: {exc}")
                return False
        return True

    def _emit_point(self, entry, result: dict, step_idx: int, voltage_key: Optional[float],
                    v_read_f: Optional[float], i_read_f: Optional[float],
                    v_read_b: Optional[float], i_read_b: Optional[float],
                    a_voltage: Optional[float] = None, b_voltage: Optional[float] = None,
                    pump2: Optional[bool] = None) -> None:
        payload = {
            "entry": entry,
            "image": result["image"],
            "accum_n": result["accum_n"],
            "filepath": result["filepath"],
            "step_idx": step_idx,
            "voltage_key": voltage_key,
            "grid": self.grid_sweep,
            "a_voltage": a_voltage,
            "b_voltage": b_voltage,
            "v_front": v_read_f,
            "i_front": i_read_f,
            "v_back": v_read_b,
            "i_back": i_read_b,
            "pump2": pump2,
        }
        self.point_done.emit(payload)

    # ------------------------------------------------------------ TRPL (SPAD23 / SPAD93 over pSPAD)
    def _save_trpl_csv(self, base_step: str, snapshot) -> Optional[str]:
        if not self.save_dir:
            return None
        path = self._next_filename(f"TRPL_{base_step}", ext=".csv")
        n_bins = int(snapshot.counts_matrix.shape[0])
        time_ns = (np.arange(n_bins, dtype=np.float64) * float(snapshot.bin_width_ps)) / 1000.0
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_ns"] + [f"pixel_{i}" for i in range(snapshot.counts_matrix.shape[1])])
                for i in range(n_bins):
                    writer.writerow([f"{time_ns[i]:.6f}"] + snapshot.counts_matrix[i, :].astype(np.int64).tolist())
        except Exception:
            return None
        return path

    def _note_trpl_problem(self, base_step: str, what: str, attempt: int, exc: Exception,
                           saved: Optional[str], outcome: str) -> None:
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "point": base_step, "what": what, "attempt": attempt,
               "error": f"{type(exc).__name__}: {exc}", "partial_file": saved or "", "outcome": outcome}
        self.trpl_problems.append(rec)
        if self.save_dir:
            try:
                with open(os.path.join(self.save_dir, PROBLEMS_NAME), "a", encoding="utf-8") as f:
                    f.write(" | ".join(f"{k}={v}" for k, v in rec.items()) + "\n")
            except Exception:
                pass

    def _save_failed_partial(self, base_step: str, what: str, attempt: int, snap) -> Optional[str]:
        """Keep what a failed stream had received (trpl_failed/, not a sweep point)."""
        if not self.save_dir or snap is None or getattr(snap, "counts_matrix", None) is None:
            return None
        c = np.asarray(snap.counts_matrix)
        if c.ndim != 2 or c.shape[0] == 0:
            return None
        folder = os.path.join(self.save_dir, FAILED_DIR)
        tag = what.replace(" ", "_")
        try:
            os.makedirs(folder, exist_ok=True)
            k = 0
            while True:
                path = os.path.join(folder, f"TRPL_{base_step}_{tag}_attempt{attempt}" + (f"_{k}" if k else "") + ".csv")
                if not os.path.exists(path):
                    break
                k += 1
            tdyn.write_trpl_csv(path, c, float(snap.bin_width_ps))
            return os.path.relpath(path, self.save_dir)
        except Exception:
            return None

    def _trpl_stream(self, measurement_ms: int, bin_ps: int, on_update, base_step: str, what: str):
        """One TRPL stream on its own connection, retried after a failure; returns the snapshot (a cancelled
        one on abort), None when stopped during a failure, or raises ``_TrplFailed``."""
        last_exc: Optional[Exception] = None
        attempts = 1 + TRPL_RETRIES
        for attempt in range(1, attempts + 1):
            if self._stop:
                return None
            try:
                with self.detector.tcspc_client(host=self.tcspc_host, port=self.tcspc_port,
                                                stream_idle_timeout_s=TRPL_DONE_GRACE_S) as client:
                    return client.acquire_trpl_stream(measurement_ms=measurement_ms, bin_width_ps=bin_ps,
                                                      on_update=on_update, update_interval_s=0.4,
                                                      should_stop=lambda: self._stop)
            except Exception as exc:
                if self._stop:
                    return None
                last_exc = exc
                saved = self._save_failed_partial(base_step, what, attempt, getattr(exc, "snapshot", None))
                retry = attempt < attempts
                self._note_trpl_problem(base_step, what, attempt, exc, saved, "retried" if retry else "gave up")
                self.status.emit(f"TRPL {what} failed (attempt {attempt}/{attempts}): {exc}"
                                 + (f"; retrying in {TRPL_RETRY_WAIT_S:g} s" if retry else ""))
                if retry:
                    t_end = time.monotonic() + TRPL_RETRY_WAIT_S
                    while time.monotonic() < t_end and not self._stop:
                        time.sleep(0.05)
        raise _TrplFailed(f"{what} failed {attempts} times; last error: {last_exc}")

    def _acquire_trpl(self, entry, base_step: str, step_idx: int, tags: dict, box: dict) -> None:
        """Quick count snapshot, then the TRPL stream; result (or error) in ``box``.

        Runs in the sweep thread (TRPL only) or a helper thread (PL + TRPL).  The
        pSPAD server is used by one connection at a time: count, then stream.
        """
        t0 = time.monotonic()
        try:
            self._acquire_trpl_body(entry, base_step, step_idx, tags, box)
        except Exception as exc:  # a bug, not the stream: still not worth losing the whole sweep
            if self._stop:
                box["aborted"] = True
            else:
                self._note_trpl_problem(base_step, "TRPL", 0, exc, None, "point kept without TRPL")
                box["failed"] = f"TRPL at {base_step}: unexpected {type(exc).__name__}: {exc}"
        finally:
            box["dt"] = time.monotonic() - t0

    def _acquire_trpl_body(self, entry, base_step: str, step_idx: int, tags: dict, box: dict) -> None:
        try:
            with self.detector.count_client(host=self.tcspc_host, port=self.tcspc_port) as count_client:
                c_snap = count_client.get_counts_snapshot(integration_ms=self.spad_count_integration_ms)
            self.spad_count_ready.emit(dict(tags, snapshot=c_snap, entry=entry, step_idx=step_idx))
        except Exception as exc:
            self.status.emit(f"SPAD quick count failed: {exc}")

        if self._stop:
            box["aborted"] = True
            return
        if self.trpl_dynamic is not None:
            self._acquire_trpl_dynamic(entry, base_step, step_idx, tags, box)
            return
        measurement_ms = max(1, int(round(self.tcspc_acq_s * 1000.0)))
        bin_ps = max(1, int(self.tcspc_bin_ps))
        self.status.emit(f"TRPL acquire @ point (t={self.tcspc_acq_s:.3g}s, bin={bin_ps} ps)...")
        try:
            snap = self._trpl_stream(
                measurement_ms, bin_ps,
                lambda s: self.trpl_update.emit(dict(tags, snapshot=s, entry=entry, step_idx=step_idx)),
                base_step, "stream")
        except _TrplFailed as exc:
            box["failed"] = f"TRPL at {base_step}: {exc}"
            return
        if snap is None or self._stop or getattr(snap, "cancelled", False):
            box["aborted"] = True
            return
        box["snapshot"] = snap
        box["filepath"] = self._save_trpl_csv(base_step, snap)

    def _acquire_trpl_dynamic(self, entry, base_step: str, step_idx: int, tags: dict, box: dict) -> None:
        """TRPL in chunks (one complete ``S,<ms>`` each, on one connection) until the brightest pixel's
        background-subtracted peak reaches the target, or the max time.  The chunks are summed bin-for-bin
        (zero-padded at the end: row 0 is t = 0 in every chunk) and each one is saved in trpl_chunks/.
        dI/dP pump 2 on repeats pump 2 off's chunk list (equal time), without the count check."""
        st = self.trpl_dynamic
        bin_ps = max(1, int(self.tcspc_bin_ps))
        plan = tdyn.ChunkPlan(st.first_s, st.step_s, st.max_s)
        pump2 = tags.get("pump2")
        fixed = list(self._dyn_schedule) if (pump2 is True and self._dyn_schedule) else None
        res = tdyn.SpotResult(target=float(st.target_counts), max_s=plan.max_ms / 1000.0)
        path = self._next_filename(f"TRPL_{base_step}", ext=".csv") if self.save_dir else None
        stem = os.path.splitext(os.path.basename(path))[0] if path else f"TRPL_{base_step}"
        chunk_dir = os.path.join(self.save_dir, tdyn.CHUNK_DIR) if self.save_dir else None
        acc: Optional[np.ndarray] = None
        n_events = 0
        oow = 0
        warns: List[str] = []
        last = None
        t_start = time.monotonic()

        def emit_live(s, done_ms: int, chunk_ms: int, prev, prev_events: int):
            # the running total: earlier chunks + this chunk so far (time = what the counts cover)
            part_ms = int(min(chunk_ms, max(0.0, float(s.elapsed_s)) * 1000.0))
            live = tdyn.combine_snapshot(s, tdyn.pad_add(prev, s.counts_matrix), max(1, done_ms + part_ms),
                                         prev_events + int(s.n_events), oow, (), time.monotonic() - t_start,
                                         done=False, cancelled=bool(getattr(s, "cancelled", False)))
            self.trpl_update.emit(dict(tags, snapshot=live, entry=entry, step_idx=step_idx))

        def finish_box():
            box["dyn"] = res.as_dict()
            box["trpl_acq_s"] = res.total_s
            box["trpl_chunks"] = len(res.chunk_ms)

        try:
            while True:
                if self._stop:
                    res.reason = "aborted"
                    break
                if fixed is not None:
                    if len(res.chunk_ms) >= len(fixed):
                        res.reason = "fixed_schedule"
                        break
                    ms = int(fixed[len(res.chunk_ms)])
                else:
                    ms = plan.next_ms(res.chunk_ms)
                    if ms is None:
                        res.reason = "max_time"
                        break
                k = len(res.chunk_ms) + 1
                done_ms = int(sum(res.chunk_ms))
                goal = (f"{sum(fixed) / 1000.0:g} s (= pump 2 off)" if fixed is not None
                        else f"max {plan.max_ms / 1000.0:g} s")
                self.status.emit(f"TRPL chunk {k} ({ms / 1000.0:g} s) · {done_ms / 1000.0:g} s done, {goal} · "
                                 f"peak {res.peak:.0f}/{st.target_counts:g}")
                snap = self._trpl_stream(
                    ms, bin_ps,
                    lambda s, _d=done_ms, _m=ms, _p=acc, _n=n_events: emit_live(s, _d, _m, _p, _n),
                    base_step, f"chunk {k}")
                if snap is None or self._stop or getattr(snap, "cancelled", False):
                    res.reason = "aborted"  # the partial chunk is dropped
                    break
                acc = tdyn.pad_add(acc, snap.counts_matrix)
                res.chunk_ms.append(ms)
                n_events += int(snap.n_events)
                oow += int(getattr(snap, "out_of_window", 0))
                warns.extend(snap.warnings)
                last = snap
                if chunk_dir:
                    cpath = os.path.join(chunk_dir, f"{stem}_c{k:02d}.csv")
                    try:
                        os.makedirs(chunk_dir, exist_ok=True)
                        tdyn.write_trpl_csv(cpath, snap.counts_matrix, bin_ps)
                        res.chunk_files.append(os.path.basename(cpath))
                    except Exception as exc:
                        self.status.emit(f"TRPL chunk file not saved: {exc}")
                pk = tdyn.peak_counts(acc, bin_ps, res.total_s, st, pre_pulse_window)
                res.peak, res.pixel, res.t_ns = pk["peak"], pk["pixel"], pk["t_ns"]
                res.background = pk["background"]
                res.reached = res.peak >= float(st.target_counts)
                if fixed is None and res.reached:
                    res.reason = "target"
                    break
        except _TrplFailed as exc:
            # keep the chunks already summed (the spot is flagged 'error'); none: the point has no TRPL
            res.reason = "error"
            res.reached = False
            if last is None:
                box["failed"] = f"TRPL at {base_step}: {exc}"
                finish_box()
                self._log_dynamic(None, res, entry, step_idx, tags)
                return
            self.status.emit(f"TRPL at {base_step}: {exc}; keeping the {len(res.chunk_ms)} chunk(s) taken")
        finish_box()
        if res.reason == "aborted" or last is None:
            box["aborted"] = True
            self._log_dynamic(None, res, entry, step_idx, tags)
            return
        total_ms = int(sum(res.chunk_ms))
        box["snapshot"] = tdyn.combine_snapshot(last, acc, total_ms, n_events, oow, warns,
                                                time.monotonic() - t_start)
        if pump2 is False:
            self._dyn_schedule = list(res.chunk_ms)
        if path:
            try:
                tdyn.write_trpl_csv(path, acc, bin_ps)
                box["filepath"] = path
            except Exception as exc:
                self.status.emit(f"TRPL file not saved: {exc}")
                box["filepath"] = None
        self._log_dynamic(box.get("filepath"), res, entry, step_idx, tags)
        flag = "" if res.reached else f"  ** below target ({res.reason}) **"
        self.status.emit(f"TRPL done: {len(res.chunk_ms)} chunk(s), {res.total_s:g} s, "
                         f"peak {res.peak:.0f}/{st.target_counts:g} (pixel {res.pixel}){flag}")

    def _log_dynamic(self, filepath, res, entry, step_idx: int, tags: dict) -> None:
        if self._dyn_log is None:
            return
        try:
            self._dyn_log.append(filepath, res, step_idx=step_idx, power_w=getattr(entry, "power_w", None),
                                 v_front=tags.get("a_voltage"), v_back=tags.get("b_voltage"),
                                 pump2=tags.get("pump2"))
        except Exception as exc:
            self.status.emit(f"{tdyn.LOG_NAME} not written: {exc}")

    def _acquire_at_step(self, entry, base: str, step_idx: int,
                         v_front: Optional[float], v_back: Optional[float]) -> bool:
        """Read VI, check trip, acquire image, emit point_done. Returns False on error/abort."""
        ok = self._acquire_at_step_body(entry, base, step_idx, v_front, v_back)
        if ok:
            self._emit_point_timing()
        return ok

    def _acquire_at_step_body(self, entry, base: str, step_idx: int,
                              v_front: Optional[float], v_back: Optional[float]) -> bool:
        with self._timed("read"):
            v_read_f, i_read_f = (
                self._read_vi(self.front_dev, target_v=v_front, icomp_a=self.front_icomp_a)
                if self.front_dev is not None else (None, None)
            )
            v_read_b, i_read_b = (
                self._read_vi(self.back_dev, target_v=v_back, icomp_a=self.back_icomp_a)
                if self.back_dev is not None else (None, None)
            )

        self.vi_ready.emit({
            "v_front": v_read_f,
            "i_front": i_read_f,
            "v_back": v_read_b,
            "i_back": i_read_b,
            "step_idx": step_idx,
            "entry": entry,
        })

        try:
            self._check_trip(i_read_f, self.front_icomp_a, "Front")
            self._check_trip(i_read_b, self.back_icomp_a, "Back")
        except ComplianceTrip:
            self._set_done("trip")
            return False

        vsuffix = self._voltage_suffix(v_front, v_back)
        base_step = f"{base}{vsuffix}" if vsuffix else base
        vi = (v_read_f, i_read_f, v_read_b, i_read_b)
        self._dyn_schedule = None
        if not self.didp:
            with self._timed("acq"):
                return self._acquire_one(entry, base_step, step_idx, v_front, v_back, vi, pump2=None)
        # dI/dP: pump 2 off, then pump 2 on, before the next point
        for pump2 in (False, True):
            if self._stop:
                self._set_done("aborted")
                return False
            with self._timed("flip"):
                flipped = self._set_pump2(pump2)
            if not flipped:
                return False
            with self._timed("acq"):
                acquired = self._acquire_one(entry, f"{base_step}_{self.pump2_tags[pump2]}", step_idx, v_front,
                                             v_back, vi, pump2=pump2)
            if not acquired:
                return False
        return True

    # ------------------------------------------------------------ dI/dP (pump-2 flipper)
    def _set_pump2(self, on: bool) -> bool:
        """Flip pump 2 off / on and wait for the settle time; False (done emitted) on abort or error."""
        target = self.pump2_on_position if on else self.pump2_off_position
        self.status.emit(f"Pump 2 {'on' if on else 'off'}: flipper to position {target}")
        try:
            res = self.flipper.move_to(target, controller=self._flipper_controller)
        except Exception as exc:
            if self._stop:
                self._set_done("aborted")
            else:
                msg = f"error: flipper move failed: {exc}"
                self.status.emit(msg)
                self._set_done(msg)
            return False
        if self._stop or getattr(res, "aborted", False):
            self._set_done("aborted")
            return False
        t_end = time.monotonic() + self.flipper_settle_ms / 1000.0
        while time.monotonic() < t_end:
            if self._stop:
                self._set_done("aborted")
                return False
            time.sleep(0.01)
        self._acq_meta = {
            "pump2": "on" if on else "off",
            "flipper_position": target,
            "flipper_serial": getattr(self.flipper, "serial", ""),
        }
        return True

    def _acquire_one(self, entry, base_step: str, step_idx: int, v_front: Optional[float],
                     v_back: Optional[float], vi, pump2: Optional[bool]) -> bool:
        """One PL and/or TRPL acquisition at the current settings (the PL GUI's per-point body)."""
        v_read_f, i_read_f, v_read_b, i_read_b = vi
        voltage_key = self._voltage_key_from_step(v_front, v_back)
        tags = {"voltage_key": voltage_key, "grid": self.grid_sweep, "a_voltage": v_front, "b_voltage": v_back,
                "pump2": pump2}

        trpl_box: dict = {}
        trpl_thread = None
        if self.measure_trpl:
            if self.measure_pl:
                trpl_thread = threading.Thread(
                    target=self._acquire_trpl, args=(entry, base_step, step_idx, tags, trpl_box), daemon=True
                )
                trpl_thread.start()
            else:
                self._acquire_trpl(entry, base_step, step_idx, tags, trpl_box)

        result = {"image": None, "accum_n": 0, "filepath": None}
        err = None
        if self.measure_pl:
            result, err = self._acquire_image(entry, v_read_f, v_read_b, i_read_f, i_read_b, base_step)
            if err:
                self._stop = self._stop or err == "aborted"
        if trpl_thread is not None:
            if err and not self._stop:
                self._stop = True  # PL failed: cancel the stream and end the sweep
            trpl_thread.join()
        if "dt" in trpl_box:
            self._tacc["trpl"] = self._tacc.get("trpl", 0.0) + float(trpl_box["dt"])
        if "trpl_acq_s" in trpl_box:  # dynamic: the requested TRPL time and chunks (scan estimate)
            self._tacc["trpl_acq_s"] = self._tacc.get("trpl_acq_s", 0.0) + float(trpl_box["trpl_acq_s"])
            self._tacc["trpl_chunks"] = self._tacc.get("trpl_chunks", 0) + int(trpl_box["trpl_chunks"])
        if err:
            if err == "aborted":
                self._set_done("aborted")
            else:
                self.status.emit(err)
                self._set_done(err)
            return False
        if trpl_box.get("aborted"):
            self._set_done("aborted")
            return False
        if trpl_box.get("error"):
            self.status.emit(trpl_box["error"])
            self._set_done(trpl_box["error"])
            return False
        if self.measure_trpl:
            if trpl_box.get("failed"):
                self._trpl_fail_points += 1
                if self._trpl_fail_points >= MAX_TRPL_FAIL_POINTS:
                    msg = (f"error: TRPL failed at {self._trpl_fail_points} points in a row; last: "
                           f"{trpl_box['failed']}")
                    self.status.emit(msg)
                    self._set_done(msg)
                    return False
                self.status.emit(f"{trpl_box['failed']} -- point kept without TRPL, the sweep goes on "
                                 f"({PROBLEMS_NAME})")
            else:
                self._trpl_fail_points = 0

        if self.measure_pl:
            self._emit_point(entry, result, step_idx, voltage_key, v_read_f, i_read_f, v_read_b, i_read_b,
                             a_voltage=v_front, b_voltage=v_back, pump2=pump2)
        if self.measure_trpl:
            self.trpl_done.emit(dict(tags, snapshot=trpl_box.get("snapshot"), entry=entry, step_idx=step_idx,
                                     filepath=trpl_box.get("filepath"), dyn=trpl_box.get("dyn"),
                                     failed=trpl_box.get("failed")))
        return True

    def _run_power_outer(self, power_plan: List[dict]) -> bool:
        """Outer loop = powers, inner loop = voltages (with snake). Current default behavior."""
        snake_voltage = bool(self.sweep_power and self.sweep_voltage and max(1, int(self.nsteps)) > 1)
        sweep_forward = True

        for idx, item in enumerate(power_plan, start=1):
            if self._stop or self._gate_controller.is_aborted():
                self._set_done("aborted")
                return False

            entry = item.get("entry")
            base = item.get("base") or "PL"

            if not self._move_stage_for_entry(entry, idx, len(power_plan)):
                return False

            if snake_voltage:
                direction = "forward" if sweep_forward else "reverse"
                self.status.emit(f"Voltage sweep ({direction})")
            reverse_steps = bool(snake_voltage and not sweep_forward)
            steps = self._iter_voltage_steps(reverse=reverse_steps)

            # Pre-ramp only the gate(s) that this mode actually sweeps to the
            # first step of this pass (V0 forward, V_end on a snake reverse pass,
            # where the previous pass ended). In single-gate mode the other gate
            # is left untouched.
            if self.sweep_voltage:
                _, vf_first, vb_first = steps[0]
                if self.sweep_mode in ("front", "dual") and self.front_dev is not None:
                    with self._timed("gate"):
                        self._gate_ramp(self.front_dev, float(vf_first), self.front_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_front": vi.v, "i_front": vi.i}))
                if self.sweep_mode in ("back", "dual") and self.back_dev is not None:
                    with self._timed("gate"):
                        self._gate_ramp(self.back_dev, float(vb_first), self.back_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_back": vi.v, "i_back": vi.i}))
                if self._stop or self._gate_controller.is_aborted():
                    self._set_done("aborted")
                    return False

            for step_idx, v_front, v_back in steps:
                if self._stop or self._gate_controller.is_aborted():
                    self._set_done("aborted")
                    return False

                if self.sweep_voltage:
                    with self._timed("gate"):
                        if v_front is not None and self.front_dev is not None:
                            self._gate_ramp(self.front_dev, v_front, self.front_icomp_a,
                                            on_microstep=lambda vi: self.vi_ready.emit({"v_front": vi.v, "i_front": vi.i}))
                        if v_back is not None and self.back_dev is not None:
                            self._gate_ramp(self.back_dev, v_back, self.back_icomp_a,
                                            on_microstep=lambda vi: self.vi_ready.emit({"v_back": vi.v, "i_back": vi.i}))
                    self._settle(self.settle_ms)

                if self._is_zero_step(v_front, v_back):
                    self._settle(self.zero_v_extra_settle_ms)

                if not self._acquire_at_step(entry, base, step_idx, v_front, v_back):
                    return False

            if self.sweep_power and entry is not None:
                self.power_done.emit({"entry": entry})
            if snake_voltage:
                sweep_forward = not sweep_forward

        return True

    def _run_voltage_outer(self, power_plan: List[dict]) -> bool:
        """Outer loop = voltages, inner loop = powers. No snake — voltage steps in fixed order."""
        for step_idx, v_front, v_back in self._iter_voltage_steps():
            if self._stop or self._gate_controller.is_aborted():
                self._set_done("aborted")
                return False

            if self.sweep_voltage:
                with self._timed("gate"):
                    if v_front is not None and self.front_dev is not None:
                        self._gate_ramp(self.front_dev, v_front, self.front_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_front": vi.v, "i_front": vi.i}))
                    if v_back is not None and self.back_dev is not None:
                        self._gate_ramp(self.back_dev, v_back, self.back_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_back": vi.v, "i_back": vi.i}))
                if self._stop or self._gate_controller.is_aborted():
                    self._set_done("aborted")
                    return False
                self._settle(self.settle_ms)
                if self._is_zero_step(v_front, v_back):
                    self._settle(self.zero_v_extra_settle_ms)

            n_voltages = max(1, int(self.nsteps)) if self.sweep_voltage else 1
            self.status.emit(f"Voltage step {step_idx + 1}/{n_voltages}" if self.sweep_voltage else "Acquiring")

            for pidx, item in enumerate(power_plan, start=1):
                if self._stop or self._gate_controller.is_aborted():
                    self._set_done("aborted")
                    return False

                entry = item.get("entry")
                base = item.get("base") or "PL"

                if not self._move_stage_for_entry(entry, pidx, len(power_plan)):
                    return False

                if not self._acquire_at_step(entry, base, step_idx, v_front, v_back):
                    return False

        # Emit power_done once per power entry after all voltage steps complete
        if self.sweep_power:
            for item in power_plan:
                entry = item.get("entry")
                if entry is not None:
                    self.power_done.emit({"entry": entry})

        return True

    def _run_grid(self, power_plan: List[dict]) -> bool:
        """Grid sweep: for each power, hold gate A at each step while sweeping
        gate B through its full list, then advance A. Gate A only re-ramps when
        its target changes between steps."""
        for idx, item in enumerate(power_plan, start=1):
            if self._stop or self._gate_controller.is_aborted():
                self._set_done("aborted")
                return False

            entry = item.get("entry")
            base = item.get("base") or "PL"

            if not self._move_stage_for_entry(entry, idx, len(power_plan)):
                return False

            # Pre-ramp both gates to their starting values
            with self._timed("gate"):
                if self.front_dev is not None:
                    self._gate_ramp(self.front_dev, float(self.front_v0), self.front_icomp_a,
                                    on_microstep=lambda vi: self.vi_ready.emit({"v_front": vi.v, "i_front": vi.i}))
                if self.back_dev is not None:
                    self._gate_ramp(self.back_dev, float(self.back_v0), self.back_icomp_a,
                                    on_microstep=lambda vi: self.vi_ready.emit({"v_back": vi.v, "i_back": vi.i}))
            if self._stop or self._gate_controller.is_aborted():
                self._set_done("aborted")
                return False

            prev_vf = None
            for step_idx, v_front, v_back in self._iter_grid_steps():
                if self._stop or self._gate_controller.is_aborted():
                    self._set_done("aborted")
                    return False

                # Ramp gate A only when its value changes (once per A row)
                with self._timed("gate"):
                    if self.front_dev is not None and (prev_vf is None or v_front != prev_vf):
                        self._gate_ramp(self.front_dev, v_front, self.front_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_front": vi.v, "i_front": vi.i}))
                        prev_vf = v_front
                    if self.back_dev is not None:
                        self._gate_ramp(self.back_dev, v_back, self.back_icomp_a,
                                        on_microstep=lambda vi: self.vi_ready.emit({"v_back": vi.v, "i_back": vi.i}))
                self._settle(self.settle_ms)
                if self._is_zero_step(v_front, v_back):
                    self._settle(self.zero_v_extra_settle_ms)

                if not self._acquire_at_step(entry, base, step_idx, v_front, v_back):
                    return False

            if self.sweep_power and entry is not None:
                self.power_done.emit({"entry": entry})

        return True

    def run(self):
        prev_timeout_scale = None
        self.timing.emit({"event": "start", "t": time.monotonic()})
        try:
            if self.measure_pl and self.cam is None:
                self.status.emit("Sweep error: camera not connected")
                self._set_done("error: device not connected")
                return
            if self.didp and self.flipper is None:
                self.status.emit("Sweep error: dI/dP needs the pump-2 flipper")
                self._set_done("error: flipper not connected")
                return

            if self.measure_pl:  # TRPL only: the camera is not touched
                try:
                    if hasattr(self.cam, "stop_stream"):
                        self.cam.stop_stream()
                    self.cam.set_frame_api("Snap")
                    self.cam.set_acquisition_mode("Single")
                    self.cam.set_trigger_mode("Internal")
                    self.cam.set_shutter("auto")
                    self.cam.set_exposure_ms(self.exposure_ms)
                    self.cam.set_baseline_clamp(True)
                except Exception:
                    pass

                if self.output_amp and self.readout_rate and self.preamp_gain:
                    try:
                        self.cam.set_amp_mode_by_labels(
                            self.output_amp, self.readout_rate, self.preamp_gain, force_preamp=True
                        )
                    except Exception:
                        pass

                try:
                    prev_timeout_scale = getattr(self.cam, "_timeout_scale", None)
                except Exception:
                    prev_timeout_scale = None
                try:
                    timeout_scale = 5.0
                    if self.exposure_ms > 10000.0:
                        timeout_scale = max(timeout_scale, (self.exposure_ms / 10000.0) * 2.0)
                    self.cam._timeout_scale = timeout_scale
                except Exception:
                    pass

            if self.measure_pl:
                self._resolve_wavelength_axis()

            power_plan = self.plan if self.sweep_power else [{"entry": None, "base": "PL_manual"}]

            if self.grid_sweep:
                ok = self._run_grid(power_plan)
            elif self.sweep_order == "voltage_outer":
                ok = self._run_voltage_outer(power_plan)
            else:
                ok = self._run_power_outer(power_plan)

            if ok:
                self._set_done("ok")
        except Exception as exc:
            self.status.emit(f"Unexpected sweep error: {exc}")
            self._set_done(f"error: {exc}")
        finally:
            if self.didp and self.flipper is not None:
                # leave pump 2 blocked
                try:
                    self.flipper.move_to(self.pump2_off_position)
                except Exception as exc:
                    self.status.emit(f"Flipper not returned to pump 2 off: {exc}")
            if self.measure_pl:
                try:
                    if prev_timeout_scale is not None:
                        self.cam._timeout_scale = prev_timeout_scale
                    else:
                        self.cam._timeout_scale = 1.0
                except Exception:
                    pass
            self.timing.emit(dict(self._tacc, event="end", t=time.monotonic(), points=self._point_count))
            self.done.emit(self._done_status or "error: sweep ended without a status")
