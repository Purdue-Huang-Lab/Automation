"""Per-device controllers for the PL + TRPL v2 GUI.

Every device call (connect, readback polling, settings) runs in a worker
thread; the widget only reacts to signals.  This is what keeps the GUI
responsive: v1 polled the camera, spectrograph, stages, gates and the SPAD
from GUI timers, blocking the event loop for most of every second.

Each controller has
    connect_device(**params) / disconnect_device()   (CallThread)
    a PollThread that emits ``readback(dict)`` while connected
    pause() / resume() and ``quiet`` (paused and no call in flight), so a job
    (live view, sweep, gate ramp) can start only once the poll has let go of
    the device -- AndorSystem has no lock of its own
    call(fn, on_done)   one blocking call off the GUI thread (amp settings, homing, ...)

The SPAD (SPAD23 or SPAD93) uses that detector's gui_v2.device.DeviceController
(see detectors.py), which hands the pSPAD server to one client at a time.
"""

import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from PyQt5 import QtCore

from keithley.keithley_wrapper import SweepController

DISCONNECTED = "disconnected"
CONNECTING = "connecting"
CONNECTED = "connected"
DISCONNECTING = "disconnecting"


SHUTDOWN_WAIT_S = 10.0  # window close: longest wait for a device's in-flight connect / close / call

# Threads still running when the window closed (see wait_threads): never destroyed while the app lives
_PARKED: List[QtCore.QThread] = []


def _wavelength_axis_problem(wl) -> str:
    """Why a calibration is not usable ("" if it is): the PL GUI's rule, finite and not all zero."""
    if wl is None:
        return "no calibration"
    try:
        arr = np.asarray(wl, dtype=float).ravel()
    except Exception as exc:
        return f"not numeric ({exc})"
    if arr.size < 2:
        return f"{arr.size} values"
    if not np.all(np.isfinite(arr)):
        return "contains NaN / inf"
    if np.allclose(arr, 0.0):
        return "all zeros"
    return ""


def _valid_wavelength_axis(wl) -> bool:
    return not _wavelength_axis_problem(wl)


def wait_threads(threads: Iterable[Optional[QtCore.QThread]], timeout_s: float) -> None:
    """Wait for ``threads`` (one shared deadline); park any that are still running.

    A QThread destroyed while it runs makes Qt abort the program ("QThread: Destroyed
    while thread is still running"), e.g. when the window closes during a slow
    Kinesis or Andor open.  A thread that does not finish in time is detached from
    its parent and kept in ``_PARKED`` instead.
    """
    deadline = time.monotonic() + float(timeout_s)
    for th in list(threads):
        if th is None:
            continue
        left = deadline - time.monotonic()
        if th.isRunning() and left > 0:
            th.wait(max(1, int(left * 1000)))
        if th.isRunning():
            th.setParent(None)
            _PARKED.append(th)


class CallThread(QtCore.QThread):
    """Run one blocking call and report ``(result, error text)``."""

    done = QtCore.pyqtSignal(object, str)

    def __init__(self, fn: Callable[[], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            res = self._fn()
        except Exception as exc:
            self.done.emit(None, str(exc) or type(exc).__name__)
            return
        self.done.emit(res, "")


class PollThread(QtCore.QThread):
    """Call ``fn`` every ``interval_s`` (rate-limited) and emit its dict, until ``stop()``."""

    readback = QtCore.pyqtSignal(object)

    def __init__(self, fn: Callable[[], Dict], interval_s: float, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._min_interval_s = float(interval_s)
        self._stop = False
        self._paused = False
        self._in_call = False

    def stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def quiet(self) -> bool:
        return (self._paused or self._stop) and not self._in_call

    def run(self):
        while not self._stop:
            loop_start = time.monotonic()
            if not self._paused:
                self._in_call = True
                try:
                    if not self._paused and not self._stop:
                        res = self._fn()
                        if res is not None:
                            self.readback.emit(res)
                except Exception as exc:
                    self.readback.emit({"error": str(exc) or type(exc).__name__})
                finally:
                    self._in_call = False
            while not self._stop:
                remaining = self._min_interval_s - (time.monotonic() - loop_start)
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))


class LatestJobThread(QtCore.QThread):
    """Run submitted callables one at a time off the GUI thread; only the newest pending one runs.

    Used for the 2D Gaussian fit of the Histogram tab: during a TRPL acquisition a
    new snapshot arrives every 0.4 s, and a fit on a low-count frame can take
    hundreds of ms -- v1 ran it in the GUI thread.  ``done(job_id, result, error)``.
    """

    done = QtCore.pyqtSignal(int, object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending = None  # (job_id, fn)
        self._next_id = 0
        self._stop = False

    def submit(self, fn: Callable[[], object]) -> int:
        with self._cond:
            self._next_id += 1
            self._pending = (self._next_id, fn)
            self._cond.notify()
            return self._next_id

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job_id, fn = self._pending
                self._pending = None
            try:
                res, err = fn(), ""
            except Exception as exc:
                res, err = None, str(exc) or type(exc).__name__
            self.done.emit(job_id, res, err)


class DeviceBase(QtCore.QObject):
    """Connect / poll / disconnect one device off the GUI thread.

    Subclasses implement ``_open(params) -> info``, ``_close()`` and ``_poll() -> dict``
    (all run in worker threads).
    """

    state_changed = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(object)  # info returned by _open
    readback = QtCore.pyqtSignal(object)  # dict from _poll
    message = QtCore.pyqtSignal(str)

    poll_interval_s = 1.0

    def __init__(self, name: str, io_lock=None, parent=None):
        super().__init__(parent)
        self.name = name
        # every hardware call of this device (open, poll, call, close) holds this lock;
        # devices sharing one driver object (Andor camera + spectrograph) share it
        self._io_lock = io_lock if io_lock is not None else threading.RLock()
        self._state = DISCONNECTED
        self._poller: Optional[PollThread] = None
        self._threads = set()
        self._paused = False
        self._job: Optional[CallThread] = None

    # ------------------------------------------------------------ state
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == CONNECTED

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    @property
    def quiet(self) -> bool:
        """No poll call in flight (paused or not polling) and no ``call`` running."""
        busy_call = self._job is not None and self._job.isRunning()
        return (self._poller is None or self._poller.quiet) and not busy_call

    @property
    def call_running(self) -> bool:
        return self._job is not None and self._job.isRunning()

    def pause(self) -> None:
        self._paused = True
        if self._poller is not None:
            self._poller.pause()

    def resume(self) -> None:
        self._paused = False
        if self._poller is not None:
            self._poller.resume()

    def _locked(self, fn: Callable, *args):
        with self._io_lock:
            return fn(*args)

    def _keep(self, th: QtCore.QThread) -> QtCore.QThread:
        self._threads.add(th)
        th.finished.connect(lambda th=th: self._threads.discard(th))
        return th

    # ------------------------------------------------------------ connect / disconnect
    def connect_device(self, **params) -> bool:
        if self._state != DISCONNECTED:
            return False
        self._set_state(CONNECTING)
        self.message.emit(f"{self.name}: connecting...")
        th = self._keep(CallThread(lambda: self._locked(self._open, params), self))
        th.done.connect(self._on_open_done)
        th.start()
        return True

    def _on_open_done(self, info, err: str) -> None:
        if err:
            self._set_state(DISCONNECTED)
            self.message.emit(f"{self.name}: connect failed: {err}")
            return
        self._set_state(CONNECTED)
        self._start_poll()
        self.connected.emit(info)
        self.message.emit(f"{self.name}: connected")

    def _start_poll(self) -> None:
        if self.poll_interval_s <= 0:
            return
        poll = self._keep(PollThread(self._poll_once, self.poll_interval_s, self))
        poll.readback.connect(self.readback)
        if self._paused:
            poll.pause()
        self._poller = poll
        poll.start()

    def disconnect_device(self) -> bool:
        if self._state != CONNECTED:
            return False
        self._set_state(DISCONNECTING)
        if self._poller is not None:
            self._poller.stop()
        QtCore.QTimer.singleShot(50, self._check_stopped)
        return True

    def _check_stopped(self) -> None:
        if (self._poller is not None and self._poller.isRunning()) or self.call_running:
            QtCore.QTimer.singleShot(100, self._check_stopped)
            return
        self._poller = None
        th = self._keep(CallThread(lambda: self._locked(self._close), self))
        th.done.connect(self._on_close_done)
        th.start()

    def _on_close_done(self, _res, err: str) -> None:
        self._set_state(DISCONNECTED)
        self.message.emit(f"{self.name}: disconnected" + (f" ({err})" if err else ""))

    # ------------------------------------------------------------ one-off calls
    def call(self, fn: Callable[[], object], on_done: Optional[Callable[[object, str], None]] = None) -> bool:
        """Run ``fn`` in a worker thread (one at a time per device)."""
        if self.call_running:
            self.message.emit(f"{self.name}: busy")
            return False
        th = self._keep(CallThread(lambda: self._locked(fn), self))
        if on_done is not None:
            th.done.connect(on_done)
        self._job = th
        th.start()
        return True

    def shutdown(self, timeout_s: float = SHUTDOWN_WAIT_S) -> None:
        """Window close: stop the poll, let in-flight calls finish, close the device.

        Waits for every thread of this device -- the poll, a ``call`` and also a
        connect or disconnect still in progress (a Kinesis or Andor open can take
        seconds) -- because the threads are destroyed with the window.
        """
        if self._poller is not None:
            self._poller.stop()
        wait_threads(list(self._threads), timeout_s)
        self._poller = None
        if self._state != DISCONNECTED:
            # CONNECTING too: the open may have finished while we waited (its done slot never runs now).
            # A parked thread may still hold the I/O lock: do not block the close on it.
            if self._io_lock.acquire(timeout=2.0):
                try:
                    self._close()
                except Exception:
                    pass
                finally:
                    self._io_lock.release()
        self._state = DISCONNECTED

    # ------------------------------------------------------------ hooks
    def _poll_once(self) -> Optional[Dict]:
        return self._locked(self._poll)

    def _open(self, params: Dict):
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError

    def _poll(self) -> Optional[Dict]:
        return None


# ----------------------------------------------------------------------
class CameraDevice(DeviceBase):
    """Andor camera (AndorSystem.cam).  Poll: CCD temperature."""

    def __init__(self, cam, setup_fn: Callable[[], Dict], poll_interval_s: float, io_lock=None, parent=None):
        super().__init__("Andor", io_lock, parent)
        self.cam = cam
        self._setup_fn = setup_fn  # camera defaults + amp info / choices (worker thread)
        self.poll_interval_s = float(poll_interval_s)

    def _open(self, params: Dict):
        self.cam.connect()
        return self._setup_fn()

    def _close(self) -> None:
        try:
            if self.cam.spec is not None:
                self.cam.disconnect_spectrograph()
        except Exception:
            pass
        self.cam.disconnect()

    def _poll(self) -> Dict:
        return {"temp": self.cam.get_temperature_c()}


class SpectrographDevice(DeviceBase):
    """Andor Shamrock spectrograph (AndorSystem.spec; needs the camera for the pixel calibration)."""

    def __init__(self, cam, spec_index: int, slit_id, poll_interval_s: float, io_lock=None, parent=None):
        super().__init__("Spectrograph", io_lock, parent)
        self.cam = cam
        self.spec_index = int(spec_index)
        self.slit_id = slit_id
        self.poll_interval_s = float(poll_interval_s)
        self.last_wl_error = ""  # why the last wavelength-axis read failed

    def _open(self, params: Dict):
        if not self.cam.connect_spectrograph(self.spec_index):
            raise RuntimeError("spectrograph not found")
        return self.apply_settings(params)

    def apply_settings(self, params: Dict) -> Dict:
        """Grating / center / slit (worker thread), then the wavelength axis (``None`` at 0 nm center)."""
        if params.get("grating") is not None:
            try:
                self.cam.spec_set_grating(int(params["grating"]))
            except Exception:
                pass
        if params.get("center_nm") is not None:
            try:
                self.cam.set_center_wavelength_nm(float(params["center_nm"]))
            except Exception:
                pass
        if params.get("slit_um") is not None:
            try:
                self.cam.spec_set_slit_width_um(self.slit_id, float(params["slit_um"]))
            except Exception:
                pass
        wl = None
        center = params.get("center_nm")
        if center is None or abs(float(center)) > 1e-9:
            wl = self.read_wavelength_axis()
        return {"wavelength_axis": wl}

    def read_wavelength_axis(self):
        """Wavelength (nm) of every detector column, or ``None`` (worker thread).

        Asks the open spectrograph for its calibration at the current grating / center
        (``AndorSystem.get_wavelength_axis(force=True)`` re-opens it, which is only the fallback).
        """
        reasons = []
        wl = None
        spec = getattr(self.cam, "spec", None)
        if spec is not None and hasattr(spec, "get_calibration"):
            try:
                wl = np.asarray(spec.get_calibration(), dtype=float).ravel() * 1e9
            except Exception as exc:
                wl = None
                reasons.append(f"get_calibration: {type(exc).__name__}: {str(exc)[:80]}")
            else:
                if _wavelength_axis_problem(wl):
                    reasons.append(f"get_calibration: {_wavelength_axis_problem(wl)}")
        else:
            reasons.append("spectrograph not open")
        if not _valid_wavelength_axis(wl):
            try:
                wl = self.cam.get_wavelength_axis(force=True)
            except Exception as exc:
                wl = None
                reasons.append(f"re-open: {type(exc).__name__}: {str(exc)[:80]}")
            else:
                if _wavelength_axis_problem(wl):
                    reasons.append(f"re-open: {_wavelength_axis_problem(wl)}")
        if not _valid_wavelength_axis(wl):
            self.last_wl_error = "; ".join(reasons) or "unknown"
            return None
        self.last_wl_error = ""
        wl = np.asarray(wl, dtype=float).ravel()
        self.cam._wavelength_axis_nm = wl  # AndorSystem's cache (save_ascii's default axis)
        return wl

    def _close(self) -> None:
        self.cam.disconnect_spectrograph()

    def _poll(self) -> Dict:
        return {
            "grating": self.cam.spec_get_grating(),
            "center_nm": self.cam.get_center_wavelength_nm(),
            "slit_um": self.cam.spec_get_slit_width_um(self.slit_id),
        }


class StageDevice(DeviceBase):
    """Kinesis rotation stage.  Optional homing right after connect (as v1's Initialize)."""

    def __init__(self, name: str, stage_factory: Callable[[str], object], poll_interval_s: float, parent=None):
        super().__init__(name, None, parent)
        self._factory = stage_factory
        self.stage = None
        self.poll_interval_s = float(poll_interval_s)

    def _open(self, params: Dict):
        stage = self._factory(str(params["serial"]))
        stage.open()
        self.stage = stage
        velocity = stage.get_velocity() if hasattr(stage, "get_velocity") else None  # for the scan time estimate
        return {"serial": stage.serial, "home": bool(params.get("home")), "velocity": velocity}

    def _close(self) -> None:
        stage, self.stage = self.stage, None
        if stage is not None:
            stage.close()

    def _poll(self) -> Optional[Dict]:
        st = self.stage
        if st is None:
            return None
        out = {"serial": st.serial}
        try:
            out["pos"] = float(st.get_position())
        except Exception:
            out["pos"] = None
        try:
            out["homed"] = st.is_homed()
        except Exception:
            out["homed"] = None
        try:
            out["moving"] = bool(st.is_moving())
        except Exception:
            out["moving"] = False
        return out


class GateDevice(DeviceBase):
    """Keithley SMU gate.  Poll: V/I; after ``error_limit`` failed reads the device is dropped and
    reconnected every ``reconnect_s`` in the poll thread (v1's auto-reconnect, now off the GUI thread).

    Disconnect ramps the output to 0 V (``ramp_step_v`` / ``ramp_dwell_s``) before switching it off;
    if that ramp cannot finish (link error, ``abort_rampdown``) the output is left on where it is.
    ``leave_output_on = True`` closes without touching the output (window closed with the gates on)."""

    RAMPDOWN_CHUNK_V = 0.5  # progress message and verify read every this many volts

    def __init__(self, name: str, open_fn: Callable[[str, float], Dict], read_timeout_ms: int, error_limit: int,
                 poll_interval_s: float, reconnect_s: float, ramp_step_v: float = 0.01, ramp_dwell_s: float = 0.1,
                 parent=None):
        super().__init__(name, None, parent)
        self.ramp_step_v = float(ramp_step_v)
        self.ramp_dwell_s = float(ramp_dwell_s)
        self.leave_output_on = False
        self.last_close_error = ""
        self._rampdown_ctrl: Optional[SweepController] = None
        self._open_fn = open_fn  # (resource, icomp_a) -> {"dev", "idn", "v_now", "output_on"}
        self.dev = None
        self._resource = ""
        self._icomp_a = None
        self._read_timeout_ms = int(read_timeout_ms)
        self._error_limit = max(1, int(error_limit))
        self._reconnect_s = float(reconnect_s)
        self._failures = 0
        self._next_reconnect = 0.0
        self._lock = threading.Lock()
        self.poll_interval_s = float(poll_interval_s)

    def set_compliance_a(self, icomp_a: Optional[float]) -> None:
        self._icomp_a = icomp_a

    def _open(self, params: Dict):
        self._resource = str(params["resource"])
        self._icomp_a = params.get("icomp_a")
        info = self._open_fn(self._resource, self._icomp_a)
        with self._lock:
            self.dev = info["dev"]
            self._failures = 0
        return info

    def _drop(self, output_off: bool = True) -> None:
        dev, self.dev = self.dev, None
        if dev is not None:
            if output_off:
                try:
                    dev.set_output(False)
                except Exception:
                    pass
            try:
                dev.close()
            except Exception:
                pass

    def _close(self) -> None:
        """Ramp to 0 V, output off, close (worker thread).  Raises with the reason if the output stays on."""
        self.last_close_error = ""
        with self._lock:
            dev, self.dev = self.dev, None
            if dev is None:
                return
            err = None if self.leave_output_on else self._ramp_down(dev)
            try:
                if err is None and not self.leave_output_on:
                    dev.set_output(False)
            except Exception as exc:
                err = f"output off failed: {exc}"
            finally:
                try:
                    dev.close()
                except Exception:
                    pass
        if err:
            self.last_close_error = err
            raise RuntimeError(err)

    def abort_rampdown(self) -> None:
        ctrl = self._rampdown_ctrl
        if ctrl is not None:
            ctrl.abort()

    def _ramp_down(self, dev) -> Optional[str]:
        """Ramp ``dev`` to 0 V in chunks (progress in ``message`` / ``readback``); None, or why it stopped."""
        ctrl = self._rampdown_ctrl = SweepController()
        try:
            v = float(dev.read_source_setpoint()) if hasattr(dev, "read_source_setpoint") else 0.0
            if abs(v) > 1e-3:
                n = abs(v) / max(1e-6, self.ramp_step_v)
                self.message.emit(f"{self.name}: ramping {v:.4g} -> 0 V before switching the output off "
                                  f"(~{n * (self.ramp_dwell_s + 0.02):.0f} s; Abort leaves it on)")
            while True:
                nxt = 0.0 if abs(v) <= self.RAMPDOWN_CHUNK_V else v - np.sign(v) * self.RAMPDOWN_CHUNK_V
                vi = dev.ramp_to_voltage(nxt, ramp_step_v=self.ramp_step_v, ramp_dwell_s=self.ramp_dwell_s,
                                         controller=ctrl, micro_read=False, sync_each_step=True,
                                         verify=not ctrl.is_aborted())
                if ctrl.is_aborted():
                    return (f"ramp to 0 V aborted at {float(getattr(dev, '_last_set_v', v)):.4g} V: "
                            f"output left ON")
                self.readback.emit({"v": float(vi.v), "i": float(vi.i)})
                v = nxt
                if v == 0.0:
                    return None
                self.message.emit(f"{self.name}: ramping to 0 V, now at {v:.4g} V")
        except Exception as exc:
            return (f"could not ramp to 0 V ({type(exc).__name__}: {str(exc)[:100]}): output left ON at "
                    f"{float(getattr(dev, '_last_set_v', float('nan'))):.4g} V")
        finally:
            self._rampdown_ctrl = None

    def _poll(self) -> Optional[Dict]:
        with self._lock:
            dev = self.dev
            if dev is None:
                now = time.monotonic()
                if now < self._next_reconnect:
                    return None
                self._next_reconnect = now + self._reconnect_s
                try:
                    info = self._open_fn(self._resource, self._icomp_a)
                except Exception as exc:
                    return {"lost": True, "reconnect_failed": str(exc)}
                self.dev = info["dev"]
                self._failures = 0
                return {"reconnected": True, "info": info}
            try:
                vi = dev.read_vi_with_timeout(self._read_timeout_ms)
                self._failures = 0
                return {"v": float(vi.v), "i": float(vi.i)}
            except Exception as exc:
                self._failures += 1
                if self._failures >= self._error_limit:
                    # keep the output on: switching it off would step the gate from V to 0 V at once,
                    # and the link may only be slow; the SMU keeps holding the voltage until reconnect
                    self._drop(output_off=False)
                    self._next_reconnect = time.monotonic() + self._reconnect_s
                    return {"lost": True, "error": str(exc)}
                return {"poll_error": str(exc), "failures": self._failures}


class FlipperDevice(DeviceBase):
    """Thorlabs MFF filter flipper (pump 2 on / off).  Poll: position (1, 2 or None in transit)."""

    def __init__(self, poll_interval_s: float, parent=None):
        super().__init__("Flipper", None, parent)
        self.flipper = None
        self.poll_interval_s = float(poll_interval_s)

    @staticmethod
    def _factory(serial: str):
        from flipper.flipper_wrapper import FilterFlipper

        return FilterFlipper(serial)

    def _open(self, params: Dict):
        ff = self._factory(str(params["serial"]))
        info = ff.open()
        self.flipper = ff
        return {"info": info.text(), "position": ff.last_position, "serial": ff.serial}

    def _close(self) -> None:
        ff, self.flipper = self.flipper, None
        if ff is not None:
            ff.close()

    def _poll(self) -> Optional[Dict]:
        ff = self.flipper
        if ff is None:
            return None
        return {"position": ff.get_position()}
