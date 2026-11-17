"""Device controllers of the Montana pump-probe GUI (all device I/O off the GUI thread).

Reuses the PL+TRPL v2 device layer (``DeviceBase``: connect / poll / disconnect in
worker threads, ``StageDevice`` for the Kinesis rotation stages, ``GateDevice`` for the
Keithley SMUs) and adds:

* ``DelayStageDevice`` -- Newport SMC100 (``smc100.smc100_wrapper.SMC100Stage``);
* ``CameraDevice`` -- PixeLink ``SimpleCamera`` (old GUI code, unchanged), or
  ``SimulatedCamera`` for tests without hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import numpy as np

from measurements.pl_trpl_power_voltage_v2.devices import (  # noqa: F401  (re-exported)
    CONNECTED,
    CONNECTING,
    DISCONNECTED,
    DISCONNECTING,
    CallThread,
    DeviceBase,
    GateDevice,
    StageDevice,
    wait_threads,
)
from measurements.pump_probe_gate_dep_montana.config import DEFAULT_STAGE_SCALE, VISA_DLL
from measurements.pump_probe_gate_dep_montana.pixelink_compat import load_camera_module
from smc100.smc100_wrapper import SMC100Stage


# ----------------------------------------------------------------------------- delay stage
class DelayStageDevice(DeviceBase):
    """Newport SMC100 delay stage.  Poll: position, state, homed, moving."""

    def __init__(self, poll_interval_s: float, parent=None):
        super().__init__("Delay stage", None, parent)
        self.stage: Optional[SMC100Stage] = None
        self.info = None
        self.poll_interval_s = float(poll_interval_s)

    def _open(self, params: Dict):
        stage = SMC100Stage(str(params["port"]), int(params.get("address", 1)),
                            simulate=bool(params.get("simulate")))
        info = stage.open()
        self.stage = stage
        self.info = info
        return info

    def _close(self) -> None:
        stage, self.stage = self.stage, None
        if stage is not None:
            stage.close()

    def _poll(self) -> Optional[Dict]:
        st = self.stage
        if st is None:
            return None
        out: Dict = {}
        try:
            state = st.get_state()
            out.update(state=state.state, state_text=state.text(), homed=state.is_referenced,
                       ready=state.is_ready, moving=state.is_moving)
        except Exception as exc:
            out["error"] = str(exc)
        try:
            out["pos"] = float(st.get_position())
        except Exception:
            out["pos"] = None
        return out


# ----------------------------------------------------------------------------- camera
class SimulatedCamera:
    """Stand-in for ``SimpleCamera``: 128 x 480 MONO16 frames, pump ON on every other frame.

    The reference ROI area (default x 625 is outside 480 columns, so the whole right edge
    is bright on ON frames) and a Gaussian spot whose depth follows the stage delay make
    the pump ON / OFF split and the log ratio display something sensible.
    """

    def __init__(self, height: int = 128, width: int = 700):
        self.height = int(height)
        self.width = int(width)
        self.is_initialized = False
        self.delay_ps_fn = None  # callable -> current delay (ps), set by the widget
        self._k = 0
        rr, cc = np.mgrid[0:self.height, 0:self.width]
        self._spot = np.exp(-(((rr - self.height / 2) / 12.0) ** 2 + ((cc - 240) / 60.0) ** 2))
        self._rng = np.random.default_rng(1)

    def initialize(self):
        self.is_initialized = True
        return True

    def capture_frames(self, n_frames, verbose=False, timeout=1.0):
        time.sleep(min(2.0, 0.002 * n_frames))
        delay = 0.0
        if self.delay_ps_fn is not None:
            try:
                delay = float(self.delay_ps_fn())
            except Exception:
                delay = 0.0
        amp = 0.02 * np.exp(-max(delay, 0.0) / 50.0) if delay >= 0 else 0.0
        frames = []
        for _ in range(int(n_frames)):
            on = (self._k % 2) == 0
            self._k += 1
            base = 5000.0 * (1.0 - amp * self._spot) if on else np.full(self._spot.shape, 5000.0)
            img = base + self._rng.normal(0, 20.0, base.shape)
            if on:
                img[:, 600:] += 800.0
            frames.append(np.clip(img, 0, 65535).astype(np.uint16))
        return frames

    def close(self):
        self.is_initialized = False


class CameraDevice(DeviceBase):
    """PixeLink camera (``SimpleCamera``: no settings changed, PixeLink Capture's are used).  No poll."""

    poll_interval_s = 0.0

    def __init__(self, parent=None):
        super().__init__("Camera", None, parent)
        self.camera = None

    def _open(self, params: Dict):
        if params.get("simulate"):
            cam = SimulatedCamera()
        else:
            module, err = load_camera_module()
            if module is None:
                raise RuntimeError(err)
            cam = module.SimpleCamera()
        cam.initialize()
        self.camera = cam
        return {"width": int(cam.width), "height": int(cam.height),
                "dtype": np.dtype(getattr(cam, "dtype", np.uint16)).name,
                "simulated": bool(params.get("simulate"))}

    def _close(self) -> None:
        cam, self.camera = self.camera, None
        if cam is not None:
            cam.close()


# ----------------------------------------------------------------------------- factories
def make_rotation_stage(serial: str):
    """ND wheel / polarization stage: Kinesis via ``rot.rot_wrapper`` (same as the ND calibration GUI)."""
    from rot.rot_wrapper import RotationStage
    return RotationStage(serial, scale=DEFAULT_STAGE_SCALE)


class GateOpener:
    """``open_fn`` of ``GateDevice``: KeithleySMU open, compliance, output on (pl_trpl v2 ``_open_gate_io``)."""

    def __init__(self):
        self._rm = None
        self._lock = threading.Lock()

    def _ensure_rm(self):
        with self._lock:
            if self._rm is None:
                from keithley.keithley_wrapper import make_resource_manager
                self._rm = make_resource_manager(VISA_DLL)
            return self._rm

    def __call__(self, resource: str, icomp_a: Optional[float]) -> Dict:
        from keithley.keithley_wrapper import KeithleySMU
        resource = (resource or "").strip()
        if not resource:
            raise RuntimeError("no VISA resource")
        rm = self._ensure_rm()
        dev = KeithleySMU(rm, resource, timeout_ms=20000, query_delay_s=0.0, verbose=False)
        idn = dev.open()
        if icomp_a is not None:
            try:
                dev.set_compliance(float(icomp_a))
            except Exception:
                pass
        v_now = None
        try:
            v_now = float(dev.query(":SOUR:VOLT?").strip())
        except Exception:
            v_now = None
        output_on = False
        try:
            dev.set_output(True)
            output_on = True
        except Exception:
            output_on = False
        return {"dev": dev, "idn": idn, "v_now": v_now, "output_on": output_on}
