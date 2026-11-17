"""Newport SMC100CC / SMC100PP single-axis motion controller (RS-232 / USB-serial).

Used as the pump-probe delay stage of the Montana setup (replaces the Thorlabs
BBD301 driven through ``thorlabs_apt``).  No Qt; thread-safe through one
``RLock`` (every command / reply pair holds it).

Protocol (SMC100 user manual): ASCII ``<address><CMD><parameter>\\r\\n`` at
57600 baud, 8N1, XON/XOFF.  Queries are answered as ``<address><CMD><value>``.
Commands used here:

    ID?  stage identifier          VE   controller version
    TS   error bits + state        TP   current position
    SL?  / SR?  software limits    VA?  / VA  velocity
    OR   home search               PA   absolute move
    PR   relative move             ST   stop motion
    TE   last command error        RS   reset
    MM1  leave DISABLE (-> READY)

After power-on the controller is NOT REFERENCED: ``home()`` must run once
before any absolute move.

Command line::

    python -m smc100.smc100_wrapper --list
    python -m smc100.smc100_wrapper --port COM3 --info
    python -m smc100.smc100_wrapper --port COM3 --home --move 10
    python -m smc100.smc100_wrapper --simulate --home --move 5    # no hardware
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
    import serial  # pyserial
    from serial.tools import list_ports
except Exception:  # pyserial missing: only the simulator works
    serial = None
    list_ports = None


DEFAULT_PORT = "COM3"
DEFAULT_ADDRESS = 1
DEFAULT_BAUDRATE = 57600
DEFAULT_TIMEOUT_S = 1.0
MOVE_POLL_S = 0.05
HOME_TIMEOUT_S = 120.0
MOVE_TIMEOUT_EXTRA_S = 10.0  # move timeout = distance / velocity * 2 + this
RESET_WAIT_S = 3.0

# Controller states (2 hex digits of the TS reply)
STATE_NAMES = {
    "0A": "NOT REFERENCED from RESET",
    "0B": "NOT REFERENCED from HOMING",
    "0C": "NOT REFERENCED from CONFIGURATION",
    "0D": "NOT REFERENCED from DISABLE",
    "0E": "NOT REFERENCED from READY",
    "0F": "NOT REFERENCED from MOVING",
    "10": "NOT REFERENCED - ESP stage error",
    "11": "NOT REFERENCED from JOGGING",
    "14": "CONFIGURATION",
    "1E": "HOMING (RS-232)",
    "1F": "HOMING (keypad)",
    "28": "MOVING",
    "32": "READY from HOMING",
    "33": "READY from MOVING",
    "34": "READY from DISABLE",
    "35": "READY from JOGGING",
    "36": "READY T from READY",
    "37": "READY T from TRACKING",
    "38": "READY T from DISABLE T",
    "3C": "DISABLE from READY",
    "3D": "DISABLE from MOVING",
    "3E": "DISABLE from JOGGING",
    "3F": "DISABLE from TRACKING",
    "46": "JOGGING from READY",
    "47": "JOGGING from DISABLE",
}
NOT_REFERENCED = {"0A", "0B", "0C", "0D", "0E", "0F", "10", "11"}
READY = {"32", "33", "34", "35", "36", "37", "38"}
HOMING = {"1E", "1F"}
MOVING = {"28"}
DISABLE = {"3C", "3D", "3E", "3F"}
JOGGING = {"46", "47"}

# Positioner error bits (4 hex digits of the TS reply)
ERROR_BITS = (
    (0x0200, "80 W output power exceeded"),
    (0x0100, "DC voltage too low"),
    (0x0080, "wrong ESP stage"),
    (0x0040, "homing time out"),
    (0x0020, "following error"),
    (0x0010, "short circuit detection"),
    (0x0008, "RMS current limit"),
    (0x0004, "peak current limit"),
    (0x0002, "positive end of run"),
    (0x0001, "negative end of run"),
)

# Command error codes (TE reply)
COMMAND_ERRORS = {
    "@": "no error",
    "A": "unknown message code or floating point controller address",
    "B": "controller address not correct",
    "C": "parameter missing or out of range",
    "D": "command not allowed",
    "E": "home sequence already started",
    "F": "ESP stage name unknown",
    "G": "displacement out of limits",
    "H": "command not allowed in NOT REFERENCED state",
    "I": "command not allowed in CONFIGURATION state",
    "J": "command not allowed in DISABLE state",
    "K": "command not allowed in READY state",
    "L": "command not allowed in HOMING state",
    "M": "command not allowed in MOVING state",
    "N": "current position out of software limit",
    "S": "communication time out",
    "U": "error during EEPROM access",
    "V": "error during command execution",
    "W": "command not allowed for PP version",
    "X": "command not allowed for CC version",
}


class SMC100Error(RuntimeError):
    pass


class MoveController:
    """Abort flag for a home / move wait: the stage is stopped (``ST``) when it is set."""

    def __init__(self):
        self._abort = threading.Event()

    def abort(self) -> None:
        self._abort.set()

    def reset(self) -> None:
        self._abort.clear()

    def is_aborted(self) -> bool:
        return self._abort.is_set()


@dataclass
class SMC100State:
    error_bits: int
    state: str  # 2 hex digits

    @property
    def name(self) -> str:
        return STATE_NAMES.get(self.state, f"unknown state {self.state}")

    @property
    def errors(self) -> List[str]:
        return [txt for bit, txt in ERROR_BITS if self.error_bits & bit]

    @property
    def is_ready(self) -> bool:
        return self.state in READY

    @property
    def is_referenced(self) -> bool:
        return self.state not in NOT_REFERENCED and self.state != "14"

    @property
    def is_moving(self) -> bool:
        return self.state in MOVING or self.state in HOMING or self.state in JOGGING

    @property
    def is_disabled(self) -> bool:
        return self.state in DISABLE

    def text(self) -> str:
        s = self.name
        if self.error_bits:
            s += " | " + ", ".join(self.errors)
        return s


@dataclass
class SMC100Info:
    port: str
    address: int
    stage_id: str
    version: str
    limits_mm: Tuple[float, float]
    velocity_mm_s: float
    state: SMC100State
    position_mm: float

    def text(self) -> str:
        return (f"{self.version} on {self.port} (addr {self.address}), stage {self.stage_id or '?'}; "
                f"limits {self.limits_mm[0]:g} .. {self.limits_mm[1]:g} mm, v = {self.velocity_mm_s:g} mm/s; "
                f"{self.state.text()}; position {self.position_mm:.4f} mm")


def list_serial_ports() -> List[Tuple[str, str]]:
    """[(device, description)] of the serial ports present (empty without pyserial)."""
    if list_ports is None:
        return []
    out = []
    for p in sorted(list_ports.comports(), key=lambda p: p.device):
        out.append((p.device, p.description or ""))
    return out


# ----------------------------------------------------------------------------- simulator
class SimulatedSMC100:
    """Answers SMC100 commands like a controller with a 25 mm stage (for GUI tests without hardware)."""

    def __init__(self, address: int = DEFAULT_ADDRESS, limits=(0.0, 25.0), velocity: float = 5.0):
        self.address = int(address)
        self.limits = (float(limits[0]), float(limits[1]))
        self.velocity = float(velocity)
        self.state = "0A"
        self.pos = 0.0
        self._move = None  # (t0, start, target, final_state)
        self._last_error = "@"
        self._lock = threading.Lock()

    def _update(self) -> None:
        if self._move is None:
            return
        t0, start, target, final_state = self._move
        dist = abs(target - start)
        dt = time.monotonic() - t0
        if self.velocity <= 0 or dt * self.velocity >= dist:
            self.pos = target
            self.state = final_state
            self._move = None
        else:
            self.pos = start + math.copysign(dt * self.velocity, target - start)

    def _start_move(self, target: float, moving_state: str, final_state: str) -> None:
        self._move = (time.monotonic(), self.pos, float(target), final_state)
        self.state = moving_state

    def handle(self, line: str) -> Optional[str]:
        with self._lock:
            self._update()
            a = str(self.address)
            if not line.startswith(a):
                return None
            body = line[len(a):]
            cmd, arg = body[:2].upper(), body[2:]
            query = arg == "?"
            if cmd == "TS":
                return f"{a}TS0000{self.state}"
            if cmd == "TP":
                return f"{a}TP{self.pos:.6f}"
            if cmd == "TE":
                err, self._last_error = self._last_error, "@"
                return f"{a}TE{err}"
            if cmd == "ID":
                return f"{a}IDSIM-25MM"
            if cmd == "VE":
                return f"{a}VE SMC_CC - Simulated 1.0"
            if cmd == "SL" and query:
                return f"{a}SL{self.limits[0]:g}"
            if cmd == "SR" and query:
                return f"{a}SR{self.limits[1]:g}"
            if cmd == "VA":
                if query:
                    return f"{a}VA{self.velocity:g}"
                self.velocity = float(arg)
                return None
            if cmd == "OR":
                if self.state in NOT_REFERENCED:
                    self._start_move(0.0, "1E", "32")
                else:
                    self._last_error = "K"
                return None
            if cmd in ("PA", "PR"):
                if self.state not in READY:
                    self._last_error = "H" if self.state in NOT_REFERENCED else "D"
                    return None
                target = float(arg) if cmd == "PA" else self.pos + float(arg)
                if not (self.limits[0] - 1e-9 <= target <= self.limits[1] + 1e-9):
                    self._last_error = "G"
                    return None
                self._start_move(target, "28", "33")
                return None
            if cmd == "ST":
                if self._move is not None:
                    self._move = None
                    self.state = "33" if self.state in MOVING else "0B"
                return None
            if cmd == "RS":
                self._move = None
                self.state = "0A"
                return None
            if cmd == "MM":
                if arg == "1" and self.state in DISABLE:
                    self.state = "34"
                elif arg == "0" and self.state in READY:
                    self.state = "3C"
                return None
            self._last_error = "A"
            return None


# ----------------------------------------------------------------------------- controller
class SMC100Stage:
    """One SMC100 axis.  Positions in the stage unit (mm for linear stages).

    ``move_to(mm, wait=True)`` / ``get_position()`` / ``stop()`` / ``home()`` match the
    old ``DelayStage`` (Thorlabs BBD301) calls used by the pump-probe scan loops.
    """

    def __init__(self, port: str = DEFAULT_PORT, address: int = DEFAULT_ADDRESS, *,
                 baudrate: int = DEFAULT_BAUDRATE, timeout_s: float = DEFAULT_TIMEOUT_S,
                 simulate: bool = False, verbose: bool = False):
        self.port = str(port)
        self.address = int(address)
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.simulate = bool(simulate)
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self._ser = None
        self._sim: Optional[SimulatedSMC100] = None
        self.limits_mm: Tuple[float, float] = (-math.inf, math.inf)
        self.velocity_mm_s: float = float("nan")
        self.stage_id = ""
        self.version = ""
        self.info: Optional[SMC100Info] = None
        self._sim_reply: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SMC100 {self.port}] {msg}")

    @property
    def is_open(self) -> bool:
        return self._sim is not None or self._ser is not None

    @property
    def is_initialized(self) -> bool:  # name used by the old DelayStage callers
        return self.is_open

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise SMC100Error("SMC100 not connected")

    def _send(self, cmd: str) -> None:
        line = f"{self.address}{cmd}"
        self._log(f"> {line}")
        if self._sim is not None:
            reply = self._sim.handle(line)
            self._sim_reply = reply
            return
        self._ser.write((line + "\r\n").encode("ascii"))
        self._ser.flush()

    def _readline(self) -> str:
        if self._sim is not None:
            reply, self._sim_reply = self._sim_reply, None
            if reply is None:
                raise SMC100Error("no reply (timeout)")
            return reply
        raw = self._ser.read_until(b"\n")
        if not raw.endswith(b"\n"):
            raise SMC100Error(f"no reply from SMC100 on {self.port} (timeout {self.timeout_s:g} s)")
        return raw.decode("ascii", errors="replace").strip()

    def write(self, cmd: str) -> None:
        """Send a command that has no reply."""
        with self._lock:
            self._ensure_open()
            self._send(cmd)

    def query(self, cmd: str) -> str:
        """Send a query (``TS``, ``TP``, ``SL?`` ...) and return the value after the echoed command."""
        head = cmd.rstrip("?")[:2].upper()
        prefix = f"{self.address}{head}"
        with self._lock:
            self._ensure_open()
            if self._ser is not None:
                self._ser.reset_input_buffer()
            self._send(cmd)
            reply = self._readline()
            self._log(f"< {reply}")
        if not reply.upper().startswith(prefix.upper()):
            raise SMC100Error(f"unexpected reply to {cmd!r}: {reply!r}")
        return reply[len(prefix):].strip()

    def _query_float(self, cmd: str) -> float:
        val = self.query(cmd)
        try:
            return float(val)
        except ValueError as exc:
            raise SMC100Error(f"bad number in reply to {cmd!r}: {val!r}") from exc

    def last_error(self) -> Tuple[str, str]:
        """(code, text) of the last command error (``TE``; reading it clears it)."""
        code = self.query("TE")[:1] or "@"
        return code, COMMAND_ERRORS.get(code, f"error {code}")

    def _check_command(self, what: str) -> None:
        code, txt = self.last_error()
        if code != "@":
            raise SMC100Error(f"{what}: {txt} (TE {code})")

    # ------------------------------------------------------------------ open / close
    def open(self) -> SMC100Info:
        self.close()
        with self._lock:
            if self.simulate:
                self._sim = SimulatedSMC100(self.address)
            else:
                if serial is None:
                    raise SMC100Error("pyserial is not installed")
                try:
                    self._ser = serial.Serial(
                        port=self.port, baudrate=self.baudrate, bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        xonxoff=True, timeout=self.timeout_s, write_timeout=self.timeout_s)
                except Exception as exc:
                    self._ser = None
                    raise SMC100Error(f"cannot open {self.port}: {exc}") from exc
                time.sleep(0.05)
                self._ser.reset_input_buffer()
            try:
                self.last_error()  # clear a stale command error
                st = self.get_state()
                try:
                    self.stage_id = self.query("ID?")
                except SMC100Error:
                    self.stage_id = ""
                try:
                    self.version = self.query("VE")
                except SMC100Error:
                    self.version = "SMC100"
                self.limits_mm = (self._query_float("SL?"), self._query_float("SR?"))
                try:
                    self.velocity_mm_s = self._query_float("VA?")
                except SMC100Error:
                    self.velocity_mm_s = float("nan")
                if st.is_disabled:
                    self.write("MM1")  # DISABLE -> READY
                    time.sleep(0.1)
                    st = self.get_state()
                pos = self.get_position()
            except Exception as exc:
                self.close()
                raise SMC100Error(f"SMC100 on {self.port} (address {self.address}) does not answer: {exc}") from exc
            info = SMC100Info(self.port if not self.simulate else "SIM", self.address, self.stage_id,
                              self.version, self.limits_mm, self.velocity_mm_s, st, pos)
            self._log(info.text())
            return info

    def close(self) -> None:
        with self._lock:
            ser, self._ser = self._ser, None
            self._sim = None
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass

    def __enter__(self):
        self.info = self.open()
        return self

    def __exit__(self, *_exc):
        self.close()

    # ------------------------------------------------------------------ status
    def get_state(self) -> SMC100State:
        val = self.query("TS")
        if len(val) < 6:
            raise SMC100Error(f"bad TS reply: {val!r}")
        return SMC100State(int(val[:4], 16), val[4:6].upper())

    def get_position(self) -> float:
        return self._query_float("TP")

    def get_limits(self) -> Tuple[float, float]:
        return self.limits_mm

    def is_homed(self) -> bool:
        return self.get_state().is_referenced

    def is_moving(self) -> bool:
        return self.get_state().is_moving

    # ------------------------------------------------------------------ motion
    def set_velocity(self, mm_s: float) -> None:
        self.write(f"VA{float(mm_s):.6g}")
        self._check_command("set velocity")
        self.velocity_mm_s = self._query_float("VA?")

    def stop(self) -> None:
        if self.is_open:
            try:
                self.write("ST")
            except Exception:
                pass

    def reset(self) -> None:
        """Controller reset (``RS``): it comes back NOT REFERENCED; home again afterwards."""
        self.write("RS")
        time.sleep(RESET_WAIT_S)

    def _wait_ready(self, what: str, timeout_s: float, controller: Optional[MoveController],
                    on_poll: Optional[Callable[[float], None]] = None) -> SMC100State:
        t_end = time.monotonic() + float(timeout_s)
        while True:
            if controller is not None and controller.is_aborted():
                self.stop()
                raise SMC100Error(f"{what} aborted")
            st = self.get_state()
            if on_poll is not None:
                try:
                    on_poll(self.get_position())
                except Exception:
                    pass
            if st.is_ready:
                return st
            if st.is_disabled or (st.state in NOT_REFERENCED and what != "home"):
                raise SMC100Error(f"{what} failed: {st.text()}")
            if st.state in NOT_REFERENCED and what == "home" and st.error_bits:
                raise SMC100Error(f"home failed: {st.text()}")
            if time.monotonic() > t_end:
                self.stop()
                raise SMC100Error(f"{what} timed out after {timeout_s:g} s ({st.text()})")
            time.sleep(MOVE_POLL_S)

    def home(self, wait: bool = True, timeout_s: float = HOME_TIMEOUT_S,
             controller: Optional[MoveController] = None) -> None:
        """Home search (``OR``).  Only allowed in the NOT REFERENCED state; a referenced stage is left as is."""
        st = self.get_state()
        if st.is_referenced:
            self._log("already referenced: home skipped")
            return
        if st.state == "14":
            raise SMC100Error("controller is in CONFIGURATION state: leave it (PW0) before homing")
        self.write("OR")
        self._check_command("home")
        if wait:
            time.sleep(0.1)
            self._wait_ready("home", timeout_s, controller)

    def _move_timeout(self, distance_mm: float) -> float:
        v = self.velocity_mm_s
        if not (math.isfinite(v) and v > 0):
            return 60.0
        return abs(distance_mm) / v * 2.0 + MOVE_TIMEOUT_EXTRA_S

    def move_to(self, position_mm: float, wait: bool = True, timeout_s: Optional[float] = None,
                controller: Optional[MoveController] = None,
                on_poll: Optional[Callable[[float], None]] = None) -> float:
        """Absolute move (``PA``); blocking by default.  Returns the position read after the move."""
        target = float(position_mm)
        lo, hi = self.limits_mm
        if not (lo - 1e-9 <= target <= hi + 1e-9):
            raise SMC100Error(f"target {target:.4f} mm outside the stage limits {lo:g} .. {hi:g} mm")
        st = self.get_state()
        if not st.is_referenced:
            raise SMC100Error(f"stage not homed ({st.text()}): home it first")
        if not st.is_ready:
            raise SMC100Error(f"stage not ready to move ({st.text()})")
        start = self.get_position()
        self.write(f"PA{target:.6f}")
        self._check_command(f"move to {target:.4f} mm")
        if not wait:
            return target
        time.sleep(MOVE_POLL_S)
        to = self._move_timeout(target - start) if timeout_s is None else float(timeout_s)
        self._wait_ready(f"move to {target:.4f} mm", to, controller, on_poll)
        return self.get_position()

    def move_by(self, distance_mm: float, wait: bool = True, timeout_s: Optional[float] = None,
                controller: Optional[MoveController] = None) -> float:
        return self.move_to(self.get_position() + float(distance_mm), wait=wait,
                            timeout_s=timeout_s, controller=controller)


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Newport SMC100 quick test")
    ap.add_argument("--list", action="store_true", help="list serial ports")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--address", type=int, default=DEFAULT_ADDRESS)
    ap.add_argument("--simulate", action="store_true", help="no hardware")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--move", type=float, default=None, help="absolute position (mm)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found (or pyserial missing).")
        for dev, desc in ports:
            print(f"{dev}\t{desc}")
        return 0

    with SMC100Stage(args.port, args.address, simulate=args.simulate, verbose=args.verbose) as stage:
        print(stage.info.text())
        print(f"State: {stage.get_state().text()}  Position: {stage.get_position():.4f} mm  "
              f"Limits: {stage.limits_mm}")
        if args.home:
            print("Homing...")
            stage.home()
            print(f"Homed: {stage.get_state().text()}  Position: {stage.get_position():.4f} mm")
        if args.move is not None:
            print(f"Moving to {args.move:.4f} mm...")
            pos = stage.move_to(args.move)
            print(f"Done: {pos:.4f} mm  ({stage.get_state().text()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
