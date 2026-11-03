"""Thorlabs MFF101 / MFF102 motorized filter flipper (Kinesis / APT protocol over USB).

Built on ``pylablib.devices.Thorlabs.MFF`` (FTDI via ft232, like the rotation
stages in ``rot/rot_wrapper.py``).  No Qt; thread-safe through one ``RLock``.

Positions use the Kinesis numbering printed on the device: **1** and **2**
(pylablib's states 0 and 1).  ``get_position()`` returns ``None`` while the
flipper is in transit.

The device can be opened by one program at a time: close the Kinesis app (or
any other GUI using it) before connecting.

Command line::

    python -m flipper.flipper_wrapper --list
    python -m flipper.flipper_wrapper --serial 37005944 --info
    python -m flipper.flipper_wrapper --serial 37005944 --move 2
    python -m flipper.flipper_wrapper --simulate --move 2      # no hardware
"""

from __future__ import annotations

import argparse
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
    from pylablib.devices import Thorlabs
except Exception:  # pylablib missing: only the simulator works
    Thorlabs = None


DEFAULT_SERIAL = "37005944"
# pylablib checks the model string against the serial prefix and warns on every open:
#   "model number MFF002 doesn't match the device ID prefix 37(MFF10.)".  MFF101 / MFF102
# flippers do report "MFF002", so the warning is expected; hide only that one.
warnings.filterwarnings("ignore", message=r"model number MFF\d+ doesn't match the device ID prefix",
                        category=UserWarning)

SERIAL_PREFIX = "37"  # Kinesis serial prefix of MFF101 / MFF102 flippers
POSITIONS = (1, 2)
TRANSIT_TIME_RANGE_S = (0.3, 2.8)  # range accepted by the MFF firmware
MOVE_TIMEOUT_EXTRA_S = 3.0  # move timeout = transit time + this
MOVE_POLL_S = 0.05


class FilterFlipperError(RuntimeError):
    pass


class MoveController:
    """Abort flag for a move wait.  The flipper itself cannot be stopped mid-transit."""

    def __init__(self):
        self._abort = threading.Event()

    def abort(self) -> None:
        self._abort.set()

    def is_aborted(self) -> bool:
        return self._abort.is_set()


@dataclass
class FlipperInfo:
    serial: str
    model: str = ""
    firmware: str = ""
    hardware: str = ""
    notes: str = ""

    def text(self) -> str:
        parts = [f"SN {self.serial}"]
        if self.model:
            parts.append(self.model)
        if self.firmware:
            parts.append(f"FW {self.firmware}")
        if self.hardware:
            parts.append(f"HW {self.hardware}")
        return " | ".join(parts)


@dataclass
class MoveResult:
    target: int
    position: Optional[int]
    duration_s: float
    aborted: bool = False


# ----------------------------------------------------------------------
# Device discovery
# ----------------------------------------------------------------------
def list_kinesis_devices() -> List[Tuple[str, str]]:
    """All FTDI/Kinesis devices as ``(serial, description)``.

    A device that another program holds open is listed with an empty serial.
    """
    if Thorlabs is None:
        return []
    try:
        devs = Thorlabs.list_kinesis_devices(filter_ids=False)
    except Exception:
        return []
    out = []
    for d in devs or []:
        try:
            serial, desc = d[0], d[1]
        except Exception:
            continue
        if isinstance(serial, bytes):
            serial = serial.decode("ascii", errors="ignore")
        if isinstance(desc, bytes):
            desc = desc.decode("ascii", errors="ignore")
        out.append((str(serial or "").strip(), str(desc or "").strip()))
    return out


def list_flipper_serials() -> List[str]:
    """Serials of the flippers that are free to open (MFF serials start with 37)."""
    return [s for s, _ in list_kinesis_devices() if s.startswith(SERIAL_PREFIX)]


def count_busy_devices() -> int:
    """Number of Kinesis devices that are open in another program (no serial readable)."""
    return sum(1 for s, _ in list_kinesis_devices() if not s)


# ----------------------------------------------------------------------
# Simulated device (development / GUI tests without hardware)
# ----------------------------------------------------------------------
class SimulatedMFF:
    """Minimal stand-in for ``pylablib.devices.Thorlabs.MFF`` (states 0/1, None in transit)."""

    def __init__(self, conn: str, transit_time_s: float = 0.5):
        self.conn = str(conn)
        self._state = 0
        self._target = 0
        self._move_end = 0.0
        self._transit = float(transit_time_s)
        self.blinks = 0

    def _update(self) -> None:
        if self._target != self._state and time.monotonic() >= self._move_end:
            self._state = self._target

    def move_to_state(self, state) -> None:
        self._update()
        state = 1 if state else 0
        if state != self._state or self._target != self._state:
            self._target = state
            self._move_end = time.monotonic() + self._transit

    def get_state(self):
        self._update()
        return None if self._target != self._state else self._state

    def get_status(self):
        st = self.get_state()
        if st is None:
            return ["moving_fw" if self._target else "moving_bk"]
        return ["sw_fw_lim"] if st else ["sw_bk_lim"]

    def get_flipper_parameters(self):
        return (self._transit,)

    def setup_flipper(self, transit_time=None, **_kw):
        if transit_time is not None:
            self._transit = float(transit_time)
        return (self._transit,)

    def get_device_info(self):
        return (self.conn, "MFF10x (simulated)", "", "sim", 0, "", 0)

    def blink(self) -> None:
        self.blinks += 1

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------
# Wrapper
# ----------------------------------------------------------------------
class FilterFlipper:
    """One MFF flipper.  ``open()`` / ``close()``, positions 1 and 2."""

    def __init__(self, serial: str = DEFAULT_SERIAL, *, simulate: bool = False, verbose: bool = False):
        self.serial = str(serial).strip()
        self.simulate = bool(simulate)
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self._dev = None
        self._last_position: Optional[int] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[FilterFlipper {self.serial}] {msg}")

    # ------------------------------------------------------------ connection
    @property
    def is_open(self) -> bool:
        return self._dev is not None

    def open(self) -> FlipperInfo:
        self.close()
        if not self.serial:
            raise FilterFlipperError("No serial number given")
        with self._lock:
            if self.simulate:
                self._dev = SimulatedMFF(self.serial)
            else:
                if Thorlabs is None:
                    raise FilterFlipperError("pylablib is not available")
                try:
                    self._dev = Thorlabs.MFF(self.serial)
                except Exception as exc:
                    self._dev = None
                    raise FilterFlipperError(self._open_error_text(exc)) from exc
            self._log("opened")
            try:
                info = self.get_info()
                self._last_position = self.get_position()
            except Exception as exc:
                self.close()
                raise FilterFlipperError(f"Opened {self.serial} but it does not answer: {exc}") from exc
            return info

    def _open_error_text(self, exc: Exception) -> str:
        free = [s for s, _ in list_kinesis_devices() if s]
        busy = count_busy_devices()
        if self.serial in free:
            return f"Failed to open flipper {self.serial}: {exc}"
        hint = (f" {busy} Kinesis device(s) are open in another program (Kinesis app or another GUI);"
                f" close it there and retry.") if busy else " Is it plugged in and powered?"
        return f"Flipper {self.serial} not available.{hint}"

    def close(self) -> None:
        with self._lock:
            if self._dev is not None:
                try:
                    self._dev.close()
                except Exception:
                    pass
                self._log("closed")
            self._dev = None

    def _call(self, fn_name: str, *args, **kwargs):
        with self._lock:
            if self._dev is None:
                raise FilterFlipperError("Flipper not connected")
            try:
                return getattr(self._dev, fn_name)(*args, **kwargs)
            except FilterFlipperError:
                raise
            except Exception as exc:
                raise FilterFlipperError(f"{fn_name} failed: {exc}") from exc

    # ------------------------------------------------------------ queries
    def get_info(self) -> FlipperInfo:
        raw = self._call("get_device_info")
        info = FlipperInfo(serial=self.serial)
        try:
            # pylablib TDeviceInfo(serial_no, model_no, fw_ver, hw_type, hw_ver, notes, nchannels)
            info.model = str(getattr(raw, "model_no", raw[1]) or "").strip()
            info.firmware = str(getattr(raw, "fw_ver", raw[2]) or "").strip()
            info.hardware = str(getattr(raw, "hw_ver", raw[4]) or "").strip()
            info.notes = str(getattr(raw, "notes", raw[5]) or "").strip()
        except Exception:
            pass
        return info

    def get_position(self) -> Optional[int]:
        """1 or 2, or ``None`` while the flipper is in transit."""
        state = self._call("get_state")
        pos = None if state is None else (2 if int(state) else 1)
        if pos is not None:
            self._last_position = pos
        return pos

    @property
    def last_position(self) -> Optional[int]:
        return self._last_position

    def get_status(self) -> List[str]:
        try:
            return [str(s) for s in self._call("get_status")]
        except FilterFlipperError:
            raise
        except Exception:
            return []

    def is_moving(self) -> bool:
        return self.get_position() is None

    def get_transit_time_s(self) -> float:
        params = self._call("get_flipper_parameters")
        return float(getattr(params, "transit_time", params[0]))

    def set_transit_time_s(self, seconds: float) -> float:
        lo, hi = TRANSIT_TIME_RANGE_S
        t = min(hi, max(lo, float(seconds)))
        self._call("setup_flipper", transit_time=t)
        self._log(f"transit time {t:.3f} s")
        return self.get_transit_time_s()

    def identify(self) -> None:
        """Blink the LED on the device."""
        self._call("blink")

    # ------------------------------------------------------------ motion
    def start_move(self, position: int) -> None:
        """Send the move command and return immediately."""
        position = int(position)
        if position not in POSITIONS:
            raise FilterFlipperError(f"Position must be 1 or 2, not {position}")
        self._call("move_to_state", 1 if position == 2 else 0)
        self._log(f"move to {position}")

    def wait_move(
        self,
        target: int,
        timeout_s: Optional[float] = None,
        controller: Optional[MoveController] = None,
        on_poll: Optional[Callable[[Optional[int]], None]] = None,
    ) -> MoveResult:
        """Poll until the flipper reports ``target`` (raises on timeout)."""
        if timeout_s is None:
            try:
                timeout_s = self.get_transit_time_s() + MOVE_TIMEOUT_EXTRA_S
            except FilterFlipperError:
                timeout_s = TRANSIT_TIME_RANGE_S[1] + MOVE_TIMEOUT_EXTRA_S
        t0 = time.monotonic()
        while True:
            pos = self.get_position()
            if on_poll is not None:
                try:
                    on_poll(pos)
                except Exception:
                    pass
            if pos == int(target):
                return MoveResult(int(target), pos, time.monotonic() - t0)
            if controller is not None and controller.is_aborted():
                return MoveResult(int(target), pos, time.monotonic() - t0, aborted=True)
            if time.monotonic() - t0 > float(timeout_s):
                raise FilterFlipperError(
                    f"Flipper did not reach position {target} within {timeout_s:.1f} s (now: {pos or 'in transit'})")
            time.sleep(MOVE_POLL_S)

    def move_to(
        self,
        position: int,
        *,
        wait: bool = True,
        timeout_s: Optional[float] = None,
        controller: Optional[MoveController] = None,
        on_poll: Optional[Callable[[Optional[int]], None]] = None,
    ) -> MoveResult:
        """Move to position 1 or 2; with ``wait`` block until it is reached."""
        t0 = time.monotonic()
        self.start_move(position)
        if not wait:
            return MoveResult(int(position), None, 0.0)
        res = self.wait_move(position, timeout_s=timeout_s, controller=controller, on_poll=on_poll)
        res.duration_s = time.monotonic() - t0
        return res

    def toggle(self, **kwargs) -> MoveResult:
        pos = self.get_position()
        if pos is None:
            pos = self._last_position
        target = 1 if pos == 2 else 2
        return self.move_to(target, **kwargs)

    def __enter__(self) -> "FilterFlipper":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Thorlabs MFF filter flipper")
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--simulate", action="store_true", help="use the built-in simulated flipper")
    ap.add_argument("--list", action="store_true", help="list Kinesis devices")
    ap.add_argument("--info", action="store_true", help="print device info, position and transit time")
    ap.add_argument("--move", type=int, choices=POSITIONS, help="move to position 1 or 2 and wait")
    ap.add_argument("--toggle", action="store_true")
    ap.add_argument("--transit", type=float, help="set the transit time (s, 0.3-2.8)")
    ap.add_argument("--identify", action="store_true", help="blink the LED")
    args = ap.parse_args()

    if args.list:
        devs = list_kinesis_devices()
        if not devs:
            print("No Kinesis devices found.")
        for s, d in devs:
            print(f"  {s or '(open in another program)':<28} {d}")
        if not (args.info or args.move or args.toggle or args.transit or args.identify):
            return

    with FilterFlipper(args.serial, simulate=args.simulate, verbose=True) as ff:
        if args.transit is not None:
            print(f"transit time -> {ff.set_transit_time_s(args.transit):.3f} s")
        if args.identify:
            ff.identify()
        if args.move:
            r = ff.move_to(args.move)
            print(f"moved to {r.position} in {r.duration_s:.2f} s")
        if args.toggle:
            r = ff.toggle()
            print(f"toggled to {r.position} in {r.duration_s:.2f} s")
        if args.info or not (args.move or args.toggle or args.transit or args.identify):
            print(ff.get_info().text())
            print(f"position: {ff.get_position()}  status: {ff.get_status()}")
            print(f"transit time: {ff.get_transit_time_s():.3f} s")


if __name__ == "__main__":
    main()
