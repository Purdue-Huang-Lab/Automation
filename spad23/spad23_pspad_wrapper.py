"""Count-mode wrapper for the Pi Imaging SPAD23 over the pSPAD command server (TCP).

Protocol (pSPAD v2.02, the same server as the SPAD93; taken from recorded
SPAD93 data and the vendor example scripts -- re-check the SPAD23 ids with
``--probe``):

* every connection is greeted with a banner ("pSPAD command server")
* ``I,<ms>``   -> one ``id,count`` line per pixel, ids 0..22 (no DONE line)
* ``T,v,1``    -> "TDC calibration is valid" / "TDC calibration is invalid"
* ``T,c,1``    -> run the TDC calibration
* ``T,a,<ms>`` -> align the channels to the laser clock
* ``SI``       -> reset the system after an overexposure shutdown
* a rejected command or failed run ends with ``ERROR``

Run ``python -m spad23.spad23_pspad_wrapper --probe`` against the live server to
print the raw replies and a per-id table of a short timestamp stream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from spad23.spad23_geometry import NUM_PIXELS


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9998

# Pixel count-rate limit: 6.7 Mcps (23G) and 12.5 Mcps (23R).  The lower value warns earlier.
PIXEL_LIMIT_MCPS = 6.7

# raw id = pixel + PIXEL_ID_OFFSET (the old SPAD23.exe server used 32).  Change it
# with set_pixel_id_offset() (the GUIs have --id-offset) after checking --probe.
PIXEL_ID_OFFSET = 0
ID_LUT_SIZE = 1024
# Non-pixel ids seen in SPAD93 streams: "<offset + n_pixels>,0,k" when the 16-bit
# coarse counter wraps, "119,0" as the first line of a stream.
ROLLOVER_ID = PIXEL_ID_OFFSET + NUM_PIXELS
MARKER_NAMES: Dict[int, str] = {
    ROLLOVER_ID: "coarse counter rollover",
    119: "stream start marker",
}


def set_pixel_id_offset(offset: int) -> None:
    """Set the raw-id offset used by every client created afterwards."""
    global PIXEL_ID_OFFSET, ROLLOVER_ID
    MARKER_NAMES.pop(ROLLOVER_ID, None)
    PIXEL_ID_OFFSET = int(offset)
    ROLLOVER_ID = PIXEL_ID_OFFSET + NUM_PIXELS
    MARKER_NAMES[ROLLOVER_ID] = "coarse counter rollover"


class Spad23Error(RuntimeError):
    """Base class for SPAD23 wrapper errors."""


class Spad23ConnectionError(Spad23Error):
    """Raised for connection-level failures."""


class Spad23ProtocolError(Spad23Error):
    """Raised when the server reports an error or the reply is malformed.

    ``snapshot`` carries any partial data acquired before the error.
    """

    def __init__(self, message: str, snapshot=None):
        super().__init__(message)
        self.snapshot = snapshot


class Spad23ShutdownError(Spad23ProtocolError):
    """The system is in overexposure shutdown; send ``SI`` to reset it."""


class Spad23OverflowError(Spad23Error):
    """Raised when the device reports an overflow or the rate exceeds the limit."""

    def __init__(self, message: str, snapshot: Optional["CountSnapshot"] = None):
        super().__init__(message)
        self.snapshot = snapshot


def build_id_lut(offset: Optional[int] = None, num_pixels: int = NUM_PIXELS, size: int = ID_LUT_SIZE) -> np.ndarray:
    """Lookup table raw id -> pixel index (-1 for anything that is not a pixel)."""
    if offset is None:
        offset = PIXEL_ID_OFFSET
    lut = np.full(int(size), -1, dtype=np.int64)
    lut[int(offset): int(offset) + int(num_pixels)] = np.arange(int(num_pixels))
    return lut


def raise_for_error_text(text: str, command: str, snapshot=None) -> None:
    """Raise the matching exception if a server reply reports ERROR."""
    if "ERROR" not in text.upper():
        return
    msg = f"Server returned ERROR for '{command.strip()}': {text.strip()[-300:]}"
    lower = text.lower()
    if "shut down" in lower or "shutdown" in lower:
        raise Spad23ShutdownError(msg + " (send SI to reset)", snapshot=snapshot)
    raise Spad23ProtocolError(msg, snapshot=snapshot)


def _looks_like_overflow_warning(line: str) -> bool:
    lower = line.lower()
    return ("overflow" in lower) or ("shutdown" in lower) or ("shut down" in lower)


def _count_lines(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


@dataclass(frozen=True)
class CountSnapshot:
    integration_ms: int
    pixel_counts: np.ndarray
    pixel_limit_mcps: float
    warnings: Tuple[str, ...]
    raw_response: str

    @property
    def total_counts(self) -> int:
        return int(np.sum(self.pixel_counts))

    @property
    def pixel_rates_mcps(self) -> np.ndarray:
        dt_s = self.integration_ms / 1000.0
        if dt_s <= 0:
            return np.zeros(NUM_PIXELS, dtype=np.float64)
        return self.pixel_counts.astype(np.float64) / dt_s / 1e6

    @property
    def total_rate_mcps(self) -> float:
        return float(np.sum(self.pixel_rates_mcps))

    @property
    def max_pixel_index(self) -> int:
        return int(np.argmax(self.pixel_rates_mcps))

    @property
    def max_pixel_rate_mcps(self) -> float:
        return float(np.max(self.pixel_rates_mcps))

    @property
    def overload_ratio(self) -> float:
        if self.pixel_limit_mcps <= 0:
            return 0.0
        return self.max_pixel_rate_mcps / self.pixel_limit_mcps


class Spad23CountClient:
    """
    TCP wrapper around the SPAD23 count mode and system commands.

    The socket is used by one thread at a time (``_lock``); ``close`` does not
    take the lock so another thread can interrupt a blocking read.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = 2.0,
        quiet_timeout_s: float = 0.05,
        pixel_limit_mcps: float = PIXEL_LIMIT_MCPS,
        raise_on_limit: bool = False,
        raise_on_server_overflow: bool = True,
        id_offset: Optional[int] = None,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.quiet_timeout_s = float(quiet_timeout_s)
        self.pixel_limit_mcps = float(pixel_limit_mcps)
        self.raise_on_limit = bool(raise_on_limit)
        self.raise_on_server_overflow = bool(raise_on_server_overflow)
        self.id_lut = build_id_lut(id_offset)

        self._lock = threading.RLock()
        self._sock: Optional[socket.socket] = None
        self.server_banner: str = ""
        self.drained_bytes = 0
        self.last_drained = b""

    # ----------------------------------------------------------- connection
    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> str:
        with self._lock:
            if self._sock is not None:
                return self.server_banner
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_s)
            try:
                sock.connect((self.host, self.port))
                self._sock = sock
                self.server_banner = self._read_banner()
                return self.server_banner
            except Exception as exc:
                self._sock = None
                try:
                    sock.close()
                except OSError:
                    pass
                if isinstance(exc, Spad23Error):
                    raise
                raise Spad23ConnectionError(f"Could not connect to {self.host}:{self.port}: {exc}") from exc

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def reconnect(self) -> str:
        self.close()
        return self.connect()

    def __enter__(self) -> "Spad23CountClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------- commands
    def query(self, command: str, timeout_s: Optional[float] = None, min_lines: int = 0) -> str:
        """Send ``command`` and return the reply text (raises on ERROR replies)."""
        with self._lock:
            self._ensure_connected()
            self._drain()
            if self._sock is None:  # server had closed the old connection
                self.connect()
            self._send(command)
            payload = self._recv_reply(timeout_s if timeout_s is not None else self.timeout_s, min_lines, command)
        text = payload.decode("utf8", errors="ignore")
        raise_for_error_text(text, command)
        return text

    def initialize_device(self, run_calibration_if_invalid: bool = True) -> Dict[str, str]:
        """Check the TDC calibration and optionally run it."""
        out: Dict[str, str] = {}
        status = self.check_tdc_calibration()
        out["tdc_status"] = status
        if run_calibration_if_invalid and "invalid" in status.lower():
            out["calibration"] = self.run_tdc_calibration()
        return out

    def check_tdc_calibration(self) -> str:
        return self.query("T,v,1", timeout_s=max(self.timeout_s, 5.0)).strip()

    def run_tdc_calibration(self, timeout_s: float = 120.0) -> str:
        return self.query("T,c,1", timeout_s=timeout_s).strip()

    def run_channel_alignment(self, align_ms: int = 1000) -> str:
        align_ms = int(align_ms)
        if align_ms <= 0:
            raise ValueError("align_ms must be > 0")
        return self.query(f"T,a,{align_ms}", timeout_s=align_ms / 1000.0 + 30.0).strip()

    def calibrate_and_align(self, align_ms: int = 1000, calibrate_if_invalid: bool = True) -> List[str]:
        messages: List[str] = []
        status = self.check_tdc_calibration()
        messages.append(f"TDC status: {status}")
        if calibrate_if_invalid and "invalid" in status.lower():
            messages.append(f"TDC calibration: {self.run_tdc_calibration()}")
        messages.append(f"Alignment: {self.run_channel_alignment(align_ms=align_ms)}")
        return messages

    def reset_shutdown(self) -> str:
        """Send ``SI`` to leave the overexposure shutdown state."""
        return self.query("SI", timeout_s=max(self.timeout_s, 5.0)).strip()

    def get_counts_snapshot(self, integration_ms: int) -> CountSnapshot:
        integration_ms = int(integration_ms)
        if integration_ms <= 0:
            raise ValueError("integration_ms must be > 0")

        text = self.query(
            f"I,{integration_ms}",
            timeout_s=integration_ms / 1000.0 + self.timeout_s,
            min_lines=NUM_PIXELS,
        )
        snapshot = self._parse_count_response(text=text, integration_ms=integration_ms)

        has_overflow_warning = any(_looks_like_overflow_warning(w) for w in snapshot.warnings)
        if has_overflow_warning and self.raise_on_server_overflow:
            raise Spad23OverflowError("Server reported overflow/shutdown condition.", snapshot=snapshot)
        if snapshot.overload_ratio >= 1.0 and self.raise_on_limit:
            raise Spad23OverflowError(
                f"Max pixel rate {snapshot.max_pixel_rate_mcps:.3f} Mcps exceeds "
                f"limit {snapshot.pixel_limit_mcps:.3f} Mcps.",
                snapshot=snapshot,
            )
        return snapshot

    def read_pixel_count(self, pixel: int, integration_ms: int) -> int:
        pixel = self._validate_pixel(pixel)
        return int(self.get_counts_snapshot(integration_ms).pixel_counts[pixel])

    def read_total_count(self, integration_ms: int) -> int:
        return self.get_counts_snapshot(integration_ms).total_counts

    def read_pixel_rate_mcps(self, pixel: int, integration_ms: int = 100) -> float:
        pixel = self._validate_pixel(pixel)
        return float(self.get_counts_snapshot(integration_ms).pixel_rates_mcps[pixel])

    def read_total_rate_mcps(self, integration_ms: int = 100) -> float:
        return self.get_counts_snapshot(integration_ms).total_rate_mcps

    def read_all_pixel_rates_mcps(self, integration_ms: int = 100) -> np.ndarray:
        return self.get_counts_snapshot(integration_ms).pixel_rates_mcps.copy()

    # ------------------------------------------------------------- low level
    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def _require_sock(self) -> socket.socket:
        sock = self._sock
        if sock is None:
            raise Spad23ConnectionError("Connection is closed.")
        return sock

    def _send(self, command: str) -> None:
        sock = self._require_sock()
        cmd = command.rstrip("\n") + "\n"
        try:
            sock.sendall(cmd.encode("utf8"))
        except OSError as exc:
            self.close()
            raise Spad23ConnectionError(f"Connection failed while sending '{command}': {exc}") from exc

    def _drain(self) -> bytes:
        """Discard bytes left over from an earlier reply (never blocks)."""
        sock = self._require_sock()
        chunks: List[bytes] = []
        closed = False
        sock.setblocking(False)
        try:
            while True:
                try:
                    chunk = sock.recv(65536)
                except (BlockingIOError, InterruptedError, socket.timeout):
                    break
                except OSError:
                    closed = True
                    break
                if not chunk:
                    closed = True
                    break
                chunks.append(chunk)
        finally:
            if self._sock is sock:
                sock.settimeout(self.timeout_s)
        data = b"".join(chunks)
        if data:
            self.drained_bytes += len(data)
            self.last_drained = data[-200:]
        if closed:
            self.close()
        return data

    def _read_banner(self) -> str:
        try:
            raw = self._recv_reply(timeout_s=1.0, min_lines=0, command="<banner>")
        except Spad23ConnectionError:
            return ""
        return raw.decode("utf8", errors="ignore").strip()

    def _recv_reply(self, timeout_s: float, min_lines: int, command: str) -> bytes:
        """Read one reply: wait for the first bytes, then until the line stops.

        The reply is complete after ``quiet_timeout_s`` of silence once at
        least ``min_lines`` lines arrived, or when it ends with DONE/ERROR.
        """
        sock = self._require_sock()
        deadline = time.monotonic() + float(timeout_s)
        chunks: List[bytes] = []
        sock.settimeout(max(float(timeout_s), 0.01))
        try:
            try:
                first = sock.recv(65536)
            except socket.timeout as exc:
                raise Spad23ConnectionError(f"Timed out waiting for reply to '{command}'.") from exc
            except OSError as exc:
                self.close()
                raise Spad23ConnectionError(f"Connection lost waiting for reply to '{command}': {exc}") from exc
            if not first:
                self.close()
                raise Spad23ConnectionError(f"Server closed the connection (reply to '{command}').")
            chunks.append(first)
            n_bytes_lines = _count_lines(first)
            sock.settimeout(self.quiet_timeout_s)
            while True:
                tail = chunks[-1].rstrip()
                if tail.endswith(b"DONE") or tail.endswith(b"ERROR"):
                    break
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    if n_bytes_lines >= min_lines or time.monotonic() >= deadline:
                        break
                    continue
                except OSError:
                    self.close()
                    break
                if not chunk:
                    self.close()
                    break
                chunks.append(chunk)
                n_bytes_lines = _count_lines(b"".join(chunks))
        finally:
            if self._sock is sock:
                sock.settimeout(self.timeout_s)
        return b"".join(chunks)

    def _map_id(self, raw_id: int) -> int:
        if 0 <= raw_id < self.id_lut.size:
            return int(self.id_lut[raw_id])
        return -1

    @staticmethod
    def _validate_pixel(pixel: int) -> int:
        pixel = int(pixel)
        if pixel < 0 or pixel >= NUM_PIXELS:
            raise ValueError(f"pixel must be in [0, {NUM_PIXELS - 1}], got {pixel}")
        return pixel

    def _parse_count_response(self, text: str, integration_ms: int) -> CountSnapshot:
        counts = np.zeros(NUM_PIXELS, dtype=np.int64)
        warnings: List[str] = []
        unknown: Dict[int, int] = {}
        seen = np.zeros(NUM_PIXELS, dtype=bool)

        for raw_line in text.replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "DONE":
                continue
            if _looks_like_overflow_warning(line):
                warnings.append(line)
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                warnings.append(f"Unparsed line: {line}")
                continue
            try:
                raw_id = int(parts[0])
                count_val = int(float(parts[1]))
            except ValueError:
                warnings.append(f"Unparsed line: {line}")
                continue
            pixel = self._map_id(raw_id)
            if pixel < 0:
                unknown[raw_id] = unknown.get(raw_id, 0) + 1
                continue
            counts[pixel] += max(0, count_val)
            seen[pixel] = True

        if not seen.any():
            raise Spad23ProtocolError(
                "Could not parse any pixel counts from server response. "
                f"Raw response was: {text.strip()[:300]}"
            )
        if not seen.all():
            warnings.append(f"Reply had {int(seen.sum())} of {NUM_PIXELS} pixels.")
        if unknown:
            ids = ", ".join(f"{k} (x{v})" for k, v in sorted(unknown.items()))
            warnings.append(f"Ids that are not pixels: {ids}. Check PIXEL_ID_OFFSET with --probe.")

        return CountSnapshot(
            integration_ms=integration_ms,
            pixel_counts=counts,
            pixel_limit_mcps=self.pixel_limit_mcps,
            warnings=tuple(warnings),
            raw_response=text,
        )


# --------------------------------------------------------------------- probe
def _probe(host: str, port: int, stream_ms: int, record_path: str) -> None:
    client = Spad23CountClient(host=host, port=port, timeout_s=5.0)
    print(f"Connecting to {host}:{port} ...")
    banner = client.connect()
    print(f"banner: {banner!r}")
    try:
        print(f"T,v,1 -> {client.check_tdc_calibration()!r}")
    except Spad23Error as exc:
        print(f"T,v,1 failed: {exc}")

    try:
        raw = client.query("I,100", timeout_s=5.0, min_lines=NUM_PIXELS)
        lines = [ln for ln in raw.replace("\r", "").split("\n") if ln.strip()]
        print(f"I,100 -> {len(lines)} lines; first: {lines[:3]}  last: {lines[-3:]}")
        ids = []
        for ln in lines:
            try:
                ids.append(int(ln.split(",")[0]))
            except ValueError:
                print(f"  non-numeric line: {ln!r}")
        if ids:
            print(f"  ids {min(ids)}..{max(ids)} ({len(set(ids))} distinct)")
    except Spad23Error as exc:
        print(f"I,100 failed: {exc}")

    print(f"S,{stream_ms} ...")
    sock = client._require_sock()
    client._drain()
    client._send(f"S,{stream_ms}")
    chunks: List[bytes] = []
    sock.settimeout(0.5)
    t_end = time.monotonic() + stream_ms / 1000.0 + 15.0
    while time.monotonic() < t_end:
        try:
            chunk = sock.recv(1 << 20)
        except socket.timeout:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        tail = b"".join(chunks[-2:]).rstrip()
        if tail.endswith(b"DONE") or tail.endswith(b"ERROR"):
            break
    data = b"".join(chunks)
    client.close()
    if record_path:
        with open(record_path, "wb") as fh:
            fh.write(data)
        print(f"raw stream written to {record_path} ({len(data)} bytes)")

    table: Dict[Tuple[int, int], List[int]] = {}
    texts: List[str] = []
    for ln in data.replace(b"\r", b"").split(b"\n"):
        if not ln.strip():
            continue
        parts = ln.split(b",")
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            texts.append(ln.decode("utf8", errors="ignore"))
            continue
        key = (vals[0], len(vals))
        rec = table.setdefault(key, [0, 2**62, -2**62, 2**62, -2**62])
        rec[0] += 1
        if len(vals) > 1:
            rec[1] = min(rec[1], vals[1])
            rec[2] = max(rec[2], vals[1])
        if len(vals) > 2:
            rec[3] = min(rec[3], vals[2])
            rec[4] = max(rec[4], vals[2])
    print(f"stream: {len(data)} bytes; tail {data[-60:]!r}")
    print(" id  fields    count   field2 min..max        field3 min..max")
    for (rid, nf), (cnt, a0, a1, b0, b1) in sorted(table.items()):
        f2 = f"{a0}..{a1}" if nf > 1 else "-"
        f3 = f"{b0}..{b1}" if nf > 2 else "-"
        name = MARKER_NAMES.get(rid, "pixel" if 0 <= rid - PIXEL_ID_OFFSET < NUM_PIXELS else "?")
        print(f"{rid:4d} {nf:6d} {cnt:8d}   {f2:>20s}   {f3:>20s}  {name}")
    for t in texts[:20]:
        print(f"text line: {t!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPAD23 count-mode wrapper / protocol probe")
    parser.add_argument("--probe", action="store_true", help="print raw replies of the live server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--stream-ms", type=int, default=500, help="length of the probe stream")
    parser.add_argument("--record", default="", help="write the raw probe stream to this file")
    parser.add_argument("--id-offset", type=int, default=PIXEL_ID_OFFSET, help="raw id of pixel 0")
    args = parser.parse_args()
    set_pixel_id_offset(args.id_offset)
    if args.probe:
        _probe(args.host, args.port, args.stream_ms, args.record)
        return
    with Spad23CountClient(host=args.host, port=args.port) as client:
        snap = client.get_counts_snapshot(200)
        print(f"total {snap.total_rate_mcps:.4f} Mcps, max pixel {snap.max_pixel_index}")


if __name__ == "__main__":
    main()
