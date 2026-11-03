"""TCSPC (timestamp stream) wrapper for the SPAD23 over the pSPAD command server.

``S,<ms>`` streams one text line per event:

* ``id,coarse,fine_ps``  photon on pixel ``id - PIXEL_ID_OFFSET``; ``fine_ps``
  is the arrival time after the laser clock in picoseconds
* ``23,0,k``             coarse-counter rollover (k counts the rollovers; id = offset + 23)
* ``id,coarse``          other markers, e.g. ``119,0`` as the first line

and ends with ``DONE`` (no newline) or with a message that ends in ``ERROR``.
Sending the same ``S`` command again cancels a running measurement (vendor
note in the pSPAD example scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import re
import socket
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False

from spad23.spad23_geometry import NUM_PIXELS
from spad23.spad23_pspad_wrapper import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MARKER_NAMES,
    Spad23ConnectionError,
    Spad23CountClient,
    Spad23Error,
    Spad23ProtocolError,
    Spad23ShutdownError,
    raise_for_error_text,
)

__all__ = [
    "MAX_BINS",
    "Spad23ConnectionError",
    "Spad23Error",
    "Spad23ProtocolError",
    "Spad23ShutdownError",
    "Spad23StreamError",
    "Spad23TcspcClient",
    "TrplAccumulator",
    "TrplSnapshot",
]

# Histogram size limit (bins); 200k bins x 23 pixels x int32 = 18 MB.
MAX_BINS = 200_000

UpdateCallback = Callable[["TrplSnapshot"], None]
StopCallback = Callable[[], bool]

_TEXT_RE = re.compile(rb"[A-Za-z]")
_CSV_KW = dict(
    header=None,
    names=("id", "coarse", "fine"),
    usecols=(0, 1, 2),
    engine="c",
    on_bad_lines="skip",
    skip_blank_lines=True,
    dtype=np.float64,
)


class Spad23StreamError(Spad23ProtocolError):
    """The stream ended without DONE (connection lost or timed out); ``snapshot`` has the partial data."""


@dataclass(frozen=True)
class TrplSnapshot:
    measurement_ms: int
    bin_width_ps: int
    counts_matrix: np.ndarray  # shape: [time_bins, pixels]
    warnings: Tuple[str, ...]
    elapsed_s: float
    done: bool
    n_events: int = 0
    marker_counts: Tuple[Tuple[int, int], ...] = ()
    out_of_window: int = 0
    cancelled: bool = False

    @property
    def time_axis_ps(self) -> np.ndarray:
        return np.arange(self.counts_matrix.shape[0], dtype=np.int64) * self.bin_width_ps

    @property
    def time_axis_ns(self) -> np.ndarray:
        return np.arange(self.counts_matrix.shape[0], dtype=np.float64) * (self.bin_width_ps / 1000.0)

    @property
    def pixel_totals(self) -> np.ndarray:
        return np.sum(self.counts_matrix, axis=0).astype(np.int64)

    @property
    def total_counts(self) -> int:
        return int(np.sum(self.counts_matrix, dtype=np.int64))


def _parse_numeric_slow(block: bytes) -> np.ndarray:
    rows: List[List[float]] = []
    for line in block.split(b"\n"):
        parts = line.strip().split(b",")
        if not parts[0]:
            continue
        try:
            vals = [float(p) for p in parts[:3]]
        except ValueError:
            continue
        while len(vals) < 3:
            vals.append(np.nan)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def parse_numeric_block(block: bytes) -> np.ndarray:
    """Parse numeric ``a,b[,c]`` lines into an ``[n, 3]`` float array (NaN for missing fields)."""
    if not block.strip():
        return np.zeros((0, 3), dtype=np.float64)
    if _HAS_PANDAS:
        try:
            df = pd.read_csv(io.BytesIO(block), **_CSV_KW)
            return df.to_numpy(dtype=np.float64)
        except Exception:
            pass
    return _parse_numeric_slow(block)


class TrplAccumulator:
    """Bins raw ``S`` stream bytes into a ``[time_bin, pixel]`` histogram.

    Complete lines are parsed block-wise with the pandas C parser (falls back
    to a Python loop); text lines (DONE, ERROR, messages) are found with a
    regex scan and handled one by one.
    """

    def __init__(self, bin_width_ps: int, id_lut: np.ndarray, max_bins: int = MAX_BINS):
        self.bin_width_ps = int(bin_width_ps)
        if self.bin_width_ps <= 0:
            raise ValueError("bin_width_ps must be > 0")
        self.max_bins = max(1, int(max_bins))
        self._lut = np.asarray(id_lut, dtype=np.int64)
        self._matrix = np.zeros((min(256, self.max_bins), NUM_PIXELS), dtype=np.int32)
        self._used = 0
        self._pending = b""
        self.n_events = 0
        self.out_of_window = 0
        self.negative_times = 0
        self.marker_counts: Dict[int, int] = {}
        self.text_lines: List[str] = []
        self.done = False
        self.error: Optional[str] = None

    # ------------------------------------------------------------- feeding
    def feed(self, data: bytes) -> None:
        if not data:
            return
        buf = self._pending + data
        cut = buf.rfind(b"\n")
        if cut >= 0:
            self._pending = buf[cut + 1:]
            self._consume_block(buf[: cut + 1])
        else:
            self._pending = buf
        self._check_tail()

    def finish(self) -> None:
        """Consume whatever is left after the stream ended."""
        tail = self._pending.strip()
        self._pending = b""
        if tail:
            self._consume_block(tail + b"\n")

    def _check_tail(self) -> None:
        # DONE / ERROR arrive without a trailing newline.
        tail = self._pending.strip()
        if not tail:
            return
        if tail.endswith(b"DONE"):
            head = tail[:-4].strip()
            self._pending = b""
            if head:
                self._consume_block(head + b"\n")
            self.done = True
        elif tail.endswith(b"ERROR"):
            self._pending = b""
            self.error = tail.decode("utf8", errors="ignore")[-300:]

    def _consume_block(self, block: bytes) -> None:
        if _TEXT_RE.search(block) is None:
            self._parse_numeric(block)
            return
        numeric: List[bytes] = []
        texts: List[bytes] = []
        for line in block.split(b"\n"):
            s = line.strip()
            if not s:
                continue
            if _TEXT_RE.search(s) is None:
                numeric.append(s)
            else:
                texts.append(s)
        if numeric:
            self._parse_numeric(b"\n".join(numeric) + b"\n")
        for s in texts:
            self._handle_text(s)

    def _handle_text(self, line: bytes) -> None:
        text = line.decode("utf8", errors="ignore").strip()
        upper = text.upper()
        if upper.endswith("DONE"):
            self.done = True
            return
        if "ERROR" in upper:
            self.error = text[-300:]
            return
        if len(self.text_lines) < 50:
            self.text_lines.append(text)

    def _parse_numeric(self, block: bytes) -> None:
        arr = parse_numeric_block(block)
        if arr.shape[0] == 0:
            return
        ids = arr[:, 0]
        fine = arr[:, 2]
        id_ok = np.isfinite(ids)
        ids_i = np.where(id_ok, ids, -1).astype(np.int64)
        in_lut = (ids_i >= 0) & (ids_i < self._lut.size)
        pix = np.full(ids_i.shape, -1, dtype=np.int64)
        pix[in_lut] = self._lut[ids_i[in_lut]]
        is_event = (pix >= 0) & np.isfinite(fine)

        other = id_ok & ~is_event
        if other.any():
            uniq, cnt = np.unique(ids_i[other], return_counts=True)
            for k, v in zip(uniq.tolist(), cnt.tolist()):
                self.marker_counts[k] = self.marker_counts.get(k, 0) + v
        if not is_event.any():
            return

        f = fine[is_event]
        p = pix[is_event]
        neg = f < 0
        if neg.any():
            self.negative_times += int(neg.sum())
            f = f[~neg]
            p = p[~neg]
        b = (f // self.bin_width_ps).astype(np.int64)
        inside = b < self.max_bins
        if not inside.all():
            self.out_of_window += int((~inside).sum())
            b = b[inside]
            p = p[inside]
        if b.size == 0:
            return
        self._ensure_bins(int(b.max()) + 1)
        np.add.at(self._matrix, (b, p), 1)
        self.n_events += int(b.size)

    def _ensure_bins(self, n_bins: int) -> None:
        if n_bins > self._matrix.shape[0]:
            new_n = self._matrix.shape[0]
            while new_n < n_bins:
                new_n *= 2
            new_n = min(new_n, self.max_bins)
            grown = np.zeros((new_n, NUM_PIXELS), dtype=np.int32)
            grown[: self._matrix.shape[0], :] = self._matrix
            self._matrix = grown
        self._used = max(self._used, n_bins)

    # ------------------------------------------------------------ snapshot
    def summary_warnings(self, extra: Optional[str] = None) -> Tuple[str, ...]:
        out: List[str] = []
        if self.out_of_window:
            out.append(
                f"{self.out_of_window} events beyond the {self.max_bins}-bin window "
                f"({self.max_bins * self.bin_width_ps / 1000.0:.0f} ns); increase the bin width."
            )
        if self.negative_times:
            out.append(f"{self.negative_times} events with negative time dropped.")
        unknown = {k: v for k, v in self.marker_counts.items() if k not in MARKER_NAMES}
        if unknown:
            ids = ", ".join(f"{k} (x{v})" for k, v in sorted(unknown.items())[:12])
            out.append(f"Ids that are not pixels: {ids}.")
        for t in self.text_lines[:5]:
            out.append(f"Server: {t}")
        if self.error:
            out.append(f"Server error: {self.error}")
        if extra:
            out.append(extra)
        return tuple(out)

    def snapshot(
        self,
        measurement_ms: int,
        elapsed_s: float,
        done: bool,
        cancelled: bool = False,
        extra_warning: Optional[str] = None,
    ) -> TrplSnapshot:
        return TrplSnapshot(
            measurement_ms=int(measurement_ms),
            bin_width_ps=self.bin_width_ps,
            counts_matrix=self._matrix[: self._used, :].copy(),
            warnings=self.summary_warnings(extra_warning),
            elapsed_s=float(elapsed_s),
            done=bool(done),
            n_events=int(self.n_events),
            marker_counts=tuple(sorted(self.marker_counts.items())),
            out_of_window=int(self.out_of_window),
            cancelled=bool(cancelled),
        )


class Spad23TcspcClient(Spad23CountClient):
    """
    Timestamp-stream (TCSPC) client; inherits the count and TDC commands.

    Commands used:
    - T,v,1 / T,c,1 : verify / run TDC calibration
    - T,a,<ms>      : channel alignment to the laser clock
    - S,<ms>        : text timestamp stream (resend to cancel)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = 5.0,
        quiet_timeout_s: float = 0.05,
        recv_bytes: int = 2_097_152,
        stream_idle_timeout_s: float = 15.0,
        **kwargs,
    ):
        super().__init__(host=host, port=port, timeout_s=timeout_s, quiet_timeout_s=quiet_timeout_s, **kwargs)
        self.recv_bytes = int(recv_bytes)
        self.stream_idle_timeout_s = float(stream_idle_timeout_s)

    def __enter__(self) -> "Spad23TcspcClient":
        self.connect()
        return self

    def acquire_trpl_stream(
        self,
        measurement_ms: int,
        bin_width_ps: int = 10,
        on_update: Optional[UpdateCallback] = None,
        update_interval_s: float = 0.5,
        should_stop: Optional[StopCallback] = None,
        max_bins: int = MAX_BINS,
        cancel_wait_s: float = 3.0,
    ) -> TrplSnapshot:
        """
        Run ``S,<measurement_ms>`` and bin the fine timestamps into a [time x pixel] matrix.

        ``should_stop`` is polled about every 0.2 s; when it returns True the
        command is sent again (vendor cancel) and the data received so far is
        returned with ``cancelled=True``.  Server errors raise with the partial
        data attached as ``exc.snapshot``.
        """
        measurement_ms = int(measurement_ms)
        bin_width_ps = int(bin_width_ps)
        if measurement_ms <= 0:
            raise ValueError("measurement_ms must be > 0")
        if bin_width_ps <= 0:
            raise ValueError("bin_width_ps must be > 0")

        command = f"S,{measurement_ms}"
        with self._lock:
            self._ensure_connected()
            self._drain()
            if self._sock is None:
                self.connect()
            accum = TrplAccumulator(bin_width_ps=bin_width_ps, id_lut=self.id_lut, max_bins=max_bins)
            self._send(command)
            sock = self._require_sock()

            start = time.monotonic()
            last_emit = start
            last_data = start
            expected_end = start + measurement_ms / 1000.0
            cancel_at: Optional[float] = None
            lost: Optional[str] = None
            sock.settimeout(0.2)
            try:
                while True:
                    if cancel_at is None and should_stop is not None and should_stop():
                        try:
                            sock.sendall((command + "\n").encode("utf8"))
                        except OSError as exc:
                            lost = f"Could not send cancel: {exc}"
                            break
                        cancel_at = time.monotonic()
                    try:
                        data: Optional[bytes] = sock.recv(self.recv_bytes)
                    except socket.timeout:
                        data = None
                    except OSError as exc:
                        lost = f"Connection lost during stream: {exc}"
                        break
                    now = time.monotonic()
                    if data is not None:
                        if not data:
                            lost = "Server closed the connection during the stream."
                            break
                        last_data = now
                        accum.feed(data)
                        if accum.done or accum.error is not None:
                            break
                    if cancel_at is not None and now - cancel_at > cancel_wait_s:
                        break
                    if cancel_at is None and now > expected_end and now - last_data > self.stream_idle_timeout_s:
                        lost = f"No DONE within {self.stream_idle_timeout_s:.0f} s after the measurement end."
                        break
                    if on_update is not None and now - last_emit >= update_interval_s:
                        on_update(
                            accum.snapshot(measurement_ms, now - start, done=False, cancelled=cancel_at is not None)
                        )
                        last_emit = now
            finally:
                if self._sock is sock:
                    try:
                        sock.settimeout(self.timeout_s)
                    except OSError:
                        pass

            accum.finish()
            cancelled = cancel_at is not None
            if cancelled or not accum.done:
                # The reply was not read to a clean end: never reuse this socket.
                self.close()
            note = None
            if cancelled and not accum.done:
                note = "Cancelled; no DONE received after the cancel."
            final = accum.snapshot(
                measurement_ms,
                time.monotonic() - start,
                done=accum.done and accum.error is None,
                cancelled=cancelled,
                extra_warning=lost or note,
            )

        if on_update is not None:
            on_update(final)
        if accum.error is not None:
            raise_for_error_text(accum.error, command, snapshot=final)
            raise Spad23ProtocolError(f"Stream error: {accum.error}", snapshot=final)
        if lost is not None:
            raise Spad23StreamError(lost, snapshot=final)
        return final
