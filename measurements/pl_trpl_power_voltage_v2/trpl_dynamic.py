"""Dynamic TRPL time: acquire in chunks until the brightest pixel's peak is high enough.

Pure helpers (no Qt, no device I/O) used by the sweep worker:

- ``pad_add``: sums [time_bin, pixel] histograms of different lengths.  The pSPAD clients return
  ``counts_matrix[:used]`` (rows up to the last bin that got an event), so two chunks of the same
  spot can have different row counts.  Row 0 is always t = 0 (bin = fine_time // bin_width), so
  zero-padding at the END and adding is exact.
- ``ChunkPlan``: the chunk durations (first, step, step, ...; the last one clipped at the max).
- ``peak_counts``: background-subtracted peak over all (used) pixels and bins.
- ``combine_snapshot``: one snapshot of the summed counts with the total requested time.
- ``DynamicLog``: the per-spot ``dynamic_trpl_log.csv`` in the sweep folder.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import threading
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOG_NAME = "dynamic_trpl_log.csv"  # must not start with "trpl_": glob("TRPL_*.csv") ignores case on Windows
CHUNK_DIR = "trpl_chunks"
LOG_HEADER = ["file", "step_idx", "power_w", "v_front", "v_back", "pump2", "n_chunks", "chunk_s", "total_s",
              "peak_counts", "peak_pixel", "peak_t_ns", "target_counts", "max_s", "background", "reached", "reason"]


@dataclass(frozen=True)
class DynamicSettings:
    target_counts: float = 500.0
    first_s: float = 60.0
    step_s: float = 60.0
    max_s: float = 600.0
    # dark reference (an ``an.DarkReference``: per_bin(acq_s, bin_ps)) or None -> pre-t0 mean
    dark_ref: object = None
    # pre-t0 window: None = automatic (pre_pulse_window), else (w0_ns, w1_ns)
    pre_t0_window: Optional[Tuple[float, float]] = None
    excluded: Tuple[int, ...] = ()

    def validate(self) -> Optional[str]:
        if not self.target_counts > 0:
            return "the target peak must be > 0 counts"
        if not self.first_s > 0 or not self.step_s > 0:
            return "the first chunk and the step must be > 0 s"
        if self.max_s < self.first_s - 1e-9:
            return "the max time must be at least the first chunk"
        return None


class ChunkPlan:
    """Chunk durations in ms: ``first``, then ``step`` until the total reaches ``max`` (last one clipped)."""

    def __init__(self, first_s: float, step_s: float, max_s: float):
        self.first_ms = max(1, int(round(float(first_s) * 1000.0)))
        self.step_ms = max(1, int(round(float(step_s) * 1000.0)))
        self.max_ms = max(self.first_ms, int(round(float(max_s) * 1000.0)))

    def next_ms(self, done_ms: Sequence[int]) -> Optional[int]:
        """The next chunk after the chunks ``done_ms``; None when the max time is reached."""
        total = int(sum(done_ms))
        if total >= self.max_ms:
            return None
        want = self.first_ms if not done_ms else self.step_ms
        return int(min(want, self.max_ms - total))


def pad_add(acc: Optional[np.ndarray], counts: np.ndarray) -> np.ndarray:
    """``acc + counts`` as int64 [bins, pixels], both zero-padded at the end to the longer one."""
    c = np.asarray(counts)
    if c.ndim != 2:
        raise ValueError(f"counts must be [bins, pixels], got shape {c.shape}")
    if acc is None:
        return c.astype(np.int64, copy=True)
    if acc.shape[1] != c.shape[1]:
        raise ValueError(f"pixel count changed between chunks ({acc.shape[1]} vs {c.shape[1]})")
    n = max(acc.shape[0], c.shape[0])
    out = np.zeros((n, acc.shape[1]), dtype=np.int64)
    out[: acc.shape[0]] += acc
    out[: c.shape[0]] += c
    return out


def background(counts: np.ndarray, bin_ps: float, total_s: float, st: DynamicSettings,
               pre_pulse_window=None) -> Tuple[Optional[np.ndarray], str]:
    """Per-pixel background per bin: dark reference scaled to ``total_s``, else the pre-t0 mean."""
    n_pix = counts.shape[1]
    ref = st.dark_ref
    if ref is not None and int(getattr(ref, "num_pixels", -1)) == n_pix:
        return np.asarray(ref.per_bin(float(total_s), float(bin_ps)), dtype=np.float64), "dark_ref"
    if counts.shape[0] == 0:
        return None, "none"
    t_ns = np.arange(counts.shape[0], dtype=np.float64) * float(bin_ps) / 1000.0
    win = st.pre_t0_window
    if win is None and pre_pulse_window is not None:
        try:
            win = pre_pulse_window(t_ns, counts.sum(axis=1))
        except Exception:
            win = None
    if win is None:
        return None, "none"
    w0, w1 = sorted((float(win[0]), float(win[1])))
    sel = (t_ns >= w0) & (t_ns <= w1)
    if not sel.any():
        return None, "none"
    return counts[sel].mean(axis=0).astype(np.float64), f"pre_t0 {w0:.2f}-{w1:.2f} ns"


def peak_counts(counts: np.ndarray, bin_ps: float, total_s: float, st: DynamicSettings,
                pre_pulse_window=None) -> dict:
    """Max over (bin, used pixel) of counts - background; with its pixel, time and the background used."""
    c = np.asarray(counts, dtype=np.float64)
    bg, how = background(c, bin_ps, total_s, st, pre_pulse_window)
    out = {"peak": 0.0, "pixel": -1, "t_ns": float("nan"), "background": how}
    if c.size == 0:
        return out
    sub = c - bg[None, :] if bg is not None else c.copy()
    ex = [p for p in st.excluded if 0 <= int(p) < c.shape[1]]
    if ex and len(ex) < c.shape[1]:
        sub[:, ex] = -np.inf
    k = int(np.argmax(sub))
    b, p = divmod(k, c.shape[1])
    out.update(peak=float(sub[b, p]), pixel=int(p), t_ns=float(b * float(bin_ps) / 1000.0))
    return out


def combine_snapshot(template, counts: np.ndarray, total_ms: int, n_events: int, out_of_window: int,
                     warnings: Iterable[str], elapsed_s: float, done: bool = True, cancelled: bool = False):
    """A snapshot (same class as ``template``) holding the summed counts and the total requested time."""
    seen: List[str] = []
    for w in warnings:
        if w not in seen:
            seen.append(w)
    kw = dict(measurement_ms=int(total_ms), counts_matrix=np.asarray(counts), warnings=tuple(seen),
              elapsed_s=float(elapsed_s), done=bool(done), n_events=int(n_events), cancelled=bool(cancelled))
    names = {f.name for f in dataclasses.fields(template)}
    if "out_of_window" in names:
        kw["out_of_window"] = int(out_of_window)
    return dataclasses.replace(template, **{k: v for k, v in kw.items() if k in names})


def write_trpl_csv(path: str, counts: np.ndarray, bin_ps: float) -> None:
    """The sweep's TRPL csv format: ``time_ns, pixel_0 .. pixel_N-1`` (integer counts)."""
    counts = np.asarray(counts)
    n_bins = int(counts.shape[0])
    time_ns = (np.arange(n_bins, dtype=np.float64) * float(bin_ps)) / 1000.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ns"] + [f"pixel_{i}" for i in range(counts.shape[1])])
        for i in range(n_bins):
            writer.writerow([f"{time_ns[i]:.6f}"] + counts[i, :].astype(np.int64).tolist())


@dataclass
class SpotResult:
    """What one dynamic acquisition did (also the ``dyn`` payload of trpl_done)."""
    chunk_ms: List[int] = field(default_factory=list)
    chunk_files: List[str] = field(default_factory=list)
    peak: float = 0.0
    pixel: int = -1
    t_ns: float = float("nan")
    background: str = ""
    reached: bool = False
    reason: str = ""          # "target" / "max_time" / "fixed_schedule" / "aborted"
    target: float = 0.0
    max_s: float = 0.0

    @property
    def total_s(self) -> float:
        return sum(self.chunk_ms) / 1000.0

    def as_dict(self) -> dict:
        return {"chunk_s": [m / 1000.0 for m in self.chunk_ms], "total_s": self.total_s,
                "n_chunks": len(self.chunk_ms), "chunk_files": list(self.chunk_files), "peak": self.peak,
                "pixel": self.pixel, "t_ns": self.t_ns, "background": self.background, "reached": self.reached,
                "reason": self.reason, "target": self.target, "max_s": self.max_s}


class DynamicLog:
    """Appends one row per spot to ``<folder>/dynamic_trpl_log.csv`` (header written once)."""

    def __init__(self, folder: Optional[str]):
        self.path = os.path.join(folder, LOG_NAME) if folder else None
        self._lock = threading.Lock()

    def append(self, file: Optional[str], res: SpotResult, *, step_idx=None, power_w=None, v_front=None,
               v_back=None, pump2=None) -> None:
        if not self.path:
            return
        row = [os.path.basename(file) if file else "", "" if step_idx is None else step_idx,
               _fmt(power_w), _fmt(v_front), _fmt(v_back),
               "" if pump2 is None else ("on" if pump2 else "off"), len(res.chunk_ms),
               ";".join(f"{m / 1000.0:g}" for m in res.chunk_ms), f"{res.total_s:g}", f"{res.peak:.6g}",
               res.pixel, _fmt(res.t_ns), f"{res.target:g}", f"{res.max_s:g}", res.background,
               "yes" if res.reached else "no", res.reason]
        with self._lock:
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(LOG_HEADER)
                w.writerow(row)


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if not np.isfinite(x) else f"{x:.9g}"


def read_log(folder: str) -> dict:
    """``{file name: row dict}`` from a sweep folder's log ({} if there is none)."""
    path = os.path.join(folder, LOG_NAME)
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("file"):
                out[row["file"]] = row
    return out
