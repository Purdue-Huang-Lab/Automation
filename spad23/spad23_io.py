"""File I/O for SPAD23 data (no Qt).

Copied from ``spad93/spad93_io.py``.  TRPL CSV layout (the columns of the
old ``spad23_counts_gui`` export, plus metadata rows)::

    # device: SPAD23
    # mode: spectral
    # bin_width_ps: 50
    # ...                      one "# key: value" row per metadata entry
    # Row vs wavelength. ...   optional table (comment rows without ':')
    #  row     y_um   lambda_nm  pixels
    #    0     0.00     708.480  0 1 2 3 4
    time_ns,pixel_0,...,pixel_22
    0.000000,0,1,...

Values that are not plain numbers/strings are stored as JSON.  The reader
also accepts the older layouts: no header at all (old SPAD23 exports),
the header before ``mode_description``/the row table was added, Python-style
``True``/``None`` values, a UTF-8 BOM, CRLF line ends and a commented-out
column-name row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False

from spad23.spad23_analysis import DarkReference, WavelengthCalibration
from spad23.spad23_geometry import N_ROWS, NUM_PIXELS, PIXEL_ROW, PIXEL_X_UM, PIXEL_Y_UM, ROW_Y_UM, row_pixels

PIXEL_COLUMNS = [f"pixel_{i}" for i in range(NUM_PIXELS)]
MODE_2D = "2d"  # same strings as gui_v2.state
MODE_SPECTRAL = "spectral"
ROW_TABLE_KEY = "_row_wavelength_table"  # parsed row table; keys starting with "_" are never written

_TIME_SCALE = {"time_ns": 1.0, "time(ns)": 1.0, "t_ns": 1.0, "time": 1.0, "t": 1.0,
               "time_ps": 1e-3, "time(ps)": 1e-3, "t_ps": 1e-3}
_PIXEL_RE = re.compile(r"^(?:pixel|px|ch|channel)[_\-]?(\d+)$")
_ROW_RE = re.compile(r"^#\s*(\d+)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)(?:\s+[\d ]*)?$")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_path(path: str) -> str:
    """Append _1, _2, ... if ``path`` exists."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{base}_{k}{ext}"):
        k += 1
    return f"{base}_{k}{ext}"


def _meta_value(v) -> str:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return str(v)
    return json.dumps(v)


def _write_meta(fh, metadata: Optional[Dict]) -> None:
    for key, value in (metadata or {}).items():
        if str(key).startswith("_"):
            continue
        text = _meta_value(value).replace("\n", " ")
        fh.write(f"# {key}: {text}\n")


def _decode_value(text: str):
    """JSON first; Python-style True/False/None (written by str()) are decoded too."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    py = {"True": True, "False": False, "None": None, "nan": float("nan")}
    return py.get(text, text)


def _norm_name(name) -> str:
    return re.sub(r"\s+", "", str(name).strip().lstrip("\ufeff")).lower()


def _scan_header(path: str) -> Tuple[Dict[str, object], Optional[List[str]]]:
    """Leading comment rows: ``# key: value`` metadata, the row table and a commented column-name row."""
    meta: Dict[str, object] = {}
    rows: List[Tuple[int, float, float]] = []
    names: Optional[List[str]] = None
    in_table = False
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if body.lower().startswith("row vs wavelength"):
                in_table = True
                continue
            if in_table:
                m = _ROW_RE.match(line)
                if m:
                    rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                    continue
                if body.lower().startswith("row"):
                    continue  # the table's column titles
                in_table = False
            if "," in body and ("pixel_0" in body or "time_ns" in body) and ":" not in body:
                names = [c.strip() for c in body.split(",")]
                continue
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            meta[key.strip()] = _decode_value(value.strip())
    if rows:
        meta[ROW_TABLE_KEY] = rows
    return meta, names


def read_metadata(path: str) -> Dict[str, object]:
    """Parse the leading ``# key: value`` rows (JSON or Python-style values are decoded).

    A row -> wavelength table, if present, is returned under ``ROW_TABLE_KEY``
    as ``[(row, y_um, lambda_nm), ...]``.
    """
    return _scan_header(path)[0]


def mode_from_metadata(meta: Dict[str, object]) -> Optional[str]:
    """``"2d"`` / ``"spectral"`` from any header version (None if not recorded)."""
    for key in ("mode", "mode_description"):
        v = str(meta.get(key, "") or "").strip().lower()
        if v.startswith("spec"):
            return MODE_SPECTRAL
        if v.startswith("2d") or v in ("space", "spatial"):
            return MODE_2D
    return None


@dataclass
class TrplData:
    time_ns: np.ndarray
    counts: np.ndarray  # [bins, 23]
    bin_width_ps: float
    metadata: Dict[str, object] = field(default_factory=dict)
    source: str = ""

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[0])


def trim_trailing_empty(counts: np.ndarray) -> np.ndarray:
    if counts.shape[0] == 0:
        return counts
    nz = np.flatnonzero(counts.sum(axis=1))
    return counts[: int(nz[-1]) + 1] if nz.size else counts[:1]


def _write_row_table(fh, calib_dict) -> None:
    """Readable row -> wavelength table as comment lines (no ':' so they are not parsed as metadata)."""
    try:
        calib = WavelengthCalibration.from_dict(calib_dict)
    except (KeyError, TypeError, ValueError):
        return
    lam = calib.wavelength(ROW_Y_UM)
    fh.write("# Row vs wavelength. Row r = map row r (pixels listed) at y = 19.92*r um; "
             "the wavelength is read along the center column x = 46 um\n")
    fh.write(f"# {'row':>4} {'y_um':>8} {'lambda_nm':>11}  pixels\n")
    for r in range(N_ROWS):
        pixels = " ".join(str(int(p)) for p in row_pixels(r))
        fh.write(f"# {r:>4d} {ROW_Y_UM[r]:>8.2f} {float(lam[r]):>11.3f}  {pixels}\n")


def save_trpl_csv(path: str, time_ns: np.ndarray, counts: np.ndarray, metadata: Optional[Dict] = None) -> str:
    """Columns time_ns, pixel_0..pixel_22 (raw counts per bin) after ``# key: value`` header rows.

    If the metadata holds a wavelength calibration, a row -> wavelength
    table is added to the header as plain comment lines.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != NUM_PIXELS:
        raise ValueError(f"counts must be [bins, {NUM_PIXELS}], got {counts.shape}")
    n = counts.shape[0]
    time_ns = np.asarray(time_ns, dtype=np.float64)[:n]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        _write_meta(fh, metadata)
        calib = (metadata or {}).get("wavelength_calibration")
        if isinstance(calib, dict):
            _write_row_table(fh, calib)
        if _HAS_PANDAS:
            df = pd.DataFrame(counts.astype(np.int64), columns=PIXEL_COLUMNS)
            df.insert(0, "time_ns", np.round(time_ns, 6))
            df.to_csv(fh, index=False, float_format="%.6f", lineterminator="\n")
        else:
            fh.write("time_ns," + ",".join(PIXEL_COLUMNS) + "\n")
            out = np.column_stack([time_ns, counts.astype(np.float64)])
            np.savetxt(fh, out, fmt=["%.6f"] + ["%d"] * NUM_PIXELS, delimiter=",")
    return path


def _match_columns(columns: Sequence) -> Tuple[Optional[str], float, Dict[int, str]]:
    """(time column, factor to ns, {pixel index: column}) from the column names of any version."""
    time_col, scale, pix = None, 1.0, {}
    for c in columns:
        n = _norm_name(c)
        if time_col is None and n in _TIME_SCALE:
            time_col, scale = c, _TIME_SCALE[n]
            continue
        m = _PIXEL_RE.match(n)
        if m:
            pix.setdefault(int(m.group(1)), c)
    return time_col, scale, pix


def load_trpl_csv(path: str) -> TrplData:
    """Load a SPAD23 TRPL CSV written by any version of this code (see the module docstring)."""
    meta, commented_names = _scan_header(path)
    if _HAS_PANDAS:
        df = pd.read_csv(path, comment="#", encoding="utf-8-sig", skipinitialspace=True,
                         header=None if commented_names else "infer", names=commented_names)
        df.columns = [str(c).strip() for c in df.columns]
        columns = list(df.columns)
    else:
        with open(path, "r", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.reader(line for line in fh if not line.startswith("#"))]
        if commented_names:
            rows.insert(0, commented_names)
        columns = [c.strip() for c in rows[0]]
    time_col, scale, pix = _match_columns(columns)
    if time_col is None:
        raise ValueError(f"Missing a time column (time_ns); first columns are {columns[:3]}.")
    if len(pix) != NUM_PIXELS or sorted(pix) != list(range(NUM_PIXELS)):
        hint = " (this looks like a SPAD23 file)" if len(pix) == 23 else ""
        raise ValueError(f"Expected pixel columns 0-{NUM_PIXELS - 1}, found {len(pix)}{hint}.")
    pcols = [pix[i] for i in range(NUM_PIXELS)]
    if _HAS_PANDAS:
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df[df[time_col].notna()]
        time_ns = df[time_col].to_numpy(dtype=np.float64) * scale
        counts = df[pcols].fillna(0).to_numpy(dtype=np.float64).round().astype(np.int64)
    else:
        idx_t = columns.index(time_col)
        idx_p = [columns.index(c) for c in pcols]
        data = [r for r in rows[1:] if len(r) > max(idx_p + [idx_t]) and r[idx_t].strip()]
        time_ns = np.array([float(r[idx_t]) for r in data], dtype=np.float64) * scale
        counts = np.array([[int(round(float(r[j] or 0))) for j in idx_p] for r in data], dtype=np.int64)
    if time_ns.size == 0:
        raise ValueError("File has no data rows.")
    bin_ps = meta.get("bin_width_ps")
    if not isinstance(bin_ps, (int, float)) or isinstance(bin_ps, bool) or not bin_ps > 0:
        bin_ps = float(np.median(np.diff(time_ns)) * 1000.0) if time_ns.size > 1 else 1.0
    mode = mode_from_metadata(meta)
    if mode is not None:
        meta["mode"] = mode
    return TrplData(time_ns=time_ns, counts=counts, bin_width_ps=float(bin_ps), metadata=meta, source=path)


def save_dark_csv(path: str, time_ns: np.ndarray, counts: np.ndarray, ref: DarkReference,
                  metadata: Optional[Dict] = None) -> str:
    """The raw dark histogram (beam blocked) in the TRPL CSV layout, flagged as a dark reference."""
    meta = dict(metadata or {})
    meta.update({
        "dark_reference": True,
        "content": "dark reference: TRPL histogram with the excitation beam blocked",
        "detector": ref.detector,
        "measurement_ms": ref.measurement_ms,
        "bin_width_ps": ref.bin_width_ps,
        "taken_at": ref.taken_at,
    })
    return save_trpl_csv(path, time_ns, trim_trailing_empty(np.asarray(counts)), meta)


def load_dark_csv(path: str) -> DarkReference:
    """A dark reference saved by ``save_dark_csv`` (raises ValueError for other files)."""
    data = load_trpl_csv(path)
    meta = data.metadata
    name = os.path.basename(path)
    if meta.get("dark_reference") is not True:
        raise ValueError(f"{name} is not a dark reference (no 'dark_reference: True' header row).")
    ms = meta.get("measurement_ms")
    if not isinstance(ms, (int, float)) or not ms > 0:
        raise ValueError(f"{name}: no acquisition time (measurement_ms) in the header.")
    return DarkReference.from_counts(data.counts, data.bin_width_ps, float(ms), detector=str(meta.get("detector", "")),
                                     taken_at=str(meta.get("taken_at", "")), source=path)


def calibration_from_metadata(meta: Dict[str, object]) -> Optional[WavelengthCalibration]:
    """The JSON calibration of any header version, else one rebuilt from the row table."""
    if meta.get("wavelength_calibration_applied") is False:
        return None
    raw = meta.get("wavelength_calibration")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, dict):
        try:
            return WavelengthCalibration.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            pass
    table = meta.get(ROW_TABLE_KEY)
    if isinstance(table, list) and len(table) >= 2:
        x = np.array([r[1] for r in table], dtype=np.float64)
        lam = np.array([r[2] for r in table], dtype=np.float64)
        if np.ptp(x) > 0 and np.ptp(lam) > 0:
            slope, intercept = np.polyfit(x, lam, 1)
            return WavelengthCalibration(
                slope_nm_per_um=float(slope), intercept_nm=float(intercept),
                residuals_nm=tuple(float(v) for v in lam - (slope * x + intercept)),
                created=str(meta.get("saved_at", "")),
                warnings=("Rebuilt from the row vs wavelength table in the file header.",),
            )
    return None


def save_calibration_json(path: str, calib: WavelengthCalibration) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calib.to_dict(), fh, indent=2)
    return path


def load_calibration_json(path: str) -> WavelengthCalibration:
    with open(path, "r", encoding="utf-8") as fh:
        return WavelengthCalibration.from_dict(json.load(fh))


def save_counts_snapshot(path: str, counts: np.ndarray, rates_cps: np.ndarray, metadata: Optional[Dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        _write_meta(fh, metadata)
        fh.write("pixel,row,x_um,y_um,counts,rate_cps\n")
        for p in range(NUM_PIXELS):
            fh.write(
                f"{p},{int(PIXEL_ROW[p])},{PIXEL_X_UM[p]:.3f},{PIXEL_Y_UM[p]:.3f},"
                f"{int(counts[p])},{float(rates_cps[p]):.3f}\n"
            )
    return path


def save_spectrum_csv(
    path: str,
    intensity: np.ndarray,
    wavelength_nm: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        _write_meta(fh, metadata)
        fh.write("row,y_um,lambda_nm,intensity\n")
        for c in range(ROW_Y_UM.size):
            lam = "" if wavelength_nm is None else f"{float(wavelength_nm[c]):.4f}"
            val = "" if not np.isfinite(intensity[c]) else f"{float(intensity[c]):.6g}"
            fh.write(f"{c},{ROW_Y_UM[c]:.3f},{lam},{val}\n")
    return path


def _num(v: float) -> str:
    """Plain number for Origin; NaN/inf become an empty cell (missing value)."""
    return f"{v:.10g}" if np.isfinite(v) else ""


def save_origin_matrix_csv(path: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> str:
    """Numbers only: first row = x (corner cell empty), first column = y, then z[len(y), len(x)].

    Imports into Origin as a virtual matrix / XYZ grid; there are no labels or units.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (y.size, x.size):
        raise ValueError(f"matrix shape {z.shape} does not match ({y.size}, {x.size})")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("," + ",".join(_num(v) for v in x) + "\n")
        for yi, row in zip(y, z):
            fh.write(_num(yi) + "," + ",".join(_num(v) for v in row) + "\n")
    return path


def columns_to_text(columns: Sequence[np.ndarray], sep: str = "\t") -> str:
    """Tab-separated numbers, one array per column (shorter columns padded with empty cells)."""
    cols = [np.asarray(c, dtype=np.float64).ravel() for c in columns]
    n = max((c.size for c in cols), default=0)
    lines = []
    for i in range(n):
        lines.append(sep.join(_num(c[i]) if i < c.size else "" for c in cols))
    return "\n".join(lines) + ("\n" if lines else "")


def save_spectral_matrix_csv(
    path: str,
    time_ns: np.ndarray,
    matrix: np.ndarray,
    wavelength_nm: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Time-resolved spectrum: rows = time bins, columns = the 5 center-column points (one per row)."""
    if wavelength_nm is not None:
        labels = [f"{float(v):.4f}nm" for v in wavelength_nm]
    else:
        labels = [f"y{float(v):.2f}um" for v in ROW_Y_UM]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        _write_meta(fh, metadata)
        fh.write("time_ns," + ",".join(labels) + "\n")
        out = np.column_stack([np.asarray(time_ns, dtype=np.float64), np.asarray(matrix, dtype=np.float64)])
        np.savetxt(fh, out, fmt="%.6g", delimiter=",")
    return path


def save_columns_csv(path: str, columns: Dict[str, Sequence[float]], metadata: Optional[Dict] = None) -> str:
    """Write equally long named columns (fit traces)."""
    names = list(columns.keys())
    arrays = [np.asarray(columns[n], dtype=np.float64) for n in names]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        _write_meta(fh, metadata)
        fh.write(",".join(names) + "\n")
        if arrays:
            np.savetxt(fh, np.column_stack(arrays), fmt="%.8g", delimiter=",")
    return path
