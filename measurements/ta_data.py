"""
measurements/ta_data.py

Loading of transient-absorption (pump-probe) datasets, shared by
ta_gate_power_viewer.py and ta_viewer.py.  No Qt (pyqtgraph is imported only
for the colour maps).

A dataset is a ``{prefix}para.txt`` + ``{prefix}sum.txt`` pair, as written by the
LabVIEW code, pump_probe_gate_dep/ and pump_probe_gate_dep_montana/ (ta_core.py):

    {base}_Vf<p|n>1d250VVb<p|n>2d250V_ND30p0deg_P375nW_LH_{timestamp}para.txt

Gate voltages, ND-wheel angle, power and helicity are optional tokens of the
name.  The ``_P<value><unit>`` power tag is written only by the Montana GUI.
"""
import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Colour maps
# ─────────────────────────────────────────────────────────────────────────────

_LUTS: Dict[str, np.ndarray] = {}


def get_lut(name: str) -> Optional[np.ndarray]:
    """'RdBu_r' (red = positive, the TA convention), 'viridis', or None for gray."""
    if name not in ("RdBu_r", "viridis"):
        return None
    if name not in _LUTS:
        import pyqtgraph as pg
        if name == "RdBu_r":
            pts = np.linspace(0.0, 1.0, 11)
            rgba = np.array([
                [103,   0,  31, 255],
                [178,  24,  43, 255],
                [214,  96,  77, 255],
                [244, 165, 130, 255],
                [253, 219, 199, 255],
                [247, 247, 247, 255],
                [209, 229, 240, 255],
                [146, 197, 222, 255],
                [ 67, 147, 195, 255],
                [ 33, 102, 172, 255],
                [  5,  48,  97, 255],
            ], dtype=np.ubyte)
            _LUTS[name] = pg.ColorMap(pts, rgba).getLookupTable(nPts=256)
        else:
            _LUTS[name] = pg.colormap.get("viridis").getLookupTable(nPts=256)
    return _LUTS[name]


# ─────────────────────────────────────────────────────────────────────────────
#  File-name parsing
# ─────────────────────────────────────────────────────────────────────────────

_VF_RE  = re.compile(r"Vf([pn])(\d+)d(\d+)V", re.I)
_VB_RE  = re.compile(r"Vb([pn])(\d+)d(\d+)V", re.I)
_ND_RE  = re.compile(r"ND(n?)(\d+)(?:p(\d+))?deg", re.I)
_HEL_RE = re.compile(r"(?<![A-Za-z])(LH|RH)(?![A-Za-z])", re.I)
# ta_core.power_tag: f"{p:.4g}" with '.' -> 'p' and '-' -> 'm'  (12.5 -> 12p5, 1e-05 -> 1em05)
_PWR_RE = re.compile(r"_P(m?\d+(?:p\d+)?(?:e[+m]?\d+)?)(pW|nW|uW|mW|W)(?=_|$)")

POWER_TO_UW = {"pW": 1e-6, "nW": 1e-3, "uW": 1.0, "mW": 1e3, "W": 1e6}


def _volt(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    return sign * float(f"{m.group(2)}.{m.group(3)}")


def _ang(m: re.Match) -> float:
    sign = -1.0 if m.group(1).lower() == "n" else 1.0
    dec = m.group(3) or "0"
    return sign * float(f"{m.group(2)}.{dec}")


def parse_fname(stem: str) -> Tuple[float, float, float, str]:
    """(v_front, v_back, angle, helicity) from a file name; NaN / '' if absent."""
    vf = vb = angle = np.nan
    helicity = ""
    m = _VF_RE.search(stem)
    if m:
        vf = _volt(m)
    m = _VB_RE.search(stem)
    if m:
        vb = _volt(m)
    m = _ND_RE.search(stem)
    if m:
        angle = _ang(m)
    m = _HEL_RE.search(stem)
    if m:
        helicity = m.group(1).upper()
    return vf, vb, angle, helicity


def parse_power_tag(stem: str) -> float:
    """Power in µW from the Montana GUI's ``_P<value><unit>`` tag; NaN if absent."""
    m = _PWR_RE.search(stem)
    if not m:
        return np.nan
    s = m.group(1).replace("em", "e-")
    if s.startswith("m"):
        s = "-" + s[1:]
    try:
        return float(s.replace("p", ".")) * POWER_TO_UW[m.group(2)]
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  para.txt / sum.txt
# ─────────────────────────────────────────────────────────────────────────────

def parse_para(path: Path) -> dict:
    """
    Parse a *para.txt (LabVIEW layout, written by ta_core.write_para).

    Line layout (0-indexed, comment lines excluded):
      0  : tab-separated delays in ps
      1  : averages per delay
      2  : 0
      3  : scale factor (1.0)
      4  : pixel_size (µm/pixel)
      5  : crop_w  – spectral pixels  (≈ 480)  ← N2 in TAM notation
      6  : pixel_size (repeated)
      7  : crop_h  – spatial pixels   (≈ 128)  ← N1 in TAM notation
      8  : 0
      9  : t0_ps
      10-14: 0s
      15 : 0
      16 : angle (degrees)
      17 : 1
      18 : 0
      19 : n_frames
      20 : 1
    Comment lines: # FrontGateV=…  # BackGateV=…  # Helicity=LH/RH
    """
    data_lines: List[str] = []
    comments: dict = {}
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("#"):
                if "=" in s:
                    k, v = s[1:].split("=", 1)
                    comments[k.strip()] = v.strip()
            else:
                data_lines.append(s)

    def _f(idx: int, default: float = 0.0) -> float:
        try:
            return float(data_lines[idx])
        except Exception:
            return default

    def _i(idx: int, default: int = 128) -> int:
        try:
            return int(float(data_lines[idx]))
        except Exception:
            return default

    def _row(idx: int) -> np.ndarray:
        try:
            return np.array([float(x) for x in data_lines[idx].split("\t") if x.strip()],
                            dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)

    hel = (comments.get("Helicity", "") or comments.get("helicity", "")).strip().upper()

    return dict(
        delays_ps  = _row(0),
        avg        = _row(1),
        pixel_size = _f(4, 0.07),   # µm/pixel
        crop_w     = _i(5, 480),    # spectral width  (N2)
        crop_h     = _i(7, 128),    # spatial height  (N1)
        t0_ps      = _f(9, 0.0),
        angle      = _f(16, np.nan),
        n_frames   = _i(19, 0),
        v_front    = float(comments["FrontGateV"]) if "FrontGateV" in comments else np.nan,
        v_back     = float(comments["BackGateV"])  if "BackGateV"  in comments else np.nan,
        helicity   = hel,
    )


def _read_sum_rows(path: Path, skiprows: int = 0, nrows: Optional[int] = None) -> np.ndarray:
    import pandas as pd
    return (
        # skipinitialspace: LabVIEW writes "   Inf", which pandas would otherwise keep as text
        pd.read_csv(path, delimiter="\t", header=None, skiprows=skiprows, nrows=nrows,
                    na_values=["--", "---"], skipinitialspace=True)
        .fillna(0.0)
        .values.astype(np.float32)
    )


def read_sum_stack(path: Path, n1: int, n2: int, nt: int) -> Tuple[np.ndarray, int]:
    """
    (stack, n_read): the (nt, n1, n2) = (delays, crop_h, crop_w) stack of a *sum.txt.

    sum.txt layout: for each delay, crop_w rows × crop_h columns (the transposed
    log-ratio image).  Read back: raw[t*N2 : (t+1)*N2, :N1] → rot90 → (N1, N2).
    Delays missing at the end of the file (stopped scan) stay zero; ``n_read``
    is the number of complete delays found.
    """
    raw = _read_sum_rows(path)
    stack = np.zeros((nt, n1, n2), dtype=np.float32)
    n_read = 0
    for i in range(nt):
        s = i * n2
        if s + n2 > raw.shape[0]:
            break
        stack[i] = np.rot90(raw[s : s + n2, :n1])
        n_read += 1
    return stack, n_read


def read_sum_frame(path: Path, t_idx: int, n1: int, n2: int) -> Optional[np.ndarray]:
    """The (n1, n2) frame of delay ``t_idx`` only (fast for large files); None if missing."""
    chunk = _read_sum_rows(path, skiprows=t_idx * n2, nrows=n2)
    if chunk.shape[0] < n2:
        return None
    return np.rot90(chunk[:n2, :n1])


# ─────────────────────────────────────────────────────────────────────────────
#  ScanRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanRecord:
    path_para:  Path
    path_sum:   Path
    delays_ps:  np.ndarray
    crop_h:     int        # spatial pixels  (N1, ≈128)
    crop_w:     int        # spectral pixels (N2, ≈480)
    t0_ps:      float
    pixel_size: float      # µm/pixel (spatial)
    v_front:    float      # NaN if unknown
    v_back:     float      # NaN if unknown
    angle:      float      # rotation stage degrees; NaN if absent
    power_uw:   float      # µW: power map if it covers the angle, else file-name tag; NaN if unknown
    helicity:   str        # "LH", "RH", or ""
    power_tag_uw: float = np.nan   # µW from the file name's _P tag (Montana GUI)
    n_frames:   int = 0
    avg:        np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _stack: Optional[np.ndarray] = field(default=None, repr=False)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """The common prefix of the para / sum files."""
        n = self.path_para.name
        return n[: -len("para.txt")] if n.endswith("para.txt") else self.path_para.stem

    @property
    def n_delays(self) -> int:
        return len(self.delays_ps)

    @property
    def label(self) -> str:
        parts = []
        if self.helicity:
            parts.append(self.helicity)
        if np.isfinite(self.v_front):
            parts.append(f"Vf={self.v_front:+.3f} V")
        if np.isfinite(self.v_back):
            parts.append(f"Vb={self.v_back:+.3f} V")
        if np.isfinite(self.power_uw):
            parts.append(f"{self.power_uw:.3g} µW")
        elif np.isfinite(self.angle):
            parts.append(f"ND {self.angle:.1f}°")
        return "  ".join(parts) if parts else self.path_para.stem[:50]

    def t_index(self, t_ps: float, n: Optional[int] = None) -> int:
        """Index of the delay nearest ``t_ps`` among the first ``n`` delays."""
        d = self.delays_ps[: (self.n_delays if n is None else n)]
        return int(np.argmin(np.abs(d - t_ps))) if len(d) else 0

    # ── stack loading ─────────────────────────────────────────────────────────

    def load_stack(self) -> Optional[np.ndarray]:
        """Full (T, crop_h, crop_w) stack (cached); missing delays are zero."""
        if self._stack is not None:
            return self._stack
        try:
            self._stack, _ = read_sum_stack(self.path_sum, self.crop_h, self.crop_w, self.n_delays)
            return self._stack
        except Exception as exc:
            print(f"load_stack {self.path_sum.name}: {exc}")
            return None

    def value_at(self, t_idx: int, iy: int, ix: int) -> float:
        """Signal at one pixel, loading only the required frame if the stack is not cached."""
        if self._stack is not None:
            stk = self._stack
            t = int(np.clip(t_idx, 0, stk.shape[0] - 1))
            y = int(np.clip(iy,    0, stk.shape[1] - 1))
            x = int(np.clip(ix,    0, stk.shape[2] - 1))
            return float(stk[t, y, x])
        N1, N2 = self.crop_h, self.crop_w
        t = int(np.clip(t_idx, 0, self.n_delays - 1))
        y = int(np.clip(iy, 0, N1 - 1))
        x = int(np.clip(ix, 0, N2 - 1))
        try:
            frame = read_sum_frame(self.path_sum, t, N1, N2)
            return float(frame[y, x]) if frame is not None else np.nan
        except Exception:
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Power maps (ND-wheel angle → µW)
# ─────────────────────────────────────────────────────────────────────────────

def power_from_map(angle: float, wheel_map: dict, tol_deg: float = 0.5) -> float:
    """µW of the map entry nearest ``angle`` (within ``tol_deg``); NaN otherwise."""
    if not wheel_map or not np.isfinite(angle):
        return np.nan
    best = min(wheel_map, key=lambda k: abs(k - angle))
    return wheel_map[best] if abs(best - angle) <= tol_deg else np.nan


def apply_power_map(records: List[ScanRecord], wheel_map: dict) -> None:
    """Set each record's power_uw: the map if it covers the angle, else the file-name tag."""
    for r in records:
        p = power_from_map(r.angle, wheel_map)
        r.power_uw = p if np.isfinite(p) else r.power_tag_uw


_UNIT_RE = re.compile(r"(?<![a-z])(pw|nw|uw|µw|μw|mw|w)(?![a-z])")


def _power_header_to_uw(header: str) -> float:
    """µW per unit of a power column header, e.g. 'power (nW)' → 1e-3; no unit → µW."""
    m = _UNIT_RE.search(header.lower())
    if not m:
        return 1.0
    u = m.group(1).replace("µ", "u").replace("μ", "u")
    return {"pw": 1e-6, "nw": 1e-3, "uw": 1.0, "mw": 1e3, "w": 1e6}[u]


def _wheel_map_from_rows(rows: List[List[str]]) -> dict:
    """{angle: power_µW} from table rows: the first row with an 'angle'/'position' cell is the header."""
    hdr_idx = None
    for i, row in enumerate(rows):
        if any(("angle" in str(v).lower() or "position" in str(v).lower()) for v in row):
            hdr_idx = i
            break
    if hdr_idx is None:
        return {}
    header = [str(v).strip() for v in rows[hdr_idx]]
    low = [h.lower() for h in header]
    a_col = next((i for i, h in enumerate(low) if "angle" in h or "position" in h), None)
    p_col = next((i for i, h in enumerate(low) if "power" in h), None)
    if a_col is None or p_col is None:
        return {}
    to_uw = _power_header_to_uw(header[p_col])
    # power_calibration lists may hold several ND-filter series in the 3rd column
    # ("base", "ND#1", …); use "base" only.  Numeric columns (e.g. std) are not series.
    s_col = next((i for i, h in enumerate(low) if "series" in h), None)
    if s_col is None and p_col == 1:
        s_col = 2
    result: dict = {}
    for row in rows[hdr_idx + 1 :]:
        try:
            a = float(row[a_col])
            p = float(row[p_col])
        except (ValueError, TypeError, IndexError):
            continue
        if s_col is not None and s_col < len(row):
            series = str(row[s_col]).strip()
            try:
                float(series)
            except ValueError:
                if series.lower() not in ("", "base", "nan", "none"):
                    continue
        if np.isfinite(a) and np.isfinite(p):
            result[a] = p * to_uw
    return result


def load_wheel_map(path: str) -> dict:
    """
    Read angle → power (µW) from an .xlsx/.xls or .csv file.

    The header row is the first row with a cell containing 'angle' (or
    'position'); the power column is the one containing 'power'.  Its unit is
    taken from the header ('power (nW)', 'power_uW', …); without a unit µW is assumed.
    """
    try:
        if os.path.splitext(path)[1].lower() == ".csv":
            with open(path, "r", newline="", errors="ignore") as fh:
                rows = [row for row in csv.reader(fh) if row]
        else:
            import pandas as pd
            df = pd.read_excel(path, header=None)
            rows = [["" if (isinstance(v, float) and np.isnan(v)) else v for v in r]
                    for r in df.values.tolist()]
        return _wheel_map_from_rows(rows)
    except Exception as exc:
        print(f"Power map load failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Folder helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_datasets(folder: str) -> List[Tuple[Path, Path]]:
    """All (para, sum) pairs in ``folder`` (not recursive), sorted by name."""
    pairs = []
    for para_path in sorted(Path(folder).glob("*para.txt")):
        sum_path = para_path.with_name(para_path.name[: -len("para.txt")] + "sum.txt")
        if sum_path.exists():
            pairs.append((para_path, sum_path))
    return pairs


def record_from_para(para_path: Path, sum_path: Optional[Path] = None,
                     wheel_map: Optional[dict] = None) -> ScanRecord:
    """Read one *para.txt; metadata from its comment lines, else from the file name."""
    if sum_path is None:
        sum_path = para_path.with_name(para_path.name[: -len("para.txt")] + "sum.txt")
    meta = parse_para(para_path)
    fn_vf, fn_vb, fn_ang, fn_hel = parse_fname(para_path.stem)
    vf  = meta["v_front"]  if np.isfinite(meta["v_front"]) else fn_vf
    vb  = meta["v_back"]   if np.isfinite(meta["v_back"])  else fn_vb
    ang = meta["angle"]    if np.isfinite(meta["angle"])   else fn_ang
    hel = meta["helicity"] or fn_hel
    tag = parse_power_tag(para_path.stem)
    p_map = power_from_map(ang, wheel_map or {})
    return ScanRecord(
        path_para    = para_path,
        path_sum     = sum_path,
        delays_ps    = meta["delays_ps"],
        crop_h       = meta["crop_h"],
        crop_w       = meta["crop_w"],
        t0_ps        = meta["t0_ps"],
        pixel_size   = meta["pixel_size"],
        v_front      = vf,
        v_back       = vb,
        angle        = ang,
        power_uw     = p_map if np.isfinite(p_map) else tag,
        helicity     = hel,
        power_tag_uw = tag,
        n_frames     = meta["n_frames"],
        avg          = meta["avg"],
    )


def scan_folder(folder: str, wheel_map: dict) -> List[ScanRecord]:
    """Discover all *para.txt + *sum.txt pairs and read their para files."""
    records = []
    for para_path, sum_path in find_datasets(folder):
        try:
            records.append(record_from_para(para_path, sum_path, wheel_map))
        except Exception as exc:
            print(f"Skip {para_path.name}: {exc}")
    return records
