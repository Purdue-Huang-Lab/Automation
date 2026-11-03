"""
dual_gate_asc_io.py  -- Qt-free loaders for dual-gate PL sweeps of Andor .asc frames
(PL_manual_Vf<>_Vb<>.asc): filename gate parsing, header parsing / timestamps, fast
frame read, and emission-spot (spatial row) detection.

Copied from the loader section of TMDC Gated\260216yunfanheterobilayer\
dual_gate_calibration.py (itself faithful to pl_power_gate_dep_viewer_v2.py) so this
folder is self-sufficient. Used by build_gtd1_cache.py.
"""
import re
from pathlib import Path
from datetime import datetime

import numpy as np

DEFAULT_ROW = 131                   # empirical emission row (raw); fallback only
BAND_HALF   = 6                     # +/- spatial rows summed into a linecut band
EDGE_GUARD  = 12                    # ignore emission humps within this many px of an edge
ROW_PROBE_N = 30                    # frames read in full to locate the emission row

# ----------------------------------------------------------------------------
# File parsing (faithful to pl_power_gate_dep_viewer_v2.py, Qt-free)
# ----------------------------------------------------------------------------
_VF_RE = re.compile(r"(?:^|_)Vf(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")
_VB_RE = re.compile(r"(?:^|_)Vb(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?=_|\.|$)")


def _tok(token):
    if not token:
        return None
    sign = 1.0
    if token.startswith("-"):
        sign, token = -1.0, token[1:]
    elif token.startswith("+"):
        token = token[1:]
    if token.startswith("m"):
        sign, token = sign * -1.0, token[1:]
    token = re.sub(r"em(\d+)", r"e-\1", token)
    try:
        val = sign * float(token.replace("p", "."))
    except Exception:
        return None
    return 0.0 if abs(val) < 1e-9 else val


def _vf_vb_from_name(name):
    mf, mb = _VF_RE.search(name), _VB_RE.search(name)
    return (_tok(mf.group("val")) if mf else None,
            _tok(mb.group("val")) if mb else None)


def _parse_header(path):
    meta = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if line and ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return meta


def _header_timestamp(path, meta):
    raw = meta.get("timestamp")
    if raw:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(raw, fmt).timestamp()
            except Exception:
                pass
    return float(path.stat().st_mtime)


def _has_wl_column(meta):
    """True if the frame was saved with a wavelength column ("Columns: wavelength_nm, ...").
    Without a valid wavelength axis the acquisition saves the RAW image instead:
    (n_spatial, n_wl) integers, no wavelength column."""
    return "wavelength" in meta.get("columns", "").lower()


def _read_frame(path, wl_axis=None):
    """Read the whole whitespace data block -> (n_wl, 1 + n_spatial).

    Column 0 = wavelength (nm); columns 1.. = intensity per spatial pixel.
    A full read via the pandas C engine is ~0.075 s/file; a column-subset read
    with a whitespace separator is ~10x slower (it drops to the python engine),
    so we always read full and slice the emission band in numpy.
    If wl_axis is given and the file is a raw image (no wavelength column), it is
    transposed and wl_axis is prepended, so callers always get the same layout.
    """
    try:
        import pandas as pd
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c")
        data = df.to_numpy(dtype=float)
    except Exception:
        data = np.loadtxt(path, comments="#")
    if wl_axis is not None and not _has_wl_column(_parse_header(path)):
        if data.ndim != 2 or data.shape[1] != wl_axis.size:
            raise ValueError(f"{path.name}: raw frame width {data.shape} != wavelength axis "
                             f"({wl_axis.size} px)")
        data = np.column_stack([wl_axis, data.T])
    return data


def _first_data_ncols(path):
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                return len(line.split())
    return 0


def find_reference_wavelengths(data_dir, ref_file=None):
    """Wavelength axis (nm) for a folder of RAW-image frames, which carry none.

    Borrowed from an .asc frame that HAS a wavelength column and was taken with the
    same grating, center wavelength and pixel count: ref_file if given, else the first
    match among the sibling folders of data_dir (then one level further up).
    Returns (wl, ref_path); raises LookupError if nothing matches."""
    data_dir = Path(data_dir)
    first = next(iter(sorted(data_dir.glob("*.asc"))), None)
    if first is None:
        raise FileNotFoundError(f"no .asc files in {data_dir}")
    meta = _parse_header(first)
    npix = _first_data_ncols(first)
    want = (meta.get("grating"), meta.get("center_wl_nm"))

    def _match(p):
        try:
            m = _parse_header(p)
        except Exception:
            return None
        if not _has_wl_column(m) or (m.get("grating"), m.get("center_wl_nm")) != want:
            return None
        wl = _read_frame(p)[:, 0]
        return wl if wl.size == npix and np.all(np.isfinite(wl)) else None

    if ref_file is not None:
        wl = _match(Path(ref_file))
        if wl is None:
            raise LookupError(f"{Path(ref_file).name} has no wavelength column matching grating "
                              f"{want[0]}, center {want[1]} nm, {npix} px")
        return wl, Path(ref_file)
    def _subdirs(d):
        try:
            return sorted(s for s in d.iterdir() if s.is_dir())
        except OSError:
            return []

    seen = {data_dir.resolve()}
    for depth, root in enumerate((data_dir.parent, data_dir.parent.parent)):
        dirs = [root] + _subdirs(root)
        if depth == 1:                                  # grandparent: also its sub-subfolders
            dirs += [s for d in dirs[1:] for s in _subdirs(d)]
        for d in dirs:
            if d.resolve() in seen:
                continue
            seen.add(d.resolve())
            p = next(iter(sorted(d.glob("*.asc"))), None)
            wl = _match(p) if p is not None else None
            if wl is not None:
                return wl, p
    raise LookupError(f"frames in {data_dir.name} have no wavelength column and no reference "
                      f".asc (grating {want[0]}, center {want[1]} nm, {npix} px) was found nearby")


# ----------------------------------------------------------------------------
# Emission-spot detection
# ----------------------------------------------------------------------------
def _find_humps(prof):
    """Emission humps well inside the detector (edge artifacts excluded).

    Returns the dominant hump center, plus a second only if it is genuinely
    strong (>=40% of the primary) and well separated (>=20 px).
    """
    p = np.asarray(prof, dtype=float)
    n = p.size
    interior = np.zeros(n, bool)
    interior[EDGE_GUARD:n - EDGE_GUARD] = True
    med = np.median(p[interior])
    mad = np.median(np.abs(p[interior] - med)) + 1e-9
    thresh = med + 5.0 * 1.4826 * mad
    strong = (p > thresh) & interior
    if not strong.any():
        return [int(EDGE_GUARD + np.argmax(p[interior]))]
    humps, i = [], 0
    while i < n:
        if strong[i]:
            j = i
            while j < n and strong[j]:
                j += 1
            seg = np.arange(i, j)
            c = int(round((seg * p[seg]).sum() / p[seg].sum()))
            humps.append((float(p[i:j].max()), c))
            i = j
        else:
            i += 1
    humps.sort(reverse=True)
    keep = [humps[0][1]]
    for pk, c in humps[1:]:
        if pk >= 0.40 * humps[0][0] and abs(c - humps[0][1]) >= 20:
            keep.append(c)
    return keep


def _detect_emission_bands(files, force_row, wl_axis=None):
    """Average PL-only spatial profile over a probe subset; return (profile, bands, wl).

    Each .asc line is: wavelength, intensity_spatial0..spatialN. We isolate PL from the
    flat dark floor by subtracting each spatial pixel's median-over-wavelength before
    summing, then average across probe frames. wl_axis: see _read_frame (raw frames).
    """
    n = len(files)
    idx = np.unique(np.linspace(0, n - 1, min(ROW_PROBE_N, n)).astype(int))
    frames = []
    for i in idx:
        try:
            data = _read_frame(files[i], wl_axis)
        except Exception:
            continue                               # unreadable (e.g. being written right now)
        if data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 1:
            frames.append(data)
    if not frames:
        raise ValueError("no readable .asc frames to probe")
    # an ongoing scan's newest frame may be half-written -> keep only the common frame shape
    shapes = [f.shape for f in frames]
    full = max(set(shapes), key=lambda s: (shapes.count(s), s[0]))
    frames = [f for f in frames if f.shape == full]
    wl = frames[0][:, 0].copy()
    prof = None
    for data in frames:
        img = data[:, 1:]
        pl = img - np.median(img, axis=0, keepdims=True)
        s = np.clip(pl, 0, None).sum(axis=0)
        prof = s if prof is None else prof + s
    prof = prof / len(frames)

    peaks = [int(force_row)] if force_row is not None else _find_humps(prof)
    if not peaks:
        peaks = [DEFAULT_ROW]
    bands = [(p, max(0, p - BAND_HALF), min(len(prof) - 1, p + BAND_HALF)) for p in peaks]
    return prof, bands, wl

