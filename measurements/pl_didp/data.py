"""Loading for the dI/dP viewer (no Qt).

The PL + TRPL v2 GUI's dI/dP mode writes pump 2 off and pump 2 on of every
sweep point into one folder:

    <base>[_Vf..|_Vb..]_P2off.asc        <base>[...]_P2on.asc
    TRPL_<base>[...]_P2off.csv           TRPL_<base>[...]_P2on.csv

with an optional ``_1``, ``_2``, ... after the tag for retakes.  The two files
of one point (and its PL and TRPL files) share a *stem*: the name without the
``TRPL_`` prefix, the tag, the retake counter and the extension.

TRPL points are kept as stub ``PLFrame`` objects (1×1 image, the ``TrplData``
in ``meta["_trpl"]``) so they go through the same ``_build_channel`` key
normalisation as the PL frames and line up with the same sliders.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from measurements.pl_power_gate_dep_viewer_v2 import (
    PLFrame, _load_frames, _parse_angle_from_name, _parse_from_name, _tok,
)
from spad23.spad23_io import TrplData, load_trpl_csv

from .config import PUMP2_OFF_TAG, PUMP2_ON_TAG

_TAG_RE = re.compile(rf"_(?P<tag>{PUMP2_OFF_TAG}|{PUMP2_ON_TAG})(?:_\d+)?$")
_RETAKE_RE = re.compile(r"_\d+$")
# Single-calibration names carry the power without the "P" prefix: PL_<series>_<pw><unit>
_BARE_POWER_RE = re.compile(
    r"(?:^|_)(?P<val>[+-]?(?:m)?[0-9]+(?:p[0-9]+)?(?:em?[0-9]+)?)(?P<unit>nW|uW|mW|W)(?=_|\.|$)")
_UNIT_SCALE = {"w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9}

TRPL_KEY = "_trpl"


def pump2_of(name: str) -> Optional[bool]:
    """True / False for a ``_P2on`` / ``_P2off`` file, None if untagged."""
    m = _TAG_RE.search(Path(name).stem)
    if m is None:
        return None
    return m.group("tag") == PUMP2_ON_TAG


def pair_stem(name: str) -> str:
    """Name shared by the off / on (and PL / TRPL) files of one sweep point."""
    stem = Path(name).stem
    if stem.startswith("TRPL_"):
        stem = stem[len("TRPL_"):]
    m = _TAG_RE.search(stem)
    if m is not None:
        return stem[:m.start()]
    return _RETAKE_RE.sub("", stem)


def load_didp_folder(folder: Path) -> Tuple[List[PLFrame], List[PLFrame], int]:
    """PL frames of a dI/dP folder split by tag: (pump 2 off, pump 2 on, n untagged ignored)."""
    off, on, n_untagged = [], [], 0
    for fr in _load_frames(Path(folder)):
        tag = pump2_of(fr.path.name)
        if tag is None:
            n_untagged += 1
        elif tag:
            on.append(fr)
        else:
            off.append(fr)
    return off, on, n_untagged


def by_stem(frames: List[PLFrame]) -> Dict[str, PLFrame]:
    """Stem → frame (the newest file wins, as in ``_build_channel``)."""
    out: Dict[str, PLFrame] = {}
    for fr in frames:
        key = pair_stem(fr.path.name)
        old = out.get(key)
        if old is None or fr.mtime > old.mtime:
            out[key] = fr
    return out


def _power_from_bare_name(name: str) -> Optional[float]:
    m = _BARE_POWER_RE.search(name)
    if not m:
        return None
    val = _tok(m.group("val"))
    scale = _UNIT_SCALE.get(m.group("unit").lower())
    return None if val is None or scale is None else float(val) * scale


def load_trpl_points(folder: Path, pump2: Optional[bool],
                     pl_by_stem: Optional[Dict[str, PLFrame]] = None) -> Tuple[List[PLFrame], List[str]]:
    """``TRPL_*.csv`` of one pump-2 state as stub frames; returns (frames, load errors).

    pump2      : False / True for tagged off / on files, None for untagged (legacy) files
    pl_by_stem : PL frames of the same folder; a TRPL point takes its power and gate
                 keys from the PL file of the same stem so both line up exactly.
    """
    pl_by_stem = pl_by_stem or {}
    frames: List[PLFrame] = []
    errors: List[str] = []
    for path in sorted(Path(folder).glob("TRPL_*.csv")):
        if pump2_of(path.name) != pump2:
            continue
        try:
            data: TrplData = load_trpl_csv(str(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        pl = pl_by_stem.get(pair_stem(path.name))
        if pl is not None:
            power, vf, vb, angle = pl.power_w, pl.v_front, pl.v_back, pl.angle_deg
            vf_named, vb_named = pl.vf_named, pl.vb_named
        else:
            power, vf, vb = _parse_from_name(path.name)
            angle = _parse_angle_from_name(path.name)
            if power is None and angle is None:
                power = _power_from_bare_name(path.name)
            if power is None:
                power = angle
            vf_named, vb_named = vf is not None, vb is not None
        frames.append(PLFrame(
            path=path, power_w=power, v_front=vf, v_back=vb, wavelength_nm=None,
            image=np.zeros((1, 1)), meta={TRPL_KEY: data}, mtime=float(path.stat().st_mtime),
            angle_deg=angle, vf_named=vf_named, vb_named=vb_named,
        ))
    return frames, errors


def trpl_of(fr: Optional[PLFrame]) -> Optional[TrplData]:
    return None if fr is None else fr.meta.get(TRPL_KEY)


def read_session(folder: Path) -> dict:
    """Best-effort read of the v2 GUI's ``session.md``: TRPL window, measure mode, dI/dP line."""
    out: dict = {}
    path = Path(folder) / "session.md"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        body = line.lstrip("- ").strip()
        if ":" not in body:
            continue
        key, val = (s.strip() for s in body.split(":", 1))
        low = key.lower()
        if low.startswith("histogram start/end"):
            try:
                vals = [float(v) for v in val.split(",")]
            except ValueError:
                continue
            if len(vals) >= 2 and vals[1] > vals[0]:
                out["hist_ns"] = (vals[0], vals[1])
        elif low == "measure":
            out["measure"] = val
        elif low.startswith("di/dp"):
            out["didp"] = val
    return out
