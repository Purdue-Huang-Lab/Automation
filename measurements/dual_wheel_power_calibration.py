import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DualWheelPowerEntry:
    a_deg: float
    b_deg: float
    power_w: float
    intensity: float
    index: int


@dataclass
class DualWheelPowerData:
    path: str
    unit: str
    scale: float
    entries: List[DualWheelPowerEntry]


def _normalize_unit(unit: str) -> str:
    val = unit.strip().lower()
    if val.endswith("w"):
        if val.startswith("n"):
            return "nW"
        if val.startswith("u"):
            return "uW"
        if val.startswith("m"):
            return "mW"
        return "W"
    if val in ("nw", "nanowatt"):
        return "nW"
    if val in ("uw", "micro", "microwatt"):
        return "uW"
    if val in ("mw", "milliwatt"):
        return "mW"
    if val == "w":
        return "W"
    return "nW"


def _unit_scale(unit: str) -> float:
    unit = _normalize_unit(unit)
    if unit == "W":
        return 1.0
    if unit == "mW":
        return 1e-3
    if unit == "uW":
        return 1e-6
    return 1e-9


def _row_has_alpha(row: List[str]) -> bool:
    for cell in row:
        if re.search(r"[a-zA-Z]", str(cell)):
            return True
    return False


def _parse_unit_from_header(header: List[str]) -> str:
    for cell in header:
        cell = cell.strip().lower()
        if cell.startswith("power_"):
            return _normalize_unit(cell.split("power_", 1)[-1])
        if "power" in cell and "(" in cell and ")" in cell:
            inner = cell.split("(", 1)[-1].split(")", 1)[0]
            return _normalize_unit(inner)
    return "nW"


def is_dual_wheel_calibration(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            header = [c.strip().lower() for c in row]
            if len(header) < 3:
                return False
            if "a" in header[0] and "b" in header[1] and "power" in header[2]:
                return True
            if _row_has_alpha(header):
                return False
            return False
    return False


def load_dual_wheel_calibration(path: str) -> DualWheelPowerData:
    if not path:
        raise ValueError("No calibration path provided")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            if header is None and _row_has_alpha(row):
                header = row
                continue
            rows.append(row)

    unit = _parse_unit_from_header([c.strip() for c in (header or [])])
    scale = _unit_scale(unit)

    entries: List[DualWheelPowerEntry] = []
    for idx, row in enumerate(rows):
        if len(row) < 3:
            continue
        try:
            a_deg = float(row[0])
            b_deg = float(row[1])
            power = float(row[2])
        except Exception:
            continue
        intensity = float("nan")
        if len(row) > 3:
            try:
                intensity = float(row[3])
            except Exception:
                intensity = float("nan")
        entries.append(
            DualWheelPowerEntry(
                a_deg=a_deg,
                b_deg=b_deg,
                power_w=power * scale,
                intensity=float(intensity),
                index=int(idx),
            )
        )

    if not entries:
        raise ValueError("No calibration data rows found")

    return DualWheelPowerData(
        path=path,
        unit=unit,
        scale=scale,
        entries=entries,
    )
