"""The two SPAD detectors of the PL + TRPL v2 GUI: SPAD23 (23 px) and SPAD93 (93 px).

Both sit behind the same pSPAD command server and protocol, so one GUI drives
either; what changes is the pixel layout, the client classes and the gui_v2 map
widgets.  A ``DetectorProfile`` bundles those; ``profile(key)`` builds it (and
imports that detector's package) on first use.

``detect_detector`` tells them apart by the reply to ``I,<ms>``: one ``id,count``
line per pixel, 23 or 93 of them (counting ids works whatever the id offset).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore

SPAD23 = "SPAD23"
SPAD93 = "SPAD93"
DETECTOR_KEYS = (SPAD23, SPAD93)

# DeviceController states (identical strings in spad23.gui_v2.device and spad93.gui_v2.device)
DISCONNECTED = "disconnected"
CONNECTING = "connecting"
IDLE = "idle"
LIVE = "live"
BUSY = "busy"

DETECT_INTEGRATION_MS = 20


@dataclass(frozen=True)
class DetectorProfile:
    key: str
    num_pixels: int
    center_pixel: int
    pixel_x_um: np.ndarray
    pixel_y_um: np.ndarray
    an: object  # the detector's analysis module (fit_gaussian_2d with its pitches, MIN_FIT_PIXELS_2D, ...)
    io: object  # the detector's io module (TRPL / dark reference CSV files)
    count_client: type
    tcspc_client: type
    device_controller: type
    PixelMapPlot: type
    Gauss2DFitPlot: type

    @property
    def label(self) -> str:
        return f"{self.key} ({self.num_pixels} px)"


_MODULES = {
    SPAD23: dict(geometry="spad23.spad23_geometry", analysis="spad23.spad23_analysis", io="spad23.spad23_io",
                 count=("spad23.spad23_pspad_wrapper", "Spad23CountClient"),
                 tcspc=("spad23.spad23_pspad_tcspc", "Spad23TcspcClient"),
                 device="spad23.gui_v2.device", maps="spad23.gui_v2.map_plots"),
    SPAD93: dict(geometry="spad93.spad93_geometry", analysis="spad93.spad93_analysis", io="spad93.spad93_io",
                 count=("spad93.spad93_wrapper", "Spad93CountClient"),
                 tcspc=("spad93.spad93_tcspc_wrapper", "Spad93TcspcClient"),
                 device="spad93.gui_v2.device", maps="spad93.gui_v2.map_plots"),
}


def _attr(spec: Tuple[str, str]):
    return getattr(importlib.import_module(spec[0]), spec[1])


@lru_cache(maxsize=None)
def profile(key: str) -> DetectorProfile:
    if key not in _MODULES:
        raise KeyError(f"unknown detector {key!r} (expected one of {', '.join(DETECTOR_KEYS)})")
    m = _MODULES[key]
    geo = importlib.import_module(m["geometry"])
    maps = importlib.import_module(m["maps"])
    return DetectorProfile(
        key=key,
        num_pixels=int(geo.NUM_PIXELS),
        center_pixel=int(geo.CENTER_PIXEL),
        pixel_x_um=np.asarray(geo.PIXEL_X_UM, dtype=np.float64),
        pixel_y_um=np.asarray(geo.PIXEL_Y_UM, dtype=np.float64),
        an=importlib.import_module(m["analysis"]),
        io=importlib.import_module(m["io"]),
        count_client=_attr(m["count"]),
        tcspc_client=_attr(m["tcspc"]),
        device_controller=importlib.import_module(m["device"]).DeviceController,
        PixelMapPlot=maps.PixelMapPlot,
        Gauss2DFitPlot=maps.Gauss2DFitPlot,
    )


@lru_cache(maxsize=None)
def snapshot_types() -> Tuple[tuple, tuple]:
    """(count snapshot classes, TRPL snapshot classes) of both detectors, for isinstance checks."""
    from spad23.spad23_pspad_tcspc import TrplSnapshot as T23
    from spad23.spad23_pspad_wrapper import CountSnapshot as C23
    from spad93.spad93_tcspc_wrapper import TrplSnapshot as T93
    from spad93.spad93_wrapper import CountSnapshot as C93
    return (C23, C93), (T23, T93)


def is_count_snapshot(obj) -> bool:
    return isinstance(obj, snapshot_types()[0])


def is_trpl_snapshot(obj) -> bool:
    return isinstance(obj, snapshot_types()[1])


def classify_id_count(n_ids: int) -> Optional[str]:
    """Detector from the number of distinct pixel ids in an ``I`` reply (None = unclear)."""
    if n_ids >= 60:
        return SPAD93
    if 10 <= n_ids <= 40:
        return SPAD23
    return None


def detect_detector(host: str, port: int, timeout_s: float = 5.0) -> Tuple[Optional[str], str]:
    """Ask the pSPAD server for one short count frame and count its pixel lines.

    Returns ``(key or None, explanation)``; raises on connection errors.
    """
    from spad93.spad93_wrapper import Spad93CountClient  # any client: only the raw reply is used

    with Spad93CountClient(host=host, port=int(port), timeout_s=timeout_s) as client:
        text = client.query(f"I,{DETECT_INTEGRATION_MS}", timeout_s=timeout_s, min_lines=23)
    ids = set()
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[0].strip().lstrip("-").isdigit():
            ids.add(int(parts[0]))
    key = classify_id_count(len(ids))
    span = f"ids {min(ids)}..{max(ids)}" if ids else "no id lines"
    if key is None:
        return None, (f"The pSPAD server returned {len(ids)} pixel lines ({span}); expected 23 (SPAD23) "
                      "or 93 (SPAD93). Pick the detector manually.")
    return key, f"{key}: the server returned {len(ids)} pixel lines ({span})"


class DetectWorker(QtCore.QThread):
    """``detect_detector`` off the GUI thread; ``done(key or "", message, ok)``."""

    done = QtCore.pyqtSignal(str, str, bool)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host, self.port = host, int(port)

    def run(self) -> None:
        try:
            key, msg = detect_detector(self.host, self.port)
            self.done.emit(key or "", msg, True)
        except Exception as exc:
            self.done.emit("", f"Could not reach the pSPAD server at {self.host}:{self.port}: {exc}", False)
