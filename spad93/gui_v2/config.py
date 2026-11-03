"""Defaults for the SPAD93 v2 GUI."""

import os
from typing import Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9998

BTN_W = 130
STATUS_W = 360

# Live counts
LIVE_INTEGRATION_MS = 200
LIVE_MIN_INTERVAL_S = 0.25
LIMIT_WARN_RATIO = 0.8          # state label turns red above this fraction of the pixel limit

# TRPL acquisition
TRPL_DEFAULT_ACQ_S = 10.0
TRPL_DARK_ACQ_S = 30.0  # dark reference (beam blocked)
TRPL_DEFAULT_BIN_PS = 50
TRPL_UPDATE_S = 1.0
TRPL_CANCEL_WAIT_S = 3.0
TRPL_SAVE_WORKER_ROWS = 50_000  # larger files are written in a worker thread
ALIGN_MS = 1000
CLOSE_TIMEOUT_S = 15.0

# Wavelength calibration
CALIB_CAPTURE_FRAMES = 5
CALIB_DEFAULT_ROWS = 3

SLIDER_DEBOUNCE_MS = 70

# Colors
MAP_CMAP = "inferno"
FIT_CMAP = "viridis"
SELECT_COLOR = "#06b6d4"
EXCLUDE_COLOR = "#ef4444"
NAN_COLOR = "#d1d5db"
INACTIVE_COLOR = "#cbd5e1"
REF_COLOR = "#059669"
FIT_LINE_COLOR = "#dc2626"
DATA_COLOR = "#1d4ed8"
COMPONENT_COLORS = ("#ff7f0e", "#e11d48")  # double 2D Gaussian: components 1 and 2
TRACE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

SETTINGS_ORG = "Automation"
SETTINGS_APP = "SPAD93"

_DATA_DIR: Optional[str] = None


def data_dir() -> str:
    """``AUTOMATION_DATA_DIR``, else ``measurements.config.DATA_DIR``, else the Desktop."""
    global _DATA_DIR
    if _DATA_DIR is None:
        env = os.environ.get("AUTOMATION_DATA_DIR")
        if env:
            _DATA_DIR = env
        else:
            try:
                # Imported lazily: measurements.config also loads the andor/rot GUIs.
                from measurements.config import DATA_DIR

                _DATA_DIR = DATA_DIR
            except Exception:
                _DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
    return _DATA_DIR


def save_dir(sub: str) -> str:
    path = os.path.join(data_dir(), "spad93", sub)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path
