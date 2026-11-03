from measurements.config import (
    DATA_DIR,
    DEFAULT_ACQ_NUMBER,
    DEFAULT_CENTER_WL_NM,
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_GRATING,
    DEFAULT_LINECUT_WIDTH,
    DEFAULT_OUTPUT_AMP,
    DEFAULT_PREAMP_GAIN,
    DEFAULT_READOUT_RATE,
    DEFAULT_SLIT_ID,
    DEFAULT_SLIT_UM,
    DEFAULT_SPEC_INDEX,
    POLL_MS,
)

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None

BTN_W = 120
STATUS_W = 360

# Curve colors
ON_COLOR = "#d62728"
OFF_COLOR = "#1f77b4"
DIFF_COLOR = "k"

# Map colormap (pyqtgraph / matplotlib name)
MAP_CMAP = "viridis"

__all__ = [
    "DATA_DIR",
    "DEFAULT_ACQ_NUMBER",
    "DEFAULT_CENTER_WL_NM",
    "DEFAULT_CROP_BOTTOM",
    "DEFAULT_CROP_LEFT",
    "DEFAULT_CROP_RIGHT",
    "DEFAULT_CROP_TOP",
    "DEFAULT_EXPOSURE_MS",
    "DEFAULT_GRATING",
    "DEFAULT_LINECUT_WIDTH",
    "DEFAULT_OUTPUT_AMP",
    "DEFAULT_PREAMP_GAIN",
    "DEFAULT_READOUT_RATE",
    "DEFAULT_SLIT_ID",
    "DEFAULT_SLIT_UM",
    "DEFAULT_SPEC_INDEX",
    "POLL_MS",
    "andor_cfg",
    "BTN_W",
    "STATUS_W",
    "ON_COLOR",
    "OFF_COLOR",
    "DIFF_COLOR",
    "MAP_CMAP",
]
