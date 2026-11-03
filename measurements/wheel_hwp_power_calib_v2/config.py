"""Defaults for the wheel + HWP calibration v2 GUI.

The device serials and sweep defaults of the v1 GUI are re-exported unchanged.
"""

from measurements.wheel_hwp_power_calib_config import (  # noqa: F401
    DATA_DIR,
    DEFAULT_HWP_FIXED_ANGLE,
    DEFAULT_HWP_SERIAL,
    DEFAULT_HWP_START,
    DEFAULT_HWP_STEP,
    DEFAULT_HWP_STOP,
    DEFAULT_N_READINGS,
    DEFAULT_ND_ANGLE,
    DEFAULT_ND_SERIAL,
    DEFAULT_ND_START,
    DEFAULT_ND_STEP,
    DEFAULT_ND_STOP,
    DEFAULT_PM_AVERAGING,
    DEFAULT_RAMP_STEP_DEG,
    DEFAULT_STAGE_SCALE,
)

BTN_W = 90
STATUS_W = 360

STAGE_POLL_S = 0.5  # position readback while connected
DEFAULT_WAVELENGTH_NM = 532.0
PM_ZERO_TIMEOUT_S = 30.0

# Power planner
DEFAULT_TOL_PCT = 1.0  # targets are hit within this; among those the least angle-sensitive pair wins
DEFAULT_TARGET_UNIT = "nW"
TARGET_UNITS = ("pW", "nW", "µW", "mW", "W")

# Colors
CURVE_COLOR = "#1f77b4"
RANGE_COLOR = "#9ecae1"
OK_COLOR = "#2e7d32"
BAD_COLOR = "#d62728"
BAD_ROW_BG = "#fde2e1"
