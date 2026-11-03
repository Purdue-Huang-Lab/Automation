"""Defaults for the filter flipper v2 GUI."""

from flipper.flipper_wrapper import DEFAULT_SERIAL, TRANSIT_TIME_RANGE_S

BTN_W = 110
STATUS_W = 360

POLL_MS = 250  # position readback while connected
POLL_ERROR_LIMIT = 5  # consecutive poll errors before the GUI disconnects
HISTORY_S = 300.0  # position-history window

DEFAULT_TRANSIT_S = 0.5
DEFAULT_CYCLES = 10
DEFAULT_DWELL_S = 1.0

# Which flipper position lets pump 2 through (saved in QSettings; the PL+TRPL
# GUI is meant to read the same setting when the flipper is integrated).
PUMP2_ON_DEFAULT = 2

SETTINGS_ORG = "Automation"
SETTINGS_APP = "FilterFlipper"

# Colors
POS_COLOR = "#1f77b4"
MOVE_COLOR = "#d62728"
MEAN_COLOR = "#7f7f7f"
ON_COLOR = "#2e7d32"
OFF_COLOR = "#6b7280"

__all__ = [
    "DEFAULT_SERIAL", "TRANSIT_TIME_RANGE_S", "BTN_W", "STATUS_W", "POLL_MS", "POLL_ERROR_LIMIT",
    "HISTORY_S", "DEFAULT_TRANSIT_S", "DEFAULT_CYCLES", "DEFAULT_DWELL_S", "PUMP2_ON_DEFAULT",
    "SETTINGS_ORG", "SETTINGS_APP", "POS_COLOR", "MOVE_COLOR", "MEAN_COLOR", "ON_COLOR", "OFF_COLOR",
]
