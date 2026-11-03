"""Defaults for the two-pump dI/dP viewer."""

from measurements.config import DATA_DIR

BTN_W = 90
STATUS_W = 360

# Crop T/B/L/R applied to every frame (same default as the v2 power/gate viewer)
DEFAULT_CROP = (50, 50, 200, 200)

# Baseline-correction window (nm)
DEFAULT_BG_LO_NM = 900.0
DEFAULT_BG_HI_NM = 940.0

# Pump-2 power (µW).  Enter the pump-1-equivalent power if pump 2 has a
# different wavelength / spot size, or use "Estimate P₂,eff".
DEFAULT_P2_UW = 0.01

# dlnI/dlnP is masked where I < floor × max|I| (log of ~0 is meaningless)
DEFAULT_REL_FLOOR = 0.02

# Display-grid columns used to draw maps whose sweep axis is non-uniform
MAP_DISPLAY_COLS = 600

# Colours
SET1_COLOR = "#1f77b4"      # pump 1 only
SET2_COLOR = "#d62728"      # pump 1 + pump 2
DIFF_COLOR = "#2ca02c"      # difference / measured derivative
REF_COLOR = "#7f7f7f"       # numerical derivative of set 1 (reference)
CURSOR_X_COLOR = "#00a0a0"
CURSOR_Y_COLOR = "#e08000"

SEQ_CMAP = "viridis"
DIV_CMAP = "coolwarm"

# Pump-2 tags written by the PL + TRPL v2 GUI's dI/dP mode (one folder, both states)
PUMP2_OFF_TAG = "P2off"
PUMP2_ON_TAG = "P2on"

# TRPL tab: window summed for the pixel map (ns; session.md overrides it) and rebin factor
DEFAULT_TRPL_WIN_NS = (130.0, 250.0)
DEFAULT_REBIN = 1

__all__ = [
    "DATA_DIR", "BTN_W", "STATUS_W", "DEFAULT_CROP", "DEFAULT_BG_LO_NM", "DEFAULT_BG_HI_NM",
    "DEFAULT_P2_UW", "DEFAULT_REL_FLOOR", "MAP_DISPLAY_COLS",
    "SET1_COLOR", "SET2_COLOR", "DIFF_COLOR", "REF_COLOR", "CURSOR_X_COLOR", "CURSOR_Y_COLOR",
    "SEQ_CMAP", "DIV_CMAP", "PUMP2_OFF_TAG", "PUMP2_ON_TAG", "DEFAULT_TRPL_WIN_NS", "DEFAULT_REBIN",
]
