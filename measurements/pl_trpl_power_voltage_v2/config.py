"""Defaults for the PL + TRPL v2 GUI.

Everything from the current PL power/gate GUI config is re-exported unchanged
(gates, sweeps, Andor, crop), plus the SPAD (SPAD23 or SPAD93) / TCSPC and polling settings.
"""

from measurements.pl_power_voltage_config import *  # noqa: F401,F403
from measurements.pl_power_voltage_config import __all__ as _PL_ALL
from measurements.pl_power_voltage_config import POLL_GATE_MS, POLL_MS
from spad23.spad23_pspad_wrapper import DEFAULT_PORT as _PSPAD_PORT
from spad23.spad23_pspad_wrapper import set_pixel_id_offset

# ---------------------------------------------------------------- SPAD23 / SPAD93 (TCSPC)
DEFAULT_TCSPC_HOST = "127.0.0.1"
DEFAULT_TCSPC_PORT = _PSPAD_PORT  # pSPAD command server (same port for SPAD23 and SPAD93)
DEFAULT_TCSPC_ACQ_S = 10.0
DEFAULT_TCSPC_BIN_PS = 500
DEFAULT_HIST_START_NS = 130.0
DEFAULT_HIST_END_NS = 250.0
DEFAULT_HIST_REF_NS = 130.0

# SPAD23 only: raw pixel id = pixel + offset on the pSPAD server (the SPAD23 v2 GUI's --id-offset).
# The SPAD93 clients use their own default (0, verified on hardware).
# The SPAD23 ids were never checked on hardware: run
#   python -m spad23.spad23_pspad_wrapper --probe
# and change this if the ids are offset.
SPAD_PIXEL_ID_OFFSET = 0
set_pixel_id_offset(SPAD_PIXEL_ID_OFFSET)

# Detector choice: Auto (detected on Connect from the number of pixel lines) or a fixed one;
# the last choice is remembered here.
SPAD_DETECTORS = ("SPAD23", "SPAD93")
DETECTOR_AUTO = "Auto"
DETECTOR_SETTINGS = ("Automation", "PLTRPLv2")
DETECTOR_SETTINGS_KEY = "spad_detector"
# SPAD view (2D space / spectral) and the last wavelength calibration JSON per detector (same QSettings)
SPAD_VIEW_KEY = "spad_view_mode"
WLCAL_PATH_KEY = "spad_lambda_cal_path_"  # + detector key

SPAD_LIVE_INTEGRATION_MS = 180  # live count map (v1 value)
SPAD_SWEEP_COUNT_MS = 120  # quick count snapshot at every sweep point (v1 value)
TRPL_ALIGN_MS = 1000  # calibrate + align: T,a,<ms>

# ---------------------------------------------------------------- pump-2 flipper (dI/dP mode)
from flipper.flipper_wrapper import DEFAULT_SERIAL as _FLIPPER_SERIAL  # noqa: E402

DEFAULT_FLIPPER_SERIAL = _FLIPPER_SERIAL  # Thorlabs MFF, S/N 37005944
FLIPPER_SETTLE_MS = 200  # wait after each flip before acquiring
FLIPPER_POLL_S = 0.5
# Which flipper position lets pump 2 through: shared with the flipper GUI (flipper/gui_v2)
FLIPPER_SETTINGS = ("Automation", "FilterFlipper", "pump2_on_position")
PUMP2_ON_DEFAULT = 2
PUMP2_TAGS = {False: "P2off", True: "P2on"}  # file-name tag of each acquisition

# ---------------------------------------------------------------- polling (worker threads)
CAM_POLL_S = 1.0  # Andor temperature
SPEC_POLL_S = 1.0  # spectrograph grating / center / slit readback
STAGE_POLL_S = POLL_MS / 1000.0
GATE_POLL_S = POLL_GATE_MS / 1000.0
GATE_RECONNECT_S = 3.0  # auto-reconnect interval of a gate that stopped answering
# V/I read timeout of the gate poll: a 2450 read is two measurements, and current autoranging at pA
# can take over a second (the PL GUI's 1 s timed reads out and left late replies behind)
GATE_READ_TIMEOUT_MS = 5000
LONG_RAMP_STEPS = 50  # ramps longer than this (0.5 V at 10 mV) read V/I and check compliance every microstep
QUIET_WAIT_MAX_S = 5.0  # longest wait for device polls to go quiet before a job starts

# ---------------------------------------------------------------- measure modes
MEASURE_PL = "PL"
MEASURE_TRPL = "TRPL"
MEASURE_BOTH = "PL + TRPL"
MEASURE_MODES = (MEASURE_PL, MEASURE_TRPL, MEASURE_BOTH)

# ---------------------------------------------------------------- plots
LINECUT_CMAP = "viridis"
LIFETIME_CMAP = "inferno"
GATE_LINE_COLOR = "#1f77b4"

__all__ = list(_PL_ALL) + [
    "DEFAULT_TCSPC_HOST", "DEFAULT_TCSPC_PORT", "DEFAULT_TCSPC_ACQ_S", "DEFAULT_TCSPC_BIN_PS",
    "DEFAULT_HIST_START_NS", "DEFAULT_HIST_END_NS", "DEFAULT_HIST_REF_NS",
    "SPAD_PIXEL_ID_OFFSET", "SPAD_DETECTORS", "DETECTOR_AUTO", "DETECTOR_SETTINGS", "DETECTOR_SETTINGS_KEY",
    "SPAD_VIEW_KEY", "WLCAL_PATH_KEY",
    "SPAD_LIVE_INTEGRATION_MS", "SPAD_SWEEP_COUNT_MS", "TRPL_ALIGN_MS",
    "DEFAULT_FLIPPER_SERIAL", "FLIPPER_SETTLE_MS", "FLIPPER_POLL_S", "FLIPPER_SETTINGS", "PUMP2_ON_DEFAULT",
    "PUMP2_TAGS",
    "CAM_POLL_S", "SPEC_POLL_S", "STAGE_POLL_S", "GATE_POLL_S", "GATE_RECONNECT_S", "QUIET_WAIT_MAX_S",
    "GATE_READ_TIMEOUT_MS", "LONG_RAMP_STEPS",
    "MEASURE_PL", "MEASURE_TRPL", "MEASURE_BOTH", "MEASURE_MODES",
    "LINECUT_CMAP", "LIFETIME_CMAP", "GATE_LINE_COLOR",
]
