"""Defaults of the Montana pump-probe gate / power dependence GUI."""

from measurements.config import DATA_DIR, DEFAULT_RAMP_STEP_DEG, DEFAULT_STAGE_SCALE  # noqa: F401
from measurements.wheel_hwp_power_calib_config import DEFAULT_HWP_SERIAL, DEFAULT_ND_SERIAL

try:
    from keithley.gui import config as keithley_cfg
except Exception:
    keithley_cfg = None

# ---------------------------------------------------------------- layout
BTN_W = 110
SMALL_BTN_W = 70
STATUS_W = 420
SETTINGS_MAX_H = 350  # height of the settings tabs (the live display gets the rest)
DISPLAY_REFRESH_MS = 100  # live capture: redraw at most this often (old GUI: 100 ms)

# ---------------------------------------------------------------- camera / analysis (old GUI defaults)
DEFAULT_N_FRAMES = 400
DEFAULT_AVG_TIMES = 1
DEFAULT_ROI = (625, 10, 5, 5)  # reference ROI x, y, w, h
DEFAULT_CROP_ENABLED = True
DEFAULT_CROP_X0 = 0
DEFAULT_CROP_W = 480
DEFAULT_PIXEL_SIZE_UM = 0.07
DEFAULT_WAVELENGTH_K = 1.0  # lambda = k * pixel + b (nm)
DEFAULT_WAVELENGTH_B = 0.0
DEFAULT_Z_RANGE = (-10.0, 10.0)
DEFAULT_MONITOR_Z = (3000.0, 8000.0)

# ---------------------------------------------------------------- delay scan (old GUI defaults)
DEFAULT_SIMPLE_SCAN = (-10.0, 5, 5.0)  # start ps, steps, increment ps
DEFAULT_MULTI_RANGES = ((-5.0, 5.0, 0.5, 3), (5.0, 20.0, 1.0, 2), (20.0, 100.0, 5.0, 1))
DEFAULT_DELAY_TEXT = "-10, -5, 0, 5, 10"

# ---------------------------------------------------------------- Newport SMC100 delay stage
DEFAULT_SMC_PORT = "COM3"
DEFAULT_SMC_ADDRESS = 1
SMC_POLL_S = 0.5
DELAY_PASS_FACTOR = 4.0  # time delay = factor x distance / c (double-pass retroreflector, as the old GUI)

# ---------------------------------------------------------------- ND wheel (power) + polarization stage
DEFAULT_WHEEL_SERIAL = DEFAULT_ND_SERIAL  # 27600911, as the old pump-probe GUI and the ND calibration
DEFAULT_POL_SERIAL = DEFAULT_HWP_SERIAL  # 27264008, as the old pump-probe GUI
STAGE_POLL_S = 0.5
DEFAULT_POWER_SETTLE_S = 0.5
DEFAULT_POL_SETTLE_S = 0.5
DEFAULT_LH_ANGLE = 0.0
POL_RH_OFFSET_DEG = 45.0
DEFAULT_MANUAL_ANGLES = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
POWER_UNITS = {"pW": 1e-12, "nW": 1e-9, "uW": 1e-6, "mW": 1e-3, "W": 1.0}
DEFAULT_TABLE_UNIT = "nW"

# ---------------------------------------------------------------- gates (Keithley SMUs)
VISA_DLL = getattr(keithley_cfg, "VISA_DLL", r"C:\Windows\System32\visa64.dll")
DEFAULT_FRONT_RESOURCE = "GPIB0::24::INSTR"  # as the old pump-probe GUI
DEFAULT_BACK_RESOURCE = "GPIB0::23::INSTR"
DEFAULT_ICOMP_NA = 10.0
GATE_POLL_S = 0.5
GATE_READ_TIMEOUT_MS = 5000
GATE_POLL_ERROR_LIMIT = 3
GATE_RECONNECT_S = 3.0
RAMP_STEP_V = 0.01
RAMP_DWELL_S = 0.10
LONG_RAMP_STEPS = 50  # longer ramps read V/I at every microstep and stop at the compliance
DEFAULT_GATE_SETTLE_MS = 500
DEFAULT_SWEEP_V0 = 0.0
DEFAULT_SWEEP_STEP = 1.0
DEFAULT_SWEEP_STEPS = 5
V_LIMIT = 210.0

# ---------------------------------------------------------------- saving
DEFAULT_SAVE_BASE = "pump_probe_scan"
SAVE_DIR_PREFIX = "ta_montana"

# ---------------------------------------------------------------- watchdog (hardware not responding)
# The scan threads report what they are doing; the GUI warns (it does not stop anything)
# when one step takes longer than its limit.  Terminate -> reconnect / test -> Resume.
WATCHDOG_TICK_MS = 1000
WATCHDOG_CAPTURE_S = 60.0  # camera capture: this + WATCHDOG_CAPTURE_PER_FRAME_S per frame
WATCHDOG_CAPTURE_PER_FRAME_S = 0.05
WATCHDOG_MOVE_S = 180.0  # delay stage / ND wheel / polarization stage move
WATCHDOG_EXTRA_S = 60.0  # added to settle times, gate ramps, gate reads, saving
ERROR_LOG_NAME = "ta_montana_errors.log"  # in DATA_DIR: uncaught exceptions + native crash tracebacks
