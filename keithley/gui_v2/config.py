VISA_DLL = r"C:\Windows\System32\visa64.dll"  # set "" to use system default

DEFAULT_B_RESOURCE = "TCPIP0::10.164.14.237::5025::SOCKET"
DEFAULT_A_RESOURCE = "TCPIP0::169.254.153.76::5025::SOCKET"

# DEFAULT_A_RESOURCE = "GPIB0::24::INSTR"
# DEFAULT_B_RESOURCE = "GPIB0::23::INSTR"

POLL_MS_DEFAULT = 200
PLOT_MAX_POINTS = 100

BTN_W = 130
READ_W = 120
STATUS_W = 340

DEFAULT_ICOMP_NA = 10.0
DEFAULT_SWEEP_ICOMP_NA = 10.0

# Ramp defaults
RAMP_STEP_V = 0.01
RAMP_DWELL_S = 0.05

# Sweep settle after each macro step before measuring
SWEEP_SETTLE_MS = 50

READOUT_UNKNOWN = "--"
POLL_ERROR_LIMIT = 3
POLL_TIMEOUT_MS = 1000

# Leakage test
LEAKAGE_V_DECIMALS = 2          # bin width 10**-2 V, matches RAMP_STEP_V
LEAKAGE_FLUSH_EVERY = 50        # CSV flush interval in rows (microsteps arrive at ~20 Hz)
LEAKAGE_AUTOUPDATE_S = 10       # default period for the auto-update checkbox
LEAKAGE_MAX_BINS_WARN = 5000    # status-bar warning threshold on redraw
LEAKAGE_WIN_SIZE = (1100, 620)
