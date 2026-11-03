"""Voltage-binned accumulator for the leakage current test.

No Qt here: this is plain statistics so it can be exercised without a GUI.
Units are SI throughout (volts, amps); the nA conversion happens only at the
UI boundary, as everywhere else in this package.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from ..keithley_wrapper import KEITHLEY_V_LIMIT

from .config import LEAKAGE_V_DECIMALS


class LeakageRecorder:
    """Accumulates (V, I) samples into bins of rounded voltage.

    One instance per gate. Each bin keeps a running count/mean/M2 (Welford),
    so memory is O(bins) rather than O(samples) and the variance stays stable
    at nanoamp scale -- a naive sum-of-squares at ~1e-9 A cancels
    catastrophically and routinely comes out negative.

    Bins are keyed on the *measured* voltage, not the setpoint: in compliance
    the read-back voltage wanders off the setpoint, so bins can come out a
    little ragged.
    """

    def __init__(self, decimals: int = LEAKAGE_V_DECIMALS):
        self._decimals = int(decimals)
        self._scale = float(10 ** self._decimals)
        self._bins: Dict[int, List[float]] = {}  # key -> [n, mean, m2]
        self.n_samples = 0
        self.n_rejected = 0

    # ---- properties ----

    @property
    def decimals(self) -> int:
        return self._decimals

    @property
    def n_bins(self) -> int:
        return len(self._bins)

    # ---- accumulation ----

    def add(self, v_volts: float, i_amps: float) -> bool:
        """Bin one sample. Returns False if the sample was rejected."""
        v = float(v_volts)
        i = float(i_amps)

        # The SMU returns NaN for an empty field, and ramp_to_voltage
        # manufactures VIReading(i=nan) in a few paths. int(round(nan)) raises.
        if not (math.isfinite(v) and math.isfinite(i)):
            self.n_rejected += 1
            return False
        if abs(v) > KEITHLEY_V_LIMIT:
            self.n_rejected += 1
            return False

        key = int(round(v * self._scale))
        b = self._bins.get(key)
        if b is None:
            self._bins[key] = [1.0, i, 0.0]
        else:
            b[0] += 1.0
            d = i - b[1]
            b[1] += d / b[0]
            b[2] += d * (i - b[1])

        self.n_samples += 1
        return True

    def clear(self) -> None:
        self._bins.clear()
        self.n_samples = 0
        self.n_rejected = 0

    def set_decimals(self, decimals: int) -> None:
        """Change the bin width. Discards existing bins (they are re-keyed)."""
        self._decimals = int(decimals)
        self._scale = float(10 ** self._decimals)
        self.clear()

    # ---- readout ----

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(V, mean I, sample std of I, n) sorted by voltage ascending.

        std is exactly 0.0 for single-sample bins -- never NaN, which would
        corrupt the QPainterPath the error bars are drawn from.
        """
        if not self._bins:
            empty = np.empty(0, dtype=float)
            return (empty, empty.copy(), empty.copy(), np.empty(0, dtype=int))

        keys = sorted(self._bins.keys())
        n = len(keys)
        v = np.empty(n, dtype=float)
        mean = np.empty(n, dtype=float)
        std = np.empty(n, dtype=float)
        cnt = np.empty(n, dtype=int)

        for j, key in enumerate(keys):
            b = self._bins[key]
            v[j] = key / self._scale
            mean[j] = b[1]
            cnt[j] = int(b[0])
            std[j] = math.sqrt(max(0.0, b[2] / (b[0] - 1.0))) if b[0] >= 2 else 0.0

        return (v, mean, std, cnt)

    def rows(self) -> List[Tuple[float, float, float, int]]:
        """Binned table for CSV export: (V, mean I, std I, n)."""
        v, mean, std, cnt = self.arrays()
        return [(float(v[j]), float(mean[j]), float(std[j]), int(cnt[j])) for j in range(len(v))]
