"""State shared by the Counts and TRPL tabs (mode, excluded pixels, calibration)."""

from typing import FrozenSet, Iterable, Optional

import numpy as np
from PyQt5 import QtCore

from spad93.spad93_analysis import WavelengthCalibration
from spad93.spad93_geometry import NUM_PIXELS

MODE_2D = "2d"
MODE_SPECTRAL = "spectral"


class SharedState(QtCore.QObject):
    excluded_changed = QtCore.pyqtSignal()
    calibration_changed = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._excluded: FrozenSet[int] = frozenset()
        self._calibration: Optional[WavelengthCalibration] = None
        self._calibration_path = ""
        self._mode = MODE_2D

    # ----------------------------------------------------------- excluded
    @property
    def excluded(self) -> FrozenSet[int]:
        return self._excluded

    def set_excluded(self, pixels: Iterable[int]) -> None:
        new = frozenset(int(p) for p in pixels if 0 <= int(p) < NUM_PIXELS)
        if new != self._excluded:
            self._excluded = new
            self.excluded_changed.emit()

    def toggle_excluded(self, pixel: int) -> None:
        pixel = int(pixel)
        if pixel in self._excluded:
            self.set_excluded(self._excluded - {pixel})
        else:
            self.set_excluded(self._excluded | {pixel})

    def enabled_mask(self) -> np.ndarray:
        mask = np.ones(NUM_PIXELS, dtype=bool)
        if self._excluded:
            mask[list(self._excluded)] = False
        return mask

    # -------------------------------------------------------- calibration
    @property
    def calibration(self) -> Optional[WavelengthCalibration]:
        return self._calibration

    @property
    def calibration_path(self) -> str:
        return self._calibration_path

    def set_calibration(self, calib: Optional[WavelengthCalibration], path: str = "") -> None:
        self._calibration = calib
        self._calibration_path = path if calib is not None else ""
        self.calibration_changed.emit()

    def wavelength(self, x_um):
        return None if self._calibration is None else self._calibration.wavelength(x_um)

    # --------------------------------------------------------------- mode
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def spectral(self) -> bool:
        return self._mode == MODE_SPECTRAL

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_2D, MODE_SPECTRAL):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != self._mode:
            self._mode = mode
            self.mode_changed.emit(mode)
