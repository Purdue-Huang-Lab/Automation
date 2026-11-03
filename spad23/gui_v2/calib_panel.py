"""Wavelength calibration panel: capture the center-column peak at known laser wavelengths.

The SPAD23 disperses along y, so the peak position is a y (row position) in um.
"""

import os
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from spad23.spad23_analysis import CalibPoint, WavelengthCalibration, locate_peak
from spad23.spad23_geometry import CENTER_X_UM, ROW_Y_UM, column_profile
from spad23.spad23_io import load_calibration_json, save_calibration_json, timestamp
from spad23.spad23_pspad_wrapper import CountSnapshot
from spad23.gui_v2.config import BTN_W, CALIB_CAPTURE_FRAMES, CALIB_DEFAULT_ROWS, save_dir
from spad23.gui_v2.plots import CalibrationPlot
from spad23.gui_v2.state import SharedState

COL_LAMBDA, COL_X, COL_SIGMA, COL_STATUS = range(4)


class CalibPanel(QtWidgets.QGroupBox):
    """
    Tune the laser, type its wavelength into a table row and press Capture: the
    panel skips one frame, sums the next N live frames, and fits the
    center-column profile to find the peak y.  Compute fits lambda = a + b*y.
    """

    status = QtCore.pyqtSignal(str)
    calibration_saved = QtCore.pyqtSignal(str)

    def __init__(self, state: SharedState, live_running: Callable[[], bool], parent=None):
        super().__init__("Wavelength calibration", parent)
        self.state = state
        self._live_running = live_running
        self._points: List[Optional[CalibPoint]] = [None] * CALIB_DEFAULT_ROWS
        self._capture_row: Optional[int] = None
        self._capture_skip = 0
        self._capture_n = 0
        self._capture_acc: Optional[np.ndarray] = None
        self._capture_overflow = False
        self._build_ui()
        self.state.calibration_changed.connect(self._on_calibration_changed)

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        self.table = QtWidgets.QTableWidget(CALIB_DEFAULT_ROWS, 4)
        self.table.setHorizontalHeaderLabels(["λ (nm)", "y peak (µm)", "σ (µm)", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(COL_STATUS, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setMinimumWidth(430)
        for r in range(CALIB_DEFAULT_ROWS):
            self._init_row(r)
        self.table.selectRow(0)
        self.table.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.table, 1)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        self.btn_add = QtWidgets.QPushButton("Add row")
        self.btn_remove = QtWidgets.QPushButton("Remove row")
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(1, 200)
        self.spin_frames.setValue(CALIB_CAPTURE_FRAMES)
        self.btn_capture = QtWidgets.QPushButton("Capture → row")
        self.btn_capture.setToolTip("Sum the next N live frames and fit the center-column peak for the selected table row.")
        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_save = QtWidgets.QPushButton("Save…")
        self.btn_load = QtWidgets.QPushButton("Load…")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        for b in (self.btn_add, self.btn_remove, self.btn_capture, self.btn_compute, self.btn_save,
                  self.btn_load, self.btn_clear):
            b.setMinimumWidth(BTN_W - 30)
        grid.addWidget(self.btn_add, 0, 0)
        grid.addWidget(self.btn_remove, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Frames"), 0, 2)
        grid.addWidget(self.spin_frames, 0, 3)
        grid.addWidget(self.btn_capture, 1, 0, 1, 2)
        grid.addWidget(self.btn_compute, 1, 2, 1, 2)
        grid.addWidget(self.btn_save, 2, 0)
        grid.addWidget(self.btn_load, 2, 1)
        grid.addWidget(self.btn_clear, 2, 2, 1, 2)
        left.addLayout(grid)

        self.lbl_result = QtWidgets.QLabel("Not calibrated: the spectral axis is shown in µm.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setMinimumWidth(430)
        left.addWidget(self.lbl_result)

        self.plot = CalibrationPlot()
        root.addWidget(self.plot, 1)

        self.btn_add.clicked.connect(self._on_add_row)
        self.btn_remove.clicked.connect(self._on_remove_row)
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_clear.clicked.connect(self._on_clear)

    def _init_row(self, r: int, lam: Optional[float] = None) -> None:
        self.table.blockSignals(True)
        item = QtWidgets.QTableWidgetItem("" if lam is None else f"{lam:g}")
        self.table.setItem(r, COL_LAMBDA, item)
        for c in (COL_X, COL_SIGMA, COL_STATUS):
            it = QtWidgets.QTableWidgetItem("")
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, c, it)
        self.table.blockSignals(False)

    def _row_lambda(self, r: int) -> float:
        item = self.table.item(r, COL_LAMBDA)
        try:
            return float(item.text()) if item is not None and item.text().strip() else float("nan")
        except ValueError:
            return float("nan")

    def _set_row(self, r: int, pt: Optional[CalibPoint], status: str = "") -> None:
        self.table.blockSignals(True)
        self.table.item(r, COL_X).setText("" if pt is None else f"{pt.x_um:.2f}")
        self.table.item(r, COL_SIGMA).setText("" if pt is None or not np.isfinite(pt.sigma_um) else f"{pt.sigma_um:.1f}")
        self.table.item(r, COL_STATUS).setText(status)
        self.table.blockSignals(False)

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else 0

    def _refresh_plot(self) -> None:
        pts = []
        for r, pt in enumerate(self._points):
            if pt is not None:
                lam = self._row_lambda(r)
                pt = CalibPoint(lam, pt.x_um, pt.sigma_um, pt.method, pt.flags, pt.profile, pt.captured_at)
            pts.append(pt)
        self.plot.set_points(pts, self.state.calibration)

    # ---------------------------------------------------------- capture
    def on_frame(self, snap: CountSnapshot) -> None:
        if self._capture_row is None:
            return
        if self._capture_skip > 0:
            self._capture_skip -= 1
            return
        counts = snap.pixel_counts.astype(np.int64)
        self._capture_acc = counts if self._capture_acc is None else self._capture_acc + counts
        self._capture_n += 1
        if snap.warnings or snap.overload_ratio >= 1.0:
            self._capture_overflow = True
        target = self.spin_frames.value()
        r = self._capture_row
        self._set_row(r, self._points[r], f"capturing {self._capture_n}/{target}")
        if self._capture_n >= target:
            self._finish_capture()

    def _on_capture(self) -> None:
        if not self._live_running():
            self.status.emit("Start live counts first; Capture uses the live frames.")
            return
        r = self._selected_row()
        self._capture_row = r
        self._capture_skip = 1          # the frame in flight may straddle the laser tuning
        self._capture_n = 0
        self._capture_acc = None
        self._capture_overflow = False
        self.btn_capture.setEnabled(False)
        self._set_row(r, self._points[r], "waiting for frames…")

    def _finish_capture(self) -> None:
        r = self._capture_row
        acc = self._capture_acc
        self._capture_row = None
        self.btn_capture.setEnabled(True)
        if r is None or acc is None:
            return
        profile = column_profile(acc.astype(np.float64), CENTER_X_UM, self.state.excluded)
        peak = locate_peak(ROW_Y_UM, profile)
        if peak is None:
            self._points[r] = None
            self._set_row(r, None, "no peak found")
            self.status.emit(f"Entry {r + 1}: no peak found in the center column.")
            self._refresh_plot()
            return
        flags = list(peak.flags)
        if self._capture_overflow:
            flags.append("overflow during capture")
        pt = CalibPoint(
            wavelength_nm=self._row_lambda(r),
            x_um=peak.center,
            sigma_um=peak.sigma,
            method=peak.method,
            flags=tuple(flags),
            profile=tuple(float(v) for v in profile),
            captured_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._points[r] = pt
        self._set_row(r, pt, f"{pt.method}" + (f"; {', '.join(flags)}" if flags else "; ok"))
        self.status.emit(f"Entry {r + 1}: peak at y = {pt.x_um:.2f} µm (σ = {pt.sigma_um:.1f} µm).")
        if r + 1 < self.table.rowCount():
            self.table.selectRow(r + 1)
        self._refresh_plot()

    def cancel_capture(self) -> None:
        if self._capture_row is not None:
            r = self._capture_row
            self._capture_row = None
            self.btn_capture.setEnabled(True)
            self._set_row(r, self._points[r], "capture cancelled")

    # ---------------------------------------------------------- compute
    def _on_compute(self) -> None:
        pts = []
        for r, pt in enumerate(self._points):
            lam = self._row_lambda(r)
            if pt is not None and np.isfinite(lam):
                pts.append(CalibPoint(lam, pt.x_um, pt.sigma_um, pt.method, pt.flags, pt.profile, pt.captured_at))
        try:
            calib = WavelengthCalibration.fit(pts)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Wavelength calibration", str(exc))
            return
        self.state.set_calibration(calib)
        self.status.emit("Wavelength calibration computed (not saved yet).")

    def _on_calibration_changed(self) -> None:
        calib = self.state.calibration
        if calib is None:
            self.lbl_result.setText("Not calibrated: the spectral axis is shown in µm.")
        else:
            text = calib.describe()
            if self.state.calibration_path:
                text += f"<br>File: {os.path.basename(self.state.calibration_path)}"
            if calib.warnings:
                text += "<br><span style='color:#b45309'>" + "<br>".join(calib.warnings) + "</span>"
            if not calib.geometry_matches:
                text += "<br><span style='color:#b91c1c'>Saved with different FLIP_X/FLIP_Y settings.</span>"
            self.lbl_result.setText(text)
        self._refresh_plot()

    # ------------------------------------------------------------ files
    def _on_save(self) -> None:
        calib = self.state.calibration
        if calib is None:
            QtWidgets.QMessageBox.information(self, "Save calibration", "Compute a calibration first.")
            return
        default = os.path.join(save_dir("calibration"), f"spad23_wavecal_{timestamp()}.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save wavelength calibration", default,
                                                        "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            save_calibration_json(path, calib)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save calibration", f"Could not save:\n{exc}")
            return
        self.state.set_calibration(calib, path)
        self.calibration_saved.emit(path)
        self.status.emit(f"Calibration saved: {path}")

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load wavelength calibration", save_dir("calibration"),
                                                        "JSON files (*.json);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str, quiet: bool = False) -> bool:
        try:
            calib = load_calibration_json(path)
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.critical(self, "Load calibration", f"Could not load:\n{exc}")
            return False
        self._fill_from(calib)
        self.state.set_calibration(calib, path)
        self.calibration_saved.emit(path)
        self.status.emit(f"Calibration loaded: {path}")
        return True

    def _fill_from(self, calib: WavelengthCalibration) -> None:
        n = max(CALIB_DEFAULT_ROWS, len(calib.points))
        self.table.setRowCount(n)
        self._points = [None] * n
        for r in range(n):
            pt = calib.points[r] if r < len(calib.points) else None
            self._init_row(r, None if pt is None else pt.wavelength_nm)
            self._points[r] = pt
            if pt is not None:
                self._set_row(r, pt, pt.method + (f"; {', '.join(pt.flags)}" if pt.flags else "; loaded"))

    def _on_clear(self) -> None:
        self.cancel_capture()
        self.table.setRowCount(CALIB_DEFAULT_ROWS)
        self._points = [None] * CALIB_DEFAULT_ROWS
        for r in range(CALIB_DEFAULT_ROWS):
            self._init_row(r)
        self.state.set_calibration(None)

    def _on_add_row(self) -> None:
        r = self.table.rowCount()
        self.table.setRowCount(r + 1)
        self._init_row(r)
        self._points.append(None)

    def _on_remove_row(self) -> None:
        if self.table.rowCount() <= 2:
            return
        r = self._selected_row()
        if self._capture_row is not None:
            self.cancel_capture()
        self.table.removeRow(r)
        del self._points[r]
        self._refresh_plot()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == COL_LAMBDA:
            self._refresh_plot()

    def sync_controls(self, live: bool) -> None:
        self.btn_capture.setEnabled(live and self._capture_row is None)
        if not live:
            self.cancel_capture()
