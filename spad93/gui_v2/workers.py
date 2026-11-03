"""QThread workers for the SPAD93 GUI; every device call runs here, never in the GUI thread.

Each hardware worker opens and closes its own connection (as the spad23
workers do); the DeviceController makes sure only one runs at a time.
"""

import time
from typing import Callable, Dict

from PyQt5 import QtCore

from spad93.spad93_tcspc_wrapper import MAX_BINS, Spad93TcspcClient
from spad93.spad93_wrapper import (
    PIXEL_LIMIT_MCPS,
    Spad93CountClient,
    Spad93Error,
    Spad93OverflowError,
    Spad93ProtocolError,
    Spad93ShutdownError,
)


class ConnectWorker(QtCore.QThread):
    """Open a connection, read the banner and the TDC status, close again."""

    done = QtCore.pyqtSignal(bool, str, str)  # ok, banner, TDC status or error

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)

    def run(self) -> None:
        try:
            with Spad93CountClient(host=self.host, port=self.port, timeout_s=3.0) as client:
                banner = client.server_banner
                try:
                    tdc = client.check_tdc_calibration()
                except Spad93Error as exc:
                    tdc = f"TDC status unavailable: {exc}"
            self.done.emit(True, banner, tdc)
        except Exception as exc:
            self.done.emit(False, "", str(exc))


class CountLiveWorker(QtCore.QThread):
    """Continuous count snapshots (``I,<ms>``) until ``stop()``."""

    frame_ready = QtCore.pyqtSignal(object)  # CountSnapshot
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, host: str, port: int, integration_ms: int, min_interval_s: float, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self._integration_ms = max(10, int(integration_ms))
        self._min_interval_s = float(min_interval_s)
        self._stop = False

    def set_integration_ms(self, ms: int) -> None:
        self._integration_ms = max(10, int(ms))

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        client = Spad93CountClient(host=self.host, port=self.port, pixel_limit_mcps=PIXEL_LIMIT_MCPS)
        errors = 0
        try:
            while not self._stop:
                loop_start = time.monotonic()
                try:
                    if not client.is_connected:
                        client.connect()
                    snap = client.get_counts_snapshot(self._integration_ms)
                    errors = 0
                    self.frame_ready.emit(snap)
                except Spad93OverflowError as exc:
                    if exc.snapshot is not None:
                        self.frame_ready.emit(exc.snapshot)
                    self.status.emit(f"Overflow: {exc}")
                except Spad93ShutdownError as exc:
                    self.failed.emit(str(exc))
                    return
                except Spad93Error as exc:
                    errors += 1
                    client.close()
                    if self._stop:
                        return
                    if errors >= 5:
                        self.failed.emit(f"Live counts stopped after {errors} errors: {exc}")
                        return
                    self.status.emit(f"Count read failed ({errors}/5): {exc}")
                    time.sleep(0.5)
                if (sleep_s := self._min_interval_s - (time.monotonic() - loop_start)) > 0:
                    time.sleep(sleep_s)
        except Exception as exc:
            self.failed.emit(f"Live counts crashed: {exc}")
        finally:
            client.close()


class CommandWorker(QtCore.QThread):
    """Run one blocking device command, e.g. TDC calibrate + align or SI."""

    done = QtCore.pyqtSignal(str, bool, str)  # name, ok, message

    def __init__(self, host: str, port: int, name: str, fn: Callable[[Spad93TcspcClient], str], parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self.name = name
        self._fn = fn

    def run(self) -> None:
        try:
            with Spad93TcspcClient(host=self.host, port=self.port) as client:
                msg = self._fn(client)
            self.done.emit(self.name, True, msg)
        except Exception as exc:
            self.done.emit(self.name, False, str(exc))


class TrplAcquireWorker(QtCore.QThread):
    """One ``S,<ms>`` acquisition; ``stop()`` sends the vendor cancel."""

    progress = QtCore.pyqtSignal(object)  # TrplSnapshot
    done = QtCore.pyqtSignal(object, str)  # final TrplSnapshot (or None), error text ("" when ok)

    def __init__(self, host: str, port: int, measurement_ms: int, bin_width_ps: int, update_s: float,
                 cancel_wait_s: float, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self.measurement_ms = int(measurement_ms)
        self.bin_width_ps = int(bin_width_ps)
        self.update_s = float(update_s)
        self.cancel_wait_s = float(cancel_wait_s)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        final, err = None, ""
        try:
            with Spad93TcspcClient(host=self.host, port=self.port) as client:
                final = client.acquire_trpl_stream(
                    measurement_ms=self.measurement_ms,
                    bin_width_ps=self.bin_width_ps,
                    on_update=self.progress.emit,
                    update_interval_s=self.update_s,
                    should_stop=lambda: self._stop,
                    max_bins=MAX_BINS,
                    cancel_wait_s=self.cancel_wait_s,
                )
        except Spad93ProtocolError as exc:
            final, err = exc.snapshot, str(exc)
        except Exception as exc:
            err = str(exc)
        self.done.emit(final, err)


class FitAllWorker(QtCore.QThread):
    """Run a ``spad93_analysis.fit_all_*`` function with progress and cancel."""

    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(int, object, str)  # job id, FitAllResult or None, error

    def __init__(self, job_id: int, fn: Callable, kwargs: Dict, parent=None):
        super().__init__(parent)
        self.job_id = int(job_id)
        self._fn = fn
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            res = self._fn(progress=self.progress.emit, cancelled=lambda: self._cancel, **self._kwargs)
            self.done.emit(self.job_id, res, "")
        except Exception as exc:
            self.done.emit(self.job_id, None, str(exc))


class CallWorker(QtCore.QThread):
    """Run a plain callable off the GUI thread (used for saving large files)."""

    done = QtCore.pyqtSignal(bool, str)  # ok, result text or error

    def __init__(self, fn: Callable[[], str], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(True, str(self._fn()))
        except Exception as exc:
            self.done.emit(False, str(exc))
