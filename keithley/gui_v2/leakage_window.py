import csv

from PyQt5 import QtCore, QtWidgets

from .config import (
    BTN_W,
    LEAKAGE_AUTOUPDATE_S,
    LEAKAGE_MAX_BINS_WARN,
    LEAKAGE_WIN_SIZE,
)
from .plots import LeakageIVPlot


class LeakageWindow(QtWidgets.QWidget):
    """Separate top-level window holding both gates' leakage I-V plots.

    Closing it only hides it: recording continues and the main window's
    "I-V Window" button brings it back.
    """

    def __init__(self, main_widget):
        super().__init__(main_widget, QtCore.Qt.Window)
        self._main = main_widget

        self.setWindowTitle("Leakage I-V (mean +/- std per voltage bin)")
        self.resize(*LEAKAGE_WIN_SIZE)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        vbox.addLayout(top)

        self.modeLbl = QtWidgets.QLabel("MODE: POLL ONLY")

        self.updateBtn = QtWidgets.QPushButton("Update Plots"); self.updateBtn.setFixedWidth(BTN_W)
        self.autoChk = QtWidgets.QCheckBox("Auto-update every")
        self.autoSpin = QtWidgets.QSpinBox()
        self.autoSpin.setRange(1, 600); self.autoSpin.setValue(LEAKAGE_AUTOUPDATE_S); self.autoSpin.setSuffix(" s")
        self.clearBtn = QtWidgets.QPushButton("Clear I-V"); self.clearBtn.setFixedWidth(BTN_W)
        self.saveBtn = QtWidgets.QPushButton("Save I-V..."); self.saveBtn.setFixedWidth(BTN_W)

        top.addWidget(self.modeLbl)
        top.addSpacing(16)
        top.addWidget(self.updateBtn)
        top.addSpacing(8)
        top.addWidget(self.autoChk); top.addWidget(self.autoSpin)
        top.addStretch()
        top.addWidget(self.clearBtn); top.addWidget(self.saveBtn)

        # Plots
        plots = QtWidgets.QHBoxLayout()
        vbox.addLayout(plots, 1)

        self.plotA = LeakageIVPlot(self)
        self.plotB = LeakageIVPlot(self)
        self.headerA = QtWidgets.QLabel("Source A")
        self.headerB = QtWidgets.QLabel("Source B")

        for header, plot in ((self.headerA, self.plotA), (self.headerB, self.plotB)):
            header.setStyleSheet("font-weight: 600;")
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(header)
            col.addWidget(plot, 1)
            plots.addLayout(col, 1)

        # Status bar
        self._status_bar = QtWidgets.QStatusBar(self)
        vbox.addWidget(self._status_bar)

        # Auto-update timer
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.timeout.connect(self.update_plots)

        # Signals
        self.updateBtn.clicked.connect(self.on_update_clicked)
        self.clearBtn.clicked.connect(self.on_clear)
        self.saveBtn.clicked.connect(self.on_save)
        self.autoChk.toggled.connect(self.on_auto_toggled)
        self.autoSpin.valueChanged.connect(self.on_auto_interval_changed)

        self.update_plots()

    def statusBar(self) -> QtWidgets.QStatusBar:
        return self._status_bar

    # ---- mode indicator ----

    def set_mode(self, text: str, css: str) -> None:
        self.modeLbl.setText(text)
        self.modeLbl.setStyleSheet(css)

    # ---- plotting ----

    def _pairs(self):
        return (("A", self._main.panelA, self.plotA, self.headerA),
                ("B", self._main.panelB, self.plotB, self.headerB))

    def update_plots(self) -> None:
        max_bins = 0
        for tag, panel, plot, header in self._pairs():
            v, mean, std, _cnt = panel.leak.arrays()
            plot.set_data(v, mean, std)
            header.setText(
                f"Source {tag} - {panel.leak.n_samples} pts, {panel.leak.n_bins} bins"
            )
            max_bins = max(max_bins, panel.leak.n_bins)

        msg = f"I-V updated ({max_bins} bins)"
        if max_bins > LEAKAGE_MAX_BINS_WARN:
            msg += "  [many bins: consider coarser V bin decimals]"
        self.statusBar().showMessage(msg)

    def on_update_clicked(self) -> None:
        self.update_plots()

    def on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._auto_timer.start(int(self.autoSpin.value()) * 1000)
            self.update_plots()
        else:
            self._auto_timer.stop()

    def on_auto_interval_changed(self, s: int) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.setInterval(int(s) * 1000)

    # ---- clear / save ----

    def on_clear(self) -> None:
        total = sum(p.leak.n_samples for _t, p, _pl, _h in self._pairs())
        if total:
            ans = QtWidgets.QMessageBox.question(
                self, "Clear I-V",
                f"Discard {total} recorded sample(s) from both gates?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        for _tag, panel, _plot, _header in self._pairs():
            panel.leak.clear()
        self.update_plots()
        self.statusBar().showMessage("I-V data cleared.")

    def on_save(self) -> None:
        total = sum(p.leak.n_samples for _t, p, _pl, _h in self._pairs())
        if not total:
            QtWidgets.QMessageBox.information(self, "Save I-V", "No leakage data recorded yet.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save binned I-V", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["panel", "V(V)", "I_mean(A)", "I_std(A)", "n"])
                for tag, panel, _plot, _header in self._pairs():
                    for v, mean, std, n in panel.leak.rows():
                        w.writerow([tag, f"{v:.6f}", f"{mean:.9e}", f"{std:.9e}", n])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Cannot write file:\n{e}")
            return

        self.statusBar().showMessage(f"Saved -> {path}")
