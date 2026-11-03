"""Display scaling of the PL + TRPL v2 GUI (same approach as measurements/pl_didp/ui.py).

Without Qt's high-DPI scaling, Qt 5 on Windows sizes fonts from the primary screen's magnification
only, while every pixel size (fixed widths, plot axes) stays at 100 %: text is clipped at 125 / 150 %
and too small on a screen magnified more than the primary one.  ``enable_high_dpi`` makes Qt scale
everything by each screen's exact (fractional) factor, also when the window moves to another screen.
``scale_fixed_sizes`` grows the fixed widths / heights with the UI font, so the Windows UI font
(9 pt Segoe UI instead of Qt's 8 pt default) or a larger Windows text size does not clip them.
"""

from PyQt5 import QtCore, QtGui, QtWidgets

UI_FONT = ("Segoe UI", 9)  # the Windows UI font (Qt 5 defaults to MS Shell Dlg 2, 8 pt)
_REF_HEIGHT = 13.0  # font height (px) the fixed sizes were chosen for: MS Shell Dlg 2 8.25 pt at 96 dpi
_QWIDGETSIZE_MAX = (1 << 24) - 1


def enable_high_dpi() -> None:
    """Call before the QApplication is created."""
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtGui.QGuiApplication, "setHighDpiScaleFactorRoundingPolicy"):
        # 125 % stays 1.25 (the default policy rounds it to 1 and scales the fonts alone again)
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


def apply_ui_font(app: QtWidgets.QApplication) -> None:
    family, size = UI_FONT
    if family in QtGui.QFontDatabase().families():
        app.setFont(QtGui.QFont(family, size))


def font_scale() -> float:
    app = QtWidgets.QApplication.instance()
    font = app.font() if app is not None else QtGui.QFont()
    return max(1.0, QtGui.QFontMetricsF(font).height() / _REF_HEIGHT)


def scale_fixed_sizes(root: QtWidgets.QWidget, scale: float = None) -> None:
    """Grow every fixed width / height below ``root`` with the UI font (once, after the UI is built)."""
    s = font_scale() if scale is None else float(scale)
    if s <= 1.0001:
        return
    for w in [root] + root.findChildren(QtWidgets.QWidget):
        if w.property("_ui_scaled"):
            continue
        w.setProperty("_ui_scaled", True)
        if 0 < w.minimumWidth() == w.maximumWidth() < _QWIDGETSIZE_MAX:
            w.setFixedWidth(int(round(w.minimumWidth() * s)))
        if 0 < w.minimumHeight() == w.maximumHeight() < _QWIDGETSIZE_MAX:
            w.setFixedHeight(int(round(w.minimumHeight() * s)))
