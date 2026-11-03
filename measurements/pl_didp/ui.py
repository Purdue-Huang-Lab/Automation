"""Display scaling for the dI/dP viewer.

Without Qt's high-DPI scaling, Qt 5 on Windows enlarges only the fonts at
125 % / 150 % magnification while every pixel size (fixed widths, plot axes,
colour bars) stays at 100 %, so text is clipped or overlaps.  ``enable_high_dpi``
makes Qt scale everything by the exact (fractional) factor; ``px`` sizes the
fixed widths from the font, so a larger Windows text size does not clip them.
"""

from PyQt5 import QtCore, QtGui, QtWidgets

UI_FONT = ("Segoe UI", 9)   # the Windows UI font (Qt 5 defaults to MS Shell Dlg 2, 8 pt)
MIN_PT = 9                  # smallest text in plot titles
_REF_HEIGHT = 13.0          # font height (px) the fixed widths were chosen for: MS Shell Dlg 2 8.25 pt at 96 dpi


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


def px(n: float) -> int:
    """A width in pixels at the reference font, scaled to the current font."""
    return int(round(float(n) * font_scale()))
