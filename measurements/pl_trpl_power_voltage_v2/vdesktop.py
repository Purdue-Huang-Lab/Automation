"""Windows virtual desktops: keep a window on the desktop the user is looking at.

Windows hides ("cloaks") a window when the user switches to another virtual desktop
(Win+Ctrl+arrow, Task View).  There is no documented way to pin a window to all
desktops, so ``DesktopFollower`` checks the window's DWM cloak state and, once it is
cloaked by the shell, moves it to the current desktop with the documented
IVirtualDesktopManager::MoveWindowToDesktop (raw ctypes COM, no extra packages).

The current desktop's id comes from the foreground window (it is on the desktop just
switched to) or, failing that, from Explorer's ``CurrentVirtualDesktop`` registry value.
If neither works the window is hidden and shown again, which opens it on the current
desktop.  Everything is a no-op off Windows or when the COM object is unavailable.
"""

import ctypes
import sys
import time
import uuid
from typing import Optional

from PyQt5 import QtCore, QtWidgets

_WIN = sys.platform == "win32"
if _WIN:
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    def _guid(text: str) -> "_GUID":
        g = _GUID()
        ctypes.memmove(ctypes.byref(g), uuid.UUID(text).bytes_le, 16)
        return g

    def _guid_bytes(g: "_GUID") -> bytes:
        return ctypes.string_at(ctypes.byref(g), 16)

CLSID_VIRTUAL_DESKTOP_MANAGER = "aa509086-5ca9-4c25-8f95-589d3c07b48a"
IID_IVIRTUAL_DESKTOP_MANAGER = "a5cd92ff-29be-454c-8d04-d82879fb3f1b"
DWMWA_CLOAKED = 14
DWM_CLOAKED_SHELL = 0x2  # cloaked by the shell: the window is on another virtual desktop
_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\VirtualDesktops"


class VirtualDesktopManager:
    """The documented IVirtualDesktopManager (Windows 10+); ``ok`` is False if unavailable."""

    def __init__(self):
        self.ok = False
        if not _WIN:
            return
        try:
            ole32 = ctypes.OleDLL("ole32")
            try:
                ole32.CoInitializeEx(None, 0x2)  # STA; Qt usually did it already (S_FALSE)
            except OSError:
                pass  # RPC_E_CHANGED_MODE: COM is already initialized differently, which is fine
            ptr = ctypes.c_void_p()
            ole32.CoCreateInstance(ctypes.byref(_guid(CLSID_VIRTUAL_DESKTOP_MANAGER)), None, 0x1 | 0x4,
                                   ctypes.byref(_guid(IID_IVIRTUAL_DESKTOP_MANAGER)), ctypes.byref(ptr))
            vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            hr = ctypes.HRESULT
            self._is_on = ctypes.WINFUNCTYPE(hr, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(wintypes.BOOL))(vtbl[3])
            self._get_id = ctypes.WINFUNCTYPE(hr, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(_GUID))(vtbl[4])
            self._move = ctypes.WINFUNCTYPE(hr, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(_GUID))(vtbl[5])
            self._ptr = ptr  # kept for the life of the process
            self._dwm = ctypes.WinDLL("dwmapi")
            self._user32 = ctypes.WinDLL("user32")
            self._user32.GetForegroundWindow.restype = wintypes.HWND
            self.ok = True
        except Exception:
            self.ok = False

    def cloaked_by_shell(self, hwnd: int) -> bool:
        """True if Windows hides ``hwnd`` because it belongs to another virtual desktop."""
        val = wintypes.DWORD(0)
        try:
            self._dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(val),
                                            ctypes.sizeof(val))
        except Exception:
            return False
        return bool(val.value & DWM_CLOAKED_SHELL)

    def _desktop_of(self, hwnd: int) -> Optional["_GUID"]:
        g = _GUID()
        try:
            self._get_id(self._ptr, wintypes.HWND(hwnd), ctypes.byref(g))
        except OSError:
            return None
        return None if _guid_bytes(g) == bytes(16) else g

    def _on_current(self, hwnd: int) -> bool:
        b = wintypes.BOOL(0)
        try:
            self._is_on(self._ptr, wintypes.HWND(hwnd), ctypes.byref(b))
        except OSError:
            return False
        return bool(b.value)

    def current_desktop(self) -> Optional["_GUID"]:
        """Id of the desktop the user is on: the foreground window's, else Explorer's registry value."""
        fg = self._user32.GetForegroundWindow()
        if fg and self._on_current(int(fg)):
            g = self._desktop_of(int(fg))
            if g is not None:
                return g
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY) as key:
                raw, _ = winreg.QueryValueEx(key, "CurrentVirtualDesktop")
            if isinstance(raw, bytes) and len(raw) == 16 and raw != bytes(16):
                g = _GUID()
                ctypes.memmove(ctypes.byref(g), raw, 16)
                return g
        except Exception:
            pass
        return None

    def move_to_current(self, hwnd: int) -> bool:
        g = self.current_desktop()
        if g is None:
            return False
        try:
            self._move(self._ptr, wintypes.HWND(hwnd), ctypes.byref(g))
        except OSError:
            return False
        return True


_manager: Optional[VirtualDesktopManager] = None


def manager() -> Optional[VirtualDesktopManager]:
    global _manager
    if _manager is None:
        _manager = VirtualDesktopManager()
    return _manager if _manager.ok else None


class DesktopFollower(QtCore.QObject):
    """Brings ``window`` along when the user switches virtual desktops (checked every ``interval_ms``)."""

    def __init__(self, window: QtWidgets.QWidget, interval_ms: int = 400):
        super().__init__(window)
        self._w = window
        self._last_reshow = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.check)

    def start(self) -> None:
        if manager() is not None:
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def check(self) -> None:
        mgr = manager()
        w = self._w
        if mgr is None or not w.isVisible():
            return
        hwnd = int(w.winId())
        if not mgr.cloaked_by_shell(hwnd):
            return
        if mgr.move_to_current(hwnd) and not mgr.cloaked_by_shell(hwnd):
            w.raise_()
            return
        # last resort (at most every 2 s): a window shown again opens on the current desktop
        now = time.monotonic()
        if now - self._last_reshow > 2.0:
            self._last_reshow = now
            pos = w.pos()
            w.hide()
            w.move(pos)
            w.show()
            w.raise_()
