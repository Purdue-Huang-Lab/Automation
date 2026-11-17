"""Import the PixeLink camera module on Windows builds without WMIC.

``pixelinkWrapper`` checks the PxLAPI40.dll version at import time by running
``wmic datafile ... get version``.  Windows 11 (24H2+) no longer ships WMIC, so the
import fails with ``FileNotFoundError`` although the PixeLink SDK is installed.

``load_camera_module()`` imports ``measurements.pump_probe_gate_dep.camera_simple``
(the old GUI's camera code, used unchanged) while ``subprocess.check_output``
answers that one ``wmic`` call from the DLL's version resource (ctypes,
version.dll).  Every other call goes to the real ``check_output``.
Alternative fix on the PC itself: install the "WMIC" optional Windows feature.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
from typing import Optional, Tuple


def _file_version(path: str) -> str:
    """FileVersion (a.b.c.d) from the version resource of ``path``."""
    ver = ctypes.WinDLL("version")
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        raise OSError(f"no version resource in {path}")
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(path, 0, size, buf):
        raise OSError(f"cannot read the version of {path}")
    ptr = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
        raise OSError(f"no fixed version info in {path}")

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint32) for n in (
            "dwSignature", "dwStrucVersion", "dwFileVersionMS", "dwFileVersionLS",
            "dwProductVersionMS", "dwProductVersionLS", "dwFileFlagsMask", "dwFileFlags",
            "dwFileOS", "dwFileType", "dwFileSubtype", "dwFileDateMS", "dwFileDateLS")]

    info = ctypes.cast(ptr, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    ms, ls = info.dwFileVersionMS, info.dwFileVersionLS
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def _make_shim(orig):
    def check_output(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).lower() == "wmic":
            m = re.search(r"name='(.*)'", " ".join(str(c) for c in cmd))
            if m:
                path = m.group(1).replace("\\\\", "\\")
                try:
                    return ("Version  \r\n" + _file_version(path) + "  \r\n").encode()
                except Exception:
                    pass
        return orig(cmd, *args, **kwargs)
    return check_output


_MODULE = None
_ERROR: Optional[str] = None


def load_camera_module() -> Tuple[Optional[object], Optional[str]]:
    """(camera_simple module, None) or (None, error text).  Imported once."""
    global _MODULE, _ERROR
    if _MODULE is not None or _ERROR is not None:
        return _MODULE, _ERROR
    orig = subprocess.check_output
    subprocess.check_output = _make_shim(orig)
    try:
        from measurements.pump_probe_gate_dep import camera_simple
        _MODULE = camera_simple
    except Exception as exc:
        _ERROR = f"PixeLink camera module unavailable: {type(exc).__name__}: {exc}"
    finally:
        subprocess.check_output = orig
    return _MODULE, _ERROR
