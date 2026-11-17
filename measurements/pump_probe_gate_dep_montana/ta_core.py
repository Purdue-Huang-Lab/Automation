"""Transient-absorption (pump-probe) core of the Montana GUI -- no Qt.

Everything here is the TA logic of ``measurements/pump_probe_gate_dep/gui_integrated.py``
moved into plain functions, unchanged:

* acquisition of one delay point (``_GateSweepThread._run_delay_scan`` inner loop):
  ``avg_times`` captures of ``n_frames`` frames, each split into pump ON / OFF by the
  reference ROI (``camera_simple.analyze_frames``), combined as frame-count-weighted sums;
* signal = -1000 * log10((ON + 1e-6) / (OFF + 1e-6));
* delay <-> stage position: mm = t0_mm + ps * c / 4 (double-pass retroreflector);
* delay lists (Simple and Multi-range, with the per-delay averaging map);
* the LabVIEW / ``TAM code new.py`` compatible ``*para.txt`` + ``*sum.txt`` files and
  their names (parsed by ``measurements/ta_gate_power_viewer.py``).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from measurements.pump_probe_gate_dep_montana.pixelink_compat import load_camera_module

SPEED_OF_LIGHT_MM_PS = 0.299792458  # mm/ps
DELAY_PASS_FACTOR = 4.0  # time delay (ps) = 4 x distance (mm) / c  (as the old DelayStage)
STAGE_SETTLE_S = 0.5  # sleep after every delay-stage move (old scan loops)


# ----------------------------------------------------------------------------- pump ON / OFF split
# Verbatim copy of measurements/pump_probe_gate_dep/camera_simple.analyze_frames, used only when
# that module cannot be imported (no PixeLink SDK: simulated camera).  With the SDK the original
# function is used.
def _analyze_frames_copy(frames, roi_x, roi_y, roi_w, roi_h, verbose=False):
    """
    Analyze captured frames and classify as pump on/off.

    Args:
        frames: List of numpy arrays (captured frames)
        roi_x, roi_y, roi_w, roi_h: Region of interest coordinates
        verbose: If True, print detailed analysis info

    Returns:
        dict with analysis results
    """
    if isinstance(frames, list):
        frames_array = np.array(frames)
    else:
        frames_array = frames

    n_frames = len(frames)
    frame_h, frame_w = frames[0].shape

    x = max(0, min(roi_x, frame_w - 1))
    y = max(0, min(roi_y, frame_h - 1))
    w = min(roi_w, frame_w - x)
    h = min(roi_h, frame_h - y)

    ref_intensities = np.mean(frames_array[:, y:y+h, x:x+w], axis=(1, 2))

    avg_ref = np.mean(ref_intensities)

    pump_states = ref_intensities > avg_ref

    pump_on_indices = np.where(pump_states)[0]
    pump_off_indices = np.where(~pump_states)[0]

    avg_pump_on = None
    avg_pump_off = None

    if len(pump_on_indices) > 0:
        avg_pump_on = np.mean(frames_array[pump_on_indices], axis=0)

    if len(pump_off_indices) > 0:
        avg_pump_off = np.mean(frames_array[pump_off_indices], axis=0)

    if verbose:
        print(f"Analyzed {n_frames} frames: {len(pump_on_indices)} ON, "
              f"{len(pump_off_indices)} OFF (thr={avg_ref:.1f})")

    return {
        'avg_pump_on': avg_pump_on,
        'avg_pump_off': avg_pump_off,
        'n_pump_on': len(pump_on_indices),
        'n_pump_off': len(pump_off_indices),
        'threshold': avg_ref,
        'ref_intensities': ref_intensities.tolist(),
        'pump_states': pump_states.tolist()
    }


_camera_module, _camera_error = load_camera_module()
analyze_frames = _camera_module.analyze_frames if _camera_module is not None else _analyze_frames_copy


# ----------------------------------------------------------------------------- delay <-> mm
def mm_to_ps(distance_mm: float, pass_factor: float = DELAY_PASS_FACTOR) -> float:
    return pass_factor * distance_mm / SPEED_OF_LIGHT_MM_PS


def ps_to_mm(time_ps: float, pass_factor: float = DELAY_PASS_FACTOR) -> float:
    return time_ps * SPEED_OF_LIGHT_MM_PS / pass_factor


def delay_positions_mm(delays_ps: Sequence[float], t0_mm: float,
                       pass_factor: float = DELAY_PASS_FACTOR) -> List[float]:
    return [float(t0_mm) + ps_to_mm(d, pass_factor) for d in delays_ps]


# ----------------------------------------------------------------------------- signal
def log_ratio(avg_on: np.ndarray, avg_off: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    ratio = (avg_on + epsilon) / (avg_off + epsilon)
    return -1000 * np.log10(ratio)


def crop_columns(img: Optional[np.ndarray], crop: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """Columns x0 .. x0 + w of ``img`` (the save / display region); ``crop=None``: unchanged."""
    if img is None or crop is None:
        return img
    x0, w = crop
    h, full_w = img.shape
    x0 = max(0, min(x0, full_w - 1))
    x1 = min(x0 + w, full_w)
    return img[:, x0:x1] if x1 > x0 else img


# ----------------------------------------------------------------------------- acquisition
def acquire_delay_point(camera, n_frames: int, roi: Tuple[int, int, int, int], avg_times: int,
                        is_running: Callable[[], bool],
                        on_capture: Optional[Callable[[int, int], None]] = None) -> Optional[Dict]:
    """``avg_times`` captures at the current delay, combined exactly as the old scan threads.

    Returns the old ``final_results`` dict, or None if stopped before the first capture.
    ``on_capture(avg_idx, avg_times)`` is called before every capture (progress text).
    """
    sum_on = None
    sum_off = None
    total_on = 0
    total_off = 0
    results = None
    for avg_idx in range(avg_times):
        if not is_running():
            break
        if on_capture is not None:
            on_capture(avg_idx, avg_times)
        frames = camera.capture_frames(n_frames)
        results = analyze_frames(frames, *roi)
        n_on = results['n_pump_on']
        n_off = results['n_pump_off']
        if results['avg_pump_on'] is not None and n_on > 0:
            batch_sum = results['avg_pump_on'].astype(np.float64) * n_on
            if sum_on is None:
                sum_on = batch_sum
            else:
                sum_on += batch_sum
            total_on += n_on
        if results['avg_pump_off'] is not None and n_off > 0:
            batch_sum = results['avg_pump_off'].astype(np.float64) * n_off
            if sum_off is None:
                sum_off = batch_sum
            else:
                sum_off += batch_sum
            total_off += n_off
    if results is None:
        return None
    final_on = sum_on / total_on if sum_on is not None else None
    final_off = sum_off / total_off if sum_off is not None else None
    return {
        'avg_pump_on': final_on, 'avg_pump_off': final_off,
        'n_pump_on': total_on, 'n_pump_off': total_off,
        'threshold': results['threshold'],
        'ref_intensities': results['ref_intensities'],
    }


# ----------------------------------------------------------------------------- delay lists
def parse_delay_text(text: str) -> List[float]:
    """Comma-separated delays (ps), as the old 'Generated Time Delays' box."""
    return [float(p.strip()) for p in text.split(',') if p.strip()]


def simple_delays(start_ps: float, n_steps: int, increment_ps: float, t0_offset_ps: float = 0.0) -> List[float]:
    if n_steps <= 0:
        raise ValueError("Number of steps must be positive!")
    positions = [start_ps + i * increment_ps for i in range(n_steps)]
    return [p + t0_offset_ps for p in positions]


def multi_range_delays(ranges: Sequence[Tuple[float, float, float, int]], t0_offset_ps: float = 0.0
                       ) -> Tuple[List[float], Dict[float, int], List[str]]:
    """(delays, {round(delay, 2): avg}, info lines) of the Multi-Range rows (start, end, step, avg)."""
    all_positions: List[float] = []
    range_info: List[str] = []
    avg_map: Dict[float, int] = {}
    for i, (s, e, st, av) in enumerate(ranges):
        s = float(s); e = float(e); st = float(st); av = int(av)
        if st <= 0:
            raise ValueError(f"Range {i+1}: Step must be positive!")
        n_pts = int(abs(e - s) / st) + 1
        if s <= e:
            positions = [s + j * st for j in range(n_pts) if s + j * st <= e]
        else:
            positions = [s - j * st for j in range(n_pts) if s - j * st >= e]
        positions_with_offset = [p + t0_offset_ps for p in positions]
        if all_positions and positions_with_offset:
            if abs(positions_with_offset[0] - all_positions[-1]) < 0.001:
                positions_with_offset = positions_with_offset[1:]
                positions = positions[1:]
        for pos in positions_with_offset:
            avg_map[round(pos, 2)] = av
        all_positions.extend(positions_with_offset)
        range_info.append(f"  R{i+1}: {s:+.2f} to {e:+.2f}, step={st}, avg={av} -> "
                          f"{len(positions_with_offset)} pts")
    if not all_positions:
        raise ValueError("No valid ranges!")
    return all_positions, avg_map, range_info


def avg_for_delay(delay_ps: float, avg_map: Dict[float, int], default_avg: int) -> int:
    return avg_map.get(round(delay_ps, 2), default_avg)


# ----------------------------------------------------------------------------- file names
def fmt_gate_v(v: float) -> str:
    """Old ``_save_gate_step._fmt_v``: +1.25 -> 'p1d250', -0.5 -> 'n0d500'."""
    s = f"{abs(v):.3f}".replace('.', 'd')
    return ('p' if v >= 0 else 'n') + s


def angle_tag(angle_deg: Optional[float]) -> str:
    if angle_deg is None:
        return ""
    astr = f"{angle_deg:.1f}".replace('.', 'p').replace('-', 'n')
    return f"_ND{astr}deg"


def power_tag(power_w: Optional[float], unit: str = "nW") -> str:
    """'_P12p5nW' (extra token, ignored by the old viewers); '' if the power is unknown."""
    if power_w is None or not math.isfinite(float(power_w)):
        return ""
    scale = {"pW": 1e-12, "nW": 1e-9, "uW": 1e-6, "µW": 1e-6, "mW": 1e-3, "W": 1.0}.get(unit, 1e-9)
    val = f"{float(power_w) / scale:.4g}".replace('.', 'p').replace('-', 'm')
    return f"_P{val}{unit.replace('µ', 'u')}"


def scan_prefix(base: str, vf: Optional[float], vb: Optional[float], angle: Optional[float],
                pol_tag: Optional[str], timestamp: str, power_w: Optional[float] = None,
                power_unit: str = "nW") -> str:
    """``{base}_Vf..V[Vb..V]_ND..deg[_P..][_LH]_{timestamp}`` (old ``_save_gate_step`` order)."""
    vf_tag = f"Vf{fmt_gate_v(vf)}V" if vf is not None else ""
    vb_tag = f"Vb{fmt_gate_v(vb)}V" if vb is not None else ""
    gate = f"_{vf_tag}{vb_tag}" if (vf_tag or vb_tag) else ""
    pol = f"_{pol_tag}" if pol_tag else ""
    return f"{base}{gate}{angle_tag(angle)}{power_tag(power_w, power_unit)}{pol}_{timestamp}"


# ----------------------------------------------------------------------------- para / sum files
def write_para(path: str, delays: Sequence[float], avg_used: Sequence[float], pixel_size: float,
               crop_w: int, crop_h: int, t0_ps: float, angle_deg: Optional[float], n_frames: int) -> None:
    """The LabVIEW-compatible 23-line ``*para.txt`` (byte-identical to the old GUI)."""
    with open(path, 'w') as f:
        f.write('\t'.join([f'{d:.6f}' for d in delays]) + '\n')
        f.write('\t'.join([f'{a:.6f}' for a in avg_used]) + '\n')
        f.write('0\n')
        f.write(f' {1.0:.6f} \n')
        f.write(f' {pixel_size:.6f} \n')
        f.write(f' {crop_w} \n')
        f.write(f' {pixel_size:.6f} \n')
        f.write(f' {crop_h} \n')
        f.write(f' {0.0:.6f} \n')
        f.write(f' {t0_ps:.6f}\n')
        f.write('0\n0\n0\n0\n0 \n')
        f.write(f' {0.0:.6f}\n')
        f.write(f' {angle_deg if angle_deg is not None else 0.0:.6f}\n')
        f.write(f'{1.000000}\n')
        f.write(f'{0}\n')
        f.write(f' {n_frames} \n')
        f.write(f' 1\n')


def write_sum(path: str, delay_results: Sequence[Dict], crop: Optional[Tuple[int, int]]) -> None:
    """Log ratio of every delay (cropped, transposed), rows stacked: the old ``*sum.txt``."""
    with open(path, 'w') as f:
        for r in delay_results:
            res = r['results']
            on = crop_columns(res['avg_pump_on'], crop)
            off = crop_columns(res['avg_pump_off'], crop)
            if on is not None and off is not None:
                lr = log_ratio(on, off).T
                for row in lr:
                    f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')


def save_scan(save_dir: str, prefix: str, delay_results: Sequence[Dict], *, pixel_size: float,
              crop: Optional[Tuple[int, int]], camera_width: int, crop_h: int, t0_ps: float,
              angle_deg: Optional[float], n_frames: int) -> Tuple[str, str]:
    """Write ``{prefix}para.txt`` + ``{prefix}sum.txt``; returns both paths."""
    delays = [r['time_delay_ps'] for r in delay_results]
    avg_used = [r.get('avg_times_used', 1) for r in delay_results]
    crop_w = crop[1] if crop else (camera_width or 480)
    para_path = os.path.join(save_dir, f"{prefix}para.txt")
    sum_path = os.path.join(save_dir, f"{prefix}sum.txt")
    write_para(para_path, delays, avg_used, pixel_size, crop_w, crop_h, t0_ps, angle_deg, n_frames)
    write_sum(sum_path, delay_results, crop)
    return para_path, sum_path
