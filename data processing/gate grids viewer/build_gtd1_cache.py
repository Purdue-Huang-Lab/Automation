"""
build_gtd1_cache.py  -- build the spatial-window spectral cache for a GT-D1-style
(topWSe2/botWS2) 0.2 V-step dual-gate PL sweep.

Reads every .asc frame in a data folder, keeps a +/-SPAT_HALF spatial band around the
emission spot, and writes dual_gate_speccube_gtd1.npz. The n-D viewer and the full-collapse
calibration both read that cache. The viewer also imports build_cube() to build a cache
on-the-fly for a data folder the user picks (any folder in the same .asc format).

Uses the .asc loaders in dual_gate_asc_io.py (same folder). Run with the ROOT .venv:
  & "C:\\Users\\YouAr\\OneDrive - purdue.edu\\.venv\\Scripts\\python.exe" build_gtd1_cache.py <data_folder> [out.npz]
(default out = <data_folder>\\dual_gate_speccube_gtd1.npz)
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_gate_asc_io as m               # .asc loaders / filename parsing / spot detect

CACHE_NAME = "dual_gate_speccube_gtd1.npz"      # standard cache filename dropped next to any data folder
SPAT_HALF = 35
STEP = 0.2


def _axis_values(vals):
    """Sorted lattice of the distinct gate values measured on one axis, clustering
    near-duplicates (filename float noise) that fall within 1/4 of the typical step,
    and returning each cluster's mean. Handles ARBITRARY per-axis spacing -- it does
    NOT assume a fixed 0.2 V step (e.g. a map taken with back-step = front-step/r has
    an irregular Vb step and is rectangular in (u_n,u_D))."""
    v = np.array(sorted(float(x) for x in vals), float)
    if v.size == 0:
        return v
    uniq = np.unique(np.round(v, 6))
    tol = 0.25 * float(np.median(np.diff(uniq))) if uniq.size > 1 else 1e-3
    reps, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol:               # same vertex (steps are >> tol, so no chaining)
            cur.append(x)
        else:
            reps.append(float(np.mean(cur))); cur = [x]
    reps.append(float(np.mean(cur)))
    return np.array(reps, float)


def build_cube(data_dir, cache_path=None, step=STEP, spat_half=SPAT_HALF, progress=None,
               prev=None, wl_ref=None):
    """Build the spectral cube for one .asc dual-gate sweep folder.

    Parameters
    ----------
    data_dir : path to the folder of PL_manual_Vf<>_Vb<>.asc frames.
    cache_path : if given, np.savez_compressed the result there.
    progress : optional callback progress(done, total) called per file (for a GUI
               progress bar); may raise to abort the build.
    prev : optional previously built cache dict for the SAME folder (incremental
           update of an ongoing scan). Its wavelength axis / spatial window are reused,
           and frames it already holds that are not newer than prev["built"] are copied
           instead of re-read; only new / rewritten / previously unreadable frames are read.
    wl_ref : optional reference .asc for the wavelength axis when the frames are RAW images
             (saved without a wavelength column); default: auto-search nearby folders.

    Returns a dict with the same keys the viewer/collapse expect, plus provenance
    (source_dir, built). A uniform `step`-V lattice spans the measured Vf/Vb range;
    unmeasured vertices (incl. internal gaps for an ongoing scan) stay NaN.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.asc"))
    if not files:
        raise FileNotFoundError(f"no .asc files in {data_dir}")
    t_build = time.time()
    raw = not m._has_wl_column(m._parse_header(files[0]))   # frames saved without wavelengths

    if prev is not None:
        wl = np.asarray(prev["wl"], float)
        wl_source = str(prev["wl_source"]) if "wl_source" in prev else "frames"
        rows = np.asarray(prev["spat_rows"], int); center = int(prev["spot_center"])
        lo, hi = int(rows[0]), int(rows[-1])
        p_cube, p_ts = prev["cube"], np.asarray(prev["tstamp"], float)
        p_vf, p_vb = np.asarray(prev["vf_vals"], float), np.asarray(prev["vb_vals"], float)
        p_built = float(prev["built"]) if "built" in prev else 0.0
    else:
        wl_source = "frames"
        if raw:
            wl, ref = m.find_reference_wavelengths(data_dir, wl_ref)
            wl_source = str(ref)
        prof, bands, wl = m._detect_emission_bands(files, None, wl if raw else None)
        center = int(bands[0][0]); lo = max(0, center - spat_half); hi = center + spat_half
        rows = np.arange(lo, hi + 1)

    parsed, vfs, vbs = [], [], []
    for f in files:
        vf, vb = m._vf_vb_from_name(f.name)
        if vf is None or vb is None:
            continue
        parsed.append((f, float(vf), float(vb))); vfs.append(float(vf)); vbs.append(float(vb))
    if not parsed:
        raise ValueError(f"no Vf/Vb-tagged .asc files in {data_dir}")

    # lattice = the gate values ACTUALLY measured on each axis (per-axis step inferred),
    # then map every frame to its nearest vertex (robust to filename float noise).
    vf_vals = _axis_values(vfs)
    vb_vals = _axis_values(vbs)
    nvf, nvb, nwin, nwl = vf_vals.size, vb_vals.size, rows.size, wl.size

    cube = np.full((nvf, nvb, nwin, nwl), np.nan, np.float32)
    tstamp = np.full((nvf, nvb), np.nan)
    if prev is not None:                     # vertex-match tolerance against the old lattice
        def _tol(v):
            return 0.25 * float(np.min(np.diff(v))) if v.size > 1 else 1e-3
        tol_f, tol_b = _tol(p_vf), _tol(p_vb)
    total = len(parsed)
    for k, (f, vf, vb) in enumerate(parsed):
        i = int(np.argmin(np.abs(vf_vals - vf)))
        j = int(np.argmin(np.abs(vb_vals - vb)))
        if prev is not None:
            pi = int(np.argmin(np.abs(p_vf - vf))); pj = int(np.argmin(np.abs(p_vb - vb)))
            if (abs(p_vf[pi] - vf) <= tol_f and abs(p_vb[pj] - vb) <= tol_b
                    and np.isfinite(p_ts[pi, pj]) and f.stat().st_mtime < p_built):
                cube[i, j] = p_cube[pi, pj]; tstamp[i, j] = p_ts[pi, pj]
                if progress is not None:
                    progress(k + 1, total)
                continue
        try:
            img = m._read_frame(f, wl if raw else None)[:, 1:]
            hi2 = min(rows[-1], img.shape[1] - 1)
            sT = img[:, lo:hi2 + 1].T
            cube[i, j, :sT.shape[0], :] = sT
            tstamp[i, j] = m._header_timestamp(f, m._parse_header(f))
        except Exception:
            pass
        if progress is not None:
            progress(k + 1, total)

    data = dict(cube=cube, wl=wl, vf_vals=vf_vals, vb_vals=vb_vals, tstamp=tstamp,
                spat_rows=rows, spot_center=center, wl_source=wl_source, n_files=len(files),
                source_dir=str(data_dir), built=float(t_build))   # build START: frames written
                                                                   # during the build get re-read next time
    if cache_path is not None:
        np.savez_compressed(cache_path, **data)
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: build_gtd1_cache.py <data_folder> [out.npz]")
    data_dir = Path(sys.argv[1])
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / CACHE_NAME
    t0 = time.time()
    print("dataset", data_dir, flush=True)

    def _cb(done, total):
        if done % 250 == 0 or done == total:
            print(f"  {done}/{total}  {time.time()-t0:.0f}s", flush=True)

    data = build_cube(data_dir, cache, progress=_cb)
    print("wrote", cache, data["cube"].shape,
          f"filled {int(np.isfinite(data['tstamp']).sum())}/{data['cube'].shape[0]*data['cube'].shape[1]}  "
          f"{time.time()-t0:.0f}s", flush=True)
