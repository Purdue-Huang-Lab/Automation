"""
gtd1_full_collapse.py  -- shear-collapse (Radon) calibration of r = c_b/c_t for the
GT-D1 (topWSe2/botWS2) dual-gate PL map.

Extracts r from the whole map at once: for a trial r, form u = V_f + r*V_b for every
pixel, bin by u, and score how well the map collapses onto a 1-D function of u (ANOVA
R^2). Only at the true r do all iso-density lines line up, so the score peaks sharply.
|slope| = r for iso-n and iso-D alike, so any scalar that tracks ONE of them gives r.

IMPORTANT for GT-D1: the IX energy mixes the Stark (D) shift and the doping-induced
blue-shift / trion crossover (n), so it is NOT a clean function of one axis and does NOT
collapse (its "best r" jumps around with the fit region -- see the printout). The IX
INTENSITY, dominated by doping-induced quenching, is a clean iso-n tracker and collapses
sharply. So r is taken from the intensity collapse (integrated + peak-height + its FFT
ridge); the energy collapse is printed only as a (failing) cross-check.

Self-contained (numpy + matplotlib). Reads a dual_gate_speccube_gtd1.npz cache (built by
build_gtd1_cache.py / the viewer): by default calibration\\dual_gate_speccube_gtd1.npz, or
the one given on the command line (a .npz, or a data folder holding one). Writes
gtd1_calibration.json + gtd1_collapse.png into calibration\\. Run with the ROOT .venv:
  & "C:\\Users\\YouAr\\OneDrive - purdue.edu\\.venv\\Scripts\\python.exe" gtd1_full_collapse.py [cache.npz | data_folder]
"""
import sys, json
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

EV_NM = 1239.841984
HERE = Path(__file__).resolve().parent
CACHE_NAME = "dual_gate_speccube_gtd1.npz"
OUT = HERE / "calibration"                 # sample-specific files: reference cache, calibration json, figure
CACHE = OUT / CACHE_NAME                   # default input (override on the command line)
CAL_JSON = OUT / "gtd1_calibration.json"   # written by this script; loaded by the viewer's "Load calibration"
IX_WIN = (1.28, 1.52)          # IX window; excludes the 1.20 eV (1033 nm) laser leak
BG_NM = (1050.0, 1150.0)       # reference wavelength window for per-spectrum baseline
BAND_HALF = 5                  # spatial rows summed around the emission spot
R_SCAN = np.arange(-2.60, 2.601, 0.02)
R_FINE = 0.005


def _bg_band_spectra(cube, wl, spat_rows, spot):
    c = spot - int(spat_rows[0])
    lo = max(0, c - BAND_HALF); hi = min(cube.shape[2] - 1, c + BAND_HALF)
    spec = np.nan_to_num(cube[:, :, lo:hi + 1, :]).sum(2)          # (nvf,nvb,nwl)
    bg = (wl >= BG_NM[0]) & (wl <= BG_NM[1])
    if bg.any():
        spec = spec - np.nanmean(spec[:, :, bg], axis=2, keepdims=True)
    return spec


def _ix(spec, wl):
    ev = EV_NM / wl
    win = (ev > IX_WIN[0]) & (ev < IX_WIN[1])
    Sw = np.clip(spec[:, :, win], 0, None)                         # IX-window, non-negative
    return Sw, ev[win]


def intensity_map(spec, wl):
    Sw, _ = _ix(spec, wl); return Sw.sum(2)                        # integrated IX intensity


def peakheight_map(spec, wl):
    Sw, _ = _ix(spec, wl); return Sw.max(2)


def energy_map(spec, wl):
    Sw, ew = _ix(spec, wl); return ew[np.argmax(Sw, 2)]           # IX peak energy (argmax)


def _collapse_score(field, VF, VB, r, binw=0.2):
    u = (VF + r * VB).ravel(); y = field.ravel()
    ok = np.isfinite(y); u, y = u[ok], y[ok]
    lo, hi = u.min(), u.max(); nb = max(10, int((hi - lo) / binw))
    idx = np.clip(((u - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    M = y.mean(); tot = ((y - M) ** 2).sum()
    if tot <= 0:
        return 0.0
    s = np.bincount(idx, weights=y, minlength=nb)
    c = np.bincount(idx, minlength=nb).astype(float)
    mb = np.where(c > 0, s / np.maximum(c, 1), M)
    return float((c * (mb - M) ** 2).sum() / tot)


def collapse_r(field, VF, VB):
    sc = np.array([_collapse_score(field, VF, VB, r) for r in R_SCAN])
    good = np.abs(R_SCAN) > 0.2                                    # ignore the r~0 degeneracy
    r0 = R_SCAN[good][int(np.argmax(sc[good]))]
    fine = np.arange(r0 - 0.06, r0 + 0.061, R_FINE)
    sf = np.array([_collapse_score(field, VF, VB, r) for r in fine])
    k = int(np.argmax(sf)); rb = float(fine[k])
    if 0 < k < len(fine) - 1:
        y1, y2, y3 = sf[k - 1], sf[k], sf[k + 1]; den = y1 - 2 * y2 + y3
        if den != 0:
            rb = float(fine[k] - 0.5 * R_FINE * (y3 - y1) / den)
    peak, base = sf.max(), sf.min()
    within = fine[sf >= peak - 0.02 * (peak - base)]
    sig = float((within.max() - within.min()) / 2) if within.size > 1 else R_FINE
    return dict(r=rb, absr=abs(rb), sigma=sig, score=float(peak), curve=(R_SCAN, sc))


def r_from_fft(field):
    nvf, nvb = field.shape
    win = np.hanning(nvf)[:, None] * np.hanning(nvb)[None, :]
    F = np.fft.fftshift(np.fft.fft2(np.nan_to_num(field - np.nanmean(field)) * win))
    P = np.abs(F) ** 2
    kf = np.fft.fftshift(np.fft.fftfreq(nvf)); kb = np.fft.fftshift(np.fft.fftfreq(nvb))
    KB, KF = np.meshgrid(kb, kf)
    w = P * (np.hypot(KF, KB) > 0.03)
    cff = (w * KF * KF).sum(); cbb = (w * KB * KB).sum(); cfb = (w * KF * KB).sum()
    _, evec = np.linalg.eigh(np.array([[cff, cfb], [cfb, cbb]]))
    vfd, vbd = evec[:, -1]
    return abs(vbd / vfd) if abs(vfd) > 1e-9 else np.nan


def main():
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE
    if cache.is_dir():
        cache = cache / CACHE_NAME
    if not cache.exists():
        sys.exit(f"cache not found: {cache}\nrun build_gtd1_cache.py <data_folder> first")
    print("cache", cache)
    z = np.load(cache, allow_pickle=True)
    cube = z["cube"]; wl = np.asarray(z["wl"], float)
    vf = np.asarray(z["vf_vals"], float); vb = np.asarray(z["vb_vals"], float)
    VF, VB = np.meshgrid(vf, vb, indexing="ij")
    spec = _bg_band_spectra(cube, wl, z["spat_rows"], int(z["spot_center"]))

    Ii = intensity_map(spec, wl)                       # clean density (iso-n) tracker
    Ph = peakheight_map(spec, wl)
    En = energy_map(spec, wl)                          # mixes n and D -> does NOT collapse
    cI = collapse_r(Ii, VF, VB)
    cP = collapse_r(Ph, VF, VB)
    cE = collapse_r(En, VF, VB)
    rf = r_from_fft(Ii)

    vals = [cI["absr"], cP["absr"], rf]                # intensity-based (the reliable ones)
    absr, sig = float(np.mean(vals)), float(np.std(vals))

    print("=" * 68)
    print("GT-D1  shear-collapse calibration of  r = c_b/c_t")
    print("=" * 68)
    print(f"grid {vf.size} x {vb.size}   Vf[{vf[0]:+.1f},{vf[-1]:+.1f}]  Vb[{vb[0]:+.1f},{vb[-1]:+.1f}]"
          f"   IX window {IX_WIN[0]}-{IX_WIN[1]} eV")
    print("  -- reliable (intensity = doping/quenching tracker, iso-n) --")
    print(f"  integrated-intensity collapse : |r|={cI['absr']:.3f}  (signed {cI['r']:+.3f}; score {cI['score']:.2f})")
    print(f"  peak-height collapse          : |r|={cP['absr']:.3f}  (signed {cP['r']:+.3f}; score {cP['score']:.2f})")
    print(f"  FFT ridge (intensity map)     : |r|={rf:.3f}")
    print(f"  ==> r = {absr:.3f} +/- {sig:.3f}")
    print(f"      implied thickness ratio d_back/d_front = 1/r = {1.0/absr:.2f}")
    print("  -- cross-check (UNRELIABLE: IX energy mixes Stark(D) + doping(n)) --")
    print(f"  peak-energy collapse          : |r|={cE['absr']:.3f}  (score {cE['score']:.2f}, region-dependent -> discard)")
    print("=" * 68)

    # ---- write the calibration file (r + provenance of the data used) --------
    src = str(z["source_dir"]) if "source_dir" in z.files else "(not recorded in this cache)"
    built = (datetime.fromtimestamp(float(z["built"])).isoformat(timespec="seconds")
             if "built" in z.files else None)
    n_meas = int(np.isfinite(np.asarray(z["tstamp"], float)).sum())
    cal = {
        "kind": "gtd1_dual_gate_calibration",
        "version": 1,
        "r": round(absr, 4),
        "r_sigma": round(sig, 4),
        "r_definition": ("r = c_b/c_t = d_front/d_back ;  "
                         "u_n = V_f + r*V_b (density),  u_D = V_f - r*V_b (displacement)"),
        "methods": {
            "integrated_intensity": {"abs_r": round(cI["absr"], 4), "signed_r": round(cI["r"], 4),
                                     "score_R2": round(cI["score"], 4), "reliable": True},
            "peak_height": {"abs_r": round(cP["absr"], 4), "signed_r": round(cP["r"], 4),
                            "score_R2": round(cP["score"], 4), "reliable": True},
            "fft_ridge": {"abs_r": round(float(rf), 4), "reliable": True},
            "peak_energy_crosscheck": {"abs_r": round(cE["absr"], 4), "score_R2": round(cE["score"], 4),
                                       "reliable": False,
                                       "note": "IX energy mixes Stark(D)+doping(n); does not collapse"},
        },
        "data": {
            "source_dir": src,
            "cache": cache.name,
            "cache_built": built,
            "grid": [int(vf.size), int(vb.size)],
            "vf_range": [float(vf.min()), float(vf.max())],
            "vb_range": [float(vb.min()), float(vb.max())],
            "n_measured": n_meas,
            "ix_window_eV": [IX_WIN[0], IX_WIN[1]],
            "bg_window_nm": [BG_NM[0], BG_NM[1]],
            "band_half_rows": BAND_HALF,
        },
        "fit": {
            "method": "shear-collapse (ANOVA R^2) on IX intensity; r = mean(integrated, peak-height, FFT)",
            "created": datetime.now().isoformat(timespec="seconds"),
            "script": Path(__file__).name,
        },
    }
    OUT.mkdir(exist_ok=True)
    CAL_JSON.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("wrote", CAL_JSON)

    if _HAVE_MPL:
        OUT.mkdir(exist_ok=True)
        ext = [vb.min(), vb.max(), vf.min(), vf.max()]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        im = ax[0].imshow(Ii, origin="lower", aspect="auto", extent=ext, cmap="viridis")
        ax[0].set_title("integrated IX intensity (iso-n)"); fig.colorbar(im, ax=ax[0])
        vbl = np.array([vb.min(), vb.max()])
        ax[0].plot(vbl, absr * -vbl, "w--", lw=1)            # iso-n line slope -r through origin
        ax[0].set_xlabel("V_back (V)"); ax[0].set_ylabel("V_front (V)")
        ax[0].set_ylim(vf.min(), vf.max())
        ax[1].plot(*cI["curve"], label=f"intensity (peak {cI['r']:+.2f}, R2 {cI['score']:.2f})")
        ax[1].plot(*cP["curve"], label=f"peak height (peak {cP['r']:+.2f})")
        ax[1].axvline(0, color="k", lw=0.5)
        ax[1].set_xlabel("trial r"); ax[1].set_ylabel("collapse score (ANOVA R^2)")
        ax[1].set_title(f"r = {absr:.3f} +/- {sig:.3f}"); ax[1].legend()
        ax[2].plot(*cE["curve"], color="C3")
        ax[2].axvline(0, color="k", lw=0.5)
        ax[2].set_xlabel("trial r"); ax[2].set_ylabel("collapse score")
        ax[2].set_title("peak-ENERGY collapse (fails: n-D mixing)")
        p = OUT / "gtd1_collapse.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        print("wrote", p)
    return absr, sig


if __name__ == "__main__":
    main()
