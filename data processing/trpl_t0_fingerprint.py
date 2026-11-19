"""Session-calibrated time zero for SPAD93 TRPL data ("fingerprint" alignment).

Why the per-file methods (first-peak edge / max of each trace) fail at low power
------------------------------------------------------------------------------
* Every SPAD93 pixel has its own TDC offset.  The offsets are spread over the whole histogram period
  (here 18 ... 1309 ns) and are re-randomised whenever the detector is re-initialised, so they differ
  between sessions; within one session they are stable to ~10 ps (checked over 14 h / 33 files).
* Every pixel sees the same instrument fingerprint: a few sharp spikes (laser leakage / scatter) at fixed
  delays before the PL step (2026-11-13 session: -33.22, -25.45, -11.34 ns, identical in all pixels to
  10-30 ps), then the PL step + slow decay.
* In the outer pixels the PL step is weak and the spikes are as tall as or taller than it, worse at low
  power.  A peak / edge search then locks onto the first spike, ~33-35 ns early, in a different set of
  pixels in every file.

What this module does
---------------------
1. Calibrate once per session from the SUM of all files (huge SNR): every pixel's lag is the circular lag
   that best fits   y(t) = a * Spikes(t - L) + b * PL(t - L) + c   with a, b >= 0.  The two templates are
   the pre-edge part (spike comb) and the post-edge part (PL step + decay) of the aligned session mean.
   They are bootstrapped from the bright PL-dominated pixels, whose step is unambiguous.  A spike can't be
   explained as a negative step, and the comb only repeats at one lag, so there is no 35 ns ambiguity.
2. t = 0 is the half-rise of the PL step of the aligned session mean.
3. Every file gets the same per-pixel shifts plus ONE global shift.  The global shift comes from fitting the
   skew-corrected pixel sum of that file, which is robust at any power.

CLI:  python trpl_t0_fingerprint.py <session folder> [--out calib.json] [--png diag.png]
Writes <session>/t0_fingerprint_calibration.json (+ a diagnostic PNG) unless told otherwise.
numpy only (matplotlib for the PNG).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WIN_NS = (-45.0, 60.0)    # template window around the PL edge
COARSE_NS = 15.0          # step-filter half window of the seed search (long: the PL tail beats a spike burst)
SEED_N = 15               # bright, PL-dominated pixels that seed the templates
SPLIT_NS = -0.5           # template split: spikes before, PL from here on
CONTRAST_MIN = 1.5        # pixels with a clearer best lag than this rebuild the templates
AGREE_NS = 0.3            # per-file fit within this of the calibration counts as agreeing


# ── I/O ────────────────────────────────────────────────────────────────────────────────────────────────
def load_trpl_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(time_ns [bins], counts [bins, pixels]) of a TRPL_*.csv / DARK_*.csv ('#' header lines skipped)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    a = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    return a[:, 0].astype(np.float64), a[:, 1:].astype(np.float64)


def session_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "TRPL_*.csv")))


# ── core ───────────────────────────────────────────────────────────────────────────────────────────────
def bgsub(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return c - np.median(c, axis=0)


def _step_filter(y: np.ndarray, w: int) -> np.ndarray:
    """f[i] = mean(y[i:i+w]) - mean(y[i-w:i]) (circular, per column)."""
    n = y.shape[0]
    pad = np.concatenate([y[-w:], y, y[:w]])
    cs = np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(pad, axis=0)])
    i = np.arange(n) + w
    return (cs[i + w] - cs[i]) / w - (cs[i] - cs[i - w]) / w


def _snippets(y: np.ndarray, lag: np.ndarray, rel: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    return np.stack([y[(int(round(l)) + rel) % n, p] for p, l in enumerate(lag)], axis=1)


def _templates(y: np.ndarray, lag: np.ndarray, rel: np.ndarray, dt: float) -> np.ndarray:
    """[len(rel), 2]: spike part (t < SPLIT_NS) and PL part of the aligned mean, each unit norm."""
    seg = _snippets(y, lag, rel)
    seg = seg - np.median(seg[: max(2, int(round(5.0 / dt)))], axis=0)
    nrm = np.linalg.norm(seg, axis=0)
    seg = seg[:, nrm > 0] / nrm[nrm > 0]
    pre = rel * dt < SPLIT_NS
    sp = np.clip(np.where(pre[:, None], seg, 0.0).sum(1), 0, None)
    pl = np.clip(np.where(pre[:, None], 0.0, seg).sum(1), 0, None)
    out = []
    for v in (sp, pl):
        n_ = np.linalg.norm(v)
        out.append(v / n_ if n_ > 0 else v)
    return np.stack(out, axis=1)


def fit_lags(y: np.ndarray, T: np.ndarray, rel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column: sub-bin circular lag (bins from bin 0 to the PL edge) of the best non-negative fit
    y ≈ a*T[:,0] + b*T[:,1] + c, the contrast (best score / best score >= 3 ns away) and (a, b) at the best lag."""
    y = np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, npx = y.shape
    Tf = np.zeros((n, 2))
    Tf[rel % n] = T
    Tf -= Tf.mean(axis=0)                     # constant offset decouples
    G = Tf.T @ Tf
    if abs(np.linalg.det(G)) < 1e-12 * max(np.trace(G) ** 2, 1e-30):
        G = G + 1e-9 * np.eye(2)              # degenerate template pair (e.g. no spikes at all)
    Gi = np.linalg.inv(G)
    FY = np.fft.fft(y - y.mean(axis=0), axis=0)
    r1 = np.real(np.fft.ifft(FY * np.conj(np.fft.fft(Tf[:, 0]))[:, None], axis=0))
    r2 = np.real(np.fft.ifft(FY * np.conj(np.fft.fft(Tf[:, 1]))[:, None], axis=0))
    a = Gi[0, 0] * r1 + Gi[0, 1] * r2
    b = Gi[1, 0] * r1 + Gi[1, 1] * r2
    both = np.where((a >= 0) & (b >= 0), a * r1 + b * r2, -np.inf)
    s1 = np.where(r1 > 0, r1 ** 2 / G[0, 0], -np.inf)
    s2 = np.where(r2 > 0, r2 ** 2 / G[1, 1], -np.inf)
    sc = np.maximum(both, np.maximum(s1, s2))
    sc[~np.isfinite(sc)] = 0.0
    L = np.argmax(sc, axis=0)
    lag = np.empty(npx)
    con = np.empty(npx)
    amp = np.zeros((npx, 2))
    sep = max(1, int(round(3.0 / dt)))
    idx = np.arange(n)
    for p in range(npx):
        l = int(L[p])
        u, v, w = sc[(l - 1) % n, p], sc[l, p], sc[(l + 1) % n, p]
        den = u - 2 * v + w
        d = 0.5 * (u - w) / den if den != 0 else 0.0
        lag[p] = (l + float(np.clip(d, -0.5, 0.5))) % n
        far = np.abs(((idx - l + n // 2) % n) - n // 2) > sep
        con[p] = v / max(float(sc[far, p].max()), 1e-12)
        amp[p] = max(a[l, p], 0.0), max(b[l, p], 0.0)
    return lag, con, amp


def _pl_half_rise(m: np.ndarray, dt: float) -> float:
    """Sub-bin index (circular) of the PL half-rise in an aligned mean trace m."""
    n = m.size
    w = max(2, int(round(COARSE_NS / dt)))
    c = int(np.argmax(_step_filter(m[:, None], w)[:, 0]))
    # fine: steepest 1 ns step within ±3 ns of the coarse one
    wf = max(1, int(round(1.0 / dt)))
    near = (c + np.arange(-int(3 / dt), int(3 / dt) + 1)) % n
    c = int(near[np.argmax(_step_filter(m[:, None], wf)[near, 0])])
    lvl = np.median(m[(c + np.arange(int(round(1 / dt)), int(round(4 / dt)) + 1)) % n])
    base = np.median(m[(c - np.arange(int(round(0.5 / dt)), int(round(3 / dt)) + 1)) % n])
    half = base + 0.5 * (lvl - base)
    k = c + int(round(1 / dt))
    steps = 0
    while m[(k - 1) % n] >= half and steps < int(round(4 / dt)):
        k -= 1
        steps += 1
    a_, b_ = m[(k - 1) % n], m[k % n]
    frac = (half - a_) / (b_ - a_) if b_ != a_ else 1.0
    return float((k - 1 + frac) % n)


def calibrate(ysum: np.ndarray, dt: float, n_iter: int = 3) -> Dict:
    """Per-pixel PL-edge lag (bins, sub-bin) from a session sum of background-subtracted counts."""
    n, npx = ysum.shape
    w = max(2, int(round(COARSE_NS / dt)))
    f = _step_filter(ysum, w)
    i0 = np.argmax(f, axis=0)
    h = f[i0, np.arange(npx)]
    k3 = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3.0, "same"), 0, ysum)
    pre = np.array([k3[(i0[p] + np.arange(-int(40 / dt), -int(1 / dt))) % n, p].max() for p in range(npx)])
    mad = np.median(np.abs(ysum - np.median(ysum, axis=0)), axis=0)
    snr = h / np.sqrt(np.maximum(mad, 1.0))
    cand = np.flatnonzero(pre < 0.5 * h)
    if cand.size < 3:
        cand = np.arange(npx)
    seeds = cand[np.argsort(-snr[cand])][:SEED_N]
    rel = np.arange(int(round(WIN_NS[0] / dt)), int(round(WIN_NS[1] / dt)))
    T = _templates(ysum[:, seeds], i0[seeds].astype(float), rel, dt)
    for _ in range(n_iter):
        lag, con, _ = fit_lags(ysum, T, rel, dt)
        use = np.flatnonzero(con > CONTRAST_MIN)
        if use.size < 3:
            use = seeds
        T = _templates(ysum[:, use], lag[use], rel, dt)
    lag, con, amp = fit_lags(ysum, T, rel, dt)
    # t = 0 at the PL half-rise of the aligned mean (the template's own zero may sit on a spike)
    ref = float(np.median(lag))
    sh = np.round(((ref - lag + n / 2) % n) - n / 2).astype(int)
    good = con > 1.0
    mean = np.mean(np.stack([np.roll(ysum[:, p], sh[p]) / max(np.abs(ysum[:, p]).max(), 1e-12)
                             for p in np.flatnonzero(good)], axis=1), axis=1)
    frac_ref = lag - np.round(lag)            # sub-bin part lost by the whole-bin roll
    edge_idx = _pl_half_rise(mean, dt) + float(np.median(frac_ref[good]))
    corr = ((edge_idx - ref + n / 2) % n) - n / 2   # bins from the template zero to the PL half rise
    lag = (lag + corr) % n
    T = _templates(ysum[:, good], lag[good], rel, dt)   # template zero = PL half rise from here on
    return dict(lag_bins=lag, contrast=con, amp=amp, T=T, rel=rel, dt=dt, n=n, seeds=seeds,
                zero_correction_ns=corr * dt)


# ── session driver ─────────────────────────────────────────────────────────────────────────────────────
def calibrate_session(folder: str, files: Optional[Sequence[str]] = None, verbose: bool = True) -> Dict:
    files = list(files) if files is not None else session_files(folder)
    if not files:
        raise ValueError(f"no TRPL_*.csv in {folder}")
    data = []
    for i, fp in enumerate(files):
        t, c = load_trpl_csv(fp)
        data.append((fp, t, c))
        if verbose:
            print(f"\rloading {i + 1}/{len(files)}", end="", flush=True)
    if verbose:
        print()
    dts = {round(float(np.median(np.diff(t))), 6) for _, t, _ in data}
    if len(dts) != 1:
        raise ValueError(f"mixed bin widths in the session: {sorted(dts)}")
    dt = dts.pop()
    npx = {c.shape[1] for _, _, c in data}
    if len(npx) != 1:
        raise ValueError(f"mixed pixel counts: {sorted(npx)}")
    n_min = min(c.shape[0] for _, _, c in data)
    t_first = float(data[0][1][0])
    ysum = sum(bgsub(c[:n_min]) for _, _, c in data)
    cal = calibrate(ysum, dt)
    edge_raw = t_first + cal["lag_bins"] * dt                 # raw time (ns) of every pixel's PL edge

    per_file = []
    agree = np.zeros(ysum.shape[1])
    for fp, t, c in data:
        n = c.shape[0]
        period = n * dt
        y = bgsub(c)
        pos = (edge_raw - t[0]) % period / dt                  # this file's bins
        tgt = float(np.median(pos))
        shift = np.round(((tgt - pos + n / 2) % n) - n / 2).astype(int)
        al = np.stack([np.roll(y[:, p], shift[p]) for p in range(y.shape[1])], axis=1)
        g, gcon, _ = fit_lags(al.sum(axis=1), cal["T"], cal["rel"], dt)
        own, _, _ = fit_lags(y, cal["T"], cal["rel"], dt)     # per-file per-pixel check
        dev = (((own - pos + n / 2) % n) - n / 2) * dt
        dev -= np.median(dev)
        agree += np.abs(dev) <= AGREE_NS
        t0_raw = float(t[0] + (g[0] % n) * dt)
        per_file.append(dict(file=os.path.basename(fp), n_bins=int(n), t0_raw_ns=t0_raw,
                             global_vs_median_ns=float(((g[0] - tgt + n / 2) % n - n / 2) * dt),
                             contrast=float(gcon[0]), n_pix_disagree=int((np.abs(dev) > AGREE_NS).sum())))
    agree /= len(data)
    return dict(folder=os.path.abspath(folder), created=datetime.now().isoformat(timespec="seconds"),
                bin_ns=dt, n_files=len(data), n_bins_min=int(n_min), time_first_ns=t_first,
                pixel_edge_raw_ns=edge_raw.tolist(), pixel_contrast=cal["contrast"].tolist(),
                pixel_agreement=agree.tolist(),
                unreliable_pixels=[int(p) for p in np.flatnonzero(agree < 0.5)],
                zero_correction_ns=cal["zero_correction_ns"], files=per_file,
                _cal=cal, _ysum=ysum, _data=data)


def pixel_shifts(cal_json: Dict, time_ns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Whole-bin per-pixel shifts that put every pixel's PL edge on the file's median edge, and that
    median edge as raw time.  Use with the per-file global t0 (``files[i]['t0_raw_ns']``) for t = 0."""
    t = np.asarray(time_ns, np.float64)
    dt = float(np.median(np.diff(t)))
    n = t.size
    pos = (np.asarray(cal_json["pixel_edge_raw_ns"]) - t[0]) % (n * dt) / dt
    tgt = float(np.median(pos))
    shift = np.round(((tgt - pos + n / 2) % n) - n / 2).astype(int)
    return shift, float(t[0] + tgt * dt)


def save_json(res: Dict, path: str) -> None:
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["method"] = "fingerprint (non-negative spike + PL template fit on the session sum)"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def diagnostic_png(res: Dict, path: str, low_file: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal, ysum, data = res["_cal"], res["_ysum"], res["_data"]
    dt, n = res["bin_ns"], ysum.shape[0]
    rel = np.arange(int(-50 / dt), int(60 / dt))
    lag = (np.asarray(res["pixel_edge_raw_ns"]) - res["time_first_ns"]) / dt
    sn = _snippets(ysum, lag, rel)
    sn = sn / np.maximum(np.percentile(sn, 99.5, axis=0), 1e-9)
    order = np.argsort(np.clip(sn[rel * dt < -1], 0, None).sum(0) / np.maximum(np.clip(sn[rel * dt > 0], 0, None).sum(0), 1e-9))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    ax[0].imshow(sn[:, order].T, aspect="auto", extent=[rel[0] * dt, rel[-1] * dt, sn.shape[1], 0],
                 vmin=-0.1, vmax=1, cmap="magma")
    ax[0].set_xlabel("t (ns)")
    ax[0].set_ylabel("pixel (sorted by spike/PL ratio)")
    ax[0].set_title("session sum, aligned (every row should be straight)")

    fi = [f["t0_raw_ns"] for f in res["files"]]
    ax[1].plot(np.asarray(fi) - np.median(fi), "o-", ms=3)
    ax[1].set_xlabel("file #")
    ax[1].set_ylabel("global t0 - median (ns)")
    ax[1].set_title("per-file global time zero")

    # lowest-signal file: pixel traces aligned, outer vs inner pixels
    k = int(np.argmin([np.abs(bgsub(c)).sum() for _, _, c in data])) if low_file is None else \
        [os.path.basename(fp) for fp, _, _ in data].index(low_file)
    fp, t, c = data[k]
    sh, _ = pixel_shifts(res, t)
    y = bgsub(c)
    t0 = res["files"][k]["t0_raw_ns"]
    for p in range(y.shape[1]):
        if p in res["unreliable_pixels"]:
            continue
        tr = np.convolve(np.roll(y[:, p], sh[p]), np.ones(5) / 5, "same")
        tr = tr / max(np.percentile(tr, 99.7), 1e-9)
        tt = ((t + 0.0) - t0 + n * dt / 2) % (n * dt) - n * dt / 2
        o = np.argsort(tt)
        ax[2].plot(tt[o], tr[o], lw=0.3, alpha=0.5)
    ax[2].set_xlim(-50, 60)
    ax[2].set_ylim(-0.4, 1.5)
    ax[2].axvline(0, c="k", lw=0.6)
    ax[2].set_xlabel("t (ns)")
    ax[2].set_title(f"weakest file, all pixels aligned: {os.path.basename(fp)}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="calibration JSON (default <folder>/t0_fingerprint_calibration.json)")
    ap.add_argument("--png", default=None, help="diagnostic PNG (default next to the JSON)")
    a = ap.parse_args(argv)
    res = calibrate_session(a.folder)
    out = a.out or os.path.join(a.folder, "t0_fingerprint_calibration.json")
    save_json(res, out)
    png = a.png or os.path.splitext(out)[0] + ".png"
    diagnostic_png(res, png)
    g = np.array([f["global_vs_median_ns"] for f in res["files"]])
    dis = np.array([f["n_pix_disagree"] for f in res["files"]])
    print(f"{res['n_files']} files, bin {res['bin_ns']} ns; pixel edges span "
          f"{min(res['pixel_edge_raw_ns']):.1f}..{max(res['pixel_edge_raw_ns']):.1f} ns raw")
    print(f"global t0 scatter across files: {g.std():.3f} ns (max |dev| {np.abs(g).max():.3f} ns)")
    print(f"per-file pixels disagreeing with the calibration (> {AGREE_NS} ns): median {np.median(dis):.0f}, max {dis.max()}")
    print(f"unreliable pixels (agree in < 50% of files): {res['unreliable_pixels']}")
    print(f"wrote {out}\n      {png}")


if __name__ == "__main__":
    main()
