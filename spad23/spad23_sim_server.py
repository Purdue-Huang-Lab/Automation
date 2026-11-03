"""Development-only fake pSPAD command server for testing the SPAD23 code without hardware.

    python -m spad23.spad23_sim_server [--port 9998] [--mode spot|spectral] [--two-spots]
                                       [--replay data000.txt] [--fault NAME]

Answers the commands the wrappers use (I, S, T,v/c/a, SI) with synthetic data:

* spot mode: a 2D Gaussian spot whose width grows as sigma^2(t) = sigma0^2 + 2 D t,
  optionally plus a second spot (SIM,pos2) that takes a fraction of the photons
* spectral mode: a dispersed emission line along y (one row per wavelength,
  lambda set with SIM,lambda) whose x width grows the same way and whose peak
  red-shifts with time

Simulator-only commands (any client may send them):
    SIM,mode,<spot|spectral>   SIM,lambda,<nm>   SIM,width,<nm>
    SIM,pos,<x_um>,<y_um>      SIM,rate,<cps>    SIM,fault,<name|none>
    SIM,pos2,<x_um>,<y_um>|none   SIM,frac2,<0..1>   SIM,sigma,<um>
Faults: error (S ends with ERROR), splitdone (DONE split over two packets),
silence (S sends no events), shutdown (I is rejected until SI).
"""

from __future__ import annotations

import argparse
import select
import socketserver
import threading
import time
from typing import List, Optional

import numpy as np

from spad23.spad23_geometry import (
    CENTER_X_UM,
    LINE_SPAN_UM,
    NUM_PIXELS,
    PIXEL_DIAMETER_UM,
    PIXEL_X_UM,
    PIXEL_Y_UM,
)

BANNER = b"pSPAD command server (SPAD23 simulator)"
SLICE_S = 0.05
LASER_PERIOD_PS = 50_000.0
T0_PS = 10_000.0
TAU_PS = 3_000.0
IRF_PS = 40.0
DARK_CPS = 100.0
HOT_PIXEL = 7
HOT_CPS = 2_000.0
LAMBDA_REF_NM = 750.0           # wavelength that lands on the array center
DISPERSION_NM_PER_UM = 0.25     # 0.25 nm/um -> about 20 nm across the 5 rows
COARSE_TICKS_PER_S = 100_000.0  # coarse counter wraps every 0.66 s


class SimState:
    def __init__(self, mode: str, rate_cps: float, fault: str, replay: Optional[str]):
        self.lock = threading.Lock()
        self.mode = mode
        self.rate_cps = float(rate_cps)
        self.fault = fault
        self.spot_xy = (CENTER_X_UM, LINE_SPAN_UM / 2.0)
        self.spot2_xy: Optional[tuple] = None
        self.spot2_frac = 0.4
        self.sigma0_um = 25.0
        self.diff_um2_per_ns = 150.0
        self.line_nm = LAMBDA_REF_NM
        self.line_width_nm = 3.0
        self.line_sigma_x_um = 14.0
        self.redshift_nm = 3.0
        self.tdc_valid = False
        self.shutdown = fault == "shutdown"
        self.rng = np.random.default_rng()
        self.replay_lines: Optional[List[bytes]] = None
        self.replay_weights: Optional[np.ndarray] = None
        if replay:
            self._load_replay(replay)

    def _load_replay(self, path: str) -> None:
        with open(path, "rb") as fh:
            lines = fh.read().split(b"\n")
        self.replay_lines = [ln for ln in lines if ln.strip()]
        hist = np.zeros(NUM_PIXELS)
        for ln in self.replay_lines[: min(len(self.replay_lines), 500_000)]:
            parts = ln.split(b",")
            if len(parts) == 3:
                pid = int(parts[0])
                if 0 <= pid < NUM_PIXELS:
                    hist[pid] += 1
        self.replay_weights = hist / max(hist.sum(), 1.0)
        print(f"replay: {len(self.replay_lines)} lines from {path}")

    # ---------------------------------------------------------- photon model
    def _y_of_lambda(self, lam_nm: np.ndarray) -> np.ndarray:
        return LINE_SPAN_UM / 2.0 + (lam_nm - LAMBDA_REF_NM) / DISPERSION_NM_PER_UM

    def photons(self, n: int):
        """Return (pixel, fine_ps) of ``n`` emitted photons (photons missing a pixel are dropped)."""
        rng = self.rng
        t_ps = rng.exponential(TAU_PS, n)
        t_ns = t_ps / 1000.0
        if self.mode == "spectral":
            lam = rng.normal(self.line_nm + self.redshift_nm * (1.0 - np.exp(-t_ns / 1.5)), self.line_width_nm)
            y = self._y_of_lambda(lam)
            sig_x = np.sqrt(self.line_sigma_x_um**2 + 2.0 * self.diff_um2_per_ns * t_ns)
            x = CENTER_X_UM + rng.normal(0.0, 1.0, n) * sig_x
        else:
            sig = np.sqrt(self.sigma0_um**2 + 2.0 * self.diff_um2_per_ns * t_ns)
            cx = np.full(n, self.spot_xy[0])
            cy = np.full(n, self.spot_xy[1])
            if self.spot2_xy is not None:
                two = rng.random(n) < self.spot2_frac
                cx[two] = self.spot2_xy[0]
                cy[two] = self.spot2_xy[1]
            x = cx + rng.normal(0.0, 1.0, n) * sig
            y = cy + rng.normal(0.0, 1.0, n) * sig
        d2 =(x[:, None] - PIXEL_X_UM[None, :]) ** 2 + (y[:, None] - PIXEL_Y_UM[None, :]) ** 2
        pix = np.argmin(d2, axis=1)
        hit = d2[np.arange(n), pix] <= (PIXEL_DIAMETER_UM / 2.0) ** 2
        fine = np.mod(T0_PS + t_ps + rng.normal(0.0, IRF_PS, n), LASER_PERIOD_PS)
        return pix[hit], fine[hit]

    def pixel_weights(self) -> np.ndarray:
        if self.replay_weights is not None:
            return self.replay_weights
        pix, _ = self.photons(60_000)
        hist = np.bincount(pix, minlength=NUM_PIXELS).astype(np.float64)
        return hist / 60_000.0

    def dark_rates(self) -> np.ndarray:
        rates = np.full(NUM_PIXELS, DARK_CPS)
        rates[HOT_PIXEL] = HOT_CPS
        return rates


class Handler(socketserver.BaseRequestHandler):
    state: SimState

    def handle(self) -> None:
        sock = self.request
        sock.sendall(BANNER)
        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf8", errors="ignore").strip()
                if not cmd:
                    continue
                try:
                    reply = self.dispatch(cmd)
                except OSError:
                    return
                if reply is not None:
                    try:
                        sock.sendall(reply)
                    except OSError:
                        return

    # -------------------------------------------------------------- commands
    def dispatch(self, cmd: str) -> Optional[bytes]:
        st = self.state
        parts = [p.strip() for p in cmd.split(",")]
        head = parts[0].upper()
        if head == "SIM":
            return self.sim_command(parts[1:])
        if head == "SI":
            with st.lock:
                was = st.shutdown
                st.shutdown = False
            return b"Reset of system shutdown initiated." if was else b"No shutdown active, command ignored."
        if head == "I" and len(parts) == 2:
            return self.count_reply(int(parts[1]))
        if head == "T" and len(parts) >= 3:
            sub = parts[1].lower()
            if sub == "v":
                return b"TDC calibration is valid" if st.tdc_valid else b"TDC calibration is invalid"
            if sub == "c":
                time.sleep(1.5)
                st.tdc_valid = True
                return b"TDC calibration completed"
            if sub == "a":
                time.sleep(int(parts[2]) / 1000.0)
                return b"Channel alignment completed"
        if head == "S" and len(parts) == 2:
            self.stream(int(parts[1]), cmd)
            return None
        return f"Unknown command '{cmd}'. ERROR".encode()

    def sim_command(self, args: List[str]) -> bytes:
        st = self.state
        if not args:
            return b"SIM needs arguments ERROR"
        key = args[0].lower()
        with st.lock:
            if key == "mode" and len(args) > 1:
                st.mode = args[1].lower()
            elif key == "lambda" and len(args) > 1:
                st.line_nm = float(args[1])
            elif key == "width" and len(args) > 1:
                st.line_width_nm = float(args[1])
            elif key == "pos" and len(args) > 2:
                st.spot_xy = (float(args[1]), float(args[2]))
            elif key == "pos2" and len(args) > 1:
                st.spot2_xy = None if args[1].lower() == "none" else (float(args[1]), float(args[2]))
            elif key == "frac2" and len(args) > 1:
                st.spot2_frac = float(args[1])
            elif key == "sigma" and len(args) > 1:
                st.sigma0_um = float(args[1])
            elif key == "rate" and len(args) > 1:
                st.rate_cps = float(args[1])
            elif key == "fault" and len(args) > 1:
                st.fault = args[1].lower()
                st.shutdown = st.fault == "shutdown"
            else:
                return f"Unknown SIM command {args} ERROR".encode()
        return b"OK"

    def count_reply(self, ms: int) -> bytes:
        st = self.state
        time.sleep(ms / 1000.0)
        with st.lock:
            if st.shutdown:
                return (b"Command rejected, system in shutdown mode. Please reset by sending "
                        b"the following command: SI. ERROR")
            lam = st.pixel_weights() * st.rate_cps * (ms / 1000.0) + st.dark_rates() * (ms / 1000.0)
            counts = st.rng.poisson(lam)
        return "\n".join(f"{i},{int(c)}" for i, c in enumerate(counts)).encode()

    def _cancel_requested(self, cmd: str) -> bool:
        sock = self.request
        readable, _, _ = select.select([sock], [], [], 0.0)
        if not readable:
            return False
        data = sock.recv(4096)
        return cmd.encode() in data or not data

    def stream(self, ms: int, cmd: str) -> None:
        st = self.state
        sock = self.request
        sock.sendall(b"119,0\n")
        start = time.monotonic()
        rollovers = 0
        replay_pos = 0
        fault = st.fault
        while True:
            now = time.monotonic() - start
            if now >= ms / 1000.0 or self._cancel_requested(cmd):
                break
            if fault == "error" and now >= 0.5 * ms / 1000.0:
                sock.sendall(b"Laser clock synchronization lost during the measurement. ERROR")
                return
            lines: List[bytes] = []
            if fault != "silence":
                with st.lock:
                    if st.replay_lines is not None:
                        n_take = max(1, int(len(st.replay_lines) * SLICE_S / (ms / 1000.0)))
                        lines = st.replay_lines[replay_pos: replay_pos + n_take]
                        replay_pos += n_take
                    else:
                        n = int(st.rng.poisson(st.rate_cps * SLICE_S))
                        pix, fine = st.photons(n)
                        dark = st.rng.poisson(st.dark_rates() * SLICE_S)
                        dpix = np.repeat(np.arange(NUM_PIXELS), dark)
                        dfine = st.rng.uniform(0, LASER_PERIOD_PS, dpix.size)
                        pix = np.concatenate([pix, dpix])
                        fine = np.concatenate([fine, dfine])
                        coarse = int(now * COARSE_TICKS_PER_S)
                        if coarse // 65536 > rollovers:
                            rollovers += 1
                            lines.append(f"{NUM_PIXELS},0,{rollovers - 1}".encode())
                        c = coarse % 65536
                        lines.extend(f"{p},{c},{int(f)}".encode() for p, f in zip(pix.tolist(), fine.tolist()))
            if lines:
                sock.sendall(b"\n".join(lines) + b"\n")
            time.sleep(SLICE_S)
        if fault == "splitdone":
            sock.sendall(b"DO")
            time.sleep(0.1)
            sock.sendall(b"NE")
        else:
            sock.sendall(b"DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake pSPAD server for SPAD23 GUI testing")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--mode", choices=("spot", "spectral"), default="spot")
    parser.add_argument("--rate", type=float, default=300_000.0, help="signal photons per second")
    parser.add_argument("--fault", default="none", help="error | splitdone | silence | shutdown | none")
    parser.add_argument("--replay", default="", help="stream lines from a recorded pSPAD data file")
    parser.add_argument("--two-spots", action="store_true", help="spot mode: spots at (23, 20) and (69, 60) um")
    args = parser.parse_args()

    Handler.state = SimState(args.mode, args.rate, args.fault, args.replay or None)
    if args.two_spots:
        Handler.state.spot_xy = (23.0, 20.0)
        Handler.state.spot2_xy = (69.0, 60.0)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", args.port), Handler) as server:
        print(f"SPAD23 simulator on 127.0.0.1:{args.port} (mode={args.mode}, fault={args.fault})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
