"""Check the pSPAD command-server port and test repeated count reads.

1. Port check: who listens on the port and which processes hold connections
   to it (netstat + tasklist, no extra packages).
2. Frame test: send ``I,<ms>`` over one connection, frame after frame, and
   report the time to first byte, reply time and line count of each.  When a
   frame stalls it also tries a fresh connection, to tell a hung server from a
   stuck connection.

    python -m spad93.spad93_port_check                  # port 9998, 20 frames
    python -m spad93.spad93_port_check --port 9999 --frames 50 --ms 100
    python -m spad93.spad93_port_check --passive        # port check only
    python -m spad93.spad93_port_check --reconnect      # new connection per frame
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import time
from typing import Dict, List, Optional, Tuple

NUM_PIXELS = 93


# ---------------------------------------------------------------- port check
def _process_names() -> Dict[int, str]:
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    names: Dict[int, str] = {}
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1].strip('"').isdigit():
            names[int(parts[1].strip('"'))] = parts[0].strip('"')
    return names


def _split_addr(addr: str) -> Tuple[str, int]:
    host, _, port = addr.rpartition(":")
    return host, int(port) if port.isdigit() else -1


def port_check(port: int) -> bool:
    """Print the sockets on ``port``; return True if exactly one process listens."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10).stdout
        out6 = subprocess.run(["netstat", "-ano", "-p", "TCPv6"], capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:
        print(f"netstat failed: {exc}")
        return False
    names = _process_names()
    listeners: List[Tuple[str, int]] = []
    clients: List[Tuple[str, str, str, int]] = []
    for line in (out + "\n" + out6).splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP" or not parts[-1].isdigit():
            continue
        local, remote, state, pid = parts[1], parts[2], parts[3], int(parts[-1])
        _, lport = _split_addr(local)
        _, rport = _split_addr(remote)
        if state == "LISTENING" and lport == port:
            listeners.append((local, pid))
        elif rport == port or lport == port:
            clients.append((local, remote, state, pid))

    print(f"== Port {port} ==")
    if not listeners:
        print("  NOTHING is listening. Check the command-server port set in pSPAD.")
    for local, pid in listeners:
        print(f"  LISTEN  {local:24s} pid {pid:6d}  {names.get(pid, '?')}")
    if len({pid for _, pid in listeners}) > 1:
        print("  WARNING: more than one process listens on this port.")
    for local, remote, state, pid in clients:
        print(f"  {state:12s} {local:22s} -> {remote:22s} pid {pid:6d}  {names.get(pid, '?')}")
    stale = [c for c in clients if c[2] in ("CLOSE_WAIT", "FIN_WAIT_2")]
    if stale:
        print("  NOTE: half-closed connections (CLOSE_WAIT/FIN_WAIT_2); a client or the server did not close cleanly.")
    if not clients:
        print("  no open client connections")
    return len({pid for _, pid in listeners}) == 1


# ---------------------------------------------------------------- frame test
def _connect(host: str, port: int, timeout_s: float) -> Tuple[socket.socket, str]:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(1.0)
    try:
        banner = sock.recv(65536).decode("utf8", errors="ignore").strip()
    except socket.timeout:
        banner = "<no banner within 1 s>"
    return sock, banner


def _read_frame(sock: socket.socket, timeout_s: float, quiet_s: float) -> Tuple[Optional[float], float, bytes, str]:
    """Return (time to first byte, total time, data, end reason)."""
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    chunks: List[bytes] = []
    first_t: Optional[float] = None
    while True:
        now = time.monotonic()
        if now >= deadline:
            return first_t, now - t0, b"".join(chunks), "TIMEOUT"
        buf = b"".join(chunks)  # the last line comes without a newline
        lines = buf.count(b"\n") + (1 if buf and not buf.endswith(b"\n") else 0)
        sock.settimeout(quiet_s if (first_t is not None and lines >= NUM_PIXELS) else min(0.2, deadline - now))
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            if first_t is not None and lines >= NUM_PIXELS:
                return first_t, time.monotonic() - t0, b"".join(chunks), "ok"
            continue
        except OSError as exc:
            return first_t, time.monotonic() - t0, b"".join(chunks), f"socket error: {exc}"
        if not chunk:
            return first_t, time.monotonic() - t0, b"".join(chunks), "SERVER CLOSED CONNECTION"
        if first_t is None:
            first_t = time.monotonic() - t0
        chunks.append(chunk)
        tail = b"".join(chunks).rstrip()
        if tail.endswith(b"ERROR"):
            return first_t, time.monotonic() - t0, b"".join(chunks), "ERROR reply"


def _summarize(data: bytes) -> Tuple[int, int, int, List[str]]:
    """Return (lines, distinct pixel ids, total counts, non-numeric lines)."""
    ids, total, texts = set(), 0, []
    lines = [ln.strip() for ln in data.replace(b"\r", b"").split(b"\n") if ln.strip()]
    for ln in lines:
        parts = ln.split(b",")
        try:
            rid, cnt = int(parts[0]), int(float(parts[1]))
        except (ValueError, IndexError):
            texts.append(ln.decode("utf8", errors="ignore")[:120])
            continue
        if 0 <= rid < NUM_PIXELS:
            ids.add(rid)
            total += cnt
    return len(lines), len(ids), total, texts


def frame_test(host: str, port: int, frames: int, ms: int, timeout_s: float, quiet_s: float,
               reconnect: bool, interval_s: float) -> None:
    print(f"\n== Frame test: {frames} x I,{ms} on {host}:{port} "
          f"({'new connection per frame' if reconnect else 'one connection'}) ==")
    try:
        sock, banner = _connect(host, port, timeout_s)
    except OSError as exc:
        print(f"  connect FAILED: {exc}")
        return
    print(f"  banner: {banner!r}")
    reply_timeout = ms / 1000.0 + timeout_s
    n_ok = 0
    try:
        for i in range(1, frames + 1):
            if reconnect and i > 1:
                sock.close()
                sock, _ = _connect(host, port, timeout_s)
            sock.sendall(f"I,{ms}\n".encode())
            first, total_t, data, reason = _read_frame(sock, reply_timeout, quiet_s)
            n_lines, n_ids, counts, texts = _summarize(data)
            first_s = f"{first * 1000:7.0f}" if first is not None else "      -"
            print(f"  frame {i:3d}: first byte {first_s} ms, reply {total_t * 1000:7.0f} ms, "
                  f"{n_lines:3d} lines, {n_ids:2d}/{NUM_PIXELS} px, {counts:>10d} counts  [{reason}]")
            for t in texts[:3]:
                print(f"             text: {t!r}")
            if reason != "ok":
                print(f"  -> stalled at frame {i} ({reason}); last bytes {data[-120:]!r}")
                _diagnose_stall(host, port, ms, timeout_s, quiet_s)
                break
            n_ok += 1
            if interval_s > 0:
                time.sleep(interval_s)
        else:
            print(f"  all {n_ok} frames OK")
    finally:
        sock.close()


def _diagnose_stall(host: str, port: int, ms: int, timeout_s: float, quiet_s: float) -> None:
    print("  trying a fresh connection ...")
    try:
        sock, banner = _connect(host, port, timeout_s)
    except OSError as exc:
        print(f"  fresh connect FAILED ({exc}): the server stopped accepting clients.")
        return
    try:
        sock.sendall(f"I,{ms}\n".encode())
        first, total_t, data, reason = _read_frame(sock, ms / 1000.0 + timeout_s, quiet_s)
        n_lines, n_ids, _, _ = _summarize(data)
        print(f"  fresh connection: banner {banner!r}, {n_lines} lines, {n_ids} px [{reason}]")
        if reason == "ok":
            print("  => server is alive; the old connection got stuck (try --reconnect).")
        else:
            print("  => server does not answer on a new connection either: pSPAD itself is busy/hung "
                  "(another acquisition running in the pSPAD window, or overexposure shutdown -> SI).")
    finally:
        sock.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9998)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--ms", type=int, default=100, help="integration time per frame")
    p.add_argument("--timeout", type=float, default=5.0, help="extra seconds to wait beyond the integration")
    p.add_argument("--quiet", type=float, default=0.05, help="silence that ends a reply (same as the GUI)")
    p.add_argument("--interval", type=float, default=0.0, help="pause between frames")
    p.add_argument("--reconnect", action="store_true", help="open a new connection for every frame")
    p.add_argument("--passive", action="store_true", help="only check the port, send nothing")
    args = p.parse_args()

    port_check(args.port)
    if not args.passive:
        frame_test(args.host, args.port, args.frames, args.ms, args.timeout, args.quiet,
                   args.reconnect, args.interval)


if __name__ == "__main__":
    main()
