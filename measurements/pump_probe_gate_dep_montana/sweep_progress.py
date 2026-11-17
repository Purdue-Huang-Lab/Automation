"""Progress of one gate x power sweep, kept on disk so an unfinished sweep can be resumed.

``sweep_state_<timestamp>.json`` in the sweep's save folder holds the sweep parameters
(everything needed to rebuild the same scan plan, without the device objects), the list
of scans, which of them are done, which one is running and the status.  It is rewritten
(atomically) at every scan start / end, so it survives a GUI crash.

Rule for a sweep that was terminated (Terminate button) or that died (GUI crash, the file
still says "running"): the scan in progress -- or, if none was running, the last finished
one -- is treated as bad.  It is removed from the done list (its files, if any, get a
``.bad`` suffix so the viewers ignore them) and is measured again by Resume.

A sweep that ended by itself (Stop, error, compliance trip) only redoes the scan that was
unfinished: the scans finished before are kept.

No Qt here: the sweep thread and the widget share one object; the lock makes every
method atomic, and after ``freeze()`` (Terminate) the abandoned thread can no longer
change the progress.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

STATE_PREFIX = "sweep_state_"
STATE_VERSION = 1


def state_path(save_dir: str, timestamp: str) -> str:
    return os.path.join(save_dir, f"{STATE_PREFIX}{timestamp}.json")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SweepProgress:
    def __init__(self, path: str, data: Dict):
        self.path = path
        self.data = data
        self._lock = threading.RLock()
        self._frozen = False
        self.last_save_error = ""

    # ------------------------------------------------------------------ create / load
    @classmethod
    def new(cls, save_dir: str, timestamp: str, params: Dict, labels: List[str]) -> "SweepProgress":
        data = {
            "version": STATE_VERSION,
            "timestamp": timestamp,
            "save_dir": save_dir,
            "status": "running",
            "started": _now(),
            "params": params,
            "labels": list(labels),
            "done": {},  # str(idx) -> {"prefix", "para", "sum", "finished"}
            "last_done": None,
            "current": None,
            "redo": None,
            "discarded": [],
            "history": [{"time": _now(), "event": "start"}],
        }
        prog = cls(state_path(save_dir, timestamp), data)
        prog.save()
        return prog

    @classmethod
    def load(cls, path: str) -> "SweepProgress":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "params" not in data or "labels" not in data:
            raise ValueError(f"{os.path.basename(path)} is not a sweep state file")
        data.setdefault("done", {})
        data.setdefault("discarded", [])
        data.setdefault("history", [])
        # the save folder may have been moved / opened from another PC: use the file's folder
        data["save_dir"] = os.path.dirname(os.path.abspath(path))
        return cls(os.path.abspath(path), data)

    # ------------------------------------------------------------------ read
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def timestamp(self) -> str:
        return self.data["timestamp"]

    @property
    def save_dir(self) -> str:
        return self.data["save_dir"]

    @property
    def params(self) -> Dict:
        return self.data["params"]

    @property
    def status(self) -> str:
        return self.data.get("status", "")

    @property
    def total(self) -> int:
        return len(self.data["labels"])

    def label(self, idx: Optional[int]) -> str:
        if idx is None or not (0 <= idx < self.total):
            return "-"
        return f"#{idx + 1}/{self.total} {self.data['labels'][idx]}"

    def done_set(self) -> set:
        with self._lock:
            return {int(k) for k in self.data["done"]}

    @property
    def n_done(self) -> int:
        return len(self.data["done"])

    @property
    def finished(self) -> bool:
        return self.n_done >= self.total

    def remaining(self) -> List[int]:
        done = self.done_set()
        return [i for i in range(self.total) if i not in done]

    # ------------------------------------------------------------------ sweep thread
    def start_scan(self, idx: int) -> None:
        with self._lock:
            if self._frozen:
                return
            self.data["current"] = int(idx)
            self.save()

    def finish_scan(self, idx: int, prefix: str, para: str, summ: str, complete: bool) -> None:
        """A scan was saved.  Only a complete scan counts as done; a partial one is redone."""
        with self._lock:
            if self._frozen:
                return
            if complete:
                self.data["done"][str(int(idx))] = {"prefix": prefix, "para": para, "sum": summ,
                                                    "finished": _now()}
                self.data["last_done"] = int(idx)
                if self.data.get("redo") == int(idx):
                    self.data["redo"] = None
            self.data["current"] = None
            self.save()

    def end(self, status: str) -> None:
        """The sweep thread ended by itself (ok / aborted / trip / error).  ``current`` is kept."""
        with self._lock:
            if self._frozen:
                return
            self.data["status"] = status
            if self.data.get("current") is not None:
                self.data["redo"] = self.data["current"]
            self.data["history"].append({"time": _now(), "event": "end", "status": status,
                                         "done": self.n_done, "unfinished": self.data.get("current")})
            self.save()

    # ------------------------------------------------------------------ widget
    def freeze(self) -> None:
        """From now on the (abandoned) sweep thread cannot change the progress."""
        with self._lock:
            self._frozen = True

    def discard_last(self, reason: str) -> Optional[Dict]:
        """Treat the scan in progress, or else the last finished one, as bad.

        Returns {"idx", "label", "was_done", "renamed": [paths]} or None (nothing measured yet).
        """
        with self._lock:
            cur = self.data.get("current")
            idx = cur if cur is not None else self.data.get("last_done")
            out = None
            if idx is not None:
                idx = int(idx)
                rec = self.data["done"].pop(str(idx), None)
                renamed = []
                for p in ((rec or {}).get("para"), (rec or {}).get("sum")):
                    if p and os.path.exists(p):
                        try:
                            os.replace(p, p + ".bad")
                            renamed.append(p + ".bad")
                        except OSError:
                            pass
                self.data["redo"] = idx
                self.data["discarded"].append({"time": _now(), "idx": idx, "reason": reason,
                                               "was_done": rec is not None, "renamed": renamed})
                if self.data.get("last_done") == idx:
                    self.data["last_done"] = None
                out = {"idx": idx, "label": self.label(idx), "was_done": rec is not None, "renamed": renamed}
            self.data["current"] = None
            self.data["status"] = reason
            self.data["history"].append({"time": _now(), "event": reason, "done": self.n_done,
                                         "discarded": idx})
            self.save()
            return out

    def terminate(self) -> Optional[Dict]:
        """Terminate button: freeze, then discard the scan in progress / the last finished one."""
        with self._lock:
            self.freeze()
            return self.discard_last("terminated")

    def needs_crash_recovery(self) -> bool:
        """A loaded file still says "running": the GUI died during the sweep."""
        return self.status == "running"

    def resumed(self, note: str = "") -> None:
        with self._lock:
            self._frozen = False
            self.data["status"] = "running"
            self.data["current"] = None
            ev = {"time": _now(), "event": "resume", "done": self.n_done}
            if note:
                ev["note"] = note
            self.data["history"].append(ev)
            self.save()

    # ------------------------------------------------------------------ file
    def save(self) -> None:
        """Atomic rewrite of the state file; an error is kept in ``last_save_error`` (never raised)."""
        with self._lock:
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=1)
                os.replace(tmp, self.path)
                self.last_save_error = ""
            except Exception as exc:
                self.last_save_error = f"{type(exc).__name__}: {exc}"
