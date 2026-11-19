"""Run a command, timestamp each stderr line and sample the process tree's RSS.

Writes ``OUT.log`` (``t_seconds<TAB>rss_mb<TAB>line`` per stderr line, times since start) and
``OUT.rss`` (``t_seconds<TAB>rss_mb`` every ``--interval`` seconds), and prints a JSON summary
(wall time, peak RSS, the time and last log line at the peak).

Usage: python benchmarks/timed_run.py OUT_PREFIX [--interval 0.2] -- CMD ARGS...

Used for Step 11a (``RESULTS.md``): gizapp's progress events give per-iteration times; the gaps
between events (table construction, transfers, output writing) are only visible with timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time


def tree_rss_kb(pid: int) -> int:
    """Sum of VmRSS over ``pid`` and its descendants (kB); 0 if gone."""
    total = 0
    todo = [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
            with open(f"/proc/{p}/task/{p}/children") as f:
                todo.extend(int(c) for c in f.read().split())
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, errors="replace")
    state = {"rss": 0, "peak": 0, "peak_t": 0.0, "last": "", "peak_line": ""}
    lock = threading.Lock()

    def sampler() -> None:
        with open(a.out + ".rss", "w") as f:
            while proc.poll() is None:
                kb = tree_rss_kb(proc.pid)
                t = time.monotonic() - t0
                with lock:
                    state["rss"] = kb
                    if kb > state["peak"]:
                        state.update(peak=kb, peak_t=t, peak_line=state["last"])
                f.write(f"{t:.3f}\t{kb / 1024:.1f}\n")
                time.sleep(a.interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    with open(a.out + ".log", "w") as log:
        assert proc.stderr is not None
        for line in proc.stderr:
            t = time.monotonic() - t0
            line = line.rstrip("\n")
            with lock:
                state["last"] = line
                rss = state["rss"]
            log.write(f"{t:.3f}\t{rss / 1024:.1f}\t{line}\n")
            log.flush()
    rc = proc.wait()
    wall = time.monotonic() - t0
    th.join()
    # The exact peak (getrusage of reaped children) beats the sampled one.
    ru_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    summary = {
        "cmd": cmd,
        "rc": rc,
        "wall_s": round(wall, 2),
        # CLOCK_MONOTONIC at start, to line log times up with `perf record -k CLOCK_MONOTONIC`.
        "t0_monotonic": t0,
        "peak_rss_mb": round(max(ru_peak, state["peak"]) / 1024, 1),
        "sampled_peak_mb": round(state["peak"] / 1024, 1),
        "sampled_peak_t": round(state["peak_t"], 2),
        "sampled_peak_after": state["peak_line"],
        "host_load_1m": os.getloadavg()[0],
    }
    with open(a.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return rc


if __name__ == "__main__":
    sys.exit(main())
