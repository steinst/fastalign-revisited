#!/usr/bin/env python3
"""Run a command only on a quiet machine and report how much CPU *other* processes used.

    quiet_run.py [--idle-cpus 1.0] [--max-wait 3600] -- CMD...

Waits until the rest of the machine uses less than --idle-cpus cores for 10 s in a row, runs
CMD, then prints `other_cpus=X.XX` (average cores busy outside CMD during the run: total busy
time from /proc/stat minus CMD's user+sys time, over its wall time) to stderr. Exit status is
CMD's. Step 9b: another session's benchmarks shared this machine; a run with other_cpus above
~0.5 is not used.
"""
import argparse
import os
import resource
import subprocess
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def busy() -> float:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return (sum(v[:8]) - idle) / HZ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-cpus", type=float, default=1.0)
    ap.add_argument("--max-wait", type=float, default=3600)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    t_start = time.time()
    quiet = 0
    while quiet < 10:
        b0 = busy()
        time.sleep(1)
        quiet = quiet + 1 if busy() - b0 < a.idle_cpus else 0
        if time.time() - t_start > a.max_wait:
            print("quiet_run: machine never became quiet", file=sys.stderr)
            return 3
    waited = time.time() - t_start
    b0, w0 = busy(), time.time()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    rc = subprocess.call(cmd)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    b1, w1 = busy(), time.time()
    own = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    other = max(0.0, (b1 - b0) - own) / max(w1 - w0, 1e-9)
    print(f"quiet_run: waited {waited:.0f} s, other_cpus={other:.2f}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
