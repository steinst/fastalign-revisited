#!/usr/bin/env python3
"""Edge-case parity: run real fast_align and our CLI on generated odd inputs and compare.

    edge_parity.py [--bin target/release/fastalign] [--keep DIR]

Every case is a (name, input bytes, fast_align flags) triple. Both programs run with
OMP_NUM_THREADS=1 / --threads 1 and 4. Compared: stdout (alignments) and the statistics on stderr
(progress dots removed, as parity_compare.py does). Cases where fast_align crashes or hangs are
listed under "fast_align fails" and checked only for *our* behaviour (no crash, clean error).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FA = ROOT / "reference/fast_align-src/build/fast_align"
PROGRESS = re.compile(rb"^\.*( \[\d+\])?$")
TABLE_LINE = re.compile(rb"^conditional probabilities: .*$")
NAN_SIGN = re.compile(rb"-nan")


def norm_err(b: bytes, keep_nan_sign: bool) -> list[bytes]:
    lines = [ln for ln in b.split(b"\n") if not PROGRESS.match(ln) and not TABLE_LINE.match(ln)]
    if not keep_nan_sign:
        lines = [NAN_SIGN.sub(b"nan", ln) for ln in lines]
    return lines


def ours_cmd(bin_, path, flags, threads, out_prefix):
    cmd = [str(bin_), "train", str(path), "-o", str(out_prefix), "--log", "--quiet", "-t", str(threads)]
    direction, diag, opt, vb = "fwd", False, False, False
    echo = ["i"]
    it = iter(flags)
    for f in it:
        echo.append(f[1:])
        if f == "-r": direction = "inv"
        elif f == "-d": diag = True
        elif f == "-o": opt = True
        elif f == "-v": vb = True
        elif f == "-N": cmd.append("--no-null")
        elif f == "-s": cmd.append("--scores")
        elif f == "-I": cmd += ["-I", next(it)]
        elif f == "-b": cmd += ["-b", next(it)]
        elif f == "-q": cmd += ["--p0", next(it)]
        elif f == "-T": diag = True; cmd += ["-T", next(it)]
        elif f == "-a": cmd += ["-a", next(it)]
        elif f == "-p": cmd.append("--table"); next(it)
        else: raise SystemExit(f"unhandled flag {f}")
    if not diag: cmd.append("--no-diagonal")
    if not opt: cmd.append("--no-optimize-tension")
    if not vb: cmd.append("--no-variational-bayes")
    cmd += ["--directions", direction]
    return cmd, direction, echo


def run_case(bin_, tmp, name, data, flags, threads=(1, 4)):
    path = tmp / f"{name}.txt"
    path.write_bytes(data)
    fa_flags = [str(tmp / f"{name}.fa.table") if (i and flags[i - 1] == "-p") else f
                for i, f in enumerate(flags)]
    try:
        g = subprocess.run([str(FA), "-i", str(path), *fa_flags], capture_output=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
        gout = g.stdout
        if "-p" in flags:
            tp = tmp / f"{name}.fa.table"
            gout += b"\n--table--\n" + (tp.read_bytes() if tp.exists() else b"")
        gold = (g.returncode, gout, g.stderr)
    except subprocess.TimeoutExpired:
        gold = None
    results = []
    for t in threads:
        pre = tmp / f"{name}.o{t}"
        cmd, direction, echo = ours_cmd(bin_, path, flags, t, pre)
        o = subprocess.run(cmd, capture_output=True, timeout=120)
        out = pre.with_name(pre.name + f".{direction}.pharaoh")
        log = pre.with_name(pre.name + f".{direction}.log")
        o_out = out.read_bytes() if out.exists() else b""
        o_err = (("".join(f"ARG={c}\n" for c in echo)).encode() + (log.read_bytes() if log.exists() else b""))
        tab = pre.with_name(pre.name + f".{direction}.table")
        if "-p" in flags:
            o_out += b"\n--table--\n" + (tab.read_bytes() if tab.exists() else b"")
        results.append((t, o.returncode, o_out, o_err, o.stderr))
    return gold, results


def cases():
    rnd = random.Random(7)
    words = [f"w{i}" for i in range(30)]
    def sent(n): return " ".join(rnd.choice(words) for _ in range(n))
    def corp(n, lo=1, hi=8, sep=" ||| "):
        return "\n".join(sent(rnd.randint(lo, hi)) + sep + sent(rnd.randint(lo, hi)) for _ in range(n)) + "\n"
    D = ["-d", "-o", "-v"]
    base = corp(40).encode()
    yield "base", base, D
    yield "empty-file", b"", D
    yield "only-newline", b"\n", D
    yield "empty-lines", b"a b ||| x y\n\n\nc d ||| z w\n\n", D
    yield "empty-src", b" ||| x y\na b ||| x y\nc ||| z\n", D
    yield "empty-tgt", b"a b |||\na b ||| x y\nc ||| z\n", D
    yield "empty-src-plain", b" ||| x y\na b ||| x y\nc ||| z\n", []
    yield "empty-tgt-plain", b"a b |||\na b ||| x y\nc ||| z\n", []
    yield "no-sep-line", b"a b ||| x y\nhello world\nc ||| z\n", D
    yield "multi-sep", b"a b ||| x y ||| q\nc d ||| z ||| ||| w\ne ||| f\n", D
    yield "single-token", b"a ||| x\n", D
    yield "single-token-plain", b"a ||| x\n", []
    yield "one-line", b"a b c d ||| x y z\n", D
    yield "one-line-N", b"a b c d ||| x y z\n", ["-d", "-o", "-v", "-N"]
    yield "same-token-many", b"a ||| x\n" * 20, D
    yield "dup-heavy", (b"a b c ||| x y z\n" * 200 + b"d e ||| u v\n"), D
    yield "dup-heavy-s", (b"a b c ||| x y z\n" * 200 + b"d e ||| u v\n"), D + ["-s"]
    yield "long-sent", (" ".join(rnd.choice(words) for _ in range(1500)) + " ||| " + " ".join(rnd.choice(words) for _ in range(1200)) + "\n" + corp(5)).encode(), D
    yield "long-sent-plain", (" ".join(rnd.choice(words) for _ in range(1500)) + " ||| " + " ".join(rnd.choice(words) for _ in range(1200)) + "\n" + corp(5)).encode(), []
    yield "very-long-sent", (" ".join(rnd.choice(words) for _ in range(5000)) + " ||| " + " ".join(rnd.choice(words) for _ in range(5000)) + "\n" + corp(5)).encode(), D
    yield "nonlatin", "привет мир ||| 你好 世界\nпривет ||| 你好\nмир мир ||| 世界 界\nこんにちは ||| سلام\n".encode() * 5, D
    yield "latin1", b"caf\xe9 cr\xe8me ||| \xe9t\xe9 chaud\nna\xefve caf\xe9 ||| \xe9t\xe9\n" * 6, D
    yield "crlf", corp(30).replace("\n", "\r\n").encode(), D
    yield "crlf-s", corp(30).replace("\n", "\r\n").encode(), D + ["-s"]
    yield "no-trailing-nl", corp(30).encode().rstrip(b"\n"), D
    yield "tabs-spaces", b"a\tb ||| x  y\n  c   d  ||| \t z\t\n", D
    yield "tab-sep", b"a b\tx y\nc d\tz\n", D
    yield "nul-bytes", b"a\x00b c ||| x\x00y z\nq ||| r\n", D
    yield "eps-word", b"<eps> a ||| x <eps>\n<eps> ||| y\nb ||| <eps> z\n", D
    yield "eps-N", b"<eps> a ||| x <eps>\n<eps> ||| y\nb ||| <eps> z\n", D + ["-N"]
    for I in ("0", "1", "2"):
        yield f"I{I}", base, D + ["-I", I]
    yield "I0-plain", base, ["-I", "0"]
    yield "b1", base, D + ["-b", "1"]
    yield "b0", base, D + ["-b", "0"]
    yield "b1-s", base, D + ["-s", "-b", "1"]
    yield "b3-N", base, D + ["-N", "-b", "3"]
    yield "rev", base, D + ["-r"]
    yield "rev-empty-src", b" ||| x y\na b ||| x y\nc ||| z\n", D + ["-r"]
    yield "only-eps-src", b"<eps> ||| x\n<eps> ||| y\n", D
    yield "huge-vocab-small-lines", ("\n".join(f"s{i} ||| t{i}" for i in range(500)) + "\n").encode(), D
    yield "tension-big", base, ["-d", "-o", "-v", "-T", "50"]
    yield "p0-edge", base, ["-d", "-q", "0.99"]
    yield "p0-zero", base, ["-d", "-q", "0"]
    yield "alpha-big", base, ["-v", "-a", "1000"]
    yield "ws-only-lines", b"   ||| x\n \t ||| y\na ||| b\n", D
    yield "trailing-sep-spaces", b"a b ||| x y   \n  a ||| x\n", D
    yield "sep-no-spaces", b"a b|||x y\nc|||z\nd ||| w\n", D
    yield "four-pipes", b"a |||| x\nb ||| y\n", D
    yield "long-line-diff", (sent(200) + " ||| " + sent(2) + "\n" + sent(2) + " ||| " + sent(200) + "\n" + corp(20)).encode(), D
    yield "table-base", base, D + ["-p", "T"]
    yield "table-empty-sides", b" ||| x y\na b |||\na b ||| x y\n", D + ["-p", "T"]
    yield "table-tiny", b"a ||| x\n", D + ["-p", "T"]
    yield "table-I0", base, D + ["-I", "0", "-p", "T"]
    yield "table-I1", base, D + ["-I", "1", "-p", "T"]
    yield "table-N-v", base, ["-v", "-N", "-p", "T"]
    yield "rand-small-100", b"".join(corp(rnd.randint(1, 6), 0, 4).encode() for _ in range(30)), D + ["-s"]


def forced_cases(bin_, tmp):
    """`-f TABLE` (fast_align) vs `align --table` on odd inputs."""
    rnd = random.Random(3)
    words = [f"w{i}" for i in range(30)]
    sent = lambda n: " ".join(rnd.choice(words) for _ in range(n))
    train = tmp / "f-train.txt"
    train.write_bytes(("\n".join(sent(rnd.randint(1, 8)) + " ||| " + sent(rnd.randint(1, 8))
                                for _ in range(60)) + "\n").encode())
    table = tmp / "f.table"
    subprocess.run([str(FA), "-i", str(train), "-d", "-o", "-v", "-p", str(table)],
                   capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    inputs = {
        "empty-file": b"", "empty-line": b"\n", "empty-src": b" ||| w1 w2\nw1 ||| w2\n",
        "empty-tgt": b"w1 w2 |||\nw1 ||| w2\n", "unseen": b"zzz yyy ||| w1 qqq\nw1 ||| w2\n",
        "crlf": b"w1 w2 ||| w3 w4\r\nw1 ||| w2\r\n", "no-nl": b"w1 w2 ||| w3 w4",
        "no-sep": b"w1 w2\nw1 ||| w2\n", "multi-sep": b"w1 ||| w2 ||| w3\nw4 ||| w5\n",
        "long": (sent(1200) + " ||| " + sent(1000) + "\n").encode(),
        "nonlatin": "привет ||| 你好\nw1 ||| w2\n".encode(),
    }
    flagsets = {"plain": ([], []), "diag": (["-d", "-T", "4"], ["--tension", "4"]),
                "N": (["-N"], ["--no-null"]), "m": (["-d", "-T", "3", "-m", "1.5", "-q", "0.2"],
                                                   ["--tension", "3", "-m", "1.5", "--p0", "0.2"]),
                "s": (["-d", "-T", "4", "-s"], ["--tension", "4", "--scores"]),
                "r": (["-r"], ["--direction", "inv"])}
    bad = 0
    n = 0
    for iname, data in inputs.items():
        path = tmp / f"f-{iname}.txt"
        path.write_bytes(data)
        for fname, (fa_f, our_f) in flagsets.items():
            n += 1
            g = subprocess.run([str(FA), "-i", str(path), "-f", str(table), *fa_f], capture_output=True,
                               env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
            args = [str(bin_), "align", "--table", str(table), str(path), *our_f]
            if "-d" not in fa_f and "-T" not in fa_f:
                args.append("--no-diagonal")
            o = subprocess.run(args, capture_output=True, timeout=120)
            if g.returncode != 0 and o.returncode == 0:
                print(f"NOTE forced {iname}/{fname}: fast_align rc={g.returncode}, ours rc=0 (stdout {'equal' if o.stdout == g.stdout else 'differs'})")
            if o.returncode not in (0, 1):
                print(f"FAIL forced {iname}/{fname}: our rc={o.returncode} {o.stderr[:200]!r}"); bad += 1; continue
            if o.stdout != g.stdout:
                print(f"FAIL forced {iname}/{fname}: stdout differs (rc fa={g.returncode} ours={o.returncode})"); bad += 1
            elif o.returncode != g.returncode:
                print(f"FAIL forced {iname}/{fname}: rc fa={g.returncode} ours={o.returncode}"); bad += 1
            else:
                ge = [x for x in g.stderr.split(b"\n") if not x.startswith(b"ARG=") and not PROGRESS.match(x)]
                oe = [x for x in o.stderr.split(b"\n") if not PROGRESS.match(x)]
                if norm_err(b"\n".join(ge), False) != norm_err(b"\n".join(oe), False):
                    print(f"FAIL forced {iname}/{fname}: stderr differs\n  fa:   {ge[:4]}\n  ours: {oe[:4]}"); bad += 1
    print(f"forced alignment: {n - bad}/{n} identical")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "target/release/fastalign"))
    ap.add_argument("--keep")
    ap.add_argument("--only")
    a = ap.parse_args()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp())
    tmp.mkdir(exist_ok=True)
    npass = nfail = 0
    fa_fail = []
    nansign = []
    for name, data, flags in cases():
        if a.only and a.only not in name:
            continue
        gold, res = run_case(a.bin, tmp, name, data, flags)
        if gold is None or gold[0] != 0:
            rc = "timeout" if gold is None else gold[0]
            msg = [f"fast_align rc={rc}"]
            for t, rcode, out, err, serr in res:
                msg.append(f"ours t={t}: rc={rcode} out={len(out)}B stderr={serr[:100]!r}")
                if rcode not in (0, 1) and rcode != 0:
                    msg.append("   !! crash")
                    nfail += 1
            fa_fail.append((name, msg))
            continue
        _, gout, gerr = gold
        ok = True
        for t, rcode, out, err, serr in res:
            if rcode != 0:
                print(f"FAIL {name} t={t}: our rc={rcode}: {serr[:200]!r}"); ok = False; continue
            if out != gout:
                print(f"FAIL {name} t={t}: stdout differs"); ok = False
            if norm_err(err, True) != norm_err(gerr, True):
                if norm_err(err, False) == norm_err(gerr, False):
                    nansign.append(f"{name} t={t}")
                else:
                    print(f"FAIL {name} t={t}: stderr differs")
                    ge, oe = norm_err(gerr, True), norm_err(err, True)
                    for x, y in zip(ge, oe):
                        if x != y: print(f"   fa:   {x[:150]!r}\n   ours: {y[:150]!r}"); break
                    ok = False
        npass += ok; nfail += not ok
    print(f"\n{npass} cases identical (incl. NaN-sign-only), {nfail} differing")
    print(f"NaN-sign-only stderr differences (documented, stdout identical): {len(nansign)}: "
          + ", ".join(nansign))
    print("fast_align fails:" if fa_fail else "fast_align fails: none")
    for name, msg in fa_fail:
        print(f"  {name}: " + "; ".join(msg))
    nfail += forced_cases(a.bin, tmp)
    if not a.keep:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
