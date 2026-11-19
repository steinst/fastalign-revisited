#!/usr/bin/env python3
"""Executable check of docs/SPEC-fast-align.md (Step 3).

A slow, single-threaded model of fast_align written from the spec, not from the C++. It is used to
show that the spec is complete: on the tiny corpora its stdout, stderr and -p table are compared
byte for byte with the golden outputs of OMP_NUM_THREADS=1 fast_align. It also checks the spec's
description of libstdc++ unordered_map iteration order against the stlhash_probe test vectors.

    spec_check.py hash VECTORS_DIR                 check the hash emulation against probe vectors
    spec_check.py run GOLDEN_RUN_DIR [...]         rerun golden runs (dirs with cmd/out/err) and compare
    spec_check.py atools GOLDEN_CORPUS_DIR         check the atools model against atools.* golden dirs

libm functions are called through ctypes so that they are glibc's, as in fast_align.
This is a verification tool for the spec, not a reference for the Rust code (Step 5 writes its own).
"""

import ctypes
import math
import struct
import sys
from bisect import bisect_left
from pathlib import Path

_libm = ctypes.CDLL("libm.so.6")
for _name, _n in (("exp", 1), ("log", 1), ("pow", 2), ("lgamma", 1)):
    getattr(_libm, _name).restype = ctypes.c_double
    getattr(_libm, _name).argtypes = [ctypes.c_double] * _n
c_exp, c_log, c_pow, c_lgamma = _libm.exp, _libm.log, _libm.pow, _libm.lgamma

U32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------
# Number formatting: C++ ostream default for double == printf("%.6g")
def fmt_g(x: float) -> str:
    if math.isnan(x):
        return "-nan" if struct.pack(">d", x)[0] & 0x80 else "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.6g}"


# ---------------------------------------------------------------------------------------------
# libstdc++ unordered_map node order (spec section "libstdc++ unordered_map iteration order")
FAST_BKT = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13]
PRIMES: list = []  # filled from tests/data/stlhash/primes.txt
_HEAD = object()


class StlMap:
    """Iteration order of std::unordered_map with unique keys, max_load_factor 1."""

    __slots__ = ("buckets", "hash", "n", "nb", "next_resize", "nxt")

    def __init__(self, hash_fn=lambda k: k):
        self.hash = hash_fn
        self.nb = 1
        self.buckets = [None]
        self.nxt = {_HEAD: None}
        self.next_resize = 0
        self.n = 0

    def _next_bkt(self, n):
        if n < len(FAST_BKT):
            if n == 0:
                return 1
            self.next_resize = FAST_BKT[n]
            return FAST_BKT[n]
        last = len(PRIMES) - 2  # index of the last prime proper; PRIMES[-1] is a sentinel copy
        i = bisect_left(PRIMES, n, 6, last)
        self.next_resize = (1 << 64) - 1 if i == last else PRIMES[i]
        return PRIMES[i]

    def _rehash(self, nb):
        new = [None] * nb
        p = self.nxt[_HEAD]
        self.nxt[_HEAD] = None
        bbegin = 0
        while p is not None:
            nx = self.nxt[p]
            b = self.hash(p) % nb
            if new[b] is None:
                self.nxt[p] = self.nxt[_HEAD]
                self.nxt[_HEAD] = p
                new[b] = _HEAD
                if self.nxt[p] is not None:
                    new[bbegin] = p
                bbegin = b
            else:
                before = new[b]
                self.nxt[p] = self.nxt[before]
                self.nxt[before] = p
            p = nx
        self.buckets, self.nb = new, nb

    def insert(self, k):
        if k in self.nxt:
            return
        if self.n + 1 > self.next_resize:
            min_bkts = max(self.n + 1, 0 if self.next_resize else 11)
            if min_bkts >= self.nb:
                self._rehash(self._next_bkt(max(min_bkts + 1, self.nb * 2)))
            else:
                self.next_resize = self.nb
        b = self.hash(k) % self.nb
        before = self.buckets[b]
        if before is not None:
            self.nxt[k] = self.nxt[before]
            self.nxt[before] = k
        else:
            self.nxt[k] = self.nxt[_HEAD]
            self.nxt[_HEAD] = k
            if self.nxt[k] is not None:
                self.buckets[self.hash(self.nxt[k]) % self.nb] = k
            self.buckets[b] = _HEAD
        self.n += 1

    def order(self):
        out, p = [], self.nxt[_HEAD]
        while p is not None:
            out.append(p)
            p = self.nxt[p]
        return out


def pair_hash(k):
    a, b = k  # (short, short)
    return (((a & 0xFFFF) << 16) | (b & U32)) & U32


def load_primes(vec_dir: Path):
    PRIMES[:] = [int(x) for x in (vec_dir / "primes.txt").read_text().split()]


def check_hash(vec_dir: Path) -> int:
    load_primes(vec_dir)
    cases = (vec_dir / "cases.txt").read_text().splitlines()
    expected = (vec_dir / "expected.txt").read_text().splitlines()
    bad = 0
    for idx, line in enumerate(cases):
        kind, name, *toks = line.split()
        exp = expected[idx * 5 : idx * 5 + 5]
        if kind == "u32":
            m, keys = StlMap(), [int(t) for t in toks]
        else:
            m = StlMap(pair_hash)
            keys = [tuple(_short(int(v)) for v in t.split(",")) for t in toks]
        bkts = []
        for k in keys:
            m.insert(k)
            bkts.append(m.nb)
        fmt = (lambda k: f"{k[0]},{k[1]}") if kind == "pair" else str
        got_order = " ".join(fmt(k) for k in m.order())
        ok = (
            exp[0] == f"case {name} {kind} {len(keys)}"
            and exp[1] == ("bkt " + " ".join(map(str, bkts))).rstrip()
            and all(
                e == f"{tag} {got_order}".rstrip()
                for tag, e in zip(("order", "copy", "swap"), exp[2:])
            )
        )
        if not ok:
            bad += 1
            print(f"hash case {name}: MISMATCH")
    print(f"hash emulation: {len(cases) - bad}/{len(cases)} cases match the probe")
    return bad


def _short(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


# ---------------------------------------------------------------------------------------------
# Diagonal prior and helpers (spec sections "Diagonal alignment prior", "digamma, log_poisson")
def feature(i, j, m, n):
    return -abs(float(j) / n - float(i) / m) if (n and m) else _feature_slow(i, j, m, n)


def _feature_slow(i, j, m, n):
    a = float(j) / n if n else (math.nan if j == 0 else math.copysign(math.inf, 1.0))
    b = float(i) / m if m else (math.nan if i == 0 else math.inf)
    return -abs(a - b)


def unnorm(i, j, m, n, lam):
    return c_exp(feature(i, j, m, n) * lam)


def compute_z(i, m, n, lam):
    split = float(i) * n / m
    fl = int(split) & U32
    ce = (fl + 1) & U32
    ratio = c_exp(-lam / n) if n else 0.0
    num_top = (n - fl) & U32
    ezt = ezb = 0.0
    if num_top:
        ezt = unnorm(i, ce, m, n, lam) * (1.0 - c_pow(ratio, float(num_top))) / (1.0 - ratio)
    if fl:
        ezb = unnorm(i, fl, m, n, lam) * (1.0 - c_pow(ratio, float(fl))) / (1.0 - ratio)
    return ezb + ezt


def ags(a1, g1, r, d, n):
    g_np1 = g1 * c_pow(r, float(n))
    a_n = d * float((n - 1) & U32) + a1
    x_1 = a1 * g1
    g_2 = g1 * r
    rm1 = r - 1
    return (a_n * g_np1 - x_1) / rm1 - d * (g_np1 - g_2) / (rm1 * rm1)


def compute_dlogz(i, m, n, lam):
    z = compute_z(i, n, m, lam)  # m and n swapped, as fast_align does
    split = float(i) * n / m
    fl = int(split) & U32
    ce = (fl + 1) & U32
    ratio = c_exp(-lam / n)
    d = -1.0 / n
    num_top = (n - fl) & U32
    pct = pcb = 0.0
    if num_top:
        pct = ags(feature(i, ce, m, n), unnorm(i, ce, m, n, lam), ratio, d, num_top)
    if fl:
        pcb = ags(feature(i, fl, m, n), unnorm(i, fl, m, n, lam), ratio, d, fl)
    return (pct + pcb) / z


def digamma(x):
    r = 0.0
    while x < 7:
        r -= 1 / x
        x += 1.0
    x -= 0.5
    xx = 1.0 / x
    xx2 = xx * xx
    xx4 = xx2 * xx2
    r += (
        c_log(x)
        + (1.0 / 24.0) * xx2
        - (7.0 / 960.0) * xx4
        + (31.0 / 8064.0) * xx4 * xx2
        - (127.0 / 30720.0) * xx4 * xx4
    )
    return r


def log_poisson(x, lam):
    return c_log(lam) * float(x) - c_lgamma(float((x + 1) & U32)) - lam


# ---------------------------------------------------------------------------------------------
# Command line (spec section "Command line")
class Opts:
    def __init__(self):
        self.input = ""
        self.cpf = ""
        self.mult = 1.0
        self.reverse = False
        self.iters = 5
        self.diag = False
        self.beam = -4.0
        self.p0 = 0.08
        self.tension = 4.0
        self.opt = False
        self.vb = False
        self.alpha = 0.01
        self.no_null = False
        self.bufsize = 10000
        self.force = False
        self.scores = False


def parse_args(argv, err):
    o = Opts()
    with_arg = set("iIfmtqTapb")
    k = 0
    while k < len(argv):
        a = argv[k]
        assert a.startswith("-") and len(a) == 2, a  # the golden commands use separate short flags
        c = a[1]
        val = None
        if c in with_arg:
            k += 1
            val = argv[k]
        err.append(f"ARG={c}\n")
        if c == "i":
            o.input = val
        elif c == "r":
            o.reverse = True
        elif c == "I":
            o.iters = int(val)
        elif c == "d":
            o.diag = True
        elif c == "f":
            o.force, o.cpf = True, val
        elif c == "m":
            o.mult = float(val)
        elif c == "t":
            o.beam = float(val)
        elif c == "q":
            o.p0 = float(val)
        elif c == "T":
            o.diag, o.tension = True, float(val)
        elif c == "o":
            o.opt = True
        elif c == "v":
            o.vb = True
        elif c == "a":
            o.alpha = float(val)
        elif c == "N":
            o.no_null = True
        elif c == "p":
            o.cpf = val
        elif c == "b":
            o.bufsize = int(val)
        elif c == "s":
            o.scores = True
        k += 1
    return o


# ---------------------------------------------------------------------------------------------
# Input (spec sections "Input parsing", "Dictionary")
class Dict:
    def __init__(self):
        self.ids = {}
        self.words = [b"<bad0>"]
        self.kdiv = None

    def conv(self, w: bytes) -> int:
        i = self.ids.get(w)
        if i is None:
            self.words.append(w)
            i = self.ids[w] = len(self.words) - 1
        return i

    def parse(self, line: bytes):
        if self.kdiv is None:
            self.kdiv = self.conv(b"|||")
        out, tok = [], bytearray()
        for ch in line:
            if ch in (0x20, 0x09):
                if tok:
                    out.append(self.conv(bytes(tok)))
                    tok.clear()
                if ch == 0x09:
                    out.append(self.kdiv)
            else:
                tok.append(ch)
        if tok:
            out.append(self.conv(bytes(tok)))
        if self.kdiv in out:
            k = out.index(self.kdiv)
            return out[:k], out[k + 1 :]
        return out, []


def read_lines(path) -> list:
    data = Path(path).read_bytes()
    parts = data.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


# ---------------------------------------------------------------------------------------------
def train(o: Opts, out: list, err: list, table_out: list):
    use_null = not o.no_null
    pnn = 1.0 - o.p0
    d = Dict()
    knull = d.conv(b"<eps>")
    lines = read_lines(o.input)
    rows: dict = {}  # e -> StlMap of target ids
    bufsize = o.bufsize & ((1 << 64) - 1)

    def row(e):
        r = rows.get(e)
        if r is None:
            r = rows[e] = StlMap()
        return r

    # initial pass
    err.append("INITIAL PASS \n")
    tot_len_ratio = 0.0
    ntt = 0.0
    sc_map = StlMap(pair_hash)
    sc_count: dict = {}
    ins_buf: dict = {}
    items = 0
    max_e = -1
    flag = False
    parsed = []
    for lc, line in enumerate(lines, 1):
        if lc % 1000 == 0:
            err.append(".")
            flag = True
        if lc % 50000 == 0:
            err.append(f" [{lc}]\n")
            flag = False
        src, trg = d.parse(line)
        if o.reverse:
            src, trg = trg, src
        parsed.append((src, trg))
        if not src or not trg:
            err.append(f"Error in line {lc}\n")
            err.append(line.decode("latin-1") + "\n")
        tot_len_ratio += _div(float(len(trg)), float(len(src)))
        ntt += len(trg)
        if use_null:
            for f in trg:
                row(knull).insert(f)
        for e in src:
            max_e = max(max_e, e)
            ins_buf.setdefault(e, []).extend(trg)
            items += len(trg)
        if items > bufsize * 100:
            items = 0
            for e in sorted(ins_buf):
                for f in ins_buf[e]:
                    row(e).insert(f)
            ins_buf.clear()
        key = (_short(len(trg)), _short(len(src)))
        sc_map.insert(key)
        sc_count[key] = sc_count.get(key, 0) + 1
    for e in sorted(ins_buf):
        for f in ins_buf[e]:
            row(e).insert(f)
    size_counts = [(k, sc_count[k]) for k in sc_map.order()]
    n_rows = max([max_e + 1] + [knull + 1 if knull in rows else 0])
    mean = _div(tot_len_ratio, float(len(lines)))
    if flag:
        err.append("\n")
    err.append(f"expected target length = source length * {fmt_g(mean)}\n")

    order = {e: r.order() for e, r in rows.items()}
    prob = {
        e: dict.fromkeys(ks, 0.0) for e, ks in order.items()
    }  # ttable after Freeze: copies of 0
    cnt = {e: dict.fromkeys(ks, 0.0) for e, ks in order.items()}
    probs_init = False
    tension = o.tension

    for it in range(o.iters):
        final = it == o.iters - 1
        err.append(f"ITERATION {it + 1}{' (FINAL)' if final else ''}\n")
        likelihood = c0 = emp_feat = 0.0
        flag = False
        start = 0
        for lc in range(1, len(lines) + 1):
            if lc % 1000 == 0:
                err.append(".")
                flag = True
            if lc % 50000 == 0:
                err.append(f" [{lc}]\n")
                flag = False
            if lc - start >= bufsize or lc == len(lines):
                bl, bc0, bfe = 0.0, 0.0, 0.0
                for idx in range(start, lc):
                    src, trg = parsed[idx]
                    if not src or not trg:
                        err.append(f"Error in line {lc}\n" + lines[idx].decode("latin-1") + "\n")
                    m, n = len(trg), len(src)
                    ll = 0.0
                    links = []
                    for j, f in enumerate(trg):
                        probs = [0.0] * (n + 1)
                        s = 0.0
                        pai = _div(1.0, float(n + use_null))
                        if use_null:
                            if o.diag:
                                pai = o.p0
                            probs[0] = (prob[knull][f] if probs_init else 1e-9) * pai
                            s += probs[0]
                        az = 0.0
                        if o.diag:
                            az = compute_z(j + 1, m, n, tension) / pnn
                        for i in range(1, n + 1):
                            if o.diag:
                                pai = unnorm(j + 1, i, m, n, tension) / az
                            probs[i] = (prob[src[i - 1]][f] if probs_init else 1e-9) * pai
                            s += probs[i]
                        if final:
                            best, best_p = (0, probs[0]) if use_null else (-1, -1.0)
                            for i in range(1, n + 1):
                                if probs[i] > best_p:
                                    best, best_p = i, probs[i]
                            if best > 0:
                                links.append(f"{j}-{best - 1}" if o.reverse else f"{best - 1}-{j}")
                        else:
                            if use_null:
                                c = probs[0] / s
                                bc0 += c
                                cnt[knull][f] += c
                            for i in range(1, n + 1):
                                p = probs[i] / s
                                cnt[src[i - 1]][f] += p
                                bfe += feature(j, i, m, n) * p
                        ll += c_log(s)
                    bl += ll
                    if final:
                        line_out = " ".join(links)
                        if o.scores:
                            lp = log_poisson(m, 0.05 + float(n) * mean) + ll
                            line_out += " ||| " + fmt_g(lp)
                        out.append(line_out + "\n")
                emp_feat += bfe
                c0 += bc0
                likelihood += bl
                start = lc
        b2 = likelihood / c_log(2.0)
        if flag:
            err.append("\n")
        emp_feat /= ntt
        err.append(f"  log_e likelihood: {fmt_g(likelihood)}\n")
        err.append(f"  log_2 likelihood: {fmt_g(b2)}\n")
        err.append(f"     cross entropy: {fmt_g(-b2 / ntt)}\n")
        err.append(f"        perplexity: {fmt_g(c_pow(2.0, -b2 / ntt))}\n")
        err.append(f"      posterior p0: {fmt_g(c0 / ntt)}\n")
        err.append(f" posterior al-feat: {fmt_g(emp_feat)}\n")
        err.append(f"       size counts: {len(size_counts)}\n")
        if not final:
            if o.diag and o.opt and it > 0:
                for ii in range(8):
                    mod = 0.0
                    for (tl, sl), count in size_counts:
                        for j in range(1, tl + 1):
                            mod += float(count) * compute_dlogz(j, tl & U32, sl & U32, tension)
                    mod /= ntt
                    err.append(
                        f"  {ii + 1}  model al-feat: {fmt_g(mod)} (tension={fmt_g(tension)})\n"
                    )
                    tension += (emp_feat - mod) * 20.0
                    if tension <= 0.1:  # noqa: PLR1730 (max() would drop a NaN)
                        tension = 0.1
                    if tension > 14:
                        tension = 14.0
                err.append(f"     final tension: {fmt_g(tension)}\n")
            for e, ks in order.items():
                ce = cnt[e]
                pe = prob[e]
                tot = 0.0
                if o.vb:
                    for k in ks:
                        tot += ce[k] + o.alpha
                    if tot == 0:
                        tot = 1.0
                    dt = digamma(tot)
                    for k in ks:
                        pe[k] = c_exp(digamma(ce[k] + o.alpha) - dt)
                else:
                    for k in ks:
                        tot += ce[k]
                    if tot == 0:
                        tot = 1.0
                    for k in ks:
                        pe[k] = ce[k] / tot
                for k in ks:
                    ce[k] = 0.0
            probs_init = True
    if o.cpf:
        err.append(f"conditional probabilities: {o.cpf}\n")
        for e in range(n_rows):
            if e not in order:
                continue
            ks = order[e]
            max_p = -1.0
            for k in ks:
                if prob[e][k] > max_p:  # noqa: PLR1730 (max() would drop a NaN)
                    max_p = prob[e][k]
            thr = -c_log(max_p) * o.beam
            for k in ks:
                c = c_log(prob[e][k])
                if c >= thr:
                    table_out.append(
                        d.words[e] + b"\t" + d.words[k] + b"\t" + fmt_g(c).encode() + b"\n"
                    )


def _div(a, b):
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return -math.nan  # x86 default NaN has the sign bit set
        return math.copysign(math.inf, a)
    return a / b


def force_align(o: Opts, out: list, err: list):
    use_null = not o.no_null
    pnn = 1.0 - o.p0
    d = Dict()
    knull = d.conv(b"<eps>")
    table: dict = {}
    toks = Path(o.cpf).read_bytes().split()
    c = 0
    for k in range(0, len(toks) - 2, 3):
        ie = d.conv(toks[k])
        table.setdefault(ie, {})[d.conv(toks[k + 1])] = c_exp(float(toks[k + 2]))
        c += 1
    if c:
        c += 1  # the last record is read twice (stream state checked before, not after, >>)
    err.append(f"Loaded {c} translation parameters.\n")
    n_rows = max(table) + 1 if table else 0

    def safe_prob(e, f):
        if e < n_rows:
            return table.get(e, {}).get(f, 1e-9)
        return 1e-9

    tlp = 0.0
    for lc, line in enumerate(read_lines(o.input), 1):
        src, trg = d.parse(line)
        s_out = b"".join(d.words[s] + b" " for s in src) + b"|||"
        s_out += b"".join(b" " + d.words[t] for t in trg) + b" |||"
        if o.reverse:
            src, trg = trg, src
        if not src or not trg:
            out.append(s_out)
            err.append(f"Error in line {lc}\n")
            return 1
        m, n = len(trg), len(src)
        lp = log_poisson(m, 0.05 + float(n) * o.mult)
        for j, f in enumerate(trg):
            s = 0.0
            aj = 0
            maxp = 0.0
            pai = 1.0 / float(n + use_null)
            if use_null:
                if o.diag:
                    pai = o.p0
                maxp = safe_prob(knull, f) * pai
                s += maxp
            az = 0.0
            if o.diag:
                az = compute_z(j + 1, m, n, o.tension) / pnn
            for i in range(1, n + 1):
                if o.diag:
                    pai = unnorm(j + 1, i, m, n, o.tension) / az
                p = safe_prob(src[i - 1], f) * pai
                if p > maxp:
                    maxp, aj = p, i
                s += p
            lp += c_log(s)
            if aj > 0:
                s_out += (f" {j}-{aj - 1}" if o.reverse else f" {aj - 1}-{j}").encode()
        tlp += lp
        out.append(s_out + b" ||| " + fmt_g(lp).encode() + b"\n")
    err.append(f"TOTAL LOG PROB {fmt_g(tlp)}\n")
    return 0


def run_golden(gdir: Path) -> bool:
    argv = (gdir / "cmd").read_text().split()[1:]
    out: list = []
    err: list = []
    table: list = []
    o = parse_args(argv, err)
    if o.force:
        force_align(o, out, err)
        got_out = b"".join(out)
    else:
        train(o, out, err, table)
        got_out = "".join(out).encode("latin-1")
    ok = True
    if got_out != (gdir / "out").read_bytes():
        ok = False
        _first_diff("stdout", got_out, (gdir / "out").read_bytes())
    got_err = "".join(err).encode("latin-1")
    if got_err != (gdir / "err").read_bytes():
        ok = False
        _first_diff("stderr", got_err, (gdir / "err").read_bytes())
    if o.cpf and not o.force and b"".join(table) != (gdir / "table").read_bytes():
        ok = False
        _first_diff("table", b"".join(table), (gdir / "table").read_bytes())
    print(f"{'OK  ' if ok else 'FAIL'} {gdir}", flush=True)
    return ok


def _first_diff(what, a: bytes, b: bytes):
    la, lb = a.split(b"\n"), b.split(b"\n")
    for k, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            print(f"  {what} line {k + 1}:\n    spec:   {x[:200]!r}\n    golden: {y[:200]!r}")
            return
    print(f"  {what}: lengths differ ({len(la)} vs {len(lb)} lines)")


# ---------------------------------------------------------------------------------------------
# atools (spec section "atools")
def read_grid(line: str):
    pts = []
    pos = line.rfind(" ||| ")
    body = line[pos + 5 :] if pos >= 0 else line
    for tok in body.split():
        a, b = tok.split("-")
        pts.append((int(a), int(b)))
    w = max([p[0] for p in pts], default=0) + 1
    h = max([p[1] for p in pts], default=0) + 1
    return set(pts), w, h


def atools(cmd: str, l1: str, l2):
    a, aw, ah = read_grid(l1)
    if cmd == "invert":
        return sorted((j, i) for i, j in a)
    b, bw, bh = read_grid(l2)
    w, h = max(aw, bw), max(ah, bh)
    if cmd == "intersect":
        return sorted(a & b)
    if cmd == "union":
        return sorted(a | b)
    res = set(a & b)
    ia = {i for i, _ in res}
    ja = {j for _, j in res}
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    def align(i, j):
        res.add((i, j))
        ia.add(i)
        ja.add(j)

    pending = sorted((a | b) - res)
    while True:
        added = []
        for i, j in pending:
            if (i not in ia or j not in ja) and any((i + di, j + dj) in res for di, dj in nbrs):
                align(i, j)
                added.append((i, j))
        if not added:
            break
        pending = [p for p in pending if p not in set(added)]
    if cmd in ("grow-diag-final", "grow-diag-final-and"):
        both = cmd == "grow-diag-final-and"
        for src in (a, b):
            for i, j in sorted(src):
                if (i, j) in res:
                    continue
                ok = (i not in ia and j not in ja) if both else (i not in ia or j not in ja)
                if ok:
                    align(i, j)
    assert w and h
    return sorted(res)


def check_atools(cdir: Path) -> bool:
    f = (cdir / "fwd.default" / "out").read_text("latin-1").split("\n")[:-1]
    r = (cdir / "rev.default" / "out").read_text("latin-1").split("\n")[:-1]
    ok = True
    for a in sorted(cdir.glob("atools.*")):
        cmd = a.name.split(".", 1)[1]
        got = "".join(
            " ".join(f"{i}-{j}" for i, j in atools(cmd, x, y)) + "\n" for x, y in zip(f, r)
        )
        good = got == (a / "out").read_text("latin-1")
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {a}")
    return ok


def main():
    mode, *args = sys.argv[1:]
    if mode == "hash":
        sys.exit(1 if check_hash(Path(args[0])) else 0)
    if mode == "run":
        vec = Path(__file__).resolve().parent.parent / "crates/fastalign-core/tests/data/stlhash"
        load_primes(vec)
        results = [run_golden(Path(a)) for a in args]  # run all, report every failure
        sys.exit(0 if all(results) else 1)
    if mode == "atools":
        results = [check_atools(Path(a)) for a in args]  # run all, report every failure
        sys.exit(0 if all(results) else 1)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
