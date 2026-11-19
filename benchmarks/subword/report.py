#!/usr/bin/env python3
"""Tables and curves for ``benchmarks/SUBWORD.md`` from the sweep (Step 13).

    report.py tables CORPUS [--pretok ns] [--system grow-diag-final-and]
    report.py orders CORPUS --type bpe --mode joint [--pretok ns]
    report.py methods CORPUS RUN
    report.py cost CORPUS [--pretok ns]
    report.py svg CORPUS... [--pretok ns] [--out benchmarks/subword/figures]

Every AER difference is against the word-level run of the same corpus, with its paired-
bootstrap 95 % interval (``bootstrap.py``); ``*`` marks an interval that excludes 0.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "python")]
import bootstrap as bs

MODES = ["joint", "separate", "src", "tgt"]
TYPES = ["bpe", "unigram"]


def rows(corpus: str) -> list[dict]:
    with open(bs.SWEEP / "sweep.tsv") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["corpus"] == corpus]


def run_name(corpus, typ, vocab, mode, pretok):
    tag = f"-{pretok}" if pretok else ""
    return f"{corpus}-{typ}{vocab}-{mode}{tag}"


class Diff:
    def __init__(self, corpus: str):
        self.corpus = corpus
        self.g = bs.gold(corpus)
        self.idx = bs.draws(len(self.g[0]))
        self.runs = {r["run"]: r for r in rows(corpus)}

    def __call__(self, run: str, system: str, order: str = "proj-sym", base: str | None = None,
                 base_order: str = "proj-sym"):  # fmt: skip
        base = base or f"{self.corpus}-word"
        if run not in self.runs:
            return None
        ca = bs.run_counts(self.corpus, base, f"{base_order}/{system}", self.g)
        cb = bs.run_counts(self.corpus, run, f"{order}/{system}", self.g)
        return bs.paired(ca, cb, self.idx)


def fmt(d, with_ci=False) -> str:
    if d is None:
        return "–"
    va, vb, lo, hi = d
    star = "*" if lo > 0 or hi < 0 else ""
    s = f"{vb:.2f} ({vb - va:+.2f}{star})"
    return s + f" [{lo:+.2f}, {hi:+.2f}]" if with_ci else s


def vocabs(corpus) -> list[int]:
    return sorted({int(r["vocab"]) for r in rows(corpus) if r["type"] != "word"})


def cmd_tables(a) -> int:
    diff = Diff(a.corpus)
    word = diff(f"{a.corpus}-word", a.system)
    print(f"{a.corpus}, {a.system}, project-then-symmetrise; word level {word[1]:.2f}\n")
    for typ in TYPES:
        head = ["vocab"] + [f"{typ} {m}" for m in MODES] + ["pieces/word (separate)"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for v in vocabs(a.corpus):
            cells = [f"{v // 1000}k" if v >= 1000 else str(v)]
            for m in MODES:
                cells.append(fmt(diff(run_name(a.corpus, typ, v, m, a.pretok), a.system)))
            r = diff.runs.get(run_name(a.corpus, typ, v, "separate", a.pretok))
            cells.append(f"{float(r['src_ppw']):.2f} / {float(r['tgt_ppw']):.2f}" if r else "–")
            print("| " + " | ".join(cells) + " |")
        print()
    return 0


def cmd_orders(a) -> int:
    diff = Diff(a.corpus)
    systems = ["intersect", "grow-diag", "grow-diag-final", "grow-diag-final-and", "union"]
    head = ["vocab"] + [f"{s}: p→s / s→p (s→p − p→s)" for s in systems]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for v in vocabs(a.corpus):
        run = run_name(a.corpus, a.type, v, a.mode, a.pretok)
        if run not in diff.runs:
            continue
        cells = [f"{v // 1000}k" if v >= 1000 else str(v)]
        for s in systems:
            d = diff(run, s, "sym-proj", base=run, base_order="proj-sym")
            star = "*" if d[2] > 0 or d[3] < 0 else ""
            cells.append(f"{d[0]:.2f} / {d[1]:.2f} ({d[1] - d[0]:+.2f}{star})")
        print("| " + " | ".join(cells) + " |")
    return 0


def cmd_methods(a) -> int:
    diff = Diff(a.corpus)
    systems = ["fwd", "rev", "intersect", "union", "grow-diag", "grow-diag-final",
               "grow-diag-final-and"]  # fmt: skip
    print("| system | word P / R / AER | subword P / R / AER | ΔAER [95 % CI] |")
    print("|---|---|---|---|")
    by = {(r["order"], r["system"]): r for r in rows(a.corpus) if r["run"] == a.run}
    wb = {(r["order"], r["system"]): r for r in rows(a.corpus) if r["run"] == f"{a.corpus}-word"}
    for s in systems:
        w, r = wb[("proj-sym", s)], by[("proj-sym", s)]
        d = diff(a.run, s)
        star = "*" if d[2] > 0 or d[3] < 0 else ""
        print(f"| {s} | {w['precision']} / {w['recall']} / {w['aer']} | "
              f"{r['precision']} / {r['recall']} / {r['aer']} | "
              f"{d[1] - d[0]:+.2f}{star} [{d[2]:+.2f}, {d[3]:+.2f}] |")  # fmt: skip
    return 0


def cmd_cost(a) -> int:
    rs = {r["run"]: r for r in rows(a.corpus)}
    w = rs[f"{a.corpus}-word"]
    wt = int(w["src_tokens"]) + int(w["tgt_tokens"])
    print("| run | tokens (× word) | aligner CPU s (× word) | aligner peak RSS MB | "
          "tokeniser training s |")  # fmt: skip
    print("|---|---|---|---|---|")
    print(f"| word | {wt / 1e6:.2f} M | {float(w['cpu_align']):.1f} | {w['rss_align_mb']} | – |")
    for typ in TYPES:
        for m in ("joint", "separate"):
            for v in vocabs(a.corpus):
                r = rs.get(run_name(a.corpus, typ, v, m, a.pretok))
                if not r:
                    continue
                t = int(r["src_tokens"]) + int(r["tgt_tokens"])
                print(f"| {typ} {v // 1000}k {m} | {t / 1e6:.2f} M ({t / wt:.2f}×) | "
                      f"{float(r['cpu_align']):.1f} ({float(r['cpu_align']) / float(w['cpu_align']):.2f}×) | "
                      f"{r['rss_align_mb']} | {r['t_tok']} |")  # fmt: skip
    return 0


# -- SVG ------------------------------------------------------------------------------------

SHORT = {"grow-diag-final-and": "gdfa", "grow-diag-final": "gdf"}
COLOURS = {"bpe joint": "#1f6fb2", "bpe separate": "#6aa9de", "unigram joint": "#c0392b",
           "unigram separate": "#e8927f", "bpe src": "#2e8b57", "bpe tgt": "#9b59b6"}  # fmt: skip


def svg(corpora: list[str], pretok: str, system: str) -> str:
    w, h, ml, mr, mt, mb = 360, 260, 52, 12, 28, 40
    panels = []
    series_all = list(COLOURS)
    for k, corpus in enumerate(corpora):
        diff = Diff(corpus)
        vs = vocabs(corpus)
        data = {}
        for name in series_all:
            typ, mode = name.split()
            pts = []
            for v in vs:
                d = diff(run_name(corpus, typ, v, mode, pretok), system)
                if d:
                    pts.append((v, d[1] - d[0], d[2], d[3]))
            data[name] = pts
        ys = [y for p in data.values() for _, y, lo, hi in p for y in (lo, hi)]
        ymin, ymax = math.floor(min(ys + [-1])), math.ceil(max(ys + [1]))
        ymax = min(ymax, 8)
        x0, x1 = math.log2(min(vs)), math.log2(max(vs))

        def X(v, x0=x0, x1=x1):
            return ml + (math.log2(v) - x0) / (x1 - x0) * (w - ml - mr)

        def Y(y, ymin=ymin, ymax=ymax):
            y = max(min(y, ymax), ymin)
            return mt + (ymax - y) / (ymax - ymin) * (h - mt - mb)

        g = [f'<g transform="translate({k * (w + 10)},0)">']
        g.append(
            f'<text x="{ml}" y="16" class="t">{corpus}: Δ AER vs words, {SHORT.get(system, system)} (lower = better)</text>'
        )
        for y in range(ymin, ymax + 1):
            cls = "z" if y == 0 else "grid"
            g.append(
                f'<line x1="{ml}" x2="{w - mr}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" class="{cls}"/>'
            )
            g.append(
                f'<text x="{ml - 6}" y="{Y(y) + 4:.1f}" class="l" text-anchor="end">{y:+d}</text>'
            )
        for v in vs:
            lab = f"{v // 1000}k" if v >= 1000 else str(v)
            g.append(
                f'<text x="{X(v):.1f}" y="{h - mb + 16}" class="l" text-anchor="middle">{lab}</text>'
            )
        g.append(f'<text x="{(ml + w - mr) / 2}" y="{h - 6}" class="l" text-anchor="middle">'
                 "vocabulary size (log scale)</text>")  # fmt: skip
        for name, pts in data.items():
            c = COLOURS[name]
            if not pts:
                continue
            path = " ".join(f"{X(v):.1f},{Y(y):.1f}" for v, y, _, _ in pts)
            g.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="1.8"/>')
            for v, y, lo, hi in pts:
                g.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" '
                         f'stroke="{c}" stroke-width="1" opacity="0.5"/>')  # fmt: skip
                g.append(f'<circle cx="{X(v):.1f}" cy="{Y(y):.1f}" r="2.4" fill="{c}"/>')
        g.append("</g>")
        panels.append("\n".join(g))
    total_w = len(corpora) * (w + 10)
    legend = []
    for i, name in enumerate(series_all):
        x = 10 + (i % 6) * 120
        legend.append(f'<rect x="{x}" y="{h + 6}" width="12" height="3" fill="{COLOURS[name]}"/>'
                      f'<text x="{x + 16}" y="{h + 11}" class="l">{name}</text>')  # fmt: skip
    style = (".t{font:600 12px sans-serif;fill:#222}.l{font:10px sans-serif;fill:#444}"
             ".grid{stroke:#ddd;stroke-width:1}.z{stroke:#555;stroke-width:1.2}")  # fmt: skip
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{h + 20}" '
            f'viewBox="0 0 {total_w} {h + 20}"><style>{style}</style>'
            f'<rect width="100%" height="100%" fill="#fff"/>' + "\n".join(panels)
            + "\n".join(legend) + "</svg>\n")  # fmt: skip


def cmd_svg(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = out / f"{'-'.join(a.corpora)}-{a.pretok or 'default'}.svg"
    name.write_text(svg(a.corpora, a.pretok, a.system))
    print(name)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tables", "orders", "methods", "cost", "svg"):
        p = sub.add_parser(name)
        if name == "svg":
            p.add_argument("corpora", nargs="+")
            p.add_argument("--out", default=str(HERE / "figures"))
        else:
            p.add_argument("corpus")
        if name == "methods":
            p.add_argument("run")
        p.add_argument("--pretok", default="ns")
        p.add_argument("--system", default="grow-diag-final-and")
        p.add_argument("--type", default="bpe")
        p.add_argument("--mode", default="joint")
    a = ap.parse_args(argv)
    if a.pretok == "default":
        a.pretok = ""
    return {"tables": cmd_tables, "orders": cmd_orders, "methods": cmd_methods,
            "cost": cmd_cost, "svg": cmd_svg}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
