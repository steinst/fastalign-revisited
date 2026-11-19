"""Split a perf profile of a fastalign run into phases and draw one flamegraph per phase.

Record with the frame-pointer profiling build (see Cargo.toml, ``[profile.profiling]``):

    RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=target/prof \\
        cargo build --profile profiling -p fastalign-core --bin fastalign --example phases
    perf record -F HZ -g -o DIR/perf.data -- target/prof/profiling/fastalign train ...

Both directions train concurrently, so phases overlap in time; each sample is therefore assigned
to a phase by its *stack* (the innermost frame that names a phase function, e.g.
``Trainer::phase_a``), not by a time window as gizapp's ``profile_stages.py`` does.

Writes ``OUT/TAG-PHASE.svg`` (inferno flamegraph), ``OUT/TAG-PHASE.folded.gz`` (collapsed stacks,
inlined frames included, generic arguments stripped), ``OUT/TAG-all.svg`` and
``OUT/TAG-summary.md`` (samples per phase, top functions by self and inclusive samples).
Needs ``perf`` and ``inferno-flamegraph`` (``cargo install inferno``) on PATH.

Usage: python benchmarks/profile_phases.py PERF_DATA OUT TAG [--top 15]
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
from collections import Counter
from pathlib import Path

# (phase, regex on a cleaned frame), in priority order: a sample belongs to the first phase any
# of whose patterns matches any frame of its stack. Inlined closures lose their parents' names
# (`{closure#0}` under a rayon frame), so the per-line functions are the markers; the prior
# functions count as "prior" only when no E-step/decode/M-step frame is on the stack.
PHASES = [
    ("e_step", re.compile(r"Trainer::phase_a|e_step_line")),
    ("decode", re.compile(r"Trainer::decode|decode_line")),
    ("m_step", re.compile(r"Trainer::phase_b|train::accumulate|accumulate_lines|normalise$")),
    ("tension", re.compile(r"Trainer::optimize_tension|compute_dlogz")),
    ("prior", re.compile(r"Trainer::build_prior|compute_z|unnormalized_prob")),
    ("initial_pass", re.compile(r"ttable::|initial_pass|StlOrder")),
    ("setup", re.compile(r"RowLines|LengthPairs|Trainer::new")),
    ("load", re.compile(r"corpus::|read_corpus")),
    ("symmetrize", re.compile(r"symmetrize")),
    ("model", re.compile(r"DirModel::from_trainer|io::Model|Model::save|write_text_table")),
]

GENERIC = re.compile(r"(?<=[\w)\]}])(?:::)?<[^<>]*>")
_cache: dict[str, str] = {}


def clean_symbol(sym: str) -> str:
    """Drop offsets, the "(inlined)" marker, hashes and generic arguments."""
    out = _cache.get(sym)
    if out is None:
        out = re.sub(r"\+0x[0-9a-f]+$", "", sym).removesuffix(" (inlined)")
        prev = None
        while prev != out:
            prev, out = out, GENERIC.sub("", out)
        out = re.sub(r"::h[0-9a-f]{16}$", "", out).replace(";", ",")
        _cache[sym] = out
    return out


def phase_of(frames: list[str]) -> str:
    for name, rx in PHASES:
        if any(rx.search(f) for f in frames):
            return name
    return "other"


def collapse(perf_data: Path) -> dict[str, Counter]:
    """Stream `perf script` once; fold each sample's stack into its phase's counter."""
    folded: dict[str, Counter] = {}
    proc = subprocess.Popen(
        ["perf", "script", "-i", str(perf_data), "-F", "comm,tid,ip,sym"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None
    frames: list[str] = []
    comm = ""

    def flush() -> None:
        if not frames:
            return
        phase = phase_of(frames)
        # Cut at the innermost rayon frame (rayon's recursive join makes stacks deep and unique).
        cut = next((k for k, f in enumerate(frames) if f.startswith("rayon")), None)
        kept = frames[:cut] if cut is not None else frames
        out = ["[rayon task]"] if cut is not None else []
        for f in reversed(kept):
            if not out or out[-1] != f:
                out.append(f)
        folded.setdefault(phase, Counter())[";".join([comm, *out])] += 1

    for line in proc.stdout:
        if not line.strip():
            continue
        if line[0] not in " \t":
            flush()
            frames = []
            comm = "fastalign" if line.split()[0].startswith(("fastalign", "phases")) else "worker"
            continue
        parts = line.split(None, 1)
        sym = parts[1].strip() if len(parts) > 1 else "[unknown]"
        frames.append(clean_symbol(sym))
    flush()
    proc.wait()
    return folded


def top_functions(lines: list[str], k: int) -> tuple[int, list, list]:
    self_c: Counter = Counter()
    incl: Counter = Counter()
    total = 0
    for line in lines:
        stack, n = line.rsplit(" ", 1)
        n = int(n)
        total += n
        frames = stack.split(";")[1:]
        if not frames:
            continue
        self_c[frames[-1]] += n
        for f in set(frames):
            incl[f] += n
    return total, self_c.most_common(k), incl.most_common(k)


def short(name: str, width: int = 110) -> str:
    name = name.replace("fastalign_core::", "")
    return name if len(name) <= width else name[: width - 1] + "…"


def flamegraph(lines: list[str], title: str, path: Path) -> None:
    with open(path, "w") as svg:
        subprocess.run(
            ["inferno-flamegraph", "--title", title, "--minwidth", "0.2"],
            input="\n".join(lines) + "\n",
            stdout=svg,
            text=True,
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("perf_data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    phases = collapse(a.perf_data)
    grand = sum(sum(c.values()) for c in phases.values())
    order = [p for p, _ in PHASES] + ["other"]
    summary = [
        f"# Profile {a.tag}\n",
        (
            f"`{a.perf_data}`: {grand} samples. A sample's phase is the first phase (in the order "
            "of `PHASES`) with a marker function on its stack (`profile_phases.py`). Percentages "
            "in the per-phase tables are of the phase.\n"
        ),
        "| phase | samples | share |\n|---|---|---|",
    ]
    for p in order:
        if p in phases:
            n = sum(phases[p].values())
            summary.append(f"| {p} | {n} | {100 * n / grand:.1f} % |")
    all_lines: list[str] = []
    for p in order:
        counter = phases.get(p)
        if not counter:
            continue
        lines = [f"{k} {v}" for k, v in counter.items()]
        all_lines.extend(f"{p};{k.split(';', 1)[1] if ';' in k else k} {v}" for k, v in counter.items())
        with gzip.open(a.out / f"{a.tag}-{p}.folded.gz", "wt") as g:
            g.write("\n".join(lines) + "\n")
        flamegraph(lines, f"fastalign {a.tag} {p}", a.out / f"{a.tag}-{p}.svg")
        total, top_self, top_incl = top_functions(lines, a.top)
        summary.append(f"\n## {p} ({total} samples)\n")
        summary.append("| self % | function |\n|---|---|")
        summary.extend(f"| {100 * n / total:.1f} | `{short(fn)}` |" for fn, n in top_self)
        summary.append("\n| inclusive % | function |\n|---|---|")
        summary.extend(f"| {100 * n / total:.1f} | `{short(fn)}` |" for fn, n in top_incl)
    flamegraph(all_lines, f"fastalign {a.tag} (by phase)", a.out / f"{a.tag}-all.svg")
    (a.out / f"{a.tag}-summary.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {a.out}/{a.tag}-*")


if __name__ == "__main__":
    main()
