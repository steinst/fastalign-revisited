//! Validation and timing tool for Step 4 (not part of the public CLI).
//!
//!   dump_tables -i CORPUS [-r] [-N] [-b N] [--threads N]      dump one direction
//!   dump_tables -i CORPUS --both [--threads N]                  time load + both tables
//!
//! The dump has the format of the instrumented fast_align (`benchmarks/instrument.patch`, spec
//! §20): `initial`, `dict`, `size_counts` and `row` lines. `benchmarks/check_ttable.sh` compares
//! them. `-b` is accepted and ignored: it does not change the table (spec §6).
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::ttable::{initial_pass, InitialPass, TableOptions};

/// C99 `%a` as printed by glibc for a normal or zero double.
fn hexfloat(x: f64) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { "nan" }.into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x.is_infinite() {
        return format!("{sign}inf");
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mant = bits & ((1u64 << 52) - 1);
    let (lead, e) = match exp {
        0 if mant == 0 => return format!("{sign}0x0p+0"),
        0 => (0, -1022),
        _ => (1, exp - 1023),
    };
    let mut m = format!("{mant:013x}");
    while m.ends_with('0') {
        m.pop();
    }
    let dot = if m.is_empty() {
        String::new()
    } else {
        format!(".{m}")
    };
    let es = if e >= 0 {
        format!("+{e}")
    } else {
        e.to_string()
    };
    format!("{sign}0x{lead}{dot}p{es}")
}

fn dump(c: &Corpus, ip: &InitialPass) -> std::io::Result<()> {
    let mut o = BufWriter::new(std::io::stdout().lock());
    writeln!(
        o,
        "initial n_target_tokens {} tot_len_ratio {} mean_srclen_multiplier {} vocab {} rows {}",
        hexfloat(ip.n_target_tokens),
        hexfloat(ip.tot_len_ratio),
        hexfloat(ip.mean_srclen_multiplier),
        c.vocab.len() - 1,
        ip.table.n_rows()
    )?;
    for id in 1..c.vocab.len() as u32 {
        write!(o, "dict {id} ")?;
        o.write_all(c.vocab.word(id))?;
        o.write_all(b"\n")?;
    }
    write!(o, "size_counts {}", ip.size_counts.len())?;
    for s in &ip.size_counts {
        write!(o, " {},{},{}", s.m, s.n, s.count)?;
    }
    writeln!(o)?;
    let t = &ip.table;
    for e in 0..t.n_rows() as u32 {
        write!(o, "row {e} {} {}", t.bucket_count(e), t.row(e).len())?;
        for k in t.iter_hash_order(e) {
            write!(o, " {}", t.cols()[k])?;
        }
        writeln!(o)?;
    }
    o.flush()
}

fn main() {
    let mut input = None;
    let mut reverse = false;
    let mut use_null = true;
    let mut both = false;
    let mut threads = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-i" => input = args.next().map(PathBuf::from),
            "-r" => reverse = true,
            "-N" => use_null = false,
            "-b" => {
                args.next();
            }
            "--both" => both = true,
            "--threads" => threads = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
    }
    let input = input.expect("-i CORPUS");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let t0 = Instant::now();
    let c = Corpus::read_joined(&input, &LoadOptions::default()).unwrap();
    let t_load = t0.elapsed();
    if both {
        let t1 = Instant::now();
        let (f, r) = rayon::join(
            || {
                initial_pass(
                    &c,
                    TableOptions {
                        reverse: false,
                        use_null,
                    },
                )
            },
            || {
                initial_pass(
                    &c,
                    TableOptions {
                        reverse: true,
                        use_null,
                    },
                )
            },
        );
        let t_tab = t1.elapsed();
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        eprintln!(
            "threads {} lines {} vocab {} load {:.3}s tables {:.3}s total {:.3}s",
            rayon::current_num_threads(),
            c.len(),
            c.vocab.len(),
            t_load.as_secs_f64(),
            t_tab.as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
        eprintln!("corpus {:.1} MiB", mib(c.memory_bytes()));
        for (name, ip) in [("fwd", &f), ("rev", &r)] {
            let cells = ip.table.n_cells();
            eprintln!(
                "{name}: rows {} cells {} structure {:.1} MiB (+ prob/count f64: {:.1} MiB)",
                ip.table.n_rows(),
                cells,
                mib(ip.table.memory_bytes()),
                mib(cells * 16)
            );
        }
    } else {
        let ip = initial_pass(&c, TableOptions { reverse, use_null });
        dump(&c, &ip).unwrap();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hexfloat_matches_glibc() {
        use super::hexfloat;
        assert_eq!(hexfloat(1.0), "0x1p+0");
        assert_eq!(hexfloat(0.0), "0x0p+0");
        assert_eq!(hexfloat(-2.5), "-0x1.4p+1");
        assert_eq!(hexfloat(0.1), "0x1.999999999999ap-4");
    }
}
