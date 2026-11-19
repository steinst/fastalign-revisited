//! Train both directions concurrently on one corpus load, sharing one thread pool (Step 6
//! measurement tool; Step 7's pipeline replaces it).
//!
//!     cargo run --release --example train_both -- INPUT OUT_PREFIX [THREADS] [fast_align flags]
//!
//! INPUT is a `SRC ||| TGT` file. Flags (default `-d -o -v`): `-d -o -v -N -s -I N -b N`.
//! Writes `OUT_PREFIX.{fwd,rev}.{out,err}` (stderr as fast_align's, without `ARG=` lines) and
//! prints phase timings and the trainers' heap size to stdout.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::oracle::TrainOptions;
use fastalign_core::train::{train_both, ParallelOptions, Sinks};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: train_both INPUT OUT_PREFIX [THREADS] [flags]");
        std::process::exit(2);
    }
    let (input, prefix) = (&args[1], &args[2]);
    let mut rest = &args[3..];
    let mut threads = 0;
    if let Some(t) = rest.first().and_then(|t| t.parse().ok()) {
        threads = t;
        rest = &rest[1..];
    }
    let mut opts = TrainOptions::default();
    if rest.is_empty() {
        opts.favor_diagonal = true;
        opts.optimize_tension = true;
        opts.variational_bayes = true;
    }
    let mut k = 0;
    while k < rest.len() {
        match rest[k].as_str() {
            "-d" => opts.favor_diagonal = true,
            "-o" => opts.optimize_tension = true,
            "-v" => opts.variational_bayes = true,
            "-N" => opts.use_null = false,
            "-s" => opts.print_scores = true,
            "-I" => {
                k += 1;
                opts.iterations = rest[k].parse().unwrap();
            }
            "-b" => {
                k += 1;
                opts.buffer_size = rest[k].parse().unwrap();
            }
            f => panic!("unsupported flag {f}"),
        }
        k += 1;
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    let t0 = Instant::now();
    let corpus = Corpus::read_joined(Path::new(input), &LoadOptions::default()).unwrap();
    let t_load = t0.elapsed();
    let open =
        |s: &str| BufWriter::with_capacity(1 << 20, File::create(format!("{prefix}.{s}")).unwrap());
    let (mut fo, mut fe, mut ro, mut re) = (
        open("fwd.out"),
        open("fwd.err"),
        open("rev.out"),
        open("rev.err"),
    );
    let rev = TrainOptions {
        reverse: true,
        ..opts.clone()
    };
    let t1 = Instant::now();
    let (f, r) = train_both(
        &corpus,
        opts,
        rev,
        &ParallelOptions::default(),
        Sinks {
            log: &mut fe,
            out: &mut fo,
        },
        Sinks {
            log: &mut re,
            out: &mut ro,
        },
    )
    .unwrap();
    for w in [&mut fo, &mut fe, &mut ro, &mut re] {
        w.flush().unwrap();
    }
    let t_train = t1.elapsed();
    println!(
        "threads {} load {:.2}s train(both) {:.2}s total {:.2}s corpus {} MiB trainers {} + {} MiB",
        rayon::current_num_threads(),
        t_load.as_secs_f64(),
        t_train.as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        corpus.memory_bytes() >> 20,
        f.memory_bytes() >> 20,
        r.memory_bytes() >> 20
    );
}
