//! Per-phase timing of a training run (Step 9a measurement tool).
//!
//!     cargo run --release --example phases -- INPUT OUT_PREFIX THREADS seq|both [fast_align flags]
//!
//! INPUT is a `SRC ||| TGT` file; flags as `train_both` (default `-d -o -v`). `seq` trains the
//! forward direction alone and then the reverse direction alone (clean per-phase wall times);
//! `both` trains them concurrently, as `fastalign train` does (per-direction phase times then
//! overlap). Then grow-diag-final-and, and the three Pharaoh files are written to
//! `OUT_PREFIX.{fwd,rev,gdfa}`. Prints one `phase<TAB>direction<TAB>seconds` line per phase.
//! Environment knobs (`ParallelOptions`): `FA_SPLIT_COST`, `FA_PRIOR_CACHE`, `FA_GROUP_LINES`,
//! `FA_DENSE` (dense lookup index entries), `FA_DENSE_MIN_LEN`, `FA_BLOCK_LINES`,
//! `FA_SAMPLE_MIN_LEN` (`u64::MAX` disables the sampled index).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::oracle::TrainOptions;
use fastalign_core::pipeline::symmetrize_pharaoh;
use fastalign_core::symmetrize::Method;
use fastalign_core::train::{train_both, ParallelOptions, PhaseTimes, Sinks, Trainer};

fn report(dir: &str, t: &PhaseTimes, total: f64, mem: usize) {
    let n = t.train_iterations.max(1) as f64;
    for (name, d) in [
        ("initial_pass", t.initial_pass),
        ("setup", t.setup),
        ("prior", t.prior),
        ("e_step", t.e_step),
        ("m_step", t.m_step),
        ("tension", t.tension),
        ("decode", t.decode),
    ] {
        println!("{name}\t{dir}\t{:.3}", d.as_secs_f64());
    }
    println!("e_step_per_iter\t{dir}\t{:.3}", t.e_step.as_secs_f64() / n);
    println!("m_step_per_iter\t{dir}\t{:.3}", t.m_step.as_secs_f64() / n);
    println!("train_total\t{dir}\t{total:.3}");
    println!("trainer_mib\t{dir}\t{}", mem >> 20);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: phases INPUT OUT_PREFIX THREADS seq|both [flags]");
        std::process::exit(2);
    }
    let (input, prefix) = (&args[1], &args[2]);
    let threads: usize = args[3].parse().expect("THREADS");
    let concurrent = match args[4].as_str() {
        "seq" => false,
        "both" => true,
        m => panic!("mode must be seq or both, not {m}"),
    };
    let rest = &args[5..];
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
    let rev = TrainOptions {
        reverse: true,
        ..opts.clone()
    };
    // Tuning knobs from the environment (none changes any output bit).
    let env = |k: &str| std::env::var(k).ok().map(|v| v.parse::<u64>().expect(k));
    let mut popts = ParallelOptions::default();
    if let Some(v) = env("FA_SPLIT_COST") {
        popts.split_cost = Some(v);
    }
    if let Some(v) = env("FA_PRIOR_CACHE") {
        popts.prior_cache_entries = v as usize;
    }
    if let Some(v) = env("FA_GROUP_LINES") {
        popts.group_lines = v as usize;
    }
    if let Some(v) = env("FA_DENSE") {
        popts.dense_index_entries = v as usize;
    }
    if let Some(v) = env("FA_DENSE_MIN_LEN") {
        popts.dense_min_row_len = v as usize;
    }
    if let Some(v) = env("FA_BLOCK_LINES") {
        popts.block_lines = v as usize;
    }
    if let Some(v) = env("FA_SAMPLE_MIN_LEN") {
        popts.sample_min_row_len = v as usize;
    }
    let t_start = Instant::now();
    let corpus = Corpus::read_joined(Path::new(input), &LoadOptions::default()).unwrap();
    println!("load\t-\t{:.3}", t_start.elapsed().as_secs_f64());
    let (mut fo, mut fe, mut ro, mut re) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let t_train = Instant::now();
    if concurrent {
        let (f, r) = train_both(
            &corpus,
            opts,
            rev,
            &popts,
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
        let total = t_train.elapsed().as_secs_f64();
        report("fwd", &f.phase_times(), total, f.memory_bytes());
        report("rev", &r.phase_times(), total, r.memory_bytes());
    } else {
        for (o, out, log, name) in [
            (opts, &mut fo, &mut fe, "fwd"),
            (rev, &mut ro, &mut re, "rev"),
        ] {
            let t0 = Instant::now();
            let mut t = Trainer::new(&corpus, o, popts.clone(), log).unwrap();
            t.train(log, out).unwrap();
            report(
                name,
                &t.phase_times(),
                t0.elapsed().as_secs_f64(),
                t.memory_bytes(),
            );
        }
    }
    println!("train_both\t-\t{:.3}", t_train.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let gdfa = symmetrize_pharaoh(&fo, &ro, Method::GrowDiagFinalAnd).unwrap();
    println!("symmetrize\t-\t{:.3}", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    for (ext, data) in [("fwd", &fo), ("rev", &ro), ("gdfa", &gdfa)] {
        let mut w =
            BufWriter::with_capacity(1 << 20, File::create(format!("{prefix}.{ext}")).unwrap());
        w.write_all(data).unwrap();
        w.flush().unwrap();
    }
    for (ext, data) in [("fwd.err", &fe), ("rev.err", &re)] {
        std::fs::write(format!("{prefix}.{ext}"), data).unwrap();
    }
    println!("write\t-\t{:.3}", t0.elapsed().as_secs_f64());
    println!("total\t-\t{:.3}", t_start.elapsed().as_secs_f64());
    println!("corpus_mib\t-\t{}", corpus.memory_bytes() >> 20);
}
