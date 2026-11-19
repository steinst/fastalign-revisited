//! Property test: the parallel trainer (`train.rs`) is bit-identical to the sequential oracle
//! (`oracle.rs`) after every iteration, at 1, 2, 3 and 8 threads, on random small corpora
//! (random lengths, repeated words, empty sides and lines, `<eps>` and `|||` as words, tabs,
//! tiny `-b`), with random training options and random tuning of the parallel trainer
//! (row splitting, group size, prior cache), none of which may change a bit.

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::oracle::{IterationStats, Oracle, TrainOptions};
use fastalign_core::train::{train_both, ParallelOptions, Sinks, Trainer};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

fn random_corpus(rng: &mut Rng, allow_empty: bool) -> Vec<u8> {
    let lines = 1 + rng.below(40);
    let vocab = 2 + rng.below(14);
    let mut text = Vec::new();
    for _ in 0..lines {
        if allow_empty && rng.chance(3) {
            text.push(b'\n');
            continue;
        }
        let side = |rng: &mut Rng, text: &mut Vec<u8>, target: bool| {
            let empty = allow_empty && rng.chance(8);
            let len = if empty { 0 } else { 1 + rng.below(9) };
            for _ in 0..len {
                let w = if rng.chance(3) {
                    "<eps>".to_string()
                } else if target && rng.chance(3) {
                    "|||".to_string()
                } else {
                    // Zipf-like: small ids are frequent.
                    let r = rng.below(vocab * vocab);
                    format!("w{}", (r as f64).sqrt() as u64)
                };
                text.extend_from_slice(w.as_bytes());
                text.push(b' ');
            }
        };
        side(rng, &mut text, false);
        text.extend_from_slice(if rng.chance(10) { b"\t" } else { b"||| " });
        side(rng, &mut text, true);
        text.push(b'\n');
    }
    text
}

fn random_options(rng: &mut Rng) -> TrainOptions {
    TrainOptions {
        reverse: rng.chance(50),
        iterations: rng.pick(&[0, 1, 2, 3, 5, 6]),
        favor_diagonal: rng.chance(70),
        optimize_tension: rng.chance(70),
        variational_bayes: rng.chance(50),
        alpha: rng.pick(&[0.01, 0.2]),
        use_null: rng.chance(75),
        p0: rng.pick(&[0.08, 0.3]),
        tension: rng.pick(&[4.0, 0.5, 2.0, 13.0]),
        buffer_size: rng.pick(&[0, 1, 2, 3, 7, 10000]),
        print_scores: rng.chance(50),
    }
}

fn random_parallel(rng: &mut Rng) -> ParallelOptions {
    ParallelOptions {
        prior_cache_entries: rng.pick(&[0, 7, 40, 1 << 20]),
        group_lines: rng.pick(&[1, 2, 5, 4096]),
        split_cost: rng.pick(&[Some(1), Some(5), Some(50), None]),
        dense_index_entries: rng.pick(&[0, 30, 200, 1 << 20]),
        dense_min_row_len: rng.pick(&[0, 3, 1024]),
        block_lines: rng.pick(&[0, 1, 3, 16384]),
        sample_min_row_len: rng.pick(&[0, 1, 5, usize::MAX]),
    }
}

/// Everything observable after each iteration, as bits.
#[derive(Debug, PartialEq)]
struct Trace {
    log: Vec<u8>,
    out: Vec<u8>,
    iters: Vec<(Vec<u64>, [u64; 5], u64)>,
    table: Vec<u8>,
}

fn stat_bits(s: IterationStats) -> [u64; 5] {
    [
        s.likelihood.to_bits(),
        s.c0.to_bits(),
        s.emp_feat.to_bits(),
        s.tension.to_bits(),
        s.tension_after.to_bits(),
    ]
}

fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn run_oracle(c: &Corpus, opts: &TrainOptions) -> Trace {
    let mut log = Vec::new();
    let mut out = Vec::new();
    let mut o = Oracle::new(c, opts.clone(), &mut log).unwrap();
    let n = opts.iterations.max(0) as usize;
    let mut iters = Vec::new();
    for it in 0..n {
        let s = o.iteration(it, it + 1 == n, &mut log, &mut out).unwrap();
        iters.push((bits(o.probs()), stat_bits(s), o.tension().to_bits()));
    }
    let mut table = Vec::new();
    o.write_table(&c.vocab, -4.0, &mut table).unwrap();
    Trace {
        log,
        out,
        iters,
        table,
    }
}

fn run_trainer(c: &Corpus, opts: &TrainOptions, popts: &ParallelOptions) -> Trace {
    let mut log = Vec::new();
    let mut out = Vec::new();
    let mut t = Trainer::new(c, opts.clone(), popts.clone(), &mut log).unwrap();
    let n = opts.iterations.max(0) as usize;
    let mut iters = Vec::new();
    for it in 0..n {
        let s = t.iteration(it, it + 1 == n, &mut log, &mut out).unwrap();
        iters.push((bits(t.probs()), stat_bits(s), t.tension().to_bits()));
    }
    let mut table = Vec::new();
    t.write_table(&c.vocab, -4.0, &mut table).unwrap();
    Trace {
        log,
        out,
        iters,
        table,
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn first_difference(a: &Trace, b: &Trace) -> String {
    if a.log != b.log {
        return format!(
            "log differs:\n--- oracle\n{}\n--- trainer\n{}",
            String::from_utf8_lossy(&a.log),
            String::from_utf8_lossy(&b.log)
        );
    }
    for (k, (x, y)) in a.iters.iter().zip(&b.iters).enumerate() {
        if x != y {
            return format!("iteration {} differs", k + 1);
        }
    }
    if a.out != b.out {
        return "alignments differ".into();
    }
    "table differs".into()
}

fn check_cases(seeds: std::ops::Range<u64>, allow_empty: bool) {
    let pools: Vec<(usize, rayon::ThreadPool)> = [1, 2, 3, 8].map(|t| (t, pool(t))).into();
    for seed in seeds {
        let mut rng = Rng(seed);
        let text = random_corpus(&mut rng, allow_empty);
        let opts = random_options(&mut rng);
        let popts = random_parallel(&mut rng);
        let c = Corpus::from_joined_bytes(&text, &LoadOptions::default());
        let want = run_oracle(&c, &opts);
        for (threads, p) in &pools {
            let got = p.install(|| run_trainer(&c, &opts, &popts));
            assert!(
                want == got,
                "seed {seed}, {threads} threads, {opts:?}, {popts:?}: {}\ncorpus:\n{}",
                first_difference(&want, &got),
                String::from_utf8_lossy(&text)
            );
        }
    }
}

#[test]
fn trainer_matches_oracle_on_random_corpora() {
    check_cases(0..300, false);
}

#[test]
fn trainer_matches_oracle_with_empty_sides() {
    check_cases(1000..1300, true);
}

#[test]
fn larger_corpus_many_buffers() {
    // Several hundred lines, so that rows get long and many groups and splits occur.
    let mut rng = Rng(42);
    let mut text = Vec::new();
    for _ in 0..12 {
        text.extend(random_corpus(&mut rng, false));
    }
    let c = Corpus::from_joined_bytes(&text, &LoadOptions::default());
    for (k, b) in [1u64, 13, 100, 10000].into_iter().enumerate() {
        let opts = TrainOptions {
            favor_diagonal: true,
            optimize_tension: true,
            variational_bayes: k % 2 == 0,
            print_scores: true,
            buffer_size: b,
            reverse: k >= 2,
            ..Default::default()
        };
        let want = run_oracle(&c, &opts);
        for threads in [1, 2, 3, 8] {
            for popts in [
                ParallelOptions::default(),
                ParallelOptions {
                    prior_cache_entries: 100,
                    group_lines: 3,
                    split_cost: Some(20),
                    dense_index_entries: 1 << 16,
                    dense_min_row_len: 0,
                    block_lines: 7,
                    sample_min_row_len: 1,
                },
            ] {
                let got = pool(threads).install(|| run_trainer(&c, &opts, &popts));
                assert!(
                    want == got,
                    "b {b}, {threads} threads: {}",
                    first_difference(&want, &got)
                );
            }
        }
    }
}

#[test]
fn both_directions_concurrently() {
    let mut rng = Rng(7);
    let mut text = Vec::new();
    for _ in 0..5 {
        text.extend(random_corpus(&mut rng, false));
    }
    let c = Corpus::from_joined_bytes(&text, &LoadOptions::default());
    let fwd = TrainOptions {
        favor_diagonal: true,
        optimize_tension: true,
        variational_bayes: true,
        buffer_size: 7,
        ..Default::default()
    };
    let rev = TrainOptions {
        reverse: true,
        ..fwd.clone()
    };
    let want_f = run_oracle(&c, &fwd);
    let want_r = run_oracle(&c, &rev);
    for threads in [1, 2, 8] {
        let (mut lf, mut of, mut lr, mut or) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let popts = ParallelOptions {
            split_cost: Some(10),
            ..Default::default()
        };
        let (tf, tr) = pool(threads)
            .install(|| {
                train_both(
                    &c,
                    fwd.clone(),
                    rev.clone(),
                    &popts,
                    Sinks {
                        log: &mut lf,
                        out: &mut of,
                    },
                    Sinks {
                        log: &mut lr,
                        out: &mut or,
                    },
                )
            })
            .unwrap();
        assert_eq!(lf, want_f.log);
        assert_eq!(of, want_f.out);
        assert_eq!(lr, want_r.log);
        assert_eq!(or, want_r.out);
        assert_eq!(bits(tf.probs()), want_f.iters.last().unwrap().0);
        assert_eq!(bits(tr.probs()), want_r.iters.last().unwrap().0);
    }
}
