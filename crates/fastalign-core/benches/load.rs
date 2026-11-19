//! Criterion benchmarks for corpus loading and the initial pass (table construction).
//!
//! Uses `reference/golden/_input/<corpus>.txt` (gitignored; made by benchmarks/golden.sh).
//! Corpus via `FASTALIGN_BENCH_CORPUS` (default `enfr-medium`); skipped if the file is missing.
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::ttable::{initial_pass, TableOptions};

fn input() -> Option<PathBuf> {
    let name = std::env::var("FASTALIGN_BENCH_CORPUS").unwrap_or_else(|_| "enfr-medium".into());
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/golden/_input")
        .join(format!("{name}.txt"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping benchmarks: {} not found", p.display());
        None
    }
}

fn bench(c: &mut Criterion) {
    let Some(path) = input() else { return };
    let opts = LoadOptions::default();
    let mut g = c.benchmark_group("load");
    g.sample_size(10);
    g.bench_function("read_joined", |b| {
        b.iter(|| Corpus::read_joined(&path, &opts).unwrap())
    });
    let corpus = Corpus::read_joined(&path, &opts).unwrap();
    g.bench_function("initial_pass_fwd", |b| {
        b.iter(|| initial_pass(&corpus, TableOptions::default()))
    });
    g.bench_function("initial_pass_both", |b| {
        b.iter(|| {
            rayon::join(
                || initial_pass(&corpus, TableOptions::default()),
                || {
                    initial_pass(
                        &corpus,
                        TableOptions {
                            reverse: true,
                            ..Default::default()
                        },
                    )
                },
            )
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
