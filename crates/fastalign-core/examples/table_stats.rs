//! Statistics of the translation-table lookups of one training pass (Step 9a evidence for
//! `benchmarks/PERF-TODO.md`).
//!
//!     cargo run --release --example table_stats -- INPUT [-r] [-N]
//!
//! Prints, for the direction: cells, cell visits per E-step (`Σ n·m`, plus the NULL column),
//! visits that repeat an `(e, f)` pair already looked up in the same line, the distribution of
//! row lengths weighted by visits (the length of the array a lookup searches), and the M-step's
//! source-side scan cost (`Σ over (row, line) of n`) against its useful additions, and how many
//! of the distinct-pair lookups (one per distinct `(e, f)` of a line, Step 9b) the K busiest
//! rows take (for a dense per-row index of those rows).

use std::path::Path;

use fastalign_core::corpus::{Corpus, LoadOptions, NULL_ID};
use fastalign_core::ttable::{initial_pass, TableOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = &args[1];
    let reverse = args.iter().any(|a| a == "-r");
    let use_null = !args.iter().any(|a| a == "-N");
    let corpus = Corpus::read_joined(Path::new(input), &LoadOptions::default()).unwrap();
    let pass = initial_pass(&corpus, TableOptions { reverse, use_null });
    let table = &pass.table;
    let (src, trg) = corpus.sides(reverse);
    let n_lines = corpus.len();

    let mut visits = 0u64; // source cells (NULL counted separately)
    let mut null_visits = 0u64;
    let mut repeat = 0u64;
    // M-step: Σ over distinct (row, line) of n, and Σ m × occurrences.
    let mut scan = 0u64;
    let mut adds = 0u64;
    // Visit-weighted histogram of log2(row length).
    let mut hist = [0u64; 32];
    let mut seen: Vec<(u32, u32)> = Vec::new();
    let mut distinct_e: Vec<u32> = Vec::new();
    let mut row_lookups = vec![0u64; table.n_rows()];
    for k in 0..n_lines {
        let s = src.line(k);
        let t = trg.line(k);
        let (n, m) = (s.len() as u64, t.len() as u64);
        visits += n * m;
        if use_null {
            null_visits += m;
        }
        adds += n * m;
        distinct_e.clear();
        distinct_e.extend_from_slice(s);
        distinct_e.sort_unstable();
        distinct_e.dedup();
        scan += distinct_e.len() as u64 * n;
        seen.clear();
        for &f in t {
            for &e in s {
                seen.push((e, f));
            }
        }
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        repeat += (before - seen.len()) as u64;
        for &(e, _) in &seen {
            row_lookups[e as usize] += 1;
        }
        for &e in s {
            let len = table.row(e).len() as u64;
            hist[63 - len.max(1).leading_zeros() as usize] += m;
        }
    }
    let null_len = if use_null {
        table.row(NULL_ID).len()
    } else {
        0
    };
    println!(
        "direction {} lines {n_lines} rows {} cells {} null_row_len {null_len}",
        if reverse { "rev" } else { "fwd" },
        table.n_rows(),
        table.n_cells()
    );
    println!(
        "visits per E-step: source {visits} + NULL {null_visits}; repeated (e,f) within a line: {repeat} ({:.1} %)",
        100.0 * repeat as f64 / visits as f64
    );
    println!(
        "M-step: additions {adds}, source-side scan {scan} ({:.2} compares per addition)",
        scan as f64 / adds as f64
    );
    println!("visit-weighted row length (source cells): log2 bucket, share, cumulative");
    let mut cum = 0u64;
    for (b, &h) in hist.iter().enumerate() {
        if h == 0 {
            continue;
        }
        cum += h;
        println!(
            "  [{:>7}, {:>7}): {:5.1} %  {:5.1} %",
            1u64 << b,
            1u64 << (b + 1),
            100.0 * h as f64 / visits as f64,
            100.0 * cum as f64 / visits as f64
        );
    }
    let total: u64 = row_lookups.iter().sum();
    let mut by: Vec<usize> = (0..row_lookups.len()).collect();
    by.sort_by_key(|&e| std::cmp::Reverse(row_lookups[e]));
    println!("distinct-pair lookups {total}; share taken by the K busiest rows (dense index = K x vocab x 4 B):");
    let mut cum = 0u64;
    let mut k_next = 8;
    for (k, &e) in by.iter().enumerate() {
        cum += row_lookups[e];
        if k + 1 == k_next {
            println!(
                "  K {:>6}: {:5.1} %  shortest row {:>7}  dense index {:>5} MB",
                k + 1,
                100.0 * cum as f64 / total as f64,
                by[..=k]
                    .iter()
                    .map(|&e| table.row(e as u32).len())
                    .min()
                    .unwrap(),
                ((k + 1) * corpus.vocab.len() * 4) >> 20
            );
            k_next *= 2;
        }
    }
}
