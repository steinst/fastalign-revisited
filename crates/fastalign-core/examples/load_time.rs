//! Corpus load time: `load_time JOINED [SRC TGT] [--repeat N]`, median wall time of N reads
//! (default 5) of the joined file and, if given, of the two line-aligned files. Threads from
//! `RAYON_NUM_THREADS`.
use std::path::PathBuf;
use std::time::Instant;

use fastalign_core::corpus::{Corpus, LoadOptions};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut repeat = 5;
    if let Some(i) = args.iter().position(|a| a == "--repeat") {
        repeat = args[i + 1].parse().expect("--repeat N");
        args.drain(i..i + 2);
    }
    let opts = LoadOptions::default();
    let time = |f: &dyn Fn() -> usize| {
        let t: Vec<f64> = (0..repeat)
            .map(|_| {
                let t0 = Instant::now();
                std::hint::black_box(f());
                t0.elapsed().as_secs_f64()
            })
            .collect();
        median(t)
    };
    let joined = PathBuf::from(&args[0]);
    let t = time(&|| Corpus::read_joined(&joined, &opts).unwrap().len());
    println!("joined\t{:.3}", t);
    if args.len() == 3 {
        let (s, g) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
        let t = time(&|| Corpus::read_pair(&s, &g, &opts).unwrap().len());
        println!("pair\t{:.3}", t);
    }
}
