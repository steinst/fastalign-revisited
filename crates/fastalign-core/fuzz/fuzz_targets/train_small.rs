#![no_main]
//! Whole training on a tiny corpus with options taken from the first bytes; model save/load and
//! alignment of the corpus must not panic, and the model round trip must reproduce the output
//! (with `iterations` 0 to 3, including the never-estimated `-I 1` model, and every
//! `empty_pairs` mode).
use fastalign_core::cancel::Cancel;
use fastalign_core::corpus::{Corpus, EmptyPairs, LoadOptions};
use fastalign_core::io::{Direction, Model};
use fastalign_core::pipeline::{train, TrainConfig};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > 4096 {
        return;
    }
    let (opt, text) = data.split_at(2);
    let mut cfg = TrainConfig::default();
    cfg.train.favor_diagonal = opt[0] & 1 != 0;
    cfg.train.optimize_tension = opt[0] & 2 != 0;
    cfg.train.variational_bayes = opt[0] & 4 != 0;
    cfg.train.use_null = opt[0] & 8 != 0;
    cfg.train.iterations = (opt[0] >> 4) as i32 % 4;
    cfg.train.buffer_size = (opt[1] & 7) as u64;
    let empty_pairs = if opt[1] & 8 != 0 {
        EmptyPairs::Skip
    } else {
        EmptyPairs::Keep
    };
    let opts = LoadOptions {
        empty_pairs,
        ..LoadOptions::default()
    };
    let corpus = Corpus::from_joined_bytes(text, &opts).unwrap();
    let trained = train(&corpus, &cfg, &|_| {}, &Cancel::default()).unwrap();
    let warnings = trained.warnings(&corpus);
    let mut bytes = Vec::new();
    trained.model.write(&mut bytes).unwrap();
    let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        let got = model.align_corpus(&corpus, d, false, &Cancel::default()).unwrap();
        let o = trained.output(d).unwrap();
        if cfg.train.iterations == 0 {
            assert!(!o.decoded && o.alignments.is_empty());
        } else if corpus.empty_side_lines().is_empty() && !o.nonfinite.any() {
            // a line with an empty side gets an empty line from both (no scores here)
            let got = corpus.expand_lines(&got).unwrap();
            assert_eq!(got.as_ref(), &o.alignments[..], "{d:?}");
            assert!(warnings.is_empty() || corpus.is_empty());
        }
    }
});
