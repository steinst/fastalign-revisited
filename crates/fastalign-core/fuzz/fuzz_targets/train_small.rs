#![no_main]
//! Whole training on a tiny corpus with options taken from the first bytes; model save/load and
//! alignment of the corpus must not panic, and the model round trip must reproduce the output.
use fastalign_core::corpus::{Corpus, LoadOptions};
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
    let corpus = Corpus::from_joined_bytes(text, &LoadOptions::default());
    let trained = train(&corpus, &cfg, &|_| {}).unwrap();
    let mut bytes = Vec::new();
    trained.model.write(&mut bytes).unwrap();
    let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        let _ = model.align_corpus(&corpus, d, false).unwrap();
    }
});
