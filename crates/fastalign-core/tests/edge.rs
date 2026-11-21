//! Degenerate inputs must neither panic nor produce a model that cannot be saved, loaded and
//! used (Step 10). Byte parity with fast_align on these inputs is checked by
//! `benchmarks/edge_parity.py`, which needs the reference build.

use fastalign_core::cancel::Cancel;
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::io::{Direction, Model};
use fastalign_core::pipeline::{train, TrainConfig};

const INPUTS: &[&[u8]] = &[
    b"",
    b"\n",
    b"\n\n\n",
    b" ||| \n",
    b"a ||| x\n",
    b"a ||| \n",
    b" ||| x\n",
    b"<eps> ||| <eps>\n",
    b"a b c d ||| x\n",
    b"a b ||| x y\r\nc ||| z",
    b"a ||| b ||| c\nno separator\n\t\n",
    b"\xe9 \xff ||| \x00 \xfe\n",
];

fn configs() -> Vec<TrainConfig> {
    let mut v = Vec::new();
    for (diag, vb, null, iters, buf) in [
        (true, true, true, 5, 10_000),
        (false, false, true, 3, 1),
        (true, false, false, 2, 0),
        (true, true, true, 0, 10_000),
        (true, true, false, 1, 3),
    ] {
        let mut cfg = TrainConfig::default();
        cfg.train.favor_diagonal = diag;
        cfg.train.optimize_tension = diag;
        cfg.train.variational_bayes = vb;
        cfg.train.use_null = null;
        cfg.train.iterations = iters;
        cfg.train.buffer_size = buf;
        v.push(cfg);
    }
    v
}

#[test]
fn degenerate_corpora_train_save_load_align() {
    for &text in INPUTS {
        let corpus = Corpus::from_joined_bytes(text, &LoadOptions::default()).unwrap();
        for cfg in configs() {
            let trained = train(&corpus, &cfg, &|_| {}, &Cancel::default()).unwrap();
            let mut bytes = Vec::new();
            trained.model.write(&mut bytes).unwrap();
            let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
            for d in [Direction::Forward, Direction::Inverse] {
                let text = model
                    .align_corpus(&corpus, d, false, &Cancel::default())
                    .unwrap();
                assert_eq!(text.iter().filter(|&&b| b == b'\n').count(), corpus.len());
            }
        }
    }
}
