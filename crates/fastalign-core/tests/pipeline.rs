//! End-to-end checks of the pipeline: training, model save/load, forced alignment of the
//! training corpus, symmetrisation and `-p` export, on a small synthetic corpus.

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::io::{Direction, Model};
use fastalign_core::pipeline::{train, Directions, TrainConfig};
use fastalign_core::symmetrize::{symmetrize_text, Method};

/// Deterministic pseudo-random corpus: source words w0..w39, target words v0..v39 where the
/// target of w_k is mostly v_k, some noise, some empty sides.
fn corpus_text(lines: usize) -> String {
    let mut x = 12345u64;
    let mut next = move |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    let mut s = String::new();
    for _ in 0..lines {
        let n = next(9) as usize;
        let src: Vec<u64> = (0..n).map(|_| next(40)).collect();
        let mut tgt: Vec<String> = src.iter().map(|k| format!("v{k}")).collect();
        if next(3) == 0 && !tgt.is_empty() {
            let at = next(tgt.len() as u64) as usize;
            tgt.insert(at, format!("v{}", next(40)));
        }
        if next(4) == 0 {
            tgt.reverse();
        }
        let src: Vec<String> = src.iter().map(|k| format!("w{k}")).collect();
        s.push_str(&format!("{} ||| {}\n", src.join(" "), tgt.join(" ")));
    }
    s
}

fn load(text: &str) -> Corpus {
    Corpus::from_joined_bytes(text.as_bytes(), &LoadOptions::default())
}

#[test]
fn model_roundtrip_reproduces_training_alignments() {
    let corpus = load(&corpus_text(400));
    for (diag, vb, null) in [
        (true, true, true),
        (false, false, true),
        (true, false, false),
    ] {
        let mut cfg = TrainConfig::default();
        cfg.train.favor_diagonal = diag;
        cfg.train.optimize_tension = diag;
        cfg.train.variational_bayes = vb;
        cfg.train.use_null = null;
        cfg.train.buffer_size = 7;
        let trained = train(&corpus, &cfg, &|_| {}).unwrap();
        let mut bytes = Vec::new();
        trained.model.write(&mut bytes).unwrap();
        let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        for d in [Direction::Forward, Direction::Inverse] {
            let text = model.align_corpus(&corpus, d, false).unwrap();
            let want = trained.output(d).unwrap().alignments.clone();
            assert_eq!(
                String::from_utf8_lossy(&text),
                String::from_utf8_lossy(&want),
                "{d:?} diag={diag} vb={vb} null={null}"
            );
        }
        // Symmetrised in memory == atools on the two files.
        let f = &trained.fwd.as_ref().unwrap().alignments;
        let i = &trained.inv.as_ref().unwrap().alignments;
        for m in [
            Method::Intersect,
            Method::Union,
            Method::GrowDiag,
            Method::GrowDiagFinalAnd,
        ] {
            let mut want = Vec::new();
            symmetrize_text(f, Some(i), m, &mut want).unwrap();
            assert_eq!(trained.symmetrized(m).unwrap(), want);
        }
    }
}

#[test]
fn table_export_from_model_equals_trainer_table() {
    let corpus = load(&corpus_text(200));
    let trained = train(&corpus, &TrainConfig::default(), &|_| {}).unwrap();
    // Reload: the exported table must not depend on the file round trip.
    let mut bytes = Vec::new();
    trained.model.write(&mut bytes).unwrap();
    let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        trained.model.write_table(d, -4.0, &mut a).unwrap();
        model.write_table(d, -4.0, &mut b).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }
}

#[test]
fn unknown_words_and_empty_sides_in_new_text() {
    let corpus = load(&corpus_text(200));
    let trained = train(&corpus, &TrainConfig::default(), &|_| {}).unwrap();
    let new = load("w1 w2 unseen ||| v1 v2 never\n ||| v1\nw3 ||| \nw1 ||| v1\n");
    let text = trained
        .model
        .align_corpus(&new, Direction::Forward, true)
        .unwrap();
    let lines: Vec<&str> = std::str::from_utf8(&text).unwrap().lines().collect();
    assert_eq!(lines.len(), 4);
    let o = &trained.model.fwd.as_ref().unwrap().opts; // TEMP-DEBUG
    assert!(lines[0].contains(" ||| -"), "{text:?} opts={o:?}");
    assert_eq!(lines[1], "");
    assert_eq!(lines[2], "");
    assert!(lines[3].starts_with("0-0 ||| "), "{}", lines[3]);
}

#[test]
fn directions_and_thread_counts_do_not_change_results() {
    let corpus = load(&corpus_text(300));
    let both = train(&corpus, &TrainConfig::default(), &|_| {}).unwrap();
    for (dirs, d) in [
        (Directions::Forward, Direction::Forward),
        (Directions::Inverse, Direction::Inverse),
    ] {
        let cfg = TrainConfig {
            directions: dirs,
            ..TrainConfig::default()
        };
        let one =
            fastalign_core::pipeline::with_threads(3, || train(&corpus, &cfg, &|_| {})).unwrap();
        assert_eq!(
            one.output(d).unwrap().alignments,
            both.output(d).unwrap().alignments
        );
        assert_eq!(one.output(d).unwrap().log, both.output(d).unwrap().log);
    }
    assert!(both.symmetrized(Method::Union).is_ok());
    let conc = train(
        &corpus,
        &TrainConfig {
            concurrent_directions: true,
            ..TrainConfig::default()
        },
        &|_| {},
    )
    .unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        assert_eq!(
            conc.output(d).unwrap().alignments,
            both.output(d).unwrap().alignments
        );
        assert_eq!(conc.output(d).unwrap().log, both.output(d).unwrap().log);
    }
    let lean = train(
        &corpus,
        &TrainConfig {
            keep_models: false,
            ..TrainConfig::default()
        },
        &|_| {},
    )
    .unwrap();
    assert!(lean.model.fwd.is_none() && lean.model.inv.is_none());
    assert!(both.model.fwd.is_some() && both.model.inv.is_some());
    for d in [Direction::Forward, Direction::Inverse] {
        assert_eq!(
            lean.output(d).unwrap().alignments,
            both.output(d).unwrap().alignments
        );
    }
    let one = train(
        &corpus,
        &TrainConfig {
            directions: Directions::Forward,
            ..TrainConfig::default()
        },
        &|_| {},
    )
    .unwrap();
    assert!(one.symmetrized(Method::Union).is_err());
}

#[test]
fn progress_events() {
    let corpus = load(&corpus_text(100));
    let seen = std::sync::Mutex::new(Vec::new());
    let cfg = TrainConfig {
        directions: Directions::Forward,
        ..TrainConfig::default()
    };
    train(&corpus, &cfg, &|p| seen.lock().unwrap().push(p.iteration)).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![1, 2, 3, 4, 5]);
}
