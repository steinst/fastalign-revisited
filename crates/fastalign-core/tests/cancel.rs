//! Per-operation cancellation: a requested token stops its own operation in every phase that
//! checks it (corpus reading, table setup, training, subword projection, forced alignment,
//! text-table reading, table export), and never another operation's.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;

use fastalign_core::cancel::{is_cancelled, Cancel};
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::io::{Direction, SubwordInfo};
use fastalign_core::oracle::{force_align, ForceOptions, TextTable, TrainOptions};
use fastalign_core::pipeline::{train, with_threads, Directions, Progress, TrainConfig};
use fastalign_core::subword::{project_text, Boundary};
use fastalign_core::train::{ParallelOptions, Trainer};
use fastalign_core::ttable::{initial_pass_with, TableOptions};

fn text(n: usize, seed: usize) -> String {
    (0..n)
        .map(|k| {
            let k = k + seed;
            format!("a{} b{} c ||| x{} y y{}\n", k % 7, k % 11, k % 5, k % 13)
        })
        .collect()
}

fn corpus(n: usize, seed: usize) -> Corpus {
    Corpus::from_joined_bytes(text(n, seed).as_bytes(), &LoadOptions::default()).unwrap()
}

fn requested() -> Cancel {
    let c = Cancel::new();
    c.request();
    c
}

fn assert_cancelled<T>(r: Result<T, Box<dyn Any + Send>>) {
    match r {
        Ok(_) => panic!("the operation was not cancelled"),
        Err(p) => assert!(is_cancelled(&*p), "unwound with something else"),
    }
}

fn config(directions: Directions) -> TrainConfig {
    TrainConfig {
        directions,
        ..TrainConfig::default()
    }
}

#[test]
fn every_phase_stops_on_a_requested_token() {
    let t = text(300, 0);
    let c = corpus(300, 0);
    let cancel = requested();

    // Corpus reading, joined and paired.
    let opts = LoadOptions {
        cancel: cancel.clone(),
        ..LoadOptions::default()
    };
    assert_cancelled(catch_unwind(|| {
        Corpus::from_joined_bytes(t.as_bytes(), &opts).unwrap()
    }));
    assert_cancelled(catch_unwind(|| {
        Corpus::from_pair_bytes(b"a b\n", b"x y\n", &opts)
    }));

    // Table setup and the trainer.
    assert_cancelled(catch_unwind(|| {
        initial_pass_with(&c, TableOptions::default(), &cancel)
    }));
    assert_cancelled(catch_unwind(|| {
        Trainer::new(
            &c,
            TrainOptions::default(),
            ParallelOptions::default(),
            &cancel,
            &mut Vec::new(),
        )
        .map(|_| ())
    }));
    assert_cancelled(catch_unwind(|| {
        train(&c, &config(Directions::Both), &|_| {}, &cancel)
    }));

    // Projection, forced alignment, table export, text tables.
    let trained = train(&c, &config(Directions::Both), &|_| {}, &Cancel::default()).unwrap();
    let fwd = &trained.fwd.as_ref().unwrap().alignments;
    let sw = Boundary::sentencepiece();
    assert_cancelled(catch_unwind(|| project_text(&c, &sw, fwd, false, &cancel)));
    for d in [Direction::Forward, Direction::Inverse] {
        assert_cancelled(catch_unwind(|| {
            trained.model.align_corpus(&c, d, false, &cancel)
        }));
        assert_cancelled(catch_unwind(|| {
            trained.model.align_words(&c, d, true, &cancel)
        }));
        assert_cancelled(catch_unwind(|| {
            trained.model.write_table(d, -4.0, &cancel, &mut Vec::new())
        }));
    }
    let mut table = Vec::new();
    trained
        .model
        .write_table(Direction::Forward, -4.0, &Cancel::default(), &mut table)
        .unwrap();
    assert_cancelled(catch_unwind(|| {
        TextTable::parse_with(&table, &c.vocab, &cancel)
    }));
    let tt = TextTable::parse(&table, &c.vocab);
    let fopts = ForceOptions {
        reverse: false,
        favor_diagonal: true,
        use_null: true,
        p0: 0.08,
        tension: 4.0,
        mean_srclen_multiplier: 1.0,
    };
    assert_cancelled(catch_unwind(|| {
        force_align(&c, &tt, &fopts, &cancel, &mut Vec::new(), &mut Vec::new())
    }));

    // The same calls with a fresh token work: nothing is left behind.
    let fresh = Cancel::new();
    let again = train(&c, &config(Directions::Both), &|_| {}, &fresh).unwrap();
    assert_eq!(again.fwd.unwrap().alignments, *fwd);
}

#[test]
fn a_token_requested_mid_run_stops_before_the_projection() {
    // The progress callback of the final iteration requests the token: the subword projection
    // that follows must not run to completion.
    let c = corpus(2000, 3);
    let mut cfg = config(Directions::Forward);
    cfg.train.iterations = 2;
    cfg.subword = Some(SubwordInfo {
        boundary: Boundary::sentencepiece(),
        tokenizers: Vec::new(),
        settings: Vec::new(),
    });
    let cancel = Cancel::new();
    let on_progress = |p: &Progress| {
        if p.iteration == p.iterations {
            cancel.request();
        }
    };
    assert_cancelled(catch_unwind(AssertUnwindSafe(|| {
        train(&c, &cfg, &on_progress, &cancel)
    })));
}

/// Two jobs at once (on the shared global pool, and each on its own pool): job A requests its
/// own token from its first progress event; job B, released only after that, keeps training.
/// Only A stops; B's output and log are those of a solo run, byte for byte.
#[test]
fn cancelling_one_job_leaves_a_concurrent_job_alone() {
    let (corpus_a, corpus_b) = (corpus(400, 0), corpus(600, 5));
    let cfg = config(Directions::Both);
    let solo = train(&corpus_b, &cfg, &|_| {}, &Cancel::default()).unwrap();
    for threads in [0, 1, 2] {
        let barrier = Barrier::new(2);
        let a_requested = AtomicBool::new(false);
        let (ta, tb) = (Cancel::new(), Cancel::new());
        let first = |p: &Progress| p.direction == Direction::Forward && p.iteration == 1;
        let on_a = |p: &Progress| {
            if first(p) {
                barrier.wait();
                ta.request();
                a_requested.store(true, Ordering::SeqCst);
            }
        };
        let on_b = |p: &Progress| {
            if first(p) {
                barrier.wait();
                while !a_requested.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
            }
        };
        let (ra, rb) = std::thread::scope(|s| {
            let a = s.spawn(|| {
                catch_unwind(AssertUnwindSafe(|| {
                    with_threads(threads, || train(&corpus_a, &cfg, &on_a, &ta)).unwrap()
                }))
            });
            let b =
                s.spawn(|| with_threads(threads, || train(&corpus_b, &cfg, &on_b, &tb)).unwrap());
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_cancelled(ra);
        assert!(!tb.is_requested());
        let rb = rb.unwrap();
        for d in [Direction::Forward, Direction::Inverse] {
            let (got, want) = (rb.output(d).unwrap(), solo.output(d).unwrap());
            assert_eq!(got.alignments, want.alignments, "{threads} threads");
            assert_eq!(got.log, want.log, "{threads} threads");
        }
    }
}
