//! End-to-end checks of the pipeline: training, model save/load, forced alignment of the
//! training corpus, symmetrisation and `-p` export, on a small synthetic corpus.

use fastalign_core::cancel::Cancel;
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
    Corpus::from_joined_bytes(text.as_bytes(), &LoadOptions::default()).unwrap()
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
        let trained = train(&corpus, &cfg, &|_| {}, &Cancel::default()).unwrap();
        let mut bytes = Vec::new();
        trained.model.write(&mut bytes).unwrap();
        let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        for d in [Direction::Forward, Direction::Inverse] {
            let text = model
                .align_corpus(&corpus, d, false, &Cancel::default())
                .unwrap();
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
    let trained = train(
        &corpus,
        &TrainConfig::default(),
        &|_| {},
        &Cancel::default(),
    )
    .unwrap();
    // Reload: the exported table must not depend on the file round trip.
    let mut bytes = Vec::new();
    trained.model.write(&mut bytes).unwrap();
    let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        trained
            .model
            .write_table(d, -4.0, &Cancel::default(), &mut a)
            .unwrap();
        model
            .write_table(d, -4.0, &Cancel::default(), &mut b)
            .unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }
}

#[test]
fn unknown_words_and_empty_sides_in_new_text() {
    let corpus = load(&corpus_text(200));
    let trained = train(
        &corpus,
        &TrainConfig::default(),
        &|_| {},
        &Cancel::default(),
    )
    .unwrap();
    let new = load("w1 w2 unseen ||| v1 v2 never\n ||| v1\nw3 ||| \nw1 ||| v1\n");
    let text = trained
        .model
        .align_corpus(&new, Direction::Forward, true, &Cancel::default())
        .unwrap();
    let lines: Vec<&str> = std::str::from_utf8(&text).unwrap().lines().collect();
    assert_eq!(lines.len(), 4);
    // The synthetic corpus has lines with an empty source, so the length ratio is NaN (as in
    // fast_align) and so is the score; the sign of that NaN depends on the CPU (negative on x86,
    // positive on arm64). Any score is fine here; the point is that the line has one.
    let score = lines[0].rsplit_once(" ||| ").map(|(_, s)| s.to_string());
    let score = score.unwrap_or_else(|| panic!("no score in {}", lines[0]));
    assert!(
        score.parse::<f64>().is_ok() || score.ends_with("nan"),
        "{score}"
    );
    assert_eq!(lines[1], "");
    assert_eq!(lines[2], "");
    assert!(lines[3].starts_with("0-0 ||| "), "{}", lines[3]);
}

#[test]
fn directions_and_thread_counts_do_not_change_results() {
    let corpus = load(&corpus_text(300));
    let both = train(
        &corpus,
        &TrainConfig::default(),
        &|_| {},
        &Cancel::default(),
    )
    .unwrap();
    for (dirs, d) in [
        (Directions::Forward, Direction::Forward),
        (Directions::Inverse, Direction::Inverse),
    ] {
        let cfg = TrainConfig {
            directions: dirs,
            ..TrainConfig::default()
        };
        let one = fastalign_core::pipeline::with_threads(3, || {
            train(&corpus, &cfg, &|_| {}, &Cancel::default())
        })
        .unwrap()
        .unwrap();
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
        &Cancel::default(),
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
        &Cancel::default(),
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
        &Cancel::default(),
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
    train(
        &corpus,
        &cfg,
        &|p| seen.lock().unwrap().push(p.iteration),
        &Cancel::default(),
    )
    .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![1, 2, 3, 4, 5]);
}

// ------------------------------------------------------------------------------------------
// Degenerate training semantics (Step 22: review #5, #10, tension 0)

mod degenerate {
    use super::*;
    use fastalign_core::corpus::EmptyPairs;
    use fastalign_core::io::SubwordInfo;
    use fastalign_core::math::check_tension;
    use fastalign_core::subword::Boundary;

    const MARK: &str = "\u{2581}";

    fn read(text: &str, max_len: Option<usize>, empty_pairs: EmptyPairs) -> Corpus {
        let o = LoadOptions {
            max_len,
            empty_pairs,
            ..LoadOptions::default()
        };
        Corpus::from_joined_bytes(text.as_bytes(), &o).unwrap()
    }

    fn cfg(iterations: i32, subword: bool) -> TrainConfig {
        let mut c = TrainConfig::default();
        c.train.iterations = iterations;
        c.train.print_scores = true;
        if subword {
            c.subword = Some(SubwordInfo {
                boundary: Boundary::from_parts("start", MARK.as_bytes()).unwrap(),
                tokenizers: Vec::new(),
                settings: Vec::new(),
            });
            c.keep_subword_output = true;
        }
        c
    }

    /// Word or piece corpus: in the piece form each word `wK` is `▁w K` (two pieces).
    fn text(lines: &[(&str, &str)], subword: bool) -> String {
        let side = |s: &str| -> String {
            s.split_whitespace()
                .map(|w| {
                    if subword {
                        format!("{MARK}{} {}", &w[..1], &w[1..])
                    } else {
                        w.to_owned()
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        lines
            .iter()
            .map(|(a, b)| format!("{} ||| {}\n", side(a), side(b)))
            .collect()
    }

    fn base_lines() -> Vec<(String, String)> {
        let t = corpus_text(60);
        t.lines()
            .map(|l| {
                let (a, b) = l.split_once(" ||| ").unwrap();
                (a.to_owned(), b.to_owned())
            })
            .filter(|(a, b)| !a.is_empty() && !b.is_empty())
            .collect()
    }

    fn insert_empty_lines(text: &[u8], at: &[usize], n: usize) -> Vec<u8> {
        let mut lines = text.split_inclusive(|&b| b == b'\n');
        let mut out = Vec::new();
        for k in 0..n {
            if at.contains(&k) {
                out.push(b'\n');
            } else {
                out.extend_from_slice(lines.next().unwrap());
            }
        }
        assert!(lines.next().is_none());
        out
    }

    /// Iterations 0/1/2 × no/partial/total exclusion × word/subword: no decoding iteration means
    /// no lines at all; otherwise one line per input line, the kept lines as trained alone.
    #[test]
    fn iterations_by_exclusion_by_subword() {
        let lines = base_lines();
        let long: Vec<usize> = (0..lines.len())
            .filter(|&k| {
                let (a, b) = &lines[k];
                a.split_whitespace()
                    .count()
                    .max(b.split_whitespace().count())
                    > 5
            })
            .collect();
        assert!(!long.is_empty() && long.len() < lines.len());
        let refs: Vec<(&str, &str)> = lines
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        for subword in [false, true] {
            let full = text(&refs, subword);
            // pieces double the length: max_len counts pieces
            let cap = if subword { 10 } else { 5 };
            let kept: Vec<(&str, &str)> = (0..refs.len())
                .filter(|k| !long.contains(k))
                .map(|k| refs[k])
                .collect();
            for iters in [0, 1, 2] {
                let c = cfg(iters, subword);
                for (name, max_len, dropped) in [
                    ("none", None, vec![]),
                    ("partial", Some(cap), long.clone()),
                    ("total", Some(0), (0..refs.len()).collect()),
                ] {
                    let what = format!("subword {subword} -I {iters} exclusion {name}");
                    let corpus = read(&full, max_len, EmptyPairs::Keep);
                    let t = train(&corpus, &c, &|_| {}, &Cancel::default()).unwrap();
                    for d in [Direction::Forward, Direction::Inverse] {
                        let o = t.output(d).unwrap();
                        assert_eq!(o.decoded, iters > 0, "{what}");
                        if iters == 0 {
                            assert!(o.alignments.is_empty(), "{what}");
                            assert_eq!(
                                o.subword_alignments.as_deref(),
                                subword.then_some(&b""[..])
                            );
                            continue;
                        }
                        let alone: Vec<u8> = if dropped.len() == refs.len() {
                            Vec::new()
                        } else {
                            let sub = if dropped.is_empty() {
                                refs.clone()
                            } else {
                                kept.clone()
                            };
                            let k = read(&text(&sub, subword), None, EmptyPairs::Keep);
                            let ta = train(&k, &c, &|_| {}, &Cancel::default()).unwrap();
                            ta.output(d).unwrap().alignments.clone()
                        };
                        let want = insert_empty_lines(&alone, &dropped, refs.len());
                        assert_eq!(o.alignments, want, "{what} {d:?}");
                        if subword {
                            let raw = o.subword_alignments.as_ref().unwrap();
                            assert_eq!(raw.iter().filter(|&&b| b == b'\n').count(), refs.len());
                        }
                    }
                    let sym = t.symmetrized(Method::GrowDiagFinalAnd).unwrap();
                    let n = sym.iter().filter(|&&b| b == b'\n').count();
                    assert_eq!(n, if iters == 0 { 0 } else { refs.len() }, "{what}");
                    let warn = t.warnings(&corpus);
                    assert_eq!(name == "total", !warn.is_empty(), "{what}: {warn:?}");
                }
            }
        }
    }

    /// `-I 1`: the model has no estimated probabilities (its `-p` export is all `-inf`, as
    /// fast_align's), and inference reproduces the training alignments (review #10).
    #[test]
    fn one_iteration_checkpoint_reproduces_training() {
        let lines = base_lines();
        let refs: Vec<(&str, &str)> = lines
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let corpus = load(&format!("a b c ||| x y z\n{}", text(&refs, false)));
        for (diag, null) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut c = cfg(1, false);
            c.train.favor_diagonal = diag;
            c.train.optimize_tension = diag;
            c.train.use_null = null;
            let t = train(&corpus, &c, &|_| {}, &Cancel::default()).unwrap();
            let mut bytes = Vec::new();
            t.model.write(&mut bytes).unwrap();
            let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
            for d in [Direction::Forward, Direction::Inverse] {
                let want = &t.output(d).unwrap().alignments;
                assert!(
                    want.starts_with(b"0-0 1-1 2-2 ||| ") || !diag,
                    "{diag} {null}"
                );
                let got = model
                    .align_corpus(&corpus, d, true, &Cancel::default())
                    .unwrap();
                assert_eq!(&got, want, "diag {diag} null {null} {d:?}");
                let mut table = Vec::new();
                model
                    .write_table(d, -4.0, &Cancel::default(), &mut table)
                    .unwrap();
                assert!(!table.is_empty());
                assert!(table
                    .split(|&b| b == b'\n')
                    .filter(|l| !l.is_empty())
                    .all(|l| l.ends_with(b"\t-inf")));
            }
        }
    }

    fn twenty_good_plus(extra: &str) -> String {
        format!("{}{extra}", "a b c ||| x y z\n".repeat(20))
    }

    /// Review #5: 20 good pairs + one with an empty source side (or a blank line), both
    /// directions, through save and inference, for each `empty_pairs` value.
    #[test]
    fn twenty_good_plus_one_empty_pair() {
        let good = load(&twenty_good_plus(""));
        let g = train(&good, &cfg(5, false), &|_| {}, &Cancel::default()).unwrap();
        for (extra, side) in [
            (" ||| x\n", "source side empty"),
            ("\n", "both sides empty"),
        ] {
            let text = twenty_good_plus(extra);
            // keep: fast_align's result (the parity suites cover the bytes); the damage is reported
            let corpus = read(&text, None, EmptyPairs::Keep);
            let t = train(&corpus, &cfg(5, false), &|_| {}, &Cancel::default()).unwrap();
            let warn = t.warnings(&corpus);
            assert!(
                !warn.is_empty()
                    && warn
                        .iter()
                        .all(|w| w.contains(&format!("line 21 ({side})"))),
                "{warn:?}"
            );
            let fwd = &t.fwd.as_ref().unwrap();
            assert!(fwd.nonfinite.any());
            if extra.starts_with(' ') {
                assert!(fwd.nonfinite.tension.unwrap().is_nan());
                assert!(fwd
                    .alignments
                    .split(|&b| b == b'\n')
                    .all(|l| l.starts_with(b" ||| ") || l.is_empty()));
                assert!(!t.inv.as_ref().unwrap().nonfinite.any());
            } else {
                assert!(fwd.nonfinite.mean_srclen_multiplier.unwrap().is_nan());
                assert!(fwd.alignments.starts_with(b"0-0 1-1 2-2 ||| -nan\n"));
            }
            // skip: the 20 good pairs' results and an empty line
            let corpus = read(&text, None, EmptyPairs::Skip);
            let t = train(&corpus, &cfg(5, false), &|_| {}, &Cancel::default()).unwrap();
            assert!(t.warnings(&corpus).is_empty());
            let mut bytes = Vec::new();
            t.model.write(&mut bytes).unwrap();
            let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
            let all = read(&text, None, EmptyPairs::Keep);
            for d in [Direction::Forward, Direction::Inverse] {
                let mut want = g.output(d).unwrap().alignments.clone();
                want.push(b'\n');
                assert_eq!(t.output(d).unwrap().alignments, want, "{d:?}");
                // inference on the whole file: the empty pair gets an empty line, the rest is
                // the training alignment (scores without the empty pair's length statistics)
                let got = model
                    .align_corpus(&all, d, true, &Cancel::default())
                    .unwrap();
                assert_eq!(got, want, "{d:?}");
            }
            // error
            let o = LoadOptions {
                empty_pairs: EmptyPairs::Error,
                ..LoadOptions::default()
            };
            let e = Corpus::from_joined_bytes(text.as_bytes(), &o)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("1 line pair with an empty side")
                    && e.contains(&format!("line 21 ({side})")),
                "{e}"
            );
        }
    }

    #[test]
    fn tension_zero_is_rejected_negative_is_not() {
        for t in [0.0, -0.0, 1e-300, -5e-7, f64::NAN, f64::INFINITY] {
            assert!(check_tension(t).is_err(), "{t}");
        }
        for t in [1e-6, -1e-6, 0.1, 4.0, -1.0, 50.0] {
            assert!(check_tension(t).is_ok(), "{t}");
        }
    }
}

/// Step 25 (review #9): a per-direction sink sees each finished direction's table before the next
/// direction trains, and the model is then dropped, unless it is kept; the outputs and the
/// exported tables equal those of a run that keeps both models.
#[test]
fn sink_sees_each_direction_before_the_next_trains() {
    use fastalign_core::pipeline::{train_with_sink, Finished, Progress};
    use std::sync::Mutex;

    let corpus = load(&corpus_text(300));
    let kept = train(
        &corpus,
        &TrainConfig::default(),
        &|_| {},
        &Cancel::default(),
    )
    .unwrap();

    for keep_models in [false, true] {
        let cfg = TrainConfig {
            keep_models,
            ..TrainConfig::default()
        };
        // events in order: ("iter", direction, iteration) and ("sink", direction)
        let events: Mutex<Vec<(&'static str, Direction, usize)>> = Mutex::new(Vec::new());
        let tables: Mutex<Vec<(Direction, Vec<u8>)>> = Mutex::new(Vec::new());
        let on_progress = |p: &Progress| {
            events
                .lock()
                .unwrap()
                .push(("iter", p.direction, p.iteration));
        };
        let mut sink = |f: &Finished<'_>| -> std::io::Result<()> {
            events.lock().unwrap().push(("sink", f.direction, 0));
            let mut buf = Vec::new();
            f.model
                .write_table(f.vocab, -4.0, &Cancel::default(), &mut buf)?;
            tables.lock().unwrap().push((f.direction, buf));
            Ok(())
        };
        let trained = train_with_sink(
            &corpus,
            &cfg,
            &on_progress,
            &Cancel::default(),
            Some(&mut sink),
        )
        .unwrap();

        let events = events.into_inner().unwrap();
        let sink_fwd = events
            .iter()
            .position(|e| *e == ("sink", Direction::Forward, 0))
            .expect("sink saw fwd");
        let first_inv = events
            .iter()
            .position(|e| e.0 == "iter" && e.1 == Direction::Inverse)
            .expect("inv trained");
        assert!(sink_fwd < first_inv, "fwd sink must precede inv training");
        assert!(events
            .last()
            .is_some_and(|e| e.0 == "sink" && e.1 == Direction::Inverse));

        for (d, table) in tables.into_inner().unwrap() {
            let mut want = Vec::new();
            kept.model
                .write_table(d, -4.0, &Cancel::default(), &mut want)
                .unwrap();
            assert_eq!(table, want, "{d:?} table");
        }
        for d in [Direction::Forward, Direction::Inverse] {
            assert_eq!(
                trained.output(d).unwrap().alignments,
                kept.output(d).unwrap().alignments
            );
            assert_eq!(trained.output(d).unwrap().log, kept.output(d).unwrap().log);
            // the model survives only if asked for
            assert_eq!(trained.model.dir(d).is_some(), keep_models, "{d:?} kept");
        }
        // streamed symmetrisation = collected
        let mut w = Vec::new();
        trained
            .write_symmetrized(Method::GrowDiagFinalAnd, &mut w)
            .unwrap();
        assert_eq!(w, trained.symmetrized(Method::GrowDiagFinalAnd).unwrap());
    }
}

/// A sink error stops the run and is returned.
#[test]
fn sink_error_stops_training() {
    use fastalign_core::pipeline::{train_with_sink, Finished};
    let corpus = load(&corpus_text(50));
    let mut calls = 0;
    let mut sink = |_: &Finished<'_>| -> std::io::Result<()> {
        calls += 1;
        Err(std::io::Error::other("disk full"))
    };
    let r = train_with_sink(
        &corpus,
        &TrainConfig::default(),
        &|_| {},
        &Cancel::default(),
        Some(&mut sink),
    );
    assert!(r.is_err());
    assert_eq!(
        calls, 1,
        "the inverse direction must not train after a failure"
    );
}

/// Step 25 (review #5): a command's parallel work runs on the requested pool, and every
/// parallel phase called inside it sees that many workers.
#[test]
fn with_threads_gives_the_requested_pool() {
    use fastalign_core::pipeline::with_threads;
    for n in [1, 2, 3] {
        assert_eq!(with_threads(n, rayon::current_num_threads).unwrap(), n);
    }
    // loading a corpus and exporting a table under the pool use only its workers
    let text = corpus_text(2000);
    let seen = with_threads(2, || {
        use rayon::prelude::*;
        let corpus = load(&text);
        let trained = train(
            &corpus,
            &TrainConfig::default(),
            &|_| {},
            &Cancel::default(),
        )
        .unwrap();
        let mut table = Vec::new();
        trained
            .model
            .write_table(Direction::Forward, -4.0, &Cancel::default(), &mut table)
            .unwrap();
        (0..64)
            .into_par_iter()
            .map(|_| std::thread::current().id())
            .collect::<std::collections::HashSet<_>>()
            .len()
    })
    .unwrap();
    assert!(seen <= 2);
}
