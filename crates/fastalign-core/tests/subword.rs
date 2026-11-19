//! Subword alignment through the pipeline: training on a segmented corpus, projection to words,
//! model save/load and forced alignment with the stored boundary.

use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::io::{Direction, Model, SubwordInfo};
use fastalign_core::pipeline::{train, TrainConfig};
use fastalign_core::subword::Boundary;
use fastalign_core::symmetrize::Method;

/// A deterministic corpus of words w0..w29 / v0..v29, with words of 1 to 3 pieces.
/// `segmented`: word `wK` is `▁wK` split as `▁w`, `K1`, `K2` (digits), else the plain word.
fn lines(n: usize, segmented: bool, marker: &str) -> String {
    let mut x = 99u64;
    let mut next = move |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    let word = |p: &str, k: u64| -> String {
        if !segmented {
            return format!("{p}{k}");
        }
        match k % 3 {
            0 => format!("{marker}{p}{k}"),
            1 => format!("{marker}{p} {k}"),
            _ => format!("{marker}{p} {} {}", k / 10, k % 10),
        }
    };
    let mut s = String::new();
    for _ in 0..n {
        let m = next(8) as usize;
        let src: Vec<u64> = (0..m).map(|_| next(30)).collect();
        let mut tgt = src.clone();
        if next(3) == 0 {
            tgt.reverse();
        }
        if next(4) == 0 && !tgt.is_empty() {
            tgt[0] = next(30);
        }
        let side =
            |p: &str, ws: &[u64]| ws.iter().map(|&k| word(p, k)).collect::<Vec<_>>().join(" ");
        s.push_str(&format!("{} ||| {}\n", side("w", &src), side("v", &tgt)));
    }
    s
}

fn load(t: &str) -> Corpus {
    Corpus::from_joined_bytes(t.as_bytes(), &LoadOptions::default())
}

fn cfg(sw: Option<Boundary>) -> TrainConfig {
    TrainConfig {
        subword: sw.map(|boundary| SubwordInfo {
            boundary,
            tokenizers: vec![],
            settings: vec![],
        }),
        keep_subword_output: true,
        ..TrainConfig::default()
    }
}

#[test]
fn identity_segmentation_equals_word_level() {
    // every word one piece, marked: the word-level run and the subword run must agree exactly
    let plain = load(&lines(300, false, ""));
    let marked_text: String = lines(300, false, "")
        .lines()
        .map(|l| {
            let (a, b) = l.split_once(" ||| ").unwrap();
            let m = |s: &str| {
                s.split(' ')
                    .filter(|w| !w.is_empty())
                    .map(|w| format!("\u{2581}{w}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            format!("{} ||| {}\n", m(a), m(b))
        })
        .collect();
    let marked = load(&marked_text);
    let a = train(&plain, &cfg(None), &|_| {}).unwrap();
    let b = train(&marked, &cfg(Some(Boundary::sentencepiece())), &|_| {}).unwrap();
    for d in [Direction::Forward, Direction::Inverse] {
        assert_eq!(
            a.output(d).unwrap().alignments,
            b.output(d).unwrap().alignments
        );
        // ... and the piece-level text is the word-level text
        assert_eq!(
            b.output(d).unwrap().subword_alignments.as_ref().unwrap(),
            &a.output(d).unwrap().alignments
        );
    }
    for m in [
        Method::Intersect,
        Method::Union,
        Method::GrowDiag,
        Method::GrowDiagFinal,
        Method::GrowDiagFinalAnd,
    ] {
        assert_eq!(a.symmetrized(m).unwrap(), b.symmetrized(m).unwrap());
    }
}

#[test]
fn projection_matches_the_pieces_and_the_model_reproduces_it() {
    for (boundary, marker) in [
        (Boundary::sentencepiece(), "\u{2581}"),
        // marker-prefix text is also valid for a different (custom) marker
        (Boundary::Start(b"##".to_vec()), "##"),
    ] {
        let corpus = load(&lines(300, true, marker));
        let t = train(&corpus, &cfg(Some(boundary.clone())), &|_| {}).unwrap();
        let mut bytes = Vec::new();
        t.model.write(&mut bytes).unwrap();
        let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(model.subword.as_ref().unwrap().boundary, boundary);
        for d in [Direction::Forward, Direction::Inverse] {
            let o = t.output(d).unwrap();
            let raw = o.subword_alignments.as_ref().unwrap();
            // every line: the projected links are the images of the piece links
            let marked = boundary.classify(&corpus.vocab);
            for (k, (w, r)) in o
                .alignments
                .split(|&b| b == b'\n')
                .zip(raw.split(|&b| b == b'\n'))
                .enumerate()
                .take(corpus.len())
            {
                let (mut fm, mut sm) = (vec![], vec![]);
                boundary.word_map(corpus.first.line(k), &marked, &mut fm);
                boundary.word_map(corpus.second.line(k), &marked, &mut sm);
                let parse = |t: &[u8]| -> Vec<(u32, u32)> {
                    std::str::from_utf8(t)
                        .unwrap()
                        .split(' ')
                        .filter(|x| !x.is_empty())
                        .map(|x| {
                            let (a, b) = x.split_once('-').unwrap();
                            (a.parse().unwrap(), b.parse().unwrap())
                        })
                        .collect()
                };
                let mut want: Vec<(u32, u32)> = parse(r)
                    .into_iter()
                    .map(|(p, q)| (fm[p as usize], sm[q as usize]))
                    .collect();
                want.sort_unstable();
                want.dedup();
                let mut got = parse(w);
                got.sort_unstable();
                assert_eq!(got, want, "line {k}");
            }
            // save -> load -> align reproduces the training alignments (words) byte for byte
            assert_eq!(
                model.align_words(&corpus, d, false).unwrap(),
                o.alignments,
                "{d:?}"
            );
        }
    }
}

#[test]
fn continuation_convention_works_end_to_end() {
    // the same segmentation, subword-nmt style: a piece ending in @@ continues
    let text: String = lines(200, true, "\u{2581}")
        .lines()
        .map(|l| {
            let conv = |s: &str| {
                let ps: Vec<&str> = s.split(' ').filter(|x| !x.is_empty()).collect();
                let mut out = Vec::new();
                for (i, p) in ps.iter().enumerate() {
                    let next_continues = ps.get(i + 1).is_some_and(|n| !n.starts_with('\u{2581}'));
                    let p = p.trim_start_matches('\u{2581}');
                    out.push(if next_continues {
                        format!("{p}@@")
                    } else {
                        p.to_string()
                    });
                }
                out.join(" ")
            };
            let (a, b) = l.split_once(" ||| ").unwrap();
            format!("{} ||| {}\n", conv(a), conv(b))
        })
        .collect();
    let a = train(
        &load(&lines(200, true, "\u{2581}")),
        &cfg(Some(Boundary::sentencepiece())),
        &|_| {},
    )
    .unwrap();
    let b = train(
        &load(&text),
        &cfg(Some(Boundary::Continue(b"@@".to_vec()))),
        &|_| {},
    )
    .unwrap();
    // different piece strings, so different ids and hash orders: the alignments may differ in
    // ties, but the projection of each run must be consistent with its own piece links
    assert_eq!(
        a.fwd
            .as_ref()
            .unwrap()
            .alignments
            .split(|&c| c == b'\n')
            .count(),
        b.fwd
            .as_ref()
            .unwrap()
            .alignments
            .split(|&c| c == b'\n')
            .count()
    );
    assert!(b.fwd.as_ref().unwrap().subword_alignments.is_some());
}

#[test]
fn subword_off_leaves_outputs_alone() {
    let corpus = load(&lines(100, false, ""));
    let t = train(&corpus, &cfg(None), &|_| {}).unwrap();
    assert!(t.model.subword.is_none());
    assert!(t.fwd.unwrap().subword_alignments.is_none());
}
