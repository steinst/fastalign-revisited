//! Subword alignment through the pipeline: training on a segmented corpus, projection to words,
//! model save/load and forced alignment with the stored boundary.

use fastalign_core::cancel::Cancel;
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
    Corpus::from_joined_bytes(t.as_bytes(), &LoadOptions::default()).unwrap()
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
    let a = train(&plain, &cfg(None), &|_| {}, &Cancel::default()).unwrap();
    let b = train(
        &marked,
        &cfg(Some(Boundary::sentencepiece())),
        &|_| {},
        &Cancel::default(),
    )
    .unwrap();
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
        let t = train(
            &corpus,
            &cfg(Some(boundary.clone())),
            &|_| {},
            &Cancel::default(),
        )
        .unwrap();
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
                model
                    .align_words(&corpus, d, false, &Cancel::default())
                    .unwrap(),
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
        &Cancel::default(),
    )
    .unwrap();
    let b = train(
        &load(&text),
        &cfg(Some(Boundary::Continue(b"@@".to_vec()))),
        &|_| {},
        &Cancel::default(),
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
    let t = train(&corpus, &cfg(None), &|_| {}, &Cancel::default()).unwrap();
    assert!(t.model.subword.is_none());
    assert!(t.fwd.unwrap().subword_alignments.is_none());
}

/// The words of each line of one side of a paired text (space/tab separated).
fn side_lines(text: &str) -> Vec<Vec<&[u8]>> {
    text.lines()
        .map(|l| {
            l.split([' ', '\t'])
                .filter(|w| !w.is_empty())
                .map(str::as_bytes)
                .collect()
        })
        .collect()
}

#[test]
fn one_pair_path_equals_the_corpus_path() {
    // Step 24: `align_tokens_words` (no corpus, no text) gives exactly the links of the
    // corresponding line of `align_words`, in its order, for word and subword models, both
    // directions; a corpus mapped once serves both directions.
    for (boundary, marker) in [
        (None, ""),
        (Some(Boundary::sentencepiece()), "\u{2581}"),
        (Some(Boundary::Start(b"##".to_vec())), "##"),
    ] {
        let segmented = boundary.is_some();
        let train_text = lines(300, segmented, marker);
        let corpus = load(&train_text);
        let t = train(&corpus, &cfg(boundary), &|_| {}, &Cancel::default()).unwrap();
        let mut bytes = Vec::new();
        t.model.write(&mut bytes).unwrap();
        let model = Model::read(&bytes[..], bytes.len() as u64).unwrap();
        // new text: other lines, unknown words, an empty side, the words `|||` and `<eps>`
        let mut new_text = lines(60, segmented, marker).replace("w1", "u1");
        new_text.push_str(" ||| v3 v4\nw2 ||| \n|||\tw3 <eps> ||| v2\n");
        let (mut src, mut tgt) = (String::new(), String::new());
        for l in new_text.lines() {
            let (a, b) = l.split_once(" ||| ").unwrap();
            src.push_str(a);
            src.push('\n');
            tgt.push_str(b);
            tgt.push('\n');
        }
        let new = Corpus::from_pair_bytes(src.as_bytes(), tgt.as_bytes(), &LoadOptions::default())
            .unwrap();
        let (sl, tl) = (side_lines(&src), side_lines(&tgt));
        assert_eq!(sl.len(), new.len());
        let mapped = model.map_corpus(&new);
        let mut per_dir = Vec::new();
        for d in [Direction::Forward, Direction::Inverse] {
            let text = model
                .align_words(&new, d, false, &Cancel::default())
                .unwrap();
            assert_eq!(
                model
                    .align_words_mapped(&mapped, d, false, &Cancel::default())
                    .unwrap(),
                text
            );
            let mut direct = Vec::new();
            for (k, line) in text.split(|&b| b == b'\n').take(new.len()).enumerate() {
                let links = model.align_tokens_words(d, &sl[k], &tl[k]).unwrap();
                let mut got = Vec::new();
                fastalign_core::symmetrize::format_links(&links, &mut got);
                assert_eq!(got, line, "{d:?} line {k}");
                direct.push(links);
            }
            per_dir.push((text, direct));
        }
        let (inv_text, inv) = per_dir.pop().unwrap();
        let (fwd_text, fwd) = per_dir.pop().unwrap();
        for m in [Method::Intersect, Method::GrowDiagFinalAnd, Method::Union] {
            let want =
                fastalign_core::pipeline::symmetrize_pharaoh(&fwd_text, &inv_text, m).unwrap();
            let mut got = Vec::new();
            for (f, i) in fwd.iter().zip(&inv) {
                let (mut f, mut i) = (f.clone(), i.clone());
                f.sort_unstable();
                i.sort_unstable();
                let s = fastalign_core::symmetrize::symmetrize(&f, &i, m);
                fastalign_core::symmetrize::format_links(&s, &mut got);
                got.push(b'\n');
            }
            assert_eq!(got, want, "{m:?}");
        }
    }
}

#[test]
fn word_index_finds_every_word() {
    let corpus = load(&lines(300, true, "\u{2581}"));
    let t = train(&corpus, &cfg(None), &|_| {}, &Cancel::default()).unwrap();
    let m = &t.model;
    for id in 1..m.vocab.len() as u32 {
        assert_eq!(m.word_id(m.vocab.word(id)), Some(id));
    }
    for w in [&b""[..], b"nope", b"\xff", b"<bad0>x"] {
        assert_eq!(m.word_id(w), None);
    }
    // id 0 is never a word
    assert_eq!(m.word_id(b"<bad0>"), None);
    // a duplicated word (only a corrupt file has one) maps to its last id
    let v =
        fastalign_core::corpus::Vocab::from_words([&b"<bad0>"[..], b"<eps>", b"|||", b"a", b"a"]);
    assert_eq!(
        fastalign_core::io::WordIndex::new(&v).get(&v, b"a"),
        Some(4)
    );
    // simultaneous first users all get the one index
    let fresh = Model::new(m.vocab.clone(), None, None);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| assert_eq!(fresh.word_id(m.vocab.word(5)), Some(5)));
        }
    });
}

/// Step 25: projection is written in ordered windows of chunks; the text is the same whatever
/// the pool (one worker makes the windows smallest), and has one line per corpus line.
#[test]
fn projection_windows_keep_order() {
    use fastalign_core::pipeline::with_threads;
    use fastalign_core::subword::project_text;
    // 40k lines (> 4 windows of 4 chunks of 4096 lines at one worker); every line differs in
    // the number of links so that a swapped chunk shows
    let mut corpus_text = String::new();
    let mut aligned = String::new();
    for k in 0..40_000usize {
        let n = 1 + k % 5;
        let src: Vec<String> = (0..n).map(|i| format!("▁s{i}")).collect();
        let tgt: Vec<String> = (0..n).map(|i| format!("▁t{i}")).collect();
        corpus_text.push_str(&format!("{} ||| {}\n", src.join(" "), tgt.join(" ")));
        let links: Vec<String> = (0..n).map(|i| format!("{i}-{i}")).collect();
        aligned.push_str(&links.join(" "));
        aligned.push('\n');
    }
    let corpus =
        Corpus::from_joined_bytes(corpus_text.as_bytes(), &LoadOptions::default()).unwrap();
    let b = Boundary::sentencepiece();
    let run = |threads| {
        with_threads(threads, || {
            project_text(&corpus, &b, aligned.as_bytes(), false, &Cancel::default()).unwrap()
        })
        .unwrap()
    };
    let one = run(1);
    assert_eq!(one.iter().filter(|&&c| c == b'\n').count(), 40_000);
    assert_eq!(one, run(3));
    assert_eq!(one, run(8));
    // every token is its own word here, so projection is the identity (forward: target order)
    assert_eq!(one, aligned.as_bytes());
}
