#![no_main]
//! The two-file reader: the input is cut into a source and a target text at a length byte.
//! Each file is one side (decision B): every line's words are that file line's space/tab
//! separated words, whatever tabs or `|||` it holds; small limits fail cleanly.
use fastalign_core::corpus::{Corpus, CorpusError, Limits, LoadOptions};
use libfuzzer_sys::fuzz_target;

fn lines(t: &[u8]) -> Vec<&[u8]> {
    let mut v: Vec<&[u8]> = t.split(|&b| b == b'\n').collect();
    if v.last().is_some_and(|l| l.is_empty()) {
        v.pop();
    }
    v
}

fn words(l: &[u8]) -> Vec<&[u8]> {
    l.split(|&b| b == b' ' || b == b'\t')
        .filter(|w| !w.is_empty())
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let (&sel, rest) = match data.split_first() {
        Some(x) => x,
        None => return,
    };
    let cut = (sel as usize * rest.len()) / 256;
    let (src, tgt) = rest.split_at(cut);
    let small = sel & 0x40 != 0;
    let limits = if small {
        Limits {
            max_words: 3 + (sel as usize & 7),
            max_lines: 1 + (sel as usize >> 3 & 7),
            max_line_tokens: 1 + (sel as usize & 3),
        }
    } else {
        Limits::DEFAULT
    };
    let opts = LoadOptions {
        max_len: None,
        chunks: Some(1 + (sel as usize & 7)),
        limits,
        ..LoadOptions::default()
    };
    let (ls, lt) = (lines(src), lines(tgt));
    match Corpus::from_pair_bytes(src, tgt, &opts) {
        Ok(c) => {
            assert_eq!(c.len(), ls.len());
            assert!(c.len() <= limits.max_lines);
            assert!(c.vocab.len() <= limits.max_words);
            for k in 0..c.len() {
                for (side, l) in [(&c.first, ls[k]), (&c.second, lt[k])] {
                    let w = words(l);
                    let got: Vec<&[u8]> = side.line(k).iter().map(|&i| c.vocab.word(i)).collect();
                    assert_eq!(got, w);
                }
            }
        }
        Err(CorpusError::LineCountMismatch { src, tgt }) => {
            assert_eq!((src, tgt), (ls.len(), lt.len()));
        }
        Err(CorpusError::TooManyWords { .. })
        | Err(CorpusError::TooManyLines { .. })
        | Err(CorpusError::LineTooLong { .. }) => assert!(small),
        Err(e) => panic!("{e}"),
    }
});
