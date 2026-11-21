#![no_main]
//! The `source ||| target` reader: any bytes, any chunking, with and without `max_len`, with
//! every `empty_pairs` mode.
use fastalign_core::corpus::{Corpus, CorpusError, EmptyPairs, LoadOptions};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let (&sel, text) = match data.split_first() {
        Some(x) => x,
        None => return,
    };
    let empty_pairs = [EmptyPairs::Keep, EmptyPairs::Skip, EmptyPairs::Error, EmptyPairs::Keep]
        [(sel >> 4 & 3) as usize];
    let opts = LoadOptions {
        max_len: (sel & 1 == 1).then_some((sel >> 6) as usize),
        chunks: Some(1 + (sel as usize >> 1 & 7)),
        empty_pairs,
        ..LoadOptions::default()
    };
    let c = match Corpus::from_joined_bytes(text, &opts) {
        Ok(c) => c,
        Err(CorpusError::EmptyPairs(r)) => {
            assert_eq!(empty_pairs, EmptyPairs::Error);
            assert!(r.total() > 0 && !r.first.is_empty());
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let n = c.len();
    let _ = (c.n_input_lines(), c.is_filtered());
    for k in 0..n {
        let _ = c.line_number(k);
        if empty_pairs != EmptyPairs::Keep {
            assert!(c.first.line_len(k) > 0 && c.second.line_len(k) > 0);
        }
    }
    let lines = vec![b'\n'; n];
    let expanded = c.expand_lines(&lines).unwrap();
    assert_eq!(expanded.len(), c.n_input_lines());
    if n > 0 {
        assert!(c.expand_lines(&vec![b'\n'; n - 1]).is_err());
    }
});
