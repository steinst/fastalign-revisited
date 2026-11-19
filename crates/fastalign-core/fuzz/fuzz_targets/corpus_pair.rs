#![no_main]
//! The two-file reader: the input is cut into a source and a target text at a length byte.
use fastalign_core::corpus::{Corpus, LoadOptions};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let (&sel, rest) = match data.split_first() {
        Some(x) => x,
        None => return,
    };
    let cut = (sel as usize * rest.len()) / 256;
    let (src, tgt) = rest.split_at(cut);
    let opts = LoadOptions {
        max_len: None,
        chunks: Some(1 + (sel as usize & 7)),
    };
    let _ = Corpus::from_pair_bytes(src, tgt, &opts);
});
