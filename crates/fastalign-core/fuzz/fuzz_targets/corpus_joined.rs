#![no_main]
//! The `source ||| target` reader: any bytes, any chunking, with and without `max_len`.
use fastalign_core::corpus::{Corpus, LoadOptions};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let (&sel, text) = match data.split_first() {
        Some(x) => x,
        None => return,
    };
    let opts = LoadOptions {
        max_len: (sel & 1 == 1).then_some((sel >> 4) as usize),
        chunks: Some(1 + (sel as usize >> 1 & 7)),
    };
    let c = Corpus::from_joined_bytes(text, &opts);
    let n = c.len();
    let _ = (c.n_input_lines(), c.is_filtered());
    for k in 0..n {
        let _ = c.line_number(k);
    }
    let _ = c.expand_lines(&vec![b'\n'; n]);
});
