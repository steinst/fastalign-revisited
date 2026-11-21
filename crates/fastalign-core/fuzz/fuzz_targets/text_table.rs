#![no_main]
//! fast_align's `-f` text table reader plus forced alignment against a corpus: the input is
//! `table bytes`, a 0xFF separator byte, then the corpus text.
use fastalign_core::cancel::Cancel;
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::oracle::{force_align, ForceOptions, TextTable};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let at = data.iter().position(|&b| b == 0xff).unwrap_or(data.len());
    let (table, text) = data.split_at(at);
    let corpus = Corpus::from_joined_bytes(text.get(1..).unwrap_or(&[]), &LoadOptions::default()).unwrap();
    let t = TextTable::parse(table, &corpus.vocab);
    for (diag, null, rev) in [(true, true, false), (false, false, true)] {
        let opts = ForceOptions {
            reverse: rev,
            favor_diagonal: diag,
            use_null: null,
            p0: 0.08,
            tension: 4.0,
            mean_srclen_multiplier: 1.0,
        };
        let _ = force_align(&corpus, &t, &opts, &Cancel::default(), &mut std::io::sink(), &mut std::io::sink());
    }
});
