#![no_main]
//! The word-boundary parser and projection on arbitrary corpora and alignment text. The input is
//! split at its first two 0xFF bytes into: corpus text (`src ||| tgt` lines), Pharaoh text and a
//! marker; every boundary kind and both orders are tried.
use fastalign_core::cancel::Cancel;
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::subword::{project_text, Boundary};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut parts = data.splitn(3, |&b| b == 0xff);
    let corpus_text = parts.next().unwrap_or(&[]);
    let pharaoh = parts.next().unwrap_or(&[]);
    let marker = parts.next().unwrap_or(b"@@");
    let corpus = Corpus::from_joined_bytes(corpus_text, &LoadOptions::default()).unwrap();
    for kind in ["start", "continue"] {
        let Ok(b) = Boundary::from_parts(kind, marker) else {
            continue;
        };
        for inverse in [false, true] {
            let _ = project_text(&corpus, &b, pharaoh, inverse, &Cancel::default());
        }
    }
    // a text with the right number of lines exercises the projection itself
    let n = corpus.len();
    let own: String = (0..n)
        .map(|k| {
            let (f, s) = (corpus.first.line_len(k), corpus.second.line_len(k));
            let links: Vec<String> = (0..f.min(s)).map(|i| format!("{i}-{}", s - 1 - i)).collect();
            format!("{}\n", links.join(" "))
        })
        .collect();
    for kind in ["start", "continue"] {
        if let Ok(b) = Boundary::from_parts(kind, marker) {
            let _ = project_text(&corpus, &b, own.as_bytes(), false, &Cancel::default());
        }
    }
});
