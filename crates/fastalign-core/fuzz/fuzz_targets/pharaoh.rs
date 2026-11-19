#![no_main]
//! Pharaoh text parsing and every symmetrisation method on arbitrary text. The input is split in
//! two at its first 0xFF byte.
use fastalign_core::symmetrize::{parse_line, symmetrize_text, Method};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let at = data.iter().position(|&b| b == 0xff).unwrap_or(data.len());
    let (a, b) = data.split_at(at);
    let b = b.get(1..).unwrap_or(&[]);
    for line in a.split(|&c| c == b'\n') {
        let _ = parse_line(line);
    }
    for m in [
        Method::Intersect,
        Method::Union,
        Method::GrowDiag,
        Method::GrowDiagFinal,
        Method::GrowDiagFinalAnd,
        Method::Invert,
    ] {
        let _ = symmetrize_text(a, Some(b), m, &mut std::io::sink());
    }
});
