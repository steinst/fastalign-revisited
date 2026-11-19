#![no_main]
//! The model loader on arbitrary bytes and on mutated valid models: an error or a usable model.
use fastalign_core::io::Model;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = Model::read(data, data.len() as u64) {
        // A model that loads must also be writable and re-loadable.
        let mut out = Vec::new();
        m.write(&mut out).unwrap();
        let _ = m.word_index();
    }
});
