//! fastalign-core: pure-Rust, bit-exact reimplementation of fast_align (no Python dependency).

pub mod align;
pub mod cancel;
pub mod corpus;
pub mod io;
pub mod math;
pub mod oracle;
pub mod pipeline;
pub mod stlhash;
pub mod subword;
pub mod symmetrize;
pub mod train;
pub mod ttable;

/// Crate version, taken from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::version().is_empty());
    }
}
