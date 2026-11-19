//! math.rs against fast_align's own functions: vectors from `benchmarks/math_probe.cc`
//! (regenerate with `benchmarks/math_vectors.sh`). Every result must be bit-identical.

use fastalign_core::math;

fn f(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

fn u(s: &str) -> u32 {
    s.parse().unwrap()
}

#[test]
fn math_matches_fast_align() {
    let text = include_str!("data/math/vectors.txt");
    let mut seen = std::collections::BTreeMap::<&str, usize>::new();
    let mut bad = Vec::new();
    for line in text.lines() {
        let w: Vec<&str> = line.split(' ').collect();
        let (got, want) = match w[0] {
            "feature" => (
                math::feature(u(w[1]), u(w[2]), u(w[3]), u(w[4])).to_bits(),
                f(w[5]).to_bits(),
            ),
            "uprob" => (
                math::unnormalized_prob(u(w[1]), u(w[2]), u(w[3]), u(w[4]), f(w[5])).to_bits(),
                f(w[6]).to_bits(),
            ),
            "z" => (
                math::compute_z(u(w[1]), u(w[2]), u(w[3]), f(w[4])).to_bits(),
                f(w[5]).to_bits(),
            ),
            "dlogz" => (
                math::compute_dlogz(u(w[1]), u(w[2]), u(w[3]), f(w[4])).to_bits(),
                f(w[5]).to_bits(),
            ),
            "digamma" => (math::digamma(f(w[1])).to_bits(), f(w[2]).to_bits()),
            "logpois" => (
                math::log_poisson(u(w[1]), f(w[2])).to_bits(),
                f(w[3]).to_bits(),
            ),
            "pow2" => (math::pow(2.0, f(w[1])).to_bits(), f(w[2]).to_bits()),
            "g" => {
                let s = math::fmt_g(f(w[1]));
                if s != w[2] {
                    bad.push(format!("{line}: got {s}"));
                }
                *seen.entry("g").or_default() += 1;
                continue;
            }
            other => panic!("unknown vector kind {other}"),
        };
        *seen.entry(w[0]).or_default() += 1;
        if got != want {
            bad.push(format!("{line}: got {got:016x}"));
        }
    }
    assert_eq!(seen.len(), 8, "{seen:?}");
    assert!(
        bad.is_empty(),
        "{} mismatches, first: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}
