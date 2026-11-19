//! The libstdc++ unordered_map emulation against the vectors printed by
//! `benchmarks/stlhash_probe.cc` (regenerate with `benchmarks/stlhash_vectors.py`).

use fastalign_core::stlhash::{hash_len_pair, hash_u32, StlOrder};

fn data(name: &str) -> String {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/stlhash/");
    std::fs::read_to_string(format!("{p}{name}")).unwrap()
}

#[test]
fn probe_vectors() {
    let cases = data("cases.txt");
    let expected = data("expected.txt");
    let exp: Vec<&str> = expected.lines().collect();
    let mut n_cases = 0;
    for (idx, line) in cases.lines().enumerate() {
        let mut toks = line.split_whitespace();
        let kind = toks.next().unwrap();
        let name = toks.next().unwrap();
        let keys: Vec<&str> = toks.collect();
        let e = &exp[idx * 5..idx * 5 + 5];
        assert_eq!(e[0], format!("case {name} {kind} {}", keys.len()));
        let mut s = StlOrder::new();
        let mut index = std::collections::HashMap::new();
        let mut names = Vec::new();
        let mut bkt = String::from("bkt");
        for k in &keys {
            if !index.contains_key(k) {
                let h = if kind == "u32" {
                    hash_u32(k.parse().unwrap())
                } else {
                    let (a, b) = k.split_once(',').unwrap();
                    hash_len_pair(a.parse().unwrap(), b.parse().unwrap())
                };
                index.insert(*k, s.insert_new(h));
                names.push(*k);
            }
            bkt.push_str(&format!(" {}", s.bucket_count()));
        }
        assert_eq!(e[1], bkt, "{name}: bucket counts");
        let order: Vec<&str> = s.order().map(|i| names[i as usize]).collect();
        let mut want = e[2].split_whitespace();
        assert_eq!(want.next(), Some("order"));
        assert_eq!(order, want.collect::<Vec<_>>(), "{name}: iteration order");
        // Copy-assignment and vector swap keep the order (spec §14.5).
        assert_eq!(e[3]["copy".len()..], e[2]["order".len()..], "{name}: copy");
        assert_eq!(e[4]["swap".len()..], e[2]["order".len()..], "{name}: swap");
        n_cases += 1;
    }
    assert_eq!(n_cases, 28);
}

#[test]
fn reuse_after_clear() {
    let mut s = StlOrder::new();
    for k in 0..100 {
        s.insert_new(hash_u32(k * 7));
    }
    let a: Vec<u32> = s.order().collect();
    s.clear();
    for k in 0..100 {
        s.insert_new(hash_u32(k * 7));
    }
    assert_eq!(a, s.order().collect::<Vec<_>>());
}
