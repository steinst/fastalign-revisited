//! Emulation of the iteration order of libstdc++'s `std::unordered_map` (unique keys,
//! `max_load_factor` 1.0), as used by fast_align's translation table and `size_counts`.
//!
//! The algorithm is described in `docs/SPEC-fast-align.md` §14 and is tested against the
//! `stlhash_probe` vectors in `tests/data/stlhash/`. Only the node *order* is modelled: keys are
//! identified by their insertion index and carry their hash code; values are irrelevant.

/// libstdc++'s bucket-count table (`std::__detail::__prime_list` in `libstdc++.so.6.0.33`,
/// LP64): 304 primes and a repeated last prime as sentinel.
#[rustfmt::skip]
const PRIMES: [u64; 305] = [
    2, 3, 5, 7,
    11, 13, 17, 19,
    23, 29, 31, 37,
    41, 43, 47, 53,
    59, 61, 67, 71,
    73, 79, 83, 89,
    97, 103, 109, 113,
    127, 137, 139, 149,
    157, 167, 179, 193,
    199, 211, 227, 241,
    257, 277, 293, 313,
    337, 359, 383, 409,
    439, 467, 503, 541,
    577, 619, 661, 709,
    761, 823, 887, 953,
    1031, 1109, 1193, 1289,
    1381, 1493, 1613, 1741,
    1879, 2029, 2179, 2357,
    2549, 2753, 2971, 3209,
    3469, 3739, 4027, 4349,
    4703, 5087, 5503, 5953,
    6427, 6949, 7517, 8123,
    8783, 9497, 10273, 11113,
    12011, 12983, 14033, 15173,
    16411, 17749, 19183, 20753,
    22447, 24281, 26267, 28411,
    30727, 33223, 35933, 38873,
    42043, 45481, 49201, 53201,
    57557, 62233, 67307, 72817,
    78779, 85229, 92203, 99733,
    107897, 116731, 126271, 136607,
    147793, 159871, 172933, 187091,
    202409, 218971, 236897, 256279,
    277261, 299951, 324503, 351061,
    379787, 410857, 444487, 480881,
    520241, 562841, 608903, 658753,
    712697, 771049, 834181, 902483,
    976369, 1056323, 1142821, 1236397,
    1337629, 1447153, 1565659, 1693859,
    1832561, 1982627, 2144977, 2320627,
    2510653, 2716249, 2938679, 3179303,
    3439651, 3721303, 4026031, 4355707,
    4712381, 5098259, 5515729, 5967347,
    6456007, 6984629, 7556579, 8175383,
    8844859, 9569143, 10352717, 11200489,
    12117689, 13109983, 14183539, 15345007,
    16601593, 17961079, 19431899, 21023161,
    22744717, 24607243, 26622317, 28802401,
    31160981, 33712729, 36473443, 39460231,
    42691603, 46187573, 49969847, 54061849,
    58488943, 63278561, 68460391, 74066549,
    80131819, 86693767, 93793069, 101473717,
    109783337, 118773397, 128499677, 139022417,
    150406843, 162723577, 176048909, 190465427,
    206062531, 222936881, 241193053, 260944219,
    282312799, 305431229, 330442829, 357502601,
    386778277, 418451333, 452718089, 489790921,
    529899637, 573292817, 620239453, 671030513,
    725980837, 785430967, 849749479, 919334987,
    994618837, 1076067617, 1164186217, 1259520799,
    1362662261, 1474249943, 1594975441, 1725587117,
    1866894511, 2019773507, 2185171673, 2364114217,
    2557710269, 2767159799, 2993761039, 3238918481,
    3504151727, 3791104843, 4101556399, 4294967291,
    6442450933, 8589934583, 12884901857, 17179869143,
    25769803693, 34359738337, 51539607367, 68719476731,
    103079215087, 137438953447, 206158430123, 274877906899,
    412316860387, 549755813881, 824633720731, 1099511627689,
    1649267441579, 2199023255531, 3298534883309, 4398046511093,
    6597069766607, 8796093022151, 13194139533241, 17592186044399,
    26388279066581, 35184372088777, 52776558133177, 70368744177643,
    105553116266399, 140737488355213, 211106232532861, 281474976710597,
    562949953421231, 1125899906842597, 2251799813685119, 4503599627370449,
    9007199254740881, 18014398509481951, 36028797018963913, 72057594037927931,
    144115188075855859, 288230376151711717, 576460752303423433, 1152921504606846883,
    2305843009213693951, 4611686018427387847, 9223372036854775783, 18446744073709551557,
    18446744073709551557,
];

/// Bucket counts for small requests (`_Prime_rehash_policy::_M_next_bkt`).
const FAST_BKT: [u64; 14] = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13];

const NIL: u32 = u32::MAX;
/// The list-head sentinel (`_M_before_begin`), stored in a bucket slot.
const HEAD: u32 = u32::MAX - 1;

/// Hash of a translation-table key (`std::hash<unsigned>`): the identity.
#[inline]
pub fn hash_u32(key: u32) -> u64 {
    u64::from(key)
}

/// fast_align's `PairHash` for a `size_counts` key `(target length, source length)` as `i16`.
#[inline]
pub fn hash_len_pair(m: i16, n: i16) -> u64 {
    u64::from((u32::from(m as u16) << 16) | (n as i32 as u32))
}

/// Node order of one `unordered_map` under a sequence of insertions of *new* keys.
///
/// Node `k` is the `k`-th inserted key. The structure can be reused with [`StlOrder::clear`] to
/// avoid reallocating.
#[derive(Default, Clone, Debug)]
pub struct StlOrder {
    hash: Vec<u64>,
    next: Vec<u32>,
    buckets: Vec<u32>,
    scratch: Vec<u32>,
    head: u32,
    next_resize: u64,
}

impl StlOrder {
    pub fn new() -> Self {
        let mut s = Self::default();
        s.clear();
        s
    }

    /// Reset to a default-constructed map (one bucket, nothing allocated).
    pub fn clear(&mut self) {
        self.hash.clear();
        self.next.clear();
        self.buckets.clear();
        self.buckets.push(NIL);
        self.head = NIL;
        self.next_resize = 0;
    }

    pub fn len(&self) -> usize {
        self.hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hash.is_empty()
    }

    pub fn bucket_count(&self) -> u64 {
        self.buckets.len() as u64
    }

    #[inline]
    fn get_next(&self, x: u32) -> u32 {
        if x == HEAD {
            self.head
        } else {
            self.next[x as usize]
        }
    }

    #[inline]
    fn set_next(&mut self, x: u32, v: u32) {
        if x == HEAD {
            self.head = v;
        } else {
            self.next[x as usize] = v;
        }
    }

    fn next_bkt(&mut self, n: u64) -> u64 {
        if n < FAST_BKT.len() as u64 {
            if n == 0 {
                return 1;
            }
            let b = FAST_BKT[n as usize];
            self.next_resize = b;
            return b;
        }
        // lower_bound over PRIMES[6..303); 303 = index of the last prime proper.
        let last = PRIMES.len() - 2;
        let i = 6 + PRIMES[6..last].partition_point(|&p| p < n);
        if i == last {
            self.next_resize = u64::MAX;
        } else {
            self.next_resize = (PRIMES[i] as f64).floor() as u64;
        }
        PRIMES[i]
    }

    fn rehash(&mut self, nb: u64) {
        let nb_us = usize::try_from(nb).expect("bucket count fits in memory");
        let mut newb = std::mem::take(&mut self.scratch);
        newb.clear();
        newb.resize(nb_us, NIL);
        let mut p = self.head;
        self.head = NIL;
        let mut bbegin = 0usize;
        while p != NIL {
            let nx = self.next[p as usize];
            let b = (self.hash[p as usize] % nb) as usize;
            if newb[b] == NIL {
                self.next[p as usize] = self.head;
                self.head = p;
                newb[b] = HEAD;
                if self.next[p as usize] != NIL {
                    newb[bbegin] = p;
                }
                bbegin = b;
            } else {
                let before = newb[b];
                let after = self.get_next(before);
                self.next[p as usize] = after;
                self.set_next(before, p);
            }
            p = nx;
        }
        self.scratch = std::mem::replace(&mut self.buckets, newb);
    }

    /// Insert a key that is not yet present, with hash code `hash`. Returns its node index.
    pub fn insert_new(&mut self, hash: u64) -> u32 {
        let n = self.hash.len() as u64;
        let nb = self.buckets.len() as u64;
        if n + 1 > self.next_resize {
            let need = (n + 1).max(if self.next_resize == 0 { 11 } else { 0 });
            if need >= nb {
                let target = self.next_bkt((need + 1).max(nb * 2));
                self.rehash(target);
            } else {
                self.next_resize = nb;
            }
        }
        let node = u32::try_from(n).expect("fewer than 2^32-2 keys");
        assert!(node < HEAD);
        self.hash.push(hash);
        self.next.push(NIL);
        let nb = self.buckets.len() as u64;
        let b = (hash % nb) as usize;
        let before = self.buckets[b];
        if before != NIL {
            let after = self.get_next(before);
            self.next[node as usize] = after;
            self.set_next(before, node);
        } else {
            self.next[node as usize] = self.head;
            self.head = node;
            let nx = self.next[node as usize];
            if nx != NIL {
                let nb2 = (self.hash[nx as usize] % nb) as usize;
                self.buckets[nb2] = node;
            }
            self.buckets[b] = HEAD;
        }
        node
    }

    /// Iteration order: node indices (= insertion indices) from `begin()` to `end()`.
    pub fn order(&self) -> impl Iterator<Item = u32> + '_ {
        let mut p = self.head;
        std::iter::from_fn(move || {
            if p == NIL {
                None
            } else {
                let cur = p;
                p = self.next[cur as usize];
                Some(cur)
            }
        })
    }

    /// Append the iteration order to `out`.
    pub fn order_into(&self, out: &mut Vec<u32>) {
        out.extend(self.order());
    }
}

/// Iteration order of an `unordered_map<unsigned, …>` after inserting `keys` (distinct, in
/// insertion order), as indices into `keys`.
pub fn u32_order(keys: &[u32]) -> Vec<u32> {
    let mut s = StlOrder::new();
    for &k in keys {
        s.insert_new(hash_u32(k));
    }
    s.order().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_insert_allocates_13_buckets() {
        let mut s = StlOrder::new();
        assert_eq!(s.bucket_count(), 1);
        s.insert_new(5);
        assert_eq!(s.bucket_count(), 13);
    }

    #[test]
    fn growth_sequence() {
        // spec §14.3: 13, then 29 at key 14, 59 at key 30, 127 at key 60 ...
        let mut s = StlOrder::new();
        let mut seen = vec![];
        for k in 0..2400u32 {
            s.insert_new(hash_u32(k));
            if seen.last() != Some(&s.bucket_count()) {
                seen.push(s.bucket_count());
            }
        }
        assert_eq!(seen, [13, 29, 59, 127, 257, 541, 1109, 2357, 5087]);
    }

    #[test]
    fn sequential_small() {
        // probe: keys 1..=20 iterate as 20 19 .. 14 1 2 .. 13 (29 buckets)
        let keys: Vec<u32> = (1..=20).collect();
        let order: Vec<u32> = u32_order(&keys).iter().map(|&i| keys[i as usize]).collect();
        let want: Vec<u32> = (14..=20).rev().chain(1..=13).collect();
        assert_eq!(order, want);
    }

    #[test]
    fn pair_hash() {
        assert_eq!(hash_len_pair(3, 4), (3 << 16) | 4);
        assert_eq!(hash_len_pair(4, -3), 0xFFFF_FFFD);
        assert_eq!(hash_len_pair(-1, 0), 0xFFFF_0000);
    }
}
