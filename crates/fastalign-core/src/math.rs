//! Numerical building blocks of fast_align's model, evaluated exactly as the reference build
//! does (`docs/SPEC-fast-align.md` §1, §12, §13, §15.1).
//!
//! Every function performs the same IEEE-754 operations in the same order as the reference, and
//! calls the same glibc functions (`exp`, `log`, `pow`, `lgamma`). Rust does not contract
//! `a * b + c` into an FMA and does not reassociate, so the results are bit-identical; the unit
//! tests check this against vectors printed by `benchmarks/math_probe.cc`, which calls
//! fast_align's own functions.
//!
//! Index conventions follow the spec: in the prior functions `i` is the 1-based target position,
//! `j` the source position, `m` the target length and `n` the source length.

use std::fmt::Write as _;

extern "C" {
    /// glibc `lgamma` (Rust's std does not expose it on stable).
    #[link_name = "lgamma"]
    fn c_lgamma(x: f64) -> f64;
}

/// glibc `lgamma(x)`.
#[inline]
pub fn lgamma(x: f64) -> f64 {
    // SAFETY: `lgamma` is a pure libm function; it only writes the global `signgam`, which
    // nothing reads.
    unsafe { c_lgamma(x) }
}

/// glibc `pow(x, y)`, never simplified by the compiler.
///
/// LLVM may rewrite `pow` with a constant base (for example `pow(2.0, y)` into `exp2(y)`), whose
/// result can differ in the last bit. The arguments are passed through [`std::hint::black_box`]
/// so that the call always reaches glibc's `pow`, as in the reference binary.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    std::hint::black_box(x).powf(std::hint::black_box(y))
}

/// The C++ conversion `static_cast<unsigned>(double)` as compiled for x86-64 by GCC: a 64-bit
/// truncating conversion (`cvttsd2si`) whose low 32 bits are kept. Out-of-range values and NaN
/// give `0x8000_0000_0000_0000`, i.e. 0 after truncation. For values in `[0, 2^32)` this is
/// plain truncation.
#[inline]
pub fn trunc_to_u32(x: f64) -> u32 {
    const LIM: f64 = 9_223_372_036_854_775_808.0; // 2^63
    let v: i64 = if x.is_nan() || x >= LIM || x < -LIM {
        i64::MIN
    } else {
        x as i64
    };
    v as u32
}

/// `h(i, j, m, n) = −|j/n − i/m|` (spec §12).
#[inline]
pub fn feature(i: u32, j: u32, m: u32, n: u32) -> f64 {
    -(f64::from(j) / f64::from(n) - f64::from(i) / f64::from(m)).abs()
}

/// `exp(h(i, j, m, n) · λ)`.
#[inline]
pub fn unnormalized_prob(i: u32, j: u32, m: u32, n: u32, lambda: f64) -> f64 {
    (feature(i, j, m, n) * lambda).exp()
}

/// Normaliser of the diagonal prior for target position `i` (spec §12, `ComputeZ`).
pub fn compute_z(i: u32, m: u32, n: u32, lambda: f64) -> f64 {
    let split = f64::from(i) * f64::from(n) / f64::from(m);
    let floor = trunc_to_u32(split);
    let ceil = floor.wrapping_add(1);
    let ratio = (-lambda / f64::from(n)).exp();
    let num_top = n.wrapping_sub(floor);
    let mut ezt = 0.0;
    let mut ezb = 0.0;
    if num_top != 0 {
        ezt = unnormalized_prob(i, ceil, m, n, lambda) * (1.0 - pow(ratio, f64::from(num_top)))
            / (1.0 - ratio);
    }
    if floor != 0 {
        ezb = unnormalized_prob(i, floor, m, n, lambda) * (1.0 - pow(ratio, f64::from(floor)))
            / (1.0 - ratio);
    }
    ezb + ezt
}

/// Sum of the arithmetico-geometric series used by [`compute_dlogz`].
fn arith_geom_series(a1: f64, g1: f64, r: f64, d: f64, k: u32) -> f64 {
    let g_np1 = g1 * pow(r, f64::from(k));
    let a_n = d * f64::from(k.wrapping_sub(1)) + a1;
    let x_1 = a1 * g1;
    let g_2 = g1 * r;
    let rm1 = r - 1.0;
    (a_n * g_np1 - x_1) / rm1 - d * (g_np1 - g_2) / (rm1 * rm1)
}

/// Derivative of `log Z` with respect to λ (spec §12, `ComputeDLogZ`), including fast_align's
/// swapped `m`/`n` in the normaliser.
pub fn compute_dlogz(i: u32, m: u32, n: u32, lambda: f64) -> f64 {
    let z = compute_z(i, n, m, lambda);
    let split = f64::from(i) * f64::from(n) / f64::from(m);
    let floor = trunc_to_u32(split);
    let ceil = floor.wrapping_add(1);
    let ratio = (-lambda / f64::from(n)).exp();
    let d = -1.0 / f64::from(n);
    let num_top = n.wrapping_sub(floor);
    let mut pct = 0.0;
    let mut pcb = 0.0;
    if num_top != 0 {
        pct = arith_geom_series(
            feature(i, ceil, m, n),
            unnormalized_prob(i, ceil, m, n, lambda),
            ratio,
            d,
            num_top,
        );
    }
    if floor != 0 {
        pcb = arith_geom_series(
            feature(i, floor, m, n),
            unnormalized_prob(i, floor, m, n, lambda),
            ratio,
            d,
            floor,
        );
    }
    (pct + pcb) / z
}

/// fast_align's approximate digamma (spec §13).
pub fn digamma(mut x: f64) -> f64 {
    const C1: f64 = 1.0 / 24.0;
    const C2: f64 = 7.0 / 960.0;
    const C3: f64 = 31.0 / 8064.0;
    const C4: f64 = 127.0 / 30720.0;
    let mut r = 0.0;
    while x < 7.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    x -= 0.5;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    let t = x.ln() + C1 * xx2 - C2 * xx4 + C3 * xx4 * xx2 - C4 * xx4 * xx4;
    r + t
}

/// `log Poisson(x; λ) = log(λ)·x − lgamma(x + 1) − λ` (spec §13).
pub fn log_poisson(x: u32, lambda: f64) -> f64 {
    lambda.ln() * f64::from(x) - lgamma(f64::from(x.wrapping_add(1))) - lambda
}

/// Append `x` formatted as C's `printf("%.6g")` with glibc (the C++ stream default; spec §15.1).
pub fn write_g(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str(if x.is_sign_negative() { "-nan" } else { "nan" });
        return;
    }
    if x.is_infinite() {
        out.push_str(if x < 0.0 { "-inf" } else { "inf" });
        return;
    }
    if x == 0.0 {
        out.push_str(if x.is_sign_negative() { "-0" } else { "0" });
        return;
    }
    // Round to 6 significant digits (Rust's exact formatting rounds the exact binary value,
    // ties to even, as glibc does), then lay the digits out as %g would.
    let mut e = String::with_capacity(16);
    write!(e, "{:.5e}", x.abs()).unwrap();
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: Vec<u8> = mant.bytes().filter(|&c| c != b'.').collect();
    if x < 0.0 {
        out.push('-');
    }
    let push_frac = |out: &mut String, frac: &[u8]| {
        let len = frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1);
        if len > 0 {
            out.push('.');
            out.extend(frac[..len].iter().map(|&c| c as char));
        }
    };
    if (-4..6).contains(&exp) {
        if exp >= 0 {
            let k = exp as usize + 1;
            out.extend(digits[..k].iter().map(|&c| c as char));
            push_frac(out, &digits[k..]);
        } else {
            out.push('0');
            let mut frac = vec![b'0'; (-exp - 1) as usize];
            frac.extend_from_slice(&digits);
            push_frac(out, &frac);
        }
    } else {
        out.push(digits[0] as char);
        push_frac(out, &digits[1..]);
        write!(
            out,
            "e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
        .unwrap();
    }
}

/// `x` formatted as `printf("%.6g")`.
pub fn fmt_g(x: f64) -> String {
    let mut s = String::new();
    write_g(&mut s, x);
    s
}

/// A value formatted with [`write_g`], for use in `write!`/`format!`.
pub struct G(pub f64);

impl std::fmt::Display for G {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        write_g(&mut s, self.0);
        f.write_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" {
        fn snprintf(buf: *mut u8, n: usize, fmt: *const u8, ...) -> i32;
    }

    fn c_g(x: f64) -> String {
        let mut buf = [0u8; 64];
        // SAFETY: the buffer is large enough for any %.6g output and NUL-terminated by snprintf.
        let n = unsafe { snprintf(buf.as_mut_ptr(), buf.len(), c"%.6g".as_ptr().cast(), x) };
        String::from_utf8(buf[..n as usize].to_vec()).unwrap()
    }

    #[test]
    fn g_special_values() {
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(-0.0), "-0");
        assert_eq!(fmt_g(f64::INFINITY), "inf");
        assert_eq!(fmt_g(f64::NEG_INFINITY), "-inf");
        assert_eq!(fmt_g(f64::NAN), "nan");
        assert_eq!(fmt_g(-f64::NAN), "-nan");
        assert_eq!(fmt_g(1234565.0), "1.23456e+06");
        assert_eq!(fmt_g(0.08), "0.08");
        assert_eq!(fmt_g(4.0), "4");
        assert_eq!(fmt_g(-585743.0), "-585743");
        assert_eq!(fmt_g(1e9), "1e+09");
        assert_eq!(fmt_g(1e-5), "1e-05");
        assert_eq!(fmt_g(1e100), "1e+100");
        assert_eq!(fmt_g(5e-324), "4.94066e-324");
    }

    #[test]
    fn g_matches_glibc_printf() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut values = Vec::new();
        for _ in 0..200_000 {
            values.push(f64::from_bits(next())); // any bit pattern
        }
        for _ in 0..100_000 {
            let r = next();
            let mant = (r >> 11) as f64 / (1u64 << 53) as f64;
            let e = (r % 40) as i32 - 20;
            values.push((mant - 0.5) * 10f64.powi(e));
        }
        // Integers and exact halves around the 6-digit rounding boundary.
        for k in 0..20_000i64 {
            values.push((k * 10 + 5) as f64 * 100.0 + 5.0);
            values.push(1_000_000.0 + k as f64 * 0.5);
            values.push(k as f64 / 1024.0);
            values.push(99_999.5 + k as f64 / 8.0);
        }
        for p in -30..30 {
            let b = 10f64.powi(p);
            for v in [b, b * 0.999_999_5, b * 0.999_995, b * 1.000_005] {
                values.push(v);
                values.push(-v);
            }
        }
        for &x in &values {
            assert_eq!(fmt_g(x), c_g(x), "value {x:e} bits {:016x}", x.to_bits());
        }
    }

    #[test]
    fn trunc_matches_x86() {
        assert_eq!(trunc_to_u32(3.99), 3);
        assert_eq!(trunc_to_u32(4_000_000_000.5), 4_000_000_000);
        assert_eq!(trunc_to_u32(f64::INFINITY), 0);
        assert_eq!(trunc_to_u32(f64::NAN), 0);
        assert_eq!(trunc_to_u32(-1.5), u32::MAX);
    }
}
