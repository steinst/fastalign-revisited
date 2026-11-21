//! Forced alignment of one sentence pair with a fixed translation table (spec §16).
//!
//! The arithmetic is that of fast_align's `-f` mode, operation for operation, so a table
//! exported with `-p` (or a saved model) reproduces fast_align's own alignments and scores.

use crate::corpus::NULL_ID;
use crate::math;
use crate::symmetrize::Link;

/// The options that shape the alignment of a sentence pair with a fixed table.
#[derive(Clone, Copy, Debug)]
pub struct AlignOptions {
    /// The table conditions on the second corpus side (fast_align's `-r`).
    pub reverse: bool,
    /// Diagonal prior (`-d`) with `tension` and `p0`; otherwise uniform over source words.
    pub favor_diagonal: bool,
    /// NULL word (off with `-N`).
    pub use_null: bool,
    pub p0: f64,
    pub tension: f64,
    /// Expected target length per source word (`-m`), used by the length score only.
    pub mean_srclen_multiplier: f64,
    /// Without NULL, start the argmax as training's Viterbi decoding does (spec §8: any value
    /// `> −1`, so a column of zeros links to the first source word) instead of as fast_align's
    /// `-f` mode (§16: only values `> 0` link). Saved-model inference sets it so that it
    /// reproduces the training run; `-f` keeps fast_align's rule.
    pub training_argmax: bool,
}

/// Align one sentence pair. `first` and `second` are the two sides of the pair (word ids, as the
/// table's `prob(e, f)` understands them); `prob(e, f)` is the probability of target word `f`
/// given source word `e` (`NULL_ID` for the NULL word) and must return fast_align's `1e-9` for
/// pairs the table does not know.
///
/// Appends the links `(first index, second index)` to `links` in increasing target position
/// (not sorted by source index) and returns the log score, or `None` (and no links) if a side
/// is empty.
pub fn align_pair<P: Fn(u32, u32) -> f64>(
    prob: &P,
    first: &[u32],
    second: &[u32],
    opts: &AlignOptions,
    links: &mut Vec<Link>,
) -> Option<f64> {
    let (src, trg) = if opts.reverse {
        (second, first)
    } else {
        (first, second)
    };
    if src.is_empty() || trg.is_empty() {
        return None;
    }
    let pnn = 1.0 - opts.p0;
    let lambda = opts.tension;
    let (n, m) = (src.len(), trg.len());
    let (n32, m32) = (n as u32, m as u32);
    let mut lp = math::log_poisson(m32, 0.05 + n as f64 * opts.mean_srclen_multiplier);
    let uniform = 1.0 / (n + usize::from(opts.use_null)) as f64;
    for (j, &f) in trg.iter().enumerate() {
        let j32 = j as u32;
        let mut sum = 0.0f64;
        let mut best = 0usize;
        let mut best_p = if opts.training_argmax { -1.0 } else { 0.0f64 };
        let mut a = uniform;
        if opts.use_null {
            if opts.favor_diagonal {
                a = opts.p0;
            }
            best_p = prob(NULL_ID, f) * a;
            sum += best_p;
        }
        let mut az = 0.0;
        if opts.favor_diagonal {
            az = math::compute_z(j32 + 1, m32, n32, lambda) / pnn;
        }
        for i in 1..=n {
            if opts.favor_diagonal {
                a = math::unnormalized_prob(j32 + 1, i as u32, m32, n32, lambda) / az;
            }
            let p = prob(src[i - 1], f) * a;
            if p > best_p {
                best_p = p;
                best = i;
            }
            sum += p;
        }
        lp += sum.ln();
        if best > 0 {
            let s = (best - 1) as u32;
            links.push(if opts.reverse { (j32, s) } else { (s, j32) });
        }
    }
    Some(lp)
}
