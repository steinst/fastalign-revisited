# Behavioural specification of fast_align

Written during the project. This document describes what `fast_align` (github.com/clab/fast_align,
commit `cab1e9a`) and its `atools` compute, precisely enough to reproduce their output bit for bit
without reading the C++. It is written from the source (`reference/fast_align-src/src/`), the paper
(Dyer, Chahuneau and Smith 2013, `reference/paper/dyer2013.pdf`) and the libstdc++ headers, and
then checked by experiment (§19).

Where the paper and the source disagree, this document follows the source and lists the
difference in §17. "Parity" always means `OMP_NUM_THREADS=1 fast_align` (§18).

Notation. Indices are as in the code: for a sentence pair after the optional `-r` swap, `src` has
length `n` (source positions `i = 1..n`, position 0 is NULL) and `trg` has length `m` (target
positions `j = 0..m-1`). All real arithmetic is IEEE-754 binary64 (`f64`), round-to-nearest-even,
no fused multiply-add, evaluated strictly in the order written; `a ⊕ b`, `a ⊗ b`, `a ⊘ b` stress
that a single rounded operation is meant where order matters. "Sum of x over S" always means a
left-to-right running sum that starts at `+0.0` and adds the terms in the stated order.
`u32`, `i16` and so on are the usual fixed-width integer types; casts are spelled out.

Contents

1. Reference build and numerical environment
2. Command line
3. Reading the input
4. Dictionary
5. Translation table: keys, values and the two tables
6. Initial pass
7. Iteration structure
8. E-step for one sentence pair
9. Reductions and their grouping
10. Statistics, tension optimisation
11. M-step
12. Diagonal prior functions
13. digamma and log_poisson
14. libstdc++ `unordered_map` iteration order
15. Output formats and number formatting
16. Forced alignment (`-f`)
17. `atools`
18. Differences from the paper; multi-threaded fast_align
19. Edge cases and undefined behaviour
20. Verification record, instrumented build, probe
21. Notes for the implementation (Steps 4–6)

---

## 1. Reference build and numerical environment

| Item | Value |
|---|---|
| Source | fast_align commit `cab1e9a`, `reference/fast_align-src` (symlink to gizapp's copy) |
| Compiler | GCC 13.3.0 (Ubuntu 24.04), C++11 |
| Flags | `-Wall -std=c++11 -O3 -g -fopenmp -O3 -DNDEBUG` (CMake `Release`); no `-march`, so baseline x86-64 SSE2 |
| C++ runtime | `libstdc++.so.6.0.33` from the `libstdc++6 14.2.0` package; headers from GCC 13 (`/usr/include/c++/13`) |
| libm | glibc 2.39 |
| Hash maps | `std::unordered_map` (sparsehash was not found, so `HAVE_SPARSEHASH` is undefined) |

Checks on the binary (`objdump -d`):

- **No FMA**: no `vfmadd`/`vfmsub`/`vfnmadd` instruction in `fast_align` or `atools`. Without
  `-ffast-math` GCC neither contracts nor reassociates; every expression is evaluated as written.
  SSE2 arithmetic, so no x87 excess precision (`FLT_EVAL_METHOD == 0`).
- **No vector libm** (`_ZGV*` symbols are absent): glibc's SIMD math variants are only declared
  under `__FAST_MATH__`.
- **libm calls** that remain: `exp` (16 call sites), `log` (9), `pow` (9), `lgamma` (2). `fabs`
  is inlined (sign-bit clear, exact). `pow(2.0, x)` is **not** turned into `exp2`. `log(2)` in the
  statistics is folded by GCC to the correctly rounded constant `ln 2 = 0x3FE62E42FEFA39EF`, which
  is also what glibc's `log(2.0)` returns.
- Integer-to-double conversions are exact for all values that occur (< 2^53).
- `pow(double, unsigned)`: in C++11 the integer exponent is converted to `double` and the
  ordinary `pow(double, double)` is called. No repeated multiplication.
- `lgamma(unsigned)`: the argument is converted to `double`; glibc `lgamma`.

A reimplementation must therefore call glibc's `exp`, `log`, `pow`, `lgamma` with the same
arguments (in Rust: `f64::exp`, `f64::ln`, `f64::powf`, and `libc::lgamma`; beware of LLVM
constant-folding a call with constant arguments, which uses its own evaluation, and of
`f64::powi`, which is *not* `pow`).

## 2. Command line

Parsed with GNU `getopt_long`, short option string `i:rI:df:m:t:q:T:ova:Np:b:s`.

| Flag | Long name | Effect | Default |
|---|---|---|---|
| `-i FILE` | `--input` | input corpus (required; empty value ⇒ usage error) | — |
| `-r` | `--reverse`* | swap source and target of every line after parsing | off |
| `-I N` | `--iterations` | number of iterations, `atoi` | 5 |
| `-d` | `--favor_diagonal`* | use the diagonal alignment prior | off |
| `-f FILE` | `--force_align` | forced-alignment mode, read table FILE (§16) | off |
| `-m X` | `--mean_srclen_multiplier` | `atof`; **only used in `-f` mode** (training overwrites it, §6) | 1.0 |
| `-t X` | `--beam_threshold` | `atof`; only affects the `-p` export | −4.0 |
| `-q X` | `--p0` | `atof`; NULL alignment probability `p0` | 0.08 |
| `-T X` | `--diagonal_tension` | `atof`; starting tension λ, **and sets `-d`** | 4.0 |
| `-o` | `--optimize_tension`* | optimise λ (only together with `-d`) | off |
| `-v` | `--variational_bayes`* | VB M-step with Dirichlet prior | off |
| `-a X` | `--alpha` | `atof`; Dirichlet α (only with `-v`) | 0.01 |
| `-N` | `--no_null_word`* | no NULL word | off |
| `-p FILE` | `--conditional_probabilities` | write the table to FILE after training | none |
| `-b N` | `--thread_buffer_size` | `atoi`, stored in an unsigned 64-bit size | 10000 |
| `-s` | — | append the alignment score to each output line | off |

Behaviour of the parser:

- Each option returned by `getopt_long` is echoed on stderr as `ARG=<c>` followed by a newline, in
  command-line order, before anything else. Unknown options make getopt print its own message,
  then `ARG=?`, then the usage text.
- The long options marked * are "flag" options: getopt sets the flag and returns 0, which is echoed
  as `ARG=` followed by a NUL byte, and then falls into the usage error (exit 1). So `--reverse`
  and friends **do not work**; the short flags do. Long options with arguments work, and getopt
  accepts unique prefixes of long names, `--name=value`, clustered short flags (`-dov`) and
  attached values (`-I5`). Non-option arguments are permuted to the end and ignored.
- `atoi`/`atof` semantics: leading whitespace skipped, longest valid prefix parsed, 0 if none, no
  error. `-b` with a negative value becomes a huge unsigned number.
- Usage error (no `-i`, or any parse failure): usage text on stderr, exit status 1. The usage text
  mentions `-c` for the table output, which does not exist; the real flag is `-p`.
- After parsing: if `-v` is given and `α ≤ 0`, print `--alpha must be > 0` and exit 1.
- `-f FILE` sets force mode and the table filename; `-p FILE` sets the same filename variable. If
  both are given, the **last one wins for the filename**, and force mode stays on if `-f`
  appeared at all.
- Interactions that make an option inert: `-o` without `-d`/`-T` does nothing; `-q` without
  `-d` does nothing in training (the uniform prior is used for NULL as well); `-a` without `-v`
  does nothing; `-m` does nothing in training; `-t` does nothing without `-p`; `-I` and `-p` are
  ignored in force mode.

## 3. Reading the input

**Lines.** The file is read with `getline` on `\n`: every `\n`-terminated piece is a line, and a
final piece without `\n` is a line too if it is non-empty. An empty file has no lines. `\r` is an
ordinary byte (so CRLF files keep a `\r` at the end of the last target token). Bytes are not
decoded; any byte other than space and tab is a token byte.

**Tokenisation of one line** produces a sequence of ids:

- Only space (0x20) and tab (0x09) separate tokens. Runs of separators produce no empty tokens.
- **A tab additionally emits the separator id** `kDIV` at its position (after the token it
  terminates, if any). So a tab acts exactly like a ` ||| ` token.
- Every token is converted to an id with the dictionary (§4). The literal token `|||` has id
  `kDIV` too.

**Splitting into sides.** `src` = the ids before the first `kDIV`; `trg` = **all** ids after the
first `kDIV`, including any later `kDIV` (so `a ||| b ||| c` gives `src = [a]`,
`trg = [b, |||, c]`; the word `|||` is then an ordinary target word). If there is no `kDIV`,
`trg` is empty.

**`-r`** swaps `src` and `trg` after parsing, everywhere (initial pass and every iteration). The
dictionary ids are not affected (tokens are always converted in line order).

**Empty sides.** If `src` or `trg` is empty the line is still used (counted in the line count,
the length statistics and the E-step). A message is printed on stderr:
`Error in line <N>` newline `<raw line bytes>` newline. In the initial pass `N` is the true
1-based line number. **In the iterations `N` is the number of the last line of the current
buffer** (§7), the value of the line counter when the buffer is processed, and the message is
printed for every iteration, including the final one. The consequences of empty sides on the
statistics are in §19.

## 4. Dictionary

One dictionary for both languages, mapping byte strings to `u32` ids. Ids are assigned
consecutively from 1 in order of first `Convert` call; id 0 is reserved and prints as `<bad0>`.

Order of first conversions in training mode:

1. `<eps>` → id 1. This is `kNULL`, the NULL word (also when `-N` is given).
2. `|||` → id 2 (`kDIV`), converted when the first line is parsed, before its tokens.
3. Every token of every line, left to right (source side then target side, since that is the
   line order; `-r` does not change it), in corpus order, during the initial pass. The
   iterations parse the lines again but meet no new words.

A corpus token `<eps>` therefore *is* the NULL word (id 1), and a corpus token `|||` is the
separator. In force mode the order is different (§16).

## 5. Translation table: keys, values and the two tables

The model keeps two tables with **the same key set**: `ttable` (probabilities) and `counts`
(expected counts). Each is a vector indexed by source id `e` (rows) of hash maps from target id
`f` to `f64`. Row `e = kNULL` holds the NULL word's distribution.

- Keys are created only in the initial pass (§6), by inserting into `counts`; values start at 0.
- **Freeze** (after the initial pass): `ttable` is resized to the number of rows of `counts`, and
  each `ttable[e]` is copy-assigned from `counts[e]`. All values are 0. A copy has the same
  bucket count and the same iteration order as the original (§14.5). No key is ever added later.
- `prob(e, f)` (used in training): returns `1e-9` for every pair until the first M-step has run
  ("probabilities not initialised"), and the stored `ttable[e][f]` afterwards. The pair always
  exists.
- `Increment(e, f, x)`: `counts[e][f] ← counts[e][f] ⊕ x`.
- The M-step (§11) swaps the two row vectors, normalises the new `ttable` in place and sets every
  value of the new `counts` to 0. Since every row's two maps have the same order, the swap never
  changes any iteration order.

The number of rows is `max(e)+1` over source ids that ever reached a flush (§6), and at least
`kNULL+1 = 2` when the NULL word is used. Rows of ids that never occur as source words are empty.

## 6. Initial pass

Reads the whole corpus once. Prints `INITIAL PASS ` (with a trailing space) and a newline first.
State: `tot_len_ratio = 0.0`, `n_target_tokens = 0.0` (f64), line counter `lc = 0`, an *insert
buffer* (per source id, a list of target ids) with an item counter `items = 0`, and the
`size_counts` map from `(i16, i16)` to `u32` (§14.6).

For each line:

1. `lc += 1`; progress output (§15.4).
2. Parse, swap if `-r`, print the empty-side error if needed (§3).
3. `tot_len_ratio ← tot_len_ratio ⊕ (f64(m) ⊘ f64(n))`.
4. `n_target_tokens ← n_target_tokens ⊕ f64(m)` (exact).
5. If the NULL word is used: for each `f` in `trg` in order, insert `f` into `counts[kNULL]`
   **directly**.
6. For each `e` in `src` in order: append all of `trg` (in order) to the insert buffer of `e`,
   and `items += m`.
7. If `items > 100 × b` (unsigned 64-bit, `b` = `-b`): flush the insert buffer: for each source id
   `e` in the buffer, insert its list into `counts[e]` in list order, then empty the list; set
   `items = 0`.
8. `size_counts[(i16(m), i16(n))] += 1` (inserting the key with count 0 first if new). The
   conversion from length to `i16` truncates modulo 2^16 (relevant only for lengths ≥ 32768).

After the last line: copy `size_counts` into a vector **in the map's iteration order**
(§14.6), flush the insert buffer once more, then
`mean_srclen_multiplier ← tot_len_ratio ⊘ f64(lc)` (this **overwrites** `-m`), print a newline if
the progress flag is set (§15.4), and print
`expected target length = source length * <mean_srclen_multiplier>` (§15.1 formatting).
Then Freeze (§5).

"Insert `f` into `counts[e]`" means: if `f` is already a key, nothing happens (no rehash, value
unchanged); otherwise a new key with value 0 is added (§14). Consequently:

- **The key-insertion sequence of a row `e ≠ kNULL` is independent of `-b`**: it is the
  first-occurrence order of target ids over the sequence (line, occurrence of `e` in `src`,
  target position), whatever the flush points. Flushes only change *when* the inserts happen.
- The NULL row receives the target ids in corpus order (line, target position).
- If `<eps>` occurs as a *source* token, the NULL row also receives buffered inserts, but they
  never add a key when the NULL word is used: each buffered key is a target word of a line whose
  target words were all inserted directly when that line was read, before any flush. Without the
  NULL word (`-N`), row 1 receives only the buffered inserts and is an ordinary row. So **no row's
  key order depends on `-b`**. (Corrected in Step 4; checked against fast_align with `<eps>` as a
  source word and `-b 0`, `-b 1`, `-b 10000`, with and without `-N`.)
- Flushing calls "ensure `counts` has at least `size(buffer)` rows" before inserting
  (see §19 for the case of an empty buffer).

## 7. Iteration structure

For `iter = 0 .. I-1` (`I` from `-I`; no iterations if `I ≤ 0`):

1. `final = (iter == I-1)`. Print `ITERATION <iter+1>` or `ITERATION <iter+1> (FINAL)`.
2. Re-read the corpus. Accumulators `likelihood = c0 = emp_feat = 0.0`. Lines are collected into
   a buffer; whenever the buffer holds `≥ b` lines (`b` = `-b`, unsigned, so `b = 0` behaves like
   1) it is processed (§8, §9) and cleared; after the last line a non-empty remainder is
   processed. In the final iteration the output lines of each buffer are written to stdout in
   line order after the buffer is processed.
3. After the corpus: print statistics (§10.1). If not `final`: tension optimisation if enabled
   (§10.2), then the M-step (§11). The final iteration has **no M-step** and its E-step
   **accumulates no counts and no `c0`/`emp_feat`** (they print as 0); it only decodes and sums
   the likelihood.

So with the default `-I 5`, iterations 1–4 do E-step + M-step and iteration 5 decodes with the
parameters from the fourth M-step. Iteration 1 uses `prob = 1e-9` everywhere (model 1 / pure
prior). With `-I 1` the only iteration is final: decoding with `prob = 1e-9`, i.e. by the prior
alone.

After the loop, if not in force mode and a table filename is set: print
`conditional probabilities: <FILE>` on stderr and export `ttable` (§15.3).

The tension λ used by the E-step of iteration `k` is the value after the last optimisation
(initially `-T`, default 4.0). Optimisation happens only when `-d` and `-o` and `iter > 0` and not
final: with `-I 5`, after the E-steps of iterations 2, 3 and 4.

## 8. E-step for one sentence pair

Inputs: `src` (length `n`), `trg` (length `m`), λ, `p0` (`-q`), `pnn = 1.0 − p0` (computed once),
`use_null = !(-N)`, `diag = -d`. Let `u = f64(n + use_null)` (with `use_null` as 0/1).

```
ll = 0.0                                   # this line's log likelihood
for j in 0 .. m-1:
    f = trg[j]
    probs[0..n] (scratch), sum = 0.0
    a = 1.0 ⊘ u                            # uniform prior, also used for NULL without -d
    if use_null:
        if diag: a = p0
        probs[0] = prob(kNULL, f) ⊗ a
        sum = sum ⊕ probs[0]
    if diag:
        az = ComputeZ(j+1, m, n, λ) ⊘ pnn  # §12; note j+1 (1-based target position)
    for i in 1 .. n:
        if diag:
            a = UnnormalizedProb(j+1, i, m, n, λ) ⊘ az
        probs[i] = prob(src[i-1], f) ⊗ a
        sum = sum ⊕ probs[i]
    if final:
        Viterbi choice (below)
    else:
        if use_null:
            c = probs[0] ⊘ sum
            c0_buf = c0_buf ⊕ c
            Increment(kNULL, f, c)
        for i in 1 .. n:
            p = probs[i] ⊘ sum
            Increment(src[i-1], f, p)
            emp_feat_buf = emp_feat_buf ⊕ (Feature(j, i, m, n) ⊗ p)   # j, NOT j+1 (§17)
    ll = ll ⊕ log(sum)
likelihood_buf = likelihood_buf ⊕ ll
```

Notes:

- `ComputeZ` is evaluated once per target position, `UnnormalizedProb` once per cell; their
  values depend only on `(j, i, m, n, λ)`. `emp_feat` is accumulated even without `-d`.
- `Feature(j, i, m, n) = −|f64(i) ⊘ f64(n) − f64(j) ⊘ f64(m)|` with the 0-based `j` (§12).
- Divisions are real divisions (`probs[i] ⊘ sum`), not multiplications by a reciprocal.
- **Per-cell count order**: over the whole corpus, `counts[e][f]` receives its additions in the
  order (line, `j`, `i`), the NULL addition for a given `(line, j)` coming before the source ones.
  A word repeated in `src` or `trg` adds several times to the same cell, in that order.
- **Viterbi** (final iteration): start with `best = 0, best_p = probs[0]` if `use_null`, else
  `best = −1, best_p = −1.0`; for `i = 1..n` in order, if `probs[i] > best_p` (strict) then
  `best = i, best_p = probs[i]`. Ties go to the earliest position, NULL first. If `best > 0`,
  emit the link `(best−1)-j`, or `j-(best−1)` with `-r` (so the link is always written in the
  original source-target orientation of the file). A comparison with NaN is false.
- **Score** (`-s`, final iteration): `lp = log_poisson(m, 0.05 ⊕ (f64(n) ⊗ mean_srclen_multiplier)) ⊕ ll`
  with `log_poisson` from §13 and `mean_srclen_multiplier` from the initial pass.

## 9. Reductions and their grouping

At one thread, the OpenMP `reduction(+)` of a buffer gives each buffer its own three partial sums
starting at `+0.0`; at the end of the buffer the shared accumulator gets `0.0 ⊕ partial` (exactly
`partial`; a `−0.0` cannot arise because the partials start at `+0.0`), and then the totals:

- `likelihood_buf` = sum over the buffer's lines (in order) of the line's `ll`; each `ll` is the
  sum over `j` of `log(sum_j)`.
- `c0_buf` = sum over (line, `j`) of the NULL posterior.
- `emp_feat_buf` = sum over (line, `j`, `i`) of `Feature ⊗ p`.
- After each buffer: `likelihood ← likelihood ⊕ likelihood_buf`, `c0 ← c0 ⊕ c0_buf`,
  `emp_feat ← emp_feat ⊕ emp_feat_buf`, buffers in corpus order.

So **`-b` changes the floating-point grouping** of these three sums (and the line numbers in
error messages) and nothing else. Other sums:

| Sum | Order |
|---|---|
| `tot_len_ratio` | lines in order, one term per line |
| `counts[e][f]` | (line, `j`, `i`), §8; independent of `-b` |
| row total in the M-step | the row's hash iteration order (§14), one row at a time |
| `mod_feat` (tension) | `size_counts` vector order (§14.6), then `j = 1..m` |

## 10. Statistics, tension optimisation

### 10.1 Statistics (every iteration, stderr)

After the corpus pass: if the progress flag is set (§15.4) print a newline. Then, with
`T = n_target_tokens`, `L = likelihood`, `L2 = L ⊘ ln2` (constant `0x3FE62E42FEFA39EF`), and
`emp = emp_feat ⊘ T` (this normalised value is also what the tension optimisation uses):

```
  log_e likelihood: <L>
  log_2 likelihood: <L2>
     cross entropy: <(−L2) ⊘ T>
        perplexity: <pow(2.0, (−L2) ⊘ T)>
      posterior p0: <c0 ⊘ T>
 posterior al-feat: <emp>
       size counts: <number of distinct (m, n) pairs, integer>
```

(Labels are right-aligned exactly as shown; numbers per §15.1.)

### 10.2 Tension optimisation

Only if `-d`, `-o`, `iter > 0` and not final. Eight steps, `ii = 1..8`:

```
mod = 0.0
for ((m, n), count) in size_counts (vector order, §14.6):
    for j in 1 .. m:              # m as i16; no terms if m ≤ 0
        mod = mod ⊕ (f64(count) ⊗ ComputeDLogZ(j, m, n, λ))   # m, n converted to u32
mod = mod ⊘ T
print "  <ii>  model al-feat: <mod> (tension=<λ>)"
λ = λ ⊕ ((emp − mod) ⊗ 20.0)
if λ ≤ 0.1: λ = 0.1
if λ > 14:  λ = 14.0
```

then print `     final tension: <λ>`. The OpenMP reduction over `size_counts` at one thread is
`0.0 ⊕ (left-to-right sum)`, i.e. the plain sum.

## 11. M-step

Row by row (rows are independent; inside a row the order matters only for the total). For row
`e` with values `c_k` (the counts) visited in the row's hash iteration order (§14):

- **Normalize** (no `-v`): `tot` = sum of `c_k`; if `tot == 0` then `tot = 1.0`;
  each value becomes `c_k ⊘ tot`.
- **NormalizeVB** (`-v`): `tot` = sum of `(c_k ⊕ α)` (each term rounded, then added); if
  `tot == 0` then `tot = 1.0`; `dt = digamma(tot)`; each value becomes
  `exp(digamma(c_k ⊕ α) − dt)`.

The results become the probabilities (`ttable`), all counts are reset to 0, and `prob()` now
returns stored values (§5). Empty rows are skipped harmlessly.

## 12. Diagonal prior functions

Arguments: `i` target position (1-based), `j` source position, `m` target length, `n` source
length, all `u32`; `λ` an `f64`. Integer-to-`f64` conversions are written `f64(·)`.

**Feature**(i, j, m, n) = `−|f64(j) ⊘ f64(n) − f64(i) ⊘ f64(m)|` (subtract, then clear the sign
bit, then negate).

**UnnormalizedProb**(i, j, m, n, λ) = `exp(Feature(i, j, m, n) ⊗ λ)`.

**ComputeZ**(i, m, n, λ):

```
split   = (f64(i) ⊗ f64(n)) ⊘ f64(m)
fl      = u32(trunc(split))               # "floor"
ce      = fl + 1   (u32)
ratio   = exp((−λ) ⊘ f64(n))
num_top = n − fl   (u32, wrapping)
ezt = 0.0; ezb = 0.0
if num_top ≠ 0: ezt = (UnnormalizedProb(i, ce, m, n, λ) ⊗ (1.0 − pow(ratio, f64(num_top)))) ⊘ (1.0 − ratio)
if fl ≠ 0:      ezb = (UnnormalizedProb(i, fl, m, n, λ) ⊗ (1.0 − pow(ratio, f64(fl)))) ⊘ (1.0 − ratio)
return ezb ⊕ ezt
```

**ComputeDLogZ**(i, m, n, λ):

```
z       = ComputeZ(i, n, m, λ)          # m and n SWAPPED (fast_align bug, reproduce it)
split   = (f64(i) ⊗ f64(n)) ⊘ f64(m)
fl      = u32(trunc(split)); ce = fl + 1
ratio   = exp((−λ) ⊘ f64(n))
d       = −1.0 ⊘ f64(n)
num_top = n − fl  (u32, wrapping)
pct = 0.0; pcb = 0.0
if num_top ≠ 0: pct = AGS(Feature(i, ce, m, n), UnnormalizedProb(i, ce, m, n, λ), ratio, d, num_top)
if fl ≠ 0:      pcb = AGS(Feature(i, fl, m, n), UnnormalizedProb(i, fl, m, n, λ), ratio, d, fl)
return (pct ⊕ pcb) ⊘ z
```

Because of the swap, the inner `ComputeZ(i, n, m)` computes `split = i·m/n`, which can exceed
its "n" (= `m`): then `num_top = m − fl` **wraps around** as `u32` (e.g. 2^32 − 3), its `f64`
value is used in `pow` (giving 0 for `ratio < 1`) and the `num_top ≠ 0` branch is taken. This
happens in practice (whenever `i·m/n > m`, i.e. for target lengths greater than source lengths);
a reimplementation must use wrapping `u32` subtraction here.

**AGS**(a1, g1, r, d, k) (arithmetico-geometric series; `k` is `u32`):

```
g_np1 = g1 ⊗ pow(r, f64(k))
a_n   = (d ⊗ f64(k − 1)) ⊕ a1          # k − 1 in u32 (k ≥ 1 here)
x_1   = a1 ⊗ g1
g_2   = g1 ⊗ r
rm1   = r − 1.0
return ((a_n ⊗ g_np1 − x_1) ⊘ rm1) − ((d ⊗ (g_np1 − g_2)) ⊘ (rm1 ⊗ rm1))
```

Operation order inside each line is left to right as written (`a ⊗ b − c` means
`(a ⊗ b) − c`).

## 13. digamma and log_poisson

**digamma**(x):

```
r = 0.0
while x < 7.0:  r = r − (1.0 ⊘ x);  x = x ⊕ 1.0
x  = x − 0.5
xx = 1.0 ⊘ x;  xx2 = xx ⊗ xx;  xx4 = xx2 ⊗ xx2
t  = log(x) ⊕ (C1 ⊗ xx2)
t  = t − (C2 ⊗ xx4)
t  = t ⊕ ((C3 ⊗ xx4) ⊗ xx2)
t  = t − ((C4 ⊗ xx4) ⊗ xx4)
return r ⊕ t
```

with the constants `C1 = 1/24`, `C2 = 7/960`, `C3 = 31/8064`, `C4 = 127/30720`, each the
correctly rounded `f64` of the quotient (as a compiler folds `1.0/24.0`). Note the full
correction term `t` is computed first and then added to `r` once.

**log_poisson**(x: u32, λ: f64) = `(log(λ) ⊗ f64(x)) − lgamma(f64(x + 1)) − λ`, left to right
(`x + 1` in `u32`). No check that λ > 0 (the `assert` is compiled out).

## 14. libstdc++ `unordered_map` iteration order

The iteration order of a row decides the summation order of its M-step total (§11) and the line
order of the `-p` table (§15.3); the iteration order of `size_counts` decides the summation order
of the tension gradient (§10.2). This section describes the algorithm of libstdc++ (GCC 13
headers, `bits/hashtable.h`, `bits/hashtable_policy.h`; rehash policy and prime table from the
`libstdc++.so.6` runtime) for a map with unique keys and the default `max_load_factor` of 1.0.
It is verified against `benchmarks/stlhash_probe.cc` (§20).

### 14.1 Structure

A map is a singly linked list of nodes (the iteration order is list order, starting at the
head), plus an array of `B` buckets. The nodes of one bucket are always contiguous in the list.
For each non-empty bucket the array stores its *before* node: the node immediately preceding the
bucket's first node, or the list head sentinel `H` if the bucket's first node is the list head.
The bucket of a key is `hash(key) mod B` (64-bit unsigned).

State of a new (default-constructed) map: `B = 1`, no nodes, element count `N = 0`,
`next_resize R = 0`.

### 14.2 Hash functions

- Rows of the translation table (`unordered_map<unsigned, double>`): `hash(k) = k`
  (identity, zero-extended to 64 bits).
- `size_counts` (`unordered_map<pair<short,short>, unsigned, PairHash>` with key
  `(a, b) = (i16(m), i16(n))`): `hash = zero_extend_64( (u32(u16(a)) << 16) | u32(i32(b)) )`,
  computed in 32 bits: `b` is sign-extended to 32 bits before the OR (so a negative `b`
  sets the upper 16 bits too); for ordinary lengths `0 ≤ m, n < 32768` this is `m·65536 + n`.
  (libstdc++ caches this hash in the node because `PairHash` is not `noexcept`; that does not
  affect the order.)

### 14.3 Insertion of a new key `k`

Inserting a key that is already present does nothing at all. For a new key:

1. **Rehash check.** If `N + 1 > R`:
   - `need = max(N + 1, (R == 0) ? 11 : 0)`;
   - if `need ≥ B`: rehash (§14.4) to `NextBkt(max(need + 1, 2·B))`;
   - else: `R = B` (no rehash).
2. `b = hash(k) mod B` (with the possibly new `B`).
3. **Link the node.**
   - If bucket `b` is non-empty: insert `k` immediately after `before[b]`, i.e. as the **first
     node of bucket `b`'s run**.
   - If bucket `b` is empty: insert `k` at the **head of the list**; if there was a previous head
     node `h`, set `before[bucket(h)] = k`; set `before[b] = H`.
4. `N += 1`.

**NextBkt(x)** (also sets `R`):

- if `x < 14`: if `x == 0` return 1 (R unchanged); else return `FAST[x]` and set `R = FAST[x]`,
  with `FAST = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13]`;
- else: let `P[0..304]` be libstdc++'s prime table (`std::__detail::__prime_list`, 305 entries on
  LP64: 304 primes ascending plus a repeated last prime as sentinel; saved as
  `crates/fastalign-core/tests/data/stlhash/primes.txt`). Return the smallest `P[t] ≥ x` with
  `6 ≤ t < 303`; if there is none return `P[303]` and set `R = 2^64 − 1`; otherwise set
  `R = P[t]` (`floor(P[t] × 1.0)` in `f64`; exact for every reachable size).

With these rules the bucket count of a map that grows one key at a time is 13 after the first
key, then 29 when the 14th key arrives, then 59 (30th), 127 (60th), 257 (128th), 541 (258th),
1109, 2357, 5087, 10273, 20753, 42043, 85229, 172933, 351061, 712697, 1447153, 2938679, 5967347…
(the rehash happens on inserting key number `B + 1`).

### 14.4 Rehash to `B'` buckets

Take the nodes in current list order; start a new empty list and an empty bucket array of size
`B'`. For each node `p` in that order, with `b = hash(p) mod B'`:

- if bucket `b` is empty in the new array: put `p` at the **head** of the new list; if the new
  list already had a head `q`, set `before[bucket(q)] = p`; set `before[b] = H`;
- else: insert `p` immediately after `before[b]` (as the first node of bucket `b`'s run).

(So buckets appear in the reverse order of their first visit, and within a bucket the nodes end
up in the reverse of the order in which they were visited.)

### 14.5 Copy-assignment and swap

- `copy = original` (as in Freeze, into a default-constructed map): the copy gets the same bucket
  count, the same rehash state `R`, and its nodes are created **in the original's iteration
  order**, each appended after the previous one. The copy's iteration order is identical.
- `vector<map>::swap` (as in the M-step) exchanges the vectors' buffers; no map is touched.
- Moving maps (when `counts` grows by `resize`) keeps nodes, buckets and order.

Hence every row of `ttable` and `counts` iterates in the order produced by §14.3 from that row's
key-insertion sequence in the initial pass (§6), for the whole run.

### 14.6 `size_counts`

Keys `(i16(m), i16(n))` are inserted in order of first occurrence over the corpus (line order),
with the hash of §14.2; the count is incremented on every line. At the end of the initial pass
the `(key, count)` pairs are copied into a vector in the map's iteration order, and that vector
order is used by the tension optimisation. The dictionary's `string → id` map is never iterated,
so its order is irrelevant.

## 15. Output formats and number formatting

### 15.1 Numbers

Every `double` written to stdout, stderr or the table uses the C++ stream default, which is
**`printf("%.6g")`** with glibc:

- at most 6 significant digits, correctly rounded from the exact binary value (ties — only
  possible when the binary value is exactly halfway — round to even, glibc's behaviour in the
  default rounding mode);
- fixed notation if the decimal exponent `X` satisfies `−4 ≤ X < 6`, else exponent notation
  `d.ddddde±XX` (at least two exponent digits: `1e+09`, `2.7464e+09`, `1e-05`);
- trailing zeros and a trailing decimal point removed (`0.08`, `4`, `-585743`);
- negative zero prints as `-0`;
- `inf`, `-inf`; NaN prints as `nan` or **`-nan` depending on its sign bit**. On x86-64 a NaN
  created by an invalid operation (`0/0`, `inf − inf`, `0 × inf`) has the sign bit set, so these
  print as `-nan` (seen in §19). A reimplementation must preserve the sign of NaN through
  arithmetic as the hardware does and must not substitute a canonical NaN (Rust's `f64::NAN` is
  positive, and LLVM folds constant `0.0/0.0` to a positive NaN).

Integers (`size counts`, line numbers, alignment indices, counts) print in plain decimal.

### 15.2 Alignments (stdout)

One line per input line, in input order, written after the final iteration's buffers. Links
`a-b` (source index, target index, 0-based, in the file's original orientation, §8), in
increasing target position, separated by single spaces; no link for target words aligned to NULL.
With `-s`: then ` ||| ` and the score (§8), even when there are no links (the line is then
` ||| <score>`). Each line ends with `\n`. A line with an empty side gives an empty alignment.

### 15.3 `-p` table

Written to FILE after training, from `ttable` (the probabilities of the last M-step; or the
all-zero values of Freeze if no M-step ran, i.e. `-I` ≤ 1). For each row `e = 0, 1, 2, …` up to the
number of rows − 1 (so in **id order**, NULL row first since `kNULL = 1`), with `e`'s word
(`<eps>` for the NULL row):

```
max_p = −1.0;  for each (f, p) in hash order: if p > max_p: max_p = p
thr = (−log(max_p)) ⊗ t            # t = -t option, default −4.0, so thr = 4·log(max_p)
for each (f, p) in hash order:
    c = log(p)
    if c ≥ thr: write  word(e) TAB word(f) TAB c(%.6g) LF
```

Entries appear in the row's hash iteration order (§14). Zero probabilities print `-inf` (and pass
the threshold, which is then `−inf`). The file is written even if empty rows produce no lines.

### 15.4 stderr

In training mode, in this order:

```
ARG=<c>                                  one per option (§2)
INITIAL PASS                             (trailing space)
[progress]  [Error in line … (§3)]
expected target length = source length * <mean>
then per iteration:
ITERATION <k>[ (FINAL)]
[progress]  [Error in line …]
<statistics §10.1>
[tension steps and final tension §10.2]
after the loop, with -p:
conditional probabilities: <FILE>
```

Progress (initial pass and every iteration): after reading line `lc`, if `lc % 1000 == 0` print
`.` and set the flag; if `lc % 50000 == 0` print ` [<lc>]` and a newline and clear the flag.
Before the statistics (and before the "expected target length" line) a newline is printed if
the flag is set. Error messages of the iterations are printed when the buffer is processed, i.e.
after the progress dot of the buffer's last line.

## 16. Forced alignment (`-f`)

**Table reading.** The dictionary first gets `<eps>` = 1. The table file is read as a stream of
whitespace-separated fields (whitespace = space, `\t`, `\n`, `\v`, `\f`, `\r`), three at a time:
`e`, `f`, `p` (a decimal parsed as `f64`, correctly rounded). For each record: `e` is converted
to an id first, then `f`; `ttable[e][f] = exp(p)` (overwriting duplicates). Then `|||` gets the
next id when the first input line is parsed, then input words. Quirk: the loop tests the stream
*before* reading, so after the last record one more iteration runs with the previous values
(re-storing the same entry); the message `Loaded <c> translation parameters.` therefore reports
**records + 1** (or 0 for an empty file). A field that does not parse as a number (e.g. `-inf`,
`nan` written by a `-I 1` training run) stops reading at that record; the record is stored with
`p = 0` (probability 1) and counted. If the file cannot be opened, nothing is read
(`Loaded 0 …`).

**Probabilities.** `safe_prob(e, f)` = the stored value if `e` is below the number of rows and
`f` is in row `e`, else `1e-9`. The NULL row is the `<eps>` row of the table.

**Input.** `-i FILE`, or stdin if `FILE` is `-`. The training options `-d`, `-T`, `-q`, `-N`, `-r`,
`-m`, `-s` apply; there are no iterations, no tension optimisation, and `mean_srclen_multiplier`
is the `-m` value (default 1.0). `-s` has no effect (the score is always printed).

**Per line** (`lc` from 1):

1. Parse (§3). Write to stdout every *source* token followed by a space, then `|||`, then a space
   before each *target* token, then ` |||`. Tokens are printed as their original bytes; this is
   **before** the `-r` swap, so it shows the line's own order.
2. Swap if `-r`. If either side is empty: print `Error in line <lc>` on stderr and **exit with
   status 1** (the partial stdout line, without newline, has been written).
3. `lp = log_poisson(m, 0.05 ⊕ (f64(n) ⊗ mean_srclen_multiplier))`.
4. For each target position `j`: compute the cell values exactly as in §8 with `safe_prob`, and
   `sum` the same way, but the argmax differs: `best = 0, best_p = probs[0]` if `use_null`,
   else `best = 0, best_p = 0.0` (not −1); then `i = 1..n` with strict `>`. `lp = lp ⊕ log(sum)`.
   If `best > 0`, write ` ` and the link (`(best−1)-j`, or `j-(best−1)` with `-r`).
5. Write ` ||| `, `lp` (%.6g) and `\n`; `tlp = tlp ⊕ lp` (from 0.0).

At the end: `TOTAL LOG PROB <tlp>` on stderr. stderr is `ARG=` lines, `Loaded …`, error
messages, `TOTAL LOG PROB …`.

Example output line: `the house . ||| la maison . ||| 0-0 1-1 2-2 ||| -12.3456`.

## 17. `atools`

`atools -c COMMAND -i FILE1 [-j FILE2]` (`-` = stdin). Commands used here: `invert` (one file),
`intersect`, `union`, `grow-diag`, `grow-diag-final`, `grow-diag-final-and` (two files; `-i` is
the forward alignment `a`, `-j` the reverse one `b`, both in source-target orientation). Also
present but out of scope: `convert`, `display`, `fmeasure`.

**Reading.** Lines are read in parallel from both files. If one file ends before the other:
`Mismatched number of lines!` and exit 1. Quirk: **if the last line of FILE1 has no trailing
newline, it is processed twice** (the loop tests the stream before reading and does not clear the
line on failure); with a trailing newline there is no duplicate. A line is parsed as a grid:
if it contains ` ||| `, parsing starts after the **last** occurrence. Links are `x-y` (decimal
digits, `-`, digits) separated by spaces; anything else aborts with `BAD ALIGNMENT`. The grid
of a line has width `max x + 1` and height `max y + 1` (1×1 for an empty line). Grid cell
`(x, y)`: `x` = source index `i`, `y` = target index `j`.

**Output.** One line per input line: all set cells in **increasing `(i, j)`** order (row-major by
source index), as `i-j` separated by spaces, then `\n` (empty line if no cell).

**invert**: cell `(i, j)` set iff input `(j, i)` set.

**intersect**, **union**: per cell AND / OR (the result grid has the larger width and height).

**grow-diag family.** Let `A` and `B` be the two link sets. State: `R = A ∩ B`; `alignedI` /
`alignedJ` = the source / target indices covered by `R`. `Align(i, j)` adds `(i, j)` to `R` and
marks `i` and `j` covered.

1. **Grow** (all three methods). `P` = `(A ∪ B) \ R`, ordered by `(i, j)`. Repeat passes until a
   pass adds nothing: in a pass, visit the points of `P` in `(i, j)` order; a point is added
   (`Align`, immediately visible to later points of the same pass) if
   `(i ∉ alignedI or j ∉ alignedJ)` **and** one of its 8 neighbours
   `(i±1, j), (i, j±1), (i±1, j±1)` is in `R`; at the end of the pass remove the added points
   from `P`.
2. **grow-diag** stops here.
3. **grow-diag-final**: for `X = A` then `X = B`: for each `(i, j) ∈ X` in `(i, j)` order, if
   `(i, j) ∉ R` and `(i ∉ alignedI or j ∉ alignedJ)`: `Align(i, j)`.
4. **grow-diag-final-and**: as 3, with the condition `(i ∉ alignedI and j ∉ alignedJ)`.

**Difference from Moses `symal` (gizapp's `symmetrize.rs`).** The final passes and their
predicates are the same. The **grow step differs**: `symal` (and gizapp) iterates over the
*currently aligned* points (target position outer, source inner) and adds each qualifying
union neighbour of such a point, in a fixed neighbour order; atools iterates over the
*candidate* points in `(i, j)` order and adds a candidate that has any aligned neighbour.
The two can produce different alignments for the same inputs, so gizapp's code cannot be reused
unchanged. atools also takes grid sizes from the largest indices in each line rather than from
sentence lengths, which never changes the output.

## 18. Differences from the paper; multi-threaded fast_align

**Source versus paper** (the source wins for parity):

1. The paper's `h(i, j, m, n) = −|i/m − j/n|`. The E-step prior uses it with the 1-based target
   position, but the *empirical* feature `emp_feat` uses the **0-based** target position
   (`Feature(j, i, m, n)` with `j` from 0), so the two terms of the gradient (paper eq. 2) are
   computed at positions shifted by one.
2. `ComputeDLogZ` normalises by `ComputeZ(i, n, m)` with **m and n swapped** (§12); the
   arithmetico-geometric sums use the correct `(m, n)`.
3. λ is optimised by **8 fixed gradient steps of size 20** on the corpus-averaged gradient,
   clamped to `[0.1, 14]`, only after the E-steps of iterations 2 … I−1. The paper does not give
   the procedure.
4. `p0` is a fixed hyperparameter (`-q`), not learned. With `-N` the source prior is still
   scaled by `1 − p0` (so the cell values sum to `1 − p0`, which changes the likelihood but not
   posteriors or alignments).
5. VB: mean-field update `exp(ψ(c + α) − ψ(Σ(c + α)))` with an approximate ψ (§13), α = 0.01
   (the paper's μ). The sum runs only over target words seen with the source word.
6. The last iteration does not update parameters; `-I N` means N−1 EM updates followed by Viterbi
   decoding.
7. The `-s` score adds a Poisson length term with mean `0.05 + n × (mean m/n over the corpus)`;
   not in the paper.
8. Without `-d` the NULL word has the same uniform prior `1/(n+1)` as every source word (the
   paper's model 1 limit), and `-q` is not used.

**Multi-threaded fast_align** (not our target, for reference): `Increment` is an unsynchronised
`+=` from all threads (lost updates possible), the three per-buffer reductions combine per-thread
partial sums in completion order, the initial-pass flush and the M-step run rows in parallel
(harmless). Results therefore vary with the thread count and from run to run (Step 2 measured
1–2.3 % differing alignment lines on medium at 20 threads). With `OMP_NUM_THREADS=1` every
OpenMP loop runs its iterations in order on one thread, and the run is deterministic (Step 2
checked that two runs are byte-identical).

## 19. Edge cases and undefined behaviour

Observed with the reference build (Step 3 experiments); Step 10 decides what we do.

| Input | fast_align behaviour |
|---|---|
| Empty source side, non-empty target (`" ||| y z"`) | error message; `tot_len_ratio` gets `+inf`, so `mean_srclen_multiplier` is `inf`; E-step: only the NULL term (`prob × 1` or `× p0`); with `-N` the sum is 0, `log(0) = −inf` |
| Empty target side | error message; `m/n = 0`; no E-step work; `size_counts` gets `(0, n)` |
| Empty line | both sides empty: `0/0` makes `tot_len_ratio` NaN, printed `expected target length = source length * -nan`; all `-s` scores become `-nan` |
| `-s` after any empty side | the score uses `mean_srclen_multiplier` (inf or NaN) ⇒ `inf`/`-nan` scores on every line |
| `-I 0` | no iterations, empty stdout; with `-p` the table has every key with `-inf` |
| `-I 1` | decoding with `prob = 1e-9`; `-p` table all `-inf` |
| Tab in a line | acts as `|||` (§3) |
| Several separators | first separates; later `|||` are target words (rows/columns named `|||`); with `-r` a `|||` row appears in the table |
| Token `<eps>` in the corpus | is the NULL word (id 1); as a source token it adds its lines' target words to the NULL row (with `-N`, row 1 consists of exactly those); independent of `-b` (§6) |
| No source token in the whole corpus | at the final flush the insert buffer is empty and "ensure `size − 1` rows" wraps to "resize to 0 rows", deleting the NULL row; later lookups are undefined |
| Lengths ≥ 32768 | `size_counts` keys truncate to `i16`; negative lengths give no tension terms; hash per §14.2 |
| `ComputeZ`/`ComputeDLogZ` with `n = 0` | `split = ±inf` or NaN, `u32(trunc(·))` is undefined in C++ (x86 `cvttsd2si` gives an out-of-range result); only reachable via tension optimisation on a corpus with an empty source side. The tension then becomes NaN, and from the second step on two NaNs of opposite sign meet in commutative additions (`λ + Δ`, likelihood sums): which sign survives depends on the operand order GCC chose for `addsd`, so the printed `nan`/`-nan` is a property of the compiled binary, not of the arithmetic. Alignments are unaffected (Step 5 checked: stdout identical, only NaN signs on stderr differ) |
| Long flag options (`--reverse` …) | usage error, exit 1 (§2) |
| Unreadable input | training: `Can't read <file>` (printed before `INITIAL PASS `), the initial pass continues with 0 lines (mean `-nan`), then `ITERATION 1`, `Can't read <file>` again and exit 1 |
| `-b 0` | every line is its own buffer, and the insert buffer flushes after every line with target words |
| `-f` table with `-inf`/`nan` values | reading stops there (§16) |
| atools: last line without `\n` | processed twice (§17) |

## 20. Verification record, instrumented build, probe

All checks below were run in Step 3 and can be rerun.

**Instrumented build.** `benchmarks/build_instrumented.sh` copies the reference source to
`reference/fast_align-instrumented/`, applies `benchmarks/instrument.patch` (our diff; it only
reads state) and builds it with the reference flags; it also builds the probe. With `FA_DUMP=PATH`
the binary writes a dump with all floating-point values as C99 hex floats:

```
initial n_target_tokens X tot_len_ratio X mean_srclen_multiplier X vocab N rows N freeze_copy_same_order 1
dict ID WORD                              (FA_DUMP_DICT=1)
size_counts K m,n,count ...               (vector order, §14.6)
row E BUCKETS SIZE f1 f2 ...              (iteration order; FA_DUMP_ROWS=all or a list, default 1..10)
hash counts H ttable H                    (FNV-1a 64 of (row, key, value bits) in iteration order)
buffer ITER LC NLINES emp_feat X c0 X likelihood X      (FA_DUMP_BUFFERS=1; per-buffer partial sums)
iter K tension X likelihood X c0 X emp_feat_sum X       (totals before normalisation by T)
tension_step II tension X mod_feat_sum X
counts_hash H final_tension X             (before the M-step)
ttable_hash H                             (after the M-step)
end
```

`benchmarks/check_instrumented.sh` runs it (with every dump enabled) over the golden runs and
compares stdout, stderr, exit status and the `-p` table byte for byte with the plain build's
golden outputs. Result: identical for all 72 runs of en-fr tiny and ro-en tiny, all 72 of en-fr
medium and ro-en medium, and en-fr large (default config, both directions). The dump also
confirms `freeze_copy_same_order 1` (§14.5).

**Probe.** `benchmarks/stlhash_probe.cc` inserts key sequences into the two map types used by
fast_align and prints the bucket count after each insert, the iteration order, the order after
copy-assignment and after vector swap, and (with `primes`) the prime table.
`benchmarks/stlhash_vectors.py` writes 28 cases (sequential, reversed, colliding, random, Zipf-like
with repeats up to 25,000 keys, 64-bit-wrapping keys, and `size_counts` pairs including negative
shorts) and the probe's output to `crates/fastalign-core/tests/data/stlhash/`
(`cases.txt`, `expected.txt`, `primes.txt`), the test vectors for Step 4.

**Executable spec.** `benchmarks/spec_check.py` is a slow single-threaded Python model written
from this document (libm through `ctypes`). Results:

- `spec_check.py hash crates/fastalign-core/tests/data/stlhash`: the §14 algorithm reproduces
  all 28 probe cases (bucket counts and orders).
- `spec_check.py run reference/golden/{enfr,roen}-tiny/{fwd,rev}.*`: **all 72 golden runs
  byte-identical** (stdout, stderr including progress, `-p` tables), covering every configuration
  of the matrix, both directions, and forced alignment (`fa-plain`, `fa-diag`, `fa-diag-s`).
- Medium corpora (several `-b` buffers, more rehashes): en-fr medium `fwd.default`,
  `rev.b1000`, `fwd.dov-p`, `rev.fa-diag-s`; ro-en medium `rev.default`, `fwd.misc`, `fwd.dovN`,
  `rev.I10` — all byte-identical.
- `spec_check.py atools reference/golden/*-{tiny,medium}`: all 24 atools golden outputs
  identical.
- Real-data hash order: the instrumented binary on en-fr tiny with `-d -o -v -b 50` (many insert
  flushes), both directions, `FA_DUMP_ROWS=all FA_DUMP_DICT=1`, against the §6 + §14 emulation:
  every non-empty row (4,222 forward, 4,759 reverse) has the same iteration order and bucket
  count, every dictionary id matches, and the `size_counts` vector order matches.

## 21. Notes for the implementation (Steps 4–6)

Facts that follow from the above and that the fast implementation may exploit without changing a
bit:

- Row iteration orders are a pure function of each row's distinct-key first-occurrence sequence
  (§6, §14); they do not depend on `-b` or on thread
  count. They can be computed per row in parallel. Only the *order* is needed (as a permutation
  of the row's sorted keys); the bucket structure can be discarded after construction.
- `-b` affects exactly two things: the grouping of the three per-iteration sums (§9) and the line
  numbers in iteration error messages.
- Per-cell counts must receive their additions in (line, `j`, `i`) order (NULL first); any
  parallel scheme must reproduce that sequence per cell, since f64 addition is not associative.
- The prior values (`ComputeZ`, `UnnormalizedProb`, `Feature`) depend only on
  `(j, i, m, n, λ)`: cache them per length pair and iteration. `prob(e, f)` is a pure lookup.
- The tension gradient depends only on `size_counts` and λ; its eight evaluations are cheap but
  must run sequentially in the §14.6 order.
- The final iteration's likelihood is still computed (statistics are printed); its `c0` and
  `emp_feat` are 0.
- Formatting: implement `%.6g` exactly (correct rounding with ties-to-even on exact halfway
  values, `-nan`/`nan` by sign bit, `inf`), and test it against glibc's `printf` on many values,
  including integers with seven or more digits ending in 5 (`1234565` → `1.23456e+06`).
- `-m` is meaningful only for forced alignment. A gizapp-style CLI may still accept it.
