# Profile t20-seq

`reference/runs/9a/prof-t20-seq/perf.data`: 104743 samples. A sample's phase is the first phase (in the order of `PHASES`) with a marker function on its stack (`profile_phases.py`). Percentages in the per-phase tables are of the phase.

| phase | samples | share |
|---|---|---|
| e_step | 54608 | 52.1 % |
| decode | 17617 | 16.8 % |
| m_step | 28186 | 26.9 % |
| tension | 406 | 0.4 % |
| prior | 90 | 0.1 % |
| initial_pass | 1252 | 1.2 % |
| setup | 135 | 0.1 % |
| load | 592 | 0.6 % |
| symmetrize | 678 | 0.6 % |
| other | 1179 | 1.1 % |

## e_step (54608 samples)

| self % | function |
|---|---|
| 46.6 | `select_unpredictable` |
| 35.8 | `prob` |
| 8.4 | `binary_search_by` |
| 2.8 | `<train::Ctx>::column` |
| 1.3 | `e_step_line` |
| 0.7 | `[unknown]` |
| 0.6 | `eq` |
| 0.5 | `next` |
| 0.5 | `ok` |
| 0.3 | `add` |
| 0.3 | `get` |
| 0.3 | `checked_sub` |
| 0.2 | `row` |
| 0.2 | `abs` |
| 0.2 | `from` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `next` |
| 100.0 | `{closure#0}` |
| 100.0 | `map` |
| 100.0 | `[rayon task]` |
| 100.0 | `e_step_line` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 100.0 | `call_once` |
| 97.0 | `<train::Ctx>::column` |
| 93.2 | `prob` |
| 57.2 | `find` |
| 55.8 | `binary_search_by` |
| 55.8 | `binary_search` |
| 46.9 | `select_unpredictable` |

## decode (17617 samples)

| self % | function |
|---|---|
| 47.7 | `select_unpredictable` |
| 36.0 | `prob` |
| 8.7 | `binary_search_by` |
| 2.9 | `<train::Ctx>::column` |
| 0.8 | `[unknown]` |
| 0.6 | `ok` |
| 0.5 | `decode_line` |
| 0.5 | `eq` |
| 0.2 | `get` |
| 0.2 | `checked_sub` |
| 0.2 | `get_offset_len_noubcheck` |
| 0.2 | `row` |
| 0.2 | `exp@@GLIBC_2.29` |
| 0.2 | `__ieee754_exp_fma` |
| 0.2 | `find` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `next` |
| 100.0 | `{closure#0}` |
| 100.0 | `map` |
| 100.0 | `decode_line` |
| 100.0 | `[rayon task]` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 100.0 | `call_once` |
| 98.5 | `<train::Ctx>::column` |
| 94.7 | `prob` |
| 58.6 | `find` |
| 57.2 | `binary_search_by` |
| 57.2 | `binary_search` |
| 47.9 | `select_unpredictable` |

## m_step (28186 samples)

| self % | function |
|---|---|
| 71.2 | `accumulate_lines` |
| 8.4 | `line` |
| 7.3 | `eq` |
| 3.3 | `index` |
| 1.4 | `__ieee754_exp_fma` |
| 1.2 | `next` |
| 0.9 | `[unknown]` |
| 0.9 | `as_slice` |
| 0.8 | `<train::Ctx>::prior` |
| 0.7 | `push_mut` |
| 0.6 | `exp@@GLIBC_2.29` |
| 0.6 | `normalise` |
| 0.5 | `digamma` |
| 0.5 | `{closure#0}` |
| 0.4 | `index_mut` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `{closure#0}` |
| 100.0 | `[rayon task]` |
| 100.0 | `fold` |
| 100.0 | `for_each` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 100.0 | `{closure#4}` |
| 97.5 | `train::accumulate` |
| 96.8 | `accumulate_lines` |
| 9.4 | `line` |
| 9.0 | `next` |
| 7.4 | `eq` |
| 4.6 | `index` |
| 3.9 | `<train::Ctx>::prior` |

## tension (406 samples)

| self % | function |
|---|---|
| 19.2 | `__ieee754_exp_fma` |
| 15.5 | `arith_geom_series` |
| 14.5 | `exp_inline` |
| 10.1 | `exp@@GLIBC_2.29` |
| 7.1 | `pow@@GLIBC_2.29` |
| 6.9 | `log_inline` |
| 4.7 | `compute_z` |
| 3.4 | `trunc_to_u32` |
| 3.0 | `__ieee754_pow_fma` |
| 2.5 | `powf` |
| 1.7 | `compute_dlogz` |
| 1.7 | `feature` |
| 1.5 | `from` |
| 1.5 | `__exp_finite@GLIBC_2.15@plt` |
| 1.5 | `__pow_finite@GLIBC_2.15@plt` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `{closure#0}` |
| 100.0 | `compute_dlogz` |
| 100.0 | `[rayon task]` |
| 100.0 | `fold` |
| 100.0 | `for_each` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 43.1 | `compute_z` |
| 37.9 | `pow` |
| 37.4 | `powf` |
| 36.2 | `arith_geom_series` |
| 32.8 | `exp` |
| 27.1 | `unnormalized_prob` |
| 25.4 | `__ieee754_pow_fma` |

## prior (90 samples)

| self % | function |
|---|---|
| 35.6 | `__ieee754_exp_fma` |
| 23.3 | `exp@@GLIBC_2.29` |
| 8.9 | `compute_z` |
| 6.7 | `exp_inline` |
| 5.6 | `exp` |
| 4.4 | `log_inline` |
| 3.3 | `pow@@GLIBC_2.29` |
| 3.3 | `from` |
| 2.2 | `__exp_finite@GLIBC_2.15@plt` |
| 2.2 | `feature` |
| 1.1 | `__ieee754_pow_fma` |
| 1.1 | `top12` |
| 1.1 | `__pow_finite@GLIBC_2.15@plt` |
| 1.1 | `unnormalized_prob` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `{closure#0}` |
| 100.0 | `[rayon task]` |
| 100.0 | `fold` |
| 100.0 | `for_each` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 72.2 | `unnormalized_prob` |
| 66.7 | `exp` |
| 60.0 | `{closure#3}` |
| 40.0 | `compute_z` |
| 35.6 | `__ieee754_exp_fma` |
| 23.3 | `exp@@GLIBC_2.29` |
| 17.8 | `powf` |
| 17.8 | `pow` |

## initial_pass (1252 samples)

| self % | function |
|---|---|
| 31.5 | `as_mut_slice` |
| 8.8 | `ttable::ordinary_row_sequence` |
| 7.7 | `line` |
| 7.2 | `<stlhash::StlOrder>::insert_new` |
| 6.0 | `push_mut` |
| 5.9 | `push_new` |
| 5.2 | `[unknown]` |
| 5.0 | `{closure#0}` |
| 3.6 | `rehash` |
| 3.1 | `index` |
| 2.8 | `eq` |
| 2.6 | `index_mut` |
| 1.1 | `non_null` |
| 1.0 | `copy` |
| 0.7 | `add` |

| inclusive % | function |
|---|---|
| 99.7 | `[rayon task]` |
| 97.4 | `consume_iter` |
| 97.4 | `fold_with` |
| 88.7 | `call` |
| 88.4 | `{closure#2}` |
| 88.4 | `<rayon::iter::map_with::MapWithFolder as rayon::iter::plumbing::Folder>::consume_iter::with::{closure#0}` |
| 87.9 | `finish_row` |
| 63.7 | `len` |
| 63.2 | `ttable::ordinary_row_sequence` |
| 44.5 | `push_new` |
| 34.5 | `index_mut` |
| 31.9 | `as_mut_slice` |
| 31.9 | `deref_mut` |
| 20.8 | `next` |
| 19.2 | `eq` |

## setup (135 samples)

| self % | function |
|---|---|
| 58.5 | `{closure#0}` |
| 8.9 | `[unknown]` |
| 8.9 | `index` |
| 7.4 | `eq` |
| 5.2 | `__memset_avx2_unaligned_erms` |
| 3.7 | `index_mut` |
| 1.5 | `line_len` |
| 0.7 | `push_mut` |
| 0.7 | `next` |
| 0.7 | `as_slice` |
| 0.7 | `__memmove_avx_unaligned_erms` |
| 0.7 | `add_assign` |
| 0.7 | `build` |
| 0.7 | `lt` |
| 0.7 | `get` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 95.6 | `consume_iter` |
| 95.6 | `fold_with` |
| 90.4 | `{closure#0}` |
| 85.9 | `<&<train::RowLines>::build::{closure#0} as core::ops::function::FnMut>::call_mut` |
| 10.4 | `index` |
| 8.9 | `[unknown]` |
| 8.9 | `next` |
| 8.1 | `__memset_avx2_unaligned_erms` |
| 7.4 | `eq` |
| 3.7 | `build` |
| 3.7 | `index_mut` |
| 3.0 | `do_call` |
| 3.0 | `halt_unwinding` |
| 3.0 | `catch_unwind` |

## load (592 samples)

| self % | function |
|---|---|
| 20.4 | `__memcmp_avx2_movbe` |
| 12.3 | `tokenize` |
| 10.3 | `[unknown]` |
| 10.1 | `eq` |
| 6.1 | `write` |
| 4.9 | `push_mut` |
| 3.9 | `hash_bytes` |
| 3.0 | `is_some` |
| 2.9 | `lowest_set_bit` |
| 2.5 | `<corpus::LocalDict>::intern` |
| 2.4 | `corpus::process_chunk::{closure#1}` |
| 2.4 | `equal_same_length` |
| 2.4 | `{closure#0}` |
| 1.7 | `probe_seq` |
| 1.7 | `_mm_movemask_epi8` |

| inclusive % | function |
|---|---|
| 100.0 | `call_mut` |
| 100.0 | `next` |
| 100.0 | `{closure#0}` |
| 100.0 | `map` |
| 100.0 | `process_chunk` |
| 100.0 | `[rayon task]` |
| 100.0 | `consume_iter` |
| 100.0 | `fold_with` |
| 100.0 | `call_once` |
| 99.3 | `corpus::process_chunk::{closure#1}` |
| 47.5 | `push` |
| 47.5 | `push_mut` |
| 35.6 | `<corpus::LocalDict>::intern` |
| 30.6 | `rustc_entry` |
| 30.6 | `entry` |

## symmetrize (678 samples)

| self % | function |
|---|---|
| 8.7 | `symmetrize` |
| 8.3 | `eq` |
| 5.5 | `_int_malloc` |
| 4.4 | `[unknown]` |
| 4.4 | `index` |
| 4.1 | `lt` |
| 3.7 | `core::str::validations::run_utf8_validation` |
| 2.8 | `dedup_by` |
| 2.8 | `realloc` |
| 2.7 | `{closure#0}` |
| 2.7 | `symmetrize::parse_line` |
| 2.5 | `from_ascii_bytes_radix_impl` |
| 2.1 | `core::str::converts::from_utf8` |
| 1.9 | `spec_eq` |
| 1.9 | `copy_nonoverlapping` |

| inclusive % | function |
|---|---|
| 95.0 | `next` |
| 91.9 | `{closure#1}` |
| 91.3 | `call_mut` |
| 91.3 | `call_once` |
| 91.3 | `map` |
| 91.3 | `[rayon task]` |
| 91.3 | `consume_iter` |
| 91.3 | `fold_with` |
| 58.8 | `symmetrize` |
| 31.1 | `symmetrize::parse_line` |
| 22.9 | `non_null` |
| 22.9 | `ptr` |
| 13.7 | `from_iter` |
| 13.7 | `collect` |
| 13.0 | `ok_or` |

## other (1179 samples)

| self % | function |
|---|---|
| 24.1 | `[unknown]` |
| 15.5 | `__ieee754_log_fma` |
| 5.8 | `exp@@GLIBC_2.29` |
| 4.1 | `{closure#0}` |
| 3.4 | `log@@GLIBC_2.29` |
| 3.3 | `__memset_avx2_unaligned_erms` |
| 2.6 | `__memmove_avx_unaligned_erms` |
| 2.5 | `copy_nonoverlapping` |
| 2.4 | `swap_if_less` |
| 2.0 | `as_slice` |
| 1.8 | `copy` |
| 1.8 | `row` |
| 1.4 | `index` |
| 1.3 | `{closure#3}` |
| 1.2 | `get` |

| inclusive % | function |
|---|---|
| 80.2 | `[rayon task]` |
| 44.5 | `consume_iter` |
| 44.5 | `fold_with` |
| 36.1 | `{closure#0}` |
| 27.9 | `call_mut` |
| 25.6 | `{closure#2}` |
| 24.1 | `[unknown]` |
| 21.0 | `call` |
| 19.3 | `{closure#1}` |
| 17.6 | `fold` |
| 17.6 | `for_each` |
| 17.6 | `finish_row` |
| 17.6 | `<rayon::iter::map_with::MapWithFolder as rayon::iter::plumbing::Folder>::consume_iter::with::{closure#0}` |
| 16.5 | `__ieee754_log_fma` |
| 13.8 | `wait_until` |
