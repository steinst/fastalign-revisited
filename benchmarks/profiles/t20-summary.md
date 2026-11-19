# Profile t20

`reference/runs/9a/prof-t20/perf.data`: 110073 samples. A sample's phase is the first phase (in the order of `PHASES`) with a marker function on its stack (`profile_phases.py`). Percentages in the per-phase tables are of the phase.

| phase | samples | share |
|---|---|---|
| e_step | 59827 | 54.4 % |
| decode | 18925 | 17.2 % |
| m_step | 26599 | 24.2 % |
| tension | 435 | 0.4 % |
| prior | 76 | 0.1 % |
| initial_pass | 1494 | 1.4 % |
| setup | 131 | 0.1 % |
| load | 579 | 0.5 % |
| symmetrize | 835 | 0.8 % |
| other | 1172 | 1.1 % |

## e_step (59827 samples)

| self % | function |
|---|---|
| 48.4 | `select_unpredictable` |
| 35.5 | `prob` |
| 7.7 | `binary_search_by` |
| 2.8 | `<train::Ctx>::column` |
| 1.1 | `e_step_line` |
| 0.6 | `[unknown]` |
| 0.5 | `next` |
| 0.5 | `eq` |
| 0.4 | `ok` |
| 0.3 | `add` |
| 0.3 | `get` |
| 0.2 | `abs` |
| 0.2 | `row` |
| 0.2 | `checked_sub` |
| 0.2 | `__ieee754_exp_fma` |

| inclusive % | function |
|---|---|
| 100.0 | `consume_iter` |
| 100.0 | `next` |
| 100.0 | `[rayon task]` |
| 100.0 | `call_once` |
| 100.0 | `fold_with` |
| 100.0 | `e_step_line` |
| 100.0 | `map` |
| 100.0 | `{closure#0}` |
| 100.0 | `call_mut` |
| 97.4 | `<train::Ctx>::column` |
| 93.6 | `prob` |
| 58.0 | `find` |
| 56.8 | `binary_search` |
| 56.8 | `binary_search_by` |
| 48.6 | `select_unpredictable` |

## decode (18925 samples)

| self % | function |
|---|---|
| 48.9 | `select_unpredictable` |
| 35.9 | `prob` |
| 8.3 | `binary_search_by` |
| 2.5 | `<train::Ctx>::column` |
| 0.6 | `[unknown]` |
| 0.5 | `decode_line` |
| 0.5 | `ok` |
| 0.4 | `eq` |
| 0.3 | `checked_sub` |
| 0.3 | `get` |
| 0.2 | `get_offset_len_noubcheck` |
| 0.2 | `row` |
| 0.2 | `__ieee754_exp_fma` |
| 0.1 | `find` |
| 0.1 | `exp@@GLIBC_2.29` |

| inclusive % | function |
|---|---|
| 100.0 | `consume_iter` |
| 100.0 | `next` |
| 100.0 | `decode_line` |
| 100.0 | `[rayon task]` |
| 100.0 | `call_once` |
| 100.0 | `fold_with` |
| 100.0 | `map` |
| 100.0 | `{closure#0}` |
| 100.0 | `call_mut` |
| 98.6 | `<train::Ctx>::column` |
| 95.3 | `prob` |
| 59.3 | `find` |
| 58.0 | `binary_search` |
| 58.0 | `binary_search_by` |
| 49.1 | `select_unpredictable` |

## m_step (26599 samples)

| self % | function |
|---|---|
| 70.9 | `accumulate_lines` |
| 8.5 | `line` |
| 7.1 | `eq` |
| 3.5 | `index` |
| 1.4 | `__ieee754_exp_fma` |
| 1.3 | `next` |
| 0.9 | `as_slice` |
| 0.9 | `[unknown]` |
| 0.8 | `<train::Ctx>::prior` |
| 0.8 | `push_mut` |
| 0.7 | `exp@@GLIBC_2.29` |
| 0.5 | `digamma` |
| 0.5 | `normalise` |
| 0.5 | `{closure#0}` |
| 0.5 | `index_mut` |

| inclusive % | function |
|---|---|
| 100.0 | `consume_iter` |
| 100.0 | `[rayon task]` |
| 100.0 | `fold_with` |
| 100.0 | `for_each` |
| 100.0 | `{closure#0}` |
| 100.0 | `call_mut` |
| 100.0 | `fold` |
| 100.0 | `{closure#4}` |
| 97.6 | `train::accumulate` |
| 97.0 | `accumulate_lines` |
| 9.6 | `line` |
| 9.0 | `next` |
| 7.1 | `eq` |
| 4.8 | `index` |
| 4.1 | `<train::Ctx>::prior` |

## tension (435 samples)

| self % | function |
|---|---|
| 20.0 | `__ieee754_exp_fma` |
| 15.9 | `exp_inline` |
| 10.1 | `exp@@GLIBC_2.29` |
| 9.4 | `arith_geom_series` |
| 8.5 | `log_inline` |
| 6.0 | `__ieee754_pow_fma` |
| 5.3 | `pow@@GLIBC_2.29` |
| 5.1 | `math::compute_z` |
| 3.4 | `from` |
| 3.0 | `trunc_to_u32` |
| 3.0 | `feature` |
| 1.6 | `exp` |
| 1.6 | `top12` |
| 1.6 | `compute_dlogz` |
| 1.4 | `__pow_finite@GLIBC_2.15@plt` |

| inclusive % | function |
|---|---|
| 100.0 | `consume_iter` |
| 100.0 | `[rayon task]` |
| 100.0 | `compute_dlogz` |
| 100.0 | `fold_with` |
| 100.0 | `for_each` |
| 100.0 | `{closure#0}` |
| 100.0 | `call_mut` |
| 100.0 | `fold` |
| 57.0 | `math::compute_z` |
| 40.0 | `pow` |
| 39.5 | `powf` |
| 32.2 | `__ieee754_pow_fma` |
| 31.0 | `exp` |
| 30.3 | `arith_geom_series` |
| 24.6 | `unnormalized_prob` |

## prior (76 samples)

| self % | function |
|---|---|
| 42.1 | `__ieee754_exp_fma` |
| 22.4 | `exp@@GLIBC_2.29` |
| 6.6 | `compute_z` |
| 5.3 | `log_inline` |
| 3.9 | `exp` |
| 3.9 | `exp_inline` |
| 2.6 | `feature` |
| 2.6 | `trunc_to_u32` |
| 2.6 | `__exp_finite@GLIBC_2.15@plt` |
| 2.6 | `pow@@GLIBC_2.29` |
| 1.3 | `top12` |
| 1.3 | `black_box` |
| 1.3 | `__ieee754_pow_fma` |
| 1.3 | `unnormalized_prob` |

| inclusive % | function |
|---|---|
| 100.0 | `consume_iter` |
| 100.0 | `[rayon task]` |
| 100.0 | `fold_with` |
| 100.0 | `for_each` |
| 100.0 | `{closure#0}` |
| 100.0 | `call_mut` |
| 100.0 | `fold` |
| 71.1 | `exp` |
| 71.1 | `unnormalized_prob` |
| 52.6 | `{closure#3}` |
| 47.4 | `compute_z` |
| 42.1 | `__ieee754_exp_fma` |
| 22.4 | `exp@@GLIBC_2.29` |
| 15.8 | `pow` |
| 14.5 | `powf` |

## initial_pass (1494 samples)

| self % | function |
|---|---|
| 33.3 | `as_mut_slice` |
| 9.8 | `line` |
| 7.5 | `ttable::ordinary_row_sequence` |
| 6.6 | `push_new` |
| 6.1 | `[unknown]` |
| 5.6 | `<stlhash::StlOrder>::insert_new` |
| 4.8 | `push_mut` |
| 4.3 | `{closure#0}` |
| 3.7 | `rehash` |
| 3.5 | `index` |
| 2.8 | `eq` |
| 2.1 | `index_mut` |
| 0.9 | `non_null` |
| 0.7 | `copy_nonoverlapping` |
| 0.6 | `set_next` |

| inclusive % | function |
|---|---|
| 99.5 | `[rayon task]` |
| 97.3 | `consume_iter` |
| 97.3 | `fold_with` |
| 89.7 | `call` |
| 89.2 | `{closure#2}` |
| 89.2 | `<rayon::iter::map_with::MapWithFolder as rayon::iter::plumbing::Folder>::consume_iter::with::{closure#0}` |
| 88.6 | `finish_row` |
| 66.1 | `len` |
| 65.5 | `ttable::ordinary_row_sequence` |
| 46.0 | `push_new` |
| 35.7 | `index_mut` |
| 33.6 | `as_mut_slice` |
| 33.6 | `deref_mut` |
| 20.3 | `next` |
| 18.5 | `eq` |

## setup (131 samples)

| self % | function |
|---|---|
| 45.0 | `{closure#0}` |
| 12.2 | `index` |
| 9.9 | `[unknown]` |
| 7.6 | `eq` |
| 5.3 | `__memset_avx2_unaligned_erms` |
| 5.3 | `index_mut` |
| 3.8 | `next` |
| 3.8 | `lt` |
| 2.3 | `add_assign` |
| 1.5 | `line_len` |
| 1.5 | `__ieee754_log_fma` |
| 0.8 | `__memmove_avx_unaligned_erms` |
| 0.8 | `{closure#3}` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 93.1 | `consume_iter` |
| 93.1 | `fold_with` |
| 83.2 | `{closure#0}` |
| 77.9 | `<&<train::RowLines>::build::{closure#0} as core::ops::function::FnMut>::call_mut` |
| 15.3 | `next` |
| 12.2 | `index` |
| 11.5 | `__memset_avx2_unaligned_erms` |
| 9.9 | `[unknown]` |
| 7.6 | `eq` |
| 5.3 | `index_mut` |
| 4.6 | `build` |
| 3.8 | `spec_next` |
| 3.8 | `lt` |
| 3.1 | `halt_unwinding` |

## load (579 samples)

| self % | function |
|---|---|
| 21.4 | `__memcmp_avx2_movbe` |
| 13.0 | `tokenize` |
| 11.6 | `eq` |
| 8.8 | `[unknown]` |
| 6.4 | `push_mut` |
| 5.9 | `write` |
| 4.0 | `hash_bytes` |
| 3.8 | `<corpus::LocalDict>::intern` |
| 2.6 | `{closure#0}` |
| 2.4 | `corpus::process_chunk::{closure#1}` |
| 2.4 | `lowest_set_bit` |
| 2.2 | `_mm_movemask_epi8` |
| 2.1 | `is_some` |
| 1.9 | `checked_sub` |
| 1.6 | `copied` |

| inclusive % | function |
|---|---|
| 96.4 | `{closure#0}` |
| 95.2 | `[rayon task]` |
| 95.0 | `consume_iter` |
| 95.0 | `next` |
| 95.0 | `process_chunk` |
| 95.0 | `call_once` |
| 95.0 | `fold_with` |
| 95.0 | `map` |
| 95.0 | `call_mut` |
| 93.8 | `corpus::process_chunk::{closure#1}` |
| 47.2 | `push` |
| 47.2 | `push_mut` |
| 34.0 | `<corpus::LocalDict>::intern` |
| 29.9 | `entry` |
| 29.9 | `rustc_entry` |

## symmetrize (835 samples)

| self % | function |
|---|---|
| 6.1 | `symmetrize` |
| 6.0 | `eq` |
| 4.2 | `[unknown]` |
| 4.1 | `copy_nonoverlapping` |
| 3.4 | `symmetrize::parse_line` |
| 3.2 | `_int_malloc` |
| 3.1 | `index` |
| 2.9 | `_int_free` |
| 2.6 | `realloc` |
| 2.4 | `dedup_by` |
| 2.4 | `push_mut` |
| 2.4 | `lt` |
| 2.3 | `{closure#0}` |
| 2.2 | `get` |
| 2.2 | `<core::fmt::Formatter>::pad_integral` |

| inclusive % | function |
|---|---|
| 100.0 | `{closure#1}` |
| 95.0 | `next` |
| 91.7 | `map` |
| 91.6 | `consume_iter` |
| 91.6 | `[rayon task]` |
| 91.6 | `call_once` |
| 91.6 | `fold_with` |
| 91.6 | `call_mut` |
| 53.3 | `symmetrize` |
| 24.1 | `symmetrize::parse_line` |
| 21.8 | `push` |
| 21.8 | `push_mut` |
| 21.4 | `ptr` |
| 21.4 | `non_null` |
| 14.0 | `{closure#0}` |

## other (1172 samples)

| self % | function |
|---|---|
| 25.5 | `[unknown]` |
| 14.0 | `__ieee754_log_fma` |
| 5.0 | `exp@@GLIBC_2.29` |
| 4.6 | `__memmove_avx_unaligned_erms` |
| 4.5 | `__memset_avx2_unaligned_erms` |
| 3.3 | `{closure#0}` |
| 3.0 | `swap_if_less` |
| 2.9 | `as_slice` |
| 2.9 | `log@@GLIBC_2.29` |
| 2.2 | `{closure#0} u32>` |
| 2.1 | `index` |
| 2.1 | `row` |
| 2.0 | `copy_nonoverlapping` |
| 2.0 | `copy` |
| 1.4 | `lt` |

| inclusive % | function |
|---|---|
| 87.5 | `[rayon task]` |
| 42.5 | `consume_iter` |
| 42.5 | `fold_with` |
| 35.2 | `{closure#0}` |
| 30.9 | `{closure#2}` |
| 28.3 | `call` |
| 25.5 | `[unknown]` |
| 22.4 | `<rayon::iter::map_with::MapWithFolder as rayon::iter::plumbing::Folder>::consume_iter::with::{closure#0}` |
| 22.4 | `finish_row` |
| 20.1 | `do_call` |
| 20.1 | `halt_unwinding` |
| 20.1 | `catch_unwind` |
| 19.3 | `call_mut` |
| 16.9 | `for_each` |
| 16.9 | `fold` |
