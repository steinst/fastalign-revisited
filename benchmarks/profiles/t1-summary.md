# Profile t1

`reference/runs/9a/prof-t1/perf.data`: 91215 samples. A sample's phase is the first phase (in the order of `PHASES`) with a marker function on its stack (`profile_phases.py`). Percentages in the per-phase tables are of the phase.

| phase | samples | share |
|---|---|---|
| e_step | 55001 | 60.3 % |
| decode | 19875 | 21.8 % |
| m_step | 13282 | 14.6 % |
| tension | 306 | 0.3 % |
| prior | 45 | 0.0 % |
| initial_pass | 1523 | 1.7 % |
| setup | 51 | 0.1 % |
| load | 491 | 0.5 % |
| other | 641 | 0.7 % |

## e_step (55001 samples)

| self % | function |
|---|---|
| 43.8 | `select_unpredictable` |
| 38.2 | `prob` |
| 8.8 | `binary_search_by` |
| 3.4 | `<train::Ctx>::column` |
| 1.1 | `e_step_line` |
| 0.8 | `ok` |
| 0.4 | `next` |
| 0.4 | `checked_sub` |
| 0.3 | `get_offset_len_noubcheck` |
| 0.3 | `find` |
| 0.3 | `__ieee754_exp_fma` |
| 0.3 | `get` |
| 0.3 | `add` |
| 0.3 | `[unknown]` |
| 0.2 | `from` |

| inclusive % | function |
|---|---|
| 100.0 | `next` |
| 100.0 | `[rayon task]` |
| 100.0 | `consume_iter` |
| 100.0 | `call_mut` |
| 100.0 | `map` |
| 100.0 | `call_once` |
| 100.0 | `fold_with` |
| 100.0 | `e_step_line` |
| 100.0 | `{closure#0}` |
| 97.4 | `<train::Ctx>::column` |
| 93.0 | `prob` |
| 54.7 | `find` |
| 52.7 | `binary_search_by` |
| 52.7 | `binary_search` |
| 43.9 | `select_unpredictable` |

## decode (19875 samples)

| self % | function |
|---|---|
| 46.3 | `select_unpredictable` |
| 37.3 | `prob` |
| 8.4 | `binary_search_by` |
| 3.1 | `<train::Ctx>::column` |
| 0.8 | `decode_line` |
| 0.7 | `ok` |
| 0.4 | `get_offset_len_noubcheck` |
| 0.3 | `find` |
| 0.3 | `row` |
| 0.3 | `checked_sub` |
| 0.2 | `__ieee754_exp_fma` |
| 0.2 | `get` |
| 0.2 | `[unknown]` |
| 0.2 | `<core::fmt::Formatter>::pad_integral` |
| 0.2 | `exp@@GLIBC_2.29` |

| inclusive % | function |
|---|---|
| 100.0 | `next` |
| 100.0 | `[rayon task]` |
| 100.0 | `consume_iter` |
| 100.0 | `call_mut` |
| 100.0 | `map` |
| 100.0 | `call_once` |
| 100.0 | `fold_with` |
| 100.0 | `decode_line` |
| 100.0 | `{closure#0}` |
| 98.0 | `<train::Ctx>::column` |
| 94.1 | `prob` |
| 56.7 | `find` |
| 54.8 | `binary_search_by` |
| 54.8 | `binary_search` |
| 46.4 | `select_unpredictable` |

## m_step (13282 samples)

| self % | function |
|---|---|
| 54.8 | `accumulate_lines` |
| 18.1 | `eq` |
| 9.9 | `line` |
| 3.0 | `index` |
| 2.7 | `next` |
| 2.3 | `__ieee754_exp_fma` |
| 1.3 | `digamma` |
| 1.1 | `<train::Ctx>::prior` |
| 1.1 | `push_mut` |
| 0.9 | `exp@@GLIBC_2.29` |
| 0.7 | `as_slice` |
| 0.7 | `normalise` |
| 0.6 | `[unknown]` |
| 0.4 | `non_null` |
| 0.4 | `index_mut` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 100.0 | `{closure#4}` |
| 100.0 | `fold` |
| 100.0 | `consume_iter` |
| 100.0 | `call_mut` |
| 100.0 | `fold_with` |
| 100.0 | `for_each` |
| 100.0 | `{closure#0}` |
| 96.5 | `train::accumulate` |
| 95.9 | `accumulate_lines` |
| 21.0 | `next` |
| 18.1 | `eq` |
| 10.7 | `line` |
| 3.9 | `index` |
| 3.5 | `normalise` |

## tension (306 samples)

| self % | function |
|---|---|
| 23.9 | `__ieee754_exp_fma` |
| 12.4 | `log_inline` |
| 11.1 | `exp_inline` |
| 6.9 | `exp@@GLIBC_2.29` |
| 6.9 | `__ieee754_pow_fma` |
| 5.6 | `pow@@GLIBC_2.29` |
| 4.6 | `arith_geom_series` |
| 4.6 | `math::compute_z` |
| 3.6 | `from` |
| 2.9 | `feature` |
| 2.9 | `compute_dlogz` |
| 2.3 | `__exp_finite@GLIBC_2.15@plt` |
| 2.3 | `__pow_finite@GLIBC_2.15@plt` |
| 2.3 | `top12` |
| 2.3 | `powf` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 100.0 | `fold` |
| 100.0 | `compute_dlogz` |
| 100.0 | `consume_iter` |
| 100.0 | `call_mut` |
| 100.0 | `fold_with` |
| 100.0 | `for_each` |
| 100.0 | `{closure#0}` |
| 57.8 | `math::compute_z` |
| 44.1 | `pow` |
| 43.1 | `powf` |
| 33.3 | `exp` |
| 32.7 | `__ieee754_pow_fma` |
| 23.9 | `__ieee754_exp_fma` |
| 23.5 | `unnormalized_prob` |

## prior (45 samples)

| self % | function |
|---|---|
| 37.8 | `__ieee754_exp_fma` |
| 13.3 | `exp@@GLIBC_2.29` |
| 6.7 | `__exp_finite@GLIBC_2.15@plt` |
| 6.7 | `exp` |
| 6.7 | `math::compute_z` |
| 4.4 | `feature` |
| 4.4 | `top12` |
| 4.4 | `compute_z` |
| 2.2 | `[unknown]` |
| 2.2 | `exp_inline` |
| 2.2 | `from` |
| 2.2 | `trunc_to_u32` |
| 2.2 | `__ieee754_pow_fma` |
| 2.2 | `pow@@GLIBC_2.29` |
| 2.2 | `black_box` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 93.3 | `fold` |
| 93.3 | `consume_iter` |
| 93.3 | `call_mut` |
| 93.3 | `fold_with` |
| 93.3 | `for_each` |
| 93.3 | `{closure#0}` |
| 64.4 | `exp` |
| 57.8 | `unnormalized_prob` |
| 46.7 | `compute_z` |
| 46.7 | `{closure#3}` |
| 37.8 | `__ieee754_exp_fma` |
| 13.3 | `exp@@GLIBC_2.29` |
| 13.3 | `pow` |
| 11.1 | `powf` |

## initial_pass (1523 samples)

| self % | function |
|---|---|
| 33.2 | `as_mut_slice` |
| 8.7 | `ttable::ordinary_row_sequence` |
| 8.7 | `line` |
| 8.5 | `[unknown]` |
| 6.1 | `<stlhash::StlOrder>::insert_new` |
| 4.6 | `push_new` |
| 4.4 | `{closure#0}` |
| 4.3 | `rehash` |
| 4.1 | `push_mut` |
| 2.0 | `index` |
| 1.9 | `copy` |
| 1.5 | `eq` |
| 1.5 | `lt` |
| 1.4 | `swap_if_less` |
| 1.2 | `copy_nonoverlapping` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 95.3 | `consume_iter` |
| 95.3 | `fold_with` |
| 92.5 | `{closure#2}` |
| 92.5 | `call` |
| 92.5 | `<rayon::iter::map_with::MapWithFolder as rayon::iter::plumbing::Folder>::consume_iter::with::{closure#0}` |
| 92.0 | `finish_row` |
| 59.9 | `len` |
| 59.9 | `ttable::ordinary_row_sequence` |
| 41.7 | `push_new` |
| 33.9 | `index_mut` |
| 33.2 | `deref_mut` |
| 33.2 | `as_mut_slice` |
| 15.2 | `next` |
| 14.4 | `eq` |

## setup (51 samples)

| self % | function |
|---|---|
| 58.8 | `{closure#0}` |
| 15.7 | `[unknown]` |
| 7.8 | `index` |
| 3.9 | `eq` |
| 3.9 | `{closure#2}` |
| 2.0 | `rotate_left` |
| 2.0 | `find_inner` |
| 2.0 | `index_mut` |
| 2.0 | `spec_next` |
| 2.0 | `saturating_add` |

| inclusive % | function |
|---|---|
| 100.0 | `[rayon task]` |
| 100.0 | `{closure#0}` |
| 88.2 | `consume_iter` |
| 88.2 | `fold_with` |
| 88.2 | `<&<train::RowLines>::build::{closure#0} as core::ops::function::FnMut>::call_mut` |
| 15.7 | `[unknown]` |
| 7.8 | `index` |
| 7.8 | `run_inline` |
| 7.8 | `{closure#2}` |
| 5.9 | `next` |
| 3.9 | `entry` |
| 3.9 | `build` |
| 3.9 | `rustc_entry` |
| 3.9 | `catch_unwind` |
| 3.9 | `call_once` |

## load (491 samples)

| self % | function |
|---|---|
| 16.7 | `__memcmp_avx2_movbe` |
| 9.6 | `tokenize` |
| 9.4 | `write` |
| 7.9 | `push_mut` |
| 7.7 | `<corpus::LocalDict>::intern` |
| 6.5 | `eq` |
| 5.7 | `[unknown]` |
| 5.5 | `{closure#0}` |
| 5.3 | `is_some` |
| 5.1 | `non_null` |
| 4.7 | `corpus::process_chunk::{closure#1}` |
| 3.7 | `hash_bytes` |
| 3.1 | `equal_same_length` |
| 1.2 | `full` |
| 1.2 | `probe_seq` |

| inclusive % | function |
|---|---|
| 98.0 | `{closure#0}` |
| 96.7 | `call_mut` |
| 96.7 | `fold_with` |
| 96.7 | `next` |
| 96.7 | `[rayon task]` |
| 96.7 | `consume_iter` |
| 96.7 | `map` |
| 96.7 | `call_once` |
| 96.7 | `process_chunk` |
| 95.7 | `corpus::process_chunk::{closure#1}` |
| 46.0 | `push` |
| 46.0 | `push_mut` |
| 37.9 | `tokenize` |
| 28.3 | `<corpus::LocalDict>::intern` |
| 21.0 | `rustc_entry` |

## other (641 samples)

| self % | function |
|---|---|
| 31.7 | `__ieee754_log_fma` |
| 16.2 | `[unknown]` |
| 14.4 | `exp@@GLIBC_2.29` |
| 5.8 | `log@@GLIBC_2.29` |
| 4.7 | `<train::Ctx>::column` |
| 3.6 | `top16` |
| 3.1 | `__log_finite@GLIBC_2.15@plt` |
| 3.1 | `<train::Ctx>::prior` |
| 2.2 | `__memset_avx2_unaligned_erms` |
| 2.0 | `__memmove_avx_unaligned_erms` |
| 1.7 | `row` |
| 1.6 | `<u64 as core::fmt::Display>::fmt` |
| 1.4 | `{closure#2}` |
| 1.2 | `{closure#0}` |
| 0.9 | `as_slice` |

| inclusive % | function |
|---|---|
| 58.8 | `[rayon task]` |
| 54.6 | `{closure#0}` |
| 52.6 | `halt_unwinding` |
| 52.6 | `do_call` |
| 52.6 | `catch_unwind` |
| 44.8 | `drop_glue` |
| 35.4 | `__ieee754_log_fma` |
| 35.3 | `branch` |
| 33.5 | `run` |
| 33.5 | `{closure#1}` |
| 33.5 | `pipeline::train` |
| 33.5 | `wait_until_out_of_work` |
| 33.5 | `main_loop` |
| 33.5 | `call` |
| 33.5 | `<std::thread::lifecycle::spawn_unchecked<<rayon_core::registry::DefaultSpawn as rayon_core::registry::ThreadS…` |
