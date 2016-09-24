# `basic_dsp`
Changes:
## Version 0.4.0 (Breaking changes)
Added support for matrix operations. In order to allow matrices and vectors to implement the same traits the existing traits had to be renamed and restructured.

- Breaking change: Reorganized existing interfaces so that they can be reused for the matrix types. For that traits have been renamed and sometimes traits have been splitted in several smaller traits.
- Breaking change: Interop facade is now only compiled with `--features interop`
- Breaking change: Removed `complex_data`, `data` and `override_data` (which also had a spelling mistake) in favor for implementations of several Indexer traits.
- Crate now compiles with Rust stable

## Version 0.3.2
Bugfix release:

- Bugfix: Potential SEGVAULT in add, sub, mul, div, add_smaller, sub_smaller, mul_smaller and div_smaller methods
- Bugfix: SEGVAULT in get_magnitude and complex_dot_product

## Version 0.3.1
Bugfix release with a minor enhancement:

- New feature: Added `sum` and `sum_sq` operations.
- Bugfix: SEGVAULT in complex magnitude op with recent versions of Rust nightly

## Version 0.3.0 (Breaking changes)
Added prepared operations/multi operations, see [combined_ops](https://liebharc.github.io/basic_dsp/basic_dsp/combined_ops)

- Breaking change: Renamed a lot of operations so that their names match (more closely) with the `num` crate. Also renamed traits so that all traits which mainly implement operations have similiar names.
- Breaking change: RededicateVector trait now defines exactly one conversion. Vectors implement the trait several times so the same functionality is still available.
- Breaking change: perform_operations and related types appear on the API differently now.

## Version 0.2.2
Added Apache-2.0 as license option.

## Version 0.2.1
Bugfix release: Fixed implementation of AVX operations.

## Version 0.2.0
First release: Vectors with a couple of operations are available.
