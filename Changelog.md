# `basic_dsp`
Changes:

## Version 0.3.2
Bugfix release:

- Bugfix: SEGVAULT in add, sub, mul, div, add_smaller, sub_smaller, mul_smaller and div_smaller methods
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
