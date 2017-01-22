# `basic_dsp`
Changes:

## Version 0.4.3

- New feature: `MultiCoreSettings` now allows more control over how the library is spawning threads.

## Version 0.4.2
Bugfix release.

- Fix: Not all implementations of `to_complex_time_vec` and `to_complex_freq_vec` set the vector length to 0 if a storage of odd length is passed.
- Fix: Offset in `zero_pad` if `Surround` or `Center` was chosen and the original vector length is odd. As a result the convolution theorem didn't hold true, that means a multiplication in frequency domain didn't give the same results as a convolution.
- Fix: `zero_pad_b` failed to copy the vector completely if `Center` was the selected option.
- Fix: Convolution for vectors produced random results if the allocated size of a vector was different than it's actual size.
- Fix: Buffer `convolve_vector` for matrices now returns all buffers back to the pool.
- Performance: Convolution and interpolation now rely on the overlap and add algorithm or spawn worker threads for larger data sizes.
- Fix: `use_sse` and `use_avx` failed to pick the faster implementations. However with the current status of the `simd` crate it's not recommended to use those feature flags. The Rust lib team is right now discussing about the future of the `simd`crate.
- Fix: Out or range panic in `add`, `sub`, `mul` and `div` for large vectors.
- New feature: Added a `TypeMetaData` struct which allows to create a new vector with the same meta data as an existing vector.

## Version 0.4.1
Minor additions and improvements.

- Added a method to convolve a matrix with a matrix of impulse responses.
- Added traits with more precise versions of sum, statistics and dot products.
- Performance improvement for `swap_halves` and `swap_halves_b`, `swap_halves_b` at the same time has been marked as deprecated since it offers no advantage to `swap_halves`

## Version 0.4.0 (Breaking changes)
Added support for matrix operations. In order to allow matrices and vectors to implement the same traits the existing traits had to be renamed and restructured.

- Breaking change: Reorganized existing interfaces so that they can be reused for the matrix types. For that traits have been renamed and sometimes traits have been split in several smaller traits.
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
