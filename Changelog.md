# `basic_dsp`
Changes:

## Version 0.8.1

- Updated `lazy_static` and `arrayvec` dependencies.

## Version 0.8.0
This release aims to improve the conversions from Rust standard library types to/from DSP vectors.

- Renamed `complex(..)` function on vectors to `cdata(..)`
- Redefined index functions for better usability. If the old functionality is required then replace calls to indexers
with call to `.data(..)` and `.data_mut(..)`, e.g. (`&vec[5..10]` becomes `vec.data(5..10)` and `&mut vec[5..10]`
becomes `vec.data_mut(5..10)`)
- Changed `FromVector` trait. Call `getf` instead of `get` if the old functionality is required.

## Version 0.7.2

- Fixed https://github.com/liebharc/basic_dsp/issues/40 . Thanks to Carlos Tuñón.
- Replaced `simd` with `packed_simd`

## Version 0.7.1

- Removed `simd` from default compilation options due to https://github.com/rust-lang/rust/issues/57940 . Compile with `--features use_sse2,use_avx2` to get the previous behavior. 

## Version 0.7.0

- Update to Rust 2018
- Updated dependencies, e.g. `rustfft`

## Version 0.6.0

- Removed `combined_ops` module as it at best showed performance improvements of 10% which didn't justify the complexity within it.
- `to_real` function of `ComplexTimeLinearTableLookup` and `ComplexFrequencyLinearTableLookup` now takes `&self` instead of `self`
- Renamed feature `use_sse` to `use_sse2` and feature `use_avx` to `use_avx2` to avoid confusion with SIMD naming.
`use_sse2` and `use_avx2` are now active on default. Compile with `--no-default-features --features std` to revert to the
compile options before this update.
- `ComplexData` is now in `meta::Complex`. `RealData`, `RealOrComplexData`, `TimeData`, `FrequencyData` and 
`TimeOrFrequencyData` have been moved similarly. If you used those types then consider to implement against the vector 
type definitions or traits instead.
- `MultiCoreSettings` now allow more fine grained control over the threading behavior.

## Version 0.5.6

**Behaviour changes**:

The default [MultiCoreSettings](https://liebharc.github.io/basic_dsp/basic_dsp_vector/struct.MultiCoreSettings.html) 
now have a `core_limit` of 1. That's because enabling multi threading should be a conscious decision, 
e.g. due to the overhead of multi threading. The conversion functions with the `_par` suffix (e.g. `to_real_time_vec_par`)
can be used to get the old behaviour. 

Changes:

- Fixed performance issue for small vectors
- If `core_limit > 1` then at first call to a vector function which utilizes multi threading a small internal performance
test will be executed to determine reasonable values for how work will be spread among the available threads. Running
this benchmark will of course slow down this call (around 200 to 500ms). 
The assumption is that this is okay as it only happens once. 
- Only features `use_sse` or `use_avx`: Fixed index out of bounds exceptions in if the input vector is smaller than 
the SIMD register length.
- Fixed calculation error in `dot_product` and `dot_product_prec` for vector in real number space.
- AVX fixes (only feature `use_avx`): Fixed various calculations for the AVX optimizations. The AVX feature now 
passes the same tests as SSE and without optimizations. There is only one error in the doc tests left which seems to 
be related to https://github.com/rust-lang-nursery/stdsimd/issues/253 although that should be closed.

## Version 0.5.5
It's now decided at runtime which SIMD optimization should be used.
Also updated `clFFT` dependency.

## Version 0.5.4
- Went back from `stdsimd` to `simd` crate. 

## Version 0.5.3
- Switched from `simd` to `stdsimd` crate. 
- Updated `crossbeam`, `ocl` and `clFFT` dependencies.

## Version 0.5.2
Updated to `crossbeam`, `arrayvec`, `ocl` and `clFFT` dependencies.

## Version 0.5.1
Updated to `rustfft` version `2.0.0`.

## Version 0.5.0
API cleanup and fixes. Most API changes should be transparent to users. A few tips for the version update: 
- Deprecated functions have been removed. The traits have replacements available, so the documentation should provide an idea what's the intended replacement.
- In some cases in was possible to pass references of references (e.g. `&&vec`) as argument for binary ops (e.g. `div`) and that's no longer legal. 
- Some traits have been redefined. It's assumed that most users won't use the traits directly except for calling functions on vectors. And so most users shouldn't be affected by this change. If this doesn't hold true for your project then feedback in form of a bug or enhancement request is welcomed.

Changes:
- `zero_pad_b` now returns a result, which may contain an error if the passed argument is smaller than the actual vector length
- Renamed `interpolate_vector` to `interpolate_signal`
- Removed deprecated functions `swap_halves_b`, `statistics_splitted`
- Removed `MapInplaceNoArgsOps` for combined ops since they seem to not fit in their independent on how this module progresses.
- Restructured statistics traits.
- Mapping operations now only borrow a function. That also allows them to work internally without any reference counting.
- Binary operations are now more flexible regarding what kind of argument they accept.
- Restructured buffers to avoid two classes of errors.

## Version 0.4.3
Minor additions.

- Updated `simd` dependency to version `0.2`.
- New feature: Added `ApproximatedOps` trait.
- New feature: New `interpolate` method which offers the same features as `interpolatef`, but the performance should be closer to `interpolatei`.
- Fix: `multiply_complex_exponential` didn't consistently took `delta` into account.

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
