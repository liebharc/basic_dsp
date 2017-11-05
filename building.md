# Building `basic_dsp_vector`

`basic_dsp_vector` offers several feature flags which determine how the calculation is performed. All feature flags are still experimental and under investigation, so it's highly recommended to read their description below before activating it.

The next table marks the allowed combinations for feature flags with an `X`.

|            | use_sse | use_avx | use_avx512 | use_gpu | no_std |
|:----------:|:-------:|:-------:|:----------:|:------:|:------:|
| use_sse    |    X    |    X    |      X     |    X   |    X   |
| use_avx    |    X    |    X    |      X     |    X   |    X   |
| use_avx512 |    X    |    X    |      X     |    X   |    X   |
| use_gpu    |    X    |    X    |      X     |    X   |        |
| no_std     |    X    |    X    |      X     |        |    X   |

# SIMD support: `use_sse` and `use_avx`
Activate with: `--features use_sse` or `--features use_avx`.

This enables explicit [SSE2](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) and [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) support. `SSE2` and `AVX2` are vector instructions which are supported by some CPUs (e.g. Intel and AMD) which speed up the calculations. Without any of those feature flags the lib relies on `rustc` to vectorize the code, and typically `rustc` does a very good job on that as long as it's called with the correct arguments (`-C target-cpu=native -C target-feature=+sse2,+sse3,+avx2,+avx`). There are however some cases where explicit vectorization is useful. The feature flags require the `stdsimd` crate. 
# GPU support: `use_gpu`
Activate with: `--features use_gpu`.

`use_gpu` relies on `opencl` and `clFFT` to implement some operations (right now `fft` and `convolve_signal`) on the GPU. GPUs have the advantage that they allow highly parallel processing which is beneficial for many DSP operations. However GPUs also have the disadvantage that they have a high latency, which means that it takes time to move data to the GPU and back to the host CPU. For a library like `basic_dsp_vector` with many low level DSP operations, the high latency is a major drawback. Therefore the GPU is only used if the vector length exceeds a predefined threshold. 

In future the `combined_ops` may be an option to make further use of a GPU. But as of today that's not implemented.

To run `basic_dsp_vector` with `use_gpu` at least one `opencl` driver must be installed (e.g. one of Intel, NVidia or AMD) and the `clFFT` library must be in the library path. Users should keep in mind that many consumer GPUs are optimized for `f32` operations and will be rather slow on `f64` data.

# Embedded support: `no_std`
Activate with: `--no-default-features`.

`--no-default-features` will disable the `std` feature flag. `basic_dsp_vector` will then compile using only [Rust core](https://doc.rust-lang.org/core/) and without any threading. This allows the lib to be used in more restrictive environments, e.g. embedded devices.

Keep in mind that `rustfft` still depends on `std`.