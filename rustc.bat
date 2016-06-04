@echo off
rem f32x8/avx causes a crash right now. See comment on https://github.com/huonw/simd/pull/18
rem rustc.exe -C target-feature=+sse2,+sse3,+avx2,+avx %*
rustc.exe -C target-feature=+sse2,+sse3 %*
