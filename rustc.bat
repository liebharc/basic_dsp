@echo off
rem Using a hard coded rust path right now
rem In future check the path variable for a rust path
rem f32x8/avx causes a crash right now. See comment on https://github.com/huonw/simd/pull/18
rem "C:\Program Files\Rust nightly 1.7\bin\rustc.exe" -C target-cpu=native -C target-feature=+sse2,+sse3,+avx2,+avx %*
"C:\Program Files\Rust nightly 1.7\bin\rustc.exe" -C target-cpu=native -C target-feature=+sse2,+sse3 %*