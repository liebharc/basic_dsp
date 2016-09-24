@echo off
rem Runs rustc and enables SIMD CPU features
rem Choose one of the two rustc.exe calls depending on your CPU architecture

rem rustc.exe -C target-cpu=native -C target-feature=+sse2,+sse3,+avx2,+avx %*
rustc.exe -C target-cpu=native -C target-feature=+sse2,+sse3 %*
