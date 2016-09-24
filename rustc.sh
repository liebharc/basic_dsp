#!/usr/bin/env bash
# Runs rustc and enables SIMD CPU features
# Choose one of the two rustc.exe calls depending on your CPU architecture

#rustc -C target-cpu=native -C target-feature=+sse2,+sse3,+avx2,+avx $@
rustc -C target-cpu=native -C target-feature=+sse2,+sse3 $@
