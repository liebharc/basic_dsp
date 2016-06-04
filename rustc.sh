#!/usr/bin/env bash
# Runs rustc and enables SIMD CPU features
# f32x8/avx causes a crash right now. See comment on https://github.com/huonw/simd/pull/18
#rustc -C target-feature=+sse2,+sse3,+avx2,+avx $@
rustc -C target-feature=+sse2,+sse3 $@
