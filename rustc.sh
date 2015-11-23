#!/usr/bin/env bash
# Runs rustc and enables SIMD CPU features
rustc -C target-cpu=native -C target-feature=+sse2,+sse3 $@