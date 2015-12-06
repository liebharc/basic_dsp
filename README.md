# `basic_dsp`

[![Build Status](https://api.travis-ci.org/liebharc/basic_dsp.png)](https://travis-ci.org/liebharc/basic_dsp)

This crate uses Rust nightly and might be unstable for longer periods. Right now the code seems to trigger the Rust internal compiler error https://github.com/rust-lang/rust/issues/28502 and possibly also https://github.com/rust-lang/rust/issues/26997 on Linux.

Basic digital signal processing (DSP) operations

The basic building blocks are 1xN (one times N) or Nx1 real or compelex vectors where N is typically at least in the order of magnitude of a couple of thousand elements. This crate tries to balance between a clear API and performance and the intent is to use this crate in other languages which aren't as fast as Rust, e.g. because of missing SIMD optimization.

This project started as small pet project to learn more about DSP, CPU architecture and Rust. Since learning involves making mistakes, don't expect things to be flawless or even close to flawless.