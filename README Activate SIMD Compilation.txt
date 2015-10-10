From https://huonw.github.io/blog/2015/08/simd-in-rust/
Passing the -C target-feature flag to a whole compilation with cargo is somewhat annoying at the moment. It requires a custom target spec or intercepting how rustc is called via the RUSTC environment variable (for my own experiments, I’m doing the latter: pointing the variable to a shell script that runs rustc -C target-feature='...' "$@"). This is covered by #1137. ↩

So set the RUSTC env variable, e.g. to: C:\Program Files\Rust nightly 1.5\bin\rustc -C target-feature=+neon