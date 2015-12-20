@echo off
rem Using a hard coded rust path right now
rem In future check the path variable for a rust path
"C:\Program Files\Rust nightly 1.7\bin\rustc.exe" -C target-cpu=native -C target-feature=+sse2,+sse3,+avx2,+avx %*