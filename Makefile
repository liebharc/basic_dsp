CARGO_CMD ?= cargo

RUST_VERSION=$(shell rustc --version)
RUST_NIGHTLY = $(findstring nightly,$(RUST_VERSION))

test:
ifeq ($(RUST_NIGHTLY), nightly)
	$(MAKE) run-all TASK="test"
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --manifest-path vector/Cargo.toml --no-default-features --lib
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --no-default-features --features std,use_sse2,matrix
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --no-default-features --features std,use_avx2,matrix
else
	$(MAKE) run-all TASK="test"
endif
	
bench:
ifeq ($(RUST_NIGHTLY), nightly)
	$(CARGO_CMD) bench
else
	@echo "Bench requires Rust nigthly, skipping bench for $(RUST_VERSION)"
endif

clean:
	$(MAKE) run-all TASK="clean"

update:
	$(MAKE) run-all TASK="update"
    
build:
	$(MAKE) run-all TASK="build"
      
build_all: build
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features --features std
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features --features std,use_sse2
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features --features std,use_avx2
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features
	
clippy:
ifeq ($(RUST_NIGHTLY), nightly)
	#$(CARGO_CMD) clean 
	#$(CARGO_CMD) clippy --no-default-features --features std,use_sse2,matrix
	@echo "Skipping clippy due to issue in nightly: component 'clippy' for target 'x86_64-unknown-linux-gnu' is unavailable for download"
else
	@echo "Skipping clippy for $(RUST_VERSION)"	
endif	
    
test_all: test
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --manifest-path vector/Cargo.toml --no-default-features --lib
	$(CARGO_CMD) test --no-default-features --features std,use_sse2
	$(CARGO_CMD) test --no-default-features --features std,use_avx2

run-all:
	$(CARGO_CMD) $(TASK) --no-default-features --features std,use_sse2,matrix
	$(CARGO_CMD) $(TASK) --manifest-path vector/Cargo.toml  --no-default-features --features std,use_sse2
	$(CARGO_CMD) $(TASK) --manifest-path matrix/Cargo.toml
	$(CARGO_CMD) $(TASK) --manifest-path interop/Cargo.toml
